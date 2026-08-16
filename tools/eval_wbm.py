"""WBM / Matbench-Discovery evaluation for ECENet checkpoints.

Two stages, so the compute-heavy part carries minimal dependencies:

``relax`` — the discovery protocol's expensive half. Loads a checkpoint
through ``load_calculator`` (joint-LES checkpoints get E_sr + E_lr
automatically), relaxes WBM *initial* structures with FIRE under a full cell
filter, and writes one json.gz shard of final energies + compositions.
Sliceable for a job array (``--slice``), resumable (already-done ids in an
existing ``--out`` are skipped), and structures containing elements outside
the checkpoint's type map are counted and skipped, not crashed on.

``score`` — pure post-processing (numpy + stdlib csv, no torch). Merges relax
shards, computes each structure's predicted formation energy per atom from
MP elemental reference energies, shifts the DFT hull distance by the
formation-energy error (the hull is fixed, so
``pred_e_hull = dft_e_hull + (pred_e_form − dft_e_form)``), and reports the
Matbench-Discovery-style metrics: e_form MAE/RMSE/R², and the stability
classification (F1, precision, recall, accuracy, DAF) at ``e_hull ≤ 0``.

Energy referencing: ``ECENetCalculator`` adds the checkpoint's per-element
``e_ref`` back, so predicted energies are absolute totals on the TRAINING
energy scale. For an MPtrj checkpoint prepared from ``corrected_total_energy``
that is the MP2020-corrected scale — compare against the summary's
``*_mp2020_corrected`` columns and MP's corrected elemental references
(both auto-detected).

Data files (once, e.g. via the matbench-discovery package or its figshare
links — https://matbench-discovery.materialsproject.org):
  * WBM initial structures  (jsonl[.gz] — the current release format — or
    json[.bz2|.gz] as a plain ``{id: structure}`` mapping / pandas column form)
  * WBM summary CSV         (DFT e_form + e_above_hull per id)
  * MP elemental reference energies (json; entry dicts or ``{symbol: eV/atom}``)

Usage (from the repo root):
    python tools/eval_wbm.py relax --checkpoint mptrj.mdl \
        --structures wbm-initial-structures.json.bz2 \
        --slice 0:1000 --out out/wbm_000.json.gz
    python tools/eval_wbm.py score --pred 'out/wbm_*.json.gz' \
        --summary wbm-summary.csv.gz --ref mp-elemental-refs.json \
        --out wbm_metrics.json [--csv per_structure.csv]
"""

import argparse
import bz2
import glob
import gzip
import json
import os
import sys  # repo root + scripts/ on path (run from anywhere)
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import numpy as np


def _open_auto(path, mode='rt'):
    low = path.lower()
    if low.endswith('.bz2'):
        return bz2.open(path, mode)
    if low.endswith('.gz'):
        return gzip.open(path, mode)
    return open(path, mode)


def _load_jsonl_structures(f):
    """JSON Lines (the current matbench-discovery release format): one record
    per line, e.g. {"material_id": ..., "initial_structure": {...}}."""
    out = {}
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        mid = rec.get('material_id') or rec.get('id')
        struct = next((v for k, v in rec.items()
                       if isinstance(v, dict) and 'sites' in v), None)
        if mid is None or struct is None:
            raise ValueError(f"jsonl record without id/structure: {list(rec)}")
        out[mid] = struct
    return out


def load_wbm_structures(path):
    """Return {material_id: pymatgen-style structure dict}. Accepts JSON Lines
    (one {"material_id": ..., "initial_structure": {...}} record per line —
    the current matbench-discovery format), a plain ``{id: structure}``
    mapping, or a pandas column-oriented dump (the single column whose values
    hold 'lattice'/'sites' dicts is taken)."""
    if '.jsonl' in path.lower():
        with _open_auto(path) as f:
            return _load_jsonl_structures(f)
    with _open_auto(path) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:                  # jsonl in .json clothing
            f.seek(0)
            return _load_jsonl_structures(f)
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"{path}: expected a non-empty JSON mapping")
    probe = next(iter(obj.values()))
    if isinstance(probe, dict) and 'sites' in probe:
        return obj                                    # plain {id: structure}
    for key, col in obj.items():                      # pandas column form
        if isinstance(col, dict) and col:
            inner = next(iter(col.values()))
            if isinstance(inner, dict) and 'sites' in inner:
                return col
    raise ValueError(f"{path}: no structure column found "
                     f"(top-level keys: {list(obj)[:5]})")


# ---------------------------------------------------------------------------
# relax
# ---------------------------------------------------------------------------

def run_relax(args):
    import torch  # noqa: F401  (keeps the heavy import out of `score`)
    from ase import Atoms
    from ase.optimize import FIRE
    from train_ecenet_mptrj import _structure_dict_to_arrays

    from ecenet import elements
    from ecenet.calculator import load_calculator
    try:
        from ase.filters import FrechetCellFilter as CellFilter
    except ImportError:                                # older ASE
        from ase.constraints import ExpCellFilter as CellFilter

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    calc = load_calculator(args.checkpoint, device=args.device)
    known = set(calc.element_to_type)

    structures = load_wbm_structures(args.structures)
    ids = sorted(structures)
    if args.slice:
        a, b = args.slice.split(':')
        ids = ids[int(a or 0):int(b) if b else None]
    print(f"{len(ids)} structures in this slice "
          f"(of {len(structures)} loaded)", flush=True)

    # Resume: keep prior results, skip their ids.
    results = {}
    if os.path.exists(args.out):
        with _open_auto(args.out) as f:
            results = json.load(f).get('results', {})
        print(f"resuming: {len(results)} already done in {args.out}", flush=True)

    def flush():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with gzip.open(args.out, 'wt') as f:
            json.dump({'meta': {'checkpoint': os.path.abspath(args.checkpoint),
                                'structures': os.path.abspath(args.structures),
                                'fmax': args.fmax, 'max_steps': args.max_steps},
                       'results': results}, f)

    t0, done0 = time.time(), len(results)
    for k, mid in enumerate(ids):
        if mid in results:
            continue
        try:
            numbers, positions, cell = _structure_dict_to_arrays(structures[mid])
            syms = [elements.symbol(int(z)) for z in numbers]
            unknown = sorted(set(syms) - known)
            if unknown:
                results[mid] = {'skipped': f"unknown elements: {unknown}"}
                continue
            atoms = Atoms(numbers=numbers, positions=positions, cell=cell,
                          pbc=True)
            atoms.calc = calc
            opt = FIRE(CellFilter(atoms), logfile=None)
            converged = bool(opt.run(fmax=args.fmax, steps=args.max_steps))
            comp = {}
            for s in syms:
                comp[s] = comp.get(s, 0) + 1
            results[mid] = {
                'energy': float(atoms.get_potential_energy()),   # eV, absolute
                'n_atoms': len(atoms),
                'composition': comp,
                'converged': converged,
                'n_steps': int(opt.nsteps),
                'volume': float(atoms.get_volume()),
            }
        except Exception as e:                     # keep the shard going
            results[mid] = {'error': f"{type(e).__name__}: {e}"}
        if len(results) % args.flush_every == 0:
            flush()
            rate = (len(results) - done0) / max(1e-9, time.time() - t0)
            left = sum(1 for m in ids if m not in results)
            print(f"  {len(results)}/{len(ids)} done "
                  f"({rate:.2f} struct/s, ~{left / max(rate, 1e-9) / 60:.0f} min left)",
                  flush=True)
    flush()
    n_err = sum(1 for r in results.values() if 'error' in r)
    n_skip = sum(1 for r in results.values() if 'skipped' in r)
    print(f"done: {len(results)} total | {n_err} errors | {n_skip} skipped "
          f"(unknown elements) → {args.out}", flush=True)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def load_elemental_refs(path):
    """{symbol: eV/atom}. Accepts plain floats, or (lists of) entry dicts with
    'energy' + 'composition' (Materials Project ComputedEntry style)."""
    with _open_auto(path) as f:
        obj = json.load(f)

    def per_atom(entry):
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, list):               # multiple entries → lowest
            return min(per_atom(e) for e in entry)
        comp = entry.get('composition')
        n = sum(comp.values()) if comp else entry.get('nsites', 1)
        if 'energy_per_atom' in entry:
            return float(entry['energy_per_atom'])
        # ComputedEntry serialization keeps the correction separate from the
        # raw energy; pymatgen's entry.energy is their sum (zero for all
        # current MP elemental refs, but honor it).
        return (float(entry['energy']) + float(entry.get('correction', 0.0))) / n
    return {sym: per_atom(entry) for sym, entry in obj.items()}


def _find_col(columns, must, prefer):
    """Column whose name contains all of `must`; among those, the one that
    also contains the most `prefer` substrings."""
    cands = [c for c in columns if all(m in c.lower() for m in must)]
    if not cands:
        raise ValueError(f"no column matching {must} in {list(columns)}")
    return max(cands, key=lambda c: sum(p in c.lower() for p in prefer))


def load_wbm_summary(path, unique_only=False):
    """{material_id: (e_form_dft, e_hull_dft)} from the WBM summary CSV,
    auto-detecting the id / formation-energy / hull-distance columns
    (preferring the MP2020-corrected variants). Plain csv module — the
    score stage deliberately needs nothing beyond numpy.

    unique_only: keep only rows flagged in the ``unique_prototype`` column —
    the ~215k deduplicated subset the Matbench-Discovery leaderboard's
    headline numbers are computed on (the full set double-counts repeated
    prototypes and is slightly easier)."""
    import csv
    with _open_auto(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        id_col = header.index(_find_col(header, ['id'], ['material']))
        ef_col = header.index(_find_col(header, ['e_form_per_atom'],
                                        ['mp2020', 'corrected']))
        eh_col = header.index(_find_col(header, ['e_above_hull'],
                                        ['mp2020', 'corrected', 'ppd']))
        uq_col = None
        if unique_only:
            uq_col = header.index(_find_col(header, ['unique', 'prototype'], []))
        out = {}
        for row in reader:
            if uq_col is not None and row[uq_col].strip().lower() not in (
                    'true', '1'):
                continue
            try:
                out[row[id_col]] = (float(row[ef_col]), float(row[eh_col]))
            except (ValueError, IndexError):
                continue          # missing DFT value → structure not scorable
    return out


def run_score(args):
    # merge shards
    preds = {}
    for path in sorted(sum((glob.glob(p) for p in args.pred), [])):
        with _open_auto(path) as f:
            preds.update(json.load(f)['results'])
    if not preds:
        raise SystemExit(f"no predictions found under {args.pred}")
    n_err = sum(1 for r in preds.values() if 'error' in r)
    n_skip = sum(1 for r in preds.values() if 'skipped' in r)

    refs = load_elemental_refs(args.ref)
    summary = load_wbm_summary(args.summary,
                               unique_only=args.unique_prototypes)

    rows = []      # (id, e_form_pred, e_form_dft, e_hull_dft, converged)
    for mid, r in preds.items():
        if 'energy' not in r:
            continue
        if any(s not in refs for s in r['composition']):
            n_skip += 1
            continue
        if mid not in summary:
            continue
        mu = sum(n * refs[s] for s, n in r['composition'].items())
        ef_dft, eh_dft = summary[mid]
        rows.append((mid, (r['energy'] - mu) / r['n_atoms'],
                     ef_dft, eh_dft, r.get('converged', True)))
    if not rows:
        raise SystemExit("no overlap between predictions and the summary ids")
    ids = [row[0] for row in rows]
    ef_pred, ef_dft, eh_dft = (np.array([row[k] for row in rows])
                               for k in (1, 2, 3))
    converged = np.array([row[4] for row in rows], dtype=bool)
    # Fixed reference hull → the hull-distance error IS the e_form error.
    eh_pred = eh_dft + (ef_pred - ef_dft)

    err = ef_pred - ef_dft
    thr = args.threshold
    st, sp = eh_dft <= thr, eh_pred <= thr
    tp, fp = int((st & sp).sum()), int((~st & sp).sum())
    fn, tn = int((st & ~sp).sum()), int((~st & ~sp).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    prevalence = float(st.mean())
    metrics = {
        'n_scored': len(rows),
        'unique_prototypes_only': bool(args.unique_prototypes),
        'n_pred_total': len(preds), 'n_errors': n_err, 'n_skipped': n_skip,
        'n_unconverged': int((~converged).sum()),
        'e_form_mae': float(np.abs(err).mean()),
        'e_form_rmse': float(np.sqrt((err ** 2).mean())),
        'e_form_r2': float(1 - (err ** 2).sum()
                           / max(1e-12, ((ef_dft - ef_dft.mean()) ** 2).sum())),
        'stability_threshold': thr,
        'prevalence': prevalence,
        'precision': prec, 'recall': rec,
        'f1': 2 * prec * rec / max(1e-12, prec + rec),
        'accuracy': (tp + tn) / len(rows),
        'daf': float(prec / max(prevalence, 1e-12)),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }
    uq = " [unique prototypes]" if args.unique_prototypes else ""
    print(f"scored {metrics['n_scored']:,} structures{uq} "
          f"({n_err} errors, {n_skip} skipped, "
          f"{metrics['n_unconverged']} unconverged)")
    print(f"  e_form  MAE {metrics['e_form_mae']*1e3:7.1f} meV/atom | "
          f"RMSE {metrics['e_form_rmse']*1e3:7.1f} | R² {metrics['e_form_r2']:.3f}")
    print(f"  stable@{thr:g}  F1 {metrics['f1']:.3f} | precision {prec:.3f} | "
          f"recall {rec:.3f} | DAF {metrics['daf']:.2f}")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"metrics → {args.out}")
    if args.csv:
        with open(args.csv, 'w') as f:
            f.write("material_id,e_form_pred,e_form_dft,e_hull_dft,"
                    "e_hull_pred,converged\n")
            for k, mid in enumerate(ids):
                f.write(f"{mid},{ef_pred[k]!r},{ef_dft[k]!r},{eh_dft[k]!r},"
                        f"{eh_pred[k]!r},{int(converged[k])}\n")
        print(f"per-structure table → {args.csv}")
    return metrics


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    pr = sub.add_parser('relax', help='relax WBM initial structures')
    pr.add_argument('--checkpoint', required=True)
    pr.add_argument('--structures', required=True,
                    help='WBM initial-structures json[.bz2|.gz]')
    pr.add_argument('--out', required=True, help='output shard (json.gz)')
    pr.add_argument('--slice', default=None,
                    help='A:B slice of the sorted id list (job arrays)')
    pr.add_argument('--fmax', type=float, default=0.05, help='eV/Å')
    pr.add_argument('--max_steps', type=int, default=500)
    pr.add_argument('--device', default=None)
    pr.add_argument('--tf32', action='store_true')
    pr.add_argument('--flush_every', type=int, default=500)

    ps = sub.add_parser('score', help='score relax shards against DFT')
    ps.add_argument('--pred', nargs='+', required=True,
                    help='glob(s) of relax output shards')
    ps.add_argument('--summary', required=True, help='WBM summary csv[.gz]')
    ps.add_argument('--ref', required=True,
                    help='MP elemental reference energies (json)')
    ps.add_argument('--out', default=None, help='metrics json')
    ps.add_argument('--csv', default=None, help='per-structure csv dump')
    ps.add_argument('--threshold', type=float, default=0.0,
                    help='stability threshold on e_hull (eV/atom)')
    ps.add_argument('--unique_prototypes', action='store_true',
                    help="score only the summary's unique_prototype subset "
                         "(~215k rows — the leaderboard's headline split)")

    args = p.parse_args(argv)
    if args.cmd == 'relax':
        run_relax(args)
    else:
        run_score(args)


if __name__ == '__main__':
    main()

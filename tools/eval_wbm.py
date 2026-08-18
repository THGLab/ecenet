"""WBM / Matbench-Discovery evaluation for ECENet checkpoints.

Two stages, so the compute-heavy part carries minimal dependencies:

``relax`` — the discovery protocol's expensive half. Loads a checkpoint
through ``load_calculator`` (joint-LES checkpoints get E_sr + E_lr
automatically), relaxes WBM *initial* structures with FIRE under a full cell
filter, and writes one json.gz shard of final energies + compositions.
Sliceable for a job array (``--slice``), resumable (already-done ids in an
existing ``--out`` are skipped), and structures containing elements outside
the checkpoint's type map are counted and skipped, not crashed on.

``singlepoint`` — E/F/stress error at the DFT-relaxed geometries, no
relaxation: separates pure PES accuracy from optimizer behavior. Energies
compare to the MP2020-corrected DFT totals reconstructed from the summary
(``uncorrected_energy_from_cse + n_sites·e_correction_per_atom_mp2020``);
the reported force and stress errors are against the DFT reference of ≈0 at
these relaxed geometries (equilibrium force/stress errors — WBM publishes
neither). Sliceable/resumable like ``relax``; ``--pred`` aggregates existing
shards instead of computing.

``rmsd`` — geometry quality (needs pymatgen, plus shards written with
``--save_structures``): Matbench-Discovery's exact recipe —
``StructureMatcher(stol=1.0, scale=False).get_rms_dist(pred, dft)[0]``
against the DFT-relaxed structures, unitless ((volume/atom)^(1/3)-
normalized), unmatched structures filled with 1.0 and a plain mean —
alongside matched-only mean/median for diagnostics.

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


def _find_structure(obj, depth=3):
    """Depth-limited search for a pymatgen-style structure dict ('lattice' +
    'sites') — handles both bare structures and ComputedStructureEntry
    records, where it sits one level down under 'structure'."""
    if isinstance(obj, dict):
        if 'sites' in obj and 'lattice' in obj:
            return obj
        if depth:
            for v in obj.values():
                got = _find_structure(v, depth - 1)
                if got is not None:
                    return got
    return None


def _load_jsonl_structures(f, wanted=None):
    """JSON Lines (the current matbench-discovery release format): one record
    per line, e.g. {"material_id": ..., "initial_structure": {...}} or a
    ComputedStructureEntry record with the structure nested inside.

    wanted: optional id set — other lines are skipped on a cheap regex id
    probe instead of paying the full json parse (the records are large; this
    turns a minutes-long scan into seconds when few ids are needed)."""
    import re
    id_probe = re.compile(r'"(?:material_)?id"\s*:\s*"([^"]+)"')
    out = {}
    for line in f:
        line = line.strip()
        if not line:
            continue
        if wanted is not None:
            m = id_probe.search(line[:200])
            if m and m.group(1) not in wanted:
                continue
        rec = json.loads(line)
        mid = rec.get('material_id') or rec.get('id')
        struct = _find_structure(rec)
        if mid is None or struct is None:
            raise ValueError(f"jsonl record without id/structure: {list(rec)}")
        out[mid] = struct
    return out


def load_wbm_structures(path, wanted=None):
    """Return {material_id: pymatgen-style structure dict}. Accepts JSON Lines
    (one {"material_id": ..., "initial_structure": {...}} record per line —
    the current matbench-discovery format), a plain ``{id: structure}``
    mapping, or a pandas column-oriented dump (the single column whose values
    hold 'lattice'/'sites' dicts is taken). ``wanted`` (an id set) speeds up
    the jsonl path by skipping other records."""
    if '.jsonl' in path.lower():
        with _open_auto(path) as f:
            return _load_jsonl_structures(f, wanted)
    with _open_auto(path) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:                  # jsonl in .json clothing
            f.seek(0)
            return _load_jsonl_structures(f, wanted)
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
    import warnings

    import torch  # noqa: F401  (keeps the heavy import out of `score`)
    from ase import Atoms
    from ase.optimize import FIRE

    # FrechetCellFilter parameterizes the cell via scipy's matrix logarithm,
    # which reports its internal error estimate (~1e-13 — float64 roundoff)
    # as a RuntimeWarning on every step. Benign, and 215k structures of it
    # would flood the logs.
    warnings.filterwarnings('ignore', message='logm result may be inaccurate')
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
    if args.unique_prototypes:
        # Skip structures outside the leaderboard's deduplicated split up
        # front (~16% of the set). NOTE: --slice indexes the FILTERED id
        # list, so slice ranges are not interchangeable between runs with
        # and without this flag.
        if not args.summary:
            raise SystemExit("--unique_prototypes needs --summary "
                             "(the unique_prototype flag lives there)")
        keep = load_wbm_summary(args.summary, unique_only=True)
        ids = [m for m in ids if m in keep]
        print(f"unique-prototype filter: {len(ids)} of {len(structures)} "
              "structures kept", flush=True)
    if args.slice:
        a, b = args.slice.split(':')
        ids = ids[int(a or 0):int(b) if b else None]
    print(f"{len(ids)} structures in this slice "
          f"(of {len(structures)} loaded)", flush=True)

    # Resume: keep prior successes/skips, but RETRY prior errors (they are
    # usually environment or since-fixed-bug artifacts, not properties of
    # the structure).
    results = {}
    if os.path.exists(args.out):
        with _open_auto(args.out) as f:
            results = json.load(f).get('results', {})
        n_prev_err = sum(1 for r in results.values() if 'error' in r)
        results = {m: r for m, r in results.items() if 'error' not in r}
        print(f"resuming: {len(results)} already done in {args.out}"
              + (f" ({n_prev_err} prior errors will be retried)"
                 if n_prev_err else ""), flush=True)

    def flush():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with gzip.open(args.out, 'wt') as f:
            json.dump({'meta': {'checkpoint': os.path.abspath(args.checkpoint),
                                'structures': os.path.abspath(args.structures),
                                'fmax': args.fmax, 'max_steps': args.max_steps,
                                'unique_prototypes': bool(args.unique_prototypes)},
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
            if args.save_structures:
                # Final geometry (for RMSD-vs-DFT scoring / re-use); 1e-6 Å
                # rounding keeps the shards compact.
                results[mid]['structure'] = {
                    'symbols': syms,
                    'positions': [[round(x, 6) for x in p]
                                  for p in atoms.get_positions().tolist()],
                    'cell': [[round(x, 6) for x in v]
                             for v in atoms.get_cell().tolist()],
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
# singlepoint — E/F error at the DFT-relaxed geometries (no relaxation)
# ---------------------------------------------------------------------------

def load_wbm_dft_energies(path):
    """{material_id: MP2020-corrected DFT total energy (eV)} from the summary.

    The computed-structure-entries file stores uncorrected energies with
    correction=0 — the corrections were applied in the summary pipeline — so
    the corrected total is reconstructed as
    ``uncorrected_energy_from_cse + n_sites * e_correction_per_atom_mp2020``.
    """
    import csv
    with _open_auto(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        id_col = header.index(_find_col(header, ['id'], ['material']))
        e_col = header.index(_find_col(header, ['uncorrected_energy'], ['cse']))
        c_col = header.index(_find_col(header, ['e_correction_per_atom'],
                                       ['mp2020']))
        n_col = header.index(_find_col(header, ['n_sites'], []))
        out = {}
        for row in reader:
            try:
                out[row[id_col]] = (float(row[e_col])
                                    + float(row[n_col]) * float(row[c_col]))
            except (ValueError, IndexError):
                continue
    return out


def _sp_aggregate(results):
    """Aggregate metrics over singlepoint shard entries (dicts with e_pred,
    e_dft, n_atoms, f_rms, f_max)."""
    ok = [r for r in results.values() if 'e_pred' in r]
    if not ok:
        raise SystemExit("no successful singlepoint entries")
    de = np.array([(r['e_pred'] - r['e_dft']) / r['n_atoms'] for r in ok])
    # global per-component force RMS: recombine per-structure RMS values
    # weighted by their component counts (DFT reference forces are ~0 at
    # these relaxed geometries, so this IS the force error)
    ncomp = np.array([3 * r['n_atoms'] for r in ok])
    f_ms = np.array([r['f_rms'] ** 2 for r in ok])
    # f_mae was added after the first shards; older ones aggregate to NaN.
    f_maes = [r.get('f_mae') for r in ok]
    f_mae = (float((np.array(f_maes) * ncomp).sum() / ncomp.sum())
             if all(v is not None for v in f_maes) else float('nan'))
    s_maes = [r.get('s_mae') for r in ok]
    s_mae = (float(np.mean(s_maes))          # 6 Voigt components each
             if all(v is not None for v in s_maes) else float('nan'))
    metrics = {
        'n_scored': len(ok),
        'n_errors': sum(1 for r in results.values() if 'error' in r),
        'n_skipped': sum(1 for r in results.values() if 'skipped' in r),
        'energy_mae': float(np.abs(de).mean()),          # eV/atom
        'energy_rmse': float(np.sqrt((de ** 2).mean())),
        'energy_me': float(de.mean()),                   # signed bias
        'force_mae': f_mae,
        'force_rms': float(np.sqrt((f_ms * ncomp).sum() / ncomp.sum())),
        'force_max_mean': float(np.mean([r['f_max'] for r in ok])),
        'force_max_p95': float(np.percentile([r['f_max'] for r in ok], 95)),
        'stress_mae': s_mae,                 # eV/Å³, Voigt components
    }
    print(f"singlepoint over {metrics['n_scored']:,} DFT-relaxed structures "
          f"({metrics['n_errors']} errors, {metrics['n_skipped']} skipped)")
    print(f"  energy  MAE {metrics['energy_mae']*1e3:7.1f} meV/atom | "
          f"RMSE {metrics['energy_rmse']*1e3:7.1f} | "
          f"bias {metrics['energy_me']*1e3:+7.1f}")
    print(f"  forces  MAE {metrics['force_mae']*1e3:7.1f} meV/Å | "
          f"RMS {metrics['force_rms']*1e3:7.1f} "
          f"(DFT ref ≈ 0 at relaxed geometry) | "
          f"per-structure max: mean {metrics['force_max_mean']:.3f}, "
          f"p95 {metrics['force_max_p95']:.3f} eV/Å")
    print(f"  stress  MAE {metrics['stress_mae']*1e3:7.2f} meV/Å³ "
          f"(= {metrics['stress_mae'] * 160.21766:.3f} GPa; DFT ref ≈ 0 — "
          f"relaxed cells, up to Pulay residuals)")
    return metrics


def run_singlepoint(args):
    if args.pred:                       # aggregate-only over existing shards
        results = {}
        for path in sorted(sum((glob.glob(p) for p in args.pred), [])):
            with _open_auto(path) as f:
                results.update(json.load(f)['results'])
        metrics = _sp_aggregate(results)
        target = args.out_metrics or args.out
        if target:
            with open(target, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"metrics → {target}")
        return metrics

    for req in ('checkpoint', 'structures', 'out'):
        if getattr(args, req) is None:
            raise SystemExit(f"singlepoint compute mode needs --{req} "
                             "(or pass --pred to aggregate existing shards)")
    import torch  # noqa: F401
    from ase import Atoms
    from train_ecenet_mptrj import _structure_dict_to_arrays

    from ecenet import elements
    from ecenet.calculator import load_calculator

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    if 'init' in os.path.basename(args.structures).lower():
        # The summary's DFT energies belong to the RELAXED geometries; a
        # single-point on the initial guesses would compare energies of two
        # different structures. Heuristic (filename) but worth failing loud.
        raise SystemExit(
            f"--structures {args.structures} looks like the INITIAL "
            "structures. singlepoint needs the DFT-relaxed geometries "
            "(wbm-computed-structure-entries.jsonl.gz) — the summary's DFT "
            "energies belong to those.")

    calc = load_calculator(args.checkpoint, device=args.device)
    known = set(calc.element_to_type)

    e_dft = load_wbm_dft_energies(args.summary)
    if args.unique_prototypes:
        keep = load_wbm_summary(args.summary, unique_only=True)
        e_dft = {m: v for m, v in e_dft.items() if m in keep}
    structures = load_wbm_structures(args.structures)
    ids = sorted(m for m in structures if m in e_dft)
    if args.slice:
        a, b = args.slice.split(':')
        ids = ids[int(a or 0):int(b) if b else None]
    print(f"{len(ids)} structures in this slice "
          f"(of {len(structures)} loaded)", flush=True)

    results = {}
    if os.path.exists(args.out):
        with _open_auto(args.out) as f:
            results = json.load(f).get('results', {})
        n_prev_err = sum(1 for r in results.values() if 'error' in r)
        results = {m: r for m, r in results.items() if 'error' not in r}
        print(f"resuming: {len(results)} already done"
              + (f" ({n_prev_err} prior errors will be retried)"
                 if n_prev_err else ""), flush=True)

    def flush():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with gzip.open(args.out, 'wt') as f:
            json.dump({'meta': {'checkpoint': os.path.abspath(args.checkpoint),
                                'structures': os.path.abspath(args.structures),
                                'mode': 'singlepoint'},
                       'results': results}, f)

    t0, done0 = time.time(), len(results)
    for mid in ids:
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
            # stress first: that single call computes E, F, and σ in one
            # strain-augmented evaluation; the later getters hit the cache
            st = atoms.get_stress()          # Voigt 6, eV/Å³
            fr = atoms.get_forces()
            results[mid] = {
                'e_pred': float(atoms.get_potential_energy()),
                'e_dft': float(e_dft[mid]),
                'n_atoms': len(atoms),
                'f_mae': float(np.abs(fr).mean()),
                'f_rms': float(np.sqrt((fr ** 2).mean())),
                'f_max': float(np.abs(fr).max()),
                's_mae': float(np.abs(st).mean()),
                's_max': float(np.abs(st).max()),
            }
        except Exception as e:
            results[mid] = {'error': f"{type(e).__name__}: {e}"}
        if len(results) % args.flush_every == 0:
            flush()
            rate = (len(results) - done0) / max(1e-9, time.time() - t0)
            print(f"  {len(results)}/{len(ids)} done ({rate:.2f} struct/s)",
                  flush=True)
    flush()
    metrics = _sp_aggregate({m: results[m] for m in ids if m in results})
    if args.out_metrics:
        with open(args.out_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"metrics → {args.out_metrics}")
    return metrics


# ---------------------------------------------------------------------------
# rmsd
# ---------------------------------------------------------------------------

def _as_pmg_structure(struct_dict, Structure):
    """pymatgen Structure from a structure dict, via our own site parsing
    (pymatgen's from_dict is pickier about site keys than the data warrants)."""
    lat = struct_dict['lattice']['matrix']
    syms, coords, cart = [], [], True
    for site in struct_dict['sites']:
        syms.append(site['species'][0]['element'])
        if 'xyz' in site:
            coords.append(site['xyz'])
        else:
            coords.append(site['abc'])
            cart = False
    return Structure(lat, syms, coords, coords_are_cartesian=cart)


_RMSD_MATCHER = None    # one StructureMatcher per worker process


def _alarm_handler(signum, frame):
    raise TimeoutError


def _rmsd_worker(item):
    """(material_id, pred_struct, ref_struct_dict, timeout) →
    (material_id, rmsd|None).

    Matbench-Discovery's exact recipe: StructureMatcher(stol=1.0,
    scale=False).get_rms_dist(pred, ref)[0] — pymatgen normalizes the rms by
    (volume/atom)^(1/3), so the value is unitless. None = no match found.

    Two guards against pathological relaxations (a collapsed cell — e.g.
    0.01 Å³ from a bad model — makes the matcher enumerate an astronomically
    dense lattice-vector set and hang for hours):
      * volume-ratio precheck: with scale=False and the default ltol=0.2, a
        lattice match is impossible beyond ~1.7× volume, so >2× or <0.5×
        returns unmatched immediately;
      * a per-match SIGALRM timeout as a backstop (POSIX only; skipped where
        unavailable).
    """
    global _RMSD_MATCHER
    import signal

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    if _RMSD_MATCHER is None:
        _RMSD_MATCHER = StructureMatcher(stol=1.0, scale=False)
    mid, pred, ref, timeout = item
    try:
        p = Structure(pred['cell'], pred['symbols'], pred['positions'],
                      coords_are_cartesian=True)
        r = _as_pmg_structure(ref, Structure)
        if not 0.5 <= p.volume / r.volume <= 2.0:
            return mid, None
        use_alarm = hasattr(signal, 'SIGALRM') and timeout
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(int(timeout))
        try:
            out = _RMSD_MATCHER.get_rms_dist(p, r)
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        return mid, (None if out is None else float(out[0]))
    except Exception:
        return mid, None                     # counted with the unmatched


def run_rmsd(args):
    try:
        import pymatgen  # noqa: F401
    except ImportError:
        raise SystemExit("the rmsd stage needs pymatgen: pip install pymatgen")

    preds = {}
    for path in sorted(sum((glob.glob(p) for p in args.pred), [])):
        with _open_auto(path) as f:
            preds.update(json.load(f)['results'])
    with_struct = {m: r['structure'] for m, r in preds.items()
                   if 'structure' in r}
    n_missing = sum(1 for r in preds.values()
                    if 'energy' in r and 'structure' not in r)
    if not with_struct:
        raise SystemExit("no saved geometries in the shards — relax with "
                         "--save_structures")
    if args.unique_prototypes:
        keep = load_wbm_summary(args.summary, unique_only=True)
        with_struct = {m: s for m, s in with_struct.items() if m in keep}

    print(f"loading DFT-relaxed references from {args.ref}...", flush=True)
    refs = load_wbm_structures(args.ref, wanted=set(with_struct))
    items = [(m, s, refs[m], args.match_timeout)
             for m, s in with_struct.items() if m in refs]
    n_no_ref = len(with_struct) - len(items)

    t0 = time.time()
    if args.nproc > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.nproc) as ex:
            pairs = list(ex.map(_rmsd_worker, items, chunksize=16))
    else:
        pairs = []
        for k, item in enumerate(items):
            pairs.append(_rmsd_worker(item))
            if (k + 1) % 500 == 0:
                rate = (k + 1) / (time.time() - t0)
                print(f"  {k + 1}/{len(items)} ({rate:.1f}/s)", flush=True)

    rmsds = dict(pairs)
    matched = np.array([v for v in rmsds.values() if v is not None])
    n_unmatched = sum(1 for v in rmsds.values() if v is None)
    # Leaderboard aggregation: unmatched filled with 1.0 (the stol), plain mean.
    filled = np.array([1.0 if v is None else v for v in rmsds.values()])
    metrics = {
        'rmsd': float(filled.mean()) if len(filled) else float('nan'),
        'rmsd_matched_mean': float(matched.mean()) if len(matched) else float('nan'),
        'rmsd_matched_median': float(np.median(matched)) if len(matched) else float('nan'),
        'n_scored': len(rmsds), 'n_matched': int(len(matched)),
        'n_unmatched': n_unmatched, 'n_no_reference': n_no_ref,
        'n_missing_structure': n_missing,
        'unique_prototypes_only': bool(args.unique_prototypes),
    }
    print(f"RMSD over {metrics['n_scored']:,} structures "
          f"({n_unmatched} unmatched → filled 1.0; {n_no_ref} without a DFT "
          f"reference; {n_missing} shard entries lacked saved geometries)")
    print(f"  rmsd (leaderboard fill) {metrics['rmsd']:.4f} | matched mean "
          f"{metrics['rmsd_matched_mean']:.4f} | matched median "
          f"{metrics['rmsd_matched_median']:.4f}")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"metrics → {args.out}")
    if args.csv:
        with open(args.csv, 'w') as f:
            f.write("material_id,rmsd,matched\n")
            for mid, v in rmsds.items():
                f.write(f"{mid},{(1.0 if v is None else v)!r},"
                        f"{int(v is not None)}\n")
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
    pr.add_argument('--summary', default=None,
                    help='WBM summary csv[.gz]; only needed with '
                         '--unique_prototypes')
    pr.add_argument('--unique_prototypes', action='store_true',
                    help="relax only the summary's unique_prototype subset "
                         "(--slice then indexes the filtered id list)")
    pr.add_argument('--save_structures', action='store_true',
                    help='also store each relaxed geometry (symbols, '
                         'positions, cell) in the shard — needed for later '
                         'RMSD-vs-DFT scoring')

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

    pm = sub.add_parser('rmsd', help='geometry RMSD vs the DFT-relaxed '
                                     'structures (needs pymatgen)')
    pm.add_argument('--pred', nargs='+', required=True,
                    help='relax shards written with --save_structures')
    pm.add_argument('--ref', required=True,
                    help='DFT-relaxed WBM structures (e.g. the '
                         'computed-structure-entries jsonl[.gz])')
    pm.add_argument('--out', default=None, help='metrics json')
    pm.add_argument('--csv', default=None, help='per-structure csv dump')
    pm.add_argument('--summary', default=None,
                    help='WBM summary csv[.gz]; only with --unique_prototypes')
    pm.add_argument('--unique_prototypes', action='store_true',
                    help='restrict to the unique_prototype subset')
    pm.add_argument('--nproc', type=int, default=1,
                    help='worker processes for the structure matching')
    pm.add_argument('--match_timeout', type=int, default=120,
                    help='seconds per structure match before counting it '
                         'unmatched (0 disables; POSIX only)')

    pp = sub.add_parser('singlepoint',
                        help='E/F error at the DFT-relaxed geometries '
                             '(no relaxation)')
    pp.add_argument('--checkpoint', default=None)
    pp.add_argument('--structures', default=None,
                    help='DFT-relaxed structures (the computed-structure-'
                         'entries jsonl[.gz])')
    pp.add_argument('--summary', required=True,
                    help='WBM summary csv[.gz] (supplies the corrected DFT '
                         'total energies)')
    pp.add_argument('--out', default=None, help='output shard (json.gz)')
    pp.add_argument('--out_metrics', default=None, help='metrics json')
    pp.add_argument('--pred', nargs='+', default=None,
                    help='aggregate existing singlepoint shards instead of '
                         'computing')
    pp.add_argument('--slice', default=None,
                    help='A:B slice of the sorted id list (job arrays)')
    pp.add_argument('--device', default=None)
    pp.add_argument('--tf32', action='store_true')
    pp.add_argument('--flush_every', type=int, default=500)
    pp.add_argument('--unique_prototypes', action='store_true')

    args = p.parse_args(argv)
    if args.cmd == 'relax':
        run_relax(args)
    elif args.cmd == 'score':
        run_score(args)
    elif args.cmd == 'singlepoint':
        run_singlepoint(args)
    else:
        if args.unique_prototypes and not args.summary:
            raise SystemExit("--unique_prototypes needs --summary")
        run_rmsd(args)


if __name__ == '__main__':
    main()

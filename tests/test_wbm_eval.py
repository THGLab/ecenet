# Prototype, mainly implemented by Claude
"""tools/eval_wbm.py — WBM relaxation driver + scorer, end to end on
synthetic data (no downloads, no matbench-discovery dependency).

  1. trains a tiny MPtrj model for one epoch → self-describing checkpoint;
  2. writes a fake WBM initial-structures file (pandas column format, the
     harder of the two accepted layouts) including one structure with an
     element outside the checkpoint's type map;
  3. runs `relax` (sliced, then resumed) and checks the shard's contents;
  4. runs `score` against a summary whose DFT values are constructed FROM
     the predictions — a perfect model — so the metrics must come out exact
     (MAE 0, F1 1, DAF = 1/prevalence), which pins the e_form arithmetic,
     the hull-shift trick, and the id join;
  5. re-scores with degraded predictions to check the classification moves.

Run:  python tests/test_wbm_eval.py    (from the repo root)
"""

import gzip
import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from eval_wbm import load_elemental_refs, load_wbm_structures, main
from test_mptrj_trainer import make_structures
from train_ecenet_mptrj import train_ecenet_mptrj

DTYPE = torch.float64
TINY = dict(l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
            r_cut_edge=4.0, r_cut_neighbor=3.5)
SYMBOL = {1: 'H', 6: 'C', 8: 'O', 14: 'Si', 92: 'U'}


def to_pmg_dict(s):
    return {'lattice': {'matrix': np.asarray(s['cell']).tolist()},
            'sites': [{'species': [{'element': SYMBOL[int(z)]}],
                       'xyz': list(map(float, p))}
                      for z, p in zip(s['numbers'], s['positions'])]}


def test_wbm_eval():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. tiny checkpoint (H/C/O/Si type map, e_ref, stress conv baked in)
        ckpt = os.path.join(tmp, 'wbm_test.mdl')
        train_ecenet_mptrj(train_structures=make_structures(12, seed=1),
                           n_val=3, n_epochs=1, batch_size=4,
                           checkpoint_path=ckpt, dtype=DTYPE, device='cpu',
                           seed=0, verbose=False, **TINY)

        # 2. fake WBM file: 6 in-distribution structures + 1 unknown-element
        structs = make_structures(6, seed=9)
        wbm = {f'wbm-1-{k}': to_pmg_dict(s) for k, s in enumerate(structs)}
        bad = make_structures(1, seed=10)[0]
        bad['numbers'][:] = 92                                  # U — unknown
        wbm['wbm-1-bad'] = to_pmg_dict(bad)
        struct_path = os.path.join(tmp, 'wbm-initial-structures.json')
        with open(struct_path, 'w') as f:
            json.dump({'initial_structure': wbm}, f)            # column form
        assert set(load_wbm_structures(struct_path)) == set(wbm)
        jsonl_path = os.path.join(tmp, 'wbm-initial-structures.jsonl')
        with open(jsonl_path, 'w') as f:                        # current format
            for mid, st in wbm.items():
                f.write(json.dumps({'material_id': mid, 'formula_from_cse': 'X',
                                    'initial_structure': st}) + '\n')
        assert load_wbm_structures(jsonl_path) == load_wbm_structures(struct_path)

        # 3. relax — first a slice, then resume the rest into the same shard
        out = os.path.join(tmp, 'wbm_000.json.gz')
        common = ['relax', '--checkpoint', ckpt, '--structures', struct_path,
                  '--out', out, '--fmax', '1e-3', '--max_steps', '3',
                  '--device', 'cpu']
        main(common + ['--slice', '0:3'])
        with gzip.open(out, 'rt') as f:
            n_first = len(json.load(f)['results'])
        assert n_first == 3
        main(common)                                            # resume: all 7
        with gzip.open(out, 'rt') as f:
            shard = json.load(f)['results']
        assert len(shard) == 7
        assert 'skipped' in shard['wbm-1-bad'], shard['wbm-1-bad']
        done = {m: r for m, r in shard.items() if 'energy' in r}
        assert len(done) == 6 and all(np.isfinite(r['energy'])
                                      for r in done.values())
        assert all(r['n_steps'] <= 3 for r in done.values())

        # 4. score against a summary built FROM the predictions (perfect model)
        refs = {'H': -3.4, 'C': -9.1, 'O': -4.9, 'Si': -5.4}
        ref_path = os.path.join(tmp, 'refs.json')
        with open(ref_path, 'w') as f:                # entry-dict format
            json.dump({s: {'energy': 2 * v, 'composition': {s: 2}}
                       for s, v in refs.items()}, f)
        assert load_elemental_refs(ref_path) == refs

        rng = np.random.RandomState(3)
        rows = ['material_id,e_form_per_atom_mp2020_corrected,'
                'e_above_hull_mp2020_correct_ppd,unique_prototype']
        e_hull = {}
        dup_id = sorted(done)[-1]           # one non-unique-prototype row
        for mid, r in done.items():
            mu = sum(n * refs[s] for s, n in r['composition'].items())
            ef = (r['energy'] - mu) / r['n_atoms']
            e_hull[mid] = float(rng.uniform(-0.05, 0.1))
            rows.append(f"{mid},{ef!r},{e_hull[mid]!r},"
                        f"{mid != dup_id}")
        summary_path = os.path.join(tmp, 'wbm-summary.csv')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(rows))

        metrics_path = os.path.join(tmp, 'metrics.json')
        m = main(['score', '--pred', out, '--summary', summary_path,
                  '--ref', ref_path, '--out', metrics_path])
        assert m is None or True  # main returns None; read the file instead
        with open(metrics_path) as f:
            m = json.load(f)
        assert m['n_scored'] == 6
        assert m['e_form_mae'] < 1e-12, m['e_form_mae']
        assert m['f1'] == 1.0 and m['recall'] == 1.0 and m['precision'] == 1.0
        n_stable = sum(1 for v in e_hull.values() if v <= 0)
        assert abs(m['daf'] - len(done) / n_stable) < 1e-9, (m['daf'], n_stable)

        # unique-prototype filter drops exactly the flagged-False row
        main(['score', '--pred', out, '--summary', summary_path,
              '--ref', ref_path, '--out', metrics_path, '--unique_prototypes'])
        with open(metrics_path) as f:
            mu_ = json.load(f)
        assert mu_['n_scored'] == 5 and mu_['unique_prototypes_only'], mu_

        # relax-side filter: only unique-prototype ids get relaxed at all
        # (dup_id flagged False; 'wbm-1-bad' absent from the summary)
        out_uq = os.path.join(tmp, 'wbm_uq.json.gz')
        main(common[:5] + ['--out', out_uq] + common[7:]
             + ['--summary', summary_path, '--unique_prototypes',
                '--save_structures'])
        with gzip.open(out_uq, 'rt') as f:
            uq_shard = json.load(f)['results']
        assert len(uq_shard) == 5 and dup_id not in uq_shard \
            and 'wbm-1-bad' not in uq_shard, sorted(uq_shard)
        # --save_structures: relaxed geometry stored with consistent shapes
        for v in uq_shard.values():
            st = v['structure']
            assert len(st['symbols']) == len(st['positions']) == v['n_atoms']
            assert np.asarray(st['positions']).shape == (v['n_atoms'], 3)
            assert np.asarray(st['cell']).shape == (3, 3)
        # scoring ignores the extra field
        main(['score', '--pred', out_uq, '--summary', summary_path,
              '--ref', ref_path, '--out', metrics_path])
        with open(metrics_path) as f:
            assert json.load(f)['n_scored'] == 5

        # rmsd stage (needs pymatgen; skip cleanly without it)
        try:
            import pymatgen  # noqa: F401
        except ImportError:
            print('  (pymatgen missing — rmsd stage not tested)')
            return
        # reference = the saved geometries themselves, wrapped in the nested
        # ComputedStructureEntry jsonl format → every RMSD must be exactly 0
        def to_ref(st):
            return {'lattice': {'matrix': st['cell']},
                    'sites': [{'species': [{'element': s}], 'xyz': p}
                              for s, p in zip(st['symbols'], st['positions'])]}
        ref_structs = os.path.join(tmp, 'wbm-cse.jsonl')
        with open(ref_structs, 'w') as f:
            for mid, v in uq_shard.items():
                f.write(json.dumps({
                    'material_id': mid,
                    'computed_structure_entry':
                        {'structure': to_ref(v['structure'])}}) + '\n')
        rmsd_out = os.path.join(tmp, 'rmsd.json')
        main(['rmsd', '--pred', out_uq, '--ref', ref_structs,
              '--out', rmsd_out])
        with open(rmsd_out) as f:
            rm = json.load(f)
        assert rm['n_scored'] == 5 and rm['n_matched'] == 5, rm
        assert rm['rmsd'] < 1e-10, rm['rmsd']
        # perturb one reference → its RMSD becomes nonzero and lifts the mean
        with open(ref_structs) as f:
            lines = [json.loads(ln) for ln in f]
        st = lines[0]['computed_structure_entry']['structure']
        st['sites'][0]['xyz'] = [x + 0.15 for x in st['sites'][0]['xyz']]
        with open(ref_structs, 'w') as f:
            for rec in lines:
                f.write(json.dumps(rec) + '\n')
        main(['rmsd', '--pred', out_uq, '--ref', ref_structs,
              '--out', rmsd_out])
        with open(rmsd_out) as f:
            rm2 = json.load(f)
        assert 1e-4 < rm2['rmsd'] < 0.5, rm2['rmsd']
        assert rm2['rmsd_matched_median'] < rm2['rmsd_matched_mean'], rm2

        # ── singlepoint: E/F at fixed geometries ─────────────────────────
        # pass 1 with placeholder DFT energies → shard of e_pred values
        sp_sum = os.path.join(tmp, 'sp-summary.csv')
        with open(sp_sum, 'w') as f:
            f.write('material_id,n_sites,uncorrected_energy_from_cse,'
                    'e_correction_per_atom_mp2020\n')
            for mid, s in zip(wbm, structs + [bad]):
                f.write(f"{mid},{s['n_atoms']},0.0,0.0\n")
        # the guard refuses initial-structures input (the summary's DFT
        # energies belong to relaxed geometries)...
        try:
            main(['singlepoint', '--checkpoint', ckpt,
                  '--structures', struct_path, '--summary', sp_sum,
                  '--out', os.path.join(tmp, 'x.json.gz'), '--device', 'cpu'])
            raise AssertionError('initial-structures input not refused')
        except SystemExit as e:
            assert 'INITIAL' in str(e)
        # ...so the synthetic fixture gets a relaxed-style name here (in this
        # test the summary is built from these structures, so the pairing is
        # correct by construction)
        relaxed_path = os.path.join(tmp, 'wbm-cse-structs.json')
        os.link(struct_path, relaxed_path)
        sp_out = os.path.join(tmp, 'sp.json.gz')
        main(['singlepoint', '--checkpoint', ckpt,
              '--structures', relaxed_path,
              '--summary', sp_sum, '--out', sp_out, '--device', 'cpu'])
        with gzip.open(sp_out, 'rt') as f:
            sp = json.load(f)['results']
        assert 'skipped' in sp['wbm-1-bad']
        done_sp = {m: v for m, v in sp.items() if 'e_pred' in v}
        assert len(done_sp) == 6
        # pass 2: summary built FROM the predictions, with a nonzero
        # correction split — pins the corrected-total reconstruction
        with open(sp_sum, 'w') as f:
            f.write('material_id,n_sites,uncorrected_energy_from_cse,'
                    'e_correction_per_atom_mp2020\n')
            for mid, v in done_sp.items():
                f.write(f"{mid},{v['n_atoms']},"
                        f"{v['e_pred'] - 0.05 * v['n_atoms']!r},0.05\n")
        sp_out2 = os.path.join(tmp, 'sp2.json.gz')
        sp_metrics = os.path.join(tmp, 'sp_metrics.json')
        main(['singlepoint', '--checkpoint', ckpt,
              '--structures', relaxed_path,
              '--summary', sp_sum, '--out', sp_out2, '--device', 'cpu',
              '--out_metrics', sp_metrics])
        with open(sp_metrics) as f:
            spm = json.load(f)
        assert spm['n_scored'] == 6 and spm['energy_mae'] < 1e-10, spm
        assert np.isfinite(spm['force_rms']) and spm['force_rms'] > 0
        # aggregate-only mode reproduces the same numbers from the shard
        main(['singlepoint', '--pred', sp_out2, '--summary', sp_sum,
              '--out', sp_metrics])
        with open(sp_metrics) as f:
            assert abs(json.load(f)['energy_mae'] - spm['energy_mae']) < 1e-15

        # 5. shift one stable prediction far above the hull → recall drops
        with gzip.open(out, 'rt') as f:
            blob = json.load(f)
        worst = min(e_hull, key=e_hull.get)             # the most-stable id
        blob['results'][worst]['energy'] += 1.0 * blob['results'][worst]['n_atoms']
        out2 = os.path.join(tmp, 'wbm_degraded.json.gz')
        with gzip.open(out2, 'wt') as f:
            json.dump(blob, f)
        main(['score', '--pred', out2, '--summary', summary_path,
              '--ref', ref_path, '--out', metrics_path])
        with open(metrics_path) as f:
            m2 = json.load(f)
        assert m2['e_form_mae'] > 0.1 and m2['recall'] < 1.0, m2
    print("ALL TESTS PASSED")


if __name__ == '__main__':
    test_wbm_eval()

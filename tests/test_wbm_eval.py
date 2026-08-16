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

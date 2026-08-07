"""Synthetic smoke test for train_ecenet_spice.py, focused on atom-budget batching.

No SPICE download needed: writes a small extended-XYZ file with a wide spread of
molecule sizes and runs the trainer over it.

The atom-budget path is the part worth pinning down. Under DDP every rank must
run the same number of batches or the collective in backward deadlocks — a hang,
not an error — so the rank-assignment invariant is checked directly across
simulated world sizes, alongside an end-to-end run.

Run:  python tests/test_spice_trainer.py
"""

import os
import sys  # repo root + scripts/ on path (imports ecenet and the scripts/ trainer)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))


import tempfile

import numpy as np
import torch
from train_ecenet_spice import atom_budget_batches, train_ecenet_spice

DTYPE = torch.float64
DEVICE = torch.device('cpu')
ELEMENTS = ['H', 'C', 'N', 'O']


def write_xyz(path, n_structures, seed=0, n_atoms_range=(3, 25)):
    """Extended XYZ in the format the SPICE parser expects, with a wide size spread."""
    rng = np.random.RandomState(seed)
    with open(path, 'w') as f:
        for _ in range(n_structures):
            na = rng.randint(*n_atoms_range)
            f.write(f"{na}\n")
            f.write(f"Properties=species:S:1:pos:R:3:forces:R:3 energy={rng.uniform(-20, 20):.6f}\n")
            pos = rng.uniform(-4, 4, size=(na, 3))
            frc = rng.uniform(-1, 1, size=(na, 3))
            for i in range(na):
                el = ELEMENTS[rng.randint(len(ELEMENTS))]
                f.write(f"{el} {pos[i,0]:.6f} {pos[i,1]:.6f} {pos[i,2]:.6f} "
                        f"{frc[i,0]:.6f} {frc[i,1]:.6f} {frc[i,2]:.6f}\n")
    return path


def test_atom_budget_ddp_invariant():
    """Every rank must get the same NUMBER of batches (else DDP deadlocks), no
    batch may exceed the budget, and no structure may be used twice per epoch."""
    rng = np.random.RandomState(0)
    n_atoms = rng.randint(3, 60, size=4000).astype(np.int64)
    BUDGET = 250
    for world_size in (1, 2, 4, 8):
        for epoch in (0, 1, 7):
            all_idx = np.random.RandomState(10 + epoch).choice(4000, 3000, replace=False)
            per_rank = [atom_budget_batches(all_idx, n_atoms, BUDGET, None,
                                            world_size, r, 10 + epoch)
                        for r in range(world_size)]
            counts = {len(b) for b in per_rank}
            assert len(counts) == 1, \
                f"world_size={world_size} epoch={epoch}: ranks disagree on batch " \
                f"count {counts} — this deadlocks DDP"
            over = [int(n_atoms[b].sum()) for rb in per_rank for b in rb
                    if n_atoms[b].sum() > BUDGET]
            assert not over, f"budget {BUDGET} exceeded: {over[:3]}"
            used = np.concatenate([np.concatenate(b) for b in per_rank])
            assert len(used) == len(set(used.tolist())), "a structure was used twice"
        n_b = counts.pop()
        totals = [int(n_atoms[np.concatenate(b)].sum()) for b in per_rank]
        spread = (max(totals) - min(totals)) / max(totals)
        assert spread < 0.05, f"ranks poorly balanced at world_size={world_size}: {spread:.1%}"
        print(f"  world_size={world_size}: {n_b} batches on every rank, "
              f"budget respected, rank load spread {spread:.1%}")


def test_atom_budget_respects_max_batch_count():
    n_atoms = np.full(500, 4, dtype=np.int64)          # tiny molecules
    idx = np.arange(500)
    b = atom_budget_batches(idx, n_atoms, 1000, 7, 1, 0, 0)   # budget would allow 250
    assert max(len(x) for x in b) <= 7, f"max_batch_count ignored: {max(len(x) for x in b)}"
    print(f"  max_batch_count caps structures/batch at 7 (budget alone would allow "
          f"{1000 // 4})")


def test_atom_budget_raises_when_no_full_round():
    n_atoms = np.full(4, 10, dtype=np.int64)
    try:
        atom_budget_batches(np.arange(4), n_atoms, 100, None, 8, 0, 0)   # 1 batch, 8 ranks
    except ValueError as e:
        assert 'world_size' in str(e)
        print(f"  too few batches for the world size raises: {str(e)[:58]}…")
    else:
        raise AssertionError("expected a ValueError when no full round exists")


def test_trainer_runs_with_atom_budget():
    """End-to-end: the trainer runs on both batching paths, and rejects a budget
    smaller than the largest structure rather than looping forever."""
    with tempfile.TemporaryDirectory() as d:
        train = write_xyz(os.path.join(d, 'train.xyz'), 60, seed=1)
        test = write_xyz(os.path.join(d, 'test.xyz'), 10, seed=2)
        common = dict(
            train_xyz=train, test_xyz=test, n_train=48, n_val=6, n_test=6,
            r_cut_edge=4.0, r_cut_neighbor=4.0, l_max=2, n_max=2, embed_dim=8,
            n_layers=1, n_max_d=4, m_max=2, n_epochs=2, lr=5e-3,
            eval_every=2, eval_batch_size=4, dtype=DTYPE, device=DEVICE,
            seed=0, verbose=False,
        )
        _, base = train_ecenet_spice(batch_size=4, **common)
        _, budg = train_ecenet_spice(batch_size=4, max_atoms_per_batch=60, **common)
        for r in (base, budg):
            assert np.isfinite(r['test_force_mae']), f"non-finite force MAE: {r}"

        # a budget below the largest structure can never be satisfied
        try:
            train_ecenet_spice(batch_size=4, max_atoms_per_batch=2, **common)
        except ValueError as e:
            assert 'largest training structure' in str(e)
        else:
            raise AssertionError("expected a ValueError for an impossible budget")
    print(f"  trainer runs on both paths (fixed F={base['test_force_mae']:.3f}, "
          f"atom-budget F={budg['test_force_mae']:.3f}); impossible budget rejected")


def test_tf32_is_a_noop_under_float64():
    """tf32 is a float32-only mode; under float64 it must warn, not silently
    change global torch state."""
    before = torch.backends.cuda.matmul.allow_tf32
    with tempfile.TemporaryDirectory() as d:
        train = write_xyz(os.path.join(d, 'train.xyz'), 24, seed=3)
        train_ecenet_spice(
            train_xyz=train, test_xyz=train, n_train=16, n_val=4, n_test=4,
            r_cut_edge=4.0, r_cut_neighbor=4.0, l_max=2, n_max=2, embed_dim=8,
            n_layers=1, n_max_d=4, m_max=2, n_epochs=1, batch_size=4, lr=5e-3,
            eval_every=1, eval_batch_size=4, dtype=torch.float64, device=DEVICE,
            seed=0, verbose=False, tf32=True)
    assert torch.backends.cuda.matmul.allow_tf32 == before, \
        "tf32=True under float64 should not touch torch's global TF32 state"
    print("  tf32=True under float64 leaves torch's TF32 state untouched")


if __name__ == '__main__':
    test_atom_budget_ddp_invariant()
    test_atom_budget_respects_max_batch_count()
    test_atom_budget_raises_when_no_full_round()
    test_trainer_runs_with_atom_budget()
    test_tf32_is_a_noop_under_float64()
    print("All tests passed.")

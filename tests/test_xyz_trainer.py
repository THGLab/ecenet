# Prototype, mainly implemented by Claude
"""
Tests for scripts/train_ecenet_xyz.py — the small-dataset trainer with
optional joint LES long-range training.

No data download needed: synthetic random periodic structures throughout.
Covers:
  1. end-to-end smoke, LES off (energy + force + stress);
  2. end-to-end smoke, LES on — including checkpoint save → resume (the LES
     head is built lazily by upstream, so resume exercises the
     materialise-then-load path) and best-state restore;
  3. finite-difference check of forces THROUGH the LES term (E = E_sr + E_lr
     on one graph) on a fresh model;
  4. ECENetCalculator.from_checkpoint refuses an LES checkpoint unless
     ignore_les=True.

LES-dependent tests skip cleanly when the optional `les` package is missing.

Run:  python tests/test_xyz_trainer.py     (from the repo root)
"""

import os
import sys  # repo root + scripts/ on path (imports ecenet and the scripts/ trainer)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import tempfile

import numpy as np
import torch
from train_ecenet_mptrj import compute_energy_reference
from train_ecenet_xyz import tensorize, train_ecenet_xyz

from ecenet import elements

DTYPE = torch.float64
DEVICE = torch.device('cpu')   # FD needs float64; MPS has no float64
Z_CHOICES = [1, 8, 11, 17]     # H, O, Na, Cl → n_types = 4


def _has_les():
    try:
        import les  # noqa: F401
        return True
    except ImportError:
        return False


def make_structures(n, seed=0, n_atoms_range=(4, 8), box=(7.0, 8.0)):
    """Random periodic structures with random energy/forces/stress (eV/Å³)."""
    rng = np.random.RandomState(seed)
    structs = []
    for _ in range(n):
        na = rng.randint(*n_atoms_range)
        L = rng.uniform(*box)
        cell = np.diag([L, L, L]).astype(np.float64)
        cell[0, 1] = rng.uniform(-0.5, 0.5)   # exercise triclinic shifts
        cell[1, 2] = rng.uniform(-0.5, 0.5)
        frac = rng.uniform(0, 1, size=(na, 3))
        structs.append({
            'numbers': rng.choice(Z_CHOICES, size=na).astype(np.int64),
            'positions': frac @ cell,
            'cell': cell,
            'pbc': True,
            'energy': float(rng.uniform(-5, 5) * na),
            'forces': rng.uniform(-1, 1, size=(na, 3)).astype(np.float64),
            'stress': rng.uniform(-0.05, 0.05, size=(3, 3)).astype(np.float64),
            'n_atoms': na,
        })
    return structs


COMMON = dict(
    l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
    r_cut_edge=4.0, r_cut_neighbor=3.5,
    dtype=DTYPE, device=DEVICE, seed=0, verbose=True,
)


def test_smoke_train():
    print("=== Smoke: end-to-end training, LES off (E + F + S) ===")
    _, les_module, results = train_ecenet_xyz(
        train_structures=make_structures(12, seed=1),
        test_structures=make_structures(3, seed=2),
        n_val=2, stress_weight=0.1,
        n_epochs=3, batch_size=4, lr=5e-3, **COMMON,
    )
    assert les_module is None
    for k in ('test_energy_mae', 'test_force_mae', 'test_stress_mae'):
        assert np.isfinite(results[k]), f"{k} not finite: {results[k]}"
    print(f"  results OK: E={results['test_energy_mae']:.3f} "
          f"F={results['test_force_mae']:.3f} S={results['test_stress_mae']:.3e}\n")


def test_smoke_train_les():
    if not _has_les():
        print("=== SKIP: LES smoke (optional `les` package not installed) ===\n")
        return
    print("=== Smoke: end-to-end training, LES on, + checkpoint resume ===")
    structs = make_structures(12, seed=4)
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, 'xyz_les.mdl')
        _, les_module, results = train_ecenet_xyz(
            train_structures=[dict(s) for s in structs], n_val=2,
            use_les=True, checkpoint_path=ckpt,
            n_epochs=2, batch_size=4, lr=5e-3, **COMMON,
        )
        assert les_module is not None
        assert any(p.requires_grad for p in les_module.parameters()), \
            "LES head has no trainable parameters (lazy build did not run)"
        for k in ('val_energy_mae', 'val_force_mae'):
            assert np.isfinite(results[k]), f"{k} not finite: {results[k]}"

        # Resume: fresh call restores model + LES + optimizer and continues.
        _, les2, results2 = train_ecenet_xyz(
            train_structures=[dict(s) for s in structs], n_val=2,
            use_les=True, checkpoint_path=ckpt,
            n_epochs=4, batch_size=4, lr=5e-3, **COMMON,
        )
        assert np.isfinite(results2['val_force_mae'])

        # use_les must match the checkpoint.
        try:
            train_ecenet_xyz(
                train_structures=[dict(s) for s in structs], n_val=2,
                use_les=False, checkpoint_path=ckpt,
                n_epochs=5, batch_size=4, lr=5e-3, **COMMON,
            )
            raise AssertionError("resume with use_les=False should have raised")
        except ValueError as e:
            assert 'use_les' in str(e)
    print("  LES smoke + resume OK\n")


def test_les_force_fd():
    """Forces from the joint graph (E_sr + E_lr) match finite differences."""
    if not _has_les():
        print("=== SKIP: LES force FD (optional `les` package not installed) ===\n")
        return
    print("=== FD: forces through E_sr + E_lr ===")
    from train_ecenet_mptrj import build_topology

    from ecenet import ECENet
    from ecenet.les import LESLongRange

    structs = make_structures(1, seed=7, n_atoms_range=(5, 6))
    s = structs[0]
    type_map = elements.build_type_map(int(z) for z in s['numbers'])

    torch.manual_seed(3)
    model = ECENet(n_types=len(type_map), r_cut_edge=4.0, r_cut_neighbor=3.5,
                   l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4
                   ).double().to(DEVICE)
    lr_mod = LESLongRange().double()
    types = torch.tensor([type_map[int(z)] for z in s['numbers']],
                         dtype=torch.long, device=DEVICE)
    cell_t = torch.tensor(s['cell'], dtype=DTYPE, device=DEVICE)

    def total_energy(pos_np, requires_grad=False):
        # topology rebuilt per evaluation so FD displacements stay consistent
        ei, ej, she, ni, nj, shn = build_topology(
            pos_np, s['cell'], True, 4.0, 3.5, DEVICE, DTYPE)
        pos = torch.tensor(pos_np, dtype=DTYPE, device=DEVICE,
                           requires_grad=requires_grad)
        e_sr, l0 = model.forward_pbc(pos, types, ei, ej, she, ni, nj, shn,
                                     return_embeddings=True, l0_only=True)
        e = e_sr + lr_mod(l0, pos, cell=cell_t).sum()
        return e, pos

    # materialise the lazy LES head, then perturb it away from zero-ish init
    with torch.no_grad():
        total_energy(s['positions'])
    for p in lr_mod.parameters():
        with torch.no_grad():
            p.add_(0.1 * torch.randn_like(p))

    e, pos = total_energy(s['positions'], requires_grad=True)
    forces = -torch.autograd.grad(e, pos)[0].cpu().numpy()

    h = 1e-5
    max_err = 0.0
    for a in range(min(3, s['n_atoms'])):
        for c in range(3):
            pp = s['positions'].copy(); pp[a, c] += h
            pm = s['positions'].copy(); pm[a, c] -= h
            with torch.no_grad():
                ep, _ = total_energy(pp)
                em, _ = total_energy(pm)
            f_fd = -(ep.item() - em.item()) / (2 * h)
            max_err = max(max_err, abs(f_fd - forces[a, c]))
    assert max_err < 1e-6, f"FD force mismatch through LES: {max_err:.2e}"
    print(f"  max |F_autograd - F_fd| = {max_err:.2e}  (incl. E_lr)\n")


def test_calculator_rejects_les_checkpoint():
    if not _has_les():
        print("=== SKIP: calculator LES rejection (optional `les` package "
              "not installed) ===\n")
        return
    print("=== Calculator: from_checkpoint refuses LES checkpoints ===")
    from ecenet.calculator import ECENetCalculator

    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, 'xyz_les.mdl')
        train_ecenet_xyz(
            train_structures=make_structures(8, seed=9), n_val=2,
            use_les=True, checkpoint_path=ckpt,
            n_epochs=1, batch_size=4, lr=5e-3, **COMMON,
        )
        try:
            ECENetCalculator.from_checkpoint(ckpt, device='cpu')
            raise AssertionError("from_checkpoint should refuse an LES checkpoint")
        except ValueError as e:
            assert 'LES' in str(e), f"unexpected error: {e}"
        calc = ECENetCalculator.from_checkpoint(ckpt, device='cpu', ignore_les=True)
        assert calc is not None
    print("  rejected without ignore_les, loads (SR-only) with it\n")


def test_tensorize_keeps_cell():
    print("=== tensorize: cell kept for periodic, None otherwise ===")
    structs = make_structures(2, seed=5)
    structs[1]['cell'] = None
    structs[1]['pbc'] = False
    type_map = elements.build_type_map(
        int(z) for s in structs for z in s['numbers'])
    e_ref = compute_energy_reference(structs, type_map)
    data = tensorize(structs, type_map, e_ref, 4.0, 3.5, 1.0, DTYPE, DEVICE)
    assert data[0]['cell'] is not None and data[0]['cell'].shape == (3, 3)
    assert data[1]['cell'] is None
    print("  OK\n")


if __name__ == '__main__':
    test_tensorize_keeps_cell()
    test_smoke_train()
    test_les_force_fd()
    test_smoke_train_les()
    test_calculator_rejects_les_checkpoint()
    print("ALL TESTS PASSED")

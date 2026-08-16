#Prototype, mainly implemented by Claude
"""

Behavioural tests for ECENetCalculator.

These pin the *observable* contract of the calculator and of `from_checkpoint`.
ecenet/calculator.py is dataset-agnostic: it reads only generic, self-describing
checkpoint keys (element mapping, reference energies, units) — no knowledge of
rMD17 / MD22 / SPICE / MPtrj.

What is locked down:
  * unit handling (kcal/mol → eV) via `energy_units`
  * per-element reference energies are added back atom-by-atom
  * the training mean energy is added back (in eV)
  * the element→type mapping reaches `calculate`
  * unsupported elements raise
  * from_checkpoint reconstructs the model + metadata from a saved dict, building
    energy_reference from an 'e_ref' array via the checkpoint's OWN mapping, and
    raising if no element mapping is present

Run:  python tests/test_calculator.py     (also collectable by pytest)
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

import numpy as np
import torch
from ase import Atoms
from ase import units as ase_units

from ecenet import ECENet
from ecenet.calculator import ECENetCalculator
from ecenet.equivariant import RealSpaceNonlinearity

torch.manual_seed(0)

_KCAL = ase_units.kcal / ase_units.mol

# Small float64 model config — same idiom as test_ecenet.py.
_HPARAMS = dict(
    r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
)


def _tiny_model(n_types):
    return ECENet(n_types=n_types, **_HPARAMS).double()


def _mol(symbols=('H', 'C', 'O')):
    """A small non-periodic molecule with the given elements."""
    pos = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0]]
    return Atoms(symbols=list(symbols), positions=pos[:len(symbols)])


def _periodic_mol(symbols=('H', 'C', 'O'), a=12.0):
    """Same atoms in a cubic box big enough that r_cut <= L/2 (no MIC warning)."""
    atoms = _mol(symbols)
    atoms.set_cell([a, a, a])
    atoms.set_pbc(True)
    return atoms


def _energy(calc, atoms):
    a = atoms.copy()
    a.calc = calc
    return a.get_potential_energy()


def _save_ckpt(path, model, n_types, **extra):
    """Write a checkpoint dict the way the trainers do (hparams + state + extras)."""
    hp = dict(n_types=n_types, n_mp=1, **_HPARAMS)
    ckpt = {'model': model.state_dict(), 'hparams': hp}
    ckpt.update(extra)
    torch.save(ckpt, path)


# ── unit handling ────────────────────────────────────────────────────────────

def test_energy_units_kcal_vs_ev_scaling():
    """A kcal/mol calculator scales the same model output by kcal→eV vs eV."""
    model = _tiny_model(3)
    e2t = {'H': 0, 'C': 1, 'O': 2}
    atoms = _mol()

    calc_ev   = ECENetCalculator(model, element_to_type=e2t, energy_units='eV')
    calc_kcal = ECENetCalculator(model, element_to_type=e2t, energy_units='kcal/mol')

    assert abs(calc_ev._to_ev - 1.0) < 1e-15
    assert abs(calc_kcal._to_ev - _KCAL) < 1e-15

    e_ev   = _energy(calc_ev, atoms)
    e_kcal = _energy(calc_kcal, atoms)
    # Same raw model energy, just a different unit conversion factor.
    assert abs(e_kcal - e_ev * _KCAL) < 1e-9
    print(f"  units: E(eV)={e_ev:.4f}  E(kcal→eV)={e_kcal:.6f}  ratio={_KCAL:.5f}")


# ── reference energies ───────────────────────────────────────────────────────

def test_energy_reference_added_per_atom():
    """Per-element references shift the energy by exactly Σ_atoms ref[element]."""
    model = _tiny_model(3)
    e2t = {'H': 0, 'C': 1, 'O': 2}
    eref = {'H': 1.5, 'C': -2.0, 'O': 0.25}   # eV/atom
    atoms = _mol(('H', 'C', 'O'))

    base = ECENetCalculator(model, element_to_type=e2t, energy_reference={})
    ref  = ECENetCalculator(model, element_to_type=e2t, energy_reference=eref)

    shift = _energy(ref, atoms) - _energy(base, atoms)
    expected = sum(eref[s] for s in atoms.get_chemical_symbols())
    assert abs(shift - expected) < 1e-9, f"{shift} != {expected}"
    print(f"  energy_reference: shift={shift:.4f} eV == Σref={expected:.4f}")


def test_energy_mean_added_in_ev():
    """The training mean energy is added back, converted to eV by the unit factor."""
    model = _tiny_model(3)
    e2t = {'H': 0, 'C': 1, 'O': 2}
    atoms = _mol()

    base = ECENetCalculator(model, element_to_type=e2t,
                            energy_units='kcal/mol', energy_mean=0.0)
    shifted = ECENetCalculator(model, element_to_type=e2t,
                               energy_units='kcal/mol', energy_mean=10.0)

    shift = _energy(shifted, atoms) - _energy(base, atoms)
    assert abs(shift - 10.0 * _KCAL) < 1e-9, f"{shift} != {10.0 * _KCAL}"
    print(f"  energy_mean: shift={shift:.6f} eV == 10*kcal={10.0 * _KCAL:.6f}")


# ── element mapping ──────────────────────────────────────────────────────────

def test_unsupported_element_raises():
    model = _tiny_model(3)
    calc = ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2})
    atoms = _mol(('H', 'C', 'O'))
    atoms[1].symbol = 'Fe'  # not in the mapping
    a = atoms.copy(); a.calc = calc
    try:
        a.get_potential_energy()
    except ValueError as e:
        assert 'Fe' in str(e)
        print(f"  unsupported element raises: {str(e)[:48]}…")
        return
    raise AssertionError("expected ValueError for unsupported element")


def test_forces_finite_and_shaped():
    model = _tiny_model(3)
    calc = ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2})
    a = _mol(); a.calc = calc
    f = a.get_forces()
    assert f.shape == (3, 3) and np.isfinite(f).all()
    print(f"  forces: shape={f.shape} |F|max={np.abs(f).max():.3f}")


# ── periodic path (_compute_pbc / _compute_stress) ───────────────────────────

def test_pbc_energy_forces_stress_shapes():
    """The periodic path produces finite energy, (N,3) forces, and a 6-vector
    Voigt stress (exercises _compute_pbc and the strain-based _compute_stress)."""
    model = _tiny_model(3)
    calc = ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2})
    a = _periodic_mol(); a.calc = calc
    e = a.get_potential_energy()
    f = a.get_forces()
    s = a.get_stress()                      # requests 'stress' → strain path
    assert np.isfinite(e)
    assert f.shape == (3, 3) and np.isfinite(f).all()
    assert s.shape == (6,) and np.isfinite(s).all()
    print(f"  pbc: E={e:.4f} |F|max={np.abs(f).max():.3f} |σ|max={np.abs(s).max():.3e}")


def test_pbc_forces_match_finite_difference():
    """Autograd forces on the periodic path agree with a central difference on
    the energy (validates _compute_pbc's grad wiring after the refactor)."""
    model = _tiny_model(3)
    calc = ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2})
    atoms = _periodic_mol(); atoms.calc = calc
    f = atoms.get_forces()

    eps = 1e-5
    fd = np.zeros_like(f)
    for i in range(len(atoms)):
        for d in range(3):
            a = atoms.copy(); a.calc = calc
            p = a.get_positions(); p[i, d] += eps; a.set_positions(p)
            ep = a.get_potential_energy()
            p[i, d] -= 2 * eps; a.set_positions(p)
            em = a.get_potential_energy()
            fd[i, d] = -(ep - em) / (2 * eps)
    err = np.abs(f - fd).max()
    assert err < 1e-5, f"PBC forces vs finite-difference mismatch: {err:.2e}"
    print(f"  pbc forces match finite-difference (max err {err:.1e})")


# ── from_checkpoint ──────────────────────────────────────────────────────────

def test_from_checkpoint_type_to_idx_fallback():
    """Atomic-number 'type_to_idx' fallback (converted to symbols) + kcal/mol +
    energy_mean. Trainers now write 'element_to_type', but the calculator still
    accepts an atomic-number-keyed map for convenience."""
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'tti.mdl')
        _save_ckpt(path, model, 3,
                   type_to_idx={1: 0, 6: 1, 8: 2},
                   energy_units='kcal/mol',
                   energy_mean=3.0)
        calc = ECENetCalculator.from_checkpoint(path)

    assert calc.element_to_type == {'H': 0, 'C': 1, 'O': 2}
    assert abs(calc._to_ev - _KCAL) < 1e-15
    assert abs(calc._energy_mean_ev - 3.0 * _KCAL) < 1e-12
    e = _energy(calc, _mol())
    assert np.isfinite(e)
    print(f"  from_checkpoint(type_to_idx fallback): map={calc.element_to_type} E={e:.4f}")


def test_from_checkpoint_defaults_to_ev_without_units():
    """No 'energy_units' key → defaults to eV (no dataset-based unit guessing)."""
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'nounits.mdl')
        _save_ckpt(path, model, 3, type_to_idx={1: 0, 6: 1, 8: 2})  # no energy_units
        calc = ECENetCalculator.from_checkpoint(path)
    assert abs(calc._to_ev - 1.0) < 1e-15
    print(f"  from_checkpoint(no units key): defaults eV → _to_ev={calc._to_ev:.3f}")


def test_from_checkpoint_dtype_inferred_from_weights():
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'dt.mdl')
        _save_ckpt(path, model, 3, type_to_idx={1: 0, 6: 1, 8: 2}, energy_units='eV')
        calc = ECENetCalculator.from_checkpoint(path)  # dtype=None
    assert calc.dtype == torch.float64
    print(f"  from_checkpoint(dtype): inferred {calc.dtype}")


def test_from_checkpoint_spice_style():
    """SPICE-style: a symbol-keyed 'element_to_type' + an 'e_ref' array. The
    calculator builds energy_reference from e_ref indexed by the checkpoint's
    OWN mapping — no import from the training scripts, no hardcoded element list.
    """
    e2t = {'H': 0, 'C': 1, 'O': 2}
    n = len(e2t)
    model = _tiny_model(n)
    e_ref = np.array([0.5, -1.0, 2.5])   # eV/atom, indexed by type idx
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'spice.mdl')
        _save_ckpt(path, model, n, element_to_type=e2t, e_ref=e_ref,
                   energy_units='eV')
        calc = ECENetCalculator.from_checkpoint(path)

    assert calc.element_to_type == e2t
    assert abs(calc._to_ev - 1.0) < 1e-15
    # energy_reference built generically from e_ref via the checkpoint's mapping
    for sym, idx in e2t.items():
        assert abs(calc.energy_reference[sym] - e_ref[idx]) < 1e-12
    print(f"  from_checkpoint(SPICE-style): refs={calc.energy_reference}")


def test_from_checkpoint_missing_mapping_raises():
    """A checkpoint with neither 'element_to_type' nor 'type_to_idx' fails loudly."""
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'nomap.mdl')
        _save_ckpt(path, model, 3, energy_units='eV')  # no mapping at all
        try:
            ECENetCalculator.from_checkpoint(path)
        except ValueError as e:
            assert 'element mapping' in str(e)
            print(f"  from_checkpoint(no mapping): raises — {str(e)[:46]}…")
            return
    raise AssertionError("expected ValueError when no element mapping is stored")


def test_from_checkpoint_missing_hparams_raises():
    """A checkpoint without stored 'hparams' fails loudly (no reconstruction)."""
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'nohp.mdl')
        # Write a checkpoint dict deliberately lacking 'hparams'.
        torch.save({'model': model.state_dict(),
                    'element_to_type': {'H': 0, 'C': 1, 'O': 2}}, path)
        try:
            ECENetCalculator.from_checkpoint(path)
        except ValueError as e:
            assert 'hparams' in str(e)
            print(f"  from_checkpoint(no hparams): raises — {str(e)[:46]}…")
            return
    raise AssertionError("expected ValueError when no hparams are stored")


def test_legacy_edge_mp_checkpoint_raises():
    """Checkpoints trained with the removed mp_type='edge' message passing
    (identifiable by mp_layers.*.W_msg weights) must fail loudly: those weights
    have no counterpart in the current MP layers, so a tolerant load would
    silently run a randomly initialised MP layer and return wrong energies.

    The fixture mirrors the layout of a real dev-era edge-MP checkpoint
    (hparams incl. the removed n_dist_embed/n_dist_basis keys, W_msg in the
    state dict); the 8.6 MB real one this replaced was dropped from examples/.
    The rejection fires on the W_msg key before the model is built, so a
    minimal state dict exercises the same path."""
    hp = dict(n_types=3, r_cut_edge=6.0, r_cut_neighbor=5.0, l_max=2, n_max=4,
              embed_dim=8, n_layers=2, n_max_d=4, n_grid=None,
              cutoff_type='cosine', activation='silu', use_nonlinearity=True,
              output_hidden_dims=None, analytic_ace_basis=True,
              n_dist_embed=0, n_mp=2, n_dist_basis=8)
    state = {'mp_layers.0.W_msg': torch.zeros(3, 3, 8, dtype=torch.float64)}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'legacy_edge_mp.mdl')
        torch.save({'model': state, 'hparams': hp,
                    'element_to_type': {'H': 0, 'C': 1, 'O': 2},
                    'energy_units': 'kcal/mol'}, path)
        try:
            ECENetCalculator.from_checkpoint(path)
        except ValueError as e:
            assert 'W_msg' in str(e) and 'edge' in str(e), f"unhelpful message: {e}"
            print(f"  legacy edge-MP checkpoint rejected: {str(e)[:64]}…")
        else:
            raise AssertionError("expected a ValueError for a legacy edge-MP checkpoint")


def test_trainers_save_every_architecture_hparam():
    """Every ECENet constructor argument that changes the architecture must be in
    the 'hparams' each trainer saves — from_checkpoint rebuilds the model from
    that dict alone, so a missing key silently substitutes a default (and, when
    it changes tensor shapes, makes the checkpoint fail to load at all)."""
    import ast
    import inspect

    from ecenet import ECENet

    # Args that describe the *architecture*. Excluded: n_types (data-derived) and
    # options that do not affect the built module tree.
    required = set(inspect.signature(ECENet).parameters) - {'self', 'n_types'}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trainers = ('train_ecenet.py', 'train_ecenet_spice.py', 'train_ecenet_mptrj.py')

    for name in trainers:
        tree = ast.parse(open(os.path.join(root, 'scripts', name)).read())
        saved = set()
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and getattr(call.func, 'id', None) == 'dict':
                keys = {k.arg for k in call.keywords if k.arg}
                if {'n_types', 'l_max'} <= keys:      # the hparams dict
                    saved |= keys
        assert saved, f"{name}: no hparams dict found"
        missing = sorted(required - saved)
        assert not missing, f"{name}: hparams omits architecture args {missing}"
        stale = sorted(saved - required - {'n_types'})
        assert not stale, f"{name}: hparams saves non-ECENet args {stale}"
    print(f"  all {len(trainers)} trainers save every architecture hparam "
          f"({len(required)} args)")


def test_legacy_pre_scale_buffers_are_dropped():
    """RealSpaceNonlinearity used to carry fixed pre_scale=1 / pre_shift=0 buffers
    and apply them before the activation — an exact identity, never learnable.
    They are gone, so a checkpoint that still has them must load and give the same
    energy, not trip the unexpected-key check."""
    model = _tiny_model(3)
    state = dict(model.state_dict())
    assert not any('pre_scale' in k for k in state), "buffers should no longer exist"
    # forge a pre-removal checkpoint: identity buffers on every nonlinearity
    n_added = 0
    for name, mod in model.named_modules():
        if isinstance(mod, RealSpaceNonlinearity):
            w = mod.n_features
            state[f'{name}.pre_scale'] = torch.ones(w, 1, dtype=torch.float64)
            state[f'{name}.pre_shift'] = torch.zeros(w, 1, dtype=torch.float64)
            n_added += 2
    assert n_added > 0, "no nonlinearities found to forge buffers on"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'legacy_affine.mdl')
        torch.save({'model': state, 'hparams': dict(n_types=3, n_mp=1, **_HPARAMS),
                    'element_to_type': {'H': 0, 'C': 1, 'O': 2}}, path)
        calc = ECENetCalculator.from_checkpoint(path)
    e_legacy = _energy(calc, _mol())
    e_direct = _energy(ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2}),
                       _mol())
    assert e_legacy == e_direct, \
        f"dropping the identity buffers changed the energy: {e_legacy} vs {e_direct}"
    print(f"  legacy checkpoint with {n_added} pre_scale/pre_shift buffers loads, "
          f"energy unchanged ({e_legacy:.6f} eV)")


def test_architecture_mismatch_raises():
    """A checkpoint whose weights disagree with the architecture rebuilt from
    'hparams' must raise rather than load a partly random model."""
    hp = dict(n_types=3, n_mp=2, **_HPARAMS)
    model = ECENet(**hp).double()
    # drop a genuine MP parameter → the rebuilt model has no weights for it
    dropped = 'mp_layers.0.msg_up.weights'
    assert dropped in model.state_dict(), f"{dropped} is no longer a parameter"
    state = {k: v for k, v in model.state_dict().items() if k != dropped}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'mismatch.mdl')
        torch.save({'model': state, 'hparams': hp,
                    'element_to_type': {'H': 0, 'C': 1, 'O': 2}}, path)
        try:
            ECENetCalculator.from_checkpoint(path)
        except ValueError as e:
            assert 'do not match' in str(e) and dropped in str(e), f"unhelpful message: {e}"
            print(f"  architecture mismatch rejected: {str(e)[:60]}…")
        else:
            raise AssertionError("expected a ValueError for mismatched weights")


def _direct_all_images(model, pos, cell, types_list):
    """Reference: forward_pbc on the all-images topology (the lists are
    themselves ASE-verified in test_mptrj_trainer.py)."""
    from ecenet.radial import torch_neighbor_list
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    ct = torch.tensor(cell, dtype=torch.float64)
    types = torch.tensor(types_list, dtype=torch.long)
    ei, ej, she = torch_neighbor_list(pt.detach(), ct, model.r_cut_edge)
    ni, nj, shn = torch_neighbor_list(pt.detach(), ct, model.r_cut_neighbor)
    e = model.forward_pbc(pt, types, ei, ej, she, ni, nj, shn)
    f = -torch.autograd.grad(e, pt)[0]
    return e.item(), f.numpy(), len(ei), pt


def test_pbc_small_cell_uses_all_images_topology():
    """Small crystal cells (cutoff > half the minimum perpendicular width —
    ~97% of MPtrj/WBM frames): the calculator must build the same all-images
    topology the trainers train on, self-image edges included. The
    minimum-image shortcut drops periodic-image edges there — this pins the
    automatic dispatch by matching a direct forward_pbc evaluation."""
    model = _tiny_model(2)
    rng = np.random.RandomState(0)
    cell = np.array([[4.0, 0.3, 0.0], [0.0, 4.2, 0.4], [0.0, 0.0, 3.8]])
    pos = rng.uniform(0, 1, (4, 3)) @ cell
    atoms = Atoms(symbols=['H', 'C', 'H', 'C'], positions=pos, cell=cell,
                  pbc=True)
    atoms.calc = ECENetCalculator(model, device=torch.device('cpu'),
                                  dtype=torch.float64,
                                  element_to_type={'H': 0, 'C': 1})
    e_calc = atoms.get_potential_energy()
    f_calc = atoms.get_forces()

    e_ref, f_ref, n_edges, pt = _direct_all_images(model, pos, cell,
                                                   [0, 1, 0, 1])
    assert abs(e_ref - e_calc) < 1e-10, (e_ref, e_calc)
    assert np.abs(f_ref - f_calc).max() < 1e-10
    # the regression: MIC finds strictly fewer edges in this cell, so the old
    # minimum-image path could not have produced this energy
    mi, _, _ = atoms.calc._gpu_neighbor_list(pt.detach(), cell,
                                             model.r_cut_edge)
    assert n_edges > len(mi), (n_edges, len(mi))
    print(f"  small cell: all-images topology dispatched "
          f"({n_edges} edges vs {len(mi)} under MIC), E/F match direct")


def test_pbc_large_cell_paths_agree():
    """Where MIC is valid (cutoff ≤ half the cell width) the two list
    flavours are identical, so the fast minimum-image path must reproduce
    the all-images reference exactly."""
    model = _tiny_model(2)
    rng = np.random.RandomState(1)
    cell = np.diag([12.0, 12.5, 13.0])
    pos = rng.uniform(0, 1, (6, 3)) @ cell
    atoms = Atoms(symbols=['H', 'C', 'H', 'C', 'H', 'C'], positions=pos,
                  cell=cell, pbc=True)
    atoms.calc = ECENetCalculator(model, device=torch.device('cpu'),
                                  dtype=torch.float64,
                                  element_to_type={'H': 0, 'C': 1})
    e_calc = atoms.get_potential_energy()
    f_calc = atoms.get_forces()
    e_ref, f_ref, _, _ = _direct_all_images(model, pos, cell,
                                            [0, 1, 0, 1, 0, 1])
    assert abs(e_ref - e_calc) < 1e-10, (e_ref, e_calc)
    assert np.abs(f_ref - f_calc).max() < 1e-10
    print("  large cell: MIC fast path == all-images reference")


if __name__ == '__main__':
    print("ECENetCalculator behaviour")
    test_pbc_small_cell_uses_all_images_topology()
    test_pbc_large_cell_paths_agree()
    test_energy_units_kcal_vs_ev_scaling()
    test_energy_reference_added_per_atom()
    test_energy_mean_added_in_ev()
    test_unsupported_element_raises()
    test_forces_finite_and_shaped()
    test_pbc_energy_forces_stress_shapes()
    test_pbc_forces_match_finite_difference()
    test_from_checkpoint_type_to_idx_fallback()
    test_from_checkpoint_defaults_to_ev_without_units()
    test_from_checkpoint_dtype_inferred_from_weights()
    test_from_checkpoint_spice_style()
    test_from_checkpoint_missing_mapping_raises()
    test_from_checkpoint_missing_hparams_raises()
    test_legacy_edge_mp_checkpoint_raises()
    test_trainers_save_every_architecture_hparam()
    test_legacy_pre_scale_buffers_are_dropped()
    test_architecture_mismatch_raises()
    print("All tests passed.")

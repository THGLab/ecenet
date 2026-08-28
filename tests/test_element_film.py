"""Tests for the element(+distance)-conditioned FiLM gate (ecenet/film.py),
wired into ECENet via ``element_film=True``.

The gate runs once, on the freshly built edge features in the bond frame, before
the equivariant layer stack.

Checks: γ=1 / identity at init, SO(3) invariance with an active gate, the
element-only (no radial leg) variant, the per-(channel, m) scale and its
structural-zero mask, the m=0-only shift head, and finite forces.

Run:  python tests/test_element_film.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import warnings

import torch

from ecenet import ECENet
from ecenet.film import ElementFiLM

torch.manual_seed(0)
DTYPE = torch.float64
N_TYPES = 4
COMMON = dict(
    n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
)


def random_structure(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=DTYPE) * 1.8
    types = torch.randint(0, N_TYPES, (n,), generator=g)
    return pos, types


def rand_rotation(seed=1):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=DTYPE))
    Q = Q * torch.sign(torch.diag(R))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def activate(film):
    """Perturb the gate's last layer so γ != 1 (it is zero-init = identity)."""
    with torch.no_grad():
        film.mlp[-1].weight.normal_(std=0.1)
        film.mlp[-1].bias.normal_(std=0.1)


def test_gate_identity_at_init():
    """Zero-init last layer → γ exactly 1, β absent (scale-only)."""
    f = ElementFiLM(n_features=8, n_types=N_TYPES, n_rbf=6, shift=False).double()
    gamma, beta = f(torch.tensor([0, 1, 2]), torch.tensor([1, 1, 3]),
                    torch.randn(3, 6, dtype=DTYPE))
    assert beta is None, "scale-only gate should not return a shift"
    assert torch.allclose(gamma, torch.ones_like(gamma)), "γ must be 1 at init"
    print("  ElementFiLM γ=1 at init, scale-only (β=None)")


def test_model_identity_at_init():
    """element_film=True must be a no-op at init (same model, gate on vs off)."""
    pos, types = random_structure(seed=2)
    for kw in (dict(film_n_rbf=6), dict(film_n_rbf=0), dict(film_n_rbf=6, film_per_m=True),
               dict(film_n_rbf=6, film_shift=True),
               dict(film_n_rbf=6, film_per_m=True, film_shift=True)):
        m = ECENet(**COMMON, element_film=True, **kw).double()
        e_on = m(pos, types)
        saved, m.element_film = m.element_film, None
        e_off = m(pos, types)
        m.element_film = saved
        err = (e_on - e_off).abs().item()
        assert err == 0.0, f"FiLM not identity at init for {kw}: {err:.2e}"
    print("  ECENet(element_film) identity at init (gate on == off, all variants)")


def test_so3_invariance_active():
    """With an active gate (γ != 1), energy stays SO(3)-invariant — γ depends
    only on the rotation-invariant types and distance, and scales per channel."""
    pos, types = random_structure(seed=3)
    for n_rbf, label in [(6, "element+distance"), (0, "element-only")]:
        m = ECENet(**COMMON, element_film=True, film_n_rbf=n_rbf).double()
        e_init = m(pos, types)
        activate(m.element_film)
        e0 = m(pos, types)
        err = (e0 - m(pos @ rand_rotation().T, types)).abs().item()
        # γ is a function of invariants (types, dist), so this is exact in theory;
        # the residual is float noise from recomputing dist under rotation, which
        # the gate MLP amplifies (looser than the base model's ~1e-12).
        assert err < 1e-8, f"{label} FiLM breaks SO(3): {err:.2e}"
        assert (e0 - e_init).abs().item() > 1e-6, f"{label} gate had no effect"
        print(f"  active FiLM ({label}): SO(3) err {err:.1e}, gate has effect")


def test_element_only_gate_ignores_distance():
    """film_n_rbf=0 → the gate has no radial leg, so γ depends on the element pair
    alone: the same pair must get the same γ at any separation."""
    m = ECENet(**COMMON, element_film=True, film_n_rbf=0).double()
    activate(m.element_film)
    assert m.element_film.n_rbf == 0
    ti, tj = torch.tensor([0, 0]), torch.tensor([1, 1])
    g_near, _ = m._film_params(ti, tj, torch.tensor([1.0, 1.0], dtype=DTYPE))
    g_far,  _ = m._film_params(ti, tj, torch.tensor([4.5, 4.5], dtype=DTYPE))
    assert torch.equal(g_near, g_far), "element-only gate must not depend on distance"
    # ...while the element+distance variant does
    m2 = ECENet(**COMMON, element_film=True, film_n_rbf=6).double()
    activate(m2.element_film)
    g2_near, _ = m2._film_params(ti, tj, torch.tensor([1.0, 1.0], dtype=DTYPE))
    g2_far,  _ = m2._film_params(ti, tj, torch.tensor([4.5, 4.5], dtype=DTYPE))
    assert not torch.equal(g2_near, g2_far), "radial-leg gate should vary with distance"
    print("  film_n_rbf=0: γ is distance-independent; film_n_rbf=6: γ varies with r")


def test_per_m_gate_shape_and_mask():
    """film_per_m → γ is (E, C, n_angular), = 1 at init, and pinned to 1 on the
    structural-zero slots (m > l of that channel) even with an active gate."""
    m = ECENet(**COMMON, element_film=True, film_n_rbf=6, film_per_m=True).double()
    C, M = m.n_features_per_m, m.n_angular
    args = (torch.tensor([0, 1, 2]), torch.tensor([1, 1, 3]),
            torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE))
    gamma, beta = m._film_params(*args)
    assert beta is None, "scale-only unless film_shift"
    assert gamma.shape == (3, C, M), f"expected (3, {C}, {M}), got {tuple(gamma.shape)}"
    assert torch.allclose(gamma, torch.ones_like(gamma)), "γ must be 1 at init"

    activate(m.element_film)
    gamma, _ = m._film_params(*args)
    # channel c carries degree l_of_c = c % (l_max+1); modes m > l are structural zeros
    l_of_c = torch.arange(C) % (COMMON['l_max'] + 1)
    invalid = torch.arange(M)[None, :] > l_of_c[:, None]              # (C, M)
    assert (gamma[:, invalid] == 1).all(), "invalid (m > l) slots must stay at γ=1"
    assert not torch.allclose(gamma[:, ~invalid], torch.ones_like(gamma[:, ~invalid])), \
        "active gate should move γ on the real modes"
    # a per-m gate really is per-m: γ varies across m within a channel
    valid_ch = (l_of_c == COMMON['l_max']).nonzero()[0].item()        # a channel with all m valid
    assert gamma[0, valid_ch].std() > 0, "γ should differ across m within a channel"
    print(f"  film_per_m: γ{tuple(gamma.shape)}, invalid slots pinned to 1, varies across m")


def test_per_m_so3_invariance():
    """A per-(channel, m) scale is still exactly equivariant: A_cos and A_sin of a
    mode share γ, which commutes with the bond frame's per-mode SO(2) rotation."""
    pos, types = random_structure(seed=3)
    m = ECENet(**COMMON, element_film=True, film_n_rbf=6, film_per_m=True).double()
    e_init = m(pos, types)
    activate(m.element_film)
    e0 = m(pos, types)
    err = (e0 - m(pos @ rand_rotation().T, types)).abs().item()
    assert err < 1e-8, f"per-m FiLM breaks SO(3): {err:.2e}"
    assert (e0 - e_init).abs().item() > 1e-6, "per-m gate had no effect"
    print(f"  per-m FiLM: SO(3) err {err:.1e}, gate has effect")


def test_shift_head_is_m0_only():
    """film_shift → β is an extra head on the same MLP: (E, C), zero at init, and
    applied to the m=0 slot of A_cos ONLY (m>0 and the whole sin component are
    untouched — a shift there would break equivariance / fill a structural zero)."""
    for per_m in (False, True):
        m = ECENet(**COMMON, element_film=True, film_n_rbf=6,
                   film_shift=True, film_per_m=per_m).double()
        C, M = m.n_features_per_m, m.n_angular
        # one MLP, not two: the last layer just got n_features wider
        assert len(m.element_film.mlp) == len(ElementFiLM(C, N_TYPES, n_rbf=6, shift=False).mlp)
        assert m.element_film.mlp[-1].out_features == C * (M if per_m else 1) + C

        args = (torch.tensor([0, 1]), torch.tensor([1, 2]),
                torch.tensor([2.0, 3.0], dtype=DTYPE))
        _, beta = m._film_params(*args)
        assert beta.shape == (2, C), f"β must be (E, C), got {tuple(beta.shape)}"
        assert (beta == 0).all(), "β must be 0 at init"

        activate(m.element_film)
        gamma, beta = m._film_params(*args)
        assert beta.abs().max() > 0, "active gate should produce a nonzero β"
        A = torch.randn(2, C, M, dtype=DTYPE)
        cos_b, sin_b = m._film_apply(A.clone(), A.clone(), gamma, beta)
        cos_0, sin_0 = m._film_apply(A.clone(), A.clone(), gamma, None)   # γ only
        d = (cos_b - cos_0).abs()
        assert d[..., 0].max() > 0, "β must move the m=0 slot"
        assert (d[..., 1:] == 0).all(), "β must NOT touch m>0 slots"
        assert (sin_b == sin_0).all(), "β must NOT touch the sin component"
        print(f"  film_shift (per_m={per_m}): β{tuple(beta.shape)} on m=0 cos only, one shared MLP")


def test_shift_so3_invariance():
    """β on m=0 is an invariant scalar on the invariant mode → still exactly
    SO(3)-invariant, and it moves the output."""
    pos, types = random_structure(seed=3)
    for per_m in (False, True):
        m = ECENet(**COMMON, element_film=True, film_n_rbf=6,
                   film_shift=True, film_per_m=per_m).double()
        e_init = m(pos, types)
        activate(m.element_film)
        e0 = m(pos, types)
        err = (e0 - m(pos @ rand_rotation().T, types)).abs().item()
        assert err < 1e-8, f"film_shift breaks SO(3) (per_m={per_m}): {err:.2e}"
        assert (e0 - e_init).abs().item() > 1e-6, f"shift had no effect (per_m={per_m})"
        print(f"  film_shift (per_m={per_m}): SO(3) err {err:.1e}, has effect")


def test_gate_runs_on_every_forward_path():
    """The gate sits at the top of _run_equivariant_layers, so it must apply on
    the batched training path too — not just the single-structure forward."""
    pos, types = random_structure(seed=5)
    m = ECENet(**COMMON, element_film=True, film_n_rbf=6).double()
    activate(m.element_film)
    e_single = m(pos, types)
    e_batch = m.forward_batch_multi([pos, pos], [types, types])
    assert (e_batch - e_single).abs().max().item() < 1e-10, \
        "batched path disagrees with the single-structure path under an active gate"
    print(f"  gate applies on forward and forward_batch_multi (agree to "
          f"{(e_batch - e_single).abs().max().item():.1e})")


def test_ignored_flags_warn():
    """FiLM knobs with element_film=False configure a gate that is never built
    (element_film defaults to True, so the off state must be explicit here)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ECENet(**COMMON, element_film=False, film_n_rbf=6, film_per_m=True)
        assert any('film_n_rbf' in str(x.message) for x in w), "expected an ignored-flag warning"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ECENet(**COMMON, element_film=True, film_n_rbf=6)
        assert not any('film' in str(x.message) for x in w), "should not warn when the gate is on"
    print("  FiLM knobs warn when element_film=False, stay quiet when it is on")


def test_forces_finite():
    pos, types = random_structure(seed=4)
    for per_m in (False, True):
        for shift in (False, True):
            m = ECENet(**COMMON, element_film=True, film_n_rbf=6,
                       film_per_m=per_m, film_shift=shift).double()
            activate(m.element_film)
            p = pos.clone().requires_grad_(True)
            e = m(p, types)
            f = -torch.autograd.grad(e, p, create_graph=True)[0]
            assert torch.isfinite(e) and f.shape == pos.shape and torch.isfinite(f).all()
            print(f"  forces finite with active FiLM (per_m={per_m}, shift={shift}): "
                  f"|F|max={f.abs().max():.3f}")


if __name__ == "__main__":
    print("ElementFiLM tests")
    test_gate_identity_at_init()
    test_model_identity_at_init()
    test_so3_invariance_active()
    test_element_only_gate_ignores_distance()
    test_per_m_gate_shape_and_mask()
    test_per_m_so3_invariance()
    test_shift_head_is_m0_only()
    test_shift_so3_invariance()
    test_gate_runs_on_every_forward_path()
    test_ignored_flags_warn()
    test_forces_finite()
    print("All tests passed.")

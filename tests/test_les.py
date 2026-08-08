"""Tests for the LES integration: the model's ``return_embeddings`` hook and
the optional ``ecenet.les`` wrapper around the upstream ``les`` package.

Model-side (no optional dependency needed): the per-atom l0 read-out is
rotation-invariant and identical under ``l0_only``; l1 transforms as a vector;
the energy is unchanged by the flags; the batched paths match per-structure
forwards, including a zero-edge structure mid-batch (which exercises the
embedding slicing); forward_pbc with zero shifts matches forward.

Wrapper-side: without the upstream ``les`` package, ``import ecenet.les`` still
works and constructing `LESLongRange` raises an ImportError carrying the
install hint; with it installed, a smoke forward returns a finite energy.

Run:  python tests/test_les.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

import ecenet
from ecenet import ECENet
from ecenet.les import _LES_PIN, LESLongRange

try:
    import les  # noqa: F401
    HAVE_LES = True
except ImportError:
    HAVE_LES = False

torch.manual_seed(0)
DTYPE = torch.float64
N_TYPES = 4
TOL = 1e-8

COMMON = dict(
    n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
)


def make_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    m = ECENet(**COMMON, n_mp=2, bottleneck_dim=6, **kwargs).double()
    # Activate the layer stack (zero-init up-projections make it ~identity at
    # init) so the invariance tests see non-trivial features.
    for lyr in [x for stage in m.layers for x in stage]:
        with torch.no_grad():
            lyr.linear_up.weights.normal_(std=0.2)
    return m


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


# ── Model hook: return_embeddings / l0_only ─────────────────────────────────

def test_energy_unchanged_and_l0_only_consistent():
    m = make_model()
    pos, types = random_structure()
    e_plain = m(pos, types)
    e_full, l0_full, l1 = m(pos, types, return_embeddings=True)
    e_l0, l0_only = m(pos, types, return_embeddings=True, l0_only=True)
    assert (e_plain - e_full).abs() == 0.0 and (e_plain - e_l0).abs() == 0.0
    assert (l0_full - l0_only).abs().max() == 0.0, "l0_only changed l0"
    assert l0_full.shape == (len(types), 2 * m.embed_dim)
    assert l1.shape == (len(types), 2 * m.embed_dim, 3)
    print(f"  energy unchanged by flags; l0_only == full l0; "
          f"l0 {tuple(l0_full.shape)}, l1 {tuple(l1.shape)}")


def test_l0_rotation_invariant_l1_equivariant():
    m = make_model()
    pos, types = random_structure()
    Q = rand_rotation()
    _, l0_a, l1_a = m(pos, types, return_embeddings=True)
    _, l0_b, l1_b = m(pos @ Q.T, types, return_embeddings=True)
    d0 = (l0_a - l0_b).abs().max()
    assert d0 < TOL, f"l0 not rotation-invariant: {d0:.3e}"
    # l1 is in the real SH basis (m=-1,0,+1) = (y,z,x); mapped to Cartesian
    # (x,y,z) it must transform as a vector: l1(Qr) = Q l1(r).
    cart = [2, 0, 1]
    v_a = l1_a[:, :, cart]
    v_b = l1_b[:, :, cart]
    d1 = (v_b - torch.einsum('ncj,ij->nci', v_a, Q)).abs().max()
    # The pipeline's float64 SO(3) noise floor is ~3e-8 for l1 (vs ~4e-9 for
    # l0); a wrong basis mapping fails at O(1) (measured 0.68), so 1e-6 keeps
    # six orders of margin to a real break.
    assert d1 < 1e-6, f"l1 not vector-equivariant: {d1:.3e}"
    print(f"  l0 invariant ({d0:.1e}), l1 vector-equivariant ({d1:.1e})")


def test_batch_multi_matches_loop_with_zero_edge_structure():
    # Middle structure is a single atom (zero edges): its l0 must be zero rows
    # and must NOT shift the slices of the structures after it.
    m = make_model()
    structs = [random_structure(n, seed=s) for n, s in [(5, 1), (1, 2), (7, 3)]]
    pos_list = [p for p, _ in structs]
    types_list = [t for _, t in structs]
    energies, l0_list = m.forward_batch_multi(
        pos_list, types_list, return_embeddings=True, l0_only=True)
    assert len(l0_list) == 3
    for b, (pos, types) in enumerate(structs):
        e_ref, l0_ref = m(pos, types, return_embeddings=True, l0_only=True)
        de = (energies[b] - e_ref).abs()
        dl = (l0_list[b] - l0_ref).abs().max()
        assert de < TOL and dl < TOL, f"structure {b}: dE={de:.3e}, dl0={dl:.3e}"
    assert l0_list[1].abs().max() == 0.0, "zero-edge structure must have zero l0"
    print("  forward_batch_multi == per-structure forward (incl. zero-edge mid-batch)")


def test_forward_pbc_zero_shift_matches_forward():
    m = make_model()
    pos, types = random_structure()
    with torch.no_grad():
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-30)
        ei, ej = ((dist < m.r_cut_edge) & (dist > 1e-10)).nonzero(as_tuple=True)
        ns, nd = ((dist < m.r_cut_neighbor) & (dist > 1e-10)).nonzero(as_tuple=True)
    zed = torch.zeros(len(ei), 3, dtype=DTYPE)
    znb = torch.zeros(len(ns), 3, dtype=DTYPE)
    e_a, l0_a = m(pos, types, return_embeddings=True, l0_only=True)
    e_b, l0_b = m.forward_pbc(pos, types, ei, ej, zed, ns, nd, znb,
                              return_embeddings=True, l0_only=True)
    de, dl = (e_a - e_b).abs(), (l0_a - l0_b).abs().max()
    assert de < TOL and dl < TOL, f"pbc mismatch: dE={de:.3e}, dl0={dl:.3e}"
    print(f"  forward_pbc(zero shifts) == forward: dE={de:.1e}, dl0={dl:.1e}")


# ── Wrapper: lazy import / upstream smoke ───────────────────────────────────

def test_lazy_import():
    # The wrapper module and the package attribute resolve without `les`.
    assert ecenet.LESLongRange is LESLongRange


def test_missing_dep_error():
    if HAVE_LES:
        print("  skipped (`les` is installed)")
        return
    try:
        LESLongRange()
    except ImportError as e:
        msg = str(e)
        assert "ChengUCB/les" in msg and "pip install" in msg and _LES_PIN in msg
        assert "CC BY-NC" in msg
    else:
        raise AssertionError("LESLongRange() should raise ImportError without `les`")


def test_smoke_forward():
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    torch.manual_seed(0)
    lr = LESLongRange().double()
    m = make_model()
    pos, types = random_structure()
    p = pos.clone().requires_grad_(True)
    e_sr, l0 = m(p, types, return_embeddings=True, l0_only=True)
    e = e_sr + lr(l0, p).sum()
    assert torch.isfinite(e).all(), f"non-finite total energy: {e}"
    f = -torch.autograd.grad(e, p)[0]
    assert f.shape == pos.shape and torch.isfinite(f).all()
    print(f"  smoke: E={e.item():.6f} eV, |F|max={f.abs().max():.3f}")


if __name__ == "__main__":
    print(f"LES integration tests (upstream `les` installed: {HAVE_LES})")
    test_energy_unchanged_and_l0_only_consistent()
    test_l0_rotation_invariant_l1_equivariant()
    test_batch_multi_matches_loop_with_zero_edge_structure()
    test_forward_pbc_zero_shift_matches_forward()
    test_lazy_import()
    test_missing_dep_error()
    test_smoke_forward()
    print("All tests passed.")

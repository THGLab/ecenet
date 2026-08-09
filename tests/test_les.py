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


# ── Softmax (l0,l1) read-out (les_readout='softmax') ────────────────────────

def make_softmax_model(seed=0, score_std=0.0):
    m = make_model(seed=seed, les_readout='softmax')
    if score_std > 0:
        with torch.no_grad():
            m.les_score.weight.normal_(std=score_std)
            m.les_score.bias.normal_(std=score_std)
    return m


def test_softmax_readout_so3():
    # With a non-trivial (randomized) score head, the weighted read-out must
    # keep l0 invariant and l1 vector-equivariant: the weight is an invariant
    # scalar shared by both.
    m = make_softmax_model(score_std=0.5)
    pos, types = random_structure()
    Q = rand_rotation()
    _, l0_a, l1_a = m(pos, types, return_embeddings=True)
    _, l0_b, l1_b = m(pos @ Q.T, types, return_embeddings=True)
    d0 = (l0_a - l0_b).abs().max()
    assert d0 < TOL, f"softmax read-out broke l0 invariance: {d0:.3e}"
    cart = [2, 0, 1]
    d1 = (l1_b[:, :, cart]
          - torch.einsum('ncj,ij->nci', l1_a[:, :, cart], Q)).abs().max()
    assert d1 < 1e-6, f"softmax read-out broke l1 equivariance: {d1:.3e}"
    print(f"  softmax read-out: l0 invariant ({d0:.1e}), l1 equivariant ({d1:.1e})")


def test_softmax_readout_dimer_weight():
    # On a dimer each atom has exactly one in-edge, so the softmax weight has
    # a closed form: a = f_cut² / (f_cut + eps) — i.e. ≈ f_cut, the envelope.
    # les_readout is read at aggregation time, so flipping the attribute on ONE
    # model compares both paths with identical weights.
    from ecenet.radial import get_cutoff_fn
    m = make_softmax_model()          # zero-init score → s = 0 exactly
    types = torch.tensor([0, 1])
    f = get_cutoff_fn(m.cutoff_type)
    for r in (1.5, 3.0, 4.5):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, r]], dtype=DTYPE)
        m.les_readout = 'sum'
        _, l0_sum = m(pos, types, return_embeddings=True, l0_only=True)
        m.les_readout = 'softmax'
        _, l0_soft = m(pos, types, return_embeddings=True, l0_only=True)
        f_cut = f(torch.tensor([r], dtype=DTYPE), m.r_cut_edge)
        expected = l0_sum * (f_cut ** 2 / (f_cut + 1e-6))
        d = (l0_soft - expected).abs().max()
        assert d < TOL, f"dimer weight mismatch at r={r}: {d:.3e}"
        ratio = f_cut.item() ** 2 / (f_cut.item() + 1e-6)
        print(f"  dimer r={r}: softmax/sum = {ratio:.4f} (≈ f_cut={f_cut.item():.4f})")


def test_softmax_readout_variants_consistent():
    # forward == forward_pbc(zero shifts) == forward_batch_multi, softmax
    # read-out with a randomized score, incl. a zero-edge structure mid-batch.
    m = make_softmax_model(score_std=0.5)
    structs = [random_structure(n, seed=s) for n, s in [(5, 1), (1, 2), (7, 3)]]
    pos_list = [p for p, _ in structs]
    types_list = [t for _, t in structs]
    energies, l0_list = m.forward_batch_multi(
        pos_list, types_list, return_embeddings=True, l0_only=True)
    for b, (pos, types) in enumerate(structs):
        _, l0_ref = m(pos, types, return_embeddings=True, l0_only=True)
        dl = (l0_list[b] - l0_ref).abs().max()
        assert dl < TOL, f"structure {b}: dl0={dl:.3e}"
    assert l0_list[1].abs().max() == 0.0, "zero-edge structure must have zero l0"

    pos, types = structs[0]
    with torch.no_grad():
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-30)
        ei, ej = ((dist < m.r_cut_edge) & (dist > 1e-10)).nonzero(as_tuple=True)
        ns, nd = ((dist < m.r_cut_neighbor) & (dist > 1e-10)).nonzero(as_tuple=True)
    _, l0_a = m(pos, types, return_embeddings=True, l0_only=True)
    _, l0_b = m.forward_pbc(pos, types, ei, ej, torch.zeros(len(ei), 3, dtype=DTYPE),
                            ns, nd, torch.zeros(len(ns), 3, dtype=DTYPE),
                            return_embeddings=True, l0_only=True)
    dl = (l0_a - l0_b).abs().max()
    assert dl < TOL, f"pbc mismatch: {dl:.3e}"
    print("  softmax read-out consistent across forward variants (incl. zero-edge)")


def test_edge_readout_model():
    """les_readout='edge'/'edge_basis': l0 has width 1, is SO(3)-invariant,
    consistent across forward variants, and zero for zero-edge structures."""
    for mode in ('edge', 'edge_basis'):
        m = make_model(seed=0, les_readout=mode)
        pos, types = random_structure()
        Q = rand_rotation()
        _, l0_a = m(pos, types, return_embeddings=True, l0_only=True)
        _, l0_b = m(pos @ Q.T, types, return_embeddings=True, l0_only=True)
        assert l0_a.shape == (len(types), 1), f"{mode} l0 shape: {tuple(l0_a.shape)}"
        d0 = (l0_a - l0_b).abs().max()
        assert d0 < TOL, f"{mode} read-out charge not invariant: {d0:.3e}"
        assert l0_a.abs().max() > 0, f"{mode} read-out is identically zero " \
            "(zero-init would be a gradient-free saddle — must be standard init)"

        structs = [random_structure(n, seed=s) for n, s in [(5, 1), (1, 2), (7, 3)]]
        energies, l0_list = m.forward_batch_multi(
            [p for p, _ in structs], [t for _, t in structs],
            return_embeddings=True, l0_only=True)
        for b, (pos_b, types_b) in enumerate(structs):
            _, l0_ref = m(pos_b, types_b, return_embeddings=True, l0_only=True)
            dl = (l0_list[b] - l0_ref).abs().max()
            assert dl < TOL, f"{mode} structure {b}: dl0={dl:.3e}"
        assert l0_list[1].shape == (1, 1) and l0_list[1].abs().max() == 0.0
        print(f"  {mode} read-out: width-1 invariant charge ({d0:.1e}), "
              "variants consistent, zero-edge → q=0")

    # 'edge_basis' only: the per-bond charge vanishes exactly at r_cut (the
    # dotted radial basis carries the envelope), and the head has n_max_d
    # channels rather than 1.
    m = make_model(seed=0, les_readout='edge_basis')
    assert m.les_edge_charge.weight.shape[0] == m.n_max_d
    types2 = torch.tensor([0, 1])
    eps_r = 1e-9
    pos_at = torch.tensor([[0.0, 0.0, 0.0],
                           [0.0, 0.0, m.r_cut_edge - eps_r]], dtype=DTYPE)
    _, q_at = m(pos_at, types2, return_embeddings=True, l0_only=True)
    assert q_at.abs().max() < 1e-6, \
        f"edge_basis charge does not vanish at r_cut: {q_at.abs().max():.3e}"
    print("  edge_basis: n_max_d-channel head, per-bond charge → 0 at r_cut")


def test_edge_readout_les_energy():
    """l0_is_charge=True: isolated fast path and upstream's latent_charges
    path agree, and the LES module holds no parameters (head bypassed)."""
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    torch.manual_seed(4)
    lr = LESLongRange().double()
    g = torch.Generator().manual_seed(11)
    sizes = [4, 6]
    pos = torch.randn(sum(sizes), 3, generator=g, dtype=DTYPE) * 2.0
    q_in = torch.randn(sum(sizes), 1, generator=g, dtype=DTYPE) * 0.3
    batch = torch.cat([torch.full((n,), b, dtype=torch.long)
                       for b, n in enumerate(sizes)])

    p_a = pos.clone().requires_grad_(True)
    e_fast, q_out = lr(q_in, p_a, batch=batch, n_struct=2,
                       return_charges=True, l0_is_charge=True)
    f_a = torch.autograd.grad(e_fast.sum(), p_a)[0]
    assert (q_out - q_in.reshape(-1)).abs().max() == 0.0, \
        "l0_is_charge must return the input charges untouched"
    assert len(list(lr.parameters())) == 0, \
        "LES module should hold no parameters when the head is bypassed"

    p_b = pos.clone().requires_grad_(True)
    res = lr.les(latent_charges=q_in.reshape(-1), positions=p_b, cell=None,
                 batch=batch, compute_energy=True)
    f_b = torch.autograd.grad(res["E_lr"].sum(), p_b)[0]
    de = (e_fast - res["E_lr"].reshape(e_fast.shape)).abs().max()
    df = (f_a - f_b).abs().max()
    assert de < 1e-10 and df < 1e-10, f"mismatch: dE={de:.3e}, dF={df:.3e}"
    print(f"  l0_is_charge: fast path == upstream latent_charges "
          f"(dE={de:.1e}, dF={df:.1e}), module parameter-free")


def test_les_readout_validation():
    try:
        ECENet(**COMMON, les_readout='mean')
        raise AssertionError("les_readout='mean' should have raised")
    except ValueError as e:
        assert 'les_readout' in str(e)
    print("  invalid les_readout rejected")


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
    # cell=None (fast path, no per-structure det check) must equal an explicit
    # zero cell (upstream's det<1e-6 branch) — both mean isolated.
    with torch.no_grad():
        e_none = lr(l0, p)
        e_zero = lr(l0, p, cell=torch.zeros(1, 3, 3, dtype=p.dtype))
    dz = (e_none - e_zero).abs().max()
    assert dz == 0.0, f"cell=None != zero cell: {dz:.3e}"
    f = -torch.autograd.grad(e, p)[0]
    assert f.shape == pos.shape and torch.isfinite(f).all()
    print(f"  smoke: E={e.item():.6f} eV, |F|max={f.abs().max():.3f}")


def test_isolated_batched_matches_upstream_loop():
    """The wrapper's vectorized isolated path (one masked full-batch quadratic
    form) must equal upstream's per-structure Python loop exactly — energies,
    latent charges, and position gradients — including a single-atom
    structure mid-batch."""
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    torch.manual_seed(2)
    lr = LESLongRange().double()
    g = torch.Generator().manual_seed(9)
    sizes = [5, 1, 7]
    pos = torch.randn(sum(sizes), 3, generator=g, dtype=DTYPE) * 2.0
    l0 = torch.randn(sum(sizes), 16, generator=g, dtype=DTYPE)
    batch = torch.cat([torch.full((n,), b, dtype=torch.long)
                       for b, n in enumerate(sizes)])
    with torch.no_grad():
        lr(l0, pos, batch=batch, n_struct=len(sizes))   # materialise the head
        for p in lr.parameters():
            p.add_(0.1 * torch.randn_like(p))

    p_a = pos.clone().requires_grad_(True)
    e_fast, q_fast = lr(l0, p_a, batch=batch, n_struct=len(sizes),
                        return_charges=True)
    f_a = torch.autograd.grad(e_fast.sum(), p_a)[0]

    p_b = pos.clone().requires_grad_(True)
    res = lr.les(desc=l0, positions=p_b, cell=None, batch=batch,
                 compute_energy=True)                    # upstream's own loop
    f_b = torch.autograd.grad(res["E_lr"].sum(), p_b)[0]

    de = (e_fast - res["E_lr"].reshape(e_fast.shape)).abs().max()
    dq = (q_fast.reshape(-1) - res["latent_charges"].reshape(-1)).abs().max()
    df = (f_a - f_b).abs().max()
    # The anti-coincidence grid shift costs ~eps·offset of fp cancellation
    # noise in the intra-structure distances, so equality is to ~1e-12 in
    # float64 rather than bit-exact. A real break is orders louder.
    assert de < 1e-10, f"energy mismatch vs upstream loop: {de:.3e}"
    assert dq == 0.0, f"charge mismatch vs upstream loop: {dq:.3e}"
    assert df < 1e-10, f"gradient mismatch vs upstream loop: {df:.3e}"
    print(f"  vectorized isolated path == upstream loop "
          f"(dE={de:.1e}, dq={dq:.1e}, dF={df:.1e})")


def test_isolated_batched_coincident_cross_atoms():
    """Structures in a batch are individually centered, so atoms of DIFFERENT
    structures can sit at identical coordinates. The dense kernel used to
    produce inf·0 = NaN there (masking cannot clean a NaN); the grid shift
    must keep energies and gradients finite and equal to upstream's loop."""
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    torch.manual_seed(3)
    lr = LESLongRange().double()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=DTYPE)
    l0 = torch.randn(4, 8, dtype=DTYPE)
    batch = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        lr(l0[:2], pos[:2])                       # materialise the head
        for p in lr.parameters():
            p.add_(0.1 * torch.randn_like(p))

    p_a = pos.clone().requires_grad_(True)
    e_fast = lr(l0, p_a, batch=batch, n_struct=2)
    assert torch.isfinite(e_fast).all(), f"NaN with coincident cross atoms: {e_fast}"
    f_a = torch.autograd.grad(e_fast.sum(), p_a)[0]
    assert torch.isfinite(f_a).all(), "NaN gradient with coincident cross atoms"

    p_b = pos.clone().requires_grad_(True)
    res = lr.les(desc=l0, positions=p_b, cell=None, batch=batch,
                 compute_energy=True)
    f_b = torch.autograd.grad(res["E_lr"].sum(), p_b)[0]
    de = (e_fast - res["E_lr"].reshape(e_fast.shape)).abs().max()
    df = (f_a - f_b).abs().max()
    assert de < 1e-10 and df < 1e-10, f"mismatch: dE={de:.3e}, dF={df:.3e}"
    print(f"  coincident cross-structure atoms: finite and == upstream "
          f"(dE={de:.1e}, dF={df:.1e})")


if __name__ == "__main__":
    print(f"LES integration tests (upstream `les` installed: {HAVE_LES})")
    test_energy_unchanged_and_l0_only_consistent()
    test_l0_rotation_invariant_l1_equivariant()
    test_batch_multi_matches_loop_with_zero_edge_structure()
    test_forward_pbc_zero_shift_matches_forward()
    test_softmax_readout_so3()
    test_softmax_readout_dimer_weight()
    test_softmax_readout_variants_consistent()
    test_edge_readout_model()
    test_edge_readout_les_energy()
    test_les_readout_validation()
    test_lazy_import()
    test_missing_dep_error()
    test_smoke_forward()
    test_isolated_batched_matches_upstream_loop()
    test_isolated_batched_coincident_cross_atoms()
    print("All tests passed.")

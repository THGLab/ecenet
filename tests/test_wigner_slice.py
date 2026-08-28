# Prototype, mainly implemented by Claude
"""Truncated Wigner-D: the kept-column slice (m_max < l_max) and the analytic
small-m build.

With the angular layout truncated at m_max, the layers only consume bond-frame
components with |m| <= min(l, m_max), and the packed features are zero outside
those slots — so both frame changes need only the corresponding COLUMNS of
each D^l block. Checks:

  1. build_D_slice == the kept columns of build_D_block, exactly.
  2. build_D_slice_analytic (m_max <= 1; Y for m=0, tangential gradients along
     the gauge frame for m=±1) == the recursion to fp rounding, on both gauge
     charts and on-axis edges; gradients match through the normalize chain
     (direct-r̂ gradients differ by a pure radial component — the two builds
     extend D off the unit sphere differently, and normalize projects that
     out); double backward raises (single-backward contract); m_max > 1
     rejected.
  3. Model equality: with m_max < l_max the sliced path (automatic) is
     BIT-IDENTICAL to the full-block path (escape hatch _use_d_slice=False)
     in energies, forces, and (l0, l1) embeddings, with and without MP;
     forward_batch_multi included; SO(3) invariance holds on the sliced path.
  4. The analytic toggle: forces match the recursion path; force-loss double
     backward raises; fused/m_max guards.

Run:  python tests/test_wigner_slice.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn.functional as F

from ecenet import ECENet
from ecenet.spherical import (
    build_D_block,
    build_D_slice,
    build_D_slice_analytic,
    kept_offsets,
    n_kept_columns,
)

torch.manual_seed(0)
DTYPE = torch.float64
N_TYPES = 4
COMMON = dict(n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
              l_max=3, n_max=3, embed_dim=8, n_layers=1, n_max_d=4)


def hard_directions(n=24, seed=0):
    """Random unit vectors plus the awkward ones: chart-B (|rx| >= 0.9) and
    exactly on-axis (the gauge charts' dead branches)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.nn.functional.normalize(
        torch.randn(n, 3, generator=g, dtype=DTYPE), dim=-1)
    r[0] = F.normalize(torch.tensor([0.95, 0.2, 0.05], dtype=DTYPE), dim=-1)
    r[1] = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    r[2] = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    r[3] = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    return r


def kept_cols(l_max, m_max):
    return [i for l in range(l_max + 1)
            for i in range(l * l + l - min(l, m_max),
                           l * l + l + min(l, m_max) + 1)]


def random_structure(n=7, seed=0):
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


def test_slice_matches_block_columns():
    r = hard_directions()
    for l_max in (2, 3, 4):
        Dfull = build_D_block(r, l_max)
        for m_max in (0, 1, 2, l_max):
            cols = kept_cols(l_max, m_max)
            assert len(cols) == n_kept_columns(l_max, m_max)
            assert kept_offsets(l_max, m_max)[-1] == len(cols)
            d = (build_D_slice(r, l_max, m_max) - Dfull[:, :, cols]).abs().max()
            assert d == 0.0, f"l_max={l_max} m_max={m_max}: slice != cols {d:.2e}"
    print("  build_D_slice == kept columns of build_D_block (exact)")


def test_analytic_matches_recursion():
    r = hard_directions()
    for l_max in (2, 3, 5):
        for m_max in (0, 1):
            ref = build_D_slice(r, l_max, m_max)
            d = (build_D_slice_analytic(r, l_max, m_max) - ref).abs().max()
            assert d < 1e-13, f"l_max={l_max} m_max={m_max}: analytic {d:.2e}"
    # gradients through the model's actual chain (positions → normalize → D):
    # comparing at fixed r̂ would be ill-posed — the two builds extend D off
    # the unit sphere differently (pure radial disagreement, projected out
    # by normalize's backward)
    raw = hard_directions()[:10] * 2.0
    r1 = raw.clone().requires_grad_(True)
    g1 = torch.autograd.grad(
        build_D_slice(F.normalize(r1, dim=-1), 3, 1).pow(2).sum(), r1)[0]
    r2 = raw.clone().requires_grad_(True)
    g2 = torch.autograd.grad(
        build_D_slice_analytic(F.normalize(r2, dim=-1), 3, 1).pow(2).sum(), r2)[0]
    dg = (g1 - g2).abs().max()
    assert dg < 1e-12, f"analytic gradient mismatch: {dg:.2e}"

    # single-backward contract: building a force graph (create_graph=True)
    # raises at the FIRST backward — loud and early, because a silent
    # fallback would drop this branch's second derivative (the analytic D
    # mixes the Y-based Function with the differentiable D1 block)
    r3 = hard_directions()[:4].clone().requires_grad_(True)
    try:
        torch.autograd.grad(build_D_slice_analytic(r3, 3, 1).pow(2).sum(),
                            r3, create_graph=True)
        raise AssertionError("create_graph through analytic D should raise")
    except RuntimeError as e:
        assert 'single-backward' in str(e)
    # ...while a plain (single) backward is fine, even inside enable_grad
    with torch.enable_grad():
        r4 = hard_directions()[:4].clone().requires_grad_(True)
        g = torch.autograd.grad(build_D_slice_analytic(r4, 3, 1).pow(2).sum(),
                                r4)[0]
        assert torch.isfinite(g).all()
    try:
        build_D_slice_analytic(hard_directions()[:4], 3, 2)
        raise AssertionError("m_max=2 should have been rejected")
    except ValueError as e:
        assert 'm_max' in str(e)
    print(f"  analytic == recursion (grad via normalize {dg:.1e}); "
          "double backward + m_max>1 rejected")


def test_model_slice_bit_identical():
    pos, types = random_structure()
    for n_mp, m_max in ((1, 1), (2, 1), (2, 2)):
        torch.manual_seed(0)
        m = ECENet(**COMMON, n_mp=n_mp, m_max=m_max).double()
        if n_mp >= 2:
            for L in m.mp_layers:
                with torch.no_grad():
                    L.msg_up.weights.normal_(std=0.3)
                    L.msg_up.bias.normal_(std=0.1)
        assert m._d_slice_active(), "slice should be active for m_max < l_max"

        def ef(model, p0):
            p = p0.clone().requires_grad_(True)
            e, l0, l1 = model(p, types, return_embeddings=True)
            f = torch.autograd.grad(e, p)[0]
            return e.detach(), f, l0.detach(), l1.detach()

        e1, f1, l01, l11 = ef(m, pos)
        m._use_d_slice = False           # escape hatch → full-block path
        e2, f2, l02, l12 = ef(m, pos)
        m._use_d_slice = True
        d = max((e1 - e2).abs().item(), (f1 - f2).abs().max().item(),
                (l01 - l02).abs().max().item(), (l11 - l12).abs().max().item())
        assert d == 0.0, f"n_mp={n_mp} m_max={m_max}: slice != full ({d:.2e})"

        # SO(3) invariance on the sliced path. The bit-identity assertion
        # above already proves slice == full, so any residual here is the
        # random-weight model's own float64 noise floor (the std=0.3 MP
        # perturbation amplifies rounding), not a slice-induced break.
        Q = rand_rotation()
        de = (m(pos, types) - m(pos @ Q.T, types)).abs().item()
        assert de < 1e-8, f"SO(3) broken on sliced path: {de:.2e}"

        # batched path
        _, l0_list = m.forward_batch_multi([pos, pos + 0.05], [types, types],
                                           return_embeddings=True, l0_only=True)
        m._use_d_slice = False
        _, l0_ref = m.forward_batch_multi([pos, pos + 0.05], [types, types],
                                          return_embeddings=True, l0_only=True)
        m._use_d_slice = True
        db = max((a - b).abs().max().item() for a, b in zip(l0_list, l0_ref))
        assert db == 0.0, f"batched slice != full: {db:.2e}"
        print(f"  n_mp={n_mp}, m_max={m_max}: sliced == full (E/F/l0/l1/batched "
              f"exact), SO(3) {de:.1e}")


def test_model_analytic():
    pos, types = random_structure()
    torch.manual_seed(0)
    m = ECENet(**COMMON, n_mp=2, m_max=1).double()
    for L in m.mp_layers:
        with torch.no_grad():
            L.msg_up.weights.normal_(std=0.3)
    p1 = pos.clone().requires_grad_(True)
    e1 = m(p1, types)
    f1 = torch.autograd.grad(e1, p1)[0]
    m.set_analytic_wigner(True)
    p2 = pos.clone().requires_grad_(True)
    e2 = m(p2, types)
    f2 = torch.autograd.grad(e2, p2)[0]
    de = (e1 - e2).abs().item()
    df = (f1 - f2).abs().max().item()
    assert de < 1e-12 and df < 1e-11, f"analytic model mismatch: {de:.2e}/{df:.2e}"

    # force-loss training must raise, not silently mis-train: the error fires
    # already at the create_graph force computation
    p3 = pos.clone().requires_grad_(True)
    try:
        torch.autograd.grad(m(p3, types), p3, create_graph=True)
        raise AssertionError("force-graph build through analytic D should raise")
    except RuntimeError as e:
        assert 'single-backward' in str(e)

    # guards: fused is mutually exclusive; m_max > 1 rejected
    try:
        m.set_edge_frame_fused(True)
        raise AssertionError("fused + analytic should be rejected")
    except ValueError as e:
        assert 'analytic' in str(e)
    m.set_analytic_wigner(False)
    m.set_edge_frame_fused(True)
    try:
        m.set_analytic_wigner(True)
        raise AssertionError("analytic + fused should be rejected")
    except ValueError as e:
        assert 'fused' in str(e).lower() or 'edge_frame' in str(e)
    m.set_edge_frame_fused(False)
    m2 = ECENet(**COMMON, m_max=2).double()
    try:
        m2.set_analytic_wigner(True)
        raise AssertionError("m_max=2 analytic should be rejected")
    except ValueError as e:
        assert 'm_max' in str(e)
    print(f"  analytic model: E/F match (dE={de:.1e}, dF={df:.1e}); "
          "force-loss raises; guards OK")


if __name__ == "__main__":
    print("Truncated Wigner-D (kept-column slice + analytic small-m build)")
    test_slice_matches_block_columns()
    test_analytic_matches_recursion()
    test_model_slice_bit_identical()
    test_model_analytic()
    print("All tests passed.")

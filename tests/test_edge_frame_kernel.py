"""Tests for ecenet/edge_frame_kernel.py (fused node→edge-frame transform).

Runnable as a script from the repo root:  python tests/test_edge_frame_kernel.py

Checks, in order of increasing strictness:
  1. the reference spec reproduces the model's actual ops
     (gather + wigner_rotate + SphToAngular) bit-for-bit;
  2. EdgeFrameFused matches the reference forward;
  3. its analytic backward matches autograd through the reference chain
     (dA_emb, dD, and d r_hat through build_D_block);
  4. gradcheck / gradgradcheck — the double backward force training needs;
  5. the single-source / e2n / pack-unrotate variants match the MP layers' ops;
  6. Triton paths vs the fp64 reference (skips without CUDA);
  7. whole-model integration: energy/forces identical with the flags on/off.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecenet.edge_frame_kernel import EdgeFrameFused, edge_frame_reference  # noqa: E402
from ecenet.model import SphToAngular  # noqa: E402
from ecenet.spherical import build_D_block, wigner_rotate  # noqa: E402

torch.manual_seed(0)

L_MAX, M_MAX, C, N_ATOMS, E = 2, 2, 2, 5, 8
N_SPH = (L_MAX + 1) ** 2


def _setup(dtype=torch.float64):
    sph = SphToAngular(C, L_MAX, m_max=M_MAX).to(dtype)
    A_emb = torch.randn(N_ATOMS, C, N_SPH, dtype=dtype)
    edge_i = torch.randint(0, N_ATOMS, (E,))
    edge_j = (edge_i + torch.randint(1, N_ATOMS, (E,))) % N_ATOMS  # no self-edges
    r_hat = torch.randn(E, 3, dtype=dtype)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    return sph, A_emb, edge_i, edge_j, r_hat


def _tables(sph):
    return (sph.cos_flat_idx, sph.sin_flat_idx, sph.cos_valid, sph.sin_valid)


def test_reference_matches_model_ops():
    sph, A_emb, edge_i, edge_j, r_hat = _setup()
    D = build_D_block(r_hat, L_MAX)

    # the model's actual op sequence (forward steps 3-4)
    A_both = torch.cat([A_emb[edge_i], A_emb[edge_j]], dim=1)
    A_rot = wigner_rotate(A_both, D)
    c0, s0 = sph(A_rot)

    c1, s1 = edge_frame_reference(A_emb, edge_i, edge_j, D, *_tables(sph))
    assert torch.equal(c0, c1) and torch.equal(s0, s1)
    print("test_reference_matches_model_ops: OK")


def test_fused_matches_reference_forward():
    sph, A_emb, edge_i, edge_j, r_hat = _setup()
    D = build_D_block(r_hat, L_MAX)
    c0, s0 = edge_frame_reference(A_emb, edge_i, edge_j, D, *_tables(sph))
    c1, s1 = EdgeFrameFused.apply(A_emb, edge_i, edge_j, D, *_tables(sph))
    assert torch.allclose(c0, c1, atol=1e-14) and torch.allclose(s0, s1, atol=1e-14)
    print("test_fused_matches_reference_forward: OK")


def test_backward_matches_reference():
    """Analytic dA_emb / dD / d r_hat vs autograd through the reference chain."""
    sph, A_emb0, edge_i, edge_j, r_hat0 = _setup()
    dc = torch.randn(E, sph.n_ch, sph.n_angular, dtype=torch.float64)
    ds = torch.randn(E, sph.n_ch, sph.n_angular, dtype=torch.float64)

    grads = {}
    for name, fn in [("ref", edge_frame_reference), ("fused", EdgeFrameFused.apply)]:
        A_emb = A_emb0.clone().requires_grad_(True)
        r_hat = r_hat0.clone().requires_grad_(True)
        D = build_D_block(r_hat, L_MAX)          # d r_hat flows through the recursion
        c, s = fn(A_emb, edge_i, edge_j, D, *_tables(sph))
        loss = (c * dc).sum() + (s * ds).sum()
        grads[name] = torch.autograd.grad(loss, [A_emb, r_hat])

    for g0, g1 in zip(grads["ref"], grads["fused"]):
        assert torch.allclose(g0, g1, atol=1e-12), (g0 - g1).abs().max()
    print("test_backward_matches_reference: OK")


def test_gradcheck():
    sph, A_emb, edge_i, edge_j, r_hat = _setup()
    D = build_D_block(r_hat, L_MAX)
    A_emb = A_emb.requires_grad_(True)
    D = D.detach().requires_grad_(True)
    tabs = _tables(sph)

    ok = torch.autograd.gradcheck(
        lambda a, d: EdgeFrameFused.apply(a, edge_i, edge_j, d, *tabs),
        (A_emb, D), atol=1e-9)
    assert ok
    print("test_gradcheck: OK")


def test_gradgradcheck():
    """Double backward — the path force-loss training exercises."""
    sph, A_emb, edge_i, edge_j, r_hat = _setup()
    D = build_D_block(r_hat, L_MAX)
    A_emb = A_emb.requires_grad_(True)
    D = D.detach().requires_grad_(True)
    tabs = _tables(sph)

    ok = torch.autograd.gradgradcheck(
        lambda a, d: EdgeFrameFused.apply(a, edge_i, edge_j, d, *tabs),
        (A_emb, D), atol=1e-9)
    assert ok
    print("test_gradgradcheck: OK")


def test_force_path_smoke():
    """Energy-like scalar → forces via create_graph → grad of |F|² w.r.t. A_emb.
    Exercises the full double-backward chain through the fused op AND the
    Wigner recursion, comparing fused vs reference end to end."""
    sph, A_emb0, edge_i, edge_j, _ = _setup()
    pos0 = torch.randn(N_ATOMS, 3, dtype=torch.float64)

    def force_grad(fn):
        A_emb = A_emb0.clone().requires_grad_(True)
        pos = pos0.clone().requires_grad_(True)
        rv = pos[edge_j] - pos[edge_i]
        r_hat = rv / rv.norm(dim=-1, keepdim=True)
        D = build_D_block(r_hat, L_MAX)
        c, s = fn(A_emb, edge_i, edge_j, D, *_tables(sph))
        energy = (c ** 2).sum() + (s ** 2).sum()
        (force,) = torch.autograd.grad(energy, pos, create_graph=True)
        (g,) = torch.autograd.grad((force ** 2).sum(), A_emb)
        return force, g

    f0, g0 = force_grad(edge_frame_reference)
    f1, g1 = force_grad(EdgeFrameFused.apply)
    assert torch.allclose(f0, f1, atol=1e-12)
    assert torch.allclose(g0, g1, atol=1e-11), (g0 - g1).abs().max()
    print("test_force_path_smoke: OK")


def test_single_matches_mp_ops():
    """Single-source variant == the MP layers' bmm + _unpack_sph_to_angular."""
    from ecenet.edge_frame_kernel import edge_frame_fused_single
    from ecenet.model import _unpack_sph_to_angular

    n_base, l_max, m_max, n_atoms, E_ = 6, 2, 2, 5, 8
    n_sph = (l_max + 1) ** 2
    n_ang = m_max + 1
    Delta = torch.randn(n_atoms, n_base, n_sph, dtype=torch.float64)
    ei = torch.randint(0, n_atoms, (E_,))
    r_hat = torch.randn(E_, 3, dtype=torch.float64)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    D = build_D_block(r_hat, l_max)

    # the MP layers' actual ops (steps 5-6)
    v_rot = torch.bmm(Delta[ei], D)
    dc0, ds0 = _unpack_sph_to_angular(v_rot, n_base, l_max, m_max, n_ang, n_sph)
    dc0 = dc0.reshape(E_, n_base * (l_max + 1), n_ang)
    ds0 = ds0.reshape(E_, n_base * (l_max + 1), n_ang)

    dc1, ds1 = edge_frame_fused_single(Delta, ei, D, l_max, m_max)
    assert torch.allclose(dc0, dc1, atol=1e-14) and torch.allclose(ds0, ds1, atol=1e-14)

    # m_max < l_max truncation also matches
    dc0t, ds0t = _unpack_sph_to_angular(v_rot, n_base, l_max, 1, 2, n_sph)
    dc1t, ds1t = edge_frame_fused_single(Delta, ei, D, l_max, 1)
    assert torch.allclose(dc0t.reshape(E_, -1, 2), dc1t, atol=1e-14)
    assert torch.allclose(ds0t.reshape(E_, -1, 2), ds1t, atol=1e-14)
    print("test_single_matches_mp_ops: OK")


def test_single_gradchecks():
    from ecenet.edge_frame_kernel import _ef_flat_tables

    n_base, l_max, m_max, n_atoms, E_ = 4, 2, 2, 5, 8
    n_sph = (l_max + 1) ** 2
    Delta = torch.randn(n_atoms, n_base, n_sph, dtype=torch.float64,
                        requires_grad=True)
    ei = torch.randint(0, n_atoms, (E_,))
    r_hat = torch.randn(E_, 3, dtype=torch.float64)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    D = build_D_block(r_hat, l_max).detach().requires_grad_(True)
    tabs = _ef_flat_tables(n_base, l_max, m_max, Delta.device, Delta.dtype)

    fn = lambda a, d: EdgeFrameFused.apply(a, ei, None, d, *tabs)  # noqa: E731
    assert torch.autograd.gradcheck(fn, (Delta, D), atol=1e-9)
    assert torch.autograd.gradgradcheck(fn, (Delta, D), atol=1e-9)
    print("test_single_gradchecks: OK")


def test_e2n_matches_mp_ops():
    """edge_to_node reference/Function == the MP layers' pack + bmm(Dᵀ) + scatter."""
    from ecenet.edge_frame_kernel import (
        _ef_flat_tables,
        edge_to_node_fused,
        edge_to_node_reference,
    )
    from ecenet.model import _pack_angular_to_sph

    n_base, l_max, m_max, n_atoms, E_ = 6, 2, 2, 5, 8
    n_sph = (l_max + 1) ** 2
    n_ang = m_max + 1
    n_ch = n_base * (l_max + 1)
    g_cos = torch.randn(E_, n_ch, n_ang, dtype=torch.float64)
    g_sin = torch.randn(E_, n_ch, n_ang, dtype=torch.float64)
    ej = torch.randint(0, n_atoms, (E_,))
    r_hat = torch.randn(E_, 3, dtype=torch.float64)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    D = build_D_block(r_hat, l_max)

    # the MP layers' actual ops
    h = _pack_angular_to_sph(g_cos, g_sin, n_base, l_max, m_max, n_ang, n_sph)
    h_global = torch.bmm(h, D.transpose(-1, -2))
    idx = ej[:, None, None].expand_as(h_global)
    Delta0 = torch.zeros(n_atoms, n_base, n_sph, dtype=torch.float64
                         ).scatter_add(0, idx, h_global)

    tabs = _ef_flat_tables(n_base, l_max, m_max, g_cos.device, g_cos.dtype)
    Delta1 = edge_to_node_reference(g_cos, g_sin, ej, D, n_atoms, *tabs)
    Delta2 = edge_to_node_fused(g_cos, g_sin, ej, D, n_atoms, l_max, m_max)
    assert torch.allclose(Delta0, Delta1, atol=1e-14)
    assert torch.allclose(Delta0, Delta2, atol=1e-14)

    # m_max < l_max truncation
    m_tr = 1
    h_t = _pack_angular_to_sph(g_cos[:, :, :2], g_sin[:, :, :2], n_base,
                               l_max, m_tr, 2, n_sph)
    D0t = torch.zeros(n_atoms, n_base, n_sph, dtype=torch.float64).scatter_add(
        0, idx, torch.bmm(h_t, D.transpose(-1, -2)))
    D2t = edge_to_node_fused(g_cos[:, :, :2].contiguous(),
                             g_sin[:, :, :2].contiguous(), ej, D,
                             n_atoms, l_max, m_tr)
    assert torch.allclose(D0t, D2t, atol=1e-14)
    print("test_e2n_matches_mp_ops: OK")


def test_e2n_gradchecks():
    from ecenet.edge_frame_kernel import EdgeToNodeFused, _ef_flat_tables

    n_base, l_max, m_max, n_atoms, E_ = 4, 2, 2, 5, 8
    n_ch = n_base * (l_max + 1)
    g_cos = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64,
                        requires_grad=True)
    g_sin = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64,
                        requires_grad=True)
    ej = torch.randint(0, n_atoms, (E_,))
    r_hat = torch.randn(E_, 3, dtype=torch.float64)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    D = build_D_block(r_hat, l_max).detach().requires_grad_(True)
    tabs = _ef_flat_tables(n_base, l_max, m_max, g_cos.device, g_cos.dtype)

    fn = lambda gc, gs, d: EdgeToNodeFused.apply(gc, gs, ej, d, n_atoms, *tabs)  # noqa: E731
    assert torch.autograd.gradcheck(fn, (g_cos, g_sin, D), atol=1e-9)
    assert torch.autograd.gradgradcheck(fn, (g_cos, g_sin, D), atol=1e-9)
    print("test_e2n_gradchecks: OK")


def test_pack_unrotate_matches_mp_ops():
    """PackUnrotateFused == the MP layers' _pack_angular_to_sph + bmm(Dᵀ)."""
    from ecenet.edge_frame_kernel import pack_unrotate_fused
    from ecenet.model import _pack_angular_to_sph

    n_base, l_max, m_max, E_ = 6, 2, 2, 8
    n_sph = (l_max + 1) ** 2
    n_ch = n_base * (l_max + 1)
    m_cos = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64)
    m_sin = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64)
    r_hat = torch.randn(E_, 3, dtype=torch.float64)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    D = build_D_block(r_hat, l_max)

    h = _pack_angular_to_sph(m_cos, m_sin, n_base, l_max, m_max,
                             m_max + 1, n_sph)
    hg0 = torch.bmm(h, D.transpose(-1, -2))
    hg1 = pack_unrotate_fused(m_cos, m_sin, D, l_max, m_max)
    assert torch.allclose(hg0, hg1, atol=1e-14)

    # m_max < l_max truncation
    h_t = _pack_angular_to_sph(m_cos[:, :, :2], m_sin[:, :, :2], n_base,
                               l_max, 1, 2, n_sph)
    hg0t = torch.bmm(h_t, D.transpose(-1, -2))
    hg1t = pack_unrotate_fused(m_cos[:, :, :2].contiguous(),
                               m_sin[:, :, :2].contiguous(), D, l_max, 1)
    assert torch.allclose(hg0t, hg1t, atol=1e-14)
    print("test_pack_unrotate_matches_mp_ops: OK")


def test_pack_unrotate_gradchecks():
    from ecenet.edge_frame_kernel import PackUnrotateFused, _ef_flat_tables

    n_base, l_max, m_max, E_ = 4, 2, 2, 6
    n_ch = n_base * (l_max + 1)
    m_cos = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64,
                        requires_grad=True)
    m_sin = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64,
                        requires_grad=True)
    r_hat = torch.randn(E_, 3, dtype=torch.float64)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    D = build_D_block(r_hat, l_max).detach().requires_grad_(True)
    tabs = _ef_flat_tables(n_base, l_max, m_max, m_cos.device, m_cos.dtype)

    fn = lambda mc, ms, d: PackUnrotateFused.apply(mc, ms, d, *tabs)  # noqa: E731
    assert torch.autograd.gradcheck(fn, (m_cos, m_sin, D), atol=1e-9)
    assert torch.autograd.gradgradcheck(fn, (m_cos, m_sin, D), atol=1e-9)
    print("test_pack_unrotate_gradchecks: OK")


def test_triton_paths():
    """CUDA-only: Triton fwd/bwd vs the float64 eager reference.

    fp32 tolerances: the kernel computes ieee fp32 dots, so ~1e-5 relative vs
    the fp64 truth is the expected agreement (the eager fp32 path differs from
    fp64 by the same order).
    """
    if not torch.cuda.is_available():
        print("test_triton_paths: SKIP (no CUDA)")
        return
    from ecenet.edge_frame_kernel import _HAS_TRITON
    assert _HAS_TRITON, "CUDA available but triton missing"
    dev = "cuda"

    for l_max, m_max, C_, n_atoms, E_ in [(3, 3, 8, 32, 200), (4, 4, 5, 17, 133),
                                          (2, 1, 4, 9, 40)]:
        n_sph = (l_max + 1) ** 2
        sph = SphToAngular(C_, l_max, m_max=m_max).to(dev).double()
        A64 = torch.randn(n_atoms, C_, n_sph, dtype=torch.float64, device=dev)
        ei = torch.randint(0, n_atoms, (E_,), device=dev)
        ej = (ei + torch.randint(1, n_atoms, (E_,), device=dev)) % n_atoms
        r_hat = torch.randn(E_, 3, dtype=torch.float64, device=dev)
        r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
        D64 = build_D_block(r_hat, l_max)
        tabs64 = (sph.cos_flat_idx, sph.sin_flat_idx, sph.cos_valid, sph.sin_valid)

        # fp64 eager truth (grads via the reference chain)
        A_r = A64.clone().requires_grad_(True)
        D_r = D64.clone().requires_grad_(True)
        c0, s0 = edge_frame_reference(A_r, ei, ej, D_r, *tabs64)
        dc = torch.randn_like(c0)
        ds = torch.randn_like(s0)
        gA0, gD0 = torch.autograd.grad((c0 * dc).sum() + (s0 * ds).sum(), [A_r, D_r])

        # fp32 triton (plain backward → kernel path)
        sph32 = SphToAngular(C_, l_max, m_max=m_max).to(dev)
        tabs32 = (sph32.cos_flat_idx, sph32.sin_flat_idx,
                  sph32.cos_valid, sph32.sin_valid)
        A_t = A64.float().requires_grad_(True)
        D_t = D64.float().requires_grad_(True)
        c1, s1 = EdgeFrameFused.apply(A_t, ei, ej, D_t, *tabs32)
        torch.autograd.backward((c1 * dc.float()).sum() + (s1 * ds.float()).sum())

        scale = c0.abs().max().item()
        assert (c1.double() - c0).abs().max().item() < 1e-4 * max(scale, 1.0)
        assert (s1.double() - s0).abs().max().item() < 1e-4 * max(scale, 1.0)
        gscale = max(gA0.abs().max().item(), gD0.abs().max().item(), 1.0)
        assert (A_t.grad.double() - gA0).abs().max().item() < 1e-3 * gscale
        assert (D_t.grad.double() - gD0).abs().max().item() < 1e-3 * gscale
        print(f"test_triton_paths[l_max={l_max}, m_max={m_max}]: OK")

    # single-source variant (MP steps 5-6) on the same kernels
    from ecenet.edge_frame_kernel import _ef_flat_tables
    n_base, l_max, m_max, n_atoms, E_ = 12, 3, 3, 20, 150
    n_sph = (l_max + 1) ** 2
    Delta64 = torch.randn(n_atoms, n_base, n_sph, dtype=torch.float64, device=dev)
    ei = torch.randint(0, n_atoms, (E_,), device=dev)
    r_hat = torch.randn(E_, 3, dtype=torch.float64, device=dev)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    D64 = build_D_block(r_hat, l_max)
    t64 = _ef_flat_tables(n_base, l_max, m_max, dev, torch.float64)
    t32 = _ef_flat_tables(n_base, l_max, m_max, dev, torch.float32)

    A_r = Delta64.clone().requires_grad_(True)
    D_r = D64.clone().requires_grad_(True)
    c0, s0 = EdgeFrameFused.apply(A_r, ei, None, D_r, *t64)  # fp64 → eager path
    dc = torch.randn_like(c0)
    ds = torch.randn_like(s0)
    gA0, gD0 = torch.autograd.grad((c0 * dc).sum() + (s0 * ds).sum(), [A_r, D_r])

    A_t = Delta64.float().requires_grad_(True)
    D_t = D64.float().requires_grad_(True)
    c1, s1 = EdgeFrameFused.apply(A_t, ei, None, D_t, *t32)  # fp32 → triton
    torch.autograd.backward((c1 * dc.float()).sum() + (s1 * ds.float()).sum())
    assert (c1.double() - c0).abs().max().item() < 1e-4 * max(c0.abs().max().item(), 1.0)
    gscale = max(gA0.abs().max().item(), gD0.abs().max().item(), 1.0)
    assert (A_t.grad.double() - gA0).abs().max().item() < 1e-3 * gscale
    assert (D_t.grad.double() - gD0).abs().max().item() < 1e-3 * gscale
    print("test_triton_paths[single-source]: OK")

    # e2n (fused pack+unrotate+accumulate + role-swapped backward kernels)
    from ecenet.edge_frame_kernel import EdgeToNodeFused
    n_ch = n_base * (l_max + 1)
    gc64 = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64, device=dev,
                       requires_grad=True)
    gs64 = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64, device=dev,
                       requires_grad=True)
    D_r = D64.clone().requires_grad_(True)
    Delta0 = EdgeToNodeFused.apply(gc64, gs64, ei, D_r, n_atoms, *t64)
    dDelta = torch.randn_like(Delta0)
    gc0, gs0, gD0 = torch.autograd.grad((Delta0 * dDelta).sum(),
                                        [gc64, gs64, D_r])

    gc_t = gc64.detach().float().requires_grad_(True)
    gs_t = gs64.detach().float().requires_grad_(True)
    D_t = D64.float().requires_grad_(True)
    Delta1 = EdgeToNodeFused.apply(gc_t, gs_t, ei, D_t, n_atoms, *t32)
    torch.autograd.backward((Delta1 * dDelta.float()).sum())
    assert (Delta1.double() - Delta0).abs().max().item() \
        < 1e-4 * max(Delta0.abs().max().item(), 1.0)
    gscale = max(gc0.abs().max().item(), gD0.abs().max().item(), 1.0)
    assert (gc_t.grad.double() - gc0).abs().max().item() < 1e-3 * gscale
    assert (gs_t.grad.double() - gs0).abs().max().item() < 1e-3 * gscale
    assert (D_t.grad.double() - gD0).abs().max().item() < 1e-3 * gscale
    print("test_triton_paths[e2n]: OK")

    # pack+unrotate (the set_edge_frame_fused e2n model path)
    from ecenet.edge_frame_kernel import PackUnrotateFused
    mc64 = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64, device=dev,
                       requires_grad=True)
    ms64 = torch.randn(E_, n_ch, m_max + 1, dtype=torch.float64, device=dev,
                       requires_grad=True)
    D_r = D64.clone().requires_grad_(True)
    hg0 = PackUnrotateFused.apply(mc64, ms64, D_r, *t64)
    dh = torch.randn_like(hg0)
    gm0, gs0_, gD0 = torch.autograd.grad((hg0 * dh).sum(), [mc64, ms64, D_r])

    mc_t = mc64.detach().float().requires_grad_(True)
    ms_t = ms64.detach().float().requires_grad_(True)
    D_t = D64.float().requires_grad_(True)
    hg1 = PackUnrotateFused.apply(mc_t, ms_t, D_t, *t32)
    torch.autograd.backward((hg1 * dh.float()).sum())
    assert (hg1.double() - hg0).abs().max().item() \
        < 1e-4 * max(hg0.abs().max().item(), 1.0)
    gscale = max(gm0.abs().max().item(), gD0.abs().max().item(), 1.0)
    assert (mc_t.grad.double() - gm0).abs().max().item() < 1e-3 * gscale
    assert (ms_t.grad.double() - gs0_).abs().max().item() < 1e-3 * gscale
    assert (D_t.grad.double() - gD0).abs().max().item() < 1e-3 * gscale
    print("test_triton_paths[pack_unrotate]: OK")


def test_model_integration():
    """Full ECENet (no MP): energy and autograd forces identical flag on/off."""
    import ecenet as _ecenet

    torch.manual_seed(1)
    model = _ecenet.ECENet(n_types=2, r_cut_edge=5.0, r_cut_neighbor=4.0,
                           l_max=2, n_max=3, embed_dim=4, n_max_d=4).double()
    pos0 = torch.randn(6, 3, dtype=torch.float64) * 2.0
    types = torch.randint(0, 2, (6,))

    def run():
        pos = pos0.clone().requires_grad_(True)
        e = model(pos, types)
        (f,) = torch.autograd.grad(e, pos)
        return e.detach(), f

    e0, f0 = run()
    model.set_edge_frame_fused(True)
    e1, f1 = run()
    model.set_edge_frame_fused(False)

    assert torch.allclose(e0, e1, atol=1e-12), (e0 - e1).abs().item()
    assert torch.allclose(f0, f1, atol=1e-12), (f0 - f1).abs().max()
    print("test_model_integration: OK")


def test_mp_integration():
    """Full ECENet with message passing: energy/forces identical flag on/off,
    across both aggregations and the gate variants (heads, l_attention) whose
    node-frame weighting the fused paths must commute with."""
    import ecenet as _ecenet

    configs = [
        dict(mp_type="softmax"),
        dict(mp_type="softmax", mp_n_heads=4, mp_l_attention=True),
        dict(mp_type="sum"),
        dict(mp_type="sum", mp_n_heads=2, mp_l_attention=True),
    ]
    for cfg in configs:
        torch.manual_seed(2)
        model = _ecenet.ECENet(n_types=2, r_cut_edge=5.0, r_cut_neighbor=4.0,
                               l_max=2, n_max=3, embed_dim=4, n_max_d=4,
                               n_mp=2, **cfg).double()
        with torch.no_grad():           # activate the zero-init residual paths
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        pos0 = torch.randn(6, 3, dtype=torch.float64) * 2.0
        types = torch.randint(0, 2, (6,))

        def run():
            pos = pos0.clone().requires_grad_(True)
            e = model(pos, types)
            (f,) = torch.autograd.grad(e, pos)
            return e.detach(), f

        e0, f0 = run()
        model.set_edge_frame_fused(True, e2n=True)    # everything fused
        assert model.mp_layers[0].edge_frame_fused_e2n, "flag reached no MP layer"
        e1, f1 = run()
        model.set_edge_frame_fused(True, e2n=False)   # n2e only
        e2, f2 = run()
        model.set_edge_frame_fused(False)

        for ex, fx in ((e1, f1), (e2, f2)):
            assert torch.allclose(e0, ex, atol=1e-12), (cfg, (e0 - ex).abs().item())
            assert torch.allclose(f0, fx, atol=1e-12), (cfg, (f0 - fx).abs().max())
        print(f"test_mp_integration[{cfg}]: OK")


if __name__ == "__main__":
    test_reference_matches_model_ops()
    test_fused_matches_reference_forward()
    test_backward_matches_reference()
    test_gradcheck()
    test_gradgradcheck()
    test_force_path_smoke()
    test_single_matches_mp_ops()
    test_single_gradchecks()
    test_e2n_matches_mp_ops()
    test_e2n_gradchecks()
    test_pack_unrotate_matches_mp_ops()
    test_pack_unrotate_gradchecks()
    test_triton_paths()
    test_model_integration()
    test_mp_integration()
    print("\nAll tests passed.")

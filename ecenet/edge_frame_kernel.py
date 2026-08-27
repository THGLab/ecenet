"""ecenet/edge_frame_kernel.py — fused node→edge-frame transform.

Isolates the gather → Wigner-rotate → SphToAngular chain from ECENet.forward
(steps 3-4) so it has a single testable spec and a home for a fused Triton
kernel. Per edge e = (i→j):

    A_both[e] = concat(A_emb[edge_i[e]], A_emb[edge_j[e]])   # gather (E, 2C, n_sph)
    A_rot[e]  = A_both[e] @ D_block[e]                        # Wigner rotation
    A_cos[e]  = select_{+m}(A_rot[e]) * cos_valid             # (E, n_ch, n_ang)
    A_sin[e]  = select_{-m}(A_rot[e]) * sin_valid

Run unfused this is ~5 passes over an (E, 2C, n_sph) tensor (two gathers, cat,
bmm, index_select), and autograd saves A_both for the bmm backward. The op is
memory-bound: fusing drops the intermediates and the saved tensor — backward
re-gathers A_both from A_emb (n_atoms-sized, ≪ E-sized) instead.

Three layers, cheapest-to-fastest, all matching the same math:

1. ``edge_frame_reference`` — the plain-PyTorch spec (the exact ops model.py
   runs today; what every test compares against).
2. ``EdgeFrameFused`` — a ``torch.autograd.Function`` with the analytic
   backward and recompute-in-backward. The backward is composed of
   differentiable ops, so the double backward needed for force training works.
   Already a memory win with no kernel; validates the math on CPU.
3. Triton forward/backward kernels (``_ef_*_kernel``), dispatched inside
   ``EdgeFrameFused`` on CUDA + float32. The gather/cat is folded into the A
   load, the SphToAngular select into the D load (column tables), and the
   output permute into the store addressing — none of the (E, 2C, n_sph)
   intermediates ever reach HBM. The single backward uses kernels; under
   ``create_graph`` (force-loss training) backward runs in grad mode and
   automatically takes the differentiable eager path, so double backward
   stays exact. Prior art: fairchem's node_to_edge_wigner_permute.

The adjoint direction (edge-frame → node: pack → rotate-back → scatter, used by
``_aggregate_node_sph``) is this op's transpose; its fusion reuses the same
building blocks and lands here when needed.

Tables come from ``SphToAngular`` (model.py): ``cos_flat_idx``/``sin_flat_idx``
index the flattened (2C·n_sph) axis and already encode the per-l channel
repeat; ``cos_valid``/``sin_valid`` zero the |m| > l (and m=0 sin) slots.
"""

import torch
from torch.profiler import record_function

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:                       # CPU-only / no-triton env → eager fallback
    _HAS_TRITON = False


def edge_frame_reference(A_emb, edge_i, edge_j, D_block,
                         cos_flat_idx, sin_flat_idx, cos_valid, sin_valid):
    """The unfused chain, verbatim (spec for the fused paths).

    Args:
        A_emb:    (n_atoms, C, n_sph) embedded per-atom ACE features
        edge_i/j: (E,) directed edge endpoint indices
        D_block:  (E, n_sph, n_sph) block-diagonal Wigner-D per edge
        *_idx:    (n_ch * n_ang,) flat gather indices from SphToAngular
        *_valid:  (n_ch, n_ang) validity masks from SphToAngular
    Returns:
        A_cos, A_sin: (E, n_ch, n_ang) with n_ch = 2C(l_max+1), n_ang = m_max+1
    """
    n_ch, n_ang = cos_valid.shape
    A_both = torch.cat([A_emb[edge_i], A_emb[edge_j]], dim=1)   # (E, 2C, n_sph)
    A_rot = torch.bmm(A_both, D_block)                          # (E, 2C, n_sph)
    A_flat = A_rot.reshape(A_rot.shape[0], -1)                  # (E, 2C*n_sph)
    A_cos = (A_flat.index_select(1, cos_flat_idx)
             .view(-1, n_ch, n_ang)) * cos_valid
    A_sin = (A_flat.index_select(1, sin_flat_idx)
             .view(-1, n_ch, n_ang)) * sin_valid
    return A_cos, A_sin


def _pack_grads(dA_cos, dA_sin, cos_flat_idx, sin_flat_idx,
                cos_valid, sin_valid, n_flat):
    """Adjoint of the masked select: scatter (dA_cos·valid, dA_sin·valid) back
    into the flat (E, 2C*n_sph) rotated layout. Invalid slots point at flat
    index 0 by construction, so masking BEFORE the scatter is what keeps them
    from contaminating the (l=0, m=0) entry."""
    E = dA_cos.shape[0]
    g_cos = (dA_cos * cos_valid).reshape(E, -1)
    g_sin = (dA_sin * sin_valid).reshape(E, -1)
    g = torch.zeros(E, n_flat, dtype=dA_cos.dtype, device=dA_cos.device)
    g = g.scatter_add(1, cos_flat_idx[None].expand(E, -1), g_cos)
    g = g.scatter_add(1, sin_flat_idx[None].expand(E, -1), g_sin)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernels (CUDA, float32). Design notes:
#
# * The SphToAngular select is folded into the D LOAD: a (P,) column table
#   (col[p] = l²+l±m for p = l·n_ang+m, ok[p] = |m| ≤ l) gathers exactly the
#   D columns each output slot needs, so A_rot never exists. P = (l_max+1)·n_ang.
# * The output (E, n_ch, n_ang) is memory-identical to (E, 2C, P): channel
#   c_out = base·(l_max+1)+l with inner axis m ⟺ flat base·P + (l·n_ang+m).
#   So the result tile stores contiguously — the "permute" is free.
# * The gather+cat is folded into the A LOAD: tile row r reads atom edge_i[e]
#   (r < C) or edge_j[e] (r ≥ C) — A_both never exists either.
# * dD stores need no atomics: cos columns (l²+l+m, m≥0) and sin columns
#   (l²+l−m, m≥1) are disjoint, and each is hit by exactly one p.
# * All tl.dot use input_precision="ieee" (repo rule: fused paths numerically
#   match the reference — no TF32).
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _ef_fwd_kernel(a_ptr, ei_ptr, ej_ptr, d_ptr,
                       cosc_ptr, cosk_ptr, sinc_ptr, sink_ptr,
                       outc_ptr, outs_ptr,
                       C, R, S, P,
                       SP: tl.constexpr, P16: tl.constexpr,
                       BLOCK_R: tl.constexpr):
        e = tl.program_id(0)
        offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)   # rows of A_both
        offs_k = tl.arange(0, SP)                                     # SH index
        offs_p = tl.arange(0, P16)                                    # (l, m) slot

        # gather+cat folded into the load: row r < C ← edge_i, else ← edge_j
        ei = tl.load(ei_ptr + e)
        ej = tl.load(ej_ptr + e)
        row_ok = offs_r < R
        atom = tl.where(offs_r < C, ei, ej)
        ch = tl.where(offs_r < C, offs_r, offs_r - C)
        a_ptrs = a_ptr + (atom * C + ch)[:, None] * S + offs_k[None, :]
        A = tl.load(a_ptrs, mask=row_ok[:, None] & (offs_k[None, :] < S), other=0.0)

        # select folded into the D load: gather the needed columns
        p_ok = offs_p < P
        cos_col = tl.load(cosc_ptr + offs_p, mask=p_ok, other=0)
        cos_ok = tl.load(cosk_ptr + offs_p, mask=p_ok, other=0)
        sin_col = tl.load(sinc_ptr + offs_p, mask=p_ok, other=0)
        sin_ok = tl.load(sink_ptr + offs_p, mask=p_ok, other=0)
        d_base = d_ptr + e * S * S + offs_k[:, None] * S
        Dc = tl.load(d_base + cos_col[None, :],
                     mask=(offs_k[:, None] < S) & (cos_ok[None, :] > 0), other=0.0)
        Ds = tl.load(d_base + sin_col[None, :],
                     mask=(offs_k[:, None] < S) & (sin_ok[None, :] > 0), other=0.0)

        OC = tl.dot(A, Dc, input_precision="ieee")                    # (BLOCK_R, P16)
        OS = tl.dot(A, Ds, input_precision="ieee")

        out_off = e * (R * P) + offs_r[:, None] * P + offs_p[None, :]
        st_mask = row_ok[:, None] & p_ok[None, :]
        tl.store(outc_ptr + out_off, OC, mask=st_mask)
        tl.store(outs_ptr + out_off, OS, mask=st_mask)

    @triton.jit
    def _ef_bwd_dx_kernel(dc_ptr, ds_ptr, d_ptr,
                          cosc_ptr, cosk_ptr, sinc_ptr, sink_ptr,
                          dab_ptr,
                          C, R, S, P,
                          SP: tl.constexpr, P16: tl.constexpr,
                          BLOCK_R: tl.constexpr):
        # dA_both[e, r, k] = Σ_p dcos[e,r,p]·D[e,k,cos_col(p)] + sin part
        e = tl.program_id(0)
        offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, SP)
        offs_p = tl.arange(0, P16)
        row_ok = offs_r < R
        p_ok = offs_p < P

        cos_col = tl.load(cosc_ptr + offs_p, mask=p_ok, other=0)
        cos_ok = tl.load(cosk_ptr + offs_p, mask=p_ok, other=0)
        sin_col = tl.load(sinc_ptr + offs_p, mask=p_ok, other=0)
        sin_ok = tl.load(sink_ptr + offs_p, mask=p_ok, other=0)

        # the ok-mask on the grad load IS the ·valid of the eager path
        g_off = e * (R * P) + offs_r[:, None] * P + offs_p[None, :]
        dC = tl.load(dc_ptr + g_off, mask=row_ok[:, None] & (cos_ok[None, :] > 0), other=0.0)
        dS = tl.load(ds_ptr + g_off, mask=row_ok[:, None] & (sin_ok[None, :] > 0), other=0.0)

        d_base = d_ptr + e * S * S + offs_k[None, :] * S
        DcT = tl.load(d_base + cos_col[:, None],                       # (P16, SP)
                      mask=(cos_ok[:, None] > 0) & (offs_k[None, :] < S), other=0.0)
        DsT = tl.load(d_base + sin_col[:, None],
                      mask=(sin_ok[:, None] > 0) & (offs_k[None, :] < S), other=0.0)

        dA = tl.dot(dC, DcT, input_precision="ieee") \
           + tl.dot(dS, DsT, input_precision="ieee")                   # (BLOCK_R, SP)

        dab_off = e * (R * S) + offs_r[:, None] * S + offs_k[None, :]
        tl.store(dab_ptr + dab_off, dA,
                 mask=row_ok[:, None] & (offs_k[None, :] < S))

    @triton.jit
    def _ef_bwd_dd_kernel(a_ptr, ei_ptr, ej_ptr, dc_ptr, ds_ptr,
                          cosc_ptr, cosk_ptr, sinc_ptr, sink_ptr,
                          dd_ptr,
                          C, R, S, P,
                          SP: tl.constexpr, P16: tl.constexpr,
                          BLOCK_R: tl.constexpr):
        # dD[e, k, cos_col(p)] = Σ_r A_both[e,r,k]·dcos[e,r,p]   (A_both re-gathered)
        e = tl.program_id(0)
        offs_k = tl.arange(0, SP)
        offs_p = tl.arange(0, P16)
        p_ok = offs_p < P

        cos_col = tl.load(cosc_ptr + offs_p, mask=p_ok, other=0)
        cos_ok = tl.load(cosk_ptr + offs_p, mask=p_ok, other=0)
        sin_col = tl.load(sinc_ptr + offs_p, mask=p_ok, other=0)
        sin_ok = tl.load(sink_ptr + offs_p, mask=p_ok, other=0)

        ei = tl.load(ei_ptr + e)
        ej = tl.load(ej_ptr + e)

        acc_c = tl.zeros((SP, P16), dtype=tl.float32)
        acc_s = tl.zeros((SP, P16), dtype=tl.float32)
        for r0 in range(0, R, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            row_ok = offs_r < R
            atom = tl.where(offs_r < C, ei, ej)
            ch = tl.where(offs_r < C, offs_r, offs_r - C)
            a_ptrs = a_ptr + (atom * C + ch)[:, None] * S + offs_k[None, :]
            A = tl.load(a_ptrs, mask=row_ok[:, None] & (offs_k[None, :] < S), other=0.0)

            g_off = e * (R * P) + offs_r[:, None] * P + offs_p[None, :]
            dC = tl.load(dc_ptr + g_off,
                         mask=row_ok[:, None] & (cos_ok[None, :] > 0), other=0.0)
            dS = tl.load(ds_ptr + g_off,
                         mask=row_ok[:, None] & (sin_ok[None, :] > 0), other=0.0)

            acc_c += tl.dot(tl.trans(A), dC, input_precision="ieee")   # (SP, P16)
            acc_s += tl.dot(tl.trans(A), dS, input_precision="ieee")

        # cos and sin column sets are disjoint → plain stores, no atomics
        dd_base = dd_ptr + e * S * S + offs_k[:, None] * S
        tl.store(dd_base + cos_col[None, :], acc_c,
                 mask=(offs_k[:, None] < S) & (cos_ok[None, :] > 0))
        tl.store(dd_base + sin_col[None, :], acc_s,
                 mask=(offs_k[:, None] < S) & (sin_ok[None, :] > 0))

    @triton.jit
    def _ef_bwd_merged_kernel(dc_ptr, ds_ptr, a_ptr, ei_ptr, ej_ptr, d_ptr,
                              cosc_ptr, cosk_ptr, sinc_ptr, sink_ptr,
                              dab_ptr, dd_ptr,
                              C, R, S, P,
                              SP: tl.constexpr, P16: tl.constexpr,
                              BLOCK_R: tl.constexpr):
        # dX and dD in one pass: the grad tiles (the backward's dominant read)
        # are loaded ONCE and feed both the @Dᵀ dot (→ dA_both) and the outer
        # product with the re-gathered A rows (→ dD). One program per edge;
        # requires BLOCK_R ≥ R (dispatch falls back to the split kernels else).
        e = tl.program_id(0)
        offs_r = tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, SP)
        offs_p = tl.arange(0, P16)
        row_ok = offs_r < R
        p_ok = offs_p < P
        k_ok = offs_k < S

        cos_col = tl.load(cosc_ptr + offs_p, mask=p_ok, other=0)
        cos_ok = tl.load(cosk_ptr + offs_p, mask=p_ok, other=0)
        sin_col = tl.load(sinc_ptr + offs_p, mask=p_ok, other=0)
        sin_ok = tl.load(sink_ptr + offs_p, mask=p_ok, other=0)

        # grad tiles — the shared read (ok-mask = the ·valid of the eager path)
        g_off = e * (R * P) + offs_r[:, None] * P + offs_p[None, :]
        dC = tl.load(dc_ptr + g_off,
                     mask=row_ok[:, None] & (cos_ok[None, :] > 0), other=0.0)
        dS = tl.load(ds_ptr + g_off,
                     mask=row_ok[:, None] & (sin_ok[None, :] > 0), other=0.0)

        # dA_both = dC @ DcT + dS @ DsT   (DxT[p, k] = D[e, k, col(p)])
        d_base_t = d_ptr + e * S * S + offs_k[None, :] * S
        DcT = tl.load(d_base_t + cos_col[:, None],
                      mask=(cos_ok[:, None] > 0) & k_ok[None, :], other=0.0)
        DsT = tl.load(d_base_t + sin_col[:, None],
                      mask=(sin_ok[:, None] > 0) & k_ok[None, :], other=0.0)
        dA = tl.dot(dC, DcT, input_precision="ieee") \
           + tl.dot(dS, DsT, input_precision="ieee")
        dab_off = e * (R * S) + offs_r[:, None] * S + offs_k[None, :]
        tl.store(dab_ptr + dab_off, dA, mask=row_ok[:, None] & k_ok[None, :])

        # dD = Aᵀ @ grads  (A re-gathered: atom-sized source, L2-resident)
        ei = tl.load(ei_ptr + e)
        ej = tl.load(ej_ptr + e)
        atom = tl.where(offs_r < C, ei, ej)
        ch = tl.where(offs_r < C, offs_r, offs_r - C)
        a_ptrs = a_ptr + (atom * C + ch)[:, None] * S + offs_k[None, :]
        A = tl.load(a_ptrs, mask=row_ok[:, None] & k_ok[None, :], other=0.0)
        acc_c = tl.dot(tl.trans(A), dC, input_precision="ieee")   # (SP, P16)
        acc_s = tl.dot(tl.trans(A), dS, input_precision="ieee")
        dd_base = dd_ptr + e * S * S + offs_k[:, None] * S
        tl.store(dd_base + cos_col[None, :], acc_c,
                 mask=k_ok[:, None] & (cos_ok[None, :] > 0))
        tl.store(dd_base + sin_col[None, :], acc_s,
                 mask=k_ok[:, None] & (sin_ok[None, :] > 0))

    @triton.jit
    def _pu_bwd_merged_kernel(dh_ptr, mc_ptr, ms_ptr, d_ptr,
                              cosc_ptr, cosk_ptr, sinc_ptr, sink_ptr,
                              dmc_ptr, dms_ptr, dd_ptr,
                              R, S, P,
                              SP: tl.constexpr, P16: tl.constexpr,
                              BLOCK_R: tl.constexpr):
        # PackUnrotate's dm and dD in one pass: dh (the dominant read) is
        # loaded ONCE, feeding dm = select(dh @ D) and dD = dhᵀ @ pack(m).
        # One program per edge; requires BLOCK_R ≥ R.
        e = tl.program_id(0)
        offs_r = tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, SP)
        offs_p = tl.arange(0, P16)
        row_ok = offs_r < R
        p_ok = offs_p < P
        k_ok = offs_k < S

        cos_col = tl.load(cosc_ptr + offs_p, mask=p_ok, other=0)
        cos_ok = tl.load(cosk_ptr + offs_p, mask=p_ok, other=0)
        sin_col = tl.load(sinc_ptr + offs_p, mask=p_ok, other=0)
        sin_ok = tl.load(sink_ptr + offs_p, mask=p_ok, other=0)

        # the shared read
        dh_off = e * (R * S) + offs_r[:, None] * S + offs_k[None, :]
        dh = tl.load(dh_ptr + dh_off,
                     mask=row_ok[:, None] & k_ok[None, :], other=0.0)

        # dm = dh @ D-cols  (Dx[k, p] = D[e, k, col(p)])
        d_base = d_ptr + e * S * S + offs_k[:, None] * S
        Dc = tl.load(d_base + cos_col[None, :],
                     mask=k_ok[:, None] & (cos_ok[None, :] > 0), other=0.0)
        Ds = tl.load(d_base + sin_col[None, :],
                     mask=k_ok[:, None] & (sin_ok[None, :] > 0), other=0.0)
        dmc = tl.dot(dh, Dc, input_precision="ieee")              # (BLOCK_R, P16)
        dms = tl.dot(dh, Ds, input_precision="ieee")
        out_off = e * (R * P) + offs_r[:, None] * P + offs_p[None, :]
        st_mask = row_ok[:, None] & p_ok[None, :]
        tl.store(dmc_ptr + out_off, dmc, mask=st_mask)
        tl.store(dms_ptr + out_off, dms, mask=st_mask)

        # dD = dhᵀ @ h, with h's columns read straight from m (pack-on-load)
        mC = tl.load(mc_ptr + out_off,
                     mask=row_ok[:, None] & (cos_ok[None, :] > 0), other=0.0)
        mS = tl.load(ms_ptr + out_off,
                     mask=row_ok[:, None] & (sin_ok[None, :] > 0), other=0.0)
        acc_c = tl.dot(tl.trans(dh), mC, input_precision="ieee")  # (SP, P16)
        acc_s = tl.dot(tl.trans(dh), mS, input_precision="ieee")
        dd_base = dd_ptr + e * S * S + offs_k[:, None] * S
        tl.store(dd_base + cos_col[None, :], acc_c,
                 mask=k_ok[:, None] & (cos_ok[None, :] > 0))
        tl.store(dd_base + sin_col[None, :], acc_s,
                 mask=k_ok[:, None] & (sin_ok[None, :] > 0))

    @triton.jit
    def _e2n_fwd_kernel(gc_ptr, gs_ptr, d_ptr, perm_ptr, aptr_ptr,
                        srcoff_ptr, okc_ptr, oks_ptr,
                        delta_ptr,
                        NB, S, P,
                        SP: tl.constexpr, RB: tl.constexpr):
        # One program per ATOM a: loop its in-edges (sorted layout), pack the
        # gated SO(2) features into SH in-register, unrotate with D_eᵀ, and
        # accumulate — Delta[a] is written once. Fixed loop order → deterministic
        # (no atomics), same trick as tri_kernel's per-node blocks.
        a = tl.program_id(0)
        start = tl.load(aptr_ptr + a)
        end = tl.load(aptr_ptr + a + 1)
        offs_b = tl.arange(0, RB)                 # base channels (rows)
        offs_k = tl.arange(0, SP)                 # SH index
        b_ok = offs_b < NB
        k_ok = offs_k < S

        # k-indexed pack tables: SH slot k ← (l(k)·n_ang + |m(k)|) of cos or sin
        srcoff = tl.load(srcoff_ptr + offs_k, mask=k_ok, other=0)
        okc = tl.load(okc_ptr + offs_k, mask=k_ok, other=0)
        oks = tl.load(oks_ptr + offs_k, mask=k_ok, other=0)

        acc = tl.zeros((RB, SP), dtype=tl.float32)
        for idx in range(start, end):
            e = tl.load(perm_ptr + idx)
            # pack folded into the load: h[b, k] = g_cos/g_sin[b·P + srcoff(k)]
            g_off = e * (NB * P) + offs_b[:, None] * P + srcoff[None, :]
            hc = tl.load(gc_ptr + g_off,
                         mask=b_ok[:, None] & (okc[None, :] > 0), other=0.0)
            hs = tl.load(gs_ptr + g_off,
                         mask=b_ok[:, None] & (oks[None, :] > 0), other=0.0)
            h = hc + hs
            # unrotate: (h @ Dᵀ)[b, i] = Σ_k h[b,k]·D[i,k] → Dt[k, i] = D[e, i, k]
            dt = tl.load(d_ptr + e * S * S + offs_k[None, :] * S + offs_k[:, None],
                         mask=k_ok[:, None] & k_ok[None, :], other=0.0)
            acc += tl.dot(h, dt, input_precision="ieee")

        st = delta_ptr + a * (NB * S) + offs_b[:, None] * S + offs_k[None, :]
        tl.store(st, acc, mask=b_ok[:, None] & k_ok[None, :])


_EF_TABLES: dict = {}


def _next_pow2(x: int) -> int:
    """Smallest power of two ≥ x, floored at 16. tl.arange REQUIRES a power
    of two (a multiple of 16 like 48 or 96 compiles to "arange's range must
    be a power of 2"), and tl.dot needs each dim ≥ 16; every kernel load and
    store masks the padding (offs < R/S/P), so the rounding is free of
    correctness effects."""
    return max(16, 1 << (max(x, 1) - 1).bit_length())


def _ef_block_r(n_rows: int) -> int:
    """Row-tile size: cover all rows of one edge in a single program when
    reasonable (≤128), so D and the column tables are loaded once per edge —
    with BLOCK_R < n_rows each edge pays for them per row-block."""
    return min(128, _next_pow2(n_rows))


def _ef_tables(n_sph: int, n_ang: int, device):
    """Per-(l, m)-slot column tables: col[p] = l²+l±m, ok[p] = slot is valid.
    p = l·n_ang + m enumerates the (l_max+1)·n_ang output slots of one base
    channel. Cached per (n_sph, n_ang, device)."""
    key = (n_sph, n_ang, str(device))
    if key not in _EF_TABLES:
        l_max = int(round(n_sph ** 0.5)) - 1
        P = (l_max + 1) * n_ang
        cos_col = torch.zeros(P, dtype=torch.int32)
        cos_ok = torch.zeros(P, dtype=torch.int32)
        sin_col = torch.zeros(P, dtype=torch.int32)
        sin_ok = torch.zeros(P, dtype=torch.int32)
        for l in range(l_max + 1):
            for m in range(n_ang):
                p = l * n_ang + m
                if m <= l:
                    cos_col[p] = l * l + l + m
                    cos_ok[p] = 1
                    if m > 0:
                        sin_col[p] = l * l + l - m
                        sin_ok[p] = 1
        _EF_TABLES[key] = tuple(t.to(device) for t in
                                (cos_col, cos_ok, sin_col, sin_ok))
    return _EF_TABLES[key]


_EF_KTABLES: dict = {}


def _ef_ktables(n_sph: int, n_ang: int, device):
    """k-indexed pack tables for the e2n kernel: SH slot k = l²+l+m_s reads
    (l·n_ang + |m_s|) of cos (m_s ≥ 0) or sin (m_s < 0); slots with
    |m_s| ≥ n_ang (m_max truncation) contribute 0."""
    key = (n_sph, n_ang, str(device))
    if key not in _EF_KTABLES:
        l_max = int(round(n_sph ** 0.5)) - 1
        srcoff = torch.zeros(n_sph, dtype=torch.int32)
        okc = torch.zeros(n_sph, dtype=torch.int32)
        oks = torch.zeros(n_sph, dtype=torch.int32)
        for l in range(l_max + 1):
            for m_s in range(-l, l + 1):
                k = l * l + l + m_s
                if abs(m_s) < n_ang:
                    srcoff[k] = l * n_ang + abs(m_s)
                    if m_s >= 0:
                        okc[k] = 1
                    else:
                        oks[k] = 1
        _EF_KTABLES[key] = tuple(t.to(device) for t in (srcoff, okc, oks))
    return _EF_KTABLES[key]


def _e2n_layout(edge_dst, n_atoms):
    """Sorted-by-receiver CSR layout: perm (E,) stable-sorted edge ids and
    aptr (n_atoms+1,) offsets. Deterministic accumulation order."""
    perm = torch.argsort(edge_dst, stable=True)
    counts = torch.bincount(edge_dst, minlength=n_atoms)
    aptr = torch.zeros(n_atoms + 1, dtype=torch.long, device=edge_dst.device)
    torch.cumsum(counts, 0, out=aptr[1:])
    return perm.contiguous(), aptr.contiguous()


# Per-atom accumulation (deterministic, Delta written once) turned out to be
# LATENCY-BOUND: the grid is only n_atoms programs, each serially looping its
# in-degree with a dependent accumulator — ~coordination× less parallelism than
# the per-edge kernels, and measurably slower at MD sizes. Default is the
# per-edge strategy below; the per-atom kernel is kept behind this switch for
# small-atom-count/regression experiments.
_E2N_PER_ATOM = False


def _e2n_forward_triton(g_cos, g_sin, edge_dst, D_block, n_atoms, n_base):
    S = D_block.shape[-1]
    n_ang = g_cos.shape[-1]
    P = (g_cos.shape[1] // n_base) * n_ang

    if _E2N_PER_ATOM:
        srcoff, okc, oks = _ef_ktables(S, n_ang, g_cos.device)
        perm, aptr = _e2n_layout(edge_dst, n_atoms)
        Delta = torch.empty(n_atoms, n_base, S, dtype=g_cos.dtype,
                            device=g_cos.device)
        _e2n_fwd_kernel[(n_atoms,)](
            g_cos.contiguous(), g_sin.contiguous(), D_block.contiguous(),
            perm, aptr, srcoff, okc, oks, Delta,
            n_base, S, P,
            SP=_next_pow2(S), RB=_next_pow2(n_base))
        return Delta

    # Per-edge: pack+unrotate is exactly the dx backward kernel's math
    # (ok-masked grad loads → dot with the column-gathered Dᵀ), run over E
    # programs; the accumulation is a plain index_add (same atomics the
    # unfused scatter_add uses — no determinism regression vs baseline).
    cos_col, cos_ok, sin_col, sin_ok = _ef_tables(S, n_ang, g_cos.device)
    h_global = torch.empty(edge_dst.shape[0], n_base, S,
                           dtype=g_cos.dtype, device=g_cos.device)
    block_r = _ef_block_r(n_base)
    grid = (edge_dst.shape[0], triton.cdiv(n_base, block_r))
    _ef_bwd_dx_kernel[grid](
        g_cos.contiguous(), g_sin.contiguous(), D_block.contiguous(),
        cos_col, cos_ok, sin_col, sin_ok, h_global,
        n_base, n_base, S, P,
        SP=_next_pow2(S), P16=_next_pow2(P), BLOCK_R=block_r)
    Delta = torch.zeros(n_atoms, n_base, S, dtype=g_cos.dtype,
                        device=g_cos.device)
    Delta.index_add_(0, edge_dst, h_global)
    return Delta


def _ef_triton_ok(t: torch.Tensor, n_edges: int) -> bool:
    return _HAS_TRITON and t.is_cuda and t.dtype == torch.float32 and n_edges > 0


def _ef_forward_triton(A_emb, edge_i, edge_j, D_block, n_ch, n_ang):
    """edge_j=None → single-source: rows are A_emb[edge_i]'s channels (R=C, the
    r<C split never picks the second endpoint)."""
    E = edge_i.shape[0]
    C = A_emb.shape[1]
    R = C if edge_j is None else 2 * C
    ej = edge_i if edge_j is None else edge_j
    S = D_block.shape[-1]
    P = (n_ch // R) * n_ang
    cos_col, cos_ok, sin_col, sin_ok = _ef_tables(S, n_ang, A_emb.device)
    A_cos = torch.empty(E, n_ch, n_ang, dtype=A_emb.dtype, device=A_emb.device)
    A_sin = torch.empty_like(A_cos)
    block_r = _ef_block_r(R)
    grid = (E, triton.cdiv(R, block_r))
    _ef_fwd_kernel[grid](
        A_emb.contiguous(), edge_i.contiguous(), ej.contiguous(),
        D_block.contiguous(),
        cos_col, cos_ok, sin_col, sin_ok,
        A_cos, A_sin,
        C, R, S, P,
        SP=_next_pow2(S), P16=_next_pow2(P), BLOCK_R=block_r)
    return A_cos, A_sin


def _ef_backward_triton(dA_cos, dA_sin, A_emb, edge_i, edge_j, D_block,
                        n_ang, single, need_dx, need_dd):
    E = edge_i.shape[0]
    C = A_emb.shape[1]
    R = C if single else 2 * C
    S = D_block.shape[-1]
    n_ch = dA_cos.shape[1]
    P = (n_ch // R) * n_ang
    cos_col, cos_ok, sin_col, sin_ok = _ef_tables(S, n_ang, A_emb.device)
    dA_cos = dA_cos.contiguous()
    dA_sin = dA_sin.contiguous()
    D_block = D_block.contiguous()
    A_emb_c = A_emb.contiguous()
    args = (cos_col, cos_ok, sin_col, sin_ok)
    block_r = _ef_block_r(R)
    kw = dict(SP=_next_pow2(S), P16=_next_pow2(P), BLOCK_R=block_r)

    def _scatter(dA_both):
        # scatter back to atoms (torch: well-optimized, visible to compile)
        dA_emb = torch.zeros_like(A_emb)
        if single:
            dA_emb.index_add_(0, edge_i, dA_both)
        else:
            dA_emb.index_add_(0, edge_i, dA_both[:, :C])
            dA_emb.index_add_(0, edge_j, dA_both[:, C:])
        return dA_emb

    # Merged path: one kernel reads the grad tiles once and emits both dA and
    # dD — the split path pays that (dominant) read twice. Needs one row-block
    # to cover the edge (block_r ≥ R; true for n_base/2C ≤ 128).
    if need_dx and need_dd and block_r >= R:
        dA_both = torch.empty(E, R, S, dtype=dA_cos.dtype, device=dA_cos.device)
        dD = torch.zeros_like(D_block)     # columns no p touches stay 0
        _ef_bwd_merged_kernel[(E,)](
            dA_cos, dA_sin, A_emb_c, edge_i.contiguous(), edge_j.contiguous(),
            D_block, *args, dA_both, dD,
            C, R, S, P, **kw)
        return _scatter(dA_both), dD

    dA_emb = None
    if need_dx:
        dA_both = torch.empty(E, R, S, dtype=dA_cos.dtype, device=dA_cos.device)
        grid = (E, triton.cdiv(R, block_r))
        _ef_bwd_dx_kernel[grid](dA_cos, dA_sin, D_block, *args, dA_both,
                                C, R, S, P, **kw)
        dA_emb = _scatter(dA_both)

    dD = None
    if need_dd:
        dD = torch.zeros_like(D_block)     # columns no p touches stay 0
        _ef_bwd_dd_kernel[(E,)](A_emb_c, edge_i.contiguous(), edge_j.contiguous(),
                                dA_cos, dA_sin, *args, dD,
                                C, R, S, P, **kw)
    return dA_emb, dD


class EdgeFrameFused(torch.autograd.Function):
    """Fused gather → rotate → select with analytic, double-differentiable
    backward. Saves (A_emb, D_block, indices) — NOT the (E, R, n_sph)
    intermediates; the gathered rows are re-gathered in the backward.

    edge_j=None → single-source mode (MP steps 5-6): rows are A_emb[edge_i]'s
    channels alone (R = C, no cat); otherwise both endpoints concatenated
    (R = 2C)."""

    @staticmethod
    def forward(ctx, A_emb, edge_i, edge_j, D_block,
                cos_flat_idx, sin_flat_idx, cos_valid, sin_valid):
        n_ch, n_ang = cos_valid.shape
        single = edge_j is None
        with torch.no_grad():
            if _ef_triton_ok(A_emb, edge_i.shape[0]):
                A_cos, A_sin = _ef_forward_triton(
                    A_emb, edge_i, edge_j, D_block, n_ch, n_ang)
            else:
                A_both = (A_emb[edge_i] if single else
                          torch.cat([A_emb[edge_i], A_emb[edge_j]], dim=1))
                A_rot = torch.bmm(A_both, D_block)
                A_flat = A_rot.reshape(A_rot.shape[0], -1)
                A_cos = (A_flat.index_select(1, cos_flat_idx)
                         .view(-1, n_ch, n_ang)) * cos_valid
                A_sin = (A_flat.index_select(1, sin_flat_idx)
                         .view(-1, n_ch, n_ang)) * sin_valid
        ctx.save_for_backward(A_emb, edge_i,
                              edge_i if single else edge_j, D_block,
                              cos_flat_idx, sin_flat_idx, cos_valid, sin_valid)
        ctx.n_ang = n_ang
        ctx.single = single
        return A_cos, A_sin

    @staticmethod
    def backward(ctx, dA_cos, dA_sin):
        (A_emb, edge_i, edge_j, D_block,
         cos_flat_idx, sin_flat_idx, cos_valid, sin_valid) = ctx.saved_tensors
        E = dA_cos.shape[0]
        C = A_emb.shape[1]
        single = ctx.single
        R = C if single else 2 * C
        n_sph = D_block.shape[-1]

        # Triton path for the plain (single) backward. Under create_graph —
        # force-loss training — grad mode is enabled inside backward, and the
        # differentiable eager path below is required for the double backward.
        if not torch.is_grad_enabled() and _ef_triton_ok(dA_cos, E):
            dA_emb, dD = _ef_backward_triton(
                dA_cos, dA_sin, A_emb, edge_i, edge_j, D_block, ctx.n_ang,
                single=single,
                need_dx=ctx.needs_input_grad[0],
                need_dd=ctx.needs_input_grad[3])
            return dA_emb, None, None, dD, None, None, None, None

        # Adjoint of select: scatter masked grads into the rotated layout.
        g_rot = _pack_grads(dA_cos, dA_sin, cos_flat_idx, sin_flat_idx,
                            cos_valid, sin_valid, R * n_sph)
        g_rot = g_rot.view(E, R, n_sph)

        # Adjoint of rotate.
        dA_both = torch.bmm(g_rot, D_block.transpose(1, 2)) \
            if ctx.needs_input_grad[0] else None

        dD = None
        if ctx.needs_input_grad[3]:
            # Recompute the gathered rows (cheaper than saving (E,R,n_sph)).
            A_both = (A_emb[edge_i] if single else
                      torch.cat([A_emb[edge_i], A_emb[edge_j]], dim=1))
            dD = torch.bmm(A_both.transpose(1, 2), g_rot)

        # Adjoint of gather: scatter-add back to atoms.
        dA_emb = None
        if ctx.needs_input_grad[0]:
            dA_emb = torch.zeros_like(A_emb)
            if single:
                dA_emb = dA_emb.index_add(0, edge_i, dA_both)
            else:
                dA_emb = dA_emb.index_add(0, edge_i, dA_both[:, :C])
                dA_emb = dA_emb.index_add(0, edge_j, dA_both[:, C:])

        return dA_emb, None, None, dD, None, None, None, None


def edge_frame_fused(A_emb, edge_i, edge_j, D_block, sph_to_angular):
    """Convenience wrapper taking the model's SphToAngular module."""
    # Tag for --profile: replaces the unfused path's "wigner_rotate" bmm plus
    # the gather/cat/index_select traffic around it.
    with record_function("edge_frame_fused"):
        return EdgeFrameFused.apply(
            A_emb, edge_i, edge_j, D_block,
            sph_to_angular.cos_flat_idx, sph_to_angular.sin_flat_idx,
            sph_to_angular.cos_valid, sph_to_angular.sin_valid)


_EF_FLAT_TABLES: dict = {}


def _ef_flat_tables(n_base, l_max, m_max, device, dtype):
    """SphToAngular-equivalent flat tables for n_base ungathered channels
    (the MP layers' _unpack layout: channel c = b*(l_max+1)+l, mode m).
    Same (l, m) mapping as SphToAngular — +m → l²+l+m, −m → l²+l−m, |m| ≤
    min(l, m_max) — verified against _unpack_sph_to_angular in the tests."""
    key = (n_base, l_max, m_max, str(device), dtype)
    if key not in _EF_FLAT_TABLES:
        lp1, n_ang = l_max + 1, m_max + 1
        n_sph = lp1 * lp1
        n_ch = n_base * lp1
        cos_idx = torch.zeros(n_ch, n_ang, dtype=torch.long)
        sin_idx = torch.zeros(n_ch, n_ang, dtype=torch.long)
        cos_valid = torch.zeros(n_ch, n_ang)
        sin_valid = torch.zeros(n_ch, n_ang)
        c = 0
        for _ in range(n_base):
            for l in range(lp1):
                base = l * l + l
                for m in range(n_ang):
                    if m <= l:
                        cos_idx[c, m] = base + m
                        cos_valid[c, m] = 1.0
                        if m > 0:
                            sin_idx[c, m] = base - m
                            sin_valid[c, m] = 1.0
                c += 1
        ch_src = torch.arange(n_ch) // lp1
        cos_flat = (ch_src[:, None] * n_sph + cos_idx).reshape(-1)
        sin_flat = (ch_src[:, None] * n_sph + sin_idx).reshape(-1)
        _EF_FLAT_TABLES[key] = (cos_flat.to(device), sin_flat.to(device),
                                cos_valid.to(device, dtype),
                                sin_valid.to(device, dtype))
    return _EF_FLAT_TABLES[key]


def edge_frame_fused_single(A_node, edge_i, D_block, l_max, m_max):
    """Single-source variant for the MP layers' steps 5-6:
    bmm(A_node[edge_i], D_block) → _unpack_sph_to_angular → reshape, as one op.

    Args:
        A_node: (n_atoms, n_base, n_sph) aggregated node features (Delta)
        edge_i: (E,) source atom per edge
        D_block: (E, n_sph, n_sph)
    Returns:
        d_cos, d_sin: (E, n_base*(l_max+1), m_max+1)
    """
    tabs = _ef_flat_tables(A_node.shape[1], l_max, m_max,
                           A_node.device, A_node.dtype)
    with record_function("edge_frame_fused_single"):
        return EdgeFrameFused.apply(A_node, edge_i, None, D_block, *tabs)


# ─────────────────────────────────────────────────────────────────────────────
# Edge → node direction (the MP layers' pack → unrotate → scatter). Adjoint of
# the op above: any per-edge gate that is constant across m within each l
# (attention weights, msg weights, f_cut envelopes — the only gates equivariance
# permits) commutes with the block-diagonal rotation, so it is applied EAGERLY
# on (A_cos, A_sin) before this op and never enters the kernel. The backward is
# the n2e machinery with roles swapped: d(g) = rotate+select of the gathered
# dDelta (the _ef_fwd kernel), dD = the _ef_bwd_dd kernel.
# ─────────────────────────────────────────────────────────────────────────────

def edge_to_node_reference(g_cos, g_sin, edge_dst, D_block, n_atoms,
                           cos_flat_idx, sin_flat_idx, cos_valid, sin_valid):
    """The unfused chain, verbatim: pack → bmm(h, Dᵀ) → scatter_add.

    Args:
        g_cos/g_sin: (E, n_ch, n_ang) gated bond-frame features
        edge_dst:    (E,) receiver atom per edge
        D_block:     (E, n_sph, n_sph)
    Returns:
        Delta: (n_atoms, n_base, n_sph) aggregated in the common global frame
    """
    E = g_cos.shape[0]
    n_sph = D_block.shape[-1]
    n_base = (cos_flat_idx.numel() // cos_valid.shape[1]) // (
        int(round(n_sph ** 0.5)))          # n_ch // lp1
    h = _pack_grads(g_cos, g_sin, cos_flat_idx, sin_flat_idx,
                    cos_valid, sin_valid, n_base * n_sph).view(E, n_base, n_sph)
    h_global = torch.bmm(h, D_block.transpose(-1, -2))
    idx = edge_dst[:, None, None].expand_as(h_global)
    return torch.zeros(n_atoms, n_base, n_sph, device=g_cos.device,
                       dtype=g_cos.dtype).scatter_add(0, idx, h_global)


class EdgeToNodeFused(torch.autograd.Function):
    """Fused pack → unrotate → per-atom accumulate, with analytic,
    double-differentiable backward. Nothing edge-sized is saved beyond the
    inputs themselves (which the surrounding graph holds anyway)."""

    @staticmethod
    def forward(ctx, g_cos, g_sin, edge_dst, D_block, n_atoms,
                cos_flat_idx, sin_flat_idx, cos_valid, sin_valid):
        n_ch, n_ang = cos_valid.shape
        n_sph = D_block.shape[-1]
        n_base = n_ch // (int(round(n_sph ** 0.5)))
        with torch.no_grad():
            if (_ef_triton_ok(g_cos, edge_dst.shape[0])
                    and _next_pow2(n_base) <= 128):
                Delta = _e2n_forward_triton(g_cos, g_sin, edge_dst, D_block,
                                            n_atoms, n_base)
            else:
                Delta = edge_to_node_reference(
                    g_cos, g_sin, edge_dst, D_block, n_atoms,
                    cos_flat_idx, sin_flat_idx, cos_valid, sin_valid)
        ctx.save_for_backward(g_cos, g_sin, edge_dst, D_block,
                              cos_flat_idx, sin_flat_idx, cos_valid, sin_valid)
        ctx.n_ang = n_ang
        return Delta

    @staticmethod
    def backward(ctx, dDelta):
        (g_cos, g_sin, edge_dst, D_block,
         cos_flat_idx, sin_flat_idx, cos_valid, sin_valid) = ctx.saved_tensors
        E = edge_dst.shape[0]
        n_ch, n_ang = cos_valid.shape
        n_sph = D_block.shape[-1]
        n_base = n_ch // (int(round(n_sph ** 0.5)))
        need_dg = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_dd = ctx.needs_input_grad[3]

        # Triton path (plain backward): dg is exactly the n2e FORWARD on the
        # gathered dDelta; dD is the shared dd kernel with swapped roles.
        if not torch.is_grad_enabled() and _ef_triton_ok(dDelta, E):
            dDelta_c = dDelta.contiguous()
            dg_cos = dg_sin = None
            if need_dg:
                dg_cos, dg_sin = _ef_forward_triton(
                    dDelta_c, edge_dst, None, D_block, n_ch, n_ang)
            dD = None
            if need_dd:
                cos_col, cos_ok, sin_col, sin_ok = _ef_tables(
                    n_sph, n_ang, dDelta.device)
                P = (n_ch // n_base) * n_ang
                dD = torch.zeros_like(D_block)
                block_r = _ef_block_r(n_base)
                _ef_bwd_dd_kernel[(E,)](
                    dDelta_c, edge_dst.contiguous(), edge_dst.contiguous(),
                    g_cos.contiguous(), g_sin.contiguous(),
                    cos_col, cos_ok, sin_col, sin_ok, dD,
                    n_base, n_base, n_sph, P,
                    SP=_next_pow2(n_sph), P16=_next_pow2(P), BLOCK_R=block_r)
            return dg_cos, dg_sin, None, dD, None, None, None, None, None

        # Eager (double-differentiable) path.
        G = dDelta[edge_dst]                                # (E, n_base, S)
        dg_cos = dg_sin = None
        if need_dg:
            g_rot = torch.bmm(G, D_block)                   # rotate to edge frame
            g_flat = g_rot.reshape(E, -1)
            dg_cos = (g_flat.index_select(1, cos_flat_idx)
                      .view(E, n_ch, n_ang)) * cos_valid
            dg_sin = (g_flat.index_select(1, sin_flat_idx)
                      .view(E, n_ch, n_ang)) * sin_valid
        dD = None
        if need_dd:
            h = _pack_grads(g_cos, g_sin, cos_flat_idx, sin_flat_idx,
                            cos_valid, sin_valid,
                            n_base * n_sph).view(E, n_base, n_sph)
            dD = torch.bmm(G.transpose(1, 2), h)            # (E, S, S)
        return dg_cos, dg_sin, None, dD, None, None, None, None, None


class PackUnrotateFused(torch.autograd.Function):
    """Fused pack → unrotate per edge: (m_cos, m_sin) → h_global = pack(m) @ Dᵀ.

    No gather, no scatter, no gate — the attention/message weighting and the
    node accumulation stay eager in the node frame exactly as the unfused
    path. The win is dropping the packed intermediate h from HBM (forward) and
    from the bmm's saved set (backward). All three kernels are the existing
    ones: forward = _ef_bwd_dx (its "grads" input is any per-edge (E, R, P)
    tensor), backward dm = _ef_fwd with identity indices, dD = _ef_bwd_dd."""

    @staticmethod
    def forward(ctx, m_cos, m_sin, D_block,
                cos_flat_idx, sin_flat_idx, cos_valid, sin_valid):
        E = m_cos.shape[0]
        n_ch, n_ang = cos_valid.shape
        S = D_block.shape[-1]
        n_base = n_ch // (int(round(S ** 0.5)))
        with torch.no_grad():
            if _ef_triton_ok(m_cos, E) and _next_pow2(n_base) <= 128:
                cos_col, cos_ok, sin_col, sin_ok = _ef_tables(S, n_ang, m_cos.device)
                P = (n_ch // n_base) * n_ang
                h_global = torch.empty(E, n_base, S, dtype=m_cos.dtype,
                                       device=m_cos.device)
                block_r = _ef_block_r(n_base)
                grid = (E, triton.cdiv(n_base, block_r))
                _ef_bwd_dx_kernel[grid](
                    m_cos.contiguous(), m_sin.contiguous(),
                    # forward wants h @ Dᵀ where the dx kernel computes
                    # Σ_p g[·,p]·D[k, col(p)] — i.e. it contracts against D's
                    # COLUMNS, which is exactly the transpose we need.
                    D_block.contiguous(),
                    cos_col, cos_ok, sin_col, sin_ok, h_global,
                    n_base, n_base, S, P,
                    SP=_next_pow2(S), P16=_next_pow2(P), BLOCK_R=block_r)
            else:
                h = _pack_grads(m_cos, m_sin, cos_flat_idx, sin_flat_idx,
                                cos_valid, sin_valid,
                                n_base * S).view(E, n_base, S)
                h_global = torch.bmm(h, D_block.transpose(-1, -2))
        ctx.save_for_backward(m_cos, m_sin, D_block,
                              cos_flat_idx, sin_flat_idx, cos_valid, sin_valid)
        return h_global

    @staticmethod
    def backward(ctx, dh):
        (m_cos, m_sin, D_block,
         cos_flat_idx, sin_flat_idx, cos_valid, sin_valid) = ctx.saved_tensors
        E = m_cos.shape[0]
        n_ch, n_ang = cos_valid.shape
        S = D_block.shape[-1]
        n_base = n_ch // (int(round(S ** 0.5)))
        need_dm = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_dd = ctx.needs_input_grad[2]

        if not torch.is_grad_enabled() and _ef_triton_ok(dh, E):
            dh_c = dh.contiguous()
            P = (n_ch // n_base) * n_ang
            block_r = _ef_block_r(n_base)
            tabs = _ef_tables(S, n_ang, dh.device)
            kw = dict(SP=_next_pow2(S), P16=_next_pow2(P), BLOCK_R=block_r)

            # Merged path: dh (the dominant read) loaded once → dm AND dD.
            if need_dm and need_dd and block_r >= n_base:
                dm_cos = torch.empty(E, n_ch, n_ang, dtype=dh.dtype,
                                     device=dh.device)
                dm_sin = torch.empty_like(dm_cos)
                dD = torch.zeros_like(D_block)
                _pu_bwd_merged_kernel[(E,)](
                    dh_c, m_cos.contiguous(), m_sin.contiguous(),
                    D_block.contiguous(),
                    *tabs, dm_cos, dm_sin, dD,
                    n_base, S, P, **kw)
                return dm_cos, dm_sin, dD, None, None, None, None

            eye = torch.arange(E, device=dh.device)
            dm_cos = dm_sin = None
            if need_dm:
                # d(m) = select(rotate(dh)): the n2e forward on the per-edge
                # dh, identity "gather".
                dm_cos, dm_sin = _ef_forward_triton(
                    dh_c, eye, None, D_block, n_ch, n_ang)
            dD = None
            if need_dd:
                dD = torch.zeros_like(D_block)
                _ef_bwd_dd_kernel[(E,)](
                    dh_c, eye, eye,
                    m_cos.contiguous(), m_sin.contiguous(),
                    *tabs, dD,
                    n_base, n_base, S, P, **kw)
            return dm_cos, dm_sin, dD, None, None, None, None

        # Eager (double-differentiable) path.
        dm_cos = dm_sin = None
        if need_dm:
            g_rot = torch.bmm(dh, D_block)
            g_flat = g_rot.reshape(E, -1)
            dm_cos = (g_flat.index_select(1, cos_flat_idx)
                      .view(E, n_ch, n_ang)) * cos_valid
            dm_sin = (g_flat.index_select(1, sin_flat_idx)
                      .view(E, n_ch, n_ang)) * sin_valid
        dD = None
        if need_dd:
            h = _pack_grads(m_cos, m_sin, cos_flat_idx, sin_flat_idx,
                            cos_valid, sin_valid,
                            n_base * S).view(E, n_base, S)
            dD = torch.bmm(dh.transpose(1, 2), h)
        return dm_cos, dm_sin, dD, None, None, None, None


def pack_unrotate_fused(m_cos, m_sin, D_block, l_max, m_max):
    """Fused pack → unrotate for the MP layers: h_global = pack(m) @ D_blockᵀ.
    Gating and scatter remain the caller's (node-frame, unchanged).

    Args:
        m_cos/m_sin: (E, n_base*(l_max+1), m_max+1) bond-frame message
        D_block:     (E, n_sph, n_sph)
    Returns:
        h_global: (E, n_base, n_sph)
    """
    n_base = m_cos.shape[1] // (l_max + 1)
    tabs = _ef_flat_tables(n_base, l_max, m_max, m_cos.device, m_cos.dtype)
    with record_function("pack_unrotate_fused"):
        return PackUnrotateFused.apply(m_cos, m_sin, D_block, *tabs)


def edge_to_node_fused(g_cos, g_sin, edge_dst, D_block, n_atoms, l_max, m_max):
    """Fused e2n for the MP layers' pack → unrotate → scatter (gate applied by
    the caller on (g_cos, g_sin) beforehand — it commutes with the rotation).

    Args:
        g_cos/g_sin: (E, n_base*(l_max+1), m_max+1) gated bond-frame features
        edge_dst:    (E,) receiver atom per edge
        D_block:     (E, n_sph, n_sph)
    Returns:
        Delta: (n_atoms, n_base, n_sph)
    """
    n_base = g_cos.shape[1] // (l_max + 1)
    tabs = _ef_flat_tables(n_base, l_max, m_max, g_cos.device, g_cos.dtype)
    with record_function("edge_to_node_fused"):
        return EdgeToNodeFused.apply(g_cos, g_sin, edge_dst, D_block,
                                     n_atoms, *tabs)

"""ecenet/realspace_kernel.py — fused real-space angular nonlinearity.

``RealSpaceNonlinearity`` lifts per-edge Fourier coefficients to values on a θ
grid, applies a pointwise nonlinearity, and projects back:

    f = A_cos·cos_synth + A_sin·sin_synth     # synthesize: (n_e,F,n_ang) → (n_e,F,n_grid)
    f = act(f)                                # pointwise on the grid
    out = f·cos_analysis , f·sin_analysis     # analyze: back to (n_e,F,n_ang)

(The release model has no pre-activation affine — dev's fixed-buffer identity
affine was dropped in da9d478, so the fused path here is pure σ(f).)

The grid tensor ``f`` is ~``n_grid/n_ang`` ≈ 3× the input and is materialized in
HBM (and saved for backward), which dominates the step's memory once the value
aggregation is fused. But every ``(edge, feature)`` row is independent and the
input/output are the small ``n_ang`` width — so the grid blow-up can be fused
away: read ``A_cos/A_sin`` once, keep ``f`` in registers, write the output once,
and recompute ``f`` in the backward (never store it).

Structure (mirrors dev's realspace_kernel.py):

1. ``realspace_reference`` — the common-path forward, the spec every fused path
   must match.
2. ``RealSpaceFused`` — a PyTorch ``autograd.Function`` with the analytic
   backward, recompute-in-backward (so ``f_grid`` is not saved). Already a memory
   win with no kernel; validates the math on CPU.
3. Triton forward/backward kernels (``_rs_*_kernel``), dispatched inside
   ``RealSpaceFused`` on CUDA (silu); the grid tensor never reaches HBM.

Scope: ``is_fusible`` gates on the common config; the release module is always
common today, but the guard keeps future dev ports (data-dependent / edge-type /
rms-norm / channel-mix variants) from silently taking the wrong path.
"""

import os

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:                       # CPU-only / no-triton env → PyTorch path
    _HAS_TRITON = False

# Launch configuration. Rows (edge·feature) per program and warps per program;
# overridable per GPU without a code change (read once at import):
#     ECENET_RS_BLOCK=256 ECENET_RS_WARPS=8 python ...
_RS_BLOCK = int(os.environ.get('ECENET_RS_BLOCK', 128))
_RS_WARPS = int(os.environ.get('ECENET_RS_WARPS', 4))


def is_fusible(nl):
    """True if a RealSpaceNonlinearity uses the common path the fused code covers:
    a single pointwise activation (no data-dependent / edge-type scale-shift, no
    channel-mixing MLP, no RMS norm — dev features, absent from this release)."""
    return not (getattr(nl, 'data_dependent', False)
                or getattr(nl, 'edge_type_nonlin', False)
                or getattr(nl, 'rms_norm', False)
                or getattr(nl, 'mix_channels', False))


# ---------------------------------------------------------------------------
# 1. Reference: common-path forward (the spec)
# ---------------------------------------------------------------------------


def realspace_reference(A_cos, A_sin, cos_synth, sin_synth,
                        cos_analysis, sin_analysis, activation):
    """Common-path RealSpaceNonlinearity forward. ``A_cos/A_sin`` are
    ``(n_e, F, n_ang)``; the synth/analysis tensors are ``(n_ang, n_grid)`` /
    ``(n_grid, n_ang)``; ``activation`` a pointwise callable. Returns
    ``(out_cos, out_sin)``, both ``(n_e, F, n_ang)``."""
    f = A_cos @ cos_synth + A_sin @ sin_synth        # (n_e, F, n_grid)
    f = activation(f)
    return f @ cos_analysis, f @ sin_analysis


# ---------------------------------------------------------------------------
# 2. Triton kernels (CUDA + silu) — one program per row-tile, f_grid in SRAM.
# ---------------------------------------------------------------------------
#
# Every (edge, feature) pair is an independent row of n_ang Fourier coeffs. Flatten
# to R = n_e·F rows; tile R over programs. Per row tile (BLOCK rows): synthesize to
# the grid (BLOCK, n_grid) IN REGISTERS, apply silu, analyze back to (BLOCK, n_ang),
# store. The fat grid tensor never touches HBM, and the backward recomputes it (so
# it's never saved) — that's the memory win.
#
# Memory access: the op is bandwidth-bound (a handful of flops per loaded float),
# so all global traffic goes through ONE (BLOCK, N_ANG) tile load/store per
# operand — a dense row-major region, fully coalesced. The earlier per-m column
# loads walked DRAM at stride n_ang (a third of each sector wasted) and issued
# N_ANG separate transactions per operand; that access pattern, not arithmetic,
# capped the kernel well below bandwidth. Columns are then pulled out of the
# register tile with a multiply-mask reduction — arithmetically free next to the
# loads it replaces, and it keeps the live set to 2D tiles (no (BLOCK, N_ANG,
# N_GRID) intermediate to spill).
#
# n_ang (3-4) and n_grid (9-13) are far below tl.dot's 16-min, so synthesis and
# analysis are written as explicit outer-product accumulations / axis reductions
# over the (compile-time) angular and grid dims — small 2D tiles only, no tl.dot,
# no 3D tiles. silu is hardcoded (the model default); other activations fall back
# to PyTorch.


if _HAS_TRITON:

    @triton.jit
    def _rs_fwd_kernel(acos_ptr, asin_ptr, cs_ptr, ss_ptr, ca_ptr, sa_ptr,
                       oc_ptr, os_ptr,
                       R, n_ang, n_grid,
                       sc_row, sc_col, ss_row, ss_col, so_row, so_col,
                       BLOCK: tl.constexpr, N_ANG: tl.constexpr, N_GRID: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # row = edge·feature
        mask = offs < R
        a_idx = tl.arange(0, N_ANG)
        offs_g = tl.arange(0, N_GRID)
        g_mask = offs_g < n_grid

        # One coalesced (BLOCK, N_ANG) tile load per operand. Each operand
        # carries its own (row, col) strides so both supported layouts — dense
        # row-major and einsum's m-major (rows unit-stride, see _rows) — read
        # straight from where the producer left them, no .contiguous() copies.
        t_mask = mask[:, None] & (a_idx < n_ang)[None, :]
        ac_all = tl.load(acos_ptr + offs[:, None] * sc_row + a_idx[None, :] * sc_col,
                         mask=t_mask, other=0.0)
        as_all = tl.load(asin_ptr + offs[:, None] * ss_row + a_idx[None, :] * ss_col,
                         mask=t_mask, other=0.0)

        # Synthesis: f(grid) = Σ_m A_cos[:,m]·cos_synth[m,:] + A_sin[:,m]·sin_synth[m,:]
        # (column m extracted from the register tile by multiply-mask reduction)
        f = tl.zeros((BLOCK, N_GRID), dtype=tl.float32)
        for m in tl.static_range(N_ANG):
            sel = (a_idx == m).to(tl.float32)
            ac = tl.sum(ac_all * sel[None, :], axis=1)
            as_ = tl.sum(as_all * sel[None, :], axis=1)
            cs = tl.load(cs_ptr + m * n_grid + offs_g, mask=g_mask & (m < n_ang), other=0.0)
            ss = tl.load(ss_ptr + m * n_grid + offs_g, mask=g_mask & (m < n_ang), other=0.0)
            f += ac[:, None] * cs[None, :] + as_[:, None] * ss[None, :]

        h = f * tl.sigmoid(f)                                   # silu

        # Analysis: out[:,a] = Σ_k h[:,k]·cos_analysis[k,a] — accumulated into
        # (BLOCK, N_ANG) register tiles, one coalesced store per operand.
        oc_all = tl.zeros((BLOCK, N_ANG), dtype=tl.float32)
        os_all = tl.zeros((BLOCK, N_ANG), dtype=tl.float32)
        for a in tl.static_range(N_ANG):
            col = g_mask & (a < n_ang)
            ca = tl.load(ca_ptr + offs_g * n_ang + a, mask=col, other=0.0)
            sa = tl.load(sa_ptr + offs_g * n_ang + a, mask=col, other=0.0)
            sel = (a_idx == a).to(tl.float32)
            oc_all += tl.sum(h * ca[None, :], axis=1)[:, None] * sel[None, :]
            os_all += tl.sum(h * sa[None, :], axis=1)[:, None] * sel[None, :]
        tile_o = offs[:, None] * so_row + a_idx[None, :] * so_col
        tl.store(oc_ptr + tile_o, oc_all, mask=t_mask)
        tl.store(os_ptr + tile_o, os_all, mask=t_mask)

    @triton.jit
    def _rs_bwd_kernel(acos_ptr, asin_ptr, cs_ptr, ss_ptr, ca_ptr, sa_ptr,
                       goc_ptr, gos_ptr, dac_ptr, das_ptr,
                       R, n_ang, n_grid,
                       sc_row, sc_col, ss_row, ss_col,
                       gc_row, gc_col, gs_row, gs_col, so_row, so_col,
                       BLOCK: tl.constexpr, N_ANG: tl.constexpr, N_GRID: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < R
        a_idx = tl.arange(0, N_ANG)
        offs_g = tl.arange(0, N_GRID)
        g_mask = offs_g < n_grid

        # One coalesced (BLOCK, N_ANG) tile load per operand, each with its own
        # strides (see forward — inputs and incoming grads have independent
        # producers, so their layouts are independent).
        t_mask = mask[:, None] & (a_idx < n_ang)[None, :]
        ac_all = tl.load(acos_ptr + offs[:, None] * sc_row + a_idx[None, :] * sc_col,
                         mask=t_mask, other=0.0)
        as_all = tl.load(asin_ptr + offs[:, None] * ss_row + a_idx[None, :] * ss_col,
                         mask=t_mask, other=0.0)
        goc_all = tl.load(goc_ptr + offs[:, None] * gc_row + a_idx[None, :] * gc_col,
                          mask=t_mask, other=0.0)
        gos_all = tl.load(gos_ptr + offs[:, None] * gs_row + a_idx[None, :] * gs_col,
                          mask=t_mask, other=0.0)

        # Recompute f (never stored in the forward).
        f = tl.zeros((BLOCK, N_GRID), dtype=tl.float32)
        for m in tl.static_range(N_ANG):
            sel = (a_idx == m).to(tl.float32)
            ac = tl.sum(ac_all * sel[None, :], axis=1)
            as_ = tl.sum(as_all * sel[None, :], axis=1)
            cs = tl.load(cs_ptr + m * n_grid + offs_g, mask=g_mask & (m < n_ang), other=0.0)
            ss = tl.load(ss_ptr + m * n_grid + offs_g, mask=g_mask & (m < n_ang), other=0.0)
            f += ac[:, None] * cs[None, :] + as_[:, None] * ss[None, :]
        sig = tl.sigmoid(f)
        silu_prime = sig * (1.0 + f * (1.0 - sig))              # d/df [f·σ(f)]

        # dh[:,k] = Σ_a g_out_cos[:,a]·cos_analysis[k,a] + g_out_sin[:,a]·sin_analysis[k,a]
        dh = tl.zeros((BLOCK, N_GRID), dtype=tl.float32)
        for a in tl.static_range(N_ANG):
            sel = (a_idx == a).to(tl.float32)
            goc = tl.sum(goc_all * sel[None, :], axis=1)
            gos = tl.sum(gos_all * sel[None, :], axis=1)
            ca = tl.load(ca_ptr + offs_g * n_ang + a, mask=g_mask & (a < n_ang), other=0.0)
            sa = tl.load(sa_ptr + offs_g * n_ang + a, mask=g_mask & (a < n_ang), other=0.0)
            dh += goc[:, None] * ca[None, :] + gos[:, None] * sa[None, :]

        df = dh * silu_prime                                    # through silu

        # dA_cos[:,m] = Σ_k df[:,k]·cos_synth[m,k] — accumulated into (BLOCK,
        # N_ANG) register tiles, one coalesced store per operand.
        dac_all = tl.zeros((BLOCK, N_ANG), dtype=tl.float32)
        das_all = tl.zeros((BLOCK, N_ANG), dtype=tl.float32)
        for m in tl.static_range(N_ANG):
            col = g_mask & (m < n_ang)
            cs = tl.load(cs_ptr + m * n_grid + offs_g, mask=col, other=0.0)
            ss = tl.load(ss_ptr + m * n_grid + offs_g, mask=col, other=0.0)
            sel = (a_idx == m).to(tl.float32)
            dac_all += tl.sum(df * cs[None, :], axis=1)[:, None] * sel[None, :]
            das_all += tl.sum(df * ss[None, :], axis=1)[:, None] * sel[None, :]
        tile_o = offs[:, None] * so_row + a_idx[None, :] * so_col
        tl.store(dac_ptr + tile_o, dac_all, mask=t_mask)
        tl.store(das_ptr + tile_o, das_all, mask=t_mask)


def _can_use_triton(A_cos, activation):
    """Triton present, CUDA input, and the hardcoded-silu kernel applies."""
    return _HAS_TRITON and A_cos.is_cuda and isinstance(activation, torch.nn.SiLU)


def _fp32(*ts):
    return [t.to(torch.float32).contiguous() for t in ts]


def _rows(t):
    """View a (n_e, F, n_ang) operand as R = n_e·F rows of n_ang coefficients,
    returning (tensor, s_row, s_col) for the kernel's stride arithmetic.

    The copy is skipped only for row-major layouts (rows collapse AND the
    coefficient axis is unit-stride) — the case the kernel loads as one
    vectorized coalesced tile. Reading EquivariantLinear's m-major einsum
    output (strides (F, 1, R)) in place was tried and MEASURED ~4-6x slower
    on an A100 than copying it contiguous first, despite its unit-stride
    rows — the tile load degrades to strided scalar accesses across three
    ~45 MB-apart streams. So the m-major layout takes the contiguous copy;
    revisit only with a transposed-tile load variant benchmarked in hand."""
    t = t.to(torch.float32)
    n_e, F, n_ang = t.shape
    if t.stride(2) != 1 or t.stride(0) != F * t.stride(1):
        t = t.contiguous()
    return t, t.stride(1), t.stride(2)


def _realspace_forward_triton(A_cos, A_sin, cos_synth, sin_synth,
                              cos_analysis, sin_analysis):
    n_e, F, n_ang = A_cos.shape
    n_grid = cos_synth.shape[1]
    R = n_e * F
    acos, sc_r, sc_c = _rows(A_cos)
    asin, ss_r, ss_c = _rows(A_sin)
    cs, ss, ca, sa = _fp32(cos_synth, sin_synth, cos_analysis, sin_analysis)
    oc = torch.empty(R, n_ang, device=A_cos.device, dtype=torch.float32)
    os = torch.empty_like(oc)
    N_ANG, N_GRID = triton.next_power_of_2(n_ang), triton.next_power_of_2(n_grid)
    grid = (triton.cdiv(R, _RS_BLOCK),)
    _rs_fwd_kernel[grid](acos, asin, cs, ss, ca, sa, oc, os,
                         R, n_ang, n_grid,
                         sc_r, sc_c, ss_r, ss_c, oc.stride(0), oc.stride(1),
                         BLOCK=_RS_BLOCK, N_ANG=N_ANG, N_GRID=N_GRID,
                         num_warps=_RS_WARPS)
    out = lambda t: t.reshape(n_e, F, n_ang).to(A_cos.dtype)  # noqa: E731
    return out(oc), out(os)


def _realspace_backward_triton(g_out_cos, g_out_sin, A_cos, A_sin,
                               cos_synth, sin_synth, cos_analysis, sin_analysis):
    n_e, F, n_ang = A_cos.shape
    n_grid = cos_synth.shape[1]
    R = n_e * F
    acos, sc_r, sc_c = _rows(A_cos)
    asin, ss_r, ss_c = _rows(A_sin)
    goc, gc_r, gc_c = _rows(g_out_cos)
    gos, gs_r, gs_c = _rows(g_out_sin)
    cs, ss, ca, sa = _fp32(cos_synth, sin_synth, cos_analysis, sin_analysis)
    dac = torch.empty(R, n_ang, device=A_cos.device, dtype=torch.float32)
    das = torch.empty_like(dac)
    N_ANG, N_GRID = triton.next_power_of_2(n_ang), triton.next_power_of_2(n_grid)
    grid = (triton.cdiv(R, _RS_BLOCK),)
    _rs_bwd_kernel[grid](acos, asin, cs, ss, ca, sa, goc, gos, dac, das,
                         R, n_ang, n_grid,
                         sc_r, sc_c, ss_r, ss_c,
                         gc_r, gc_c, gs_r, gs_c, dac.stride(0), dac.stride(1),
                         BLOCK=_RS_BLOCK, N_ANG=N_ANG, N_GRID=N_GRID,
                         num_warps=_RS_WARPS)
    out = lambda t: t.reshape(n_e, F, n_ang).to(A_cos.dtype)  # noqa: E731
    return out(dac), out(das)


class RealSpaceFused(torch.autograd.Function):
    """Common-path nonlinearity with an analytic backward that recomputes the
    grid tensor instead of saving it.

    Matches ``realspace_reference`` numerically; the backward is its exact
    gradient. ``f_grid`` (the ~3× transient) never enters the autograd graph, so
    the saved-for-backward footprint drops to the small ``(n_e, F, n_ang)`` inputs
    — a memory win even with no Triton (CPU / non-silu take the PyTorch
    recompute path below).
    """

    @staticmethod
    def forward(ctx, A_cos, A_sin, cos_synth, sin_synth,
                cos_analysis, sin_analysis, activation):
        ctx.use_triton = _can_use_triton(A_cos, activation)
        if ctx.use_triton:
            out_cos, out_sin = _realspace_forward_triton(
                A_cos, A_sin, cos_synth, sin_synth, cos_analysis, sin_analysis)
        else:
            with torch.no_grad():
                f = A_cos @ cos_synth + A_sin @ sin_synth
                f = activation(f)
                out_cos, out_sin = f @ cos_analysis, f @ sin_analysis
        ctx.save_for_backward(A_cos, A_sin, cos_synth, sin_synth,
                              cos_analysis, sin_analysis)
        ctx.activation = activation
        return out_cos, out_sin

    @staticmethod
    def backward(ctx, g_out_cos, g_out_sin):
        (A_cos, A_sin, cos_synth, sin_synth,
         cos_analysis, sin_analysis) = ctx.saved_tensors

        if ctx.use_triton:
            dA_cos, dA_sin = _realspace_backward_triton(
                g_out_cos, g_out_sin, A_cos, A_sin,
                cos_synth, sin_synth, cos_analysis, sin_analysis)
            return dA_cos, dA_sin, None, None, None, None, None

        # PyTorch path (CPU / non-silu): analytic backward, recompute the grid.
        # grad enabled through the activation only → act'(f) exact for any activation.
        f0 = A_cos @ cos_synth + A_sin @ sin_synth
        with torch.enable_grad():
            f = f0.detach().requires_grad_(True)
            h = ctx.activation(f)
        dh = (g_out_cos @ cos_analysis.transpose(-1, -2)
              + g_out_sin @ sin_analysis.transpose(-1, -2))  # (n_e, F, n_grid)
        df, = torch.autograd.grad(h, f, dh)                 # dh ⊙ act'(f)
        dA_cos = df @ cos_synth.transpose(-1, -2)           # (n_e, F, n_ang)
        dA_sin = df @ sin_synth.transpose(-1, -2)
        return dA_cos, dA_sin, None, None, None, None, None

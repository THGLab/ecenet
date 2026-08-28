"""Spherical harmonics and Wigner-D rotation machinery.

Real spherical harmonics come from sphericart (orthonormal, e3nn convention; see
``spherical_harmonics_float64`` / ``_get_sph_cache``). The rest constructs the
Wigner-D matrices that rotate spherical-harmonic features into a per-edge bond
frame: ``build_D1_from_rhat`` / ``recursive_wigner_D`` build ``D^l`` via real
Clebsch-Gordan coupling, and ``wigner_rotate`` applies the block-diagonal
rotation. Used by ace_basis.py / model.py to express ACE atomic features in each
edge's bond frame.
"""

import functools
import math

import torch


def _get_sph_cache(l_max):
    """Lazy-load sphericart and return a cached SphericalHarmonics object."""
    if l_max not in _SPH_CACHE:
        import sphericart.torch as _sct
        _SPH_CACHE[l_max] = _sct.SphericalHarmonics(l_max, backward_second_derivatives=True)
    return _SPH_CACHE[l_max]



# ──────────────────────────────────────────────────────────────────────
# Spherical harmonics via sphericart (z-polar convention)
#
# sphericart.compute() and compute_with_gradients() return SH and their
# Cartesian Jacobians as plain tensors (no PyTorch autograd graph), which
# eliminates expensive 2nd-order backward re-traversal of the SH graph.
# Normalization: orthonormal (Y_0^0 = 1/sqrt(4π)), matching e3nn convention.
# ──────────────────────────────────────────────────────────────────────

_SPH_CACHE: dict = {}


def spherical_harmonics_float64(l_max, vectors, normalize=True):
    """Real spherical harmonics, sphericart z-polar convention + orthonormal normalization."""
    if normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    if not vectors.is_contiguous():
        vectors = vectors.contiguous()
    return _get_sph_cache(l_max).compute(vectors)


@functools.lru_cache(maxsize=None)
def _change_basis_Q(l):
    """Real-to-complex spherical harmonic basis change in complex128.

    Matches e3nn's change_basis_real_to_complex exactly, but computed directly
    in complex128 for full float64 precision (e3nn defaults to complex64).
    """
    dim = 2 * l + 1
    q = torch.zeros(dim, dim, dtype=torch.complex128)
    invsqrt2 = 1.0 / math.sqrt(2)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = invsqrt2
        q[l + m, l - abs(m)] = -1j * invsqrt2
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1)**m * invsqrt2
        q[l + m, l - abs(m)] = 1j * (-1)**m * invsqrt2
    q = (-1j) ** l * q  # e3nn phase convention for real CG coefficients
    return q


# ──────────────────────────────────────────────────────────────────────
# Recursive Wigner D-matrix via Clebsch-Gordan coupling
# Computes D^l from D^1 (the 3x3 rotation matrix in real SH basis)
# using the identity  D^l = C^T @ kron(D^1, D^{l-1}) @ C
# where C are real CG coefficients for l1=1, l2=l-1 -> l3=l.
# ──────────────────────────────────────────────────────────────────────

def _wigner_3j_scalar(j1, j2, j3, m1, m2, m3):
    """Single Wigner 3j symbol via Racah formula."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    prefactor = (-1)**(j1 - j2 - m3) * math.sqrt(
        math.factorial(j1 + m1) * math.factorial(j1 - m1) *
        math.factorial(j2 + m2) * math.factorial(j2 - m2) *
        math.factorial(j3 + m3) * math.factorial(j3 - m3)
    ) * math.sqrt(
        math.factorial(j1 + j2 - j3) * math.factorial(j1 - j2 + j3) *
        math.factorial(-j1 + j2 + j3) /
        math.factorial(j1 + j2 + j3 + 1)
    )
    s_min = int(max(0, j2 - j3 - m1, j1 - j3 + m2))
    s_max = int(min(j1 + j2 - j3, j1 - m1, j2 + m2))
    total = 0.0
    for s in range(s_min, s_max + 1):
        total += (-1)**s / (
            math.factorial(s) *
            math.factorial(int(j1 + j2 - j3 - s)) *
            math.factorial(int(j1 - m1 - s)) *
            math.factorial(int(j2 + m2 - s)) *
            math.factorial(int(j3 - j2 + m1 + s)) *
            math.factorial(int(j3 - j1 - m2 + s))
        )
    return prefactor * total


@functools.lru_cache(maxsize=None)
def _real_cg(l1, l2, l3):
    """Real Clebsch-Gordan coefficients for coupling l1 x l2 -> l3.

    Returns C of shape (2*l1+1, 2*l2+1, 2*l3+1) such that:
      D^{l3}_{m3,m3'} = sum C[m1,m2,m3] * D^{l1}[m1,m1'] * D^{l2}[m2,m2'] * C[m1',m2',m3']
    """
    dim1, dim2, dim3 = 2*l1+1, 2*l2+1, 2*l3+1
    C_complex = torch.zeros(dim1, dim2, dim3, dtype=torch.complex128)
    for im1, m1 in enumerate(range(-l1, l1+1)):
        for im2, m2 in enumerate(range(-l2, l2+1)):
            m3 = m1 + m2
            if abs(m3) > l3:
                continue
            im3 = m3 + l3
            w3j = _wigner_3j_scalar(l1, l2, l3, m1, m2, -m3)
            C_complex[im1, im2, im3] = (-1)**(l1 - l2 + m3) * math.sqrt(2*l3 + 1) * w3j
    Q1 = _change_basis_Q(l1)
    Q2 = _change_basis_Q(l2)
    Q3 = _change_basis_Q(l3)
    C_real = torch.einsum('ai,bj,ck,abc->ijk', Q1.conj(), Q2.conj(), Q3, C_complex)
    assert C_real.imag.abs().max() < 1e-10
    return C_real.real.to(torch.float64)


def build_D1_from_rhat(r_hat):
    """Build l=1 Wigner D-matrix directly from unit bond vectors.

    Gram-Schmidt construction: given r̂, build an orthonormal frame
    {e_x, e_y, e_z=r̂} using a fixed reference vector, then express the
    resulting rotation in the real SH basis (m=-1,0,+1) = (y,z,x).

    Two charts avoid singularities:

      Chart A  (|rx| < 0.9, ref = x̂):  s_x = sqrt(ry²+rz²) ≥ 0.436
        e_x = [s_x, -rx·ry/s_x, -rx·rz/s_x]
        e_y = [0,    rz/s_x,    -ry/s_x   ]
        D^1 in (y,z,x) basis:
          [[ rz/s_x,   ry,  -rx·ry/s_x ],
           [ -ry/s_x,  rz,  -rx·rz/s_x ],
           [ 0,        rx,   s_x        ]]

      Chart B  (|rx| ≥ 0.9, ref = ŷ):  s_y = sqrt(rx²+rz²) ≥ 0.9
        e_x = [-rx·ry/s_y, s_y, -ry·rz/s_y]
        e_y = [-rz/s_y,    0,    rx/s_y   ]
        D^1 in (y,z,x) basis:
          [[ 0,        ry,   s_y        ],
           [ rx/s_y,   rz,  -ry·rz/s_y ],
           [ -rz/s_y,  rx,  -rx·ry/s_y ]]

    Both s_x and s_y are always ≥ 0.436 in their respective domains —
    no singularity, no large gradients anywhere on the unit sphere.

    Args:
        r_hat: (N, 3) unit vectors
    Returns:
        D1: (N, 3, 3) Wigner D-matrices in real SH basis (y, z, x)
    """
    rx = r_hat[:, 0]
    ry = r_hat[:, 1]
    rz = r_hat[:, 2]
    z  = torch.zeros_like(rx)

    # Chart A: ref = x̂, denominator s_x = sqrt(ry²+rz²).
    # Safe-sqrt (double-where): never differentiate sqrt at exactly 0, so the
    # backward is finite even for edges exactly on the x-axis (ry=rz=0, e.g. PBC
    # self-image edges in crystals). Forward output is unchanged — when s_x→0
    # this chart is the discarded (dead) branch of the torch.where below.
    u_x      = ry * ry + rz * rz
    big_x    = u_x > 1e-20
    s_x      = torch.where(big_x, torch.where(big_x, u_x, torch.ones_like(u_x)).sqrt(),
                           torch.full_like(u_x, 1e-10))
    inv_sx = 1.0 / s_x
    D1_A = torch.stack([
        rz * inv_sx,  ry,  -rx * ry * inv_sx,
       -ry * inv_sx,  rz,  -rx * rz * inv_sx,
        z,            rx,   s_x,
    ], dim=-1).reshape(-1, 3, 3)

    # Chart B: ref = ŷ, denominator s_y = sqrt(rx²+rz²). Safe-sqrt (see Chart A):
    # finite backward for edges exactly on the y-axis (rx=rz=0), where this is
    # the dead branch. Forward output unchanged.
    u_y      = rx * rx + rz * rz
    big_y    = u_y > 1e-20
    s_y      = torch.where(big_y, torch.where(big_y, u_y, torch.ones_like(u_y)).sqrt(),
                           torch.full_like(u_y, 1e-10))
    inv_sy = 1.0 / s_y
    D1_B = torch.stack([
        z,             ry,  s_y,
        rx * inv_sy,   rz,  -ry * rz * inv_sy,
       -rz * inv_sy,   rx,  -rx * ry * inv_sy,
    ], dim=-1).reshape(-1, 3, 3)

    # Use chart B when |rx| >= 0.9 (s_y >= 0.9, s_x could be small)
    mask = (rx.abs() >= 0.9)[:, None, None].expand_as(D1_A)
    return torch.where(mask, D1_B, D1_A)


# Precompute CG coefficients as plain tensors so recursive_wigner_D has no
# lru_cache lookups or complex-number code at runtime (~10% faster).
_CG_COEFFS = {l: _real_cg(1, l-1, l).clone() for l in range(2, 13)}


def recursive_wigner_D(r_hat, l_max):
    """Compute Wigner D-matrices for l=0..l_max via CG recursion from D^1.

    D^1 is built directly from r_hat (no Euler angles or trig functions).
    Higher-l matrices are computed recursively using:
      D^l = C^T @ kron(D^1, D^{l-1}) @ C

    Args:
        r_hat: (N, 3) unit bond direction vectors
        l_max: maximum angular momentum

    Returns:
        list of D-matrices: D[l] has shape (N, 2l+1, 2l+1)
    """
    N = r_hat.shape[0]
    device = r_hat.device
    dtype = r_hat.dtype
    D = [None] * (l_max + 1)
    D[0] = torch.ones(N, 1, 1, dtype=dtype, device=device)
    if l_max >= 1:
        D[1] = build_D1_from_rhat(r_hat)
    for l in range(2, l_max + 1):
        C = _CG_COEFFS[l].to(dtype=dtype, device=device)
        step1 = torch.einsum('abc,Nbd->Nacd', C, D[l-1])
        step2 = torch.einsum('Nab,Nacd->Nbcd', D[1], step1)
        D[l] = torch.einsum('Nbcd,bde->Nce', step2, C)
    return D


# ─────────────────────────────────────────────────────────────────────────────
# Wigner-D rotation of edge features (pure PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def build_D_block_from_list(D_list, N: int, l_max: int,
                            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Assemble the (N, n_sph, n_sph) block-diagonal Wigner-D matrix from a
    precomputed per-l list (one (N, 2l+1, 2l+1) block per l)."""
    n_sph = (l_max + 1) ** 2
    D_block = torch.zeros(N, n_sph, n_sph, dtype=dtype, device=device)
    for l, Dl in enumerate(D_list):
        s, e = l * l, (l + 1) * (l + 1)
        D_block[:, s:e, s:e] = Dl
    return D_block


def build_D_block(r_hat: torch.Tensor, l_max: int) -> torch.Tensor:
    """Block-diagonal Wigner-D matrix (N, n_sph, n_sph) for all edges. Build once
    per forward pass and reuse across rotation calls with the same r_hat."""
    return build_D_block_from_list(
        recursive_wigner_D(r_hat, l_max), r_hat.shape[0], l_max, r_hat.device, r_hat.dtype)


def wigner_rotate(A_flat: torch.Tensor, D_block: torch.Tensor) -> torch.Tensor:
    """Rotate edge features into the bond frame: A_rot = A_flat @ D_block.

    Args:
        A_flat:  (N, C, n_sph)      features, n_sph = (l_max+1)^2
        D_block: (N, n_sph, n_sph)  block-diagonal Wigner-D (see build_D_block)
    Returns:
        (N, C, n_sph) rotated features.
    """
    return torch.bmm(A_flat, D_block)


# ─────────────────────────────────────────────────────────────────────────────
# Truncated (m_max < l_max) Wigner-D: only the columns the model consumes
# ─────────────────────────────────────────────────────────────────────────────
#
# With the angular layout truncated at m_max, the layers only ever read the
# bond-frame components with |m| <= min(l, m_max) — and, in the reverse
# direction, the packed features are exactly zero outside those slots. So both
# the rotate (A @ D) and the unrotate (h @ D^T) consume only the corresponding
# COLUMNS of each D^l block: a rectangular (n_sph, n_kept) slice of the
# block-diagonal, n_kept = Σ_l (2·min(l, m_max)+1). The kept layout keeps the
# per-l blocks contiguous in the SH ordering (m = -mk..mk), so index maps
# mirror the full-layout ones with the block base l²+l replaced by
# kept_offset(l) + min(l, m_max).


def n_kept_columns(l_max: int, m_max: int) -> int:
    """Width of the kept-column layout: Σ_l (2·min(l, m_max)+1)."""
    return sum(2 * min(l, m_max) + 1 for l in range(l_max + 1))


def kept_offsets(l_max: int, m_max: int):
    """Start of each l's block in the kept layout (list of l_max+2 ints;
    the last entry is n_kept)."""
    off = [0]
    for l in range(l_max + 1):
        off.append(off[-1] + 2 * min(l, m_max) + 1)
    return off


def build_D_slice(r_hat: torch.Tensor, l_max: int, m_max: int) -> torch.Tensor:
    """Kept columns of the block-diagonal Wigner-D: (N, n_sph, n_kept).

    ``A @ build_D_slice(...)`` equals the kept columns of
    ``A @ build_D_block(...)`` exactly (same recursion, column slice) — pure
    differentiable torch, safe for double-backward force-loss training.
    """
    D_list = recursive_wigner_D(r_hat, l_max)
    N = r_hat.shape[0]
    n_sph = (l_max + 1) ** 2
    blocks = []
    for l, Dl in enumerate(D_list):
        mk = min(l, m_max)
        blk = torch.zeros(N, n_sph, 2 * mk + 1,
                          dtype=r_hat.dtype, device=r_hat.device)
        blk[:, l * l:(l + 1) * (l + 1), :] = Dl[:, :, l - mk:l + mk + 1]
        blocks.append(blk)
    return torch.cat(blocks, dim=2)


class _SphYGrad(torch.autograd.Function):
    """Y and its Cartesian gradient as *values*, with a first-order backward
    via sphericart's Hessians. once_differentiable: a second backward (force-
    loss training) raises rather than silently returning wrong gradients —
    this is the single-backward leg of the analytic D-slice below."""

    @staticmethod
    def forward(ctx, r_hat, l_max):
        sph = _get_sph_cache(l_max)
        Y, dY = sph.compute_with_gradients(r_hat.detach().contiguous())
        ctx.save_for_backward(r_hat)
        ctx.l_max = l_max
        return Y, dY

    @staticmethod
    def backward(ctx, gY, gdY):
        # Grad mode is enabled inside backward only under create_graph — i.e.
        # when someone is building a force graph to differentiate again
        # (force-loss training). once_differentiable would be SILENT here:
        # the analytic D mixes this Function with differentiable ops (the D1
        # block), so a double backward would quietly drop this branch's
        # second derivative instead of erroring. Raise loudly instead.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "analytic Wigner-D is single-backward only (its gradient "
                "runs through sphericart Hessians; no third derivatives). "
                "For force-loss training use the recursion path: "
                "model.set_analytic_wigner(False).")
        (r_hat,) = ctx.saved_tensors
        sph = _get_sph_cache(ctx.l_max)
        _, dY, H = sph.compute_with_hessians(r_hat.detach().contiguous())
        grad = torch.einsum('ns,ncs->nc', gY, dY)
        grad = grad + torch.einsum('nds,ncds->nc', gdY, H)
        return grad, None


def build_D_slice_analytic(r_hat: torch.Tensor, l_max: int,
                           m_max: int) -> torch.Tensor:
    """Analytic kept columns for m_max <= 1 — no CG recursion.

    Closed forms (discovered numerically against ``recursive_wigner_D`` and
    verified to ~1e-15 in tests/test_wigner_slice.py; the code's gauge is the
    two-chart frame of ``build_D1_from_rhat``, whose ±1 columns are exactly
    the tangent frame e_±):

        D^l[:, m=0]  = sqrt(4π/(2l+1)) · Y_l(r̂)
        D^l[:, m=±1] = sqrt(4π/(2l+1)) · sqrt(2/(l(l+1))) · (∇Y_l(r̂) · e_±)

    One sphericart call replaces the recursion. SINGLE-BACKWARD ONLY (the
    gradient path runs through sphericart Hessians): fine for MD / inference
    forces and the stress strain pass, but a force-loss training step (double
    backward) raises. Use ``build_D_slice`` for training.
    """
    if m_max > 1:
        raise ValueError(f"build_D_slice_analytic supports m_max <= 1 "
                         f"(closed forms for |m| <= 1), got m_max={m_max}")
    N = r_hat.shape[0]
    n_sph = (l_max + 1) ** 2
    Y, dY = _SphYGrad.apply(r_hat, l_max)
    if m_max >= 1 and l_max >= 1:
        D1 = build_D1_from_rhat(r_hat)          # differentiable; defines gauge
        # frame vectors in Cartesian from the (y, z, x) real-SH basis rows
        e_p = torch.stack([D1[:, 2, 2], D1[:, 0, 2], D1[:, 1, 2]], dim=1)
        e_m = torch.stack([D1[:, 2, 0], D1[:, 0, 0], D1[:, 1, 0]], dim=1)
        g_p = torch.einsum('nc,ncs->ns', e_p, dY)
        g_m = torch.einsum('nc,ncs->ns', e_m, dY)
    blocks = []
    for l in range(l_max + 1):
        s, e = l * l, (l + 1) * (l + 1)
        a = math.sqrt(4.0 * math.pi / (2 * l + 1))
        col0 = a * Y[:, s:e]
        if l == 0 or m_max == 0:
            cols = col0.unsqueeze(2)
        elif l == 1:
            cols = D1                            # exact, differentiable, cheap
        else:
            b = a * math.sqrt(2.0 / (l * (l + 1)))
            cols = torch.stack([b * g_m[:, s:e], col0, b * g_p[:, s:e]], dim=2)
        blk = torch.zeros(N, n_sph, cols.shape[2],
                          dtype=r_hat.dtype, device=r_hat.device)
        blk[:, s:e, :] = cols
        blocks.append(blk)
    return torch.cat(blocks, dim=2)

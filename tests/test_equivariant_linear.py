"""Tests for the block-diagonal-GEMM EquivariantLinear (``ecenet/equivariant.py``).

The layer computes y[e,o,m] = Σ_i x[e,i,m]·W[m,o,i] (bias on the m=0 cos slot
only) as one dense row-major GEMM against a per-m block-diagonal weight. These
tests pin it to the original einsum formulation:

  1. forward equivalence vs the einsum reference (several shapes, fp64);
  2. gradient equivalence — input, weight, and bias grads vs the reference;
  3. non-contiguous (m-major, einsum-layout) inputs give the same result;
  4. batched leading dims (B, n_e, F, n_ang) round-trip unchanged.

Pure PyTorch on CPU (fp64). Run:  python tests/test_equivariant_linear.py
"""

import os
import sys  # noqa: E402 — repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from ecenet.equivariant import EquivariantLinear

torch.manual_seed(0)
DTYPE = torch.float64


def reference(A_cos, A_sin, weights, bias):
    """The original einsum formulation — the spec."""
    oc = torch.einsum('...id,doi->...od', A_cos, weights)
    os_ = torch.einsum('...id,doi->...od', A_sin, weights)
    oc = oc.clone()
    oc[..., 0] = oc[..., 0] + bias
    return oc, os_


def make_layer(Fi=12, Fo=9, m_max=2, seed=0):
    torch.manual_seed(seed)
    lin = EquivariantLinear(Fi, Fo, m_max + 1, m_max).to(DTYPE)
    with torch.no_grad():                # non-trivial bias
        lin.bias.add_(torch.randn_like(lin.bias))
    return lin


def make_inputs(n_e=40, Fi=12, m_max=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: torch.randn(n_e, Fi, m_max + 1, generator=g, dtype=DTYPE)  # noqa: E731
    return mk(), mk()


def test_forward_equivalence():
    worst = 0.0
    for Fi, Fo, m_max in ((12, 9, 2), (7, 7, 3), (1, 5, 0), (16, 3, 1)):
        lin = make_layer(Fi, Fo, m_max)
        A_cos, A_sin = make_inputs(Fi=Fi, m_max=m_max)
        oc, os_ = lin(A_cos, A_sin)
        rc, rs = reference(A_cos, A_sin, lin.weights, lin.bias)
        e = max((oc - rc).abs().max(), (os_ - rs).abs().max()).item()
        worst = max(worst, e)
        assert e < 1e-12, f"forward Fi={Fi},Fo={Fo},m={m_max}: {e:.2e}"
    print(f"  forward equivalence vs einsum reference (worst {worst:.1e})")


def test_gradient_equivalence():
    lin = make_layer()
    A_cos, A_sin = make_inputs(seed=2)
    goc = torch.randn_like(A_cos[:, :9])   # (n_e, Fo, n_ang)
    gos = torch.randn_like(A_cos[:, :9])

    def grads(fn):
        a = A_cos.detach().clone().requires_grad_(True)
        b = A_sin.detach().clone().requires_grad_(True)
        lin.zero_grad()
        oc, os_ = fn(a, b)
        (oc * goc + os_ * gos).sum().backward()
        return a.grad, b.grad, lin.weights.grad.clone(), lin.bias.grad.clone()

    new = grads(lambda a, b: lin(a, b))
    ref = grads(lambda a, b: reference(a, b, lin.weights, lin.bias))
    worst = 0.0
    for name, n, r in zip(("dA_cos", "dA_sin", "dW", "db"), new, ref):
        e = (n - r).abs().max().item()
        worst = max(worst, e)
        assert e < 1e-12, f"{name} mismatch: {e:.2e}"
    print(f"  gradient equivalence: input/weight/bias grads match (worst {worst:.1e})")


def test_noncontiguous_input():
    """m-major (einsum-layout) inputs — strides (F, 1, n_e·F) — must give the
    same result as contiguous ones (reshape copies as needed internally)."""
    lin = make_layer()
    A_cos, A_sin = make_inputs(seed=3)

    def as_m_major(t):
        n_e, F, na = t.shape
        out = torch.empty(na, n_e, F, dtype=t.dtype).permute(1, 2, 0)
        out.copy_(t)
        assert not out.is_contiguous()
        return out

    oc, os_ = lin(A_cos, A_sin)
    mc, ms = lin(as_m_major(A_cos), as_m_major(A_sin))
    e = max((oc - mc).abs().max(), (os_ - ms).abs().max()).item()
    assert e < 1e-15, f"m-major input mismatch: {e:.2e}"
    print(f"  non-contiguous (m-major) input matches contiguous ({e:.1e})")


def test_batched_leading_dims():
    lin = make_layer()
    A_cos, A_sin = make_inputs(n_e=24, seed=4)
    B_cos = A_cos.reshape(4, 6, 12, 3)
    B_sin = A_sin.reshape(4, 6, 12, 3)
    oc, os_ = lin(A_cos, A_sin)
    bc, bs = lin(B_cos, B_sin)
    assert bc.shape == (4, 6, 9, 3) and bs.shape == (4, 6, 9, 3)
    e = max((bc.reshape(24, 9, 3) - oc).abs().max(),
            (bs.reshape(24, 9, 3) - os_).abs().max()).item()
    assert e < 1e-15, f"batched mismatch: {e:.2e}"
    print(f"  batched leading dims match flat ({e:.1e})")


if __name__ == "__main__":
    print("EquivariantLinear block-diagonal-GEMM tests")
    test_forward_equivalence()
    test_gradient_equivalence()
    test_noncontiguous_input()
    test_batched_leading_dims()
    print("All tests passed.")

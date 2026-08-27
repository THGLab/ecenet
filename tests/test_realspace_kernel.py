"""Tests for the fused real-space nonlinearity (``ecenet/realspace_kernel.py``).

Validates the recompute-in-backward ``RealSpaceFused`` autograd.Function against
the actual ``RealSpaceNonlinearity`` module:

  1. forward equivalence — fused vs module vs reference;
  2. backward equivalence — input grads (dA_cos/dA_sin) vs autograd through the
     module;
  3. finite-difference — analytic input grads vs numeric (the conservative check);
  4. fusibility gate — non-common configs report not-fusible;
  5. DFT bases are float64-accurate at model dtype (the _build_bases guarantee);
  6. Triton path vs the fp64 reference (skips without CUDA);
  7. whole-model set_activation_fused: matches reference, SO(3), conservative.

Pure PyTorch on CPU (fp64) except test 6. Run:  python tests/test_realspace_kernel.py
"""

import os
import sys  # noqa: E402 — repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from ecenet.equivariant import RealSpaceNonlinearity
from ecenet.realspace_kernel import (
    RealSpaceFused,
    is_fusible,
    realspace_reference,
)

torch.manual_seed(0)
DTYPE = torch.float64


def make_nl(F=12, m_max=2, activation='silu'):
    return RealSpaceNonlinearity(F, m_max, activation=activation).to(DTYPE)


def make_inputs(n_e=50, F=12, m_max=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: torch.randn(n_e, F, m_max + 1, generator=g, dtype=DTYPE)  # noqa: E731
    return mk(), mk()


def test_forward_equivalence():
    nl = make_nl()
    A_cos, A_sin = make_inputs()
    oc_ref, os_ref = nl(A_cos, A_sin)
    nl.fused = True
    oc_fus, os_fus = nl(A_cos, A_sin)
    nl.fused = False
    # …and the standalone reference fn matches too (same spec).
    oc_r, os_r = realspace_reference(A_cos, A_sin, nl.cos_synth, nl.sin_synth,
                                     nl.cos_analysis, nl.sin_analysis,
                                     nl.activation)
    e1 = max((oc_ref - oc_fus).abs().max(), (os_ref - os_fus).abs().max()).item()
    e2 = max((oc_ref - oc_r).abs().max(), (os_ref - os_r).abs().max()).item()
    assert e1 < 1e-11 and e2 < 1e-11, f"forward mismatch: {e1:.2e}, {e2:.2e}"
    print(f"  forward equivalence: fused {e1:.1e}, reference {e2:.1e}")


def test_backward_equivalence():
    nl = make_nl()
    A_cos, A_sin = make_inputs(seed=1)
    goc = torch.randn_like(A_cos)
    gos = torch.randn_like(A_sin)

    def grads(fn):
        a = A_cos.detach().clone().requires_grad_(True)
        b = A_sin.detach().clone().requires_grad_(True)
        oc, os = fn(a, b)
        (oc * goc + os * gos).sum().backward()
        return a.grad, b.grad

    ref = grads(lambda a, b: nl(a, b))
    nl.fused = True
    fus = grads(lambda a, b: nl(a, b))
    nl.fused = False
    worst = 0.0
    for name, r, f in zip(("dA_cos", "dA_sin"), ref, fus):
        e = (r - f).abs().max().item()
        worst = max(worst, e)
        assert e < 1e-10, f"{name} mismatch: {e:.2e}"
    print(f"  backward equivalence: input grads match (worst {worst:.1e})")


def test_finite_difference():
    """Analytic input grads from the fused backward vs central differences."""
    nl = make_nl(F=6, m_max=2)
    A_cos, A_sin = make_inputs(n_e=8, F=6, seed=2)
    goc = torch.randn_like(A_cos)
    gos = torch.randn_like(A_sin)

    def scalar(a, b):
        oc, os = RealSpaceFused.apply(a, b, nl.cos_synth, nl.sin_synth,
                                      nl.cos_analysis, nl.sin_analysis,
                                      nl.activation)
        return (oc * goc + os * gos).sum()

    a = A_cos.clone().requires_grad_(True)
    b = A_sin.clone().requires_grad_(True)
    scalar(a, b).backward()
    analytic = (a.grad, b.grad)

    eps = 1e-6
    worst = 0.0
    for idx, base in enumerate((A_cos, A_sin)):
        num = torch.zeros_like(base)
        flat = base.reshape(-1)
        for j in range(flat.numel()):
            pert = [A_cos.clone(), A_sin.clone()]
            pert[idx].reshape(-1)[j] += eps
            lp = scalar(*pert).item()
            pert[idx].reshape(-1)[j] -= 2 * eps
            lm = scalar(*pert).item()
            num.reshape(-1)[j] = (lp - lm) / (2 * eps)
        e = (num - analytic[idx]).abs().max().item()
        worst = max(worst, e)
        assert e < 1e-6, f"finite-diff mismatch on input {idx}: {e:.2e}"
    print(f"  finite-difference: analytic grad matches numeric (worst {worst:.1e})")


def test_fusibility_gate():
    nl = make_nl()
    assert is_fusible(nl)                              # common (release) path
    # The release module has no non-common variants; the gate exists so future
    # dev ports (data-dependent / edge-type / rms-norm / mix-channels) fall
    # back rather than silently fusing. Simulate one:
    nl.rms_norm = True
    assert not is_fusible(nl), "gate must reject rms_norm"
    del nl.rms_norm
    A_cos, A_sin = make_inputs(seed=3)
    nl.mix_channels = True
    ref = nl(A_cos, A_sin)                             # module forward, unaffected
    nl.fused = True
    fb = nl(A_cos, A_sin)                              # gate → reference path
    assert torch.equal(ref[0], fb[0]) and torch.equal(ref[1], fb[1])
    print("  fusibility gate: common path fuses; variants report not-fusible")


def test_dft_bases_are_full_precision():
    """The DFT synthesis/analysis pair must round-trip a bandlimited signal
    accurately *at the model's dtype* (see RealSpaceNonlinearity._build_bases:
    computed in float64 and cast, rebuilt on dtype/device change and after
    load_state_dict)."""
    nl = RealSpaceNonlinearity(6, m_max=3).to(DTYPE)   # .to() must not degrade them
    A_cos, A_sin = make_inputs(F=6, m_max=3)
    # sin(0·θ) ≡ 0 — the m=0 sin slot is a dead direction analysis always returns 0.
    A_sin[..., 0] = 0.0

    f = A_cos @ nl.cos_synth + A_sin @ nl.sin_synth     # synthesize onto the grid
    back_cos = f @ nl.cos_analysis                      # …and project straight back
    back_sin = f @ nl.sin_analysis
    err = max((back_cos - A_cos).abs().max(), (back_sin - A_sin).abs().max()).item()
    assert err < 1e-14, f"DFT round-trip is not exact at float64: {err:.2e}"
    print(f"  DFT bases round-trip to {err:.1e} in float64")


def _triton_available():
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def test_triton_vs_reference():
    """Triton kernel path (CUDA, silu) vs the fp64 module reference. Requires GPU."""
    if not _triton_available():
        print("  triton/CUDA unavailable — skipped (run on a GPU box)")
        return
    dev = "cuda"
    for F, m_max in ((48, 2), (32, 3)):           # n_grid = 9, 13
        nl = make_nl(F=F, m_max=m_max).to(dev)
        A_cos, A_sin = make_inputs(n_e=200, F=F, m_max=m_max, seed=30)
        A_cos, A_sin = A_cos.to(dev), A_sin.to(dev)
        goc, gos = torch.randn_like(A_cos), torch.randn_like(A_sin)

        def run(fn):
            a = A_cos.clone().requires_grad_(True)
            b = A_sin.clone().requires_grad_(True)
            oc, os = fn(a, b)
            (oc * goc + os * gos).sum().backward()
            return oc, os, a.grad, b.grad

        ref = run(lambda a, b: nl(a, b))                          # module, fp64
        ker = run(lambda a, b: RealSpaceFused.apply(
            a, b, nl.cos_synth, nl.sin_synth, nl.cos_analysis, nl.sin_analysis,
            nl.activation))                                       # → Triton (fp32)

        f_err = max((ref[0] - ker[0]).abs().max(), (ref[1] - ker[1]).abs().max()).item()
        b_err = max((ref[2] - ker[2]).abs().max(), (ref[3] - ker[3]).abs().max()).item()
        assert f_err < 1e-4, f"forward F={F},m={m_max}: {f_err:.2e}"
        assert b_err < 1e-3, f"input grad F={F},m={m_max}: {b_err:.2e}"
        print(f"  triton vs fp64 ref (F={F}, m_max={m_max}): "
              f"fwd {f_err:.1e}, dA {b_err:.1e}")


def _structure(n=7, n_types=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 3, generator=g, dtype=DTYPE) * 1.8,
            torch.randint(0, n_types, (n,), generator=g))


def _rotation(seed=1):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=DTYPE))
    Q = Q * torch.sign(torch.diag(R))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _model():
    from ecenet import ECENet
    torch.manual_seed(0)
    m = ECENet(n_types=4, r_cut_edge=5.0, r_cut_neighbor=4.0, l_max=2, n_max=3,
               embed_dim=8, n_layers=2, n_max_d=4, n_mp=2, mp_n_heads=2).double()
    with torch.no_grad():               # activate the zero-init residual paths
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def test_model_fused_matches_reference():
    """Whole-model energy + forces with set_activation_fused match the reference
    path, and SO(3) invariance holds (CPU → the PyTorch recompute path)."""
    pos, types = _structure(seed=4)
    m = _model()

    def ef(model):
        x = pos.clone().requires_grad_(True)
        e = model(x, types)
        return e, -torch.autograd.grad(e, x)[0]

    e_ref, f_ref = ef(m)
    m.set_activation_fused(True)
    e_fus, f_fus = ef(m)
    de = (e_ref - e_fus).abs().item()
    df = (f_ref - f_fus).abs().max().item()
    assert de < 1e-9 and df < 1e-9, f"dE={de:.2e}, dF={df:.2e}"
    inv = (m(pos, types) - m(pos @ _rotation().T, types)).abs().item()
    assert inv < 1e-8, f"fused breaks SO(3): {inv:.2e}"
    print(f"  model fused matches reference: dE {de:.1e}, dF {df:.1e}, SO(3) {inv:.1e}")


def test_model_fused_conservative():
    """Finite-difference -dE/dx vs fused-path autograd forces (whole model)."""
    pos, types = _structure(seed=5)
    m = _model().set_activation_fused(True)
    p = pos.clone().requires_grad_(True)
    f = -torch.autograd.grad(m(p, types), p)[0]
    eps = 1e-4
    fd = torch.zeros_like(pos)
    for a in range(pos.shape[0]):
        for c in range(3):
            dp, dm = pos.clone(), pos.clone()
            dp[a, c] += eps
            dm[a, c] -= eps
            fd[a, c] = -(m(dp, types).item() - m(dm, types).item()) / (2 * eps)
    err = (f - fd).abs().max().item()
    assert err < 1e-6, f"fused forces not conservative: {err:.2e}"
    print(f"  model fused forces match finite-difference (max err {err:.1e})")


if __name__ == "__main__":
    print("Fused real-space nonlinearity tests")
    test_forward_equivalence()
    test_backward_equivalence()
    test_finite_difference()
    test_fusibility_gate()
    test_dft_bases_are_full_precision()
    test_triton_vs_reference()
    test_model_fused_matches_reference()
    test_model_fused_conservative()
    print("All tests passed.")

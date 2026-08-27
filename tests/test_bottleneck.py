"""Tests for the low-rank bottleneck in the equivariant layers
(``ECENet(bottleneck_dim=r)``): down-project → nonlinearity(r) → up-project,
added via residual with a zero-init up-projection (identity at init).

Checks: layer identity at init, correct down/up shapes + nonlin width, SO(3)
invariance with an active bottleneck, and finite forces.

Run:  python tests/test_bottleneck.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from ecenet import ECENet
from ecenet.model import ECENetLayer

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


def test_layer_identity_at_init():
    """Zero-init up-projection → the bottleneck block adds 0 → layer is identity."""
    layer = ECENetLayer(n_features=16, m_max=3, bottleneck_dim=4).double()
    ac = torch.randn(5, 16, 4, dtype=DTYPE)
    as_ = torch.randn(5, 16, 4, dtype=DTYPE)
    as_[:, :, 0] = 0.0
    oc, os_ = layer(ac, as_)
    assert (oc - ac).abs().max() == 0.0 and (os_ - as_).abs().max() == 0.0, "not identity at init"
    # shapes: down C→r, up r→C, nonlinearity at the bottleneck width r
    assert layer.linear_down.in_features == 16 and layer.linear_down.out_features == 4
    assert layer.linear_up.in_features == 4 and layer.linear_up.out_features == 16
    assert layer.nonlin.n_features == 4
    print("  bottleneck layer: identity at init, down 16→4, up 4→16, nonlin at r=4")


def test_no_bottleneck_is_full_width():
    """bottleneck_dim=None → the plain full-width linear (no down/up)."""
    layer = ECENetLayer(n_features=16, m_max=3).double()
    assert layer.bottleneck_dim is None
    assert hasattr(layer, 'linear') and not hasattr(layer, 'linear_down')
    print("  bottleneck_dim=None → full-width linear (no down/up)")


def test_so3_invariance_active():
    """With an active (non-zero) up-projection the model stays SO(3)-invariant."""
    pos, types = random_structure(seed=3)
    m = ECENet(**COMMON, bottleneck_dim=6).double()
    for lyr in m.layers:
        with torch.no_grad():
            lyr.linear_up.weights.normal_(std=0.2)
    e0 = m(pos, types)
    err = (e0 - m(pos @ rand_rotation().T, types)).abs().item()
    assert err < 1e-9, f"bottleneck breaks SO(3): {err:.2e}"
    # the active bottleneck actually changes the output vs the identity init
    base = ECENet(**COMMON, bottleneck_dim=6).double()
    base.load_state_dict(m.state_dict())
    for lyr in base.layers:
        with torch.no_grad():
            lyr.linear_up.weights.zero_()
    assert (e0 - base(pos, types)).abs().item() > 1e-6, "active bottleneck had no effect"
    print(f"  active bottleneck: SO(3) err {err:.1e}, has effect")


def test_forces_finite():
    pos, types = random_structure(seed=4)
    m = ECENet(**COMMON, bottleneck_dim=6, n_mp=2).double()
    for lyr in [x for stage in m.layers for x in stage]:
        with torch.no_grad():
            lyr.linear_up.weights.normal_(std=0.2)
    p = pos.clone().requires_grad_(True)
    e = m(p, types)
    f = -torch.autograd.grad(e, p, create_graph=True)[0]
    assert torch.isfinite(e) and f.shape == pos.shape and torch.isfinite(f).all()
    print(f"  forces finite with active bottleneck + MP: |F|max={f.abs().max():.3f}")


if __name__ == "__main__":
    print("Bottleneck (low-rank equivariant layer) tests")
    test_layer_identity_at_init()
    test_no_bottleneck_is_full_width()
    test_so3_invariance_active()
    test_forces_finite()
    print("All tests passed.")

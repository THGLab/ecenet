"""Integration test: ECENet constructs, runs energy/forces, is SO(3)-invariant,
and the training path forward_batch_multi works (with and without message passing).

Run:  python test_ecenet.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from ecenet import ECENet

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


def rand_rotation(seed=1): #Cheap way to do a random rotation, scipy.spatial.transform.Rotation.random()
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=DTYPE)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R)) # fix QR sign ambiguity -> Haar-uniform on O(3).
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _energy_and_forces(model, pos, types):
    p = pos.clone().requires_grad_(True)
    e = model(p, types)
    f = -torch.autograd.grad(e, p, create_graph=True)[0]
    return e, f


def test_constructs_and_runs():
    pos, types = random_structure()
    model = ECENet(**COMMON).double()
    e, f = _energy_and_forces(model, pos, types)
    assert e.dim() == 0 and torch.isfinite(e), "energy not a finite scalar"
    assert f.shape == pos.shape and torch.isfinite(f).all(), "bad forces"
    print(f"  ECENet runs: E={e.item():.4f}, |F|max={f.abs().max():.3f}")


def test_so3_invariance():
    pos, types = random_structure(seed=2)
    model = ECENet(**COMMON).double()
    Q = rand_rotation()
    e1 = model(pos, types)
    e2 = model(pos @ Q.T, types)
    err = (e1 - e2).abs().item()
    assert err < 1e-9, f"energy not SO(3)-invariant: {err:.2e}"
    print(f"  SO(3) invariance: |E(Rx) - E(x)| = {err:.1e}")


def test_forces_finite_difference():
    pos, types = random_structure(seed=3)
    model = ECENet(**COMMON).double()
    _, f = _energy_and_forces(model, pos, types)
    eps = 1e-5
    fd = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            p = pos.clone(); p[i, d] += eps
            ep = model(p, types)
            p = pos.clone(); p[i, d] -= eps
            em = model(p, types)
            fd[i, d] = -(ep - em) / (2 * eps)
    err = (f - fd).abs().max().item()
    assert err < 1e-5, f"analytic vs FD forces mismatch {err:.2e}"
    print(f"  forces match finite-difference (max err {err:.1e})")


def test_training_path_forward_batch_multi():
    model = ECENet(**COMMON).double()
    structs = [random_structure(n=5 + b, seed=10 + b) for b in range(3)]
    pos_list = [s[0].clone().requires_grad_(True) for s in structs]
    typ_list = [s[1] for s in structs]
    energies = model.forward_batch_multi(pos_list, typ_list)
    assert energies.shape == (3,) and torch.isfinite(energies).all()
    grads = torch.autograd.grad(energies.sum(), pos_list, create_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)
    print(f"  forward_batch_multi (training path): energies {energies.detach().numpy().round(3)}")


def test_ecenet_mp():
    """ECENet with message passing (n_mp=2): SO(3)-invariant through the MP
    unrotate/rotate (via D_block), energy/forces finite."""
    pos, types = random_structure(seed=6)
    model = ECENet(**COMMON, n_mp=2).double()
    e, f = _energy_and_forces(model, pos, types)
    assert torch.isfinite(e) and f.shape == pos.shape and torch.isfinite(f).all()
    Q = rand_rotation(seed=4)
    err = (model(pos, types) - model(pos @ Q.T, types)).abs().item()
    assert err < 1e-9, f"ECENet(n_mp=2) not SO(3)-invariant: {err:.2e}"
    print(f"  ECENet(n_mp=2) runs: E={e.item():.4f}, SO(3) err {err:.1e}")


def test_forward_batch_topology_list_formats():
    """forward_batch's variable-topology fallback must consume BOTH
    per-structure list formats — build_topology tuples AND
    scripts/train_ecenet-style dicts — bit-identically to the on-the-fly
    path. Regression: forwarding a dict list used to unpack the dicts as
    tuples (i.e. their KEYS) and crash with a str + int TypeError."""
    model = ECENet(**COMMON).double()
    g = torch.Generator().manual_seed(3)
    types = torch.randint(0, N_TYPES, (6,), generator=g)
    pos_list = [torch.randn(6, 3, generator=g, dtype=DTYPE) * 1.8
                for _ in range(3)]     # same composition, different geometries

    tuples = model.build_topology(pos_list)
    dicts = [{'edge_i': t[0], 'edge_j': t[1], 'nb_src': t[2], 'nb_dst': t[3]}
             for t in tuples]

    e_fly = model.forward_batch(pos_list, types)
    e_tup = model.forward_batch(pos_list, types, topology=tuples)
    e_dic = model.forward_batch(pos_list, types, topology=dicts)
    d = max((e_fly - e_tup).abs().max().item(),
            (e_fly - e_dic).abs().max().item())
    assert d == 0.0, f"topology-list formats diverge from on-the-fly: {d:.3e}"
    print(f"  forward_batch consumes tuple AND dict topology lists "
          f"(d={d:.1e} vs on-the-fly)")


def test_zero_edge_energy_is_atomic_and_consistent():
    """A structure with no edges keeps Σ atomic_energy: forward and
    forward_batch_multi must agree, and the energy must be continuous across
    the last edge leaving r_cut (no jump by the per-element constants)."""
    model = ECENet(**COMMON).double()
    with torch.no_grad():
        model.atomic_energy.normal_(std=0.5)   # trained models are nonzero here

    # lone atom + a 2-atom structure with every pair beyond r_cut_edge
    for pos, types in (
        (torch.zeros(1, 3, dtype=DTYPE), torch.tensor([2])),
        (torch.tensor([[0.0, 0, 0], [50.0, 0, 0]], dtype=DTYPE),
         torch.tensor([1, 3])),
    ):
        e_single = model(pos, types)
        e_batch = model.forward_batch_multi([pos], [types])[0]
        e_ref = model.atomic_energy[types].sum()
        assert torch.allclose(e_single, e_ref), \
            f"forward zero-edge energy {e_single.item():.6f} != Σ atomic_energy {e_ref.item():.6f}"
        assert torch.allclose(e_single, e_batch), \
            f"forward {e_single.item():.6f} != forward_batch_multi {e_batch.item():.6f}"

    # continuity: dimer just inside vs just outside r_cut — the envelope takes
    # the edge energy to 0, so the difference must be tiny, not Σ atomic_energy
    r = model.r_cut_edge
    types = torch.tensor([1, 3])
    e_in = model(torch.tensor([[0.0, 0, 0], [r - 1e-6, 0, 0]], dtype=DTYPE), types)
    e_out = model(torch.tensor([[0.0, 0, 0], [r + 1e-6, 0, 0]], dtype=DTYPE), types)
    gap = (e_in - e_out).abs().item()
    assert gap < 1e-8, f"energy jump {gap:.3e} across r_cut_edge"
    print(f"  zero-edge energy = Σ atomic_energy, forward == batch, "
          f"continuous at r_cut (gap={gap:.1e})")


def test_identity_activation():
    """activation='identity': every RealSpaceNonlinearity — the layer stack AND
    the MP trunk/receiver — becomes an exact no-op (grid round-trip is exact for
    bandlimited features), the read-out MLPs keep SiLU, SO(3) invariance holds,
    and an unknown activation string is rejected loudly."""
    import torch.nn as nn

    from ecenet.equivariant import RealSpaceNonlinearity

    model = ECENet(**COMMON, n_mp=2, activation='identity').double()

    # every equivariant nonlinearity got Identity — layers and MP alike
    nls = [m for m in model.modules() if isinstance(m, RealSpaceNonlinearity)]
    assert len(nls) >= 3, "expected nonlinearities in layers and MP"
    assert all(isinstance(m.activation, nn.Identity) for m in nls)
    # ...and each is an exact no-op
    nl = nls[0]
    g = torch.Generator().manual_seed(3)
    a_cos = torch.randn(5, nl.n_features, nl.n_angular, generator=g, dtype=DTYPE)
    a_sin = torch.randn(5, nl.n_features, nl.n_angular, generator=g, dtype=DTYPE)
    a_sin[:, :, 0] = 0.0                    # m=0 sin slot is a structural zero
    o_cos, o_sin = nl(a_cos, a_sin)
    rt = max((o_cos - a_cos).abs().max(), (o_sin - a_sin).abs().max())
    assert rt < 1e-12, f"identity nonlinearity is not a no-op: {rt:.3e}"

    # read-out MLPs stay nonlinear (deliberate: identity would collapse them)
    assert isinstance(model.output_net.activation, nn.SiLU)

    # the linearized model still runs and is SO(3)-invariant
    pos, types = random_structure()
    e, f = _energy_and_forces(model, pos, types)
    Q = rand_rotation()
    e_rot = model(pos @ Q.T, types)
    de = (e - e_rot).abs().item()
    assert torch.isfinite(e) and torch.isfinite(f).all()
    assert de < 1e-9, f"SO(3) invariance broken under identity activation: {de:.3e}"

    try:
        ECENet(**COMMON, activation='nope')
        raise AssertionError("unknown activation should have raised")
    except ValueError as err:
        assert 'activation' in str(err)
    print(f"  identity activation: {len(nls)} nonlinearities no-op (rt={rt:.1e}), "
          f"MLPs keep SiLU, SO(3) dE={de:.1e}, unknown string rejected")


if __name__ == "__main__":
    print("ECENet integration")
    test_constructs_and_runs()
    test_so3_invariance()
    test_forces_finite_difference()
    test_training_path_forward_batch_multi()
    test_ecenet_mp()
    test_forward_batch_topology_list_formats()
    test_zero_edge_energy_is_atomic_and_consistent()
    test_identity_activation()
    print("All tests passed.")

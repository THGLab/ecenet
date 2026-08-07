"""Tests for the attention message passing (``ECENet(mp_type=...)``).

Per edge: a low-rank residual message + an invariant scalar score; messages are
score-weighted over each receiver atom's incoming edges, aggregated in the
global frame, then a per-edge receiver residual. Two aggregations share that
structure and differ only in the weight:

  'transformer' (default) — softmax over the receiver's in-edges (intensive)
  'sum'                   — raw signed score × cutoff envelope (extensive)

Message and scores come from ONE fused trunk whose zero-init up-projection emits
n_ch message channels plus one score channel per head.

Checks: SO(3) invariance (the key property), continuity across r_cut, softmax
weights sum to 1 per (atom, head), fused-trunk zero-init, sum extensivity,
finite forces, and multi-head splitting.

Run:  python tests/test_transformer_mp.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import warnings

import torch

from ecenet import ECENet
from ecenet.model import ECENetTransformerMPLayer

torch.manual_seed(0)
DTYPE = torch.float64
N_TYPES = 4
COMMON = dict(
    n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=3, embed_dim=8, n_layers=1, n_max_d=4,
)


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


def _activate_scores(layer, std=0.5, bias=None):
    """Make the fused trunk's score channels non-trivial. The scores are the m=0
    components of the trailing n_scores output channels, so perturbing just those
    rows leaves the message residual at its zero-init identity — the score effect
    is isolated."""
    with torch.no_grad():
        layer.msg_up.weights[0, -layer.n_scores:, :].normal_(std=std)
        if bias is not None:
            layer.msg_up.bias[-layer.n_scores:].fill_(bias)


def test_default_is_transformer():
    m = ECENet(**COMMON, n_mp=2).double()
    assert isinstance(m.mp_layers[0], ECENetTransformerMPLayer)
    assert m.mp_layers[0].aggregation == 'transformer'
    m_s = ECENet(**COMMON, n_mp=2, mp_type='sum').double()
    assert m_s.mp_layers[0].aggregation == 'sum'
    # the removed 'edge' MP is rejected, with a message that says so
    try:
        ECENet(**COMMON, n_mp=2, mp_type='edge')
    except ValueError as e:
        assert 'edge' in str(e) and 'removed' in str(e)
    else:
        raise AssertionError("expected mp_type='edge' to be rejected")
    print("  default mp_type='transformer'; 'sum' selects the weighted-sum aggregation; "
          "'edge' is rejected")


def test_so3_invariance():
    pos, types = random_structure(seed=2)
    for mp_type in ('transformer', 'sum'):
        for n_mp in (2, 3):
            m = ECENet(**COMMON, n_mp=n_mp, mp_type=mp_type).double()
            for L in m.mp_layers:     # scores are zero-init → activate them
                _activate_scores(L)
            err = (m(pos, types) - m(pos @ rand_rotation().T, types)).abs().item()
            assert err < 1e-9, f"{mp_type} MP breaks SO(3) at n_mp={n_mp}: {err:.2e}"
            print(f"  {mp_type}, n_mp={n_mp}: SO(3) invariance {err:.1e}")


def test_cutoff_continuity():
    """Energy must be continuous as an edge crosses r_cut_edge. Both aggregations
    carry the cutoff envelope, so a departing edge's contribution vanishes
    smoothly (no jump)."""
    RC = 5.0
    common = dict(n_types=N_TYPES, r_cut_edge=RC, r_cut_neighbor=4.0,
                  l_max=2, n_max=3, embed_dim=8, n_layers=1, n_max_d=4, n_mp=2)
    for mp_type in ('transformer', 'sum'):
        torch.manual_seed(1)
        m = ECENet(**common, mp_type=mp_type).double()
        for L in m.mp_layers:
            _activate_scores(L)
        types = torch.tensor([0, 1, 2])

        def energy(d, m=m):
            pos = torch.tensor([[0., 0, 0], [1.5, 0, 0], [d, 0, 0]], dtype=DTYPE)
            return m(pos, types).item()

        # one edge (atoms 0-2) crosses r_cut at d=RC; atoms 0-1 stay bonded
        Es = {d: energy(d) for d in (4.990, 4.998, 4.9995, 5.0005, 5.002, 5.010)}
        jump   = abs(Es[5.0005] - Es[4.9995])    # straddles the cutoff
        smooth = abs(Es[4.998] - Es[4.990])      # same-size step, same side
        assert jump < 10 * max(smooth, 1e-12), \
            f"{mp_type}: energy discontinuous across r_cut_edge: " \
            f"jump {jump:.2e} vs smooth {smooth:.2e}"
        print(f"  {mp_type}: continuity across r_cut, jump {jump:.1e} <= ~smooth {smooth:.1e}")


def test_forces_finite():
    pos, types = random_structure(seed=3)
    for mp_type in ('transformer', 'sum'):
        m = ECENet(**COMMON, n_mp=2, mp_type=mp_type).double()
        for L in m.mp_layers:
            _activate_scores(L)
        p = pos.clone().requires_grad_(True)
        e = m(p, types)
        f = -torch.autograd.grad(e, p, create_graph=True)[0]
        assert torch.isfinite(e) and f.shape == pos.shape and torch.isfinite(f).all()
        print(f"  forces finite ({mp_type} MP): |F|max={f.abs().max():.3f}")


def test_fused_trunk_zero_init():
    """The fused trunk's up-projection is zero-init, so the message residual and
    every score start at 0. For 'sum' that makes the whole MP layer an exact
    no-op at init; for 'transformer' exp(0)=1 leaves attention uniform instead."""
    torch.manual_seed(11)
    inp = _layer_inputs()

    for mp_type in ('sum', 'transformer'):
        layer = ECENetTransformerMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2,
                                         aggregation=mp_type).double()
        # one trunk, no separate message block or score head
        assert not hasattr(layer, 'message') and not hasattr(layer, 'score_w')
        assert layer.msg_up.out_features == layer.n_ch + layer.n_scores
        assert layer.msg_up.weights.abs().max() == 0.0
        assert layer.msg_up.bias.abs().max() == 0.0

        oc, os_ = layer(inp['A_cos'], inp['A_sin'], inp['r_hat'], inp['dist_ij'],
                        inp['edge_i'], inp['edge_j'], inp['n_atoms'],
                        inp['type_i'], inp['type_j'])
        d = max((oc - inp['A_cos']).abs().max().item(),
                (os_ - inp['A_sin']).abs().max().item())
        if mp_type == 'sum':
            assert d == 0.0, f"sum MP is not an exact no-op at init (off by {d:.2e})"
            print("  sum: fused trunk zero-init → exact identity at init")
        else:
            # uniform attention still mixes messages, so this must NOT be a no-op
            assert d > 1e-9, "transformer MP unexpectedly a no-op at init"
            w = _recompute_weights(layer, inp['A_cos'], inp['A_sin'], inp['dist_ij'],
                                   inp['edge_j'], inp['n_atoms'])
            # equal scores → weights are f_cut normalized per receiver, not equal in
            # general, but the *scores* must all be exactly 0
            u_cos, _ = layer.msg_up(*layer.msg_nonlin(*layer.msg_down(
                inp['A_cos'], inp['A_sin'])))
            s = u_cos[:, layer.n_ch:, 0]
            assert s.abs().max() == 0.0, "scores are not zero at init"
            assert w.min() > 0, "uniform attention should give positive weights"
            print("  transformer: fused trunk zero-init → scores 0, attention uniform")

    # ...and once the score channels are active the output moves.
    pos, types = random_structure(seed=6)
    m = ECENet(**COMMON, n_mp=2, mp_type='sum').double()
    e_init = m(pos, types).item()
    _activate_scores(m.mp_layers[0])
    assert abs(e_init - m(pos, types).item()) > 1e-9, "active sum MP had no effect"
    print("  sum MP becomes active once the score channels learn")


def test_sum_is_extensive():
    """The sum aggregation has no normalizer, so adding identical in-edges scales
    a receiver's total weight linearly; the softmax normalizes it to 1 however
    many there are. Driven on a star topology of *identical* edges (same distance,
    same features → same score), so neighbour count is the only variable."""
    def total_weight(mp_type, n_neigh):
        torch.manual_seed(2)
        layer = ECENetTransformerMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2,
                                         aggregation=mp_type).double()
        _activate_scores(layer, std=0.3, bias=0.5)   # scores are zero-init
        # n_neigh edges, all into atom 0, all carrying identical features/distance
        g = torch.Generator().manual_seed(4)
        one_c = torch.randn(1, 48, 3, generator=g, dtype=DTYPE)
        one_s = torch.randn(1, 48, 3, generator=g, dtype=DTYPE)
        one_s[:, :, 0] = 0.0
        A_cos, A_sin = one_c.repeat(n_neigh, 1, 1), one_s.repeat(n_neigh, 1, 1)
        dist_ij = torch.full((n_neigh,), 2.0, dtype=DTYPE)
        edge_j = torch.zeros(n_neigh, dtype=torch.long)
        return _recompute_weights(layer, A_cos, A_sin, dist_ij, edge_j,
                                  n_atoms=n_neigh + 1).sum().item()

    for mp_type, expected in (('sum', 2.0), ('transformer', 1.0)):
        w3, w6 = total_weight(mp_type, 3), total_weight(mp_type, 6)
        ratio = w6 / w3
        assert abs(ratio - expected) < 1e-6, \
            f"{mp_type}: total weight ratio for 6 vs 3 in-edges was {ratio:.4f}, expected {expected}"
        print(f"  {mp_type}: doubling the in-edge count scales the total weight {ratio:.3f}x "
              f"({'extensive' if mp_type == 'sum' else 'intensive'}; Σw={w6:.3f} at 6 edges)")


def _layer_inputs(n_atoms=6, n_ch=48, m_max=2, seed=5):
    """Synthetic edge batch for driving an MP layer directly (fully connected)."""
    g = torch.Generator().manual_seed(seed)
    ei, ej = torch.meshgrid(torch.arange(n_atoms), torch.arange(n_atoms), indexing='ij')
    mask = ei != ej
    edge_i, edge_j = ei[mask].contiguous(), ej[mask].contiguous()
    n_e = edge_i.shape[0]
    r_hat = torch.randn(n_e, 3, generator=g, dtype=DTYPE)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    dist_ij = torch.rand(n_e, generator=g, dtype=DTYPE) * 4.0 + 0.5   # inside r_cut=5
    A_cos = torch.randn(n_e, n_ch, m_max + 1, generator=g, dtype=DTYPE)
    A_sin = torch.randn(n_e, n_ch, m_max + 1, generator=g, dtype=DTYPE)
    A_sin[:, :, 0] = 0.0                       # m=0 sin slot is a structural zero
    types = torch.randint(0, N_TYPES, (n_atoms,), generator=g)
    return dict(A_cos=A_cos, A_sin=A_sin, r_hat=r_hat, dist_ij=dist_ij,
                edge_i=edge_i, edge_j=edge_j, n_atoms=n_atoms,
                type_i=types[edge_i], type_j=types[edge_j])


def _recompute_weights(layer, A_cos, A_sin, dist_ij, edge_j, n_atoms):
    """The layer's per-edge weights, recomputed from its own fused trunk —
    independently of the forward's max-subtraction path. (n_e, n_heads)."""
    H = layer.n_heads
    u_cos, u_sin = layer.msg_down(A_cos, A_sin)
    u_cos, u_sin = layer.msg_nonlin(u_cos, u_sin)
    u_cos, u_sin = layer.msg_up(u_cos, u_sin)
    s = u_cos[:, layer.n_ch:layer.n_ch + layer.n_scores, 0]  # (n_e, H)
    f_cut = layer.cutoff_fn(dist_ij, layer.r_cut)
    if layer.aggregation == 'sum':
        return s * f_cut[:, None]
    num = torch.exp(s) * f_cut[:, None]
    ej = edge_j[:, None].expand(-1, H)
    denom = torch.zeros(n_atoms, H, dtype=A_cos.dtype).scatter_add(0, ej, num)
    return num / (denom[edge_j] + layer.softmax_eps)


def test_softmax_weights_sum_to_one():
    """The per-(receiver, head) attention weights are a softmax: they sum to 1
    (up to the +eps normalizer floor). This is what makes the aggregation a
    weighted average — intensive in coordination rather than growing with it."""
    for H in (1, 2, 4):
        torch.manual_seed(3)
        layer = ECENetTransformerMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2,
                                         n_heads=H).double()
        inp = _layer_inputs()
        a = _recompute_weights(layer, inp['A_cos'], inp['A_sin'], inp['dist_ij'],
                               inp['edge_j'], inp['n_atoms'])
        ej = inp['edge_j'][:, None].expand(-1, H)
        sums = torch.zeros(inp['n_atoms'], H, dtype=DTYPE).scatter_add(0, ej, a)
        err = (sums - 1.0).abs().max().item()
        assert err < 1e-6, f"attention weights do not sum to 1 per (atom, head): off by {err:.2e}"
        print(f"  n_heads={H}: softmax weights sum to 1 per (atom, head) (max dev {err:.1e})")


def test_multihead():
    """Heads split the value channels (n_base) into contiguous whole-n_sph groups,
    each gated by its own softmax. Check: the score head widens with n_heads, the
    split is validated, SO(3) still holds, and heads actually change the output."""
    pos, types = random_structure(seed=2)
    outs = {}
    for H in (1, 2, 4):
        torch.manual_seed(7)
        m = ECENet(**COMMON, n_mp=2, mp_type='transformer', mp_n_heads=H).double()
        L = m.mp_layers[0]
        assert L.n_heads == H and L.n_scores == H
        # the fused trunk widens by one score channel per head
        assert L.msg_up.out_features == L.n_ch + H
        assert L.n_base % H == 0
        # active (non-identity) attention: perturb the score channels per edge
        _activate_scores(L)
        err = (m(pos, types) - m(pos @ rand_rotation().T, types)).abs().item()
        assert err < 1e-9, f"multi-head transformer MP breaks SO(3) at n_heads={H}: {err:.2e}"
        outs[H] = m(pos, types).item()
        print(f"  n_heads={H}: SO(3) {err:.1e}, fused trunk out={L.msg_up.out_features} "
              f"(= n_ch {L.n_ch} + {H} scores)")
    assert abs(outs[1] - outs[2]) > 1e-9 and abs(outs[2] - outs[4]) > 1e-9, \
        "n_heads had no effect on the output"

    # n_base must divide evenly across heads
    try:
        ECENet(**COMMON, n_mp=2, mp_type='transformer', mp_n_heads=3)
    except ValueError as e:
        assert 'divisible' in str(e)
        print(f"  indivisible n_heads raises: {str(e)[:60]}…")
    else:
        raise AssertionError("expected a ValueError for n_base % n_heads != 0")


def test_ignored_flags_warn():
    """mp_n_heads does nothing without message passing: silently ignoring it
    would look like it had been applied, so it warns."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ECENet(**COMMON, n_mp=1, mp_n_heads=4)            # no MP layers
        assert any('mp_n_heads' in str(x.message) for x in w), "expected an ignored-flag warning"
    try:
        ECENet(**COMMON, n_mp=2, mp_type='nope')
    except ValueError as e:
        assert 'Unknown mp_type' in str(e)
    else:
        raise AssertionError("expected a ValueError for an unknown mp_type")
    print("  mp_n_heads warns when ignored; unknown mp_type raises")


if __name__ == "__main__":
    print("Attention message-passing tests (mp_type='transformer' / 'sum')")
    test_default_is_transformer()
    test_so3_invariance()
    test_cutoff_continuity()
    test_forces_finite()
    test_fused_trunk_zero_init()
    test_sum_is_extensive()
    test_softmax_weights_sum_to_one()
    test_multihead()
    test_ignored_flags_warn()
    print("All tests passed.")

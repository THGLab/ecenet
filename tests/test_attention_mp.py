"""Tests for the attention message passing (``ECENet(mp_type=...)``).

Per edge: a low-rank residual message + an invariant scalar score; messages are
score-weighted over each receiver atom's incoming edges, aggregated in the
global frame, then a per-edge receiver residual. Two aggregations share that
structure and differ only in the weight:

  'sum' (default) — raw signed score × cutoff envelope (extensive)
  'softmax'       — softmax over the receiver's in-edges (intensive)

Message and scores come from ONE fused trunk whose zero-init up-projection emits
n_ch message channels plus one score channel per head.

Checks: SO(3) invariance (the key property), continuity across r_cut, softmax
weights sum to 1 per (atom, head), fused-trunk zero-init, sum extensivity,
finite forces, and multi-head splitting.

Run:  python tests/test_attention_mp.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import warnings

import torch

from ecenet import ECENet
from ecenet.model import ECENetAttentionMPLayer

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


def test_default_is_sum():
    m = ECENet(**COMMON, n_mp=2).double()
    assert isinstance(m.mp_layers[0], ECENetAttentionMPLayer)
    assert m.mp_layers[0].aggregation == 'sum'
    m_s = ECENet(**COMMON, n_mp=2, mp_type='softmax').double()
    assert m_s.mp_layers[0].aggregation == 'softmax'
    # the removed 'edge' MP is rejected, with a message that says so
    try:
        ECENet(**COMMON, n_mp=2, mp_type='edge')
    except ValueError as e:
        assert 'edge' in str(e) and 'removed' in str(e)
    else:
        raise AssertionError("expected mp_type='edge' to be rejected")
    print("  default mp_type='sum'; 'softmax' selects the attention aggregation; "
          "'edge' is rejected")


def test_so3_invariance():
    pos, types = random_structure(seed=2)
    for mp_type in ('softmax', 'sum'):
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
    for mp_type in ('softmax', 'sum'):
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
    for mp_type in ('softmax', 'sum'):
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
    no-op at init; for 'softmax' exp(0)=1 leaves attention uniform instead."""
    torch.manual_seed(11)
    inp = _layer_inputs()

    for mp_type in ('sum', 'softmax'):
        layer = ECENetAttentionMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2,
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
            assert d > 1e-9, "softmax MP unexpectedly a no-op at init"
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


def test_l_attention_shapes_and_map():
    """l_attention widens the fused trunk to one score per (head, l), and l_of_s
    maps every spherical index to its degree — constant within each l-block, which
    is exactly what makes a per-l weight equivariant."""
    l_max = COMMON['l_max']
    for H in (1, 2):
        off = ECENet(**COMMON, n_mp=2, mp_n_heads=H).mp_layers[0]
        on = ECENet(**COMMON, n_mp=2, mp_n_heads=H, mp_l_attention=True).mp_layers[0]
        assert off.n_scores_per_head == 1 and off.n_scores == H
        assert on.n_scores_per_head == l_max + 1 and on.n_scores == H * (l_max + 1)
        assert on.msg_up.out_features == on.n_ch + H * (l_max + 1)
        # l_of_s: degree of each spherical index, l-major contiguous blocks
        expected = torch.cat([torch.full((2 * l + 1,), l, dtype=torch.long)
                              for l in range(l_max + 1)])
        assert torch.equal(on.l_of_s, expected), f"bad l_of_s: {on.l_of_s.tolist()}"
        assert torch.equal(off.l_of_s, torch.zeros(off.n_sph, dtype=torch.long))
        # every m of a given l shares one slot — no split within an l
        for l in range(l_max + 1):
            block = on.l_of_s[l * l:(l + 1) ** 2]
            assert (block == l).all(), f"l={l} block is not uniform: {block.tolist()}"
    # the buffer is derived from l_max, so it must not be in the checkpoint
    sd = ECENet(**COMMON, n_mp=2, mp_l_attention=True).state_dict()
    assert not any('l_of_s' in k for k in sd), "l_of_s should be non-persistent"
    print(f"  l_attention: n_scores {H}→{H * (l_max + 1)}, l_of_s={expected.tolist()}, "
          f"uniform within each l-block, non-persistent")


def test_l_attention_independent_per_l():
    """Each (head, l) runs its OWN softmax: weights sum to 1 for every (atom, head,
    l) slot, and genuinely differ across l — it is not one weight broadcast."""
    torch.manual_seed(5)
    layer = ECENetAttentionMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2, n_heads=2,
                                     l_attention=True, msg_envelope=False).double()
    _activate_scores(layer, std=0.8)
    inp = _layer_inputs()
    a = _recompute_weights(layer, inp['A_cos'], inp['A_sin'], inp['dist_ij'],
                           inp['edge_j'], inp['n_atoms'])          # (n_e, K)
    K = layer.n_scores
    assert a.shape[1] == K == 2 * (layer.l_max + 1)
    ej = inp['edge_j'][:, None].expand(-1, K)
    sums = torch.zeros(inp['n_atoms'], K, dtype=DTYPE).scatter_add(0, ej, a)
    # Each slot sums to denom/(denom+softmax_eps), so the only deviation from 1 is
    # the eps floor: it can undershoot but never overshoot, and only by ~eps/denom.
    err = (sums - 1.0).abs().max().item()
    assert (sums <= 1.0 + 1e-12).all(), "a softmax slot summed to more than 1"
    assert err < 1e-4, f"per-(head, l) softmax does not sum to 1: off by {err:.2e}"
    # slots must differ: reshape to (n_e, H, l_max+1) and compare across l
    a_hl = a.reshape(-1, layer.n_heads, layer.n_scores_per_head)
    spread = (a_hl.max(dim=2).values - a_hl.min(dim=2).values).max().item()
    assert spread > 1e-3, f"weights are effectively identical across l ({spread:.2e})"
    print(f"  l_attention: {K} independent softmaxes, each sums to 1 (dev {err:.1e}), "
          f"weights differ across l by up to {spread:.3f}")


def test_l_attention_so3_and_forces():
    """A per-l weight is still exactly equivariant (the Wigner-D block is
    l-diagonal), for both aggregations, and forces stay finite."""
    pos, types = random_structure(seed=2)
    for mp_type in ('softmax', 'sum'):
        m = ECENet(**COMMON, n_mp=2, mp_type=mp_type, mp_n_heads=2,
                   mp_l_attention=True).double()
        L = m.mp_layers[0]
        e_init = m(pos, types)
        _activate_scores(L, std=0.5)
        e0 = m(pos, types)
        err = (e0 - m(pos @ rand_rotation().T, types)).abs().item()
        assert err < 1e-9, f"{mp_type} l_attention breaks SO(3): {err:.2e}"
        assert (e0 - e_init).abs().item() > 1e-9, f"{mp_type} l_attention had no effect"
        p = pos.clone().requires_grad_(True)
        f = -torch.autograd.grad(m(p, types), p, create_graph=True)[0]
        assert torch.isfinite(f).all() and f.shape == pos.shape
        print(f"  l_attention ({mp_type}): SO(3) {err:.1e}, |F|max={f.abs().max():.3f}")

    # zero-init still makes 'sum' an exact identity, with the wider score block
    layer = ECENetAttentionMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2, n_heads=2,
                                     aggregation='sum', l_attention=True).double()
    inp = _layer_inputs()
    oc, os_ = layer(inp['A_cos'], inp['A_sin'], inp['r_hat'], inp['dist_ij'],
                    inp['edge_i'], inp['edge_j'], inp['n_atoms'],
                    inp['type_i'], inp['type_j'])
    d = max((oc - inp['A_cos']).abs().max().item(), (os_ - inp['A_sin']).abs().max().item())
    assert d == 0.0, f"sum + l_attention is not identity at init: {d:.2e}"
    print("  l_attention: sum still an exact identity at init")


def test_msg_envelope_restores_absolute_decay():
    """Without the envelope, the softmax normalizer divides the ABSOLUTE f_cut back
    out: a lone neighbour near r_cut still gets weight ~1, so the message is flat
    in distance. mp_msg_envelope (on by default) multiplies it back in, so the
    weight tracks f_cut exactly. 'sum' is already enveloped and must not change."""
    RC = 5.0

    def weight(aggregation, envelope, r):
        torch.manual_seed(0)
        layer = ECENetAttentionMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2, r_cut=RC,
                                         aggregation=aggregation,
                                         msg_envelope=envelope).double()
        _activate_scores(layer, std=0.5, bias=0.5)
        g = torch.Generator().manual_seed(3)
        A_cos = torch.randn(1, 48, 3, generator=g, dtype=DTYPE)
        A_sin = torch.randn(1, 48, 3, generator=g, dtype=DTYPE)
        A_sin[:, :, 0] = 0.0
        a = _recompute_weights(layer, A_cos, A_sin, torch.tensor([r], dtype=DTYPE),
                               torch.zeros(1, dtype=torch.long), n_atoms=2)
        f_cut = layer.cutoff_fn(torch.tensor([r], dtype=DTYPE), RC).item()
        return a.abs().max().item(), f_cut

    for r in (1.5, 2.5, 3.5, 4.5):
        on, f_cut = weight('softmax', True, r)
        off, _ = weight('softmax', False, r)
        # enveloped: a single in-edge's weight is exactly f_cut (softmax gives 1)
        assert abs(on / f_cut - 1.0) < 1e-3, \
            f"enveloped weight should equal f_cut at r={r}: {on:.6f} vs {f_cut:.6f}"
        # unenveloped: ~1 regardless of distance — the flat behaviour
        assert abs(off - 1.0) < 1e-3, f"unenveloped weight should be ~1 at r={r}: {off:.6f}"
    # over 1.5 -> 4.5 Å the enveloped weight falls ~32x while the bare one is flat
    on_near, _ = weight('softmax', True, 1.5)
    on_far,  _ = weight('softmax', True, 4.5)
    off_near, _ = weight('softmax', False, 1.5)
    off_far,  _ = weight('softmax', False, 4.5)
    assert on_far / on_near < 0.05, "enveloped weight should decay strongly toward r_cut"
    assert off_far / off_near > 0.99, "unenveloped weight should be ~flat in r"
    # 'sum' is enveloped by construction: the flag must not change it
    for r in (1.5, 4.5):
        assert weight('sum', True, r)[0] == weight('sum', False, r)[0], \
            "mp_msg_envelope must not alter the sum aggregation"
    print(f"  mp_msg_envelope: weight == f_cut (decays {on_far/on_near:.3f}x over 1.5→4.5 Å); "
          f"off → flat ({off_far/off_near:.3f}x); sum unaffected")


def test_msg_envelope_defaults_and_flag():
    """On by default for 'softmax'; structurally already on (and not
    disableable) for 'sum' — the default — which warns rather than silently
    ignoring."""
    assert ECENet(**COMMON, n_mp=2,
                  mp_type='softmax').mp_layers[0].msg_envelope is True
    assert ECENet(**COMMON, n_mp=2, mp_type='softmax',
                  mp_msg_envelope=False).mp_layers[0].msg_envelope is False
    # 'sum' never sets the flag — its weight is s*f_cut, so f_cut twice would be f_cut²
    assert ECENet(**COMMON, n_mp=2).mp_layers[0].msg_envelope is False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ECENet(**COMMON, n_mp=2, mp_type='sum', mp_msg_envelope=False)
        assert any('mp_msg_envelope' in str(x.message) for x in w), \
            "disabling an envelope that is structural should warn"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ECENet(**COMMON, n_mp=2, mp_type='sum')
        assert not any('envelope' in str(x.message) for x in w), "plain sum should be quiet"
    print("  mp_msg_envelope: default on for softmax, structural for sum, warns if disabled there")


def test_sum_is_extensive():
    """The sum aggregation has no normalizer, so adding identical in-edges scales
    a receiver's total weight linearly; the softmax normalizes it to 1 however
    many there are. Driven on a star topology of *identical* edges (same distance,
    same features → same score), so neighbour count is the only variable."""
    def total_weight(mp_type, n_neigh):
        torch.manual_seed(2)
        layer = ECENetAttentionMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2,
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

    for mp_type, expected in (('sum', 2.0), ('softmax', 1.0)):
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
    independently of the forward's max-subtraction path. (n_e, n_scores): one
    slot per head, or per (head, l) with l_attention."""
    H = layer.n_scores
    u_cos, u_sin = layer.msg_down(A_cos, A_sin)
    u_cos, u_sin = layer.msg_nonlin(u_cos, u_sin)
    u_cos, u_sin = layer.msg_up(u_cos, u_sin)
    s = u_cos[:, layer.n_ch:layer.n_ch + layer.n_scores, 0]  # (n_e, H)
    f_cut = layer.cutoff_fn(dist_ij, layer.r_cut)
    if layer.aggregation == 'sum':
        return s * f_cut[:, None]
    # Mirror the layer's per-(receiver, head) max-subtraction. It is an exact
    # softmax either way, but softmax_eps is compared against the max-subtracted
    # normalizer, so omitting it changes the result once the normalizer nears eps.
    ej = edge_j[:, None].expand(-1, H)
    s_max = torch.full((n_atoms, H), float('-inf'), dtype=A_cos.dtype).scatter_reduce(
        0, ej, s.detach(), reduce='amax', include_self=True)
    num = torch.exp(s - s_max[edge_j]) * f_cut[:, None]
    denom = torch.zeros(n_atoms, H, dtype=A_cos.dtype).scatter_add(0, ej, num)
    a = num / (denom[edge_j] + layer.softmax_eps)
    if layer.msg_envelope:
        a = a * f_cut[:, None]
    return a


def test_softmax_weights_sum_to_one():
    """The per-(receiver, head) attention weights are a softmax: they sum to 1
    (up to the +eps normalizer floor). This is what makes the aggregation a
    weighted average — intensive in coordination rather than growing with it.

    Built with msg_envelope=False to isolate the normalizer: the envelope (on by
    default) multiplies the weights by f_cut afterwards, so they then sum to the
    f_cut-weighted average rather than to 1. That is deliberate — the sum-to-1
    property being checked here is a property of the softmax, not of the layer's
    final weights."""
    for H in (1, 2, 4):
        torch.manual_seed(3)
        layer = ECENetAttentionMPLayer(48, 2, 8, n_types=N_TYPES, m_max=2,
                                         n_heads=H, msg_envelope=False).double()
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
        m = ECENet(**COMMON, n_mp=2, mp_type='softmax', mp_n_heads=H).double()
        L = m.mp_layers[0]
        assert L.n_heads == H and L.n_scores == H
        # the fused trunk widens by one score channel per head
        assert L.msg_up.out_features == L.n_ch + H
        assert L.n_base % H == 0
        # active (non-identity) attention: perturb the score channels per edge
        _activate_scores(L)
        err = (m(pos, types) - m(pos @ rand_rotation().T, types)).abs().item()
        assert err < 1e-9, f"multi-head softmax MP breaks SO(3) at n_heads={H}: {err:.2e}"
        outs[H] = m(pos, types).item()
        print(f"  n_heads={H}: SO(3) {err:.1e}, fused trunk out={L.msg_up.out_features} "
              f"(= n_ch {L.n_ch} + {H} scores)")
    assert abs(outs[1] - outs[2]) > 1e-9 and abs(outs[2] - outs[4]) > 1e-9, \
        "n_heads had no effect on the output"

    # n_base must divide evenly across heads
    try:
        ECENet(**COMMON, n_mp=2, mp_type='softmax', mp_n_heads=3)
    except ValueError as e:
        assert 'divisible' in str(e)
        print(f"  indivisible n_heads raises: {str(e)[:60]}…")
    else:
        raise AssertionError("expected a ValueError for n_base % n_heads != 0")


def test_transformer_is_a_deprecated_alias():
    """mp_type was renamed 'transformer' -> 'softmax'. The old name still works —
    it is stored in the hparams of checkpoints written before the rename — but it
    warns and normalises."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = ECENet(**COMMON, n_mp=2, mp_type='transformer').double()
        assert any('renamed' in str(x.message) for x in w), "expected a rename warning"
    assert m.mp_type == 'softmax', f"alias should normalise, got {m.mp_type!r}"
    assert m.mp_layers[0].aggregation == 'softmax'
    # ...and produces exactly the same model as the new name
    torch.manual_seed(3)
    a = ECENet(**COMMON, n_mp=2, mp_type='softmax').double()
    torch.manual_seed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = ECENet(**COMMON, n_mp=2, mp_type='transformer').double()
    pos, types = random_structure(seed=2)
    assert a(pos, types).item() == b(pos, types).item(), "alias built a different model"
    print("  mp_type='transformer' still accepted, warns, normalises to 'softmax'")


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
    print("Attention message-passing tests (mp_type='softmax' / 'sum')")
    test_default_is_sum()
    test_so3_invariance()
    test_cutoff_continuity()
    test_forces_finite()
    test_fused_trunk_zero_init()
    test_l_attention_shapes_and_map()
    test_l_attention_independent_per_l()
    test_l_attention_so3_and_forces()
    test_msg_envelope_restores_absolute_decay()
    test_msg_envelope_defaults_and_flag()
    test_sum_is_extensive()
    test_softmax_weights_sum_to_one()
    test_multihead()
    test_transformer_is_a_deprecated_alias()
    test_ignored_flags_warn()
    print("All tests passed.")

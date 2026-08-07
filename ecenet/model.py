"""ecenet/model.py — ECENet: equivariant Cartesian-edge interatomic potential.

Pipeline:
  1. ACE atomic basis:
       A[i, t, n, s] = Σ_k R_n(r_ik) Y_s(r̂_ik) δ(type_k=t)
       shape: (n_atoms, n_types, n_max, n_sph)

  2. Joint contraction per central atom type:
       A_emb[i, c, s] = Σ_{t,n} A[i, t, n, s] * W[types[i], t, n, c]
       shape: (n_atoms, embed_dim, n_sph)
       W: (n_types, n_types, n_max, embed_dim)

  3. Gather for edge endpoints + Wigner rotation into bond frame:
       stack [A_emb[edge_i], A_emb[edge_j]] → rotate by D(r̂_ij)
       shape: (n_edges, 2*embed_dim, n_sph)

  4. Reshape to A_cos / A_sin:
       shape: (n_edges, 2*embed_dim, n_angular)  where n_angular = l_max + 1

  5. Equivariant layers × n_layers (EquivariantLinear → nonlinearity → residual)

  6. Contract to invariants:
       m=0: A_cos[:, :, 0]
       m>0: A_cos[:,:,m]² + A_sin[:,:,m]²
       Optional outer product with radial basis f_d(r_ij) of rank n_max_d.

  7. Output MLP([invariants, r_ij_scaled]) → per-edge scalar → sum over edges
     + per-type atomic energy baseline
"""

import functools
import warnings

import torch
import torch.nn as nn

from ecenet.ace_basis import ACEBasisAnalytic
from ecenet.equivariant import EquivariantLinear, RealSpaceNonlinearity
from ecenet.radial import find_edges, get_cutoff_fn, radial_basis
from ecenet.spherical import build_D_block, spherical_harmonics_float64, wigner_rotate

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ECENet(nn.Module):
    """ECENet — SO(3)-equivariant interatomic potential using per-edge SO(2) features.

    Args:
        n_types:        number of atom types
        r_cut_edge:     edge formation cutoff (Å)
        r_cut_neighbor: neighbour-list cutoff for the ACE basis (Å)
        l_max:          max angular momentum of the spherical-harmonic / ACE basis
        n_max:          radial basis functions per (type, l)
        embed_dim:      embedding dim after the joint (n_types, n_max) contraction
        n_layers:       equivariant layers per stage
        n_mp:           number of stages; one equivariant message-passing layer is
                        inserted between consecutive stages (n_mp-1 MP layers, no
                        trailing MP). n_mp=1 (default) is the plain model with no
                        message passing. n_mp=K is equivalent to the old
                        (n_mp_steps=K-1, n_final_layers=n_layers) layout.
        n_max_d:        if set, outer-product the invariants with f_d(r_ij) of this rank
        m_max:          max angular mode |m| kept after the equivariant layers
                        (default: l_max); lower it to cut cost at large l_max
        cutoff_type:    'cosine' or 'poly'
        activation:     pointwise activation in the realspace nonlinearity ('silu', 'tanh', ...)
        n_grid:         θ-grid points for the realspace nonlinearity (default: 4*m_max+1)
        output_hidden_dims: hidden widths of the readout MLP (default: [64])
        analytic_ace_basis: use ACEBasisAnalytic (recommended for force training)
        bottleneck_dim: if set, each equivariant layer becomes a low-rank block
                        (down → nonlin at this width → up, zero-init up so the
                        layer is identity at init); None → full-width layers
        mp_type:        how messages are aggregated at each receiver atom (n_mp
                        >= 2 only). Both styles share the same per-edge structure
                        — a fused message/score trunk and a receiver transform —
                        and differ only in the weight applied to each incoming
                        message:
                          'transformer' (default): softmax over the receiver's
                            incoming edges, so the aggregate is a weighted
                            *average* (intensive in coordination). Zero-init
                            scores make the attention uniform at init.
                          'sum': the raw signed score times the cutoff envelope,
                            summed (extensive in coordination). Zero-init scores
                            make the layer an exact no-op at init.
        mp_dim:         bottleneck width of the fused message/score trunk and of
                        the receiver block (default: n_features_per_m // 4)
        mp_n_heads:     number of attention heads; the value channels
                        (n_base = 2*embed_dim) split evenly across them
                        (default 1). Ignored (with a warning) when n_mp=1.
    """

    def __init__(
        self,
        n_types: int,
        r_cut_edge: float = 5.0,
        r_cut_neighbor: float = 4.0,
        l_max: int = 3,
        n_max: int = 4,
        embed_dim: int = 16,
        n_layers: int = 2,
        n_mp: int = 1,
        n_max_d: int = None,
        cutoff_type: str = 'cosine',
        activation: str = 'silu',
        use_nonlinearity: bool = True,
        n_grid: int = None,
        analytic_ace_basis: bool = True,
        output_hidden_dims: list = None,
        m_max: int = None,
        bottleneck_dim: int = None,
        mp_type: str = 'transformer',
        mp_dim: int = None,
        mp_n_heads: int = 1,
    ):
        super().__init__()
        self.n_types = n_types
        self.r_cut_edge = r_cut_edge
        self.r_cut_neighbor = r_cut_neighbor
        l_max = int(l_max)
        self.l_max = l_max
        self.n_max = n_max
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_max_d = n_max_d
        self.cutoff_type = cutoff_type
        self.activation = activation
        self.use_nonlinearity = use_nonlinearity
        self.n_grid = n_grid
        self.analytic_ace_basis = analytic_ace_basis
        self.bottleneck_dim = bottleneck_dim
        self.mp_type = mp_type
        self.n_sph = (l_max + 1) ** 2
        self.m_max = int(m_max) if m_max is not None else l_max
        self.n_angular = self.m_max + 1   # m = 0..m_max (layers only use up to m_max)

        # ── Joint (n_types, n_max) → embed_dim contraction per central atom type ──
        # W[type_i, t, n, c]: for central atom of type type_i, contract
        # neighbor type t and radial channel n into embed channel c.
        # (Initial Atomic embedding)
        self.W = nn.Parameter(
            torch.randn(n_types, n_types, n_max, embed_dim)
            / (n_types * n_max) ** 0.5
        )

        # ── SH → A_cos/A_sin reshape ──────────────────────────────────────
        # m_max controls output angular modes; ACE basis always uses full l_max.
        # Going from node to edge frame
        self.sph_to_angular = SphToAngular(embed_dim, l_max, m_max=self.m_max)
        # n_features_per_m = 2 * embed_dim * (l_max+1): one channel per (side, embed, l)
        self.n_features_per_m = 2 * embed_dim * (l_max + 1)

        # ── Equivariant layers: Linear → RealSpaceNonlinearity → residual ────
        # Message passing: the model is `n_mp` stages of `n_layers` equivariant
        # layers each, with one equivariant MP layer *between* consecutive stages
        # (n_mp-1 MP layers total, no trailing MP). n_mp == 1 is the plain model:
        # a flat list of n_layers equivariant layers and no MP. n_mp >= 2 groups
        # the layers into stages and adds the interleaved MP layers.
        self.n_mp = n_mp
        self.layers = nn.ModuleList([
            ECENetLayer(self.n_features_per_m, self.m_max, activation=activation,
                        use_nonlinearity=use_nonlinearity, n_grid=n_grid,
                        bottleneck_dim=bottleneck_dim)
            for _ in range(n_mp * n_layers)
        ])
        # n_mp >= 2: regroup the flat layers into `n_mp` stages and build the
        # `n_mp - 1` MP layers that sit between them.
        if mp_type not in ('transformer', 'sum'):
            raise ValueError(
                f"Unknown mp_type '{mp_type}' (expected 'transformer' or 'sum'). "
                "The old distance/type-weighted 'edge' message passing has been removed.")
        # Warn rather than silently ignore: an MP-only knob left at a non-default
        # value with n_mp=1 does nothing, and a silent no-op looks like the
        # setting was applied.
        if mp_n_heads != 1 and n_mp == 1:
            warnings.warn(
                f"mp_n_heads={mp_n_heads} is ignored: message passing is off "
                f"(n_mp=1).", stacklevel=2)
        if n_mp > 1:
            flat = list(self.layers)
            self.layers = nn.ModuleList([
                nn.ModuleList(flat[g * n_layers:(g + 1) * n_layers])
                for g in range(n_mp)
            ])
            self.mp_layers = nn.ModuleList([
                ECENetTransformerMPLayer(
                    self.n_features_per_m, self.l_max, self.embed_dim,
                    n_types=n_types,
                    r_cut=self.r_cut_edge, cutoff_type=self.cutoff_type,
                    m_max=self.m_max, mp_dim=mp_dim,
                    activation=activation, n_grid=n_grid, n_heads=mp_n_heads,
                    aggregation=mp_type,
                )
                for _ in range(n_mp - 1)
            ])

        # ── Output MLP ──────────────────────────────────────────────────────
        # inv → MLP → n_max_d, then dot with rij_basis (see _apply_output).
        hidden_dims = output_hidden_dims or [64]
        in_dim = self.n_features_per_m
        n_output_out = n_max_d if n_max_d is not None else 1
        mlp_dims = [in_dim] + list(hidden_dims) + [n_output_out]
        act = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'relu': nn.ReLU,
               'gelu': nn.GELU}.get(activation, nn.SiLU)
        self.output_net = OutputMLP(mlp_dims, activation=act())

        # ── Per-type atomic energy baseline ──────────────────────────────
        self.atomic_energy = nn.Parameter(torch.zeros(n_types))


    # ── Helpers ────────────────────────────────────────────────────────────

    def _compute_ace_basis(self, pos_batch, nb_src, nb_dst, types, shift_vecs_nb=None):
        """Compute ACE atomic basis: (B, N, n_types, n_max, n_sph)."""
        if self.analytic_ace_basis:
            cutoff_type_id = 0 if self.cutoff_type == 'cosine' else 1
            return ACEBasisAnalytic.apply(
                pos_batch, nb_src, nb_dst, types,
                self.r_cut_neighbor, self.n_max, self.l_max,
                self.n_types, cutoff_type_id, shift_vecs_nb)

        B, N, _ = pos_batch.shape
        n_nb = nb_src.shape[0]
        device, dtype = pos_batch.device, pos_batch.dtype

        if n_nb == 0:
            return torch.zeros(B, N, self.n_types, self.n_max, self.n_sph,
                               device=device, dtype=dtype)

        diff_ik = pos_batch[:, nb_dst] - pos_batch[:, nb_src]
        if shift_vecs_nb is not None:
            diff_ik = diff_ik + shift_vecs_nb.to(dtype=dtype)[None]
        r_ik = torch.sqrt((diff_ik ** 2).sum(-1) + 1e-30)
        r_hat_ik = diff_ik / r_ik.unsqueeze(-1)

        f_R = radial_basis(r_ik.reshape(-1), self.r_cut_neighbor, self.n_max,
                           cutoff_type=self.cutoff_type).reshape(B, n_nb, self.n_max)
        Y = spherical_harmonics_float64(self.l_max, r_hat_ik.reshape(-1, 3),
                                        normalize=False).reshape(B, n_nb, self.n_sph)
        contributions = f_R.unsqueeze(-1) * Y.unsqueeze(-2)  # (B, n_nb, n_max, n_sph)

        neighbor_types = types[nb_dst]
        flat_idx = nb_src * self.n_types + neighbor_types
        flat_idx_exp = flat_idx[None, :, None, None].expand(B, n_nb, self.n_max, self.n_sph)
        A_flat = torch.zeros(B, N * self.n_types, self.n_max, self.n_sph,
                             device=device, dtype=dtype)
        A_flat = A_flat.scatter_add(1, flat_idx_exp, contributions)
        return A_flat.reshape(B, N, self.n_types, self.n_max, self.n_sph)

    def _embed(self, A, types):
        """Joint (n_types, n_max) → embed_dim contraction per central atom type.

        Args:
            A:     (n_atoms, n_types, n_max, n_sph)
            types: (n_atoms,) central atom type indices

        Returns:
            A_emb: (n_atoms, embed_dim, n_sph)
        """
        W_i = self.W[types]  # (n_atoms, n_types, n_max, embed_dim)
        return torch.einsum('itns,itnc->ics', A, W_i)

    def _contract(self, A_cos, A_sin):
        """Extract m=0 invariants: (n_edges, n_features_per_m, n_angular) → (n_edges, n_features_per_m)."""
        return A_cos[:, :, 0]

    def _apply_output(self, invariants, dist_ij):
        """output_net(inv) → per-edge energies.

        n_max_d=None: the readout emits a single number per edge, multiplied by
        the cutoff envelope f(r) so the per-edge energy still decays smoothly to
        0 at r_cut_edge (continuous energy/forces) without an explicit radial
        basis — i.e. energy_edge = MLP(inv) · f(r_ij). The n_max_d>=1 path
        instead dots the MLP output with the (cutoff-enveloped) radial basis."""
        if self.n_max_d is not None:
            rij_basis = radial_basis(dist_ij, self.r_cut_edge, self.n_max_d,
                                     cutoff_type=self.cutoff_type)
            return (self.output_net(invariants) * rij_basis).sum(-1)
        env = get_cutoff_fn(self.cutoff_type)(dist_ij, self.r_cut_edge)   # (n_e,) smooth → 0 at r_cut
        return self.output_net(invariants).squeeze(-1) * env


    def _run_equivariant_layers(self, A_cos, A_sin, **kwargs):
        """Run the equivariant layers, interleaving a message-passing layer
        between consecutive stages when n_mp >= 2 (n_mp-1 MP layers, no trailing MP)."""
        type_i   = kwargs.get('type_i')
        type_j   = kwargs.get('type_j')
        if self.n_mp == 1:
            # Plain model: a flat list of equivariant layers, no message passing.
            for layer in self.layers:
                A_cos, A_sin = layer(A_cos, A_sin)
            return A_cos, A_sin
        # Message-passing path: stage, MP, stage, MP, ..., stage  (MP only between stages).
        r_hat   = kwargs.get('r_hat')
        edge_i  = kwargs.get('edge_i')
        edge_j  = kwargs.get('edge_j')
        dist_ij = kwargs.get('dist_ij')
        n_atoms = kwargs.get('n_atoms')
        D_block = kwargs.get('D_block')
        for gi, stage in enumerate(self.layers):
            for layer in stage:
                A_cos, A_sin = layer(A_cos, A_sin)
            if gi < len(self.mp_layers):          # no MP after the final stage
                A_cos, A_sin = self.mp_layers[gi](
                    A_cos, A_sin, r_hat, dist_ij, edge_i, edge_j,
                    n_atoms, type_i, type_j,
                    D_block=D_block)
        return A_cos, A_sin

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, positions: torch.Tensor, types: torch.Tensor):
        """Compute total energy.

        Args:
            positions:         (n_atoms, 3)
            types:             (n_atoms,) int tensor of atom-type indices

        Returns:
            energy: scalar tensor
        """
        device, dtype = positions.device, positions.dtype

        # ── Edges ─────────────────────────────────────────────────────────
        edge_i_undir, edge_j_undir = find_edges(positions, self.r_cut_edge)
        if len(edge_i_undir) == 0:
            return torch.zeros(1, device=device, dtype=dtype).squeeze()

        edge_i = torch.cat([edge_i_undir, edge_j_undir])
        edge_j = torch.cat([edge_j_undir, edge_i_undir])

        diff_ij = positions[edge_j] - positions[edge_i]
        dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
        r_hat   = diff_ij / dist_ij.unsqueeze(-1)

        # ── Neighbor list ─────────────────────────────────────────────────
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-30)
        nb_mask = (dist_mat < self.r_cut_neighbor) & (dist_mat > 1e-10)
        nb_src, nb_dst = nb_mask.nonzero(as_tuple=True)

        # ── Step 1: ACE atomic basis ───────────────────────────────────────
        pos_batch = positions.unsqueeze(0)   # (1, N, 3)
        A_batch = self._compute_ace_basis(pos_batch, nb_src, nb_dst, types)
        A = A_batch.squeeze(0)               # (N, n_types, n_max, n_sph)

        # ── Step 2: Joint contraction → (N, embed_dim, n_sph) ────────────
        A_emb = self._embed(A, types)

        # ── Step 3: Gather + Wigner rotation ──────────────────────────────
        A_src = A_emb[edge_i]   # (n_edges, embed_dim, n_sph)
        A_tgt = A_emb[edge_j]
        A_both = torch.cat([A_src, A_tgt], dim=1)   # (n_edges, 2*embed_dim, n_sph)
        D_block = build_D_block(r_hat, self.l_max)
        A_rot = wigner_rotate(A_both, D_block)  # (n_edges, 2*embed_dim, n_sph)

        # ── Step 4: Reshape to A_cos / A_sin ──────────────────────────────
        A_cos, A_sin = self.sph_to_angular(A_rot)

        # ── Step 5: Equivariant layers ────────────────────────────────────
        ti, tj = types[edge_i], types[edge_j]
        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i, edge_j=edge_j,
            dist_ij=dist_ij, n_atoms=len(types),
            type_i=ti, type_j=tj, D_block=D_block)

        # ── Step 6+7: m=0 invariants → output_net → dot(rij_basis) ──────────
        invariants = self._contract(A_cos, A_sin)   # (n_edges, n_features_per_m)
        per_edge_energy = self._apply_output(invariants, dist_ij)
        return per_edge_energy.sum() + self.atomic_energy[types].sum()

    def forward_pbc(self, positions: torch.Tensor, types: torch.Tensor,
                    edge_i: torch.Tensor, edge_j: torch.Tensor,
                    shift_vecs_edge: torch.Tensor,
                    nb_src: torch.Tensor, nb_dst: torch.Tensor,
                    shift_vecs_nb: torch.Tensor):
        """Compute total energy with periodic boundary conditions.

        Args:
            positions:         (N, 3) atom positions in Cartesian Å (wrapped to unit cell)
            types:             (N,) int tensor of atom-type indices
            edge_i, edge_j:    (n_edges,) directed edge indices (both i→j and j→i)
            shift_vecs_edge:   (n_edges, 3) Cartesian PBC shift vectors for edges
            nb_src, nb_dst:    (n_nb,) directed neighbor pair indices
            shift_vecs_nb:     (n_nb, 3) Cartesian PBC shift vectors for neighbors

        Returns:
            energy: scalar tensor
        """
        device, dtype = positions.device, positions.dtype
        n_edges = len(edge_i)

        if n_edges == 0:
            return torch.zeros(1, device=device, dtype=dtype).squeeze()

        # ── Edges with PBC shifts ──────────────────────────────────────────
        diff_ij = (positions[edge_j] - positions[edge_i]
                   + shift_vecs_edge.to(dtype=dtype))
        dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
        r_hat   = diff_ij / dist_ij.unsqueeze(-1)

        # ── Step 1: ACE atomic basis with PBC neighbor shifts ─────────────
        pos_batch = positions.unsqueeze(0)   # (1, N, 3)
        A_batch = self._compute_ace_basis(pos_batch, nb_src, nb_dst, types,
                                          shift_vecs_nb=shift_vecs_nb)
        A = A_batch.squeeze(0)               # (N, n_types, n_max, n_sph)

        # ── Steps 2–7: identical to forward() ─────────────────────────────
        A_emb = self._embed(A, types)

        A_src  = A_emb[edge_i]
        A_tgt  = A_emb[edge_j]
        A_both = torch.cat([A_src, A_tgt], dim=1)
        D_block = build_D_block(r_hat, self.l_max)
        A_rot  = wigner_rotate(A_both, D_block)

        A_cos, A_sin = self.sph_to_angular(A_rot)

        ti, tj = types[edge_i], types[edge_j]
        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i, edge_j=edge_j,
            dist_ij=dist_ij, n_atoms=len(types),
            type_i=ti, type_j=tj, D_block=D_block)

        invariants = self._contract(A_cos, A_sin)
        per_edge_energy = self._apply_output(invariants, dist_ij)
        return per_edge_energy.sum() + self.atomic_energy[types].sum()

    def forward_batch_multi(self, positions_list, types_list):
        """Batch forward for variable-size, variable-composition structures.

        Topology is built per-structure in a cheap Python loop; the expensive
        ops (Wigner rotation, equivariant layers, output MLP) run once on the
        full flat edge set.

        Args:
            positions_list:  list of B tensors, each (N_b, 3)
            types_list:      list of B tensors, each (N_b,) of type indices

        Returns:
            energies: (B,) tensor
        """
        B = len(positions_list)
        device = positions_list[0].device
        dtype  = positions_list[0].dtype

        A_src_list, A_tgt_list = [], []
        r_hat_list, dist_ij_list = [], []
        type_i_list, type_j_list = [], []
        edge_i_list, edge_j_list = [], []   # flat atom indices with offsets (for MP)
        struct_ids = []
        atomic_e_list = []
        atom_offset = 0
        atom_counts = []   # N_b per structure, for slicing embeddings

        for b, (pos, types) in enumerate(zip(positions_list, types_list)):
            N_b = pos.shape[0]
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)              # (N_b, N_b, 3)
            dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-30)      # (N_b, N_b)

            ei, ej = ((dist_mat < self.r_cut_edge) & (dist_mat > 1e-10)).nonzero(as_tuple=True)

            atomic_e_list.append(self.atomic_energy[types].sum())
            if len(ei) == 0:
                atom_offset += N_b
                continue

            nb_src, nb_dst = ((dist_mat < self.r_cut_neighbor) & (dist_mat > 1e-10)).nonzero(as_tuple=True)

            diff_ij = pos[ej] - pos[ei]
            dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
            r_hat   = diff_ij / dist_ij.unsqueeze(-1)

            A = self._compute_ace_basis(pos.unsqueeze(0), nb_src, nb_dst, types).squeeze(0)
            A_emb = self._embed(A, types)   # (N_b, embed_dim, n_sph)

            A_src_list.append(A_emb[ei])
            A_tgt_list.append(A_emb[ej])
            r_hat_list.append(r_hat)
            dist_ij_list.append(dist_ij)
            type_i_list.append(types[ei])
            type_j_list.append(types[ej])
            edge_i_list.append(ei + atom_offset)
            edge_j_list.append(ej + atom_offset)
            struct_ids.append(torch.full((len(ei),), b, dtype=torch.long, device=device))
            atom_offset += N_b
            atom_counts.append(N_b)

        energies = torch.stack(atomic_e_list)   # (B,)

        total_edges = sum(len(x) for x in r_hat_list)
        if total_edges == 0:
            return energies

        # Merge flat edge arrays
        A_src      = torch.cat(A_src_list)
        A_tgt      = torch.cat(A_tgt_list)
        r_hat      = torch.cat(r_hat_list)
        dist_ij    = torch.cat(dist_ij_list)
        type_i     = torch.cat(type_i_list)
        type_j     = torch.cat(type_j_list)
        edge_i_flat = torch.cat(edge_i_list)
        edge_j_flat = torch.cat(edge_j_list)
        struct_idx  = torch.cat(struct_ids)

        A_both  = torch.cat([A_src, A_tgt], dim=1)
        D_block = build_D_block(r_hat, self.l_max)
        A_rot  = wigner_rotate(A_both, D_block)

        A_cos, A_sin = self.sph_to_angular(A_rot)

        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i_flat, edge_j=edge_j_flat,
            dist_ij=dist_ij, n_atoms=atom_offset,
            type_i=type_i, type_j=type_j, D_block=D_block)

        invariants = self._contract(A_cos, A_sin)
        per_edge_energy = self._apply_output(invariants, dist_ij)

        energies = energies + torch.zeros(B, dtype=dtype, device=device).scatter_add(
            0, struct_idx, per_edge_energy)

        return energies

    def forward_batch(self, positions_list, types, topology=None):
        """Compute energies for a batch of structures sharing the same atom types.

        Args:
            positions_list: list of B (N, 3) tensors
            types:          (N,) int tensor of atom-type indices (same for all structures)
            topology:       dict with precomputed 'edge_i', 'edge_j', 'nb_src', 'nb_dst'
                            (and optionally 'shift_vecs_edge', 'shift_vecs_nb' for PBC)
                            for the fixed-topology (same molecule) case, or None to
                            fall back to per-structure self.forward calls.

        Returns:
            energies: (B,) tensor
        """
        if not isinstance(topology, dict):
            # Variable-topology fallback: forward_batch_multi subsumes this case
            # (shared types is just every structure carrying the same type
            # vector). It builds topology per-structure and runs the expensive
            # ops once on the merged flat edge set — identical result.
            return self.forward_batch_multi(
                positions_list, [types] * len(positions_list))

        # ── Fixed topology: vectorized over B ─────────────────────────────
        B = len(positions_list)
        edge_i = topology['edge_i']
        edge_j = topology['edge_j']
        nb_src = topology['nb_src']
        nb_dst = topology['nb_dst']
        shift_vecs_edge = topology.get('shift_vecs_edge', None)
        shift_vecs_nb   = topology.get('shift_vecs_nb',   None)
        n_edges = edge_i.shape[0]

        pos_batch = torch.stack(positions_list)  # (B, N, 3)

        # ── Edges ────────────────────────────────────────────────────────
        diff_ij = pos_batch[:, edge_j] - pos_batch[:, edge_i]  # (B, n_edges, 3)
        if shift_vecs_edge is not None:
            diff_ij = diff_ij + shift_vecs_edge[None].to(dtype=pos_batch.dtype)
        dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)   # (B, n_edges)
        r_hat   = diff_ij / dist_ij.unsqueeze(-1)               # (B, n_edges, 3)

        # ── Step 1: ACE atomic basis (B, N, n_types, n_max, n_sph) ──────
        A_batch = self._compute_ace_basis(pos_batch, nb_src, nb_dst, types, shift_vecs_nb)

        # ── Step 2: Joint contraction → (B, N, embed_dim, n_sph) ────────
        W_i   = self.W[types]  # (N, n_types, n_max, embed_dim)
        A_emb = torch.einsum('bitns,itnc->bics', A_batch, W_i)

        # ── Step 3: Gather + Wigner rotation (flatten B*n_edges) ────────
        type_i = types[edge_i]   # (n_edges,)
        type_j = types[edge_j]
        A_src  = A_emb[:, edge_i]                                # (B, n_edges, embed_dim, n_sph)
        A_tgt  = A_emb[:, edge_j]
        A_both = torch.cat([A_src, A_tgt], dim=2)               # (B, n_edges, 2*embed_dim, n_sph)

        r_hat_flat  = r_hat.reshape(B * n_edges, 3)
        A_both_flat = A_both.reshape(B * n_edges, 2 * self.embed_dim, self.n_sph)
        D_block = build_D_block(r_hat_flat, self.l_max)
        A_rot_flat  = wigner_rotate(A_both_flat, D_block)

        # ── Step 4: Reshape to A_cos / A_sin ─────────────────────────────
        A_cos_flat, A_sin_flat = self.sph_to_angular(A_rot_flat)
        # shapes: (B*n_edges, n_features_per_m, n_angular)

        # ── Step 5: Equivariant layers ────────────────────────────────────
        # For batched MP: offset edge indices so scatter targets B*N atoms
        N = pos_batch.shape[1]
        offset = torch.arange(B, device=edge_i.device).repeat_interleave(n_edges) * N
        edge_i_flat = edge_i.repeat(B) + offset
        edge_j_flat = edge_j.repeat(B) + offset
        type_i_flat = type_i.repeat(B)
        type_j_flat = type_j.repeat(B)

        A_cos_flat, A_sin_flat = self._run_equivariant_layers(
            A_cos_flat, A_sin_flat,
            r_hat=r_hat_flat, edge_i=edge_i_flat, edge_j=edge_j_flat,
            dist_ij=dist_ij.reshape(B * n_edges), n_atoms=B * N,
            type_i=type_i_flat, type_j=type_j_flat, D_block=D_block)

        # ── Step 6+7: m=0 invariants → output_net → dot(rij_basis) ──────────
        invariants = self._contract(A_cos_flat, A_sin_flat)      # (B*n_edges, n_features_per_m)
        per_edge_energy = self._apply_output(invariants, dist_ij.reshape(B * n_edges))  # (B*n_edges,)
        energies = per_edge_energy.reshape(B, n_edges).sum(dim=1)        # (B,)
        energies = energies + self.atomic_energy[types].sum()

        return energies


# ---------------------------------------------------------------------------
# Equivariant layer: Linear → RealSpaceNonlinearity → residual
# ---------------------------------------------------------------------------


class ECENetLayer(nn.Module):
    """One equivariant layer: EquivariantLinear → nonlinearity.

    Without bottleneck: linear(n_ch → n_ch) → nonlin(n_ch) → residual.
    With bottleneck:    linear_down(n_ch → r) → nonlin(r) → linear_up(r → n_ch) → residual.

    The bottleneck is a low-rank update: the nonlinearity runs at the (smaller)
    bottleneck width r, and the up-projection is zero-init so the whole block is
    identity at init (the residual carries the input through unchanged).

    Args:
        n_features:        number of feature channels (= n_features_per_m)
        m_max:             maximum angular frequency (= l_max)
        activation:        pointwise activation (used by the realspace nonlinearity)
        use_nonlinearity:  if False, skip nonlinearity entirely (linear-only layer)
        bottleneck_dim:    if set, use the down → nonlin(r) → up bottleneck structure
                           (low-rank); None → full-width linear → nonlin
    """

    def __init__(self, n_features: int, m_max: int, activation: str = 'silu',
                 use_nonlinearity: bool = True, n_grid: int = None,
                 bottleneck_dim: int = None):
        super().__init__()
        n_angular = m_max + 1
        self.bottleneck_dim = bottleneck_dim
        # nonlin_features: dimension at which the nonlinearity operates — the
        # bottleneck width r when bottlenecking, else the full feature width.
        nonlin_features = bottleneck_dim if bottleneck_dim is not None else n_features

        if bottleneck_dim is not None:
            self.linear_down = EquivariantLinear(n_features, bottleneck_dim, n_angular, m_max)
            self.linear_up   = EquivariantLinear(bottleneck_dim, n_features, n_angular, m_max)
            # Zero-init the up-projection → bottleneck starts as identity via residual.
            nn.init.zeros_(self.linear_up.weights)
            nn.init.zeros_(self.linear_up.bias)
        else:
            self.linear = EquivariantLinear(n_features, n_features, n_angular, m_max)

        self.nonlin = None
        if use_nonlinearity:
            self.nonlin = RealSpaceNonlinearity(nonlin_features, m_max, n_grid=n_grid,
                                                activation=activation)
        self.use_nonlinearity = self.nonlin is not None

    def forward(self, A_cos, A_sin):
        A_cos_in, A_sin_in = A_cos, A_sin

        # (Down-)linear: project to the bottleneck width r, else stay full width.
        if self.bottleneck_dim is not None:
            A_cos, A_sin = self.linear_down(A_cos, A_sin)
        else:
            A_cos, A_sin = self.linear(A_cos, A_sin)
        if self.nonlin is not None:
            A_cos, A_sin = self.nonlin(A_cos, A_sin)
        # Up-projection back to the full feature width.
        if self.bottleneck_dim is not None:
            A_cos, A_sin = self.linear_up(A_cos, A_sin)

        return A_cos + A_cos_in, A_sin + A_sin_in


# ---------------------------------------------------------------------------
# Message passing layer
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _sph_pack_index(l_max, m_max, n_angular, n_sph, device):
    """Precompute gather indices + validity masks for the angular↔SH pack/unpack.

    The (l, m) → SH-slot mapping is data-independent (fixed by l_max/m_max), so
    the per-l slice-assign+flip loop is exactly a fixed gather.

    Returns long index + bool mask tensors on `device` (cached per shape/device):
      pack_*   (n_sph,)            — gather source in the flat (lp1*n_angular) grid
      unpack_* (lp1*n_angular,)    — gather source in the (n_sph) SH grid
    cos = m≥0 modes, sin = m<0 modes (mirrors the flip). Masks zero the slots the
    loop left untouched (|m|>m_out when m_max<l). Bit-identical to the loop.
    """
    lp1 = l_max + 1
    pack_c  = torch.zeros(n_sph, dtype=torch.long)
    pack_s  = torch.zeros(n_sph, dtype=torch.long)
    pack_cm = torch.zeros(n_sph, dtype=torch.bool)
    pack_sm = torch.zeros(n_sph, dtype=torch.bool)
    up_c  = torch.zeros(lp1 * n_angular, dtype=torch.long)
    up_s  = torch.zeros(lp1 * n_angular, dtype=torch.long)
    up_cm = torch.zeros(lp1 * n_angular, dtype=torch.bool)
    up_sm = torch.zeros(lp1 * n_angular, dtype=torch.bool)
    for l in range(lp1):
        m_out = min(l, m_max)
        for m in range(0, m_out + 1):          # cos: SH slot l²+l+m ← angular (l, m)
            s = l * l + l + m
            pack_c[s] = l * n_angular + m
            pack_cm[s] = True
            up_c[l * n_angular + m] = s
            up_cm[l * n_angular + m] = True
        for k in range(1, m_out + 1):          # sin: SH slot l²+l-k ← angular (l, k)
            s = l * l + l - k
            pack_s[s] = l * n_angular + k
            pack_sm[s] = True
            up_s[l * n_angular + k] = s
            up_sm[l * n_angular + k] = True
    dev = torch.device(device)
    return tuple(t.to(dev) for t in (pack_c, pack_cm, pack_s, pack_sm,
                                     up_c, up_cm, up_s, up_sm))


def _pack_angular_to_sph(A_cos, A_sin, n_base, l_max, m_max, n_angular, n_sph):
    """(n_e, n_base*lp1, n_angular) → SH (n_e, n_base, n_sph). Vectorized pack."""
    n_e = A_cos.shape[0]
    lp1 = l_max + 1
    pc, pcm, ps, psm, *_ = _sph_pack_index(l_max, m_max, n_angular, n_sph, A_cos.device)
    ac  = A_cos.reshape(n_e, n_base, lp1 * n_angular)
    asn = A_sin.reshape(n_e, n_base, lp1 * n_angular)
    ic  = pc.view(1, 1, n_sph).expand(n_e, n_base, n_sph)
    isn = ps.view(1, 1, n_sph).expand(n_e, n_base, n_sph)
    return (ac.gather(2, ic) * pcm.to(A_cos.dtype)
            + asn.gather(2, isn) * psm.to(A_cos.dtype))


def _unpack_sph_to_angular(v_rot, n_base, l_max, m_max, n_angular, n_sph):
    """SH (n_e, n_base, n_sph) → cos/sin (n_e, n_base, lp1, n_angular). Vectorized."""
    n_e = v_rot.shape[0]
    lp1 = l_max + 1
    _, _, _, _, uc, ucm, us, usm = _sph_pack_index(l_max, m_max, n_angular, n_sph,
                                                   v_rot.device)
    L = lp1 * n_angular
    ic  = uc.view(1, 1, L).expand(n_e, n_base, L)
    isn = us.view(1, 1, L).expand(n_e, n_base, L)
    d_cos = (v_rot.gather(2, ic)  * ucm.to(v_rot.dtype)).view(n_e, n_base, lp1, n_angular)
    d_sin = (v_rot.gather(2, isn) * usm.to(v_rot.dtype)).view(n_e, n_base, lp1, n_angular)
    return d_cos, d_sin


class ECENetTransformerMPLayer(nn.Module):
    """Attention-style message passing for ECENet.

    Per edge (i→j):
      * a low-rank residual *message* m_e (equivariant, edge frame), and
      * an invariant scalar *score* s_e (down → nonlin → m=0 → scalar).
    Messages are unrotated to the common global frame and aggregated at each
    receiver atom as a score-weighted combination of that atom's incoming edges.
    The result is rotated back into each edge's bond frame, passed through a
    low-rank residual *receiver* transform, and added to the features.

    ``aggregation`` selects how the per-edge weight is formed:

    ``'transformer'`` — softmax over the receiver's incoming edges, with the
    smooth cutoff envelope folded in as a multiplicative log-bias,

        a_e = exp(s_e)·f_cut(r_ij) / (Σ_{e'→j} exp(s_e')·f_cut(r_e') + eps),

    so a departing edge's weight vanishes continuously as it crosses r_cut_edge
    (it leaves numerator and normalizer together — no jump), and the +eps floor
    keeps a node's aggregate finite/continuous as its last edge leaves. Because
    the aggregation is a normalized weighted average it is *intensive* in
    coordination.

    ``'sum'`` — the raw signed score times the same envelope, summed:

        a_e = s_e·f_cut(r_ij).

    There is no normalizer, so the aggregate is *extensive* in coordination and
    the envelope is what carries the continuity at r_cut (it is no longer divided
    back out, so the message also decays with absolute distance). Weights are
    signed, so a neighbour can contribute negatively. The score read-out is
    zero-init, which makes s_e = 0 and hence the whole layer an exact no-op at
    initialisation — the softmax path cannot do this, since exp(0) = 1.

    With ``n_heads > 1`` the layer is multi-head: the score head emits ``n_heads``
    invariant scores per edge (one shared trunk, ``n_heads`` linear read-outs) and
    the message's value channels (``n_base``) are split into ``n_heads``
    contiguous groups — whole spherical channels, full ``n_sph`` each. Head h
    weights the receiver's in-edges independently and gates *its own* value slice,
    so a neighbour can matter for one feature subspace and be suppressed for
    another (the per-subspace routing a single scalar can't represent). All splits
    are along channels (never within ``n_sph`` / across ``m``, which would break
    rotation-invariance); ``n_base = 2·embed_dim`` must be divisible by ``n_heads``.

    Message and scores come out of ONE fused trunk: down → nonlin at ``mp_dim`` →
    up, where the up-projection emits ``n_ch + n_heads`` channels. The first
    ``n_ch`` are the message (added residually to the input), and the m=0
    components of the trailing ``n_heads`` channels are the per-head scores. Since
    score and message share a trunk there is no dedicated score head, which makes
    this cheaper than computing the two separately. The up-projection is zero-init,
    so at initialisation the message residual is 0 and every score is 0 — which for
    ``'sum'`` means the layer is an exact identity, and for ``'transformer'`` means
    attention starts uniform (exp(0) = 1) over each receiver's in-edges.

    Equivariance: the trunk and receiver are EquivariantLinear +
    RealSpaceNonlinearity in a bond frame; the per-head scores (m=0 channel) and
    the cutoff (a function of the invariant distance) make the per-head weights
    rotation-invariant (for the softmax, its per-node normalizer is a sum of
    invariant scalars); the cross-edge sum happens in the common global frame via
    the Wigner-D unrotate/rotate.

    The receiver is an ``ECENetLayer`` bottleneck block (down → nonlin at
    ``mp_dim`` → up, zero-init up so it is identity at init).
    """

    def __init__(self, n_features_per_m: int, l_max: int, embed_dim: int,
                 n_types: int, r_cut: float = 5.0,
                 cutoff_type: str = 'cosine', m_max: int = None,
                 mp_dim: int = None,
                 activation: str = 'silu', n_grid: int = None, n_heads: int = 1,
                 aggregation: str = 'transformer'):
        super().__init__()
        if aggregation not in ('transformer', 'sum'):
            raise ValueError(
                f"Unknown aggregation '{aggregation}' (expected 'transformer' or 'sum').")
        self.aggregation = aggregation
        self.l_max     = l_max
        self.n_sph     = (l_max + 1) ** 2
        self.m_max     = m_max if m_max is not None else l_max
        self.n_angular = self.m_max + 1
        self.n_ch      = n_features_per_m
        self.n_base    = n_features_per_m // (l_max + 1)
        # Multi-head attention: each head emits its own score → its own weights
        # over the receiver's in-edges → gates its own contiguous slice of the
        # value channels (whole spherical channels, full n_sph each — NEVER a
        # split within n_sph / across m, which would break rotation-invariance).
        # So n_base = 2·embed_dim must be divisible by n_heads (the value split).
        if self.n_base % n_heads != 0:
            raise ValueError(
                f"{aggregation} MP: n_base (=2·embed_dim={self.n_base}) must be "
                f"divisible by n_heads ({n_heads}) to split the value across heads.")
        self.n_heads = n_heads
        self.n_scores = n_heads          # one invariant score per head
        # +eps floor on the per-node softmax normalizer: keeps it finite (no 0/0 →
        # NaN when a node's last edge reaches r_cut and every num → 0 together).
        self.softmax_eps = 1e-6
        n_ch = n_features_per_m
        mp_dim = mp_dim if mp_dim is not None else max(n_ch // 4, 1)

        # Receiver: low-rank residual block (down → nonlin(mp_dim) → up).
        self.receiver = ECENetLayer(n_ch, self.m_max, activation=activation,
                                    n_grid=n_grid, bottleneck_dim=mp_dim)
        # Fused message + score trunk: ONE low-rank trunk (down → nonlin → up)
        # whose up-projection emits n_ch message channels PLUS n_scores score
        # channels; the m=0 invariants of the extra channels are the per-head
        # scores. This replaces a separate message block *and* a separate score
        # head, so it is cheaper than computing the two independently. up is
        # zero-init → message residual = 0 and scores = 0 at init.
        self.msg_down   = EquivariantLinear(n_ch, mp_dim, self.n_angular, self.m_max)
        self.msg_nonlin = RealSpaceNonlinearity(mp_dim, self.m_max, n_grid=n_grid,
                                                activation=activation)
        self.msg_up     = EquivariantLinear(mp_dim, n_ch + self.n_scores,
                                            self.n_angular, self.m_max)
        nn.init.zeros_(self.msg_up.weights)
        nn.init.zeros_(self.msg_up.bias)

        # Smooth cutoff envelope for the per-edge weight (→ 0 at r_cut_edge).
        self.r_cut = r_cut
        self.cutoff_fn = get_cutoff_fn(cutoff_type)

    def _pack(self, A_cos, A_sin):
        """(n_e, n_ch, n_angular) → SH (n_e, n_base, n_sph)."""
        return _pack_angular_to_sph(A_cos, A_sin, self.n_base, self.l_max,
                                    self.m_max, self.n_angular, self.n_sph)

    def _unpack(self, v_rot, n_e):
        """SH (n_e, n_base, n_sph) → (n_e, n_ch, n_angular)."""
        d_cos, d_sin = _unpack_sph_to_angular(
            v_rot, self.n_base, self.l_max, self.m_max, self.n_angular, self.n_sph)
        return (d_cos.reshape(n_e, self.n_ch, self.n_angular),
                d_sin.reshape(n_e, self.n_ch, self.n_angular))

    def forward(self, A_cos, A_sin, r_hat, dist_ij, edge_i, edge_j,
                n_atoms, type_i, type_j, D_block=None):
        n_e = A_cos.shape[0]
        device, dtype = A_cos.device, A_cos.dtype
        if D_block is None:
            D_block = build_D_block(r_hat, self.l_max)

        # 1+2. Fused trunk: down → nonlin → up(n_ch + n_scores). The first n_ch
        #      channels are the message (added residually, so the block is a
        #      low-rank update); the m=0 components of the trailing n_scores
        #      channels are the per-head scores (invariant, hence equivariance-safe).
        u_cos, u_sin = self.msg_down(A_cos, A_sin)
        u_cos, u_sin = self.msg_nonlin(u_cos, u_sin)
        u_cos, u_sin = self.msg_up(u_cos, u_sin)          # (n_e, n_ch+n_scores, n_angular)
        m_cos = u_cos[:, :self.n_ch] + A_cos
        m_sin = u_sin[:, :self.n_ch] + A_sin
        s = u_cos[:, self.n_ch:self.n_ch + self.n_scores, 0]   # (n_e, n_heads)

        # 3. Pack message → SH, unrotate to the common global frame.
        h = self._pack(m_cos, m_sin)
        h_global = torch.bmm(h, D_block.transpose(-1, -2))  # transposed view, no copy

        # 4. Per-edge weights, formed INDEPENDENTLY per head. Either way the smooth
        #    cutoff f_cut enters multiplicatively so a departing edge's weight
        #    vanishes continuously as it crosses r_cut, and every factor is an
        #    invariant scalar, so SO(3)-equivariance is preserved.
        H = self.n_heads
        f_cut = self.cutoff_fn(dist_ij, self.r_cut)          # (n_e,) smooth → 0 at r_cut
        if self.aggregation == 'sum':
            #    a_e^k = s_e^k·f_cut_e — a plain signed weighted sum. No normalizer,
            #    so the aggregate is extensive in coordination and decays with
            #    absolute distance (f_cut is not divided back out).
            a = s * f_cut[:, None]                           # (n_e, H)
        else:
            #    Segment-softmax over the edges arriving at each receiver atom j,
            #    with the cutoff as a multiplicative log-bias on exp(s):
            #        a_e^k = exp(s_e^k)·f_cut_e / (Σ_{e'→j} exp(s_e'^k)·f_cut_e' + eps)
            #    A normalized weighted average → the aggregation is intensive in
            #    coordination. The +eps floor keeps it finite and continuous as a
            #    node's last edge leaves (all f_cut → 0 ⇒ Delta → 0, no jump).
            #    Per-(receiver, head) max-subtraction for numerical stability
            #    (detached, so this stays an exact softmax — invariant to the
            #    constant per-node shift). Receivers with no incoming edge keep
            #    -inf but are never gathered (every edge has a receiver in edge_j).
            ej_k = edge_j[:, None].expand(-1, H)             # (n_e, H)
            s_max = torch.full((n_atoms, H), float('-inf'), device=device, dtype=dtype
                               ).scatter_reduce(0, ej_k, s.detach(), reduce='amax',
                                                include_self=True)
            num = torch.exp(s - s_max[edge_j]) * f_cut[:, None]   # (n_e, H)
            denom = torch.zeros(n_atoms, H, device=device, dtype=dtype).scatter_add(0, ej_k, num)
            a = num / (denom[edge_j] + self.softmax_eps)     # (n_e, H) softmax weights

        # 5. Weighted aggregation to receiver atoms. The value channels (n_base)
        #    split into H contiguous groups (full n_sph each); head h's weight gates
        #    head h's value slice, uniformly across all m (equivariant). Heads then
        #    concat back to n_base.
        hb = self.n_base // H
        contrib = h_global.reshape(n_e, H, hb, self.n_sph) * a[:, :, None, None]
        idx = edge_j[:, None, None, None].expand_as(contrib)
        Delta = torch.zeros(n_atoms, H, hb, self.n_sph, device=device, dtype=dtype
                            ).scatter_add(0, idx, contrib).reshape(n_atoms, self.n_base, self.n_sph)

        # 6. Gather to edges (source atom), rotate back to the edge frame.
        v_rot = torch.bmm(Delta[edge_i], D_block)            # (n_e, n_base, n_sph)
        d_cos, d_sin = self._unpack(v_rot, n_e)

        # 7. Receiver: low-rank residual (equivariant, edge frame).
        r_cos, r_sin = self.receiver(d_cos, d_sin)

        return A_cos + r_cos, A_sin + r_sin


# ---------------------------------------------------------------------------
# SH → A_cos / A_sin reshape
# ---------------------------------------------------------------------------


class SphToAngular(nn.Module):
    """Convert rotated features (n_edges, 2*embed_dim, n_sph) to A_cos/A_sin.

    Reshapes n_sph = (l_max+1)² into an (l_max+1, 2*l_max+1) block indexed by
    (l, m), zero-padded where |m| > l, then merges the l axis into the channel
    dimension and separates m into cos (m>=0) and sin (m<0) components.

    Output shape: (n_edges, 2*embed_dim*(l_max+1), l_max+1)
      channel layout: [(side=0, embed=0, l=0), (side=0, embed=0, l=1), ...,
                       (side=0, embed=1, l=0), ..., (side=1, embed=embed_dim-1, l=l_max)]
      angular mode m = 0..l_max (the azimuthal frequency |m|).

    The triangular zero structure (|m| > l → 0) is preserved naturally.
    """

    def __init__(self, embed_dim: int, l_max: int, m_max: int = None):
        super().__init__()
        self.l_max = l_max
        m_max = m_max if m_max is not None else l_max
        self.m_max = m_max
        self.n_angular = m_max + 1          # m = 0..m_max (may be < l_max+1)
        self.n_ch = 2 * embed_dim * (l_max + 1)   # (side, embed, l) channels

        n_ch_base = 2 * embed_dim           # channels before l expansion

        # For each (embed_channel, l) and each m = 0..m_max, store the flat SH index
        # +m → index l²+l+m,  −m → index l²+l-m.
        # Channels with l < m have no valid component → index 0, masked to 0.
        # Only m=0..m_max are included; higher modes are discarded.
        cos_idx = torch.zeros(self.n_ch, self.n_angular, dtype=torch.long)
        sin_idx = torch.zeros(self.n_ch, self.n_angular, dtype=torch.long)
        cos_valid = torch.zeros(self.n_ch, self.n_angular)
        sin_valid = torch.zeros(self.n_ch, self.n_angular)

        c = 0
        for _ in range(n_ch_base):          # one entry per (side, embed_channel)
            for l in range(l_max + 1):
                base = l * l + l            # index of m=0 for this l
                for m in range(self.n_angular):
                    if m <= l:
                        cos_idx[c, m] = base + m    # +m component
                        cos_valid[c, m] = 1.0
                        if m > 0:
                            sin_idx[c, m] = base - m  # −m component
                            sin_valid[c, m] = 1.0
                c += 1

        self.register_buffer('cos_idx', cos_idx)
        self.register_buffer('sin_idx', sin_idx)
        self.register_buffer('cos_valid', cos_valid)
        self.register_buffer('sin_valid', sin_valid)

    def forward(self, A_rot):
        """
        Args:
            A_rot: (n_edges, 2*embed_dim, n_sph)
        Returns:
            A_cos, A_sin: (n_edges, 2*embed_dim*(l_max+1), l_max+1)
        """
        n_edges = A_rot.shape[0]
        # Repeat each embed channel l_max+1 times to align with (embed, l) layout
        A_exp = A_rot.repeat_interleave(self.l_max + 1, dim=1)  # (n_edges, n_ch, n_sph)
        # Gather cos (+m) and sin (−m) components
        A_cos = A_exp.gather(2, self.cos_idx[None].expand(n_edges, -1, -1)) * self.cos_valid
        A_sin = A_exp.gather(2, self.sin_idx[None].expand(n_edges, -1, -1)) * self.sin_valid
        return A_cos, A_sin


# ---------------------------------------------------------------------------
# Output MLP
# ---------------------------------------------------------------------------


class OutputMLP(nn.Module):
    """Plain MLP readout over the per-edge invariants.

    Weights use fan-avg Gaussian init; the last layer is near-zero initialised
    so per-edge energies start close to 0 and the atomic-energy baseline
    dominates early training.
    """

    def __init__(self, dims: list, activation: nn.Module, zero_init_last: bool = True):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        for lin in self.linears:
            std = (2.0 / (lin.in_features + lin.out_features)) ** 0.5
            nn.init.normal_(lin.weight, std=std)
            nn.init.zeros_(lin.bias)
        self.activation = activation
        if zero_init_last:
            nn.init.normal_(self.linears[-1].weight, std=0.01)
            nn.init.zeros_(self.linears[-1].bias)

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = self.activation(x)
        return x

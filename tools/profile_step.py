"""Profile individual components of a single MLIP calculator step.

Usage (run from the repo root):
    python tools/profile_step.py --checkpoint spice.mdl --box water_box.xyz
    python tools/profile_step.py --checkpoint spice.mdl --box water_box.xyz --float32
"""
import argparse
import os
import sys
import time

import torch
from ase.io import read
from ase.neighborlist import neighbor_list as ase_neighbor_list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root → import ecenet
from ecenet.calculator import load_calculator
from ecenet.radial import radial_basis
from ecenet.spherical import build_D_block_from_list, recursive_wigner_D, wigner_rotate

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--box',        required=True)
parser.add_argument('--frame_idx',  type=int, default=-1)
parser.add_argument('--float32',    action='store_true')
parser.add_argument('--tf32',       action='store_true',
                    help='Enable TF32 for float32 matmuls (Ampere+ GPUs). No effect '
                         'unless --float32 (TF32 is a float32-only mode).')
parser.add_argument('--n_warmup',   type=int, default=3)
parser.add_argument('--n_time',     type=int, default=10)
parser.add_argument('--fuse_nonlin', action='store_true',
                    help='Enable the fused recompute-in-backward RealSpaceNonlinearity '
                         '(set_activation_fused; Triton on CUDA+silu, else PyTorch).')
parser.add_argument('--edge_frame_fused', action='store_true',
                    help='Enable the fused gather+Wigner-rotate+reshape edge-frame op '
                         '(set_edge_frame_fused; Triton on CUDA+float32, else eager '
                         'recompute-in-backward).')
parser.add_argument('--edge_frame_e2n', action='store_true',
                    help="With --edge_frame_fused: ALSO fuse the MP layers' "
                         'pack+unrotate (their step 3).')
args = parser.parse_args()

dtype  = torch.float32 if args.float32 else torch.float64
if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    if dtype == torch.float64:
        print("[tf32] requested but dtype=float64 → no effect (TF32 is float32-only); "
              "add --float32 to use it")
atoms  = read(args.box, index=args.frame_idx)
atoms.set_pbc(True)

# load_calculator dispatches: a joint-LES checkpoint yields the LES-aware
# calculator with the module already materialised and loaded, so the E_lr
# cost is profiled too (flags read off the model — the single source).
calc   = load_calculator(args.checkpoint, dtype=dtype)
model  = calc.model
device = calc.device

les_module = getattr(calc, 'les_module', None)
if les_module is not None:
    les_flags = model.les_flags
    print(f"[les] checkpoint carries LES (readout={model.les_readout}"
          f"{', dipole' if model.les_dipole else ''}) — profiling E_lr too")

if args.fuse_nonlin:
    model.set_activation_fused(True)
    n_set = sum(1 for m in model.modules() if getattr(m, 'fused', False))
    print(f"[fuse_nonlin] fused RealSpaceNonlinearity on {n_set} module(s)")

if args.edge_frame_fused:
    from ecenet.edge_frame_kernel import _HAS_TRITON as _ef_triton
    model.set_edge_frame_fused(True, e2n=args.edge_frame_e2n)
    path = ('Triton' if (_ef_triton and device.type == 'cuda') else 'eager recompute')
    e2n = 'on' if args.edge_frame_e2n else 'off'
    print(f"[edge_frame_fused] fused gather+rotate+reshape ({path}, e2n {e2n})")

# ── Utilities ─────────────────────────────────────────────────────────────────

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

def time_fn(label, fn, n=args.n_time, indent=2):
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        result = fn()
    sync()
    ms = (time.perf_counter() - t0) / n * 1000
    pad = ' ' * indent
    print(f"{pad}{label:<44s} {ms:8.2f} ms")
    return ms, result

def sep(title=''):
    if title:
        print(f"\n── {title} {'─'*(54-len(title))}")
    else:
        print()

# ── Warm up ───────────────────────────────────────────────────────────────────
print(f"Warming up ({args.n_warmup} steps)...")
atoms.calc = calc
for _ in range(args.n_warmup):
    atoms.get_potential_energy()
    atoms.get_forces()

# ── Setup inputs ──────────────────────────────────────────────────────────────
symbols      = atoms.get_chemical_symbols()
positions_np = atoms.get_positions()
cell         = atoms.get_cell().array

types = torch.tensor([calc.element_to_type[s] for s in symbols],
                     dtype=torch.long, device=device)
pos_d = torch.tensor(positions_np, dtype=dtype, device=device)

print(f"\nSystem: {len(atoms)} atoms, "
      f"cell={atoms.cell.lengths().round(2)}, dtype={dtype}\n")

# ── 1. Neighbor list ──────────────────────────────────────────────────────────
sep("Neighbor list")
time_fn("ASE (CPU)",
        lambda: (ase_neighbor_list('ijS', atoms, model.r_cut_edge),
                 ase_neighbor_list('ijS', atoms, model.r_cut_neighbor)))
time_fn("GPU O(N²)",
        lambda: (calc._gpu_neighbor_list(pos_d, cell, model.r_cut_edge),
                 calc._gpu_neighbor_list(pos_d, cell, model.r_cut_neighbor)))

edge_i, edge_j, shift_e = calc._gpu_neighbor_list(pos_d, cell, model.r_cut_edge)
nb_src, nb_dst, shift_n  = calc._gpu_neighbor_list(pos_d, cell, model.r_cut_neighbor)

print(f"  edges: {len(edge_i):,}   neighbors: {len(nb_src):,}")

# Fresh pos with grad for all model timing
def fresh_pos():
    return pos_d.clone().requires_grad_(True)

p = fresh_pos()

# ── 2. Forward sub-components ─────────────────────────────────────────────────
sep("Forward pass sub-components")

# Geometry
diff_ij = pos_d[edge_j] - pos_d[edge_i] + shift_e
dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
r_hat   = diff_ij / dist_ij.unsqueeze(-1)

time_fn("edge geometry (diff, dist, r_hat)",
        lambda: (pos_d[edge_j] - pos_d[edge_i] + shift_e,))

# ACE basis
time_fn("ACE basis (_compute_ace_basis)",
        lambda: model._compute_ace_basis(
            pos_d.unsqueeze(0), nb_src, nb_dst, types, shift_vecs_nb=shift_n))

A_batch = model._compute_ace_basis(
    pos_d.unsqueeze(0), nb_src, nb_dst, types, shift_vecs_nb=shift_n)
A = A_batch.squeeze(0)

# Embed
time_fn("embed (einsum A×W → A_emb)",
        lambda: model._embed(A, types))

A_emb = model._embed(A, types)
A_both = torch.cat([A_emb[edge_i], A_emb[edge_j]], dim=1)

# Wigner rotation breakdown
_, D_list = time_fn("  Wigner D (recursive_wigner_D)",
        lambda: recursive_wigner_D(r_hat, model.l_max))
_, D_block_main = time_fn("  Wigner D (build_D_block_from_list)",
        lambda: build_D_block_from_list(D_list, len(r_hat), model.l_max,
                                        r_hat.device, r_hat.dtype))
time_fn("  Wigner bmm (A_both @ D)",
        lambda: torch.bmm(A_both, D_block_main))
time_fn("Wigner rotate total (main, w/ cached D)",
        lambda: wigner_rotate(A_both, D_block_main))

A_rot = wigner_rotate(A_both, D_block_main)

# sph_to_angular
time_fn("sph_to_angular (repeat_interleave + gather)",
        lambda: model.sph_to_angular(A_rot))

A_cos, A_sin = model.sph_to_angular(A_rot)

type_i = types[edge_i]
type_j = types[edge_j]

# Equivariant layers + message passing.
# n_mp >= 2: model.layers is a list of stages with an MP layer between
# consecutive stages (n_mp-1 MP layers, none after the final stage).
# n_mp = 1: model.layers is a flat list of layers and mp_layers does not exist.
mp_layers = list(getattr(model, 'mp_layers', []))
stages = list(model.layers) if mp_layers else [list(model.layers)]

for gi, layer_group in enumerate(stages):
    sep(f"Layer stage {gi+1}/{len(stages)}")
    for li, layer in enumerate(layer_group):
        # Low-rank (bottleneck_dim) layers have linear_down/linear_up instead
        # of a single full-width linear; the nonlinearity runs at the
        # bottleneck width, between the two.
        if layer.bottleneck_dim is not None:
            time_fn(f"  EquivariantLinear down [{gi},{li}]",
                    lambda l=layer: l.linear_down(A_cos, A_sin))
            lin_out = layer.linear_down(A_cos, A_sin)
            if layer.use_nonlinearity:
                time_fn(f"  RealSpaceNonlinearity [{gi},{li}]",
                        lambda l=layer, c=lin_out: l.nonlin(*c))
            nl_out = layer.nonlin(*lin_out) if layer.use_nonlinearity else lin_out
            time_fn(f"  EquivariantLinear up [{gi},{li}]",
                    lambda l=layer, c=nl_out: l.linear_up(*c))
        else:
            time_fn(f"  EquivariantLinear [{gi},{li}]",
                    lambda l=layer: l.linear(A_cos, A_sin))
            lin_out = layer.linear(A_cos, A_sin)
            if layer.use_nonlinearity:
                time_fn(f"  RealSpaceNonlinearity [{gi},{li}]",
                        lambda l=layer, c=lin_out: l.nonlin(*c))
        A_cos, A_sin = layer(A_cos, A_sin)

    if gi < len(mp_layers):
        mp = mp_layers[gi]
        sep(f"MP layer {gi+1} ({mp.aggregation})")
        # Message passing internals (message + score → weights → aggregate → receiver)
        n_e = len(edge_i)
        n_atoms = len(types)
        H = mp.n_heads

        def _trunk():
            u_cos, u_sin = mp.msg_down(A_cos, A_sin)
            u_cos, u_sin = mp.msg_nonlin(u_cos, u_sin)
            return mp.msg_up(u_cos, u_sin)
        _, (u_cos, u_sin) = time_fn("  MP fused message/score trunk", _trunk)
        m_cos = u_cos[:, :mp.n_ch] + A_cos
        m_sin = u_sin[:, :mp.n_ch] + A_sin
        s = u_cos[:, mp.n_ch:mp.n_ch + mp.n_scores, 0]

        _, h_packed = time_fn("  MP pack cos/sin → n_sph",
                              lambda: mp._pack(m_cos, m_sin))

        D_block_main_T = D_block_main.transpose(-1, -2)
        time_fn("  MP bmm unrotate (h @ D^T)",
                lambda: torch.bmm(h_packed, D_block_main_T))
        h_global = torch.bmm(h_packed, D_block_main_T)

        def _weights():
            # Mirrors ECENetAttentionMPLayer.forward step 4. K = one score slot
            # per head, or per (head, l) with mp_l_attention.
            K = mp.n_scores
            f_cut = mp.cutoff_fn(dist_ij, mp.r_cut)
            if mp.aggregation == 'sum':
                return s * f_cut[:, None]
            ej_k = edge_j[:, None].expand(-1, K)
            s_max = torch.full((n_atoms, K), float('-inf'), device=device, dtype=dtype
                               ).scatter_reduce(0, ej_k, s.detach(), reduce='amax',
                                                include_self=True)
            num = torch.exp(s - s_max[edge_j]) * f_cut[:, None]
            denom = torch.zeros(n_atoms, K, device=device, dtype=dtype).scatter_add(0, ej_k, num)
            a = num / (denom[edge_j] + mp.softmax_eps)
            if mp.msg_envelope:
                a = a * f_cut[:, None]
            return a
        _, a = time_fn(f"  MP {mp.aggregation} weights", _weights)

        hb = mp.n_base // H
        # Per-(head, l) weights expand to per-(head, spherical index) via l_of_s
        # (all-zero when l_attention is off → the single per-head weight, as before).
        a_full = a.reshape(n_e, H, mp.n_scores_per_head)[:, :, mp.l_of_s]
        contrib = h_global.reshape(n_e, H, hb, mp.n_sph) * a_full[:, :, None, :]
        idx = edge_j[:, None, None, None].expand_as(contrib)
        time_fn("  MP scatter_add (aggregate to atoms)",
                lambda: torch.zeros(n_atoms, H, hb, mp.n_sph, device=device, dtype=dtype
                                    ).scatter_add(0, idx, contrib))
        Delta = torch.zeros(n_atoms, H, hb, mp.n_sph, device=device, dtype=dtype
                            ).scatter_add(0, idx, contrib).reshape(n_atoms, mp.n_base, mp.n_sph)

        time_fn("  MP bmm rotate back (Delta @ D)",
                lambda: torch.bmm(Delta[edge_i], D_block_main))
        v_rot = torch.bmm(Delta[edge_i], D_block_main)
        _, d = time_fn("  MP unpack n_sph → cos/sin", lambda: mp._unpack(v_rot, n_e))
        time_fn("  MP receiver block", lambda: mp.receiver(*d))

        A_cos, A_sin = mp(A_cos, A_sin, r_hat, dist_ij, edge_i, edge_j,
                          n_atoms, types[edge_i], types[edge_j],
                          D_block=D_block_main)

sep("Output")
invariants = model._contract(A_cos, A_sin)
if model.n_max_d is not None:
    rij_basis = radial_basis(dist_ij, model.r_cut_edge, model.n_max_d,
                             cutoff_type=model.cutoff_type)
    time_fn("output MLP",
            lambda: model.output_net(invariants))
    mlp_out = model.output_net(invariants)
    time_fn("dot(MLP_out × rij_basis)",
            lambda: (mlp_out * rij_basis).sum(-1))
else:
    time_fn("output MLP",
            lambda: model.output_net(invariants))

# ── 3. LES long range (when the checkpoint carries it) ────────────────────────
if les_module is not None:
    sep("LES long range")
    cell_t = torch.tensor(cell, dtype=dtype, device=device)

    with torch.no_grad():
        _, l0 = model.forward_pbc(pos_d, types, edge_i, edge_j, shift_e,
                                  nb_src, nb_dst, shift_n,
                                  return_embeddings=True, l0_only=True)

    # marginal cost of the (l0, l1)/charge read-out on top of the SR forward
    time_fn("forward_pbc + l0 read-out",
            lambda: model.forward_pbc(fresh_pos(), types, edge_i, edge_j,
                                      shift_e, nb_src, nb_dst, shift_n,
                                      return_embeddings=True, l0_only=True))
    time_fn("E_lr (periodic Ewald, this cell)",
            lambda: les_module(l0, pos_d, cell=cell_t, **les_flags))
    # isolated-path reference (what a SPICE training step pays). Dense
    # (N, N[, 3, 3]) kernels — skip on big boxes rather than OOM the profile.
    if len(atoms) <= 1000:
        time_fn("E_lr (isolated path, cell=None)",
                lambda: les_module(l0, pos_d, cell=None, **les_flags))
    else:
        print(f"  (isolated-path timing skipped: {len(atoms)} atoms → dense "
              "pair kernels would dominate memory)")

# ── 4. Total forward ──────────────────────────────────────────────────────────
sep("Totals")
time_fn("forward_pbc (full)",
        lambda: model.forward_pbc(fresh_pos(), types, edge_i, edge_j,
                                  shift_e, nb_src, nb_dst, shift_n))

def fwd_forces():
    p = fresh_pos()
    with torch.enable_grad():
        e = model.forward_pbc(p, types, edge_i, edge_j, shift_e,
                              nb_src, nb_dst, shift_n)
        return torch.autograd.grad(e, p)[0]

time_fn("forward_pbc + autograd.grad (forces)", fwd_forces)

if les_module is not None:
    def fwd_forces_les():
        p = fresh_pos()
        with torch.enable_grad():
            e_sr, l0_g = model.forward_pbc(p, types, edge_i, edge_j, shift_e,
                                           nb_src, nb_dst, shift_n,
                                           return_embeddings=True, l0_only=True)
            e = e_sr + les_module(l0_g, p, cell=cell_t, **les_flags).sum()
            return torch.autograd.grad(e, p)[0]

    time_fn("forward_pbc + E_lr + forces (joint graph)", fwd_forces_les)
print()

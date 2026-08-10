"""Predict LES latent charges for one frame with a trained ECENet+LES model.

Loads a checkpoint saved by ``scripts/train_ecenet_xyz.py`` with
``use_les=True`` (the top-level ``les`` dict is required), runs the model's
``l0`` read-out on one frame of an ASE-readable file, and prints the per-atom
latent charges plus the long-range energy. If the frame carries reference
charges (a ``q`` column in extxyz, e.g. electrolyte.xyz), it also reports
MAE and correlation against them.

Sign note: the LES energy is quadratic in the latent charges, so a global
sign flip of ALL charges leaves every prediction unchanged — which sign the
model converges to is chance. The comparison therefore also reports the
sign-aligned numbers (flip applied if it improves the correlation).

Usage (from the repo root):
    python tools/predict_charges.py --checkpoint electrolyte_les.mdl \
        --data ../imports/data/electrolyte.xyz --frame 0
    python tools/predict_charges.py ... --save charges.npz   # positions+q to npz
"""

import argparse
import os
import sys  # repo root + scripts/ on path (run from anywhere)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import numpy as np
import torch


def load_les_model(checkpoint_path, device):
    """Rebuild (model, les_module, element_to_type, e_ref) from a checkpoint.

    Uses the best-val weights when present. The upstream LES charge head is
    built lazily on its first forward, so the caller must run one forward
    through ``les_module`` before ``load_les_state`` (below) — loading state
    into an unmaterialised head would silently no-op.
    """
    from ecenet import ECENet

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'les' not in ckpt:
        raise ValueError(
            f"{checkpoint_path} carries no 'les' dict — this tool is for "
            "checkpoints trained with train_ecenet_xyz(use_les=True).")

    hp = dict(ckpt['hparams'])
    n_mp = hp.pop('n_mp', 1)
    state = ckpt.get('best_state') or ckpt['model']
    dtype = next(iter(state.values())).dtype

    model = ECENet(**hp, n_mp=n_mp)
    if dtype == torch.float64:
        model = model.double()
    model = model.to(device)
    model.load_state_dict(state)
    model.eval()

    from ecenet.les import LESLongRange
    les_module = LESLongRange(ckpt['les'].get('arguments'))
    les_state = ckpt['les'].get('best_state') or ckpt['les']['state_dict']

    return model, les_module, les_state, ckpt['element_to_type'], dtype


def predict_charges(checkpoint_path, data_path, frame=0, device='cpu'):
    """Latent charges + E_lr for one frame. Returns a dict of numpy arrays."""
    from ase.io import read
    from train_ecenet_mptrj import build_topology

    device = torch.device(device)
    model, les_module, les_state, elem_to_type, dtype = load_les_model(
        checkpoint_path, device)

    atoms = read(data_path, index=frame)
    symbols = atoms.get_chemical_symbols()
    missing = sorted({s for s in symbols if s not in elem_to_type})
    if missing:
        raise ValueError(f"Frame contains element(s) {missing} the checkpoint "
                         f"was not trained on (knows: {sorted(elem_to_type)}).")
    types = torch.tensor([elem_to_type[s] for s in symbols],
                         dtype=torch.long, device=device)

    periodic = bool(atoms.pbc.any()) and bool(atoms.cell.any())
    cell_np = np.asarray(atoms.get_cell(), dtype=np.float64) if periodic else None
    pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    cell = (torch.tensor(cell_np, dtype=dtype, device=device)
            if periodic else None)

    hp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['hparams']
    ei, ej, she, ni, nj, shn = build_topology(
        atoms.get_positions(), cell_np, periodic,
        hp['r_cut_edge'], hp['r_cut_neighbor'], device, dtype)

    # les_readout='edge': l0 IS the charge; the LES module holds no state.
    # les_dipole: l0 is packed (N, 4) = [q | u] (bond-dipole read-out).
    is_charge = hp.get('les_readout', 'sum') in ('edge', 'edge_basis')
    les_dip = bool(hp.get('les_dipole', False))
    with torch.no_grad():
        _, l0 = model.forward_pbc(pos, types, ei, ej, she, ni, nj, shn,
                                  return_embeddings=True, l0_only=True)
        les_module(l0, pos, cell=cell, l0_is_charge=is_charge,
                   les_dipole=les_dip)          # materialise the lazy head...
        les_module = les_module.to(device=device, dtype=dtype)
        les_module.load_state_dict(les_state)   # ...then load the trained one
        les_module.eval()
        e_lr, q = les_module(l0, pos, cell=cell, return_charges=True,
                             l0_is_charge=is_charge, les_dipole=les_dip)

    out = {
        'symbols': np.array(symbols),
        'positions': atoms.get_positions(),
        'charges': q.cpu().numpy().reshape(-1),
        'e_lr': float(e_lr.sum()),
    }
    if les_dip:
        out['dipoles'] = l0[:, 1:4].cpu().numpy()
    if 'q' in atoms.arrays:
        out['charges_ref'] = np.asarray(atoms.arrays['q'], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True,
                    help='.mdl from train_ecenet_xyz(use_les=True)')
    ap.add_argument('--data', required=True, help='ASE-readable file (extxyz, ...)')
    ap.add_argument('--frame', type=int, default=0, help='frame index (default 0)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--save', default=None, help='write results to this .npz')
    ap.add_argument('--per_atom', action='store_true',
                    help='print every atom, not just the summary')
    args = ap.parse_args()

    r = predict_charges(args.checkpoint, args.data, args.frame, args.device)
    q = r['charges']

    print(f"Frame {args.frame}: {len(q)} atoms | E_lr = {r['e_lr']:+.6f} eV")
    print(f"Latent charges: sum = {q.sum():+.4f}, "
          f"min = {q.min():+.4f}, max = {q.max():+.4f}")
    if 'dipoles' in r:
        u = r['dipoles']
        mu = (q - q.mean())[:, None] * r['positions']
        mu = mu.sum(0) + u.sum(0)
        print(f"Latent dipoles: |u| mean = {np.linalg.norm(u, axis=1).mean():.4f}, "
              f"max = {np.linalg.norm(u, axis=1).max():.4f} e·Å | "
              f"molecular μ = [{mu[0]:+.4f} {mu[1]:+.4f} {mu[2]:+.4f}]")
    for s in sorted(set(r['symbols'])):
        qs = q[r['symbols'] == s]
        print(f"  {s:2s}: mean {qs.mean():+.4f}  std {qs.std():.4f}  (n={len(qs)})")

    if 'charges_ref' in r:
        ref = r['charges_ref']
        corr = float(np.corrcoef(q, ref)[0, 1])
        sign = -1.0 if corr < 0 else 1.0    # global sign is not identifiable
        mae = np.abs(sign * q - ref).mean()
        print(f"vs reference q: corr = {corr:+.4f} "
              f"(sign-aligned: {sign * corr:.4f}), sign-aligned MAE = {mae:.4f}")

    if args.per_atom:
        ref = r.get('charges_ref')
        for i, (s, qi) in enumerate(zip(r['symbols'], q)):
            extra = f"   ref {ref[i]:+.4f}" if ref is not None else ""
            print(f"  {i:4d} {s:2s} {qi:+.5f}{extra}")

    if args.save:
        np.savez(args.save, **r)
        print(f"Saved to {args.save}")


if __name__ == '__main__':
    main()

"""Evaluate latent-charge dipoles against DFT reference dipoles (SPICE slices).

For every frame the model's latent charges q_i give a molecular dipole
mu = sum_i (q_i - mean(q)) * r_i, compared against the frame's reference
``dipole`` vector. The mean-charge removal matches upstream LES's BEC module
(remove_mean=True): the frames are neutral, so it only removes the model's
spurious net charge and makes mu origin-independent. This is the success
metric MACELES-OFF used for "did the charges become physical" — far more
telling than charge magnitudes.

Data: the seven test slices from https://github.com/ChengUCB/les_fit
(``data-spice-test-bec/``; CC BY-NC 4.0 — download them yourself, they are
deliberately not part of this repository). References were computed at
wB97m-D3(BJ)/Def2SVP (Kim et al., JCTC 2025, 10.1021/acs.jctc.5c01400);
dipoles are in e*Angstrom.

Sign note: the LES energy is quadratic in the charges, so the global charge
sign is not identifiable from energy/force training — one sign is chosen for
the WHOLE checkpoint (whichever aligns the concatenated dipole components
best across all evaluated frames, never per frame) and reported.

Frames containing elements the checkpoint was not trained on are skipped
and counted (e.g. a 10-element SPICE model on the Br-containing dimers).

Usage (from the repo root):
    python tools/eval_spice_dipoles.py --checkpoint spice_les.mdl \
        --data_dir ~/Research/imports/data/spice-test-bec
    python tools/eval_spice_dipoles.py ... --files water dipeptides \
        --max_frames 20 --save dipoles.npz
"""

import argparse
import glob
import os
import sys  # repo root + scripts/ + tools/ on path (run from anywhere)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch


def predict_dipoles(checkpoint_path, xyz_files, device='cpu', max_frames=None,
                    keep_mean=False):
    """Predicted vs reference dipoles for every usable frame of each file.

    Returns (records, skipped) — records is a list of dicts with 'subset',
    'frame', 'n_atoms', 'mu_pred', 'mu_ref', 'q_sum'; mu_pred carries the
    model's arbitrary global sign (align downstream). skipped maps subset →
    number of frames dropped for unknown elements or a missing reference.
    """
    from ase.io import read
    from predict_charges import load_les_model
    from train_ecenet_mptrj import build_topology

    device = torch.device(device)
    model, les_module, les_state, elem_to_type, dtype = load_les_model(
        checkpoint_path, device)
    hp = torch.load(checkpoint_path, map_location='cpu',
                    weights_only=False)['hparams']
    is_charge = hp.get('les_readout', 'sum') in ('edge', 'edge_basis')
    les_dip = bool(hp.get('les_dipole', False))
    materialised = False

    records, skipped = [], {}
    for path in xyz_files:
        subset = os.path.splitext(os.path.basename(path))[0]
        frames = read(path, index=':')
        if max_frames is not None:
            frames = frames[:max_frames]
        skipped[subset] = 0
        for fi, atoms in enumerate(frames):
            symbols = atoms.get_chemical_symbols()
            mu_ref = (atoms.calc.results.get('dipole')
                      if atoms.calc is not None else None)
            if mu_ref is None or any(s not in elem_to_type for s in symbols):
                skipped[subset] += 1
                continue
            types = torch.tensor([elem_to_type[s] for s in symbols],
                                 dtype=torch.long, device=device)
            pos_np = atoms.get_positions()
            pos = torch.tensor(pos_np, dtype=dtype, device=device)
            ei, ej, she, ni, nj, shn = build_topology(
                pos_np, None, False,
                hp['r_cut_edge'], hp['r_cut_neighbor'], device, dtype)
            u = None
            with torch.no_grad():
                _, l0 = model.forward_pbc(pos, types, ei, ej, she, ni, nj, shn,
                                          return_embeddings=True, l0_only=True)
                if is_charge:
                    # edge modes: l0 IS the charge (packed [q | u] with
                    # les_dipole); the LES module is parameter-free, so the
                    # wrapper is not needed for prediction at all
                    q = l0[:, 0]
                    if les_dip:
                        u = l0[:, 1:4].cpu().numpy()
                else:
                    if not materialised:
                        # upstream's charge head builds lazily on first
                        # forward; only then can the trained state be loaded
                        les_module(l0, pos, cell=None)
                        les_module = les_module.to(device=device, dtype=dtype)
                        les_module.load_state_dict(les_state)
                        les_module.eval()
                        materialised = True
                    _, q = les_module(l0, pos, cell=None, return_charges=True)
            q = q.cpu().numpy().reshape(-1)
            q0 = q if keep_mean else q - q.mean()
            mu_pred = q0 @ pos_np
            if u is not None:
                mu_pred = mu_pred + u.sum(axis=0)   # μ = Σ q·r + Σ u
            records.append({
                'subset': subset, 'frame': fi, 'n_atoms': len(symbols),
                'mu_pred': mu_pred, 'mu_ref': np.asarray(mu_ref),
                'q_sum': float(q.sum()),
            })
    return records, skipped


def summarise(records, skipped):
    """Global sign alignment + per-subset metric table. Returns the sign."""
    pred = np.array([r['mu_pred'] for r in records])
    ref = np.array([r['mu_ref'] for r in records])
    # one sign for the whole checkpoint: least-squares alignment of all
    # dipole components (the quadratic LES energy cannot fix it)
    sign = 1.0 if float((pred * ref).sum()) >= 0 else -1.0
    pred = sign * pred

    def row(p, r):
        mae = np.abs(p - r).mean()
        mag_mae = np.abs(np.linalg.norm(p, axis=1)
                         - np.linalg.norm(r, axis=1)).mean()
        norm = (np.linalg.norm(p, axis=1) * np.linalg.norm(r, axis=1))
        cos = ((p * r).sum(1) / np.maximum(norm, 1e-12)).mean()
        corr = float(np.corrcoef(p.reshape(-1), r.reshape(-1))[0, 1])
        # least-squares slope of pred on ref through the origin:
        # scale < 1 → predicted dipoles uniformly too small
        scale = float((p * r).sum() / np.maximum((r * r).sum(), 1e-12))
        return mae, mag_mae, cos, corr, scale

    print(f"\nGlobal charge sign: {'+' if sign > 0 else '-'} "
          "(aligned on all frames; not identifiable from training)")
    print(f"{'subset':22s} {'frames':>6s} {'skip':>4s} {'MAE_xyz':>9s} "
          f"{'MAE_|mu|':>9s} {'cos':>7s} {'corr':>7s} {'scale':>7s}   [e*Ang]")
    subsets = sorted({r['subset'] for r in records})
    for s in subsets:
        idx = [i for i, r in enumerate(records) if r['subset'] == s]
        mae, mag_mae, cos, corr, scale = row(pred[idx], ref[idx])
        print(f"{s:22s} {len(idx):6d} {skipped.get(s, 0):4d} {mae:9.4f} "
              f"{mag_mae:9.4f} {cos:7.3f} {corr:7.3f} {scale:7.3f}")
    for s in sorted(set(skipped) - set(subsets)):
        print(f"{s:22s} {0:6d} {skipped[s]:4d}   (all frames skipped — "
              "unknown elements or no reference)")
    mae, mag_mae, cos, corr, scale = row(pred, ref)
    print(f"{'ALL':22s} {len(records):6d} {sum(skipped.values()):4d} "
          f"{mae:9.4f} {mag_mae:9.4f} {cos:7.3f} {corr:7.3f} {scale:7.3f}")
    qs = np.abs([r['q_sum'] for r in records])
    print(f"net latent charge |sum q| (pre-removal): "
          f"mean {qs.mean():.4f}, max {qs.max():.4f}")
    return sign


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True,
                    help='.mdl trained with use_les=True (xyz or SPICE trainer)')
    ap.add_argument('--data_dir', required=True,
                    help='directory with the les_fit data-spice-test-bec .xyz '
                         'files (see module docstring for the source)')
    ap.add_argument('--files', nargs='+', default=None,
                    help='subset names to evaluate (default: every .xyz found)')
    ap.add_argument('--max_frames', type=int, default=None,
                    help='cap frames per file (smoke runs)')
    ap.add_argument('--keep_mean', action='store_true',
                    help="don't remove the mean charge before the dipole "
                         "(reference frames are neutral; removal is upstream "
                         "BEC's remove_mean=True convention)")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--save', default=None,
                    help='write per-frame sign-aligned results to this file: '
                         '.csv (one row per frame, ready for a correlation '
                         'plot) or .npz (arrays)')
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if args.files:
        xyz_files = [os.path.join(data_dir, f if f.endswith('.xyz')
                                  else f + '.xyz') for f in args.files]
        missing = [f for f in xyz_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"missing: {missing}")
    else:
        xyz_files = sorted(glob.glob(os.path.join(data_dir, '*.xyz')))
        if not xyz_files:
            raise FileNotFoundError(f"no .xyz files in {data_dir}")

    records, skipped = predict_dipoles(
        args.checkpoint, xyz_files, device=args.device,
        max_frames=args.max_frames, keep_mean=args.keep_mean)
    if not records:
        raise SystemExit("no usable frames (unknown elements everywhere? "
                         f"skipped: {skipped})")
    sign = summarise(records, skipped)

    if args.save:
        mu_pred = sign * np.array([r['mu_pred'] for r in records])
        mu_ref = np.array([r['mu_ref'] for r in records])
        if args.save.endswith('.csv'):
            with open(args.save, 'w') as fh:
                fh.write('subset,frame,n_atoms,'
                         'mu_pred_x,mu_pred_y,mu_pred_z,'
                         'mu_ref_x,mu_ref_y,mu_ref_z,q_sum\n')
                for r, p, m in zip(records, mu_pred, mu_ref):
                    fh.write(f"{r['subset']},{r['frame']},{r['n_atoms']},"
                             f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},"
                             f"{m[0]:.6f},{m[1]:.6f},{m[2]:.6f},"
                             f"{r['q_sum']:.6f}\n")
        else:
            np.savez(args.save,
                     subset=np.array([r['subset'] for r in records]),
                     frame=np.array([r['frame'] for r in records]),
                     n_atoms=np.array([r['n_atoms'] for r in records]),
                     mu_pred=mu_pred, mu_ref=mu_ref,
                     q_sum=np.array([r['q_sum'] for r in records]))
        print(f"Saved to {args.save}")


if __name__ == '__main__':
    main()

"""Run an MD simulation (NVT or NVE) starting from an arbitrary xyz file.

Usage:
    python examples/run_md_xyz.py --xyz water.xyz --checkpoint model.mdl \\
        --ensemble nvt --temperature 300 --n_steps 10000 --timestep 0.5 \\
        --output traj.xyz --log md.log

The input is read with ASE, so extended-xyz headers (Lattice="..." and
pbc="...") are honoured automatically; a plain xyz is treated as a
non-periodic cluster. For a periodic box whose header lost its Lattice
field, pass --cell <side_length_angstrom> to set a cubic cell (and PBC)
by hand — check the printed "PBC:" line to confirm which case you got.
For a multi-frame file, pick the starting frame with --frame_idx
(default: last frame, ASE convention).

--ensemble nvt  -> Langevin thermostat at --temperature (uses --friction)
--ensemble nve  -> VelocityVerlet at constant energy (--temperature only
                   seeds the initial velocities)
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from ecenet.calculator import load_calculator


def run_md_xyz(checkpoint, xyz, cell=None, frame_idx=-1, seed=0, ensemble='nvt',
               temperature=300, timestep=0.5, friction=0.01, n_steps=10000,
               log_every=100, output='traj.xyz', log='md.log', device=None,
               float32=False, energy_units=None, log_timings=False,
               fuse_nonlin=False, edge_frame_fused=False, tf32=False,
               dump_charges=False, dump_bec=False):
    """Run NVT/NVE MD from an arbitrary xyz file.

    Importable entry point (see main() for the equivalent CLI). Writes a
    trajectory to `output` and a step log to `log`; returns the final Atoms.
    """
    dtype = torch.float32 if float32 else torch.float64

    if tf32:
        if dtype == torch.float64:
            print("[tf32] requested but dtype=float64 → no effect (TF32 is "
                  "float32-only); add --float32 to use it")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            print("[tf32] enabled: float32 matmuls → TF32 tensor cores "
                  "(A/B energies/forces against a tf32=False run)")

    # ── Load starting frame ────────────────────────────────────────────────
    atoms = read(xyz, index=frame_idx)
    if cell is not None:
        atoms.set_cell([cell, cell, cell])
        atoms.set_pbc(True)
    print(f"Loaded {xyz} (frame {frame_idx}): {len(atoms)} atoms")
    print(f"Elements: {sorted(set(atoms.get_chemical_symbols()))}")
    print(f"PBC: {atoms.pbc.tolist()}")
    if atoms.pbc.any():
        print(f"Cell: {atoms.cell.lengths()} Å")

    # ── Calculator ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint}")
    calc = load_calculator(checkpoint, device=device, dtype=dtype,
                           energy_units=energy_units, log_timings=log_timings)
    atoms.calc = calc

    if (dump_charges or dump_bec) and 'charges' not in calc.implemented_properties:
        raise ValueError(
            "--dump_charges/--dump_bec need a joint-LES checkpoint (the "
            "latent charges come from the LES head); this checkpoint loaded "
            f"a short-range {type(calc).__name__}.")
    if (dump_charges or dump_bec) and output.endswith('.traj'):
        raise ValueError(
            "--dump_charges/--dump_bec need an extended-xyz output: ASE's "
            ".traj format silently drops custom per-atom arrays "
            "(les_q / les_u / bec). Use --output traj.xyz.")

    # Fused kernels (opt-in). MD is forces-only (single backward), so both are
    # safe here; edge_frame_fused enables the MP pack/unrotate fusion with it
    # (the separate e2n knob is a profiling concern — see tools/profile_step.py).
    if fuse_nonlin:
        calc.model.set_activation_fused(True)
        n_set = sum(1 for m in calc.model.modules() if getattr(m, 'fused', False))
        print(f"[fuse_nonlin] fused RealSpaceNonlinearity on {n_set} module(s)")
    if edge_frame_fused:
        calc.model.set_edge_frame_fused(True, e2n=True)
        print("[edge_frame_fused] fused gather+rotate+reshape (+MP pack/unrotate)")

    # Quick sanity check
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(f"Initial energy: {e:.4f} eV")
    print(f"Initial max force: {np.abs(f).max():.4f} eV/Å")

    # ── Initialise velocities ──────────────────────────────────────────────
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,
                                 rng=np.random.default_rng(seed))

    # ── MD ─────────────────────────────────────────────────────────────────
    if ensemble == 'nvt':
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction / units.fs,
            rng=np.random.default_rng(seed + 1),
            # fixcm=True (ASE default) biases the kinetic temperature high,
            # markedly so for small systems. fixcm=False samples NVT correctly.
            fixcm=False,
        )
    else:  # nve
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep * units.fs,
        )

    # Trajectory writer. With dump_charges, copy the latent charges (and,
    # for les_dipole checkpoints, the atomic dipoles) from the calculator's
    # last force call into per-atom arrays, so the extxyz writer emits them
    # as les_q / les_u columns (guarded above: .traj drops custom arrays).
    # dump_bec additionally computes the Born effective charges per dumped
    # frame (its own forward + 3 backwards — not free like the charges) and
    # writes them as a bec column of 9 row-major components per atom, the
    # same layout as the les_fit reference files.
    def dump_les_arrays():
        if dump_charges:
            atoms.set_array('les_q', calc.results['charges'])
            if 'les_dipoles' in calc.results:
                atoms.set_array('les_u', calc.results['les_dipoles'])
        if dump_bec:
            atoms.set_array('bec', calc.compute_bec(atoms).reshape(-1, 9))

    if output.endswith('.traj'):
        traj = Trajectory(output, 'w', atoms)
        dyn.attach(traj.write, interval=log_every)
    else:
        def write_frame():
            if dump_charges or dump_bec:
                dump_les_arrays()
            write(output, atoms, append=True)
        dyn.attach(write_frame, interval=log_every)

    # Logger
    log_file = open(log, 'w')
    log_file.write(f"{'step':>8}  {'time_ps':>10}  {'E_pot (eV)':>14}  "
                   f"{'E_kin (eV)':>14}  {'T (K)':>8}\n")
    log_file.flush()

    def log_step():
        step   = dyn.get_number_of_steps()
        t_ps   = step * timestep * 1e-3
        e_pot  = atoms.get_potential_energy()
        e_kin  = atoms.get_kinetic_energy()
        temp   = atoms.get_temperature()
        line   = f"{step:>8d}  {t_ps:>10.4f}  {e_pot:>14.6f}  {e_kin:>14.6f}  {temp:>8.2f}"
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    dyn.attach(log_step, interval=log_every)

    print(f"\nRunning {n_steps} {ensemble.upper()} steps "
          f"(T={temperature} K, dt={timestep} fs)...\n")
    log_step()  # log initial state
    dyn.run(n_steps)

    log_file.close()
    print(f"\nDone. Trajectory: {output}  Log: {log}")
    return atoms


def main():
    parser = argparse.ArgumentParser(description='NVT/NVE MD with ECENet from an xyz file')
    parser.add_argument('--xyz',          required=True,
                        help='Input structure (xyz / extended-xyz)')
    parser.add_argument('--checkpoint',   required=True,
                        help='Path to trained .mdl checkpoint')
    parser.add_argument('--cell',         type=float, default=None,
                        help='Cubic cell side length in Å; sets the cell and PBC '
                             'for a box file missing its Lattice header')
    parser.add_argument('--frame_idx',    type=int, default=-1,
                        help='Frame index to start from for multi-frame files '
                             '(default: -1, the last frame)')
    parser.add_argument('--seed',         type=int, default=0)
    # MD settings
    parser.add_argument('--ensemble',     default='nvt', choices=['nvt', 'nve'],
                        help='nvt = Langevin thermostat; nve = constant-energy VelocityVerlet')
    parser.add_argument('--temperature',  type=float, default=300,
                        help='Temperature in K (NVT target; for NVE only seeds initial velocities)')
    parser.add_argument('--timestep',     type=float, default=0.5,
                        help='Timestep in fs')
    parser.add_argument('--friction',     type=float, default=0.01,
                        help='Langevin friction in 1/fs (NVT only)')
    parser.add_argument('--n_steps',      type=int,   default=10000,
                        help='Number of MD steps')
    parser.add_argument('--log_every',    type=int,   default=100,
                        help='Print/log every N steps')
    # Output
    parser.add_argument('--output',       default='traj.xyz',
                        help='Trajectory output file (.xyz or .traj)')
    parser.add_argument('--log',          default='md.log')
    # Calculator
    parser.add_argument('--device',       default=None)
    parser.add_argument('--float32',      action='store_true')
    parser.add_argument('--energy_units', default=None,
                        choices=['eV', 'kcal/mol'],
                        help='Override unit conversion (auto-detected from checkpoint '
                             'if not set). Use kcal/mol for train_ecenet.py models, '
                             'eV for train_ecenet_spice.py models.')
    parser.add_argument('--log_timings',  action='store_true',
                        help='Print per-step calculator timings')
    parser.add_argument('--fuse_nonlin',  action='store_true',
                        help='Fused RealSpaceNonlinearity (Triton on CUDA+silu)')
    parser.add_argument('--edge_frame_fused', action='store_true',
                        help='Fused edge-frame + MP pack/unrotate (Triton on CUDA+fp32)')
    parser.add_argument('--dump_charges', action='store_true',
                        help='Write the LES latent charges (les_q) — and, for '
                             'les_dipole checkpoints, the latent atomic '
                             'dipoles (les_u) — as per-atom arrays on every '
                             'dumped frame. Joint-LES checkpoints only. Note '
                             'the global sign of the latent charges is '
                             'arbitrary (E_lr is quadratic in q).')
    parser.add_argument('--dump_bec', action='store_true',
                        help='Compute the Born effective charges Z* = dP/dr '
                             '(charge-flow terms included) on every dumped '
                             'frame and write them as a per-atom bec column '
                             'of 9 row-major components. Joint-LES '
                             'checkpoints only; costs one extra forward + 3 '
                             'backwards per dumped frame. Same arbitrary '
                             'global sign as the latent charges.')
    parser.add_argument('--tf32', action='store_true',
                        help='Route float32 matmuls to TF32 tensor cores (Ampere+); '
                             'no effect without --float32. The fused Triton kernels '
                             'stay IEEE regardless.')
    args = parser.parse_args()
    run_md_xyz(**vars(args))


if __name__ == '__main__':
    main()

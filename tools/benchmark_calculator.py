"""tools/benchmark_calculator.py — same-machine ASE-calculator speed benchmark.

Times any ASE calculator on the same box and hardware so competing MLIPs can be
compared head-to-head. Paper timings are not commensurable — hardware, dtype,
system size, and kernel maturity all differ — so the credible comparison is
this: one machine, one protocol, every model at its best settings.

Competitor packages pin conflicting torch/e3nn versions: run each model in its
own conda env and collate with --csv, which appends one identical row per run.

Usage (repo root):
    python tools/benchmark_calculator.py --calc ecenet --checkpoint model.mdl \
        --box water_box.xyz --float32 --fuse --csv bench.csv
    python tools/benchmark_calculator.py --calc mace --checkpoint medium \
        --box water_box.xyz --float32 --csv bench.csv         # mace-torch env
    python tools/benchmark_calculator.py --calc custom --factory mymod:make \
        --box water_box.xyz                                   # anything else

Measures:
  * single-point E+F latency — positions are rattled before every call so
    ASE's result cache never short-circuits the calculator, and neighbor
    lists really rebuild;
  * NVE MD throughput (VelocityVerlet, steps/s and ns/day) — the number MD
    users care about; NVE so no thermostat cost muddies the comparison;
  * peak CUDA memory over each timed section (when torch + CUDA are present).

Protocol notes: benchmark every model at its BEST settings (--fuse for
ecenet; competitors' fast paths are usually already their defaults), state
the dtype, sweep sizes with --repeat for a scaling curve, and pair any speed
table with an accuracy axis — speed alone is meaningless across models
sitting at different accuracy/cost tradeoffs.

The competitor factories below are written against their 2026 APIs and kept
deliberately thin — expect drift, and adjust inside the competitor's env.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

try:
    import torch  # every competitor uses it, but the
except ImportError:                       # harness itself must not require it
    torch = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Calculator factories (lazy imports — only the installed one must work) ────

def _device(args):
    if args.device:
        return args.device
    return 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'


def _calc_ecenet(args):
    sys.path.insert(0, REPO_ROOT)
    from ecenet.calculator import load_calculator
    dtype = torch.float32 if args.float32 else torch.float64
    calc = load_calculator(args.checkpoint, dtype=dtype, verbose=False)
    if args.fuse:
        calc.model.set_edge_frame_fused(True)
        calc.model.set_activation_fused(True)   # forces = single backward: safe
    return calc


def _calc_mace(args):
    from mace.calculators import MACECalculator, mace_mp
    dt = 'float32' if args.float32 else 'float64'
    if args.checkpoint in ('small', 'medium', 'large'):   # foundation shorthand
        return mace_mp(model=args.checkpoint, device=_device(args), default_dtype=dt)
    return MACECalculator(model_paths=args.checkpoint, device=_device(args),
                          default_dtype=dt)


def _calc_chgnet(args):
    from chgnet.model.dynamics import CHGNetCalculator
    return CHGNetCalculator(use_device=_device(args))


def _calc_sevennet(args):
    from sevenn.sevennet_calculator import SevenNetCalculator
    return SevenNetCalculator(model=args.checkpoint or '7net-0', device=_device(args))


def _calc_matgl(args):
    import matgl
    from matgl.ext.ase import PESCalculator
    return PESCalculator(matgl.load_model(args.checkpoint or 'M3GNet-MP-2021.2.8-PES'))


def _calc_custom(args):
    """--factory mymodule:function — called as function(args), must return an
    ASE calculator. The module must be importable (PYTHONPATH)."""
    import importlib
    mod, fn = args.factory.rsplit(':', 1)
    return getattr(importlib.import_module(mod), fn)(args)


FACTORIES = {'ecenet': _calc_ecenet, 'mace': _calc_mace, 'chgnet': _calc_chgnet,
             'sevennet': _calc_sevennet, 'matgl': _calc_matgl, 'custom': _calc_custom}

INSTALL_HINT = {'mace': 'pip install mace-torch', 'chgnet': 'pip install chgnet',
                'sevennet': 'pip install sevenn', 'matgl': 'pip install matgl'}


# ── Timing utilities (profile_step conventions) ───────────────────────────────

def sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_mem():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_mem_mb():
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    return float('nan')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calc', required=True, choices=sorted(FACTORIES))
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--factory', default=None,
                   help="--calc custom: 'module:function', called as function(args)")
    p.add_argument('--box', required=True, help='any ASE-readable structure file')
    p.add_argument('--frame_idx', type=int, default=-1)
    p.add_argument('--repeat', type=int, nargs=3, default=None, metavar=('A', 'B', 'C'),
                   help='replicate the box into an AxBxC supercell (size scaling)')
    p.add_argument('--float32', action='store_true',
                   help='float32 where the factory supports it (state it in the table)')
    p.add_argument('--fuse', action='store_true',
                   help="enable the model's fast paths (ecenet: edge-frame + "
                        'activation fusion; no-op for factories without a hook)')
    p.add_argument('--device', default=None, help='default: cuda if available')
    p.add_argument('--n_warmup', type=int, default=5)
    p.add_argument('--n_time', type=int, default=20, help='timed single-point calls')
    p.add_argument('--md_steps', type=int, default=100, help='timed MD steps; 0 skips MD')
    p.add_argument('--md_warmup', type=int, default=10)
    p.add_argument('--timestep', type=float, default=1.0, help='MD timestep, fs')
    p.add_argument('--temperature', type=float, default=300.0, help='MB init, K')
    p.add_argument('--rattle', type=float, default=0.01,
                   help='stddev (Å) of the per-call position rattle')
    p.add_argument('--label', default=None, help='row label for --csv (default: --calc)')
    p.add_argument('--csv', default=None, help='append one result row (collate runs)')
    args = p.parse_args()

    atoms = read(args.box, index=args.frame_idx)
    if args.repeat:
        atoms = atoms.repeat(tuple(args.repeat))
    n_atoms = len(atoms)

    try:
        atoms.calc = FACTORIES[args.calc](args)
    except ImportError as e:
        hint = INSTALL_HINT.get(args.calc, '')
        sys.exit(f"[{args.calc}] import failed ({e}){' — try: ' + hint if hint else ''}")

    gpu = (torch.cuda.get_device_name() if torch is not None
           and torch.cuda.is_available() else 'cpu')
    print(f"System: {n_atoms} atoms from {args.box}"
          f"{' x' + 'x'.join(map(str, args.repeat)) if args.repeat else ''}, "
          f"pbc={atoms.pbc.any()}, device={gpu}")

    # ── Single-point E+F ──────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    base = atoms.get_positions().copy()
    deltas = rng.normal(0.0, args.rattle, (args.n_warmup + args.n_time, n_atoms, 3))

    for i in range(args.n_warmup):                      # warmup (JIT, autotune, caches)
        atoms.positions = base + deltas[i]
        atoms.get_potential_energy(), atoms.get_forces()

    reset_peak_mem()
    sync()
    t0 = time.perf_counter()
    for i in range(args.n_warmup, args.n_warmup + args.n_time):
        atoms.positions = base + deltas[i]              # bust ASE's result cache
        atoms.get_potential_energy(), atoms.get_forces()
    sync()
    sp_ms = (time.perf_counter() - t0) / args.n_time * 1000
    sp_mem = peak_mem_mb()
    print(f"  single-point E+F   {sp_ms:9.2f} ms/call   "
          f"{sp_ms / n_atoms * 1000:8.2f} us/atom   peak {sp_mem:8.1f} MB")

    # ── NVE MD ────────────────────────────────────────────────────────────────
    md_steps_s = md_ns_day = md_mem = float('nan')
    if args.md_steps > 0:
        atoms.positions = base
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
        dyn = VelocityVerlet(atoms, args.timestep * units.fs)
        dyn.run(args.md_warmup)
        reset_peak_mem()
        sync()
        t0 = time.perf_counter()
        dyn.run(args.md_steps)
        sync()
        md_steps_s = args.md_steps / (time.perf_counter() - t0)
        md_ns_day = md_steps_s * args.timestep * 1e-6 * 86400
        md_mem = peak_mem_mb()
        print(f"  NVE MD ({args.timestep} fs)     {md_steps_s:9.2f} steps/s  "
              f"{md_ns_day:8.2f} ns/day    peak {md_mem:8.1f} MB")

    # ── Collation row ─────────────────────────────────────────────────────────
    if args.csv:
        row = {'label': args.label or args.calc, 'calc': args.calc,
               'checkpoint': args.checkpoint or '', 'box': os.path.basename(args.box),
               'repeat': 'x'.join(map(str, args.repeat)) if args.repeat else '',
               'n_atoms': n_atoms,
               'dtype': 'float32' if args.float32 else 'default/float64',
               'fuse': args.fuse, 'device': gpu,
               'torch': torch.__version__ if torch is not None else '',
               'sp_ms': round(sp_ms, 3),
               'sp_us_per_atom': round(sp_ms / n_atoms * 1000, 3),
               'md_steps_per_s': round(md_steps_s, 3),
               'md_ns_per_day': round(md_ns_day, 3),
               'sp_peak_mb': round(sp_mem, 1), 'md_peak_mb': round(md_mem, 1),
               'date': time.strftime('%Y-%m-%d %H:%M')}
        new = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"  appended to {args.csv}")


if __name__ == '__main__':
    main()

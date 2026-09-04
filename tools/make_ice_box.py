"""tools/make_ice_box.py — proton-disordered hexagonal ice Ih box for benchmarks.

Builds the oxygen sublattice of ice Ih (lonsdaleite topology, orthorhombic
8-molecule cell a x a*sqrt(3) x c), replicates it, then orients every O-O
hydrogen bond so the Bernal-Fowler ice rules hold (two covalent H per O, one
H per bond) using Buch-style loop-erased path flips from a random start —
the resulting proton disorder spans the whole box, not one replicated cell.
Hydrogens sit on the O..O axis at 0.9572 A (ideal, unrelaxed).

Usage (repo root):
    python tools/make_ice_box.py --repeat 3 2 2 --out ice_box.xyz
    python tools/make_ice_box.py --repeat 4 3 3 --sel 100        # DPA budget check

Prints the density and the max/mean neighbor count within --rcut over all
atoms — the number a fixed-budget descriptor (DeePMD `sel`) is compared
against. Ice Ih at 6 A sits near 85 neighbors: below water's ~100, so the
box exercises a model on a solid with partly empty neighbor slots.
"""
import argparse

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import write
from ase.neighborlist import neighbor_list

A_HEX = 4.5181      # A, ice Ih lattice constants near 250 K (Röttger et al. 1994)
C_HEX = 7.3560
D_OH = 0.9572       # A, covalent O-H placed along the hydrogen bond
D_OO_MAX = 3.0      # A, hydrogen-bond search radius (O-O in ice Ih ~2.76 A)


def oxygen_lattice(repeat):
    """Ice Ih oxygen positions: lonsdaleite Wyckoff 4f (z=1/16) in the
    hexagonal cell, converted to the orthorhombic 8-atom cell and repeated."""
    z = 1.0 / 16.0
    frac = np.array([(1/3, 2/3, z), (2/3, 1/3, z + 1/2),
                     (2/3, 1/3, -z), (1/3, 2/3, 1/2 - z)]) % 1.0
    hexcell = [[A_HEX, 0.0, 0.0],
               [-A_HEX / 2, A_HEX * np.sqrt(3) / 2, 0.0],
               [0.0, 0.0, C_HEX]]
    ox = Atoms('O4', scaled_positions=frac, cell=hexcell, pbc=True)
    ortho = make_supercell(ox, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])
    ortho.wrap()
    return ortho.repeat(tuple(repeat))


def hydrogen_bonds(ox):
    """Unique O-O bonds as (a, b, shift) with a < b; every O must have four."""
    i, j, S = neighbor_list('ijS', ox, D_OO_MAX)
    bonds = [(int(a), int(b), tuple(int(x) for x in s))
             for a, b, s in zip(i, j, S) if a < b]
    n = len(ox)
    if len(bonds) != 2 * n:
        raise RuntimeError(f'expected {2 * n} H-bonds for {n} O, found {len(bonds)}')
    adj = [[] for _ in range(n)]
    for k, (a, b, _) in enumerate(bonds):
        adj[a].append(k)
        adj[b].append(k)
    if any(len(x) != 4 for x in adj):
        raise RuntimeError('oxygen lattice is not 4-coordinated')
    return bonds, adj


def orient_bonds(bonds, adj, n, rng):
    """Return donor[k] in {0,1}: which end of bonds[k] donates its H, such
    that every O donates on exactly two of its four bonds (ice rules)."""
    donor = rng.integers(0, 2, len(bonds))
    count = np.zeros(n, int)
    for k, (a, b, _) in enumerate(bonds):
        count[(a, b)[donor[k]]] += 1

    def outgoing(v):
        return [k for k in adj[v] if bonds[k][donor[k]] == v]

    while True:
        over = np.flatnonzero(count > 2)
        if len(over) == 0:
            return donor
        start = int(rng.choice(over))
        nodes, edges, v = [start], [], start
        while True:                       # loop-erased random walk along donations
            k = int(rng.choice(outgoing(v)))
            w = bonds[k][1 - donor[k]]
            if w in nodes:                # erase the loop: keeps the path simple
                idx = nodes.index(w)
                nodes, edges = nodes[:idx + 1], edges[:idx]
            else:
                nodes.append(w)
                edges.append(k)
            v = w
            if count[v] < 2 and v != start:
                break
        for k in edges:                   # flip the path: start -1, end +1
            donor[k] = 1 - donor[k]
        count[start] -= 1
        count[v] += 1


def build_ice(repeat, seed):
    ox = oxygen_lattice(repeat)
    bonds, adj = hydrogen_bonds(ox)
    rng = np.random.default_rng(seed)
    donor = orient_bonds(bonds, adj, len(ox), rng)

    pos, cell = ox.get_positions(), ox.cell.array
    hyd = [[] for _ in range(len(ox))]
    for k, (a, b, s) in enumerate(bonds):
        vec = pos[b] + np.asarray(s) @ cell - pos[a]      # a -> b through PBC
        u = vec / np.linalg.norm(vec)
        if donor[k] == 0:
            hyd[a].append(pos[a] + D_OH * u)
        else:
            hyd[b].append(pos[b] - D_OH * u)
    if any(len(h) != 2 for h in hyd):
        raise RuntimeError('ice rules violated after orientation')

    symbols, coords = [], []
    for o in range(len(ox)):                                # O H H per molecule
        symbols += ['O', 'H', 'H']
        coords += [pos[o], hyd[o][0], hyd[o][1]]
    atoms = Atoms(symbols, positions=coords, cell=cell, pbc=True)
    atoms.wrap()
    return atoms


def report(atoms, rcut, sel):
    n = len(atoms)
    vol = atoms.get_volume()
    n_mol = n // 3
    rho = n_mol * 18.015 / (vol * 0.602214)
    counts = np.bincount(neighbor_list('i', atoms, rcut), minlength=n)
    L = np.linalg.norm(atoms.cell.array, axis=1)
    print(f'ice Ih: {n_mol} molecules, {n} atoms, cell '
          f'{L[0]:.2f} x {L[1]:.2f} x {L[2]:.2f} A, {rho:.3f} g/cm^3, '
          f'{n / vol:.4f} atoms/A^3')
    print(f'neighbors within {rcut:.1f} A: max {counts.max()}, '
          f'mean {counts.mean():.1f}')
    if sel is not None:
        n_over = int((counts > sel).sum())
        verdict = 'no truncation' if n_over == 0 else f'{n_over} atoms truncated'
        print(f'budget sel={sel}: {verdict}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--repeat', type=int, nargs=3, default=(3, 2, 2),
                   metavar=('A', 'B', 'C'),
                   help='repeats of the 8-molecule orthorhombic cell '
                        '(4.52 x 7.83 x 7.36 A); default 3 2 2 = 96 molecules')
    p.add_argument('--seed', type=int, default=0, help='proton-disorder seed')
    p.add_argument('--rcut', type=float, default=6.0,
                   help='cutoff for the neighbor-count report')
    p.add_argument('--sel', type=int, default=None,
                   help='fixed neighbor budget to check the box against')
    p.add_argument('--out', default='ice_box.xyz', help='extxyz output path')
    args = p.parse_args()

    atoms = build_ice(args.repeat, args.seed)
    write(args.out, atoms, format='extxyz')
    report(atoms, args.rcut, args.sel)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

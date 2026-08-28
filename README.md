# ECENet

**ECENet** is an SO(3)-equivariant interatomic potential (MLIP). It uses
per-edge, SO(2)-equivariant features and passes SO(3)-equivariant messages through the nodes. The use of SO(2) features enables faster, more expressive operations than is possible with SO(3) features.

- SO(3)-invariant energies.
- Equivariant message passing.
- Energy / forces / stress via autograd; ASE calculator for MD and relaxations.

> 📄 **Paper forthcoming.** A preprint describing the method is in preparation.

## Layout

```
ecenet/
  model.py         ECENet — the model (message passing when n_mp >= 2)
  equivariant.py   EquivariantLinear, RealSpaceNonlinearity
  film.py          ElementFiLM — element(+distance)-conditioned edge gate
  les.py           LESLongRange — optional long-range add-on (wraps the `les` package)
  ace_basis.py     analytic ACE basis + Wigner-rotation autograd functions
  spherical.py     real spherical harmonics, Clebsch–Gordan, Wigner-D (recursion + rotation)
  radial.py        radial bases, cutoff envelopes, edge/neighbour lists
  calculator.py    ECENetCalculator (ASE)
  datasets/mptrj.py  MPtrj dataset loader

scripts/               training / data entry points (run from the repo root)
  train_ecenet.py        rMD17 / MD22 single-molecule training
  train_ecenet_spice.py  SPICE multi-molecule training (10 elements, DDP)
  train_ecenet_mptrj.py  MPtrj training (periodic crystals, ~89 elements, stress)
  train_ecenet_xyz.py    small ASE/extxyz datasets (single-process; optional joint LES)
  prepare_mptrj.py       tensorise raw MPtrj JSON → .pt shards
  eval_spice.py          evaluate a SPICE checkpoint on the test set

examples/              runnable examples
  run_md_*.py            ASE MD drivers (NVT/NPT); importable or CLI

tests/                 test suite (test_*.py, run from the repo root)
tools/                 developer utilities (run from the repo root)
  profile_step.py        profile a single calculator step
  equiv_vs_ref.py        numerical-equivalence check vs a git ref
  predict_charges.py     LES latent charges + E_lr for one frame of an ASE file
  eval_spice_dipoles.py  latent-charge dipoles vs DFT reference dipoles
                         (ChengUCB/les_fit SPICE test slices, downloaded separately)
  eval_spice_bec.py      zero-shot Born effective charges (dP/dr via autograd,
                         charge-flow terms included) vs the same slices' references
  eval_wbm.py            WBM / Matbench-Discovery evaluation: `relax` (job-array
                         MLIP relaxations of WBM initial structures, resumable),
                         `score` (e_form + hull metrics: F1, DAF, MAE), and
                         `rmsd` (geometry vs DFT-relaxed; needs pymatgen)
```

## Install

ECENet needs PyTorch, NumPy, ASE, and `sphericart-torch` (for spherical
harmonics). Install it as a package (editable, so `import ecenet` resolves from
anywhere):

```bash
conda create -n ecenet python=3.11
conda activate ecenet
pip install -e .
```

On a GPU machine, install the torch wheel that matches your CUDA version first
(see <https://pytorch.org/get-started/locally/>), then `pip install -e .`.

ECENet is pure PyTorch — no compiled/custom CUDA extensions to build. The
optional fused kernels (below) use Triton, which ships with CUDA builds of
PyTorch and is JIT-compiled at runtime; without Triton or a GPU the same code
paths run as pure-PyTorch fallbacks.

**Tested with:** Python 3.11 + CUDA PyTorch (cluster) and Python 3.14 + PyTorch
2.10 CPU (local); NumPy 2.4, ASE 3.28. The dependency floors in `pyproject.toml`
are deliberately conservative.

## Quickstart

Run everything from the repo root so `import ecenet` resolves. All trainers are
**import-and-call** — every option is a keyword argument of the training
function (see each function's docstring for the full list). The multi-GPU
trainers (`train_ecenet_spice`, `train_ecenet_mptrj`) additionally keep a
`__main__` entry point so they launch directly under `torchrun`.

Train on an rMD17 / MD22 molecule:

```python
from scripts.train_ecenet import train_ecenet
model, results = train_ecenet(molecule='ethanol', n_train=950,
                              l_max=3, n_max=4, embed_dim=16, n_epochs=200,
                              n_mp=2)            # n_mp ≥ 2 turns on message passing
```

Train on SPICE (10 elements), single process or DDP:

```python
from scripts.train_ecenet_spice import train_ecenet_spice
model, results = train_ecenet_spice(l_max=3, n_max=4, embed_dim=32, n_layers=2)
```

```bash
torchrun --nproc_per_node=4 scripts/train_ecenet_spice.py    # 4-GPU DDP
```

Use a trained model from Python / ASE:

```python
from ase.io import read
from ecenet.calculator import ECENetCalculator

atoms = read('molecule.xyz')
atoms.calc = ECENetCalculator.from_checkpoint('model.mdl')
print(atoms.get_potential_energy())   # eV
print(atoms.get_forces())             # eV/Å
print(atoms.get_stress())             # eV/Å³ (periodic systems)
```

```python
import ecenet
model = ecenet.ECENet(n_types=10, l_max=3, n_max=4, embed_dim=16)
energy = model(positions, types)      # positions (N,3), types (N,)
```

## Model options

All options below are keyword arguments of `ecenet.ECENet(...)` and of every
trainer; the class and trainer docstrings document the details.

**Low-rank layers** — `bottleneck_dim=16` replaces each equivariant layer with
down → nonlinearity → up (zero-init up, so each layer is the identity at
initialisation).

An element-conditioned FiLM gate on the edge features is on by default
(`element_film=False` disables it; sub-options in `ecenet/film.py`).

**Message passing** — with `n_mp >= 2`, each MP layer computes a per-edge
message and an invariant score, aggregates messages at the receiving atom, and
applies a receiver transform. `mp_type` selects the weighting:

| `mp_type` | behaviour |
| --- | --- |
| `'softmax'` (default) | attention over the receiver's incoming edges — a weighted average, intensive in coordination |
| `'sum'` | signed score × cutoff envelope — extensive, and a neighbour can contribute negatively |

Both are smooth as edges cross `r_cut_edge`. `mp_dim` sets the message trunk's
bottleneck width, `mp_n_heads` the number of attention heads, and
`mp_l_attention` gives each head one score per degree `l`. Zero-init at the
trunk output, so message passing is a no-op at initialisation. (The older
`mp_type='edge'` has been removed; its checkpoints are rejected with an
explicit error.)

**Nonlinearity** — `activation` selects the pointwise nonlinearity
(`'silu'` default); `activation='identity'` linearizes the full equivariant
stack (ablation), and `use_nonlinearity=False` skips it in the main layer
stack only.

## Long-range electrostatics (LES, optional)

The optional **LES** add-on (Latent Ewald Summation) extends ECENet beyond
`r_cut_edge`: a head predicts a latent charge per atom, and the smeared-Coulomb
interaction between those charges (reciprocal-space Ewald for periodic systems)
joins the total energy on one autograd graph, so forces and stress need no
extra code.

The implementation is **not vendored** — `ecenet.les.LESLongRange` wraps the
inventors' reference package, installed separately (pinned; it is not on PyPI):

```bash
pip install -e ".[les]"     # or directly:
pip install "les @ git+https://github.com/ChengUCB/les@c8063fad18e3d59cb4d783e0ed5a1efea8d55b8d"
```

**Joint training** (`use_les=True`) is available on all four trainers, DDP
included:

```python
from scripts.train_ecenet_xyz import train_ecenet_xyz
model, les_module, results = train_ecenet_xyz(
    train_path='data/train-H2O_RPBE-D3.xyz',
    test_path='data/test-H2O_RPBE-D3.xyz',
    use_les=True, n_epochs=200)
```

`les_readout` selects how the latent charge is produced:

| `les_readout` | aggregation |
| --- | --- |
| `'sum'` (default) | parameter-free scatter-sum of edge invariants; upstream's head maps it to charges |
| `'softmax'` | attention-weighted read-out (intensive, distance-decaying) |
| `'edge'` | Allegro-LES-style: a linear per-edge charge, summed per atom |
| `'edge_basis'` | per-edge charge head mirroring the energy readout (learnable distance profile, vanishes at `r_cut`) |

Further knobs (docstrings have the reasoning): `les_charge_scale` (fixed
multiplier on the edge-mode latent charge, à la MACE-LES's `output_scale`),
`les_dipole=True` (edge modes: every atom also gets a latent dipole `u`, fed
to upstream's charge–dipole and dipole–dipole Ewald terms — polarization the
fixed charges cannot express; molecular dipole `μ = Σ qᵢrᵢ + Σ uᵢ`), and
`les_charges=False` (dipoles-only ablation).

**MD and evaluation.** `ECENetLESCalculator` loads a joint checkpoint and
evaluates `E = E_sr + E_lr` on one graph — forces from the joint backward,
stress from a strain pass that covers the Ewald term's cell dependence
(verified against finite differences). It refuses short-range checkpoints,
and `ECENetCalculator` symmetrically refuses LES ones (`ignore_les=True`
overrides). `examples/run_md_xyz.py` picks the right calculator
automatically:

```python
from ecenet.calculator import ECENetLESCalculator
atoms.calc = ECENetLESCalculator.from_checkpoint('water_les.mdl')
print(atoms.get_potential_energy())   # E_sr + E_lr, eV
```

Every force call also exposes the latent charges via `atoms.get_charges()`
(and dipoles as `calc.results['les_dipoles']`); `calc.compute_bec(atoms)`
returns Born effective charges `Z* = ∂P/∂r` with charge-flow terms included.
`run_md_xyz --dump_charges` / `--dump_bec` write them onto every dumped frame
as extxyz columns (`les_q`, `les_u`, `bec`), giving charge/dipole/BEC
trajectories along MD. The global sign of the latent charges is arbitrary
(`E_lr` is quadratic in `q`): consistent within a checkpoint, not physically
pinned.

> **IP / licensing.** The `les` package is CC BY-NC 4.0 (**non-commercial**);
> it is an optional dependency and none of its code is included in this
> repository — installing it means accepting its terms. The Latent Ewald
> Summation algorithm additionally has a UC Berkeley provisional patent
> (academic use unrestricted). This repository's own license covers only the
> code in this repository and grants no rights to either.

## Trainer options

**Learning-rate schedules** — `lr_schedule='plateau'` (default) |
`'multistep'` (`lr_milestones`, `lr_gamma`) | `'cosine'` (`lr_min_factor`),
plus `warmup_epochs` for the latter two. `multistep` and `cosine` are pure
functions of the epoch index — resume-exact, nothing in the checkpoint, and
every DDP rank computes the same LR independently.

```python
train_ecenet(..., lr_schedule='cosine', warmup_epochs=5, lr_min_factor=0.01)
```

**Size-aware batching** (SPICE trainer, and the MPtrj prepared-shard mode) —
`bucket=True` batches similar-sized structures; `max_atoms_per_batch=250`
packs to a total-atom budget so per-step memory/compute is roughly uniform;
`max_batch_count` caps structures per batch; `bucket_sort=False` trades a
little load balance for batch diversity. All modes keep every DDP rank on the
same batch count (the collective in backward deadlocks otherwise) via a
deterministic round-alignment scheme — see the trainer docstrings.

**Other** — `precompute_topology=True` (SPICE) builds neighbour lists once at
startup (numerics-identical, skips per-step GPU syncs); `tf32=True` routes
float32 matmuls to TF32 tensor cores (A/B the validation MAE before trusting
it; float64 warns and changes nothing).

## Fused kernels (optional)

Two opt-in fused paths trade nothing numerically for memory (and, with Triton
on CUDA, HBM traffic). Both are runtime toggles on the model, off by default:

```python
model.set_edge_frame_fused(True)      # gather→Wigner-rotate→reshape as one op,
                                      # + the MP layers' pack/unrotate (e2n=True)
model.set_activation_fused(True)      # nonlinearity grid recomputed in backward
```

`set_edge_frame_fused` is safe for double-backward force-loss training;
`set_activation_fused` is single-backward oriented — leave it off when
training with a force loss. On CUDA + float32 both dispatch to Triton kernels;
elsewhere they run equivalent pure-PyTorch fallbacks. Verified bit-identical
(CPU) / fp32-accurate (kernels) in the kernel test files.

## License

Copyright ©2026. The Regents of the University of California (Regents). All
Rights Reserved. Permission to use, copy, modify, and distribute this software
and its documentation is hereby granted, provided that the above copyright
notice, this paragraph and the following two paragraphs appear in all copies,
modifications, and distributions.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

See [`LICENSE`](LICENSE) for the full text.

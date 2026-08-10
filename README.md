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

examples/              runnable examples + a small example checkpoint
  run_md_*.py            ASE MD drivers (NVT/NPT); importable or CLI
  ethanol.mdl            legacy rMD17 ethanol model (edge MP; now rejected on load —
                         kept as the fixture for the legacy-checkpoint test)

tests/                 test suite (test_*.py, run from the repo root)
tools/                 developer utilities (run from the repo root)
  profile_step.py        profile a single calculator step
  equiv_vs_ref.py        numerical-equivalence check vs a git ref
  predict_charges.py     LES latent charges + E_lr for one frame of an ASE file
  eval_spice_dipoles.py  latent-charge dipoles vs DFT reference dipoles
                         (ChengUCB/les_fit SPICE test slices, downloaded separately)
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

Run everything from the repo root so `import ecenet` resolves. All three trainers
are **import-and-call** — every option is a keyword argument of the training
function. The multi-GPU trainers (`train_ecenet_spice`, `train_ecenet_mptrj`)
additionally keep a `__main__` entry point so they launch directly under
`torchrun`; set hyperparameters in the call at the bottom of the script (or
import the function from your own driver).

Train on an rMD17 / MD22 molecule (import-and-call):

```python
from scripts.train_ecenet import train_ecenet
model, results = train_ecenet(molecule='ethanol', n_train=950,
                              l_max=3, n_max=4, embed_dim=16, n_epochs=200,
                              n_mp=2)            # n_mp ≥ 2 turns on message passing
```

Optional low-rank equivariant layers (off by default, available on every trainer
as well as `ecenet.ECENet(...)`) — down → nonlinearity at `r` → up, with a
zero-init up-projection, so each layer is the identity at initialisation:

```python
train_ecenet(..., bottleneck_dim=16)
```

### FiLM gate

`element_film=True` modulates the edge features once — right after they are
built and rotated into the bond frame, before the equivariant layer stack. A
small MLP on `[embed(type_i), embed(type_j), φ(r_ij)]` predicts a scale `γ`
(and optionally a shift `β`), applied as `γ⊙A + β`:

```python
train_ecenet(..., element_film=True, film_n_rbf=8)   # element + distance
train_ecenet(..., element_film=True)                 # element-only (film_n_rbf=0)
```

The gate MLP's last layer is zero-init, so `γ=1`, `β=0`, and the model is
unchanged at initialisation — the gate learns away from the identity.

| option | effect |
| --- | --- |
| `film_n_rbf` | size of the radial leg `φ(r_ij)`; `0` (default) makes the gate element-only, so `γ` does not vary with bond length |
| `film_embed_dim` | width of each element embedding (default 16) |
| `film_hidden` | gate-MLP hidden width(s); `None` → `[max(2*C, 32)]` |
| `film_per_m` | emit a scale per `(channel, m)` instead of one per channel broadcast over `m` |
| `film_shift` | also predict a shift `β`, as an extra head on the *same* MLP |

Both extras keep the symmetry intact, which constrains where they may act.
`A_cos` and `A_sin` of a given `(channel, m)` share `γ`, so a per-`m` scale still
commutes with the bond frame's per-mode SO(2) rotation; the structural-zero slots
(`m > l` of that channel) are masked to `γ=1`. An additive shift does *not*
commute with that rotation, so `β` lands on the `m=0` slot of `A_cos` only — the
rotation-invariant mode, and the one place a shift is exactly equivariant.

### Message passing

With `n_mp >= 2`, every message-passing layer computes, per edge, a low-rank
**message** and an invariant scalar **score**, aggregates the messages at each
receiver atom in the common global frame, and passes the result through a
**receiver** transform back in the bond frame. `mp_type` selects how the
per-edge weight is formed:

```python
train_ecenet(..., n_mp=2, mp_type='softmax', mp_n_heads=4)   # default
train_ecenet(..., n_mp=2, mp_type='sum')
```

| `mp_type` | weight | behaviour |
| --- | --- | --- |
| `'softmax'` (default) | `exp(s)·f_cut / Σ_{e→j}(exp(s)·f_cut)` | softmax over the receiver's incoming edges — a weighted *average*, intensive in coordination |
| `'sum'` | `s·f_cut` | raw signed score × cutoff envelope — *extensive* in coordination, and signed, so a neighbour can contribute negatively |

Either way the smooth cutoff envelope keeps the energy continuous as an edge
crosses `r_cut_edge`.

`mp_msg_envelope` (on by default) makes the aggregated message decay with
*absolute* distance. It matters only for `'softmax'`: the softmax normalizer
divides the absolute `f_cut` back out, leaving only the relative cutoff across a
receiver's in-edges — so without it a lone neighbour near `r_cut` still gets
weight ≈ 1 and the message is essentially flat in distance. Multiplying `f_cut`
back in fixes that (measured on a dimer, the weight then equals `f_cut` exactly,
falling ~32× from 1.5 Å to 4.5 Å instead of staying at 1.000). `'sum'` is already
enveloped by construction, so the flag is a no-op there — and setting it `False`
warns rather than silently doing nothing.

Message and scores share **one fused trunk**: a low-rank block (down →
nonlinearity at `mp_dim` → up) whose up-projection emits `2*embed_dim*(l_max+1)`
message channels plus one score channel per head, the score being that channel's
`m=0` (rotation-invariant) component. Sharing a trunk is cheaper than a separate
message block and score head. The up-projection is zero-init, so at
initialisation the message residual and every score are 0 — which makes `'sum'`
an exact no-op and leaves `'softmax'` with uniform attention (`exp(0) = 1`).

`mp_dim` sets that trunk's bottleneck width (and the receiver's); `mp_n_heads`
splits the value channels (`2*embed_dim`) into that many attention heads, so it
must divide them evenly.

`mp_l_attention` gives each head one score **per degree `l`** instead of one
overall, so a neighbour can be weighted differently for `l=1` than for `l=2`.
Each `(head, l)` then runs its own independent softmax over the receiver's
in-edges, and the fused trunk widens to `n_ch + n_heads*(l_max+1)`. This stays
equivariant because the Wigner-D block is `l`-diagonal: an invariant scalar
applied uniformly across one `l`'s whole `m`-block commutes with the rotation.
Splitting *within* an `l`, across `m`, would not — which is why the weight is
expanded through a fixed `l_of_s` map rather than being free per spherical
index.

> **Note.** The older distance/type-weighted `mp_type='edge'` message passing has
> been removed. Checkpoints trained with it (identifiable by `W_msg` weights) are
> rejected with an explicit error rather than silently loading — retrain with
> `'softmax'` or `'sum'`.

### Long-range electrostatics (LES, optional)

ECENet's message passing sees only atoms within `r_cut_edge`. The optional
**LES** add-on (Latent Ewald Summation) closes that gap: a head predicts a
scalar latent charge per atom from the model's invariant `l0` embedding, and
the long-range energy is the smeared-Coulomb interaction between those charges
(reciprocal-space Ewald for periodic systems), summed into the total energy on
one shared autograd graph so forces and stress need no extra code.

The implementation is **not vendored** — `ecenet.les.LESLongRange` wraps the
inventors' reference package, installed separately (pinned; it is not on PyPI):

```bash
pip install -e ".[les]"     # or directly:
pip install "les @ git+https://github.com/ChengUCB/les@c8063fad18e3d59cb4d783e0ed5a1efea8d55b8d"
```

`LESLongRange()(l0, positions, cell=None, batch=None)` returns the long-range
energy; the upstream package's own head maps `l0` to latent charges, so the
wrapper's state dict is exactly the upstream module's. The model exposes `l0`
on every forward variant — `l0_only=True` skips the `l=1` work (`l0` is
rotation-invariant, so it needs no frame change; a charge is a scalar, so the
head never sees `l1`):

```python
lr = ecenet.LESLongRange()
E_sr, l0 = model(pos, types, return_embeddings=True, l0_only=True)
E = E_sr + lr(l0, pos).sum()      # one autograd graph → forces via autograd
```

The batched paths (`forward_batch`, `forward_batch_multi`) return `l0` as a
per-structure list.

**Joint training** is available for small datasets:
`scripts/train_ecenet_xyz.py` is a single-process, in-memory trainer for any
ASE-readable file (extxyz etc.) that minimises `E = E_sr + E_lr` on one
autograd graph when `use_les=True` — forces come from the same graph, and
stress strain-transforms the cell alongside positions and shifts so the
Ewald part is covered too:

```python
from scripts.train_ecenet_xyz import train_ecenet_xyz
model, les_module, results = train_ecenet_xyz(
    train_path='data/train-H2O_RPBE-D3.xyz',
    test_path='data/test-H2O_RPBE-D3.xyz',
    use_les=True, n_epochs=200)
```

`les_readout` selects how the per-atom `l0` fed to the charge head is
aggregated from the final edge invariants (available on all trainers and
`ecenet.ECENet(...)`):

| `les_readout` | aggregation |
| --- | --- |
| `'sum'` (default) | parameter-free scatter-sum over the atom's in-edges — extensive in coordination |
| `'softmax'` | attention: a zero-init linear score on each edge's invariants, segment-softmax over the receiver's in-edges with `f_cut` as a multiplicative log-bias, envelope multiplied back in (exactly the MP layers' softmax + `mp_msg_envelope` recipe) — intensive, and decaying with absolute distance |
| `'edge'` | Allegro-LES-style per-edge charge decomposition: a linear scalar head on each edge's invariants, scatter-summed per atom — the width-1 `l0` **is** the latent charge, so upstream's atomwise head is bypassed (the wrapper is called with `l0_is_charge=True`; standard init, since a zero-init charge head would sit on the quadratic energy's gradient-free saddle) |
| `'edge_basis'` | `'edge'` upgraded to mirror the per-edge **energy** readout end to end: an MLP with the energy head's architecture (same full invariant input set, hidden widths, and activation — but standard init, since near-zero charges would start on the quadratic energy's saddle) emits `n_max_d` channels dotted with the cutoff-enveloped radial basis of the bond length, so each bond's charge contribution has a learnable distance profile and vanishes exactly at `r_cut` |

The softmax weight is an invariant scalar shared by the `l0` and `l1`
messages, so SO(3) behaviour is untouched (verified in `tests/test_les.py`,
including the closed-form dimer weight `f_cut²/(f_cut+ε)`).

For the edge modes, `les_charge_scale` (default 1.0) multiplies the emitted
latent charge by a fixed factor (MACE-LES ships 0.1 as `output_scale`): with
a standard-init head the charges then start small but nonzero — clear of the
quadratic energy's q = 0 saddle, while E_lr (quadratic in q) is suppressed
~scale² early, so the short-range fit leads and the charges grow gently
relative to it. It is not a parameter and is recorded in the checkpoint's
hparams; for `'sum'`/`'softmax'` it cannot apply (the charge is produced
inside the upstream head) and setting it warns.

The SPICE trainer takes the same flags (`use_les=True`, `les_arguments`) and
trains jointly under DDP: the LES head lives inside the DDP-wrapped forward
module (so its gradients join the bucket reduction and run on every step,
keeping `find_unused_parameters=False` valid), and the long-range energy is
computed in **one batched LES call** per step — concatenated atoms plus a
structure-index vector, zero cells → the isolated pairwise path (verified
bit-identical to per-structure calls in `tests/test_spice_trainer.py`).

Upstream builds its charge head lazily on the first forward, so both trainers
materialise it with one throwaway forward before the DDP wrap / optimiser /
checkpoint restore. LES checkpoints carry a top-level `les` dict;
`ECENetCalculator.from_checkpoint` **refuses** them rather than silently
dropping the long-range term (`ignore_les=True` loads the short-range part
deliberately). An LES-aware calculator and joint training in the MPtrj
trainer are not yet ported.

> **IP / licensing.** The `les` package is CC BY-NC 4.0 (**non-commercial**);
> it is an optional dependency and none of its code is included in this
> repository — installing it means accepting its terms. The Latent Ewald
> Summation algorithm additionally has a UC Berkeley provisional patent
> (academic use unrestricted). This repository's own license covers only the
> code in this repository and grants no rights to either.

### Fused kernels (optional)

Two opt-in fused paths trade nothing numerically for memory (and, with Triton
on CUDA, HBM traffic). Both are runtime toggles on the model, off by default:

```python
model.set_edge_frame_fused(True)      # gather→Wigner-rotate→reshape as one op,
                                      # + the MP layers' pack/unrotate (e2n=True)
model.set_activation_fused(True)      # nonlinearity grid recomputed in backward
```

`set_edge_frame_fused` re-gathers in the backward instead of saving the
`(n_edges, 2C, n_sph)` intermediates; its backward is built from differentiable
ops, so it is safe for double-backward force-loss training.
`set_activation_fused` drops the `(n_edges, F, n_grid)` grid transient from the
saved-for-backward set; it is single-backward oriented — leave it off for
force-loss training. On CUDA + float32 both dispatch to Triton kernels
(`ecenet/edge_frame_kernel.py`, `ecenet/realspace_kernel.py`); elsewhere they
run equivalent pure-PyTorch fallbacks. Verified bit-identical (CPU) /
fp32-accurate (kernels) in `tests/test_edge_frame_kernel.py` and
`tests/test_realspace_kernel.py`.

### Learning-rate schedule

All three trainers take `lr_schedule`, defaulting to `'plateau'`
(`ReduceLROnPlateau` on the validation metric, as before):

```python
train_ecenet(..., lr_schedule='multistep',
             lr_milestones=[80, 130, 170, 190], lr_gamma=0.5)

train_ecenet(..., lr_schedule='cosine', warmup_epochs=5, lr_min_factor=0.01)
```

| option | applies to | effect |
| --- | --- | --- |
| `lr_milestones` | `multistep` | epochs at which the LR is multiplied by `lr_gamma` |
| `lr_gamma` | `multistep` | decay factor at each milestone (default `0.1`) |
| `warmup_epochs` | `multistep`, `cosine` | linear ramp `0 → lr` over the first N epochs |
| `lr_min_factor` | `cosine` | LR floor as a fraction of `lr`, reached exactly on the last epoch |
| `scheduler_patience` | `plateau` | unchanged |

`multistep` and `cosine` are computed as **pure functions of the epoch index**
rather than through torch's stateful schedulers. That has three consequences
worth knowing: a resumed run lands on exactly the LR a fresh run would have at
that epoch (torch's `MultiStepLR` counts `.step()` calls, so it would replay from
the initial LR); nothing needs to go in the checkpoint; and every DDP rank
computes the same LR independently, with no state to keep in sync. The
`multistep` curve is verified identical to `torch.optim.lr_scheduler.MultiStepLR`
over a full run.

### Batching and precision (SPICE / MPtrj trainers)

Two size-aware batching modes for the SPICE trainer, both off by default:

```python
train_ecenet_spice(..., bucket=True)               # size-bucketed, fixed batch_size
train_ecenet_spice(..., max_atoms_per_batch=250)   # atom budget; implies bucket
train_ecenet_spice(..., max_atoms_per_batch=250, max_batch_count=16)
```

`bucket=True` sorts the epoch's structures by atom count and batches consecutive
ones, so a batch holds similar-sized molecules rather than one giant molecule
alongside several small ones (measured: mean within-batch atom spread 0.2 vs 44.1
unsorted). `max_atoms_per_batch` goes further and packs to a total-atom budget, so
memory and compute per step are roughly uniform and a batch of several large
molecules can no longer OOM; `max_batch_count` optionally caps structures per
batch, bounding the per-structure Python overhead when a batch is all tiny
molecules.

`bucket_sort=True` (the default) sorts by atom count before packing.
`bucket_sort=False` greedy-packs the already-shuffled order instead, trading a
little DDP balance for batch diversity: with sorting, the largest, rare-size
structures keep the *same* batch-mates every epoch (measured: they retain 84% of
their batch-mates across epochs, vs 10% unsorted), which costs gradient diversity
exactly where there is least data. The atom budget still bounds per-batch cost
either way, so the per-rank load spread at `world_size=8` only loosens from 0.9%
to 2.4%. Mainly meaningful together with `max_atoms_per_batch` — without a budget,
unsorted packing is just fixed-size batches plus the round alignment.

Both share the cross-rank alignment, which is what makes them useful under DDP.
Every rank must run the same *number* of batches or the collective in backward
deadlocks, so the assignment is derived identically on every rank rather than
communicated: sort by atom count, form batches, group into rounds of
`world_size`, drop any partial final round, shuffle the round order with a shared
seed, and give rank r the r-th batch of each round. Adjacent (similar-cost)
batches share a round, so per-step work is aligned across ranks and the
molecule-size straggler disappears — the biggest win multi-node. Measured spread
in per-rank total atoms at `world_size=8`: 3.4% bucketed, 0.9% atom-budget.

`tf32=True` (both trainers) routes float32 matmuls to TF32 tensor cores on
Ampere+. TF32 keeps ~10 mantissa bits, so A/B the validation MAE before trusting
it. It is a float32-only mode: under `dtype=torch.float64` it warns and changes
nothing.

Train on SPICE dataset (10 elements):

```python
from scripts.train_ecenet_spice import train_ecenet_spice
model, results = train_ecenet_spice(l_max=3, n_max=4, embed_dim=32, n_layers=2)
```

Multi-GPU via `torchrun`
(`LOCAL_RANK`/`RANK`/`WORLD_SIZE` are read from the environment for DDP):

```bash
python scripts/train_ecenet_spice.py            # single process

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

## Tests

The test suite is pure PyTorch and runs on CPU. Each file is runnable as a script:

```bash
python tests/test_ecenet.py                  # ECENet integration: SO(3) invariance, forces, MP
python tests/test_bottleneck.py              # low-rank layers: identity at init, SO(3)
python tests/test_element_film.py            # FiLM gate: identity at init, SO(3), per-m, shift
python tests/test_spice_trainer.py            # SPICE trainer: atom-budget batching, DDP invariant
python tests/test_attention_mp.py            # attention MP: SO(3), cutoff continuity, sum vs softmax
python tests/test_les.py                     # LES: l0/l1 read-out SO(3) + batch/PBC consistency; wrapper lazy import
python tests/test_edge_frame_kernel.py       # fused edge-frame/e2n: gradchecks, model on/off equality (Triton legs need CUDA)
python tests/test_realspace_kernel.py        # fused nonlinearity: backward equivalence, DFT precision (Triton legs need CUDA)
python tests/test_mptrj_trainer.py           # end-to-end MPtrj trainer smoke (synthetic)
python tests/test_xyz_trainer.py             # small-dataset trainer: smoke, LES resume, force-FD through E_lr
```

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

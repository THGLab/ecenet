"""Synthetic smoke test for train_ecenet_spice.py, focused on atom-budget batching.

No SPICE download needed: writes a small extended-XYZ file with a wide spread of
molecule sizes and runs the trainer over it.

The atom-budget path is the part worth pinning down. Under DDP every rank must
run the same number of batches or the collective in backward deadlocks — a hang,
not an error — so the rank-assignment invariant is checked directly across
simulated world sizes, alongside an end-to-end run.

Run:  python tests/test_spice_trainer.py
"""

import os
import sys  # repo root + scripts/ on path (imports ecenet and the scripts/ trainer)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))


import tempfile

import numpy as np
import torch
from train_ecenet_spice import size_aware_batches, train_ecenet_spice

DTYPE = torch.float64
DEVICE = torch.device('cpu')
ELEMENTS = ['H', 'C', 'N', 'O']


def write_xyz(path, n_structures, seed=0, n_atoms_range=(3, 25)):
    """Extended XYZ in the format the SPICE parser expects, with a wide size spread."""
    rng = np.random.RandomState(seed)
    with open(path, 'w') as f:
        for _ in range(n_structures):
            na = rng.randint(*n_atoms_range)
            f.write(f"{na}\n")
            f.write(f"Properties=species:S:1:pos:R:3:forces:R:3 energy={rng.uniform(-20, 20):.6f}\n")
            pos = rng.uniform(-4, 4, size=(na, 3))
            frc = rng.uniform(-1, 1, size=(na, 3))
            for i in range(na):
                el = ELEMENTS[rng.randint(len(ELEMENTS))]
                f.write(f"{el} {pos[i,0]:.6f} {pos[i,1]:.6f} {pos[i,2]:.6f} "
                        f"{frc[i,0]:.6f} {frc[i,1]:.6f} {frc[i,2]:.6f}\n")
    return path


def test_ddp_invariant_both_modes():
    """Every rank must get the same NUMBER of batches (else DDP deadlocks), and no
    structure may be used twice per epoch. Holds for size-bucketing (fixed
    batch_size) and for atom-budget packing, which must also respect the budget."""
    rng = np.random.RandomState(0)
    n_atoms = rng.randint(3, 60, size=4000).astype(np.int64)
    BUDGET = 250
    for label, kw in (('bucket   ', dict(batch_size=8)),
                      ('budget   ', dict(max_atoms_per_batch=BUDGET))):
        for world_size in (1, 2, 4, 8):
            for epoch in (0, 1, 7):
                all_idx = np.random.RandomState(10 + epoch).choice(4000, 3000, replace=False)
                per_rank = [size_aware_batches(all_idx, n_atoms, world_size, r,
                                               epoch, 10, **kw)
                            for r in range(world_size)]
                counts = {len(b) for b in per_rank}
                assert len(counts) == 1, \
                    f"{label} world_size={world_size} epoch={epoch}: ranks disagree " \
                    f"on batch count {counts} — this deadlocks DDP"
                if 'max_atoms_per_batch' in kw:
                    over = [int(n_atoms[b].sum()) for rb in per_rank for b in rb
                            if n_atoms[b].sum() > BUDGET]
                    assert not over, f"budget {BUDGET} exceeded: {over[:3]}"
                else:
                    sizes = {len(b) for rb in per_rank for b in rb}
                    assert sizes <= {8}, f"bucket mode should give fixed batches: {sizes}"
                used = np.concatenate([np.concatenate(b) for b in per_rank])
                assert len(used) == len(set(used.tolist())), "a structure was used twice"
            n_b = counts.pop()
            totals = [int(n_atoms[np.concatenate(b)].sum()) for b in per_rank]
            spread = (max(totals) - min(totals)) / max(totals)
            assert spread < 0.05, \
                f"{label} ranks poorly balanced at world_size={world_size}: {spread:.1%}"
            print(f"  {label} world_size={world_size}: {n_b} batches on every rank, "
                  f"rank load spread {spread:.1%}")


def test_bucket_groups_similar_sizes():
    """The point of bucketing: batches contain similar-sized structures, so no
    step is dominated by one giant molecule padded out with small ones."""
    rng = np.random.RandomState(1)
    n_atoms = rng.randint(3, 60, size=2000).astype(np.int64)
    all_idx = np.arange(2000)
    bucketed = size_aware_batches(all_idx, n_atoms, 1, 0, 0, 10, batch_size=8)
    unsorted = [all_idx[i:i + 8] for i in range(0, 2000, 8)]
    spread_b = np.mean([n_atoms[b].max() - n_atoms[b].min() for b in bucketed])
    spread_u = np.mean([n_atoms[b].max() - n_atoms[b].min() for b in unsorted])
    assert spread_b < 0.2 * spread_u, \
        f"bucketing did not group sizes: within-batch spread {spread_b:.1f} vs {spread_u:.1f}"
    print(f"  bucket: mean within-batch atom spread {spread_b:.1f} vs {spread_u:.1f} unsorted")


def test_bucket_sort_controls_batch_diversity():
    """Why bucket_sort=False exists. Sorting groups similar sizes, which tightens
    DDP balance but makes the largest, rare-size structures co-occur with the SAME
    neighbours every epoch — least data, least gradient diversity. Greedy-packing
    the shuffled order mixes them freshly each epoch, and the atom budget still
    bounds per-batch cost either way.

    Note this is about *who shares a batch*, not literal batch identity: structures
    of equal atom count are tie-broken by the epoch's shuffle, so even sorted
    batches are not byte-identical across epochs.
    """
    rng = np.random.RandomState(0)
    n_atoms = rng.randint(3, 60, size=3000).astype(np.int64)
    BUDGET = 250
    biggest = set(np.argsort(n_atoms)[-30:].tolist())

    def epoch_batches(bucket_sort, epoch):
        all_idx = np.random.RandomState(100 + epoch).permutation(3000)
        return size_aware_batches(all_idx, n_atoms, 1, 0, epoch, 10,
                                  max_atoms_per_batch=BUDGET, bucket_sort=bucket_sort)

    def neighbours_of_biggest(batches):
        out = set()
        for b in batches:
            if biggest & set(b.tolist()):
                out |= set(b.tolist())
        return out

    results = {}
    for bucket_sort in (True, False):
        e0, e1 = epoch_batches(bucket_sort, 0), epoch_batches(bucket_sort, 1)
        assert max(int(n_atoms[b].sum()) for b in e0) <= BUDGET, "budget exceeded"
        spread = np.mean([n_atoms[b].max() - n_atoms[b].min() for b in e0])
        p0, p1 = neighbours_of_biggest(e0), neighbours_of_biggest(e1)
        jaccard = len(p0 & p1) / len(p0 | p1)
        results[bucket_sort] = (spread, jaccard)
        print(f"  bucket_sort={str(bucket_sort):5s}: within-batch atom spread "
              f"{spread:5.1f}, the 30 largest keep {jaccard:.0%} of their "
              f"batch-mates across epochs")

    spread_on, jac_on = results[True]
    spread_off, jac_off = results[False]
    assert spread_on < 0.1 * spread_off, \
        f"sorting should group sizes: {spread_on:.1f} vs {spread_off:.1f}"
    assert jac_on > 0.5, \
        f"sorted: the largest should keep their batch-mates, got {jac_on:.2f}"
    assert jac_off < 0.3, \
        f"unsorted: the largest should get fresh batch-mates, got {jac_off:.2f}"


def test_bucket_sort_false_keeps_ddp_aligned():
    """Diversity must not cost the DDP invariant: unsorted packing still gives
    every rank the same batch count (only the per-rank load spread loosens)."""
    rng = np.random.RandomState(0)
    n_atoms = rng.randint(3, 60, size=4000).astype(np.int64)
    for world_size in (2, 8):
        all_idx = np.random.RandomState(5).permutation(4000)[:3000]
        per_rank = [size_aware_batches(all_idx, n_atoms, world_size, r, 0, 10,
                                       max_atoms_per_batch=250, bucket_sort=False)
                    for r in range(world_size)]
        counts = {len(b) for b in per_rank}
        assert len(counts) == 1, f"unsorted packing desynced ranks: {counts}"
        used = np.concatenate([np.concatenate(b) for b in per_rank])
        assert len(used) == len(set(used.tolist())), "a structure was used twice"
        totals = [int(n_atoms[np.concatenate(b)].sum()) for b in per_rank]
        spread = (max(totals) - min(totals)) / max(totals)
        assert spread < 0.20, f"unsorted load spread too large at ws={world_size}: {spread:.1%}"
        print(f"  bucket_sort=False world_size={world_size}: {counts.pop()} batches on "
              f"every rank, rank load spread {spread:.1%}")


def test_atom_budget_respects_max_batch_count():
    n_atoms = np.full(500, 4, dtype=np.int64)          # tiny molecules
    idx = np.arange(500)
    b = size_aware_batches(idx, n_atoms, 1, 0, 0, 0,
                           max_atoms_per_batch=1000, max_batch_count=7)
    assert max(len(x) for x in b) <= 7, f"max_batch_count ignored: {max(len(x) for x in b)}"
    print(f"  max_batch_count caps structures/batch at 7 (budget alone would allow "
          f"{1000 // 4})")


def test_degenerate_epoch_keeps_ranks_aligned():
    """Fewer batches than ranks is degenerate, but must NOT desync: every rank
    still gets exactly one batch (ranks then share structures)."""
    n_atoms = np.full(4, 10, dtype=np.int64)
    per_rank = [size_aware_batches(np.arange(4), n_atoms, 8, r, 0, 0,
                                   max_atoms_per_batch=100)          # 1 batch, 8 ranks
                for r in range(8)]
    assert {len(b) for b in per_rank} == {1}, "degenerate epoch desynced the ranks"
    print("  fewer batches than ranks: every rank still gets 1 batch (no deadlock)")


def test_trainer_runs_with_atom_budget():
    """End-to-end: the trainer runs on both batching paths, and rejects a budget
    smaller than the largest structure rather than looping forever."""
    with tempfile.TemporaryDirectory() as d:
        train = write_xyz(os.path.join(d, 'train.xyz'), 60, seed=1)
        test = write_xyz(os.path.join(d, 'test.xyz'), 10, seed=2)
        common = dict(
            train_xyz=train, test_xyz=test, n_train=48, n_val=6, n_test=6,
            r_cut_edge=4.0, r_cut_neighbor=4.0, l_max=2, n_max=2, embed_dim=8,
            n_layers=1, n_max_d=4, m_max=2, n_epochs=2, lr=5e-3,
            eval_every=2, eval_batch_size=4, dtype=DTYPE, device=DEVICE,
            seed=0, verbose=False,
        )
        _, base = train_ecenet_spice(batch_size=4, **common)
        _, budg = train_ecenet_spice(batch_size=4, max_atoms_per_batch=60, **common)
        _, buck = train_ecenet_spice(batch_size=4, bucket=True, **common)
        _, nosort = train_ecenet_spice(batch_size=4, max_atoms_per_batch=60,
                                       bucket_sort=False, **common)
        for r in (base, budg, buck, nosort):
            assert np.isfinite(r['test_force_mae']), f"non-finite force MAE: {r}"

        # a budget below the largest structure can never be satisfied
        try:
            train_ecenet_spice(batch_size=4, max_atoms_per_batch=2, **common)
        except ValueError as e:
            assert 'largest training structure' in str(e)
        else:
            raise AssertionError("expected a ValueError for an impossible budget")
    print(f"  trainer runs on all three paths (fixed F={base['test_force_mae']:.3f}, "
          f"bucket F={buck['test_force_mae']:.3f}, "
          f"atom-budget F={budg['test_force_mae']:.3f}); impossible budget rejected")


def test_precomputed_topology_matches_on_the_fly():
    """build_topology + forward_batch_multi(topology=...) must be bit-identical
    to the on-the-fly nonzero path — energies, forces, AND parameter grads —
    including a zero-edge single-atom structure. Then end-to-end: a trainer run
    with precompute_topology=True reproduces the plain run exactly."""
    from ecenet import ECENet

    torch.manual_seed(0)
    model = ECENet(n_types=4, r_cut_edge=4.0, r_cut_neighbor=4.0,
                   l_max=2, n_max=2, embed_dim=8, n_layers=2, n_mp=2,
                   n_max_d=4).double()
    rng = np.random.RandomState(7)
    pos_list = [torch.tensor(rng.uniform(-3, 3, size=(n, 3)), dtype=DTYPE)
                for n in (5, 1, 8)]     # incl. a zero-edge single atom
    typ_list = [torch.tensor(rng.randint(0, 4, size=p.shape[0])) for p in pos_list]

    topo = model.build_topology(pos_list)

    def run(topology):
        pos_rg = [p.detach().clone().requires_grad_(True) for p in pos_list]
        e = model.forward_batch_multi(pos_rg, typ_list, topology=topology)
        grads = torch.autograd.grad(e.sum(), pos_rg, create_graph=True,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(pos_rg[k])
                 for k, g in enumerate(grads)]
        loss = e.sum() + sum((g ** 2).sum() for g in grads)
        model.zero_grad()
        loss.backward()
        pgrads = [p.grad.clone() if p.grad is not None else None
                  for p in model.parameters()]
        return e.detach(), [g.detach() for g in grads], pgrads

    e_fly, f_fly, pg_fly = run(None)
    e_pre, f_pre, pg_pre = run(topo)

    de = (e_fly - e_pre).abs().max().item()
    df = max((a - b).abs().max().item() for a, b in zip(f_fly, f_pre))
    dp = max((a - b).abs().max().item()
             for a, b in zip(pg_fly, pg_pre) if a is not None)
    assert de == 0.0 and df == 0.0 and dp == 0.0, \
        f"precomputed topology diverges: dE={de:.3e} dF={df:.3e} dgrad={dp:.3e}"
    print(f"  precomputed topology == on-the-fly (dE={de:.1e}, dF={df:.1e}, "
          f"dparam-grad={dp:.1e}, incl. zero-edge structure)")

    # End-to-end: same seed, with vs without precompute → identical metrics.
    with tempfile.TemporaryDirectory() as d:
        train = write_xyz(os.path.join(d, 'train.xyz'), 40, seed=4)
        test = write_xyz(os.path.join(d, 'test.xyz'), 8, seed=5)
        common = dict(
            train_xyz=train, test_xyz=test, n_train=32, n_val=4, n_test=6,
            r_cut_edge=4.0, r_cut_neighbor=4.0, l_max=2, n_max=2, embed_dim=8,
            n_layers=1, n_max_d=4, m_max=2, n_epochs=2, batch_size=4, lr=5e-3,
            eval_every=2, eval_batch_size=4, dtype=DTYPE, device=DEVICE,
            seed=0, verbose=False,
        )
        _, plain = train_ecenet_spice(**common)
        _, pre = train_ecenet_spice(precompute_topology=True, **common)
    dtrain = abs(plain['test_force_mae'] - pre['test_force_mae'])
    assert dtrain == 0.0, \
        f"precompute_topology changed the training trajectory: dF={dtrain:.3e}"
    print(f"  trainer with precompute_topology=True reproduces the plain run "
          f"exactly (F={pre['test_force_mae']:.4f})")


def _has_les():
    try:
        import les  # noqa: F401
        return True
    except ImportError:
        return False


def test_les_wrapper_batched_matches_single():
    """The wrapper's ONE batched LES call (concatenated atoms + batch vector,
    zero cells) must equal per-structure single LES calls — the path the xyz
    trainer validated against the analytic dimer."""
    if not _has_les():
        print("  SKIP LES wrapper consistency (`les` not installed)")
        return
    from train_ecenet_spice import _MultiForwardWrapper

    from ecenet import ECENet
    from ecenet.les import LESLongRange

    torch.manual_seed(0)
    model = ECENet(n_types=4, r_cut_edge=4.0, r_cut_neighbor=4.0,
                   l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4).double()
    rng = np.random.RandomState(5)
    pos_list = [torch.tensor(rng.uniform(-3, 3, size=(n, 3)), dtype=DTYPE)
                for n in (4, 1, 7)]     # incl. a zero-edge single atom
    typ_list = [torch.tensor(rng.randint(0, 4, size=p.shape[0])) for p in pos_list]

    les_mod = LESLongRange().double()
    with torch.no_grad():
        _, l0_list = model.forward_batch_multi(pos_list, typ_list,
                                               return_embeddings=True, l0_only=True)
        les_mod(l0_list[0], pos_list[0])   # materialise
        for p in les_mod.parameters():
            p.add_(0.1 * torch.randn_like(p))

        wrapped = _MultiForwardWrapper(model, les_mod)
        e_batched = wrapped(pos_list, typ_list)

        e_sr, l0_list = model.forward_batch_multi(pos_list, typ_list,
                                                  return_embeddings=True, l0_only=True)
        e_single = torch.stack([
            e_sr[b] + les_mod(l0_list[b], pos_list[b]).sum()
            for b in range(len(pos_list))])
    d = (e_batched - e_single).abs().max()
    assert d < 1e-10, f"batched LES != per-structure LES: {d:.3e}"
    print(f"  LES wrapper: batched call == per-structure calls (d={d:.1e}, "
          "incl. zero-edge structure)")

    # les_dipole variant: packed l0 [q | u] through the DDP wrapper's one
    # batched call vs per-structure calls (both on the vectorized path /
    # its per-structure limit)
    torch.manual_seed(1)
    model_d = ECENet(n_types=4, r_cut_edge=4.0, r_cut_neighbor=4.0,
                     l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
                     les_readout='edge_basis', les_dipole=True).double()
    with torch.no_grad():
        model_d.les_edge_charge.linears[-1].weight[model_d.n_max_d:
                                                   ].normal_(std=0.5)
        les_mod_d = LESLongRange().double()   # parameter-free (head bypassed)
        wrapped_d = _MultiForwardWrapper(model_d, les_mod_d)
        e_batched = wrapped_d(pos_list, typ_list)
        e_sr, l0_list = model_d.forward_batch_multi(
            pos_list, typ_list, return_embeddings=True, l0_only=True)
        e_single = torch.stack([
            e_sr[b] + les_mod_d(l0_list[b], pos_list[b], l0_is_charge=True,
                                les_dipole=True).sum()
            for b in range(len(pos_list))])
    dd = (e_batched - e_single).abs().max()
    assert dd < 1e-10, f"batched dipole LES != per-structure: {dd:.3e}"
    assert any(l0.shape[1] == 4 and l0[:, 1:].abs().max() > 0
               for l0 in l0_list), "dipoles are identically zero in the test"
    print(f"  LES wrapper: dipole (packed l0) batched == per-structure (d={dd:.1e})")


def test_trainer_runs_with_les():
    """End-to-end SPICE trainer with use_les=True: runs, finite MAEs, LES state
    checkpointed, resume restores it, use_les mismatch rejected."""
    if not _has_les():
        print("  SKIP LES trainer smoke (`les` not installed)")
        return
    with tempfile.TemporaryDirectory() as d:
        train = write_xyz(os.path.join(d, 'train.xyz'), 40, seed=4)
        test = write_xyz(os.path.join(d, 'test.xyz'), 8, seed=5)
        ckpt = os.path.join(d, 'spice_les.mdl')
        common = dict(
            train_xyz=train, test_xyz=test, n_train=32, n_val=4, n_test=4,
            r_cut_edge=4.0, r_cut_neighbor=4.0, l_max=2, n_max=2, embed_dim=8,
            n_layers=1, n_max_d=4, m_max=2, batch_size=4, lr=5e-3,
            eval_every=1, eval_batch_size=4, dtype=DTYPE, device=DEVICE,
            seed=0, verbose=False, checkpoint_path=ckpt,
        )
        _, r = train_ecenet_spice(use_les=True, les_readout='softmax',
                                  n_epochs=2, **common)
        assert np.isfinite(r['test_force_mae']), f"non-finite force MAE: {r}"
        assert r['les_module'] is not None
        saved = torch.load(ckpt, weights_only=False)
        assert 'les' in saved and saved['les']['state_dict'], "LES state not checkpointed"

        # resume continues with the LES head restored
        _, r2 = train_ecenet_spice(use_les=True, les_readout='softmax',
                                   n_epochs=3, **common)
        assert np.isfinite(r2['test_force_mae'])

        # use_les must match the checkpoint
        try:
            train_ecenet_spice(use_les=False, n_epochs=4, **common)
        except ValueError as e:
            assert 'use_les' in str(e)
        else:
            raise AssertionError("resume with use_les=False should have raised")
    print(f"  LES trainer smoke + checkpoint resume OK "
          f"(F={r['test_force_mae']:.3f})")


def test_tf32_is_a_noop_under_float64():
    """tf32 is a float32-only mode; under float64 it must warn, not silently
    change global torch state."""
    before = torch.backends.cuda.matmul.allow_tf32
    with tempfile.TemporaryDirectory() as d:
        train = write_xyz(os.path.join(d, 'train.xyz'), 24, seed=3)
        train_ecenet_spice(
            train_xyz=train, test_xyz=train, n_train=16, n_val=4, n_test=4,
            r_cut_edge=4.0, r_cut_neighbor=4.0, l_max=2, n_max=2, embed_dim=8,
            n_layers=1, n_max_d=4, m_max=2, n_epochs=1, batch_size=4, lr=5e-3,
            eval_every=1, eval_batch_size=4, dtype=torch.float64, device=DEVICE,
            seed=0, verbose=False, tf32=True)
    assert torch.backends.cuda.matmul.allow_tf32 == before, \
        "tf32=True under float64 should not touch torch's global TF32 state"
    print("  tf32=True under float64 leaves torch's TF32 state untouched")


if __name__ == '__main__':
    test_ddp_invariant_both_modes()
    test_bucket_groups_similar_sizes()
    test_bucket_sort_controls_batch_diversity()
    test_bucket_sort_false_keeps_ddp_aligned()
    test_atom_budget_respects_max_batch_count()
    test_degenerate_epoch_keeps_ranks_aligned()
    test_trainer_runs_with_atom_budget()
    test_precomputed_topology_matches_on_the_fly()
    test_les_wrapper_batched_matches_single()
    test_trainer_runs_with_les()
    test_tf32_is_a_noop_under_float64()
    print("All tests passed.")

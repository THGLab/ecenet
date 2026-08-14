# Prototype, mainly implemented by Claude
"""Size-aware (atom-budget) batching over prepared MPtrj shards.

Covers, without any MPtrj download:
  1. pack_by_atom_budget: budget/frame-cap respected, exact coverage;
  2. the DDP invariant on MPtrjShardDataset batch mode — every rank yields the
     SAME number of batches per epoch (a mismatch deadlocks the collective in
     backward), ranks are frame-disjoint, iteration is deterministic, and
     truncation drops rotate across epochs;
  3. ensure_atom_counts: loads the prepare-written sidecar, and back-fills an
     identical one for prepared dirs that predate it;
  4. frame mode is untouched by the new arguments (eval-path regression);
  5. end-to-end trainer smoke on a synthetic prepared dir with
     max_atoms_per_batch, plus the legacy-path rejection.

Run:  python tests/test_mptrj_shard_batching.py    (from the repo root)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))   # trainer lives in scripts/

import numpy as np
import torch
from test_mptrj_trainer import make_structures
from train_ecenet_mptrj import (
    STRESS_KBAR_TO_EVA3,
    compute_energy_reference,
    to_device_tensors,
    train_ecenet_mptrj,
)

from ecenet import elements
from ecenet.datasets.mptrj import (
    ATOM_COUNTS_FILE,
    MPtrjShardDataset,
    ensure_atom_counts,
    pack_by_atom_budget,
)

DTYPE = torch.float64
DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def write_fake_prepared(out_dir, n_shards=6, shard_size=10, seed=0):
    """Minimal prepared dir for dataset-level tests: frames are tiny dicts
    carrying 'n_atoms' plus a globally unique 'uid'. Returns per-shard count
    arrays (in shard order, i.e. what atom_counts.pt should hold)."""
    out_dir = Path(out_dir)
    rng = np.random.RandomState(seed)
    names, counts, uid = [], [], 0
    for si in range(n_shards):
        na = rng.randint(2, 40, size=shard_size)
        shard = [{'n_atoms': int(a), 'uid': (uid := uid + 1)} for a in na]
        nm = f'shard_{si:05d}.pt'
        torch.save(shard, out_dir / nm)
        names.append(nm)
        counts.append(torch.tensor(na, dtype=torch.int64))
    torch.save(counts, out_dir / ATOM_COUNTS_FILE)
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump({'shards': names, 'n_frames': n_shards * shard_size,
                   'shard_size': shard_size}, f)
    return names, [np.asarray(c) for c in counts]


def write_real_prepared(out_dir, structs, shard_size, r_cut_edge=4.0,
                        r_cut_neighbor=3.5):
    """Full prepared dir (real tensors + manifest + type_map + e_ref +
    atom_counts) from synthetic structures — what prepare_mptrj.py would write,
    minus the JSON round-trip."""
    out_dir = Path(out_dir)
    type_map = elements.build_type_map(z for s in structs for z in s['numbers'])
    e_ref = compute_energy_reference(structs, type_map)
    frames = to_device_tensors(structs, type_map, e_ref, r_cut_edge,
                               r_cut_neighbor, STRESS_KBAR_TO_EVA3, DTYPE, DEVICE)
    names, counts = [], []
    for si in range(0, len(frames), shard_size):
        shard = frames[si:si + shard_size]
        nm = f'shard_{si // shard_size:05d}.pt'
        torch.save(shard, out_dir / nm)
        names.append(nm)
        counts.append(torch.tensor([f['n_atoms'] for f in shard], dtype=torch.int64))
    torch.save(counts, out_dir / ATOM_COUNTS_FILE)
    torch.save(type_map, out_dir / 'type_map.pt')
    torch.save(torch.from_numpy(np.asarray(e_ref, dtype=np.float64)),
               out_dir / 'e_ref.pt')
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump({'n_frames': len(frames), 'n_types': len(type_map),
                   'n_shards': len(names), 'shard_size': shard_size,
                   'r_cut_edge': r_cut_edge, 'r_cut_neighbor': r_cut_neighbor,
                   'dtype': 'float64', 'energy_key': 'synthetic',
                   'include_stress': True, 'shuffle_seed': 0,
                   'source': 'synthetic', 'shards': names}, f)


def make_datasets(prepared_dir, world_size, **kw):
    """One dataset per rank over the fake prepared dir's shards."""
    counts = ensure_atom_counts(prepared_dir)
    with open(Path(prepared_dir) / 'manifest.json') as f:
        names = json.load(f)['shards']
    paths = [str(Path(prepared_dir) / n) for n in names]
    return [MPtrjShardDataset(paths, rank=r, world_size=world_size, seed=3,
                              atom_counts=[counts[n] for n in names], **kw)
            for r in range(world_size)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pack_by_atom_budget():
    print("=== pack_by_atom_budget: budget, frame cap, coverage ===")
    rng = np.random.RandomState(0)
    counts = rng.randint(1, 30, size=200)
    counts[7] = 100                      # single frame over the budget
    order = rng.permutation(200)
    for cap in (None, 3):
        batches = pack_by_atom_budget(order, counts, 50, max_batch_count=cap)
        got = np.sort(np.concatenate(batches))
        assert np.array_equal(got, np.arange(200)), "not an exact partition"
        for b in batches:
            atoms = counts[b].sum()
            assert atoms <= 50 or len(b) == 1, \
                f"batch of {len(b)} frames holds {atoms} > 50 atoms"
            if cap:
                assert len(b) <= cap, f"batch holds {len(b)} > {cap} frames"
    # sorted input → batches are size-runs (bucket behaviour)
    srt = np.argsort(counts, kind='stable')
    batches = pack_by_atom_budget(srt, counts, 50)
    flat = counts[np.concatenate(batches)]
    assert np.all(np.diff(flat) >= 0), "sorted packing should preserve order"
    print(f"  partitions exact; budget + frame cap respected "
          f"({len(batches)} batches from 200 frames)\n")


def test_ddp_alignment_and_coverage():
    print("=== batch mode: DDP count alignment, disjointness, determinism ===")
    with tempfile.TemporaryDirectory() as tmp:
        write_fake_prepared(tmp, n_shards=6, shard_size=10)
        for world_size in (1, 2, 3):
            for bucket_sort in (True, False):
                dss = make_datasets(tmp, world_size, shuffle=True,
                                    max_atoms_per_batch=60,
                                    bucket_sort=bucket_sort)
                per_rank = []
                for ds in dss:
                    ds.set_epoch(2)
                    per_rank.append(list(ds))
                # 1. identical batch counts on every rank (the DDP invariant)
                ns = [len(b) for b in per_rank]
                assert len(set(ns)) == 1, f"unequal batch counts: {ns}"
                assert ns[0] == len(dss[0]), f"__len__ {len(dss[0])} != yielded {ns[0]}"
                # 2. batches respect the budget (single oversize frame excepted)
                for batches in per_rank:
                    for b in batches:
                        atoms = sum(fr['n_atoms'] for fr in b)
                        assert atoms <= 60 or len(b) == 1, f"{atoms} atoms in a batch"
                # 3. ranks are frame-disjoint
                uids = [set(fr['uid'] for b in batches for fr in b)
                        for batches in per_rank]
                for a in range(len(uids)):
                    for b in range(a + 1, len(uids)):
                        assert not (uids[a] & uids[b]), "ranks share frames"
                # 4. deterministic: re-iterating the same epoch is identical
                again = [[fr['uid'] for fr in b] for b in dss[0]]
                first = [[fr['uid'] for fr in b] for b in per_rank[0]]
                assert again == first, "same-epoch iteration not deterministic"
        print("  counts equal across ranks (ws=1,2,3), budget kept, "
              "ranks disjoint, deterministic")

        # 5. truncation drops rotate: over a few epochs, the union of frames a
        #    2-rank world sees exceeds any single epoch's coverage.
        dss = make_datasets(tmp, 2, shuffle=True, max_atoms_per_batch=60)
        seen_by_epoch = []
        for ep in range(4):
            got = set()
            for ds in dss:
                ds.set_epoch(ep)
                got |= {fr['uid'] for b in ds for fr in b}
            seen_by_epoch.append(got)
        union = set().union(*seen_by_epoch)
        assert len(union) > max(len(s) for s in seen_by_epoch), \
            "dropped frames never rotate across epochs"
        cov = min(len(s) for s in seen_by_epoch) / 60
        print(f"  per-epoch coverage ≥ {cov:.0%}, drops rotate "
              f"(union {len(union)}/60 over 4 epochs)\n")


def test_atom_counts_backfill():
    print("=== ensure_atom_counts: sidecar load + back-fill ===")
    with tempfile.TemporaryDirectory() as tmp:
        names, counts = write_fake_prepared(tmp, n_shards=4, shard_size=7)
        from_file = ensure_atom_counts(tmp)
        os.remove(Path(tmp) / ATOM_COUNTS_FILE)
        rebuilt = ensure_atom_counts(tmp)          # back-fills from the shards
        assert (Path(tmp) / ATOM_COUNTS_FILE).exists(), "back-fill did not save"
        for nm, want in zip(names, counts):
            assert np.array_equal(from_file[nm], want)
            assert np.array_equal(rebuilt[nm], want)
    print("  prepare-written and back-filled sidecars identical\n")


def test_frame_mode_untouched():
    print("=== frame mode: unchanged by the new arguments (eval regression) ===")
    with tempfile.TemporaryDirectory() as tmp:
        write_fake_prepared(tmp, n_shards=3, shard_size=5)
        with open(Path(tmp) / 'manifest.json') as f:
            names = json.load(f)['shards']
        paths = [str(Path(tmp) / n) for n in names]
        ds = MPtrjShardDataset(paths, rank=0, world_size=1, seed=0, shuffle=False)
        uids = [fr['uid'] for fr in ds]
        assert uids == list(range(1, 16)), "on-disk order broken in frame mode"
        # batch mode refuses to run without counts
        try:
            MPtrjShardDataset(paths, max_atoms_per_batch=50)
            raise AssertionError("missing atom_counts not rejected")
        except ValueError:
            pass
    print("  shuffle=False walks on-disk order; missing atom_counts rejected\n")


def test_trainer_smoke_size_aware():
    print("=== trainer smoke: prepared dir + max_atoms_per_batch ===")
    with tempfile.TemporaryDirectory() as tmp:
        structs = make_structures(48, seed=5)
        write_real_prepared(tmp, structs, shard_size=8)
        _, results = train_ecenet_mptrj(
            prepared_dir=tmp, val_frac=0.2,
            max_atoms_per_batch=24, max_batch_count=6,
            l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
            r_cut_edge=4.0, r_cut_neighbor=3.5,
            stress_weight=0.1, force_weight=1.0, energy_weight=1.0,
            n_epochs=2, batch_size=4, lr=5e-3,
            dtype=DTYPE, device=DEVICE, seed=0, verbose=True,
        )
        for k in ('val_energy_mae', 'val_force_mae'):
            assert np.isfinite(results[k]), f"{k} not finite: {results[k]}"
    # legacy in-memory path rejects the flag rather than silently ignoring it
    try:
        train_ecenet_mptrj(train_structures=make_structures(4),
                           max_atoms_per_batch=24, n_epochs=1,
                           dtype=DTYPE, device=DEVICE)
        raise AssertionError("legacy path accepted max_atoms_per_batch")
    except ValueError:
        pass
    print("  2-epoch size-aware run finite; legacy path rejects the flag\n")


if __name__ == '__main__':
    test_pack_by_atom_budget()
    test_ddp_alignment_and_coverage()
    test_atom_counts_backfill()
    test_frame_mode_untouched()
    test_trainer_smoke_size_aware()
    print("ALL TESTS PASSED")

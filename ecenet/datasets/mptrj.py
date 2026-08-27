"""IterableDataset over pre-tensorized MPtrj shards.

Pairs with ``prepare_mptrj.py``: that script writes ``shard_NNNNN.pt`` (each
holding a list of per-frame tensor dicts), plus ``manifest.json``,
``type_map.pt``, ``e_ref.pt``, and ``atom_counts.pt`` (per-frame atom counts
per shard — metadata for size-aware batching). This module loads those shards
on demand, yielding per-frame dicts in a DDP-aware, shuffled,
per-worker-disjoint order.

Sharding strategy (per epoch, with shuffle=True):
  1. permute the full shard index list (seed = base_seed + epoch).
  2. take the rank's slice via stride: my_shards = perm[rank::world_size].
  3. if a DataLoader worker is attached, sub-slice again by worker id.
  4. for each assigned shard: load it from disk, permute frame order
     (seed = base_seed + epoch + shard_idx), yield each frame.

This gives every frame on the node exactly once per epoch with cheap
shard-level locality (one disk read per ~10k frames). Cross-epoch entropy
comes from the (shard permutation × intra-shard permutation) per epoch.

Size-aware batch mode (``max_atoms_per_batch``): instead of frames, ``__iter__``
yields ready-packed *batches* (lists of frame dicts) whose total atom count is
bounded by the budget — the SPICE trainer's ``size_aware_batches`` recipe
applied per shard. A shard is a valid packing window because frames were
globally shuffled at prepare time, so each shard's size distribution matches
the dataset's. The DDP invariant (every rank runs the SAME number of batches,
or the collective in backward deadlocks) is restored by round alignment:

  1. permute shards; group into rounds of ``world_size`` (partial round dropped);
     rank r owns the r-th shard of each round.
  2. every rank *plans* all shards in the round from ``atom_counts`` alone
     (packing depends only on counts + seed + epoch, never on tensors), giving
     each shard's batch count without touching disk.
  3. all ranks truncate to the round's minimum count. WHICH batches survive is
     a seeded per-shard permutation, so with ``bucket_sort`` the dropped
     batches are not systematically the largest-structure tail.

Shards are statistically exchangeable, so per-shard counts under a fixed
budget concentrate tightly and the truncation loss is well under 1%.
"""

import json
import warnings
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# Sidecar file holding a list of (n_frames,) int64 tensors, one per shard in
# manifest order. Written by prepare_mptrj.py; back-filled by
# ensure_atom_counts for prepared dirs that predate it.
ATOM_COUNTS_FILE = 'atom_counts.pt'


def load_manifest(prepared_dir):
    """Return (manifest_dict, type_map_dict, e_ref_np, abs_shard_paths)."""
    prepared_dir = Path(prepared_dir)
    with open(prepared_dir / 'manifest.json') as f:
        manifest = json.load(f)
    type_map = torch.load(prepared_dir / 'type_map.pt', weights_only=False)
    e_ref = torch.load(prepared_dir / 'e_ref.pt', weights_only=False).numpy()
    shard_paths = [str(prepared_dir / s) for s in manifest['shards']]
    return manifest, type_map, e_ref, shard_paths


def pack_by_atom_budget(order, counts, max_atoms_per_batch, max_batch_count=None):
    """Pack ``order`` (frame indices, in packing order) into consecutive batches
    holding at most ``max_atoms_per_batch`` total atoms (frame i contributes
    ``counts[i]``), optionally capped at ``max_batch_count`` frames per batch.
    A single frame over the budget still forms its own batch. Returns a list of
    int arrays. Same packing rule as the SPICE trainer's size_aware_batches.
    """
    batches, cur, cur_atoms = [], [], 0
    for idx in order:
        a = int(counts[idx])
        if cur and (cur_atoms + a > max_atoms_per_batch
                    or (max_batch_count and len(cur) >= max_batch_count)):
            batches.append(np.array(cur))
            cur, cur_atoms = [], 0
        cur.append(int(idx))
        cur_atoms += a
    if cur:
        batches.append(np.array(cur))
    return batches


def ensure_atom_counts(prepared_dir):
    """Per-frame atom counts for every shard: {shard_basename: (n_frames,) int64}.

    Reads ``atom_counts.pt`` (written by prepare_mptrj.py). For prepared dirs
    that predate the sidecar, back-fills it by reading every shard once and
    saves the result next to the manifest — a one-time cost. Under DDP, call
    on rank 0 first (build + save), barrier, then on the other ranks (which
    then only read the file).
    """
    prepared_dir = Path(prepared_dir)
    with open(prepared_dir / 'manifest.json') as f:
        shards = json.load(f)['shards']
    path = prepared_dir / ATOM_COUNTS_FILE
    if path.exists():
        counts = torch.load(path, map_location='cpu', weights_only=False)
        if len(counts) != len(shards):
            raise ValueError(
                f"{path} holds {len(counts)} shards but the manifest lists "
                f"{len(shards)}; delete the file to rebuild it")
    else:
        # One-time back-fill: unpickles every shard once (≈ one epoch's worth
        # of shard reads), then the file is cached for all later runs.
        import time
        print(f"[atom_counts] {path} missing — building it from "
              f"{len(shards)} shards (one-time; ~one epoch of shard reads)...",
              flush=True)
        t0 = time.time()
        counts = []
        for k, s in enumerate(shards):
            shard = torch.load(prepared_dir / s, map_location='cpu',
                               weights_only=False)
            counts.append(torch.tensor([int(fr['n_atoms']) for fr in shard],
                                       dtype=torch.int64))
            if (k + 1) % 10 == 0 or k + 1 == len(shards):
                print(f"[atom_counts]   {k + 1}/{len(shards)} shards "
                      f"({time.time() - t0:.0f}s)", flush=True)
        torch.save(counts, path)
        print(f"[atom_counts] saved {path} ({time.time() - t0:.0f}s)", flush=True)
    return {s: np.asarray(c, dtype=np.int64) for s, c in zip(shards, counts)}


class MPtrjShardDataset(IterableDataset):
    """Streams per-frame dicts from a list of .pt shard files.

    Parameters
    ----------
    shard_paths : sequence of str
        Absolute paths to ``shard_*.pt`` files. Each shard is a ``list[dict]``.
    rank, world_size : int
        DDP rank info. Each rank receives a disjoint subset of shards per epoch
        (round-robin over a shard permutation).
    seed : int
        Base seed; per-epoch shuffles are deterministic from (seed, epoch).
    shuffle : bool
        Shuffle shards across epochs and frames within each shard. Default True.
        Set False for eval to walk shards in their on-disk order.
    max_atoms_per_batch : int, optional
        Switch to size-aware batch mode: ``__iter__`` yields lists of frame
        dicts packed per shard to this total-atom budget, with per-round
        min-truncation so every rank yields the same number of batches per
        epoch (the DDP invariant). Requires ``atom_counts``.
    max_batch_count : int, optional
        Batch-mode only: cap on frames per packed batch (bounds per-structure
        Python overhead when a batch is all tiny structures).
    bucket_sort : bool
        Batch-mode only. True (default): sort each shard by atom count before
        packing — size-homogeneous batches, but a shard's fixed population
        makes its batches nearly identical every epoch. False: greedy-pack the
        epoch's shuffled order — diverse batches, cost still bounded by the
        budget.
    atom_counts : sequence of int arrays, optional
        Per-frame atom counts, one array per entry of ``shard_paths`` (from
        ``ensure_atom_counts``). Lets every rank plan every shard's packing
        without loading it. Required in batch mode.
    """

    def __init__(self, shard_paths: Sequence[str], rank: int = 0,
                 world_size: int = 1, seed: int = 0, shuffle: bool = True,
                 max_atoms_per_batch=None, max_batch_count=None,
                 bucket_sort: bool = True, atom_counts=None):
        super().__init__()
        self.shard_paths: List[str] = list(shard_paths)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.max_atoms_per_batch = max_atoms_per_batch
        self.max_batch_count = max_batch_count
        self.bucket_sort = bool(bucket_sort)
        if max_atoms_per_batch is not None:
            if atom_counts is None:
                raise ValueError(
                    "max_atoms_per_batch requires atom_counts (per-frame atom "
                    "counts per shard; see ensure_atom_counts)")
            if len(atom_counts) != len(self.shard_paths):
                raise ValueError(
                    f"atom_counts covers {len(atom_counts)} shards but the "
                    f"dataset holds {len(self.shard_paths)}")
            self.atom_counts = [np.asarray(c, dtype=np.int64) for c in atom_counts]
        else:
            self.atom_counts = None
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (mirrors torch.utils.data.DistributedSampler)."""
        self._epoch = int(epoch)

    def _shard_order(self):
        """This epoch's shard-index permutation (identical on every rank)."""
        rng = np.random.RandomState(self.seed + self._epoch)
        if self.shuffle:
            return rng.permutation(len(self.shard_paths))
        return np.arange(len(self.shard_paths))

    # ── Size-aware batch mode ────────────────────────────────────────────

    def _rounds(self):
        """Shard indices grouped into rounds of world_size; rank r owns column
        r of each round. The partial final round is dropped so every rank has
        a shard in every round (with shuffle, the dropped shards rotate)."""
        order = self._shard_order()
        n_rounds = len(order) // self.world_size
        return order[:n_rounds * self.world_size].reshape(n_rounds, self.world_size)

    def _plan_batches(self, si):
        """Batch index lists for shard ``si`` this epoch, from atom counts
        alone. Deterministic in (seed, epoch, si), so every rank derives the
        same plan — and hence the same per-round counts — without disk I/O."""
        counts = self.atom_counts[si]
        if self.shuffle:
            idx = np.random.RandomState(
                self.seed + self._epoch * 100003 + int(si)).permutation(len(counts))
        else:
            idx = np.arange(len(counts))
        if self.bucket_sort:
            idx = idx[np.argsort(counts[idx], kind='stable')]
        return pack_by_atom_budget(idx, counts, self.max_atoms_per_batch,
                                   self.max_batch_count)

    def _iter_batches(self) -> Iterator[list]:
        rounds = self._rounds()
        if len(rounds) == 0:
            if self.shard_paths:
                warnings.warn(
                    f"world_size={self.world_size} exceeds the "
                    f"{len(self.shard_paths)} available shards; every rank "
                    "yields zero batches this epoch")
            return
        round_ids = range(len(rounds))
        wi = get_worker_info()
        if wi is not None:
            round_ids = list(round_ids)[wi.id::wi.num_workers]

        for r in round_ids:
            shard_ids = rounds[r]
            # All ranks compute the same minimum over the round's shards.
            m = min(len(self._plan_batches(int(si))) for si in shard_ids)
            si = int(shard_ids[self.rank])
            plan = self._plan_batches(si)
            if self.shuffle:
                # Randomize both WHICH batches survive truncation (so the
                # sorted plan's largest-structure tail is not always the part
                # dropped) and the yield order (no small→large curriculum).
                keep = np.random.RandomState(
                    self.seed + self._epoch * 999983 + si).permutation(len(plan))[:m]
            else:
                keep = np.arange(m)
            shard = torch.load(self.shard_paths[si], map_location='cpu',
                               weights_only=False)
            if len(shard) != len(self.atom_counts[si]):
                raise ValueError(
                    f"{self.shard_paths[si]} holds {len(shard)} frames but "
                    f"atom_counts lists {len(self.atom_counts[si])} — stale "
                    f"{ATOM_COUNTS_FILE}? Delete it to rebuild.")
            for b in keep:
                yield [shard[int(fi)] for fi in plan[int(b)]]

    # ── Iteration ────────────────────────────────────────────────────────

    # Approximate length. Frame mode: shards per rank (off-by-one due to a
    # non-full last shard is fine). Batch mode: this epoch's exact per-rank
    # batch count (identical on every rank by construction).
    def __len__(self) -> int:
        if self.max_atoms_per_batch is None:
            return len(self.shard_paths) // max(1, self.world_size)
        return sum(min(len(self._plan_batches(int(si))) for si in rd)
                   for rd in self._rounds())

    def __iter__(self) -> Iterator[dict]:
        if self.max_atoms_per_batch is not None:
            yield from self._iter_batches()
            return

        order = self._shard_order()

        # Rank-disjoint slice.
        my_shards = order[self.rank::self.world_size]

        # Sub-slice across DataLoader workers (if any).
        wi = get_worker_info()
        if wi is not None:
            my_shards = my_shards[wi.id::wi.num_workers]

        for si in my_shards:
            shard = torch.load(self.shard_paths[int(si)], map_location='cpu',
                               weights_only=False)
            if self.shuffle:
                inner_rng = np.random.RandomState(self.seed + self._epoch * 100003
                                                  + int(si))
                idx = inner_rng.permutation(len(shard))
            else:
                idx = np.arange(len(shard))
            for fi in idx:
                yield shard[int(fi)]


def split_shards(shard_paths: Sequence[str], val_frac: float,
                 seed: int = 0) -> tuple:
    """Split shard list into (train_shards, val_shards) by holding out a
    fraction of shards. Frames were pre-shuffled at prepare time, so a random
    shard-level holdout is a uniform random frame-level holdout."""
    shard_paths = list(shard_paths)
    n = len(shard_paths)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
    val_idx = set(perm[:n_val].tolist())
    train = [s for i, s in enumerate(shard_paths) if i not in val_idx]
    val = [shard_paths[i] for i in sorted(val_idx)]
    return train, val


def collate_keep_list(batch):
    """No-op collate: the trainer's predict() expects a list of per-frame
    dicts, not a stacked batch tensor. Each DataLoader iteration yields a
    list of dicts."""
    return batch

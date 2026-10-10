"""Batch loading shared by every dataset: an epoch-seeded sampler, the
DataLoader over a resident dataset, and the CUDA prefetcher that overlaps
the next batch's host-to-device copy with compute.

A dataset item is one already-drawn sample dict indexed by (item, epoch),
so collation is a plain stack. residency "ram": the per-rank shard lives in
CPU RAM and loader workers draw the samples; residency "gpu": the shard
lives on the GPU and the dataset draws on the main process (no workers).
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Sampler

from ..utils.env import limit_worker_threads


def _worker_init(_worker_id: int) -> None:
    limit_worker_threads()


def collate_tree(samples: list):
    """Stack a list of samples element-wise through dicts, tuples and lists."""
    first = samples[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(samples).contiguous()
    if isinstance(first, dict):
        return {k: collate_tree([s[k] for s in samples]) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(collate_tree([s[i] for s in samples]) for i in range(len(first)))
    return first


def collate_single(samples: list):
    """Micro-batch 1 by protocol: the sample passes through untouched."""
    assert len(samples) == 1, "this view is micro-batch 1 by protocol"
    return samples[0]


def move_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch


class EpochSampler(Sampler):
    """Yields (item_idx, epoch) for one epoch, reshuffled by set_epoch()."""

    def __init__(self, n_items: int, seed: int, shuffle: bool = True):
        self.n_items, self.seed, self.shuffle = int(n_items), int(seed), bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_items

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.n_items, generator=g).tolist()
        else:
            order = range(self.n_items)
        return iter([(i, self.epoch) for i in order])


def build_dataloader(dataset, batch_size: int, num_workers: int,
                     prefetch_factor: int, seed: int, collate_fn,
                     residency: str = "ram") -> tuple[DataLoader, EpochSampler]:
    sampler = EpochSampler(len(dataset), seed, shuffle=True)
    if residency == "gpu":
        num_workers = 0
    kwargs = dict(batch_size=batch_size, sampler=sampler, collate_fn=collate_fn,
                  num_workers=num_workers, drop_last=True,
                  pin_memory=(residency == "ram" and torch.cuda.is_available()))
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True,
                      worker_init_fn=_worker_init)
    return DataLoader(dataset, **kwargs), sampler


class CudaPrefetcher:
    """Copies the next batch to the GPU on a side stream while the current
    batch computes. Batches already on the GPU pass through."""

    def __init__(self, loader, device: torch.device):
        self._it = iter(loader)
        self._device = device
        self._stream = torch.cuda.Stream(device)
        self._next = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self._it)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self._stream):
            self._next = move_to_device(batch, self._device)

    def __iter__(self):
        return self

    def __next__(self):
        if self._next is None:
            raise StopIteration
        batch = self._next
        torch.cuda.current_stream().wait_stream(self._stream)
        cur = torch.cuda.current_stream()

        def _rec(v):
            if isinstance(v, torch.Tensor):
                v.record_stream(cur)
            elif isinstance(v, dict):
                for x in v.values():
                    _rec(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    _rec(x)
        _rec(batch)
        self._preload()
        return batch

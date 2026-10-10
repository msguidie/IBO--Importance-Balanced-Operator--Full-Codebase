"""Stochastic weight averaging on disk.

Every epoch rank 0 writes an fp32 model snapshot to
epoch_weights/epoch_{N:04d}.pt. After training, swa_model.pt is the mean of
the snapshots of the last swa_fraction of the epochs, accumulated as one
fp32 running sum over the files (each read once, a few prefetched).
"""
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Peel torch.compile (_orig_mod) and DDP (module) wrappers."""
    for _ in range(4):
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        else:
            break
    return model


def atomic_save(payload, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, str(path))


def _cpu_copy(v: torch.Tensor) -> torch.Tensor:
    """Detached fp32 CPU copy, never an alias of the live parameter."""
    out = v.detach().to("cpu", dtype=torch.float32) if v.is_floating_point() \
        else v.detach().cpu()
    return out.clone() if out.data_ptr() == v.data_ptr() else out


def save_epoch_weights(path: str | Path, model: nn.Module, epoch: int):
    sd = {k: _cpu_copy(v) for k, v in unwrap_model(model).state_dict().items()}
    atomic_save({"format": "epoch_weights", "epoch": int(epoch),
                 "model_state_dict": sd}, path)


def available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def average_epoch_weights(epoch_weights_dir: str | Path, lo: int, hi: int
                          ) -> tuple[dict[str, torch.Tensor], int]:
    """fp32 mean of epoch_{lo..hi}.pt (inclusive); non-floating buffers take
    the last file's value. Returns (state_dict, count)."""
    epoch_weights_dir = Path(epoch_weights_dir)
    paths = [epoch_weights_dir / f"epoch_{n:04d}.pt" for n in range(lo, hi + 1)]
    missing = [p.name for p in paths if not p.exists()]
    assert not missing, f"missing epoch weights for the SWA window: {missing[:5]}"

    def load(p: Path) -> dict:
        return torch.load(str(p), map_location="cpu", weights_only=True)["model_state_dict"]

    total: dict[str, torch.Tensor] = {}
    nonfloat: dict[str, torch.Tensor] = {}
    lookahead = min(8, max(2, available_cpu_count()))
    with ThreadPoolExecutor(max_workers=lookahead) as pool:
        pending: deque = deque(pool.submit(load, p) for p in paths[:lookahead])
        i = lookahead
        while pending:
            sd = pending.popleft().result()
            if i < len(paths):
                pending.append(pool.submit(load, paths[i]))
                i += 1
            for k, v in sd.items():
                if v.is_floating_point():
                    vf = v.to(dtype=torch.float32)
                    total[k] = vf.clone() if k not in total else total[k] + vf
                else:
                    nonfloat[k] = v.clone()
    n = len(paths)
    out = {k: v / n for k, v in total.items()}
    out.update(nonfloat)
    return out, n

"""Checkpoints.

Resume state is two files per epoch: epoch_weights/epoch_{N:04d}.pt (fp32
model weights, also the addends of the final SWA average) and
full_epoch{N:04d}.pt (optimizer, scheduler, epoch). A resume needs both
halves of the same epoch. Older epochs are pruned as new ones are written,
except those inside the SWA window.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from .swa import atomic_save, save_epoch_weights, unwrap_model

_FULL_EPOCH_RE = re.compile(r"^full_epoch(\d+)\.pt$")


def find_latest_trainer_state(ckpt_dir: str | Path) -> tuple[int, Path] | None:
    """Newest resumable (epoch, trainer_state_path): the newest
    full_epoch{N}.pt whose weights file also exists (a job killed between
    the two writes leaves a half-pair, which is skipped)."""
    ckpt_dir = Path(ckpt_dir)
    weights_dir = ckpt_dir / "epoch_weights"
    cands = []
    for p in ckpt_dir.glob("full_epoch*.pt"):
        m = _FULL_EPOCH_RE.match(p.name)
        if m:
            cands.append((int(m.group(1)), p))
    for epoch, path in sorted(cands, reverse=True):
        if (weights_dir / f"epoch_{epoch:04d}.pt").exists():
            return epoch, path
    return None


def _state_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_state_to_cpu(v) for v in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    return obj


def save_epoch(ckpt_dir: Path, model, optimizer, scheduler, epoch: int,
               identity: dict, rolling_keep: int, swa_start: int) -> None:
    """One epoch's disk footprint, in the order a resume can trust: weights,
    then the trainer state that pairs with them, then the prunes. identity
    (dataset, model, world_size, cases_per_gpu, grad_accum) is recorded so a
    relaunch with a different step geometry is refused at resume."""
    weights_dir = ckpt_dir / "epoch_weights"
    save_epoch_weights(weights_dir / f"epoch_{epoch:04d}.pt", model, epoch)
    atomic_save({"format": "trainer_state", "epoch": int(epoch), **identity,
                 "optimizer_state_dict": _state_to_cpu(optimizer.state_dict()),
                 "scheduler_state_dict": scheduler.state_dict()
                 if scheduler is not None else None},
                ckpt_dir / f"full_epoch{epoch:04d}.pt")
    prune_rolling(ckpt_dir, rolling_keep)
    prune_epoch_weights(weights_dir, epoch, rolling_keep, swa_start)


def prune_epoch_weights(weights_dir: Path, epoch: int, rolling_keep: int,
                        swa_start: int) -> None:
    """Keep the SWA addends (epoch >= swa_start) and the newest rolling_keep
    epochs (the model half of a resume pair); delete the rest."""
    if rolling_keep <= 0:
        return
    cutoff = epoch - rolling_keep
    for old in weights_dir.glob("epoch_*.pt"):
        try:
            n = int(old.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if n <= cutoff and n < swa_start:
            old.unlink(missing_ok=True)


def prune_rolling(ckpt_dir: str | Path, keep: int):
    fulls = sorted(Path(ckpt_dir).glob("full_epoch*.pt"))
    for old in fulls[:-keep] if keep > 0 else fulls:
        old.unlink(missing_ok=True)


def load_state_strict(model: torch.nn.Module, sd: dict) -> None:
    """Upcast floating tensors to fp32, then load with strict=True."""
    sd = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point()
              else v) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


def load_swa_weights(path: str | Path, model: torch.nn.Module) -> dict:
    """Strict-load swa_model.pt into the (unwrapped) model; returns the
    payload's metadata (epoch window, count)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: finish training first")
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    load_state_strict(unwrap_model(model), payload["model_state_dict"])
    return {k: v for k, v in payload.items() if k != "model_state_dict"}


def load_full_checkpoint(path, model, optimizer, scheduler, ckpt_dir: Path,
                         identity: dict) -> int:
    """Resume from a trainer-state file plus its epoch weights. Refuses a
    mismatch in any identity field (step geometry or experiment)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if ckpt.get("format") != "trainer_state":
        raise ValueError(f"{path} is not a trainer-state checkpoint")
    for key, current in identity.items():
        saved = ckpt.get(key)
        if saved is not None and current is not None and saved != current:
            raise ValueError(f"resume {key} mismatch: checkpoint {saved!r}, "
                             f"this launch {current!r}; start a fresh run dir")
    epoch = int(ckpt["epoch"])
    weights_path = Path(ckpt_dir) / "epoch_weights" / f"epoch_{epoch:04d}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"resume needs {weights_path}")
    w = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    load_state_strict(unwrap_model(model), w["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return epoch

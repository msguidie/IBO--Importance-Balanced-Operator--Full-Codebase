"""Deterministic seeding helpers."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

# Seed of the paper's base runs. Per-step draws are keyed by (case, epoch)
# alone under this seed, so seed 42 reproduces the original draws; any other
# training seed salts them (train_seed_salt).
BASE_SEED = 42


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_case_epoch_seed(case_name: str, epoch: int) -> int:
    """Stable 32-bit seed per (case, epoch): deterministic query subsampling."""
    digest = hashlib.blake2b(f"{case_name}:{epoch}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def train_seed_salt(seed: int) -> str:
    """Suffix mixed into every per-step training draw's seed string (IBO's
    query subsample, AB-UPT's training anchors). Empty for BASE_SEED so the
    base run keeps the original draws; never applied to evaluation draws or
    to anything stored in a cache, so every seed is scored on identical
    samples."""
    return "" if int(seed) == BASE_SEED else f":s{int(seed)}"

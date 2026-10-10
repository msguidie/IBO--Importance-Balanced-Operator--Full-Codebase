"""Streaming mean / standard deviation with a parallel (Chan) merge: each
case reduces locally in float64, the parent merges in a fixed order."""
from __future__ import annotations

import numpy as np


def welford_local(x) -> tuple[int, np.ndarray, np.ndarray]:
    """x [N, C] -> (n, mean [C], M2 [C]) in float64."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        c = x.shape[1] if x.ndim == 2 else 1
        return 0, np.zeros(c), np.zeros(c)
    mean = x.mean(axis=0)
    return n, mean, ((x - mean) ** 2).sum(axis=0)


class WelfordAccumulator:
    def __init__(self, n_channels: int):
        self.count = 0
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.m2 = np.zeros(n_channels, dtype=np.float64)

    def merge_local(self, n_b: int, mean_b, m2_b) -> None:
        if n_b == 0:
            return
        if not (np.all(np.isfinite(mean_b)) and np.all(np.isfinite(m2_b))):
            raise ValueError("Welford merge received non-finite statistics")
        n_a = self.count
        n_new = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n_new)
        self.m2 = self.m2 + m2_b + delta ** 2 * (n_a * n_b / n_new)
        self.count = n_new

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std); std is 1 with fewer than two samples."""
        if self.count < 2:
            return self.mean.copy(), np.ones_like(self.mean)
        return self.mean.copy(), np.sqrt(np.maximum(self.m2 / self.count, 1e-16))

"""Shared layers of the IBO model."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_linear(module: nn.Linear, mode: str = "trunc_normal") -> None:
    """mode "trunc_normal": N(0, 0.02) truncated, zero bias; mode "default":
    PyTorch's own initialisation (the UrbanWind runs were trained that way)."""
    if mode == "trunc_normal":
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32, eps 1e-6."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms).to(x.dtype) * self.weight.to(x.dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0,
                 init: str = "trunc_normal"):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        init_linear(self.gate_up, init)
        init_linear(self.down, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.drop(self.down(F.silu(gate) * up))


class Fourier(nn.Module):
    """Fourier features with bands 2^k pi, k < n_freqs, plus the raw input:
    [..., d] -> [..., d * (1 + 2 n_freqs)], computed in fp32."""

    def __init__(self, n_freqs: int, in_dim: int):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(n_freqs).float()) * math.pi)
        self.out_dim = in_dim * (1 + 2 * n_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        theta = x[..., None] * self.freq
        sc = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        return torch.cat([x, sc.reshape(*x.shape[:-1], -1)], dim=-1)

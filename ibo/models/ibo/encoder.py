"""Token projection: cat(pooled slot features, per-token descriptor) -> D,
through one linear layer or two with a SiLU between, optionally modulated
by a zero-initialised FiLM on the case vector."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import init_linear


class TokenEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, layers: int = 1,
                 cond_dim: int = 0, film: bool = False, init: str = "default"):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim) if layers == 2 else None
        init_linear(self.proj1, init)
        if self.proj2 is not None:
            init_linear(self.proj2, init)
        self.film = None
        if film and cond_dim > 0:
            self.film = nn.Linear(cond_dim, 2 * latent_dim)   # no-op at init
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, pooled: torch.Tensor, stats: torch.Tensor,
                cond: torch.Tensor | None) -> torch.Tensor:
        x = torch.cat([pooled, stats.to(pooled.dtype)], dim=-1)
        token = self.proj1(x)
        if self.proj2 is not None:
            token = self.proj2(F.silu(token))
        if self.film is not None:
            gamma, beta = self.film(cond.to(token.dtype)).chunk(2, dim=-1)
            token = token * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return token

"""Slot encoder: a shared two-layer MLP over the member slots of every
token, then masked max / mean / std pooling (fp32) concatenated to
3 * pn_out. A token without a valid slot pools to exactly zero."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 act: str = "relu", fp32_first: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.act = F.relu if act == "relu" else F.silu
        # near-wall relative positions are small and would round to zero in
        # bf16; the 2-D and DrivAerML settings run the first layer in fp32
        self.fp32_first = bool(fp32_first)
        for m in (self.fc1, self.fc2):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, slots: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """slots [B, L, S, F], mask [B, L, S] bool -> [B, L, 3 * out_dim]."""
        m = mask[..., None]
        x = torch.where(m, slots, torch.zeros_like(slots))
        if self.fp32_first:
            with torch.autocast("cuda", enabled=False):
                h = self.act(self.fc1(x.float()))
            h = self.fc2(h.to(self.fc2.weight.dtype))
        else:
            h = self.fc2(self.act(self.fc1(x)))
        with torch.autocast("cuda", enabled=False):
            hf = h.float()
            neg = torch.finfo(hf.dtype).min
            h_max = torch.where(m, hf, torch.full_like(hf, neg)).max(dim=-2).values
            cnt = m.sum(dim=-2).clamp(min=1).to(hf.dtype)
            h_valid = torch.where(m, hf, torch.zeros_like(hf))
            mu = h_valid.sum(dim=-2) / cnt
            dev = torch.where(m, hf - mu.unsqueeze(-2), torch.zeros_like(hf))
            std = torch.sqrt(dev.pow(2).sum(dim=-2) / cnt + 1e-6)
        out = torch.cat([h_max, mu, std], dim=-1).to(h.dtype)
        any_valid = mask.any(dim=-1, keepdim=True)
        return torch.where(any_valid, out, torch.zeros_like(out))

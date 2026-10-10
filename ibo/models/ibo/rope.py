"""Rotary position embeddings over d axes.

The head dimension is split across the axes (rope_dims); each axis has its
own frequency ladder (base) and scale. Positions either are already
normalised (domain None: theta = pos * scale * inv_freq, the UrbanWind
setting with positions divided by 550 m) or are mapped into a calibrated
domain first ((pos - lo) / (hi - lo), the 2-D and DrivAerML settings).
Register tokens get the identity rotation. Computed in fp32.
"""
from __future__ import annotations

import torch


def rope_cos_sin(pos: torch.Tensor, rope_dims, rope_base, rope_scales,
                 domain_lo, domain_hi, n_register: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """pos [B, L, d] -> (cos, sin) each [B, L + R, head_dim // 2] fp32."""
    pos = pos.float()
    parts = []
    for axis, dim in enumerate(rope_dims):
        half = dim // 2
        idx = torch.arange(half, dtype=torch.float32, device=pos.device)
        inv = float(rope_base[axis]) ** (-2.0 * idx / dim)
        x = pos[..., axis]
        if domain_lo is not None:
            lo, hi = float(domain_lo[axis]), float(domain_hi[axis])
            x = (x - lo) / max(hi - lo, 1e-8)
        parts.append((x * float(rope_scales[axis]))[..., None] * inv)
    theta = torch.cat(parts, dim=-1)
    if n_register > 0:
        theta = torch.cat([theta, theta.new_zeros(theta.shape[0], n_register,
                                                  theta.shape[-1])], dim=1)
    return torch.cos(theta), torch.sin(theta)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k [B, H, T, D]; cos, sin [B, T, D // 2]."""
    c = cos.unsqueeze(1).to(q.dtype)
    s = sin.unsqueeze(1).to(q.dtype)

    def rot(x: torch.Tensor) -> torch.Tensor:
        xp = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = xp[..., 0], xp[..., 1]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)

    return rot(q), rot(k)

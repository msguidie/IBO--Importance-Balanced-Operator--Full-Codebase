"""Global processor: a pre-norm Transformer over the latent tokens.

Blocks use QK-RMSNorm, rotary embeddings derived from the token positions,
full scaled-dot-product attention and a SwiGLU feed-forward layer, followed
by a final RMSNorm. Optional parts: conditioning on the case vector by
adaptive layer norm ("trunk": shared MLP cond -> hidden plus one
zero-initialised head per block; "zero": one zero-initialised linear per
block), register tokens, and gated long skips between mirrored blocks
(x = x + gate * proj(RMSNorm(cached input)), gate 0.01 at init).
forward returns the full [B, L + R, D] sequence, registers included.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import RMSNorm, SwiGLUFFN, init_linear
from .rope import apply_rope, rope_cos_sin


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, ffn_hidden: int,
                 dropout: float, init: str):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden, dropout, init)
        init_linear(self.qkv, init)
        init_linear(self.o_proj, init)

    def forward(self, x, cos, sin, mod):
        """mod [B, 6 * dim] = (gamma1, beta1, alpha1, gamma2, beta2, alpha2),
        or None for the unconditioned block."""
        B, T, D = x.shape
        if mod is not None:
            g1, b1, a1, g2, b2, a2 = (m[:, None, :] for m in mod.chunk(6, dim=-1))
            h = self.norm1(x) * (1 + g1) + b1
        else:
            h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(self.q_norm(q), self.k_norm(k), cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = self.o_proj(attn.transpose(1, 2).reshape(B, T, D))
        if mod is not None:
            x = x + a1 * attn
            x = x + a2 * self.ffn(self.norm2(x) * (1 + g2) + b2)
        else:
            x = x + attn
            x = x + self.ffn(self.norm2(x))
        return x


class Processor(nn.Module):
    def __init__(self, dim: int, n_blocks: int, num_heads: int, head_dim: int,
                 ffn_hidden: int, n_register: int, dropout: float,
                 rope_dims, rope_base, rope_scales, rope_domain_lo, rope_domain_hi,
                 cond_dim: int, adaln: str, adaln_hidden: int, long_skips: bool,
                 init: str, grad_checkpoint: bool = False):
        super().__init__()
        self.n_register = n_register
        self.rope = (tuple(rope_dims), tuple(rope_base), tuple(rope_scales),
                     rope_domain_lo, rope_domain_hi)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.register = nn.Parameter(torch.randn(n_register, dim) * 0.02) \
            if n_register > 0 else None
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, head_dim, ffn_hidden, dropout, init)
            for _ in range(n_blocks)])
        # a pair (i, n-1-i) is a long skip only when it spans a block
        n_skip = ((n_blocks - 1) // 2) if long_skips else 0
        self.skip_pairs = [(i, n_blocks - 1 - i) for i in range(n_skip)]
        self.save_at = {i: si for si, (i, _) in enumerate(self.skip_pairs)}
        self.consume_at = {j: si for si, (_, j) in enumerate(self.skip_pairs)}
        self.skip_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in self.skip_pairs])
        self.skip_norm = nn.ModuleList([RMSNorm(dim) for _ in self.skip_pairs])
        for w in self.skip_proj:
            init_linear(w, "trunc_normal")
        self.skip_gate = nn.Parameter(torch.full((n_skip,), 1e-2)) if n_skip else None
        self.final_norm = RMSNorm(dim)

        self.adaln = adaln if cond_dim > 0 else "none"
        if self.adaln == "trunk":
            self.trunk = nn.Sequential(nn.Linear(cond_dim, adaln_hidden), nn.SiLU(),
                                       nn.Linear(adaln_hidden, adaln_hidden))
            self.heads = nn.ModuleList([nn.Linear(adaln_hidden, 6 * dim)
                                        for _ in range(n_blocks)])
        elif self.adaln == "zero":
            self.heads = nn.ModuleList([nn.Linear(cond_dim, 6 * dim)
                                        for _ in range(n_blocks)])
        if self.adaln != "none":
            for head in self.heads:          # identity modulation at init
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, tokens: torch.Tensor, pos_geo: torch.Tensor,
                cond: torch.Tensor | None) -> torch.Tensor:
        """tokens [B, L, D]; pos_geo [B, L, d]; cond [B, C] -> [B, L + R, D]."""
        B = tokens.shape[0]
        x = tokens
        if self.register is not None:
            x = torch.cat([x, self.register.unsqueeze(0).expand(B, -1, -1).to(x.dtype)], dim=1)
        cos, sin = rope_cos_sin(pos_geo, *self.rope, self.n_register)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        if self.adaln == "trunk":
            h_act = F.silu(self.trunk(cond))
            mods = [head(h_act) for head in self.heads]
        elif self.adaln == "zero":
            mods = [head(cond.to(x.dtype)) for head in self.heads]
        else:
            mods = [None] * len(self.blocks)
        cache = []
        for i, block in enumerate(self.blocks):
            if i in self.consume_at:
                si = self.consume_at[i]
                x = x + self.skip_gate[si] * self.skip_proj[si](self.skip_norm[si](cache[si]))
            if i in self.save_at:
                cache.append(x)
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, cos, sin, mods[i], use_reentrant=False)
            else:
                x = block(x, cos, sin, mods[i])
        return self.final_norm(x)

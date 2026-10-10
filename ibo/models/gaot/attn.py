"""GAOT ViT processor — from src/model/layers/attn.py, verbatim except:

[PORT] rotary_embedding_torch is replaced by a minimal inline 1-D rotary
implementation with the SAME semantics as the upstream usage
(RotaryEmbedding(dim=head_dim).rotate_queries_or_keys on [B, H, S, hd]:
positions are the sequence indices 0..S-1, theta base 10000, full-width
rotation). omegaconf / dataclass-utils plumbing is dropped; configs are
plain dataclasses constructed directly.

Upstream structure kept exactly: GQA flash attention (SDPA), RMSNorm,
SwiGLU FFN (w2(silu(w1 x) * w3 x)), U-ViT long-range skips (encoder half
saves, decoder half pops through a Linear(2C -> C) merge), optional
middle layer when num_layers is odd.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    num_heads: int = 8
    num_kv_heads: int = 8
    atten_dropout: float = 0.0


@dataclass
class TransformerConfig:
    patch_size: int = 1
    hidden_size: int = 256
    use_attn_norm: bool = True
    use_ffn_norm: bool = True
    norm_eps: float = 1e-6
    num_layers: int = 3
    positional_embedding: str = "absolute"     # ['absolute', 'rope']
    use_long_range_skip: bool = True
    ffn_multiplier: int = 4
    attn_config: AttentionConfig = field(default_factory=AttentionConfig)


class _MinimalRotary(nn.Module):
    """[PORT] Inline stand-in for rotary_embedding_torch.RotaryEmbedding
    as upstream uses it: 1-D sequence rope over positions arange(S),
    theta 10000, rotating the full head_dim (paired halves)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_queries_or_keys(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, S, hd]
        seq = torch.arange(x.shape[-2], device=x.device,
                           dtype=self.inv_freq.dtype)
        freqs = torch.einsum("s,f->sf", seq, self.inv_freq)   # [S, hd/2]
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        x1 = x[..., 0::2].float()
        x2 = x[..., 1::2].float()
        out = torch.empty_like(x, dtype=torch.float32)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(x)


class GroupQueryFlashAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 8,
                 num_kv_heads: int = 8, atten_dropout: float = 0.0,
                 positional_embedding: str = "absolute"):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_repeat = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.atten_dropout = atten_dropout

        kv_hidden_size = self.head_dim * self.num_kv_heads
        self.q_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, input_size, bias=False)

        if positional_embedding == "rope":
            self.rotary_emb = _MinimalRotary(dim=self.head_dim)

    def forward(self, x, relative_positions: Optional[torch.Tensor] = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_repeat, dim=1)
            v = v.repeat_interleave(self.num_repeat, dim=1)
        if relative_positions is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        dp = self.atten_dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                           dropout_p=dp)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(x)


class FFN(nn.Module):
    def __init__(self, input_size: int, ffn_hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.w2 = nn.Linear(ffn_hidden_size, input_size, bias=False)
        self.w3 = nn.Linear(input_size, ffn_hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, input_size: int, config: TransformerConfig,
                 skip_connection: bool = False):
        super().__init__()
        hidden_size = config.hidden_size
        self.attn = GroupQueryFlashAttention(
            input_size=input_size, hidden_size=hidden_size,
            num_heads=config.attn_config.num_heads,
            num_kv_heads=config.attn_config.num_kv_heads,
            atten_dropout=config.attn_config.atten_dropout,
            positional_embedding=config.positional_embedding)
        self.ffn = FFN(input_size=input_size,
                       ffn_hidden_size=hidden_size * config.ffn_multiplier)
        self.attn_norm = RMSNorm(input_size, eps=config.norm_eps) \
            if config.use_attn_norm else None
        self.ffn_norm = RMSNorm(input_size, eps=config.norm_eps) \
            if config.use_ffn_norm else None
        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_proj = nn.Linear(input_size * 2, input_size)

    def forward(self, x, relative_positions=None, skip=None):
        if self.skip_connection and skip is not None:
            x = torch.cat([x, skip], dim=-1)
            x = self.skip_proj(x)
        h = x if self.attn_norm is None else self.attn_norm(x)
        h = x + self.attn(h, relative_positions=relative_positions)
        h2 = h if self.ffn_norm is None else self.ffn_norm(h)
        return h + self.ffn(h2)


class Transformer(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 config: TransformerConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        self.use_long_range_skip = config.use_long_range_skip

        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
            working_size = hidden_size
        else:
            self.input_proj = nn.Identity()
            working_size = input_size
        if working_size != output_size:
            self.output_proj = nn.Linear(working_size, output_size)
        else:
            self.output_proj = nn.Identity()

        num_encoder_layers = num_layers // 2
        num_decoder_layers = num_layers // 2
        middle_layer_exists = (num_layers % 2 == 1)

        self.encoder_layers = nn.ModuleList([
            TransformerBlock(working_size, config, skip_connection=False)
            for _ in range(num_encoder_layers)])
        self.middle_layer = TransformerBlock(
            working_size, config, skip_connection=False) \
            if middle_layer_exists else None
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(working_size, config, skip_connection=True)
            for _ in range(num_decoder_layers)])

    def forward(self, x, relative_positions=None):
        x = self.input_proj(x)
        skips = []
        for layer in self.encoder_layers:
            x = layer(x, relative_positions=relative_positions)
            skips.append(x)
        if self.middle_layer is not None:
            x = self.middle_layer(x, relative_positions=relative_positions)
        for layer in self.decoder_layers:
            skip = skips.pop() if self.use_long_range_skip else None
            x = layer(x, relative_positions=relative_positions, skip=skip)
        return self.output_proj(x)

"""AB-UPT: Anchored-Branched Universal Physics Transformer (Alkin et al.,
2025), ported from https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers
(Emmi AI Non-Production License, see LICENSES/ENPL-AB-UPT.txt and NOTICE).
Modified from the reference: the geometry branch is not ported (no
supernode pooling, geometry transformer or perceiver block), einops is
removed, RoPE uses real arithmetic, the coordinate affine moved into the
input stem, a case-condition MLP is added to the stem for the 2-D
benchmarks, and anchor keys / values are cached for full-field inference.

Shared trunk "sscscs" (the reference's "pscscs" with the perceiver block
replaced by a third split-attention block) + 6 surface and 6 volume
AnchorAttention blocks; dim 192, 3 heads, pre-norm LayerNorm(eps 1e-6),
MLP 4x GELU, trunc_normal(0.02) + zero bias on every Linear.

Input stem: z = branch_mlp(sincos(pos_abupt)) [+ cond_mlp(cond)], with
pos_abupt = (pos + pos_offset) * pos_scale, a fixed affine onto the
reference's non-negative coordinate scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_linear_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ContinuousSincosEmbed(nn.Module):
    def __init__(self, dim: int, ndim: int, max_wavelength: float = 10000.0):
        super().__init__()
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (arange / effective_dim_per_wave))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        theta = coords.float().unsqueeze(-1) * self.omega      # [B, N, ndim, E/2]
        emb = torch.cat([theta.sin(), theta.cos()], dim=-1).flatten(-2)
        if self.padding > 0:
            emb = torch.cat([emb, emb.new_zeros(*emb.shape[:-1], self.padding)], dim=-1)
        return emb


class RopeFrequency(nn.Module):
    """(cos, sin) of the reference's complex rotary frequencies; padded
    pairs get the identity rotation."""

    def __init__(self, dim: int, ndim: int, max_wavelength: float = 10000.0):
        super().__init__()
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        assert self.padding % 2 == 0
        effective_dim_per_wave = (dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (arange / effective_dim_per_wave))

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = (coords.float().unsqueeze(-1) * self.omega).flatten(-2)
        theta = torch.cat([theta, theta.new_zeros(*theta.shape[:-1], self.padding // 2)], dim=-1)
        return theta.cos(), theta.sin()


def apply_rope(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """x [B, H, T, hd]; cos / sin [B, T, hd // 2]; pair j rotated by theta_j."""
    cos, sin = freqs
    xf = x.float()
    xp = xf.reshape(*xf.shape[:-1], -1, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2).type_as(x)


def _slice_freqs(freqs, n: int):
    cos, sin = freqs
    return cos[:, :n], sin[:, :n]


class DotProductAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _qkv(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    def _merge_heads(self, x):
        B, _, T, _ = x.shape
        return x.transpose(1, 2).reshape(B, T, self.dim)

    def forward(self, x, freqs):
        q, k, v = self._qkv(x)
        out = F.scaled_dot_product_attention(apply_rope(q, freqs), apply_rope(k, freqs), v)
        return self.proj(self._merge_heads(out))

    # anchors never read query tokens, so their K/V are chunk-invariant and
    # computed once per case; queries then attend the cached K/V
    def kv_from(self, x_normed, freqs):
        B, T, _ = x_normed.shape
        w, b = self.qkv.weight, self.qkv.bias
        k = F.linear(x_normed, w[self.dim:2 * self.dim], None if b is None else b[self.dim:2 * self.dim])
        v = F.linear(x_normed, w[2 * self.dim:], None if b is None else b[2 * self.dim:])
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return apply_rope(k, freqs), v

    def query_attend(self, x_normed, k_cached, v_cached, freqs_q):
        B, T_q, _ = x_normed.shape
        b = self.qkv.bias
        q = F.linear(x_normed, self.qkv.weight[:self.dim], None if b is None else b[:self.dim])
        q = apply_rope(q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2), freqs_q)
        return self.proj(self._merge_heads(F.scaled_dot_product_attention(q, k_cached, v_cached)))


class SharedweightsSplitattnAttention(DotProductAttention):
    """`s` block: one parameter set, attention within each branch; queries
    attend their own branch's anchors only."""

    def forward(self, x, split_size, freqs):
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        qs, ks, vs = q.split(split_size, dim=2), k.split(split_size, dim=2), v.split(split_size, dim=2)
        if len(split_size) == 4:
            out1 = F.scaled_dot_product_attention(torch.cat([qs[0], qs[1]], dim=2), ks[0], vs[0])
            out2 = F.scaled_dot_product_attention(torch.cat([qs[2], qs[3]], dim=2), ks[2], vs[2])
            out = torch.cat([out1, out2], dim=2)
        elif len(split_size) == 2:
            if split_size[0] == split_size[1]:
                o = F.scaled_dot_product_attention(torch.cat(qs, dim=0), torch.cat(ks, dim=0),
                                                   torch.cat(vs, dim=0))
                out = torch.cat(o.chunk(2, dim=0), dim=2)
            else:
                out = torch.cat([F.scaled_dot_product_attention(qs[i], ks[i], vs[i])
                                 for i in range(2)], dim=2)
        else:
            raise NotImplementedError
        return self.proj(self._merge_heads(out))


class SharedweightsCrossattnAttention(DotProductAttention):
    """`c` block: one parameter set, each branch reads the other branch's anchors."""

    def forward(self, x, split_size, freqs):
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        ks, vs = k.split(split_size, dim=2), v.split(split_size, dim=2)
        if len(split_size) == 4:
            qs = q.split([split_size[0] + split_size[1], split_size[2] + split_size[3]], dim=2)
            out = torch.cat([F.scaled_dot_product_attention(qs[0], ks[2], vs[2]),
                             F.scaled_dot_product_attention(qs[1], ks[0], vs[0])], dim=2)
        elif len(split_size) == 2:
            if split_size[0] == split_size[1]:
                B, H, _, hd = q.shape
                n = split_size[0]
                q2 = q.view(B, H, 2, n, hd).permute(2, 0, 1, 3, 4).reshape(2 * B, H, n, hd)
                o = F.scaled_dot_product_attention(q2, torch.cat([ks[1], ks[0]], dim=0),
                                                   torch.cat([vs[1], vs[0]], dim=0))
                out = o.view(2, B, H, n, hd).permute(1, 2, 0, 3, 4).reshape(B, H, 2 * n, hd)
            else:
                qs = q.split(split_size, dim=2)
                out = torch.cat([F.scaled_dot_product_attention(qs[0], ks[1], vs[1]),
                                 F.scaled_dot_product_attention(qs[1], ks[0], vs[0])], dim=2)
        else:
            raise NotImplementedError
        return self.proj(self._merge_heads(out))


class AnchorAttention(DotProductAttention):
    """Branch block: anchors self-attend; query tokens (after the anchors)
    get a Q-only projection and cross-attend the anchors' K/V."""

    def forward(self, x, freqs, num_anchor_tokens: int | None = None):
        if num_anchor_tokens is None:
            return super().forward(x, freqs)
        n_a = num_anchor_tokens
        anchors, queries = x.split([n_a, x.size(1) - n_a], dim=1)
        q, k, v = self._qkv(anchors)
        B, T_q, _ = queries.shape
        b = self.qkv.bias
        q_query = F.linear(queries, self.qkv.weight[:self.dim], None if b is None else b[:self.dim])
        q = torch.cat([q, q_query.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)], dim=2)
        out = F.scaled_dot_product_attention(apply_rope(q, freqs), apply_rope(k, _slice_freqs(freqs, n_a)), v)
        return self.proj(self._merge_heads(out))


class Mlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_ctor=DotProductAttention):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = attn_ctor(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(self, x, attn_kwargs=None):
        x = x + self.attn(self.norm1(x), **(attn_kwargs or {}))
        return x + self.mlp(self.norm2(x))


def _branch_mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))


class AnchoredBranchedUPT(nn.Module):
    def __init__(self, ndim: int, dim: int = 192, num_heads: int = 3,
                 shared_blocks: str = "sscscs", num_surface_blocks: int = 6,
                 num_volume_blocks: int = 6, output_dim_surface: int = 1,
                 output_dim_volume: int = 5, pos_offset=(1.0, 1.0, 0.0),
                 pos_scale: float = 500.0, cond_dim: int = 0):
        super().__init__()
        self.dim = dim
        self.rope = RopeFrequency(dim=dim // num_heads, ndim=ndim)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.surface_bias = _branch_mlp(dim)
        self.volume_bias = _branch_mlp(dim)
        self.cond_dim = int(cond_dim)
        if self.cond_dim > 0:
            # the case condition is global, so one shared MLP serves both branches
            self.cond_mlp = nn.Sequential(nn.Linear(self.cond_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        ctors = {"s": SharedweightsSplitattnAttention, "c": SharedweightsCrossattnAttention}
        self.shared_pattern = str(shared_blocks)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, ctors[ch]) for ch in shared_blocks])
        self.surface_blocks = nn.ModuleList([TransformerBlock(dim, num_heads, AnchorAttention)
                                             for _ in range(num_surface_blocks)])
        self.volume_blocks = nn.ModuleList([TransformerBlock(dim, num_heads, AnchorAttention)
                                            for _ in range(num_volume_blocks)])
        self.surface_decoder = nn.Linear(dim, output_dim_surface)
        self.volume_decoder = nn.Linear(dim, output_dim_volume)
        self.register_buffer("pos_offset", torch.tensor(tuple(pos_offset), dtype=torch.float32),
                             persistent=False)
        self.pos_scale = float(pos_scale)
        self.apply(_init_linear_weights)

    def _abupt_coords(self, pos: torch.Tensor) -> torch.Tensor:
        return (pos.float() + self.pos_offset) * self.pos_scale

    def _branch_tokens(self, pos_abupt, is_surface: bool, cond):
        p = (self.surface_bias if is_surface else self.volume_bias)(self.pos_embed(pos_abupt))
        if self.cond_dim > 0:
            p = p + self.cond_mlp(cond.to(p.dtype)).unsqueeze(1)
        return p

    def _stem(self, surface_position, volume_position, cond):
        pos_s, pos_v = self._abupt_coords(surface_position), self._abupt_coords(volume_position)
        freqs_s, freqs_v = self.rope(pos_s), self.rope(pos_v)
        freqs_all = (torch.cat([freqs_s[0], freqs_v[0]], dim=1), torch.cat([freqs_s[1], freqs_v[1]], dim=1))
        x = torch.cat([self._branch_tokens(pos_s, True, cond), self._branch_tokens(pos_v, False, cond)], dim=1)
        return x, freqs_s, freqs_v, freqs_all

    def forward(self, surface_position, volume_position, cond=None):
        """Anchors only -> (pred_surface [B, Ns, Cs], pred_volume [B, Nv, Cv])."""
        ns, nv = surface_position.size(1), volume_position.size(1)
        x, freqs_s, freqs_v, freqs_all = self._stem(surface_position, volume_position, cond)
        split_size = [ns, nv]
        for block in self.blocks:
            x = block(x, {"split_size": split_size, "freqs": freqs_all})
        x_s, x_v = x.split(split_size, dim=1)
        for block in self.surface_blocks:
            x_s = block(x_s, {"freqs": freqs_s})
        for block in self.volume_blocks:
            x_v = block(x_v, {"freqs": freqs_v})
        return self.surface_decoder(x_s), self.volume_decoder(x_v)

    def forward_with_queries(self, surface_position, volume_position,
                             num_surface_anchors: int, num_volume_anchors: int, cond=None):
        """Reference anchor/query forward: the first anchors of each branch
        are computed exactly as in forward(); the rest are query tokens that
        only read anchor K/V. The equivalence reference for the cached path."""
        ns_a, nv_a = int(num_surface_anchors), int(num_volume_anchors)
        ns_q, nv_q = surface_position.size(1) - ns_a, volume_position.size(1) - nv_a
        x, freqs_s, freqs_v, freqs_all = self._stem(surface_position, volume_position, cond)
        split_size = [ns_a, ns_q, nv_a, nv_q]
        for block in self.blocks:
            x = block(x, {"split_size": split_size, "freqs": freqs_all})
        x_s, x_v = x.split([ns_a + ns_q, nv_a + nv_q], dim=1)
        for block in self.surface_blocks:
            x_s = block(x_s, {"freqs": freqs_s, "num_anchor_tokens": ns_a})
        for block in self.volume_blocks:
            x_v = block(x_v, {"freqs": freqs_v, "num_anchor_tokens": nv_a})
        return self.surface_decoder(x_s), self.volume_decoder(x_v)

    def encode_anchor_cache(self, surface_position, volume_position, cond=None):
        """Anchors once, capturing every block's post-RoPE anchor K/V.
        -> (pred_surface_anchors, pred_volume_anchors, cache)."""
        ns, nv = surface_position.size(1), volume_position.size(1)
        x, freqs_s, freqs_v, freqs_all = self._stem(surface_position, volume_position, cond)
        split_size = [ns, nv]
        trunk_kv = []
        for block in self.blocks:
            normed = block.norm1(x)
            k_s, v_s = block.attn.kv_from(normed[:, :ns], freqs_s)
            k_v, v_v = block.attn.kv_from(normed[:, ns:], freqs_v)
            trunk_kv.append((k_s, v_s, k_v, v_v))
            x = block(x, {"split_size": split_size, "freqs": freqs_all})
        x_s, x_v = x.split(split_size, dim=1)
        surf_kv, vol_kv = [], []
        for block in self.surface_blocks:
            surf_kv.append(block.attn.kv_from(block.norm1(x_s), freqs_s))
            x_s = block(x_s, {"freqs": freqs_s})
        for block in self.volume_blocks:
            vol_kv.append(block.attn.kv_from(block.norm1(x_v), freqs_v))
            x_v = block(x_v, {"freqs": freqs_v})
        cache = {"trunk": trunk_kv, "surface": surf_kv, "volume": vol_kv}
        return self.surface_decoder(x_s), self.volume_decoder(x_v), cache

    def decode_queries_cached(self, cache, surface_position, volume_position, cond=None):
        """Query rows of one chunk against the cached anchor K/V."""
        pos_s, pos_v = self._abupt_coords(surface_position), self._abupt_coords(volume_position)
        freqs_qs, freqs_qv = self.rope(pos_s), self.rope(pos_v)
        x_s = self._branch_tokens(pos_s, True, cond)
        x_v = self._branch_tokens(pos_v, False, cond)
        for ch, block, (k_s, v_s, k_v, v_v) in zip(self.shared_pattern, self.blocks, cache["trunk"]):
            if ch == "s":      # queries read their own branch's anchors
                kv_for_s, kv_for_v = (k_s, v_s), (k_v, v_v)
            else:              # 'c': the other branch's anchors
                kv_for_s, kv_for_v = (k_v, v_v), (k_s, v_s)
            x_s = x_s + block.attn.query_attend(block.norm1(x_s), *kv_for_s, freqs_qs)
            x_s = x_s + block.mlp(block.norm2(x_s))
            x_v = x_v + block.attn.query_attend(block.norm1(x_v), *kv_for_v, freqs_qv)
            x_v = x_v + block.mlp(block.norm2(x_v))
        for block, (k, v) in zip(self.surface_blocks, cache["surface"]):
            x_s = x_s + block.attn.query_attend(block.norm1(x_s), k, v, freqs_qs)
            x_s = x_s + block.mlp(block.norm2(x_s))
        for block, (k, v) in zip(self.volume_blocks, cache["volume"]):
            x_v = x_v + block.attn.query_attend(block.norm1(x_v), k, v, freqs_qv)
            x_v = x_v + block.mlp(block.norm2(x_v))
        return self.surface_decoder(x_s), self.volume_decoder(x_v)


def build_abupt(ac, spec) -> AnchoredBranchedUPT:
    """ac: config.AbuptConfig; spec: the dataset spec (dim, cond_dim, channels)."""
    return AnchoredBranchedUPT(
        ndim=spec.dim, dim=ac.hidden_dim, num_heads=ac.num_heads,
        shared_blocks=ac.shared_blocks, num_surface_blocks=ac.num_surface_blocks,
        num_volume_blocks=ac.num_volume_blocks,
        output_dim_surface=len(spec.surf_channels), output_dim_volume=len(spec.vol_channels),
        pos_offset=tuple(ac.pos_offset), pos_scale=ac.pos_scale,
        cond_dim=spec.cond_dim if spec.cond_to_baselines else 0)

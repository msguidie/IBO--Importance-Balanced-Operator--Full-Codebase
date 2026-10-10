"""Local decoder: cross-attention of each query point to its k nearest
latent tokens.

Each query gets a self-embedding assembled from the dataset's feature
groups (Fourier features, plain features, categorical embeddings), the log
mean neighbour distance and, optionally, the case vector; it is routed
through a volume or a surface first layer and given a type embedding. Per
neighbour j a pair embedding encodes the normalised offset to the token. A
stage is multi-head attention over the k neighbours,

  score_j = <RMSNorm(W_Q (h + p_j)), RMSNorm(W_K t_j)> / sqrt(d) - lambda_h log(d_j / dbar)
  out     = sum_j softmax_j(score) (W_V t_j + gamma * p_j)

followed by a SwiGLU feed-forward layer. lambda_h starts at 1 (an
inverse-distance prior). With two stages the first reads the encoder
tokens and the second the processor tokens; registers, when attended, use
a separate softmax added through a small gate. Neighbours form a tensor
axis [B, N, K, ...]: one F.embedding gather per table (segmented backward
instead of an atomic scatter), fp32 scores, and the value aggregate split
as sum_j a_j v_j + gamma * sum_j a_j p_j.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Fourier, RMSNorm, SwiGLUFFN, init_linear


class QueryEmbedding(nn.Module):
    """cat over the dataset's feature groups: fourier(x), x, or embedding(code)."""

    def __init__(self, groups):
        super().__init__()
        self.kinds = []
        self.encoders = nn.ModuleList()
        self.out_dim = 0
        for g in groups:
            if g.encode == "fourier":
                enc = Fourier(g.n_freqs, g.dim)
                self.out_dim += enc.out_dim
            elif g.encode == "embed":
                enc = nn.Embedding(g.n_classes, g.embed_dim)
                self.out_dim += g.embed_dim
            else:
                enc = nn.Identity()
                self.out_dim += g.dim
            self.kinds.append(g.encode)
            self.encoders.append(enc)

    def forward(self, tensors, dtype) -> list[torch.Tensor]:
        out = []
        for kind, enc, t in zip(self.kinds, self.encoders, tensors):
            if kind == "embed":
                out.append(enc(t.long()).to(dtype))
            elif kind == "fourier":
                out.append(enc(t).to(dtype))
            else:
                out.append(t.to(dtype))
        return out


class _Stage(nn.Module):
    def __init__(self, latent_dim: int, d_h: int, n_heads: int, registers: bool):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_h // n_heads
        self.src_norm = RMSNorm(latent_dim)
        self.k_proj = nn.Linear(latent_dim, d_h)
        self.v_proj = nn.Linear(latent_dim, d_h)
        self.h_norm = RMSNorm(d_h)
        self.q_proj = nn.Linear(d_h, d_h)
        self.q_head_norm = RMSNorm(self.head_dim)
        self.k_head_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_h, d_h)       # zero: attention adds nothing at init
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.lam = nn.Parameter(torch.ones(n_heads))
        self.gamma = nn.Parameter(torch.empty(d_h))
        nn.init.trunc_normal_(self.gamma, std=0.02)
        self.gate_reg = nn.Parameter(torch.full((n_heads,), 1e-2)) if registers else None
        self.ffn_norm = RMSNorm(d_h)
        self.ffn = SwiGLUFFN(d_h, 4 * d_h, 0.0, "trunc_normal")
        for m in (self.k_proj, self.v_proj, self.q_proj):
            init_linear(m, "trunc_normal")


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, d_h: int, n_heads: int, dim: int,
                 query_groups, pair_fourier_freqs, pair_unit_dir: bool,
                 pair_log_dbar: bool, cond_dim: int, cond_concat: bool,
                 stages: int, n_register: int, register_attn: bool,
                 head_hidden: int, n_vol: int, n_surf: int,
                 lambda_clip: float | None, log_ratio_clip: float | None):
        super().__init__()
        assert d_h % n_heads == 0
        self.d_h, self.n_heads, self.head_dim = d_h, n_heads, d_h // n_heads
        self.n_register = n_register
        self.register_attn = bool(register_attn) and n_register > 0
        self.lambda_clip = lambda_clip
        self.log_ratio_clip = log_ratio_clip
        self.n_vol = n_vol
        self.cond_concat = bool(cond_concat) and cond_dim > 0

        self.query_embed = QueryEmbedding(query_groups)
        d_self = self.query_embed.out_dim + 1 + (cond_dim if self.cond_concat else 0)
        self.pair_fourier = nn.ModuleList([Fourier(int(k), 1) for k in pair_fourier_freqs])
        assert len(self.pair_fourier) == dim
        self.pair_unit_dir, self.pair_log_dbar = bool(pair_unit_dir), bool(pair_log_dbar)
        d_pair = sum(f.out_dim for f in self.pair_fourier) + 1 \
            + (dim if self.pair_unit_dir else 0) + (1 if self.pair_log_dbar else 0)

        self.pair_proj = nn.Linear(d_pair, d_h)
        self.self_proj1_vol = nn.Linear(d_self, d_h)
        self.self_proj1_surf = nn.Linear(d_self, d_h)
        self.self_proj2 = nn.Linear(d_h, d_h)
        self.type_emb = nn.Embedding(2, d_h)
        nn.init.trunc_normal_(self.type_emb.weight, std=0.02)
        self.stages = nn.ModuleList(
            [_Stage(latent_dim, d_h, n_heads, registers=False) for _ in range(stages - 1)]
            + [_Stage(latent_dim, d_h, n_heads, registers=self.register_attn)])
        self.head = nn.Linear(d_h, head_hidden)
        self.pred = nn.Linear(head_hidden, n_vol + n_surf)
        for m in (self.pair_proj, self.self_proj1_vol, self.self_proj1_surf,
                  self.self_proj2, self.head, self.pred):
            init_linear(m, "trunc_normal")

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.n_heads, self.head_dim)

    @staticmethod
    def _gather(table: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        """[B, L, *tail] -> [B, N, K, *tail] by batch-flattened row index."""
        tail = table.shape[2:]
        flat = table.reshape(table.shape[0] * table.shape[1], -1)
        return F.embedding(flat_idx, flat).view(*flat_idx.shape, *tail)

    def forward(self, encoder_out, vit_out, query_pos_geo, feats, knn,
                latent_pos_geo, is_volume, cond):
        """feats: one tensor per spec.query_groups entry, in order (codes as
        integer tensors)."""
        knn = knn.long()
        B, N, K = knn.shape
        L = encoder_out.shape[1]
        dtype = vit_out.dtype
        H, hd = self.n_heads, self.head_dim
        flat_idx = knn + (torch.arange(B, device=knn.device)[:, None, None] * L) if B > 1 else knn
        R = self.n_register
        vit_cells = vit_out[:, :-R, :] if R > 0 else vit_out
        vit_regs = vit_out[:, -R:, :] if R > 0 else None

        # per-query geometry in fp32, one gather over the K neighbours
        cj = self._gather(latent_pos_geo.float(), flat_idx)              # [B,N,K,d]
        delta = query_pos_geo.float().unsqueeze(2) - cj
        d = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dbar = d.mean(dim=2, keepdim=True).clamp(min=1e-6)
        log_dbar = torch.log(dbar)                                        # [B,N,1,1]
        log_ratio = torch.log(d / dbar)                                   # [B,N,K,1]

        # query self-embedding
        parts = self.query_embed(feats, dtype) + [log_dbar[:, :, 0, :].to(dtype)]
        if self.cond_concat:
            parts.append(cond.to(dtype)[:, None, :].expand(B, N, -1))
        q_self = torch.cat(parts, dim=-1)
        mask = is_volume.to(dtype)[..., None]
        h = mask * F.silu(self.self_proj1_vol(q_self)) \
            + (1.0 - mask) * F.silu(self.self_proj1_surf(q_self))
        h = self.self_proj2(h) + self.type_emb(is_volume.long())

        # pair embedding, shared by every stage
        rel = delta / dbar
        p_parts = [f(rel[..., a:a + 1]).to(dtype) for a, f in enumerate(self.pair_fourier)]
        if self.pair_unit_dir:
            p_parts.append((delta / d).to(dtype))
        p_parts.append(log_ratio.to(dtype))
        if self.pair_log_dbar:
            p_parts.append(log_dbar.expand(B, N, K, 1).to(dtype))
        p = F.silu(self.pair_proj(torch.cat(p_parts, dim=-1)))           # [B,N,K,d_h]
        pj = self._split(p)

        lr = log_ratio if self.log_ratio_clip is None else \
            log_ratio.clamp(min=-self.log_ratio_clip, max=self.log_ratio_clip)
        for si, st in enumerate(self.stages):
            src_tokens = encoder_out if si < len(self.stages) - 1 else vit_cells
            src = st.src_norm(src_tokens)
            k_tab = st.k_head_norm(self._split(st.k_proj(src)))          # [B,L,H,hd]
            v_tab = st.v_proj(src)                                        # [B,L,d_h]
            lam = st.lam if self.lambda_clip is None else \
                st.lam.clamp(min=-self.lambda_clip, max=self.lambda_clip)
            h_normed = st.h_norm(h)
            q = st.q_head_norm(self._split(st.q_proj(h_normed.unsqueeze(2) + p)))  # [B,N,K,H,hd]
            kj = self._gather(k_tab, flat_idx)
            vj = self._split(self._gather(v_tab, flat_idx))
            score = (q * kj).sum(dim=-1, dtype=torch.float32) / (hd ** 0.5)  # [B,N,K,H]
            score = score - lam.view(1, 1, 1, H) * lr
            alpha = F.softmax(score, dim=2).to(dtype)
            agg = torch.einsum("bnkh,bnkhd->bnhd", alpha, vj) \
                + st.gamma.view(H, hd) * torch.einsum("bnkh,bnkhd->bnhd", alpha, pj)
            if st.gate_reg is not None:
                reg_src = st.src_norm(vit_regs)
                k_reg = st.k_head_norm(self._split(st.k_proj(reg_src)))   # [B,R,H,hd]
                v_reg = self._split(st.v_proj(reg_src))
                q_reg = st.q_head_norm(self._split(st.q_proj(h_normed)))
                reg_scores = torch.einsum("bnhd,brhd->bnrh", q_reg.float(), k_reg.float()) / (hd ** 0.5)
                alpha_reg = F.softmax(reg_scores, dim=2).to(dtype)
                agg = agg + st.gate_reg[None, None, :, None] \
                    * torch.einsum("bnrh,brhd->bnhd", alpha_reg, v_reg)
            h = h + st.out_proj(agg.reshape(B, N, self.d_h))
            h = h + st.ffn(st.ffn_norm(h))

        pred = self.pred(F.silu(self.head(h)))
        return pred[..., :self.n_vol], pred[..., self.n_vol:]

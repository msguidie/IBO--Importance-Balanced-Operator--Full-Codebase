"""GAOT: MAGNO encoder -> ViT -> MAGNO decoder over a regular latent grid.

Per-point input features are the dataset's native point features (the
case condition appended per node, GAOT's own recommended conditioning);
coordinates are the dataset's normalised coordinates, in which the latent
grid and the gathering radius are defined by the derive step. One output
head emits every channel; the recipe reads the volume and surface
predictions from it.

Batching: the B cases of a step are merged into one disjoint-union CSR
graph so each AGNO runs once per step (numerically identical to the
upstream per-case loop, since segment reductions never cross rows). The
geoembed features stay per case and are precomputed by the derive step.
No torch.compile: edge counts differ per case.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .agno import AGNO
from .attn import AttentionConfig, Transformer, TransformerConfig
from .gemb import GeoEmbedMLP
from .mlp import ChannelMLP


def gather_csr_rows(index, splits, alpha, sel):
    """Row subset of a CSR graph (each kept row keeps its full neighbour
    list) -> (index_sub, splits_sub, alpha_sub)."""
    counts = splits[sel + 1] - splits[sel]
    splits_sub = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    total = int(splits_sub[-1])
    starts = torch.repeat_interleave(splits[sel], counts)
    within = torch.arange(total, device=index.device) - torch.repeat_interleave(splits_sub[:-1], counts)
    edge_pos = starts + within
    return index[edge_pos], splits_sub, alpha[edge_pos]


def merge_csr(indexes, splitss, alphas, src_offsets):
    """Disjoint-union merge of per-case CSR graphs; src_offsets[b] is added
    to case b's edge indices."""
    idx_parts, split_parts, edge_off = [], [splitss[0].new_zeros(1)], 0
    for b, (idx, spl) in enumerate(zip(indexes, splitss)):
        idx_parts.append(idx + src_offsets[b])
        split_parts.append(spl[1:] + edge_off)
        edge_off += int(spl[-1])
    return torch.cat(idx_parts), torch.cat(split_parts), torch.cat(alphas)


class GAOT(nn.Module):
    def __init__(self, g, grid_coords: torch.Tensor, in_channels: int,
                 out_channels: int, coord_dim: int = 2):
        """g: config.GaotConfig; grid_coords [L, coord_dim] in the normalised
        coordinate space (from the derive step), kept as a buffer."""
        super().__init__()
        latent = int(g.latent_size)
        self.grid_h, self.grid_w = int(g.grid_h), int(g.grid_w)
        self.n_tokens = self.grid_h * self.grid_w
        self.patch_size = int(g.patch_size)
        assert grid_coords.shape == (self.n_tokens, coord_dim)
        assert self.grid_h % self.patch_size == 0 and self.grid_w % self.patch_size == 0
        self.register_buffer("grid_coords", grid_coords.float(), persistent=True)
        kernel_layers = [2 * coord_dim] + [int(g.kernel_hidden)] * int(g.kernel_layers) + [latent]

        self.lifting = ChannelMLP(in_channels=in_channels, out_channels=latent,
                                  hidden_channels=int(g.kernel_hidden), n_layers=1)
        self.enc_agno = AGNO(channel_mlp_layers=kernel_layers)
        self.enc_geoembed = GeoEmbedMLP(input_dim=coord_dim, output_dim=latent)
        self.enc_recovery = ChannelMLP(in_channels=2 * latent, out_channels=latent, n_layers=1)

        pv = self.patch_size ** 2
        self.patch_linear = nn.Linear(pv * latent, pv * latent)
        self.positional_embedding = str(g.positional_embedding)
        tconf = TransformerConfig(
            patch_size=self.patch_size, hidden_size=int(g.vit_hidden),
            num_layers=int(g.vit_layers), positional_embedding=self.positional_embedding,
            use_long_range_skip=bool(g.use_long_range_skip),
            ffn_multiplier=int(g.ffn_multiplier),
            attn_config=AttentionConfig(num_heads=int(g.num_heads), num_kv_heads=int(g.num_kv_heads),
                                        atten_dropout=float(g.atten_dropout)))
        self.processor = Transformer(input_size=pv * latent, output_size=pv * latent, config=tconf)
        ph, pw = self.grid_h // self.patch_size, self.grid_w // self.patch_size
        positions = torch.stack(torch.meshgrid(torch.arange(ph, dtype=torch.float32),
                                               torch.arange(pw, dtype=torch.float32),
                                               indexing="ij"), dim=-1).reshape(-1, 2)
        self.register_buffer("patch_positions", positions, persistent=False)
        if self.positional_embedding == "absolute":
            self.register_buffer("abs_pos_emb", self._absolute_embeddings(positions, pv * latent),
                                 persistent=False)

        self.dec_agno = AGNO(channel_mlp_layers=kernel_layers)
        self.dec_geoembed = GeoEmbedMLP(input_dim=coord_dim, output_dim=latent)
        self.dec_recovery = ChannelMLP(in_channels=2 * latent, out_channels=latent, n_layers=1)
        self.projection = ChannelMLP(in_channels=latent, out_channels=out_channels,
                                     hidden_channels=int(g.kernel_hidden), n_layers=1)

    @staticmethod
    def _absolute_embeddings(positions: torch.Tensor, embed_dim: int) -> torch.Tensor:
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        inv_freq = 1.0 / (10000 ** (torch.arange(dim_touse, dtype=torch.float32) / dim_touse))
        sinusoid = positions[:, :, None] * inv_freq[None, None, :]
        return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1).view(positions.size(0), -1)

    @staticmethod
    def _cmlp_flat(mlp: ChannelMLP, x: torch.Tensor) -> torch.Tensor:
        return mlp(x.t().unsqueeze(0)).squeeze(0).t()

    def _patchify(self, x):
        p, c = self.patch_size, x.shape[-1]
        if p == 1:
            return x
        b, h, w = x.shape[0], self.grid_h, self.grid_w
        x = x.view(b, h // p, p, w // p, p, c).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(b, (h // p) * (w // p), p * p * c)

    def _unpatchify(self, x, c):
        p = self.patch_size
        if p == 1:
            return x
        b, h, w = x.shape[0], self.grid_h, self.grid_w
        x = x.view(b, h // p, w // p, p, p, c).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(b, h * w, c)

    def forward(self, batch: dict) -> torch.Tensor:
        """batch: pndata [sumN, F], x_coords [sumN, d], enc_index / enc_splits /
        enc_alpha (merged encoder CSR, rows = B * n_tokens), enc_geo [B, L, 3 + 2d],
        query_coords [sumQ, d], dec_index / dec_splits / dec_alpha (merged
        decoder CSR into B * L tokens), dec_geo [sumQ, 3 + 2d], batch_size.
        -> pred [sumQ, out_channels]."""
        bsz = int(batch["batch_size"])
        lifted = self._cmlp_flat(self.lifting, batch["pndata"])
        grid_tiled = self.grid_coords.repeat(bsz, 1)
        encoded = self.enc_agno(y_coords=batch["x_coords"], x_coords=grid_tiled, f_y=lifted,
                                neighbors_index=batch["enc_index"],
                                neighbors_row_splits=batch["enc_splits"],
                                attention_weights=batch["enc_alpha"]).view(bsz, self.n_tokens, -1)
        encoded = torch.cat([encoded, self.enc_geoembed(batch["enc_geo"])], dim=-1)
        encoded = self.enc_recovery(encoded.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.patch_linear(self._patchify(encoded))
        if self.positional_embedding == "absolute":
            x = x + self.abs_pos_emb
            rel = None
        else:
            rel = self.patch_positions
        rndata = self._unpatchify(self.processor(x, relative_positions=rel), encoded.shape[-1])

        rn_flat = rndata.reshape(bsz * self.n_tokens, -1)
        decoded = self.dec_agno(y_coords=grid_tiled, x_coords=batch["query_coords"], f_y=rn_flat,
                                neighbors_index=batch["dec_index"],
                                neighbors_row_splits=batch["dec_splits"],
                                attention_weights=batch["dec_alpha"])
        decoded = torch.cat([decoded, self.dec_geoembed(batch["dec_geo"])], dim=-1)
        decoded = self._cmlp_flat(self.dec_recovery, decoded)
        return self._cmlp_flat(self.projection, decoded)


def build_gaot(g, grid_coords: torch.Tensor, in_channels: int, out_channels: int,
               coord_dim: int = 2) -> GAOT:
    return GAOT(g, grid_coords, in_channels, out_channels, coord_dim)

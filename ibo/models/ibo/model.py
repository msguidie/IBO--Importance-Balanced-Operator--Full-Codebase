"""IBO (Importance-Balanced Operator): the full model.

encode() runs the slot encoder, the token projection and the global
processor once per case; decode() predicts any set of query points from
their k nearest tokens. The two are separate so that full-field inference
encodes once and decodes in fixed-size chunks through the same code path
as training. Every forward takes explicit tensors (feature groups as
tuples in the dataset spec's order), which keeps compiled graphs clean.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import TokenEncoder
from .pointnet import SlotEncoder
from .processor import Processor


class IBO(nn.Module):
    def __init__(self, m, spec, grad_checkpoint: bool = False):
        """m: config.IBOConfig; spec: datasets.base.DatasetSpec."""
        super().__init__()
        m.validate()
        self.n_vol, self.n_surf = len(spec.vol_channels), len(spec.surf_channels)
        self.slots = SlotEncoder(spec.slot_feat_dim, m.pn_hidden, m.pn_out,
                                 act=m.pn_act, fp32_first=m.pn_fp32_first)
        self.encoder = TokenEncoder(3 * m.pn_out + spec.stats_dim, m.latent_dim,
                                    layers=m.token_proj_layers, cond_dim=spec.cond_dim,
                                    film=m.encoder_film, init=m.init)
        self.processor = Processor(
            m.latent_dim, m.n_blocks, m.num_heads, m.head_dim, m.ffn_hidden,
            m.n_register, m.dropout, m.rope_dims, m.rope_base, m.rope_scales,
            m.rope_domain_lo, m.rope_domain_hi, spec.cond_dim, m.adaln,
            m.adaln_hidden, m.long_skips, m.init, grad_checkpoint)
        self.decoder = Decoder(
            m.latent_dim, m.decoder_dh, m.decoder_heads, spec.dim,
            spec.query_groups, spec.pair_fourier_freqs, spec.pair_unit_dir,
            spec.pair_log_dbar, spec.cond_dim, m.decoder_cond_concat,
            m.decoder_stages, m.n_register, m.decoder_register_attn,
            m.head_hidden, self.n_vol, self.n_surf, m.idw_lambda_clip,
            m.idw_log_ratio_clip)

    def encode(self, latent_slots, latent_mask, latent_stats, latent_pos_geo, cond):
        """-> (encoder_out [B, L, D], vit_out [B, L + R, D])."""
        pooled = self.slots(latent_slots, latent_mask)
        encoder_out = self.encoder(pooled, latent_stats, cond)
        return encoder_out, self.processor(encoder_out, latent_pos_geo, cond)

    def decode(self, encoder_out, vit_out, query_pos_geo, feats, knn,
               latent_pos_geo, is_volume, cond):
        return self.decoder(encoder_out, vit_out, query_pos_geo, feats, knn,
                            latent_pos_geo, is_volume, cond)

    def forward(self, latent_slots, latent_mask, latent_stats, latent_pos_geo, cond,
                query_pos_geo, feats, knn, is_volume):
        encoder_out, vit_out = self.encode(latent_slots, latent_mask, latent_stats,
                                           latent_pos_geo, cond)
        return self.decode(encoder_out, vit_out, query_pos_geo, feats, knn,
                           latent_pos_geo, is_volume, cond)

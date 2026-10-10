"""Recipe "ibo": the Importance-Balanced Operator. Lion with a linear
warm-up and a cosine decay; full-field inference encodes a case once and
decodes its points in fixed-size chunks (the last chunk padded) through
the same model code as training."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..evaluation.flops import sdpa_flops
from ..models.ibo import IBO
from ..training.lion import Lion
from ..training.losses import compute_loss
from ..training.schedules import build_warmup_cosine_fraction
from .base import Recipe, cpu_defloat_bf16

ENCODE_KEYS = ("latent_slots", "latent_mask", "latent_stats", "latent_pos_geo", "cond")


def _unsq(t, device, cast_fp32: bool):
    t = t.unsqueeze(0).to(device)
    if cast_fp32 and t.is_floating_point() and t.dtype == torch.bfloat16:
        t = t.float()
    return t


def _pad_rows(t: torch.Tensor, pad: int) -> torch.Tensor:
    """Repeat the last row pad times (padded predictions are discarded)."""
    return torch.cat([t, t[-1:].expand(pad, *t.shape[1:])], dim=0)


class IBORecipe(Recipe):
    name = "ibo"
    view = "ibo"

    def build_model(self, cfg, spec) -> torch.nn.Module:
        return IBO(cfg.ibo, spec, grad_checkpoint=cfg.training.activation_checkpointing)

    def forward_loss(self, model, batch, cfg, spec):
        tc = cfg.training
        pv, ps = model(batch["latent_slots"], batch["latent_mask"], batch["latent_stats"],
                       batch["latent_pos_geo"], batch.get("cond"), batch["query_pos_geo"],
                       tuple(batch["query_feats"]), batch["query_knn"], batch["is_volume"])
        weights = tc.loss_vol_weights if tc.loss_vol_weights is not None else spec.loss_vol_weights
        return compute_loss(pv, ps, batch["target_vol"], batch["target_surf"],
                            batch["is_volume"], weights, tc.loss_reduction)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        tc = cfg.training
        optimizer = Lion(model.parameters(), lr=tc.lr, betas=tc.betas,
                         weight_decay=tc.weight_decay)
        total = tc.epochs * opt_steps_per_epoch
        sched = build_warmup_cosine_fraction(optimizer, total, tc.warmup_fraction,
                                             tc.min_lr_fraction, tc.warmup_steps)
        return optimizer, sched, "step"

    # ---- inference --------------------------------------------------------
    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> str:
        if not (use_cuda and cfg.training.compile):
            return "eager"
        # encode and decode compiled separately, one static shape each
        model.encode = torch.compile(model.encode, mode="reduce-overhead", fullgraph=True,
                                     dynamic=False)
        model.decode = torch.compile(model.decode, mode="reduce-overhead", fullgraph=True,
                                     dynamic=False)
        return "compiled (encode once, fixed decode chunks)"

    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        cast = not use_cuda
        if cast:
            case = cpu_defloat_bf16(case)
        chunk = int(cfg.evaluation.decode_chunk)
        n_vol, n_surf = int(case["n_vol"]), int(case["n_surf"])
        n_total = n_vol + n_surf
        auto = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda)
        with auto:
            cond = _unsq(case["cond"], device, cast) if case.get("cond") is not None else None
            enc, vit = model.encode(*[_unsq(case[k], device, cast) for k in ENCODE_KEYS[:4]], cond)
        if use_cuda:
            enc, vit = enc.clone(), vit.clone()      # off the compiled graph's pool
        latent_pos = _unsq(case["latent_pos_geo"], device, cast)
        pv_parts, ps_parts = [], []
        for lo in range(0, n_total, chunk):
            hi = min(lo + chunk, n_total)
            pad = chunk - (hi - lo)

            def rows(t):
                t = t[lo:hi]
                return _unsq(_pad_rows(t, pad) if pad else t, device, cast)

            feats = tuple(rows(f) for f in case["query_feats"])
            with auto:
                pv, ps = model.decode(enc, vit, rows(case["query_pos_geo"]), feats,
                                      rows(case["query_knn"]), latent_pos, rows(case["is_volume"]),
                                      cond)
            pv_parts.append(pv[0, :hi - lo].float().cpu().numpy())
            ps_parts.append(ps[0, :hi - lo].float().cpu().numpy())
        pv, ps = np.concatenate(pv_parts), np.concatenate(ps_parts)
        return pv[:n_vol], ps[n_vol:]

    # ---- FLOPs the hook meter cannot see ---------------------------------
    def manual_infer_flops(self, cfg, spec, n_vol, n_surf):
        m = cfg.ibo
        L = m.n_tokens + m.n_register
        chunk = int(cfg.evaluation.decode_chunk)
        n = int(math.ceil((n_vol + n_surf) / chunk)) * chunk
        fourier_dim = sum(g.dim * 2 * g.n_freqs for g in spec.query_groups if g.encode == "fourier")
        pair_dim = 2 * sum(spec.pair_fourier_freqs)
        return {
            "processor_sdpa": m.n_blocks * sdpa_flops(L, L, m.num_heads, m.head_dim),
            "rope_trig": 2.0 * m.n_blocks * L * m.head_dim,
            "decoder_neighbour_attention": m.decoder_stages * n * (
                4.0 * m.knn * m.decoder_dh + 3.0 * m.decoder_heads * m.knn),
            "decoder_fourier_trig": n * (fourier_dim + m.knn * pair_dim),
            "slot_pooling": 3.0 * m.n_tokens * m.n_slots * m.pn_out,
            "silu": 4.0 * (m.n_blocks * L * m.ffn_hidden + m.adaln_hidden
                           + n * (2.0 * m.decoder_dh + m.knn * m.decoder_dh + m.head_hidden)),
        }


RECIPE = IBORecipe()

"""Recipe "gaot": GAOT at its published optimizer (AdamW with the
three-phase per-epoch schedule). The dataset's "gaot" view hands over an
assembled disjoint-union graph batch (the derive step bakes the CSR graphs,
the cosine attention weights and the geometric embedding features per
case); the single output head's pressure column doubles as the surface
prediction. Runs eager: edge counts differ per case."""
from __future__ import annotations

import numpy as np
import torch

from ..datasets import get_dataset
from ..evaluation.flops import sdpa_flops
from ..models.gaot import build_gaot
from ..training.losses import compute_loss
from ..training.schedules import GaotMixScheduler
from .base import Recipe

GPU_KEYS = ("pndata", "x_coords", "target_vol", "target_surf", "is_volume", "enc_index",
            "enc_splits", "enc_alpha", "enc_geo", "query_coords", "dec_index", "dec_splits",
            "dec_alpha", "dec_geo")


class GaotRecipe(Recipe):
    name = "gaot"
    view = "gaot"

    def compile_plan(self, cfg):
        return ("", False)

    def ensure_derived(self, cfg, dataset, names: list[str]) -> None:
        dataset.ensure_derived(cfg, "gaot", names)

    def build_model(self, cfg, spec) -> torch.nn.Module:
        meta = get_dataset(cfg.run.dataset).gaot_meta(cfg)
        return build_gaot(cfg.gaot, meta["grid"], in_channels=int(meta["in_channels"]),
                          out_channels=len(spec.vol_channels), coord_dim=spec.dim)

    def batch_to_device(self, batch, device):
        out = dict(batch)
        for k in GPU_KEYS:
            if isinstance(out.get(k), torch.Tensor):
                out[k] = out[k].to(device, non_blocking=True)
        return out

    def forward_loss(self, model, batch, cfg, spec):
        tc = cfg.training
        col = spec.vol_channels.index("p")
        pred = model(batch).view(int(batch["batch_size"]), -1, len(spec.vol_channels))
        weights = tc.loss_vol_weights if tc.loss_vol_weights is not None else spec.loss_vol_weights
        return compute_loss(pred, pred[..., col:col + 1], batch["target_vol"], batch["target_surf"],
                            batch["is_volume"], weights, tc.loss_reduction)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        g = cfg.gaot
        optimizer = torch.optim.AdamW(model.parameters(), lr=g.lr, weight_decay=g.weight_decay)
        sched = GaotMixScheduler(optimizer, total_epochs=cfg.training.epochs, initial_lr=g.lr,
                                 max_lr=g.max_lr, min_lr=g.min_lr, final_lr=g.final_lr,
                                 warmup_fraction=g.warmup_fraction, cosine_fraction=g.cosine_fraction)
        return optimizer, sched, "epoch"

    # ---- inference --------------------------------------------------------
    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        """The whole case in one call (rows volume first); the pressure
        column of the surface rows is the surface prediction."""
        batch = self.batch_to_device(case["gaot"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            pred = model(batch)
        pred = pred.float().cpu().numpy().astype(np.float32)
        n_vol = int(case["n_vol"])
        col = get_dataset(cfg.run.dataset).spec.vol_channels.index("p")
        return pred[:n_vol], pred[n_vol:, col:col + 1].copy()

    def manual_infer_flops(self, cfg, spec, n_vol, n_surf):
        g = cfg.gaot
        p = int(g.patch_size)
        seq = (g.grid_h // p) * (g.grid_w // p)
        hidden = int(g.vit_hidden) * p * p
        return {"vit_sdpa": int(g.vit_layers) * sdpa_flops(seq, seq, g.num_heads, hidden // g.num_heads)}


RECIPE = GaotRecipe()

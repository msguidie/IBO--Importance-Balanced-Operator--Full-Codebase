"""Recipe "erwin": Erwin on coordinates through a Gaussian random-Fourier
encoder (plus the case condition as extra channels where the dataset has
one). AdamW, cosine per optimizer step. The ball tree and the kNN graph of
every sample or chunk are built on the CPU by the dataset / this recipe,
outside any compiled graph."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..evaluation.flops import sdpa_flops
from ..models.erwin import build_erwin_model
from ..models.erwin.balltree import build_tree, knn_edges, restore_index
from ..training.losses import compute_loss
from ..training.schedules import build_warmup_cosine
from .base import Recipe


class ErwinRecipe(Recipe):
    name = "erwin"
    view = "erwin"

    def apply(self, cfg, spec) -> None:
        cfg.erwin.dimensionality = int(spec.dim)

    def build_model(self, cfg, spec) -> torch.nn.Module:
        return build_erwin_model(cfg.erwin, out_dim_vol=len(spec.vol_channels),
                                 extra_in=spec.cond_dim if spec.cond_to_baselines else 0)

    def compile_plan(self, cfg):
        # BatchNorm buffers mutate inside the graph: no CUDA-graph capture
        return ("default" if cfg.training.compile else "", True)

    def forward_loss(self, model, batch, cfg, spec):
        tc = cfg.training
        pv, ps = model(batch["pos"], batch.get("extra"), batch.get("edge_index"), batch["tree_idx"],
                       batch["tree_mask"], batch["tree_idx_rot"], batch["restore_idx"])
        weights = tc.loss_vol_weights if tc.loss_vol_weights is not None else spec.loss_vol_weights
        return compute_loss(pv, ps, batch["target_vol"].unsqueeze(0), batch["target_surf"].unsqueeze(0),
                            batch["is_volume"].unsqueeze(0), weights, tc.loss_reduction)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        e = cfg.erwin
        optimizer = torch.optim.AdamW(model.parameters(), lr=e.lr, weight_decay=e.weight_decay)
        total = cfg.training.epochs * opt_steps_per_epoch
        sched = build_warmup_cosine(optimizer, total, e.lr, e.end_lr, e.warmup_fraction)
        return optimizer, sched, "step"

    # ---- inference --------------------------------------------------------
    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> str:
        self._infer = {}
        if not (use_cuda and cfg.training.compile):
            return "eager"
        self._infer["compiled"] = torch.compile(model, mode="reduce-overhead", fullgraph=True,
                                                dynamic=False)
        return "compiled (fixed chunk rows; trees built on the CPU per chunk)"

    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        e = cfg.erwin
        n_vol, n_surf = int(case["n_vol"]), int(case["n_surf"])
        n = n_vol + n_surf
        pos_all = case["pos"].float()
        extra_all = case.get("extra")
        out_vol = np.empty((n_vol, model.vol_head[-1].out_features), dtype=np.float32)
        out_surf = np.empty((n_surf, model.surf_head[-1].out_features), dtype=np.float32)
        compiled = getattr(self, "_infer", {}).get("compiled")
        R = int(e.infer_chunk)
        fixed = compiled is not None
        fwd = compiled if fixed else model
        n_chunks = max(1, math.ceil(n / R))
        for c in range(n_chunks):
            rows = torch.arange(c, n, n_chunks)               # density-preserving stride
            n_real = int(rows.shape[0])
            rows_run = rows.repeat((R + n_real - 1) // n_real)[:R] if (fixed and n_real < R) else rows
            pos = pos_all[rows_run]
            ti, tm, rot = build_tree(pos, e.strides, e.ball_sizes, e.rotate)
            ei = knn_edges(pos, int(e.knn)) if e.mp_steps > 0 else None
            ri = restore_index(ti, tm)
            extra = None if extra_all is None else extra_all[rows_run].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                pv, ps = fwd(pos.to(device), extra, None if ei is None else ei.to(device),
                             ti.to(device), tm.to(device), [r.to(device) for r in rot], ri.to(device))
            pv = pv.squeeze(0).float().cpu().numpy()[:n_real]
            ps = ps.squeeze(0).float().cpu().numpy()[:n_real]
            vol_mask = (rows < n_vol).numpy()
            out_vol[rows[rows < n_vol].numpy()] = pv[vol_mask]
            out_surf[(rows[rows >= n_vol] - n_vol).numpy()] = ps[~vol_mask]
        return out_vol, out_surf

    # ---- FLOPs the hook meter cannot see ---------------------------------
    def manual_infer_flops(self, cfg, spec, n_vol, n_surf):
        e = cfg.erwin
        n = n_vol + n_surf
        n_chunks = max(1, math.ceil(n / int(e.infer_chunk)))
        ball = e.ball_sizes[0]
        rows = float(ball * max(1, 2 ** math.ceil(math.log2(max(math.ceil(n / n_chunks) / ball, 1)))))
        sizes = [rows]
        for s in e.strides:
            sizes.append(sizes[-1] / s)
        levels = list(zip(e.c_hidden, e.enc_num_heads, e.enc_depths, e.ball_sizes))
        dec = list(zip(e.c_hidden[:-1], e.dec_num_heads, e.dec_depths, e.ball_sizes[:-1]))
        sdpa = cdist = act = 0.0
        for i, (dim, h, depth, b) in enumerate(levels + dec):
            r = sizes[i if i < len(levels) else i - len(levels)]
            sdpa += depth * (r / b) * sdpa_flops(b, b, h, dim // h)
            cdist += depth * (r / b) * b * b * (2 * e.dimensionality + 1)
            act += depth * r * dim * e.mlp_ratio * 4.0
        terms = {"bmsa_sdpa": sdpa, "bmsa_distance_mask": cdist, "rff_trig": rows * e.c_in,
                 "swiglu_silu": act}
        if e.mp_steps > 0:
            terms["mpnn_mean"] = e.mp_steps * rows * e.knn * e.c_hidden[0]
        return {k: v * n_chunks for k, v in terms.items()}


RECIPE = ErwinRecipe()

"""Recipe "transolver": Transolver-3 on the dataset's native point features
(one shared backbone, volume and surface heads). Optimiser per dataset
config: AdamW with a per-epoch cosine (3-D) or Adam with OneCycle (2-D).
Full-field inference is the exact two-pass global slice aggregation over
the whole case in fixed chunks (padding rows masked out of the slice
statistics)."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..evaluation.flops import einsum_flops, sdpa_flops
from ..models.transolver import build_transolver_model
from ..training.losses import compute_loss
from ..training.schedules import build_onecycle, build_warmup_cosine
from .base import Recipe


class TransolverRecipe(Recipe):
    name = "transolver"
    view = "points"

    def build_model(self, cfg, spec) -> torch.nn.Module:
        return build_transolver_model(cfg.transolver, out_dim_vol=len(spec.vol_channels),
                                      in_dim=spec.native_point_dim,
                                      use_checkpoint=cfg.training.activation_checkpointing)

    def forward_loss(self, model, batch, cfg, spec):
        tc = cfg.training
        pv, ps = model(batch["x"].unsqueeze(0))
        weights = tc.loss_vol_weights if tc.loss_vol_weights is not None else spec.loss_vol_weights
        return compute_loss(pv, ps, batch["target_vol"].unsqueeze(0), batch["target_surf"].unsqueeze(0),
                            batch["is_volume"].unsqueeze(0), weights, tc.loss_reduction)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        t = cfg.transolver
        if t.schedule == "onecycle":
            optimizer = torch.optim.Adam(model.parameters(), lr=t.lr)
            total = cfg.training.epochs * opt_steps_per_epoch
            sched = build_onecycle(optimizer, total, t.lr, t.onecycle_pct_start,
                                   t.onecycle_div_factor, t.onecycle_final_div_factor)
            return optimizer, sched, "step"
        optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
        # the reference steps its cosine once per epoch, without warm-up
        sched = build_warmup_cosine(optimizer, cfg.training.epochs, t.lr, t.min_lr, 0.0)
        return optimizer, sched, "epoch"

    # ---- inference --------------------------------------------------------
    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> str:
        self._infer = {}
        if not (use_cuda and cfg.training.compile):
            return "eager (exact forward_chunks)"
        kw = dict(mode="reduce-overhead", fullgraph=True, dynamic=False)

        def heads(fx):
            h = model.ln_out(fx)
            return model.vol_head(h), model.surf_head(h)

        self._infer["compiled"] = (torch.compile(model.embed, **kw),
                                   [torch.compile(b.stats_masked, **kw) for b in model.blocks],
                                   [torch.compile(b.apply_chunk, **kw) for b in model.blocks],
                                   torch.compile(heads, **kw))
        return "compiled (masked fixed chunks; slice attention eager)"

    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        n_vol = int(case["n_vol"])
        x = case["x"].float()
        n = x.shape[0]
        chunk = int(cfg.transolver.infer_chunk)
        compiled = getattr(self, "_infer", {}).get("compiled")
        if compiled is None:
            chunks = [x[lo:lo + chunk].unsqueeze(0).to(device) for lo in range(0, n, chunk)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                outs = model.forward_chunks(chunks)
            pv = torch.cat([o[0].squeeze(0).float().cpu() for o in outs])
            ps = torch.cat([o[1].squeeze(0).float().cpu() for o in outs])
            return pv[:n_vol].numpy().astype(np.float32), ps[n_vol:].numpy().astype(np.float32)
        embed_c, stats_c, apply_c, heads_c = compiled
        n_chunks = max(1, math.ceil(n / chunk))
        pad = n_chunks * chunk - n
        x_pad = torch.cat([x, x.new_zeros(pad, x.shape[1])]) if pad else x
        xs = [x_pad[c * chunk:(c + 1) * chunk].unsqueeze(0).to(device) for c in range(n_chunks)]
        masks = []
        for c in range(n_chunks):
            m = torch.ones(1, chunk, device=device)
            if c == n_chunks - 1 and pad:
                m[:, chunk - pad:] = 0.0
            masks.append(m)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fx = [embed_c(xc).clone() for xc in xs]      # clones: graph-owned outputs
            for li, block in enumerate(model.blocks):
                num = den = None
                for c in range(n_chunks):
                    nk, dk = stats_c[li](fx[c], masks[c])
                    num = nk.clone() if num is None else num + nk
                    den = dk.clone() if den is None else den + dk
                out_slice = block.Attn.slice_attend(num / (den[..., None] + 1e-5))
                fx = [apply_c[li](fx[c], out_slice).clone() for c in range(n_chunks)]
            pv_parts, ps_parts = [], []
            for c in range(n_chunks):
                pv_c, ps_c = heads_c(fx[c])
                pv_parts.append(pv_c.squeeze(0).float().cpu())
                ps_parts.append(ps_c.squeeze(0).float().cpu())
        pv, ps = torch.cat(pv_parts)[:n], torch.cat(ps_parts)[:n]
        return pv[:n_vol].numpy().astype(np.float32), ps[n_vol:].numpy().astype(np.float32)

    # ---- FLOPs the hook meter cannot see ---------------------------------
    def manual_infer_flops(self, cfg, spec, n_vol, n_surf):
        t = cfg.transolver
        n = n_vol + n_surf
        H, G, C = t.n_heads, t.slice_num, t.n_hidden
        D = C // H
        # forward_chunks recomputes the slice weights in the deslice pass
        terms = {"slice_weights": 2.0 * (einsum_flops(n * H * G, C) + 3.0 * n * H * G),
                 "slice_agg": einsum_flops(H * G * C, n), "slice_num_proj": einsum_flops(H * G * D, C),
                 "slice_attention": sdpa_flops(G, G, H, D), "deslice_proj": einsum_flops(H * G * C, D),
                 "deslice": einsum_flops(n * C, H * G)}
        return {k: v * t.n_layers for k, v in terms.items()}


RECIPE = TransolverRecipe()

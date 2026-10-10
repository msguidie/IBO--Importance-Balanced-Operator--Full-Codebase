"""Recipe "spider": SpiderSolver on the dataset's native point features.
Adam with OneCycle per optimizer step. Full-field inference runs disjoint
density-preserving stride chunks, every point exactly once; onion shells,
sector ids and the innermost set are recomputed per chunk as they are per
training sample."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..evaluation.flops import sdpa_flops, segment_mean_flops
from ..models.spider import build_spider_model, onion_shell_ids
from ..training.losses import compute_loss_branch
from ..training.schedules import build_onecycle
from .base import Recipe


class SpiderRecipe(Recipe):
    name = "spider"
    view = "points"
    disable_ddp_optimizer = True       # dynamo's DDP splitter cannot trace the int split point

    def build_model(self, cfg, spec) -> torch.nn.Module:
        return build_spider_model(cfg.spider, out_dim_vol=len(spec.vol_channels),
                                  surf_token_mode=cfg.spider.surf_token_mode,
                                  in_dim=spec.native_point_dim,
                                  use_checkpoint=cfg.training.activation_checkpointing)

    def ensure_derived(self, cfg, dataset, names: list[str]) -> None:
        dataset.ensure_derived(cfg, "spider", names)

    def forward_loss(self, model, batch, cfg, spec):
        tc = cfg.training
        n_vol = batch["target_vol"].shape[0]
        pv, ps = model(batch["x"].unsqueeze(0), n_vol, batch["vol_gid"], batch["inner_idx"],
                       batch.get("surf_region_id"))
        weights = tc.loss_vol_weights if tc.loss_vol_weights is not None else spec.loss_vol_weights
        return compute_loss_branch(pv, ps, batch["target_vol"].unsqueeze(0),
                                   batch["target_surf"].unsqueeze(0), weights, tc.loss_reduction)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        s = cfg.spider
        optimizer = torch.optim.Adam(model.parameters(), lr=s.lr)
        total = cfg.training.epochs * opt_steps_per_epoch
        sched = build_onecycle(optimizer, total, s.lr, s.onecycle_pct_start, s.onecycle_div_factor,
                               s.onecycle_final_div_factor)
        return optimizer, sched, "step"

    # ---- inference --------------------------------------------------------
    def _quota_shapes(self, cfg) -> tuple[int, int]:
        """The one static inference shape: the training sample geometry."""
        return int(cfg.training.n_query) - int(cfg.spider.surf_quota), int(cfg.spider.surf_quota)

    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> str:
        self._infer = {}
        if not (use_cuda and cfg.training.compile):
            return "eager (natural-proportion stride chunks)"
        self._infer["compiled"] = torch.compile(model, mode="reduce-overhead", fullgraph=True,
                                                dynamic=False)
        return "compiled (fixed training-quota chunks, padded by cycling real rows)"

    @staticmethod
    def _pad_cycle(idx: torch.Tensor, target: int) -> torch.Tensor:
        n = int(idx.shape[0])
        if n == target:
            return idx
        return idx.repeat((target + n - 1) // n)[:target]

    def _aux(self, cfg, sp, vol_rows, surf_rows):
        """gid, inner and surface-region tensors of one chunk's rows."""
        s = cfg.spider
        wall = sp["wall"][vol_rows]
        shell = onion_shell_ids(wall, int(s.onion_num))
        gid = shell * int(s.n_clusters) + sp["vol_cluster"][vol_rows].long()
        n_inner = min(int(s.n_inner), int(vol_rows.shape[0]))
        inner = torch.topk(wall, n_inner, largest=False).indices.sort().values
        reg = None if sp.get("surf_region") is None else sp["surf_region"][surf_rows].long()
        return gid, inner, reg

    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        sp = case["spider"]
        n_vol, n_surf = int(case["n_vol"]), int(case["n_surf"])
        x_all = case["x"].float()
        n_out_vol, n_out_surf = model.vol_head.out_features, model.surf_head.out_features
        out_vol = np.empty((n_vol, n_out_vol), dtype=np.float32)
        out_surf = np.empty((n_surf, n_out_surf), dtype=np.float32)
        compiled = getattr(self, "_infer", {}).get("compiled")
        auto = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda)
        if compiled is not None:
            NV, NS = self._quota_shapes(cfg)
            n_chunks = max(1, math.ceil(n_vol / NV), math.ceil(max(n_surf, 1) / NS))
            for c in range(n_chunks):
                vol_rows, surf_rows = torch.arange(c, n_vol, n_chunks), torch.arange(c, n_surf, n_chunks)
                nv_real, ns_real = int(vol_rows.shape[0]), int(surf_rows.shape[0])
                vol_pad = self._pad_cycle(vol_rows, NV)
                surf_pad = self._pad_cycle(surf_rows, NS) if ns_real else torch.zeros(NS, dtype=torch.long)
                x = torch.cat([x_all[vol_pad], x_all[surf_pad + n_vol]])
                gid, inner, reg = self._aux(cfg, sp, vol_pad, surf_pad)
                with auto:
                    pv, ps = compiled(x.unsqueeze(0).to(device), NV, gid.to(device), inner.to(device),
                                      None if reg is None else reg.to(device))
                out_vol[vol_rows.numpy()] = pv.squeeze(0).float().cpu().numpy()[:nv_real]
                if ns_real:
                    out_surf[surf_rows.numpy()] = ps.squeeze(0).float().cpu().numpy()[:ns_real]
            return out_vol, out_surf
        n = n_vol + n_surf
        n_chunks = max(1, math.ceil(n / int(cfg.spider.infer_chunk)))
        for c in range(n_chunks):
            rows = torch.arange(c, n, n_chunks)
            vol_rows, surf_rows = rows[rows < n_vol], rows[rows >= n_vol] - n_vol
            x = torch.cat([x_all[vol_rows], x_all[surf_rows + n_vol]])
            gid, inner, reg = self._aux(cfg, sp, vol_rows, surf_rows)
            with auto:
                pv, ps = model(x.unsqueeze(0).to(device), int(vol_rows.shape[0]), gid.to(device),
                               inner.to(device), None if reg is None else reg.to(device))
            out_vol[vol_rows.numpy()] = pv.squeeze(0).float().cpu().numpy()
            out_surf[surf_rows.numpy()] = ps.squeeze(0).float().cpu().numpy()
        return out_vol, out_surf

    # ---- FLOPs the hook meter cannot see ---------------------------------
    def manual_infer_flops(self, cfg, spec, n_vol, n_surf):
        s = cfg.spider
        C, H = s.n_hidden, s.n_heads
        D = C // H
        G = s.onion_num * s.n_clusters
        n_surf_tokens = 1 if s.surf_token_mode == "single" else s.n_clusters
        n_chunks = max(1, math.ceil((n_vol + n_surf) / int(s.infer_chunk)))
        nv_c, ns_c = n_vol / n_chunks, n_surf / n_chunks
        coarse = min(s.n_inner, int(nv_c)) + ns_c
        terms = {"coarse_sdpa": sdpa_flops(coarse, coarse, H, D),
                 "fine_sdpa": sdpa_flops(G + n_surf_tokens, G + n_surf_tokens, H, D),
                 "sector_segment": segment_mean_flops(nv_c, C, G) + segment_mean_flops(ns_c, C, n_surf_tokens),
                 "t2p_matmul": 2.0 * (nv_c * G * C + ns_c * G * C + ns_c * n_surf_tokens * C)}
        return {k: v * s.n_layers * n_chunks for k, v in terms.items()}


RECIPE = SpiderRecipe()

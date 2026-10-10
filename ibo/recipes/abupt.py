"""Recipe "abupt": AB-UPT without the geometry branch, supervised on its
anchor points (Lion, 5% warm-up, cosine to an absolute floor). Full-field
inference draws fixed per-case anchors, encodes them once with their
keys / values cached, and decodes every remaining point in fixed-size
chunks; the reported metric averages infer_draws independent anchor draws."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..evaluation.flops import sdpa_flops
from ..models.abupt import build_abupt
from ..training.lion import Lion
from ..training.losses import compute_loss_branch
from ..training.schedules import build_warmup_cosine
from ..utils.seed import per_case_epoch_seed
from .base import Recipe, cpu_defloat_bf16

EVAL_DRAW_EPOCH = -1     # the evaluation draws' seed epoch; training epochs are >= 1


def _clone_cache(cache):
    """Deep-clone the anchor K/V pytree off a compiled graph's memory."""
    if isinstance(cache, torch.Tensor):
        return cache.clone()
    if isinstance(cache, dict):
        return {k: _clone_cache(v) for k, v in cache.items()}
    if isinstance(cache, (list, tuple)):
        out = [_clone_cache(v) for v in cache]
        return type(cache)(out) if isinstance(cache, tuple) else out
    return cache


def _pad(t: torch.Tensor, n: int) -> torch.Tensor:
    return t if t.shape[0] == n else torch.cat([t, t[-1:].expand(n - t.shape[0], *t.shape[1:])])


class AbuptRecipe(Recipe):
    name = "abupt"
    view = "abupt"

    def build_model(self, cfg, spec) -> torch.nn.Module:
        return build_abupt(cfg.abupt, spec)

    def forward_loss(self, model, batch, cfg, spec):
        tc = cfg.training
        pred_surf, pred_vol = model(batch["surface_position"], batch["volume_position"],
                                    batch.get("cond"))
        weights = tc.loss_vol_weights if tc.loss_vol_weights is not None else spec.loss_vol_weights
        return compute_loss_branch(pred_vol, pred_surf, batch["target_vol"], batch["target_surf"],
                                   weights, tc.loss_reduction)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        ac = cfg.abupt
        optimizer = Lion(model.parameters(), lr=ac.lr, weight_decay=ac.weight_decay)
        total = cfg.training.epochs * opt_steps_per_epoch
        sched = build_warmup_cosine(optimizer, total, ac.lr, ac.end_lr, ac.warmup_fraction)
        return optimizer, sched, "step"

    # ---- inference --------------------------------------------------------
    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> str:
        self._infer = {}
        if not (use_cuda and cfg.training.compile):
            return "eager"
        kw = dict(mode="reduce-overhead", fullgraph=True, dynamic=False)
        self._infer["encode"] = torch.compile(model.encode_anchor_cache, **kw)
        self._infer["decode"] = torch.compile(model.decode_queries_cached, **kw)
        return "compiled (anchors once, fixed padded query chunks)"

    def infer_draws(self, cfg) -> int:
        return int(cfg.abupt.infer_draws)

    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        ac = cfg.abupt
        if not use_cuda:
            case = cpu_defloat_bf16(case)
        n_vol, n_surf = int(case["n_vol"]), int(case["n_surf"])
        pos = case["pos"].float()
        cond = case.get("cond")
        cond = None if cond is None else cond.float().unsqueeze(0).to(device)
        nv_a, ns_a = min(int(ac.vol_anchors), n_vol), min(int(ac.surf_anchors), n_surf)
        draw = int(self.infer_draw)
        suffix = f"_d{draw}" if draw else ""

        def fixed(n, n_a, tag):
            g = torch.Generator()
            g.manual_seed(per_case_epoch_seed(f"{case['name']}__abupt_{tag}{suffix}", EVAL_DRAW_EPOCH))
            return torch.randperm(n, generator=g)[:n_a].sort().values

        def complement(n, idx):
            mask = torch.zeros(n, dtype=torch.bool)
            mask[idx] = True
            return torch.nonzero(~mask, as_tuple=False).squeeze(1)

        v_anchor, s_anchor = fixed(n_vol, nv_a, "vol"), fixed(n_surf, ns_a, "surf")
        v_query, s_query = complement(n_vol, v_anchor), complement(n_surf, s_anchor)
        pos_v, pos_s = pos[:n_vol], pos[n_vol:]
        out_vol = np.empty((n_vol, model.volume_decoder.out_features), np.float32)
        out_surf = np.empty((n_surf, model.surface_decoder.out_features), np.float32)

        infer = getattr(self, "_infer", {})
        encode = infer.get("encode") or model.encode_anchor_cache
        decode = infer.get("decode") or model.decode_queries_cached
        auto = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda)
        with auto:
            ps_a, pv_a, cache = encode(pos_s[s_anchor].unsqueeze(0).to(device),
                                       pos_v[v_anchor].unsqueeze(0).to(device), cond)
        if infer.get("encode") is not None:
            cache = _clone_cache(cache)
        out_surf[s_anchor.numpy()] = ps_a[0].float().cpu().numpy()
        out_vol[v_anchor.numpy()] = pv_a[0].float().cpu().numpy()

        chunk = int(ac.infer_chunk)
        n_sq, n_vq = int(s_query.shape[0]), int(v_query.shape[0])
        n_steps = max(math.ceil(n_sq / chunk), math.ceil(n_vq / chunk))
        s_pad_src = pos_s[s_query[-1:] if n_sq else s_anchor[-1:]]
        v_pad_src = pos_v[v_query[-1:] if n_vq else v_anchor[-1:]]
        for step in range(n_steps):
            lo = step * chunk
            s_valid = max(min(lo + chunk, n_sq) - lo, 0)
            v_valid = max(min(lo + chunk, n_vq) - lo, 0)
            s_rows = _pad(pos_s[s_query[lo:lo + s_valid]], chunk) if s_valid else s_pad_src.expand(chunk, -1)
            v_rows = _pad(pos_v[v_query[lo:lo + v_valid]], chunk) if v_valid else v_pad_src.expand(chunk, -1)
            with auto:
                ps_q, pv_q = decode(cache, s_rows.unsqueeze(0).to(device),
                                    v_rows.unsqueeze(0).to(device), cond)
            if s_valid:
                out_surf[s_query[lo:lo + s_valid].numpy()] = ps_q[0, :s_valid].float().cpu().numpy()
            if v_valid:
                out_vol[v_query[lo:lo + v_valid].numpy()] = pv_q[0, :v_valid].float().cpu().numpy()
        return out_vol, out_surf

    # ---- FLOPs the hook meter cannot see ---------------------------------
    def _trunk_terms(self, ac, ns_a, nv_a):
        H, hd = ac.num_heads, ac.hidden_dim // ac.num_heads
        t = 0.0
        for ch in ac.shared_blocks:
            if ch == "s":
                t += sdpa_flops(ns_a, ns_a, H, hd) + sdpa_flops(nv_a, nv_a, H, hd)
            else:
                t += sdpa_flops(ns_a, nv_a, H, hd) + sdpa_flops(nv_a, ns_a, H, hd)
        return t

    def manual_infer_flops(self, cfg, spec, n_vol, n_surf):
        ac = cfg.abupt
        H, hd = ac.num_heads, ac.hidden_dim // ac.num_heads
        nv_a, ns_a = min(ac.vol_anchors, n_vol), min(ac.surf_anchors, n_surf)
        chunk = ac.infer_chunk
        n_steps = max(math.ceil(max(n_surf - ns_a, 0) / chunk), math.ceil(max(n_vol - nv_a, 0) / chunk))
        q_rows = n_steps * chunk * 2
        n_blocks = len(ac.shared_blocks) + ac.num_surface_blocks
        anchors = self._trunk_terms(ac, ns_a, nv_a) + ac.num_surface_blocks * sdpa_flops(ns_a, ns_a, H, hd) \
            + ac.num_volume_blocks * sdpa_flops(nv_a, nv_a, H, hd)
        return {"anchor_pass_sdpa": anchors,
                "query_sdpa": n_blocks * sdpa_flops(q_rows, (ns_a + nv_a) / 2, H, hd),
                "rope_trig": 2.0 * (ns_a + nv_a + q_rows) * hd * n_blocks}


RECIPE = AbuptRecipe()

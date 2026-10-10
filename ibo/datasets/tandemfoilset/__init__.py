"""TandemFoilSet: 2-D RANS over tandem airfoil pairs, five subsets
(cruise_aoa0, cruise_aoa5, cruise_random, takeoff, racecar), each its own
benchmark with an 8:1:1 case split (datasets/tandemfoilset/splits.json).
Raw data are the dataset's Part pickles; every subset gets its own cache,
statistics and calibration. Fields (u, v, p) on the mesh nodes; the
surface target is p on the wall nodes. A run addresses one subset
(--subset) or all five in turn.
"""
from __future__ import annotations

import json

import numpy as np

from ...config import Config
from ...training.data import collate_single, collate_tree
from ..base import Dataset, DatasetSpec, FeatureGroup, check_ibo_layout
from .cases import (INNER_FRACTION, UNIFIED_REF, TandemTrainSet, cache_dir, collate_gaot,
                    eval_case, load_coef_norm, load_meta, unified_grid)
from .gaot_derive import GAOT_IN_CHANNELS, bake_gaot, load_gaot_meta
from .metrics import case_metrics
from .spider_derive import build_spider_derived, load_spider_meta

SPEC = DatasetSpec(
    name="tandemfoilset", dim=2,
    vol_channels=["u", "v", "p"], surf_channels=["p"],
    cond_dim=2,                   # (U, angle of attack), z-scored
    slot_feat_dim=18,             # rel pos 2, two signed distances + gradients 6, surface flag, did 8, curvature
    stats_dim=16,                 # token: two signed distances + gradients 6, position 2, did 8
    query_groups=[
        FeatureGroup("pos_sdf", 6, "fourier", n_freqs=8),   # position, gradient x |distance| per airfoil
        FeatureGroup("did", 8, "plain"),
        FeatureGroup("curvature", 1, "plain"),
    ],
    pair_fourier_freqs=(8, 8), pair_unit_dir=False, pair_log_dbar=False,
    native_point_dim=7 + UNIFIED_REF * UNIFIED_REF,   # AirfRANS-style 7 + distances to an 8 x 8 grid
    loss_vol_weights=None,
    metric_names=["vol_mse", "surf_mse"],
)


class TandemFoilSet(Dataset):
    spec = SPEC

    def __init__(self):
        self._coef = {}

    # ---- data on disk -----------------------------------------------------
    def cache_dir(self, cfg: Config):
        return cache_dir(cfg)

    def splits(self, cfg: Config) -> dict[str, list[str]]:
        with open(self.splits_path(), encoding="utf-8") as f:
            d = json.load(f)["subsets"]
        if cfg.run.subset not in d:
            raise SystemExit(f"unknown subset {cfg.run.subset!r}; subsets: {', '.join(d)}")
        return {k: list(d[cfg.run.subset][k]) for k in ("train", "val", "test")}

    def preprocess(self, cfg: Config, from_cache: bool = False) -> None:
        from .preprocess import run_preprocess
        run_preprocess(cfg, from_cache)

    def calibration(self, cfg: Config) -> dict:
        """Per-subset values adopted at launch: IBO's RoPE domain, Fourier
        scale and slots per token; AB-UPT's coordinate affine; SpiderSolver's
        density-matched surface quota and innermost set."""
        cache = cache_dir(cfg)
        meta = load_meta(cache)
        m = cfg.run.model
        if m == "ibo":
            path = cache / "calibration_report.json"
            if not path.exists():
                raise SystemExit(f"{path} missing: rerun the preprocessing of this subset")
            with open(path, encoding="utf-8") as f:
                rep = json.load(f)
            r, d = rep["rope"], rep["decoder_fourier"]
            return {"ibo.rope_scales": tuple(r["suggested_rope_scales"]),
                    "ibo.rope_base": tuple(r["suggested_rope_base"]),
                    "ibo.rope_domain_lo": tuple(r["suggested_rope_domain_lo"]),
                    "ibo.rope_domain_hi": tuple(r["suggested_rope_domain_hi"]),
                    "ibo.fourier_sdf_scale": float(d["suggested_fourier_sdf_scale"]),
                    "ibo.n_slots": int(meta["pn_k"])}
        if m == "abupt":
            lo = np.asarray(meta["rope_domain_lo"], dtype=np.float64)
            hi = np.asarray(meta["rope_domain_hi"], dtype=np.float64)
            # domain corner -> 0, longest extent -> 1000 (AB-UPT's coordinate range)
            return {"abupt.pos_offset": tuple((-lo).tolist()),
                    "abupt.pos_scale": float(1000.0 / (hi - lo).max())}
        if m == "spider":
            sm = load_spider_meta(cache)
            n_query = int(cfg.training.n_query)
            quota = max(8, int(round(n_query * float(sm["surf_ratio_median"]))))
            return {"spider.surf_quota": quota,
                    "spider.n_inner": max(4, int(round(INNER_FRACTION * (n_query - quota)))),
                    "spider.n_clusters": 2 * int(sm["clusters_per_body"])}
        return {}

    def derive(self, cfg: Config, model: str, names: list[str]) -> None:
        cache = cache_dir(cfg)
        workers = int(cfg.tandemfoilset.n_workers)
        if model == "spider":
            build_spider_derived(cache, self.splits(cfg)["train"], names,
                                 clusters_per_body=max(1, int(cfg.spider.n_clusters) // 2),
                                 workers=cfg.spider.derive_workers or workers)
        elif model == "gaot":
            g = cfg.gaot
            bake_gaot(cache, names, int(g.grid_h), int(g.grid_w), float(g.radius_factor), workers)
        else:
            raise SystemExit(f"tandemfoilset: no derived preprocessing for {model}")

    def ensure_derived(self, cfg: Config, model: str, names: list[str]) -> None:
        if model == "spider":
            load_spider_meta(cache_dir(cfg))
        elif model == "gaot":
            self.gaot_meta(cfg)

    def gaot_meta(self, cfg: Config) -> dict:
        meta = load_gaot_meta(cache_dir(cfg))
        g = cfg.gaot
        if (int(meta["grid_h"]), int(meta["grid_w"])) != (int(g.grid_h), int(g.grid_w)):
            raise SystemExit(f"baked GAOT grid {meta['grid_h']}x{meta['grid_w']} != configured "
                             f"{g.grid_h}x{g.grid_w}: rebuild with `python -m ibo derive`")
        return {"grid": meta["grid"], "in_channels": GAOT_IN_CHANNELS}

    # ---- training ---------------------------------------------------------
    def build_train_dataset(self, cfg: Config, names: list[str], view: str, seed_salt: str):
        mode = {"ibo": "ibo", "abupt": "abupt", "erwin": "erwin", "gaot": "gaot",
                "points": "spider" if cfg.run.model == "spider" else "transolver"}[view]
        if mode == "ibo":
            check_ibo_layout(cfg, cfg.tandemfoilset.n_leaves, cfg.tandemfoilset.idw_k)
        collate = collate_tree if mode in ("ibo", "abupt") else \
            collate_gaot if mode == "gaot" else collate_single
        return TandemTrainSet(cfg, names, mode, seed_salt), collate, "ram"

    # ---- evaluation -------------------------------------------------------
    def stream_eval_cases(self, cfg: Config, names: list[str], view: str):
        cache = cache_dir(cfg)
        coef, meta = self._coef_norm(cfg), load_meta(cache)
        if view == "ibo":
            check_ibo_layout(cfg, cfg.tandemfoilset.n_leaves, cfg.tandemfoilset.idw_k)
        grid = unified_grid(meta) if view == "points" else None
        for name in names:
            yield name, eval_case(cfg, cache, name, view, coef, meta, grid)

    def _coef_norm(self, cfg: Config) -> dict:
        key = str(cache_dir(cfg))
        if key not in self._coef:
            self._coef[key] = load_coef_norm(cache_dir(cfg))
        return self._coef[key]

    def case_metrics(self, cfg: Config, case: dict, pred_vol_z, pred_surf_z) -> dict:
        return case_metrics(pred_vol_z, pred_surf_z, case["target_vol"].numpy(),
                            case["target_surf"].numpy(), self._coef_norm(cfg), case["meta"]["umag"])

    def aggregate(self, cfg: Config, rows: list[dict]) -> dict:
        """Reporting-space MSE (volume: u, v, p over volume nodes; surface: p
        over wall nodes), mean over the test cases of the subset."""
        return {k: float(np.mean([r[k] for r in rows])) for k in (*SPEC.metric_names, "zscore_mse")}


DATASET = TandemFoilSet()

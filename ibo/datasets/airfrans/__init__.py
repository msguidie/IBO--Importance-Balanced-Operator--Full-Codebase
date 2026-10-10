"""AirfRANS: 2-D RANS over single airfoils (Bonnet et al., 2022), the
official "full" task: 800 training-pool cases, of which 80 become the
validation split (fixed seed), and the 200 official test cases. Raw data
are the case folders of the official release plus its manifest.json.
Fields (u_x, u_y, p, nu_t) on the mesh nodes; the surface target is p on
the airfoil nodes; lift and drag are integrated with the dataset's own
procedure.
"""
from __future__ import annotations

import json

import numpy as np

from ...config import Config
from ...training.data import collate_tree
from ..base import Dataset, DatasetSpec, FeatureGroup, check_ibo_layout
from .cases import AirfransTrainSet, eval_case, load_norm_stats, paths
from .metrics import case_metrics

SPEC = DatasetSpec(
    name="airfrans", dim=2,
    vol_channels=["u_x", "u_y", "p", "nut"], surf_channels=["p"],
    cond_dim=2,                   # inflow (U cos a, U sin a), z-scored
    slot_feat_dim=15,             # rel pos 2, signed distance + gradient 3, surface flag, did 8, curvature
    stats_dim=13,                 # token: position 2, signed distance + gradient 3, did 8
    query_groups=[
        FeatureGroup("pos_sdf", 4, "fourier", n_freqs=8),   # position, direction x |distance|
        FeatureGroup("did", 8, "plain"),
        FeatureGroup("curvature", 1, "plain"),
    ],
    pair_fourier_freqs=(8, 8), pair_unit_dir=False, pair_log_dbar=False,
    native_point_dim=0,
    loss_vol_weights=None,
    metric_names=["volume_mse", "surface_mse", "cl_rel_err", "cl_spearman"],
)


class Airfrans(Dataset):
    spec = SPEC

    def __init__(self):
        self._stats = {}

    # ---- data on disk -----------------------------------------------------
    def cache_dir(self, cfg: Config):
        return paths(cfg)["cache"]

    def splits(self, cfg: Config) -> dict[str, list[str]]:
        """The split written by preprocessing (720 / 80 / 200), or derived
        from the official manifest when no cache exists yet."""
        p = paths(cfg)
        if p["manifest"].exists():
            with open(p["manifest"], encoding="utf-8") as f:
                d = json.load(f)
            d = d.get("splits", d)          # also the layout of earlier caches
        else:
            from .preprocess import build_manifest
            d = build_manifest(cfg.data.data_root, int(cfg.airfrans.val_seed), int(cfg.airfrans.n_val))
        return {k: list(d[k]) for k in ("train", "val", "test")}

    def preprocess(self, cfg: Config, from_cache: bool = False) -> None:
        from .preprocess import run_preprocess
        run_preprocess(cfg, from_cache)

    def calibration(self, cfg: Config) -> dict:
        """IBO's RoPE domain and Fourier scale from the calibration report."""
        if cfg.run.model != "ibo":
            return {}
        path = paths(cfg)["calibration"]
        if not path.exists():
            raise SystemExit(f"{path} missing: run `python -m ibo preprocess --dataset airfrans` first")
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
        r, d = rep["rope"], rep["decoder_fourier"]
        return {"ibo.rope_scales": tuple(r["suggested_rope_scales"]),
                "ibo.rope_base": tuple(r["suggested_rope_base"]),
                "ibo.rope_domain_lo": tuple(r["suggested_rope_domain_lo"]),
                "ibo.rope_domain_hi": tuple(r["suggested_rope_domain_hi"]),
                "ibo.fourier_sdf_scale": float(d["suggested_fourier_sdf_scale"])}

    # ---- training ---------------------------------------------------------
    def build_train_dataset(self, cfg: Config, names: list[str], view: str, seed_salt: str):
        if view not in ("ibo", "abupt"):
            raise ValueError(f"airfrans: unknown view {view!r}")
        if view == "ibo":
            self._check(cfg)
        return AirfransTrainSet(cfg, names, view, seed_salt), collate_tree, "ram"

    @staticmethod
    def _check(cfg: Config) -> None:
        pp = cfg.airfrans
        check_ibo_layout(cfg, pp.n_leaves, pp.idw_k, pp.slots)

    # ---- evaluation -------------------------------------------------------
    def stream_eval_cases(self, cfg: Config, names: list[str], view: str):
        p = paths(cfg)
        if view == "ibo":
            self._check(cfg)
        for name in names:
            yield name, eval_case(cfg, p, name, view)

    def _norm_stats(self, cfg: Config) -> dict:
        key = str(paths(cfg)["norm_stats"])
        if key not in self._stats:
            self._stats[key] = load_norm_stats(key)
        return self._stats[key]

    def case_metrics(self, cfg: Config, case: dict, pred_vol_z, pred_surf_z) -> dict:
        m = case["meta"]
        return case_metrics(cfg.data.data_root, case["name"], pred_vol_z, pred_surf_z,
                            case["target_vol"].numpy(), case["target_surf"].numpy(),
                            m["perm"].numpy(), self._norm_stats(cfg), m["Uinf"], m["alpha"])

    def aggregate(self, cfg: Config, rows: list[dict]) -> dict:
        """Mean z-scored MSE, mean relative error of the lift coefficient and
        its Spearman correlation over the test cases (also for drag)."""
        from scipy.stats import spearmanr
        out = {k: float(np.mean([r[k] for r in rows]))
               for k in ("volume_mse", "surface_mse", "cl_rel_err", "cd_rel_err", "zscore_mse")}
        for c in ("cl", "cd"):
            true = np.array([r[f"{c}_true"] for r in rows])
            pred = np.array([r[f"{c}_pred"] for r in rows])
            out[f"{c}_spearman"] = float(spearmanr(true, pred)[0]) if len(rows) > 1 else float("nan")
        return out


DATASET = Airfrans()

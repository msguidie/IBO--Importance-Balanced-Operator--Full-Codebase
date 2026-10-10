"""DrivAerML: time-averaged hybrid RANS-LES over 500 DrivAer morphs at
fixed inflow (Ashton et al., 2024); 484 cases with simulations, split
400 / 34 / 50 as in AB-UPT (datasets/drivaerml/splits.json). Raw data are
the pre-converted `run_<id>.pt` cases (field list in the README). Volume
targets (Ux, Uy, Uz, p), surface targets (p, tau_x, tau_y, tau_z); the
evaluation pool of a held-out case is a uniform volume sample plus every
surface point, so drag and lift are exact surface integrals.
"""
from __future__ import annotations

import numpy as np

from ...config import Config
from ...training.data import collate_tree
from ..base import Dataset, DatasetSpec, FeatureGroup, check_ibo_layout
from .cases import VOL_CACHE, DrivaermlTrainSet, eval_case, load_norm_stats, paths
from .metrics import case_metrics, r_squared

SPEC = DatasetSpec(
    name="drivaerml", dim=3,
    vol_channels=["Ux", "Uy", "Uz", "p"], surf_channels=["p_surf", "wss_x", "wss_y", "wss_z"],
    cond_dim=0,
    slot_feat_dim=8,              # rel centroid 3, signed distance, gradient 3, surface fraction
    stats_dim=7,                  # token: position 3, signed distance, gradient 3
    query_groups=[
        FeatureGroup("sdf", 1, "fourier", n_freqs=5),     # signed z-scored wall distance
        FeatureGroup("dir", 3, "plain"),                  # wall direction / outward normal
        FeatureGroup("pos", 3, "fourier", n_freqs=4),     # affine position
    ],
    pair_fourier_freqs=(6, 6, 6), pair_unit_dir=True, pair_log_dbar=True,
    native_point_dim=0,
    loss_vol_weights=None,
    metric_names=["p_surf_pct", "tau_pct", "p_pct", "U_pct", "cd_r2", "cl_r2"],
)
VOL_IDX = [VOL_CACHE.index(c) for c in SPEC.vol_channels]   # columns of the 5-channel cache target


class Drivaerml(Dataset):
    spec = SPEC

    def __init__(self):
        self._stats = {}

    # ---- data on disk -----------------------------------------------------
    def cache_dir(self, cfg: Config):
        return paths(cfg)["cache"]

    def preprocess(self, cfg: Config, from_cache: bool = False) -> None:
        from .preprocess import run_preprocess
        run_preprocess(cfg, self.splits(cfg))

    def calibration(self, cfg: Config) -> dict:
        """Layout grid: the rotary domain is the lattice extent, scales
        follow round(32/6 x span) and the bases keep each axis's coarsest
        cycle coverage: base' = base (scale'/scale)^(dim/(dim-2))."""
        if cfg.run.model != "ibo" or cfg.run.layout != "grid":
            return {}
        pp, ib = cfg.drivaerml, cfg.ibo
        s = float(pp.coord_scale)
        lo, hi, scales, bases = [], [], [], []
        for d in range(3):
            step = (pp.crop_max[d] - pp.crop_min[d]) / s / pp.grid_dims[d]
            c_lo, c_hi = pp.crop_min[d] / s + 0.5 * step, pp.crop_max[d] / s - 0.5 * step
            sc = float(round(32.0 / 6.0 * (c_hi - c_lo) * s))
            lo.append(c_lo)
            hi.append(c_hi)
            scales.append(sc)
            bases.append(float(ib.rope_base[d] * (sc / ib.rope_scales[d]) ** (ib.rope_dims[d] / (ib.rope_dims[d] - 2))))
        return {"ibo.rope_domain_lo": tuple(lo), "ibo.rope_domain_hi": tuple(hi),
                "ibo.rope_scales": tuple(scales), "ibo.rope_base": tuple(bases)}

    # ---- training ---------------------------------------------------------
    def build_train_dataset(self, cfg: Config, names: list[str], view: str, seed_salt: str):
        if view != "ibo":
            raise ValueError(f"drivaerml: unknown view {view!r}")
        self._check(cfg)
        return DrivaermlTrainSet(cfg, names, VOL_IDX, seed_salt), collate_tree, "ram"

    @staticmethod
    def _check(cfg: Config) -> None:
        pp = cfg.drivaerml
        check_ibo_layout(cfg, pp.n_leaves, pp.idw_k, pp.slots)

    # ---- evaluation -------------------------------------------------------
    def _norm_stats(self, cfg: Config) -> dict:
        key = str(paths(cfg)["norm_stats"])
        if key not in self._stats:
            self._stats[key] = load_norm_stats(key)
        return self._stats[key]

    def stream_eval_cases(self, cfg: Config, names: list[str], view: str):
        p, stats = paths(cfg), self._norm_stats(cfg)
        self._check(cfg)
        for name in names:
            yield name, eval_case(cfg, p, name, VOL_IDX, stats)

    def case_metrics(self, cfg: Config, case: dict, pred_vol_z, pred_surf_z) -> dict:
        return case_metrics(pred_vol_z, pred_surf_z, case["target_vol"].numpy(), case["target_surf"].numpy(),
                            self._norm_stats(cfg), SPEC.vol_channels, VOL_IDX, case["meta"])

    def aggregate(self, cfg: Config, rows: list[dict]) -> dict:
        """Relative L2 errors in percent, mean over the test cases, and the
        R^2 of the drag and lift coefficients across cases."""
        out = {f"{k}_pct": 100.0 * float(np.mean([r[k] for r in rows])) for k in ("p_surf", "tau", "p", "U")}
        for c in ("cd", "cl"):
            out[f"{c}_r2"] = r_squared([r[f"{c}_pred"] for r in rows], [r[f"{c}_true"] for r in rows])
        out["zscore_mse"] = float(np.mean([r["zscore_mse"] for r in rows]))
        return out


DATASET = Drivaerml()

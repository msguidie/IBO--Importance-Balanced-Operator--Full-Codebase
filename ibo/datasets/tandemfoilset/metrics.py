"""TandemFoilSet metrics of one case.

Training space: z-scored nondimensional fields (y / [U, U, U^2], then the
train-split z-score). Reporting space of the paper's tables: the physical
fields re-normalised with the train-split z-score of the RAW fields, so
every subset and every model share one unit. Volume MSE averages the
(u, v, p) errors over the volume nodes; surface MSE is the p error over the
wall nodes (u = v = 0 there by the no-slip condition).
"""
from __future__ import annotations

import numpy as np

P = 2
_EPS = 1e-8


def _std(x) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=np.float64), _EPS)


def case_metrics(pred_vol_z, pred_surf_z, true_vol_z, true_surf_z, coef: dict, umag: float) -> dict:
    pv = np.asarray(pred_vol_z, dtype=np.float64)
    ps = np.asarray(pred_surf_z, dtype=np.float64)[:, 0]
    tv = np.asarray(true_vol_z, dtype=np.float64)
    ts = np.asarray(true_surf_z, dtype=np.float64)[:, 0]
    # training space
    vol_mse = ((pv - tv) ** 2).mean(axis=0)
    surf_mse = float(((ps - ts) ** 2).mean()) if ps.size else float("nan")
    # physical -> reporting space
    mean, std = np.asarray(coef["mean_out"], dtype=np.float64), _std(coef["std_out"])
    mean_p, std_p = np.asarray(coef["mean_out_protocol"], dtype=np.float64), _std(coef["std_out_protocol"])
    scale = np.array([umag, umag, umag * umag], dtype=np.float64)

    def to_protocol(z, cols):
        phys = (z * std[cols] + mean[cols]) * scale[cols]
        return (phys - mean_p[cols]) / std_p[cols]

    cols = np.arange(3)
    d_vol = to_protocol(pv, cols) - to_protocol(tv, cols)
    d_surf = to_protocol(ps, P) - to_protocol(ts, P)
    return {"vol_mse": float((d_vol ** 2).mean()),
            "surf_mse": float((d_surf ** 2).mean()) if d_surf.size else float("nan"),
            "zscore_mse": float(vol_mse.mean()) + surf_mse,
            "zscore_vol": float(vol_mse.mean()), "zscore_surf": surf_mse,
            "zscore_u": float(vol_mse[0]), "zscore_v": float(vol_mse[1]), "zscore_p": float(vol_mse[2])}

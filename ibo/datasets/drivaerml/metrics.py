"""DrivAerML metrics of one case: relative L2 errors in physical units
(surface pressure; wall shear stress as one vector field; volume pressure;
velocity as one vector field) over the case's evaluation pool, and the
drag and lift coefficients integrated over the surface rows with their
face areas (Horvitz-Thompson weight n_surface / n_sampled when the pool
is a sample; 1 when it holds every surface point)."""
from __future__ import annotations

import numpy as np

from .raw import integrate_cdcl


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """||pred - true|| / ||true|| over all entries of a [N, C] block."""
    return float(np.linalg.norm(pred - true) / max(float(np.linalg.norm(true)), 1e-12))


def case_metrics(pred_vol_z, pred_surf_z, true_vol_z, true_surf_z, stats: dict, vol_channels,
                 vol_idx: list[int], meta: dict) -> dict:
    pv = np.asarray(pred_vol_z, dtype=np.float64)
    ps = np.asarray(pred_surf_z, dtype=np.float64)
    tv = np.asarray(true_vol_z, dtype=np.float64)
    ts = np.asarray(true_surf_z, dtype=np.float64)
    vol_mse = ((pv - tv) ** 2).mean(axis=0)
    surf_mse = ((ps - ts) ** 2).mean(axis=0)
    vm = np.asarray(stats["vol_mean"], dtype=np.float64)[vol_idx]
    vs = np.asarray(stats["vol_std"], dtype=np.float64)[vol_idx]
    sm = np.asarray(stats["surf_mean"], dtype=np.float64)
    ss = np.asarray(stats["surf_std"], dtype=np.float64)
    pv_phys, tv_phys = pv * vs + vm, tv * vs + vm
    ps_phys, ts_phys = ps * ss + sm, ts * ss + sm
    u = [vol_channels.index(c) for c in ("Ux", "Uy", "Uz")]
    ip = vol_channels.index("p")
    out = {"p_surf": _rel_l2(ps_phys[:, 0:1], ts_phys[:, 0:1]),
           "tau": _rel_l2(ps_phys[:, 1:4], ts_phys[:, 1:4]),
           "p": _rel_l2(pv_phys[:, ip:ip + 1], tv_phys[:, ip:ip + 1]),
           "U": _rel_l2(pv_phys[:, u], tv_phys[:, u]),
           "zscore_mse": float(np.concatenate([vol_mse, surf_mse]).mean())}
    normal = meta["surface_normal"].numpy()
    area = meta["surface_area"].numpy()
    w = float(meta["n_surf_raw"]) / max(len(area), 1)
    c = integrate_cdcl(ps_phys[:, 0], ps_phys[:, 1:4], normal, area, meta["a_ref"], weight=w)
    out.update(cd_pred=c["cd"], cl_pred=c["cl"], cd_true=float(meta["cd_true"]),
               cl_true=float(meta["cl_true"]))
    return out


def r_squared(pred, true) -> float:
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)

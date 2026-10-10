"""AirfRANS metrics of one case: z-scored field MSE (volume: the four
channels over volume nodes; surface: p over wall nodes) and the lift and
drag coefficients integrated with the dataset's own pyvista procedure
(wall shear stress from the velocity gradient with nu = 1.56e-5, pressure
and friction forces summed over the airfoil cells, rotated into the wind
frame, 2 F / U^2) on both the predicted and the shipped fields.

Compute_coefficients, WallShearStress and reorganize follow the AirfRANS
reference implementation (Bonnet et al., 2022; MIT licence).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NU = np.array(1.56e-5)
P = 2


def reorganize(in_order_points, out_order_points, quantity):
    """Rows of `quantity` (ordered like in_order_points) re-ordered like
    out_order_points, matched by exact coordinate equality."""
    n = out_order_points.shape[0]
    idx = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cond = out_order_points[i] == in_order_points
        idx[i] = np.argwhere(cond[:, 0] * cond[:, 1])[0][0]
    assert (in_order_points[idx] == out_order_points).all()
    return quantity[idx]


def WallShearStress(Jacob_U, normals):
    S = .5 * (Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    ShearStress = 2 * NU.reshape(-1, 1, 1) * S
    return (ShearStress * normals[:, :2].reshape(-1, 1, 2)).sum(axis=2)


def Compute_coefficients(internal, aerofoil, bool_surf, Uinf, angle):
    """(Cd, Cl) of the fields held by the internal mesh."""
    bool_surf = np.asarray(bool_surf, dtype=bool)
    intern = internal.copy()
    aero = aerofoil.copy()
    point_mesh = intern.points[bool_surf, :2]
    point_surf = aero.points[:, :2]
    intern = intern.compute_derivative(scalars="U", gradient="pred_grad")
    surf_grad = intern.point_data["pred_grad"].reshape(-1, 3, 3)[bool_surf, :2, :2]
    surf_p = intern.point_data["p"][bool_surf]
    surf_grad = reorganize(point_mesh, point_surf, surf_grad)
    surf_p = reorganize(point_mesh, point_surf, surf_p)
    aero.point_data["wallShearStress"] = WallShearStress(surf_grad, -aero.point_data["Normals"])
    aero.point_data["p"] = surf_p
    aero = aero.ptc(pass_point_data=True)
    WP_int = -aero.cell_data["p"][:, None] * aero.cell_data["Normals"][:, :2]
    Wss_int = (aero.cell_data["wallShearStress"] * aero.cell_data["Length"].reshape(-1, 1)).sum(axis=0)
    WP_int = (WP_int * aero.cell_data["Length"].reshape(-1, 1)).sum(axis=0)
    force = Wss_int - WP_int
    alpha = angle * np.pi / 180
    basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    coef = 2 * (basis @ force) / Uinf ** 2
    return float(coef[0]), float(coef[1])


def case_metrics(raw_dir: Path, name: str, pred_vol_z, pred_surf_z, true_vol_z, true_surf_z,
                 perm, stats: dict, Uinf: float, alpha: float) -> dict:
    import pyvista as pv

    pv_ = np.asarray(pred_vol_z, dtype=np.float64)
    ps = np.asarray(pred_surf_z, dtype=np.float64)[:, 0]
    tv = np.asarray(true_vol_z, dtype=np.float64)
    ts = np.asarray(true_surf_z, dtype=np.float64)[:, 0]
    vol_mse = ((pv_ - tv) ** 2).mean(axis=0)
    surf_mse = float(((ps - ts) ** 2).mean())
    out = {"volume_mse": float(vol_mse.mean()), "surface_mse": surf_mse,
           "zscore_mse": float(vol_mse.mean()) + surf_mse}

    # physical field in mesh order: volume rows all channels, wall rows
    # [0, 0, p, 0] (no-slip and the wall boundary condition of nu_t)
    mean, std = np.asarray(stats["target_mean"], dtype=np.float64), np.asarray(stats["target_std"], dtype=np.float64)
    n_vol = pv_.shape[0]
    phys_vf = np.zeros((n_vol + ps.shape[0], 4), dtype=np.float64)
    phys_vf[:n_vol] = pv_ * std + mean
    phys_vf[n_vol:, P] = ps * std[P] + mean[P]
    perm = np.asarray(perm)
    phys = np.empty_like(phys_vf)
    phys[perm] = phys_vf
    bool_surf = np.zeros(phys.shape[0], dtype=bool)
    bool_surf[perm[n_vol:]] = True

    case_dir = Path(raw_dir) / name
    internal = pv.read(str(case_dir / f"{name}_internal.vtu"))
    aerofoil = pv.read(str(case_dir / f"{name}_aerofoil.vtp"))
    cd_true, cl_true = Compute_coefficients(internal, aerofoil, bool_surf, Uinf, alpha)
    pred = internal.copy()
    pred.point_data["U"][:, :2] = phys[:, :2]
    pred.point_data["p"] = phys[:, 2]
    pred.point_data["nut"] = phys[:, 3]
    cd_pred, cl_pred = Compute_coefficients(pred, aerofoil, bool_surf, Uinf, alpha)
    out.update(cd_pred=cd_pred, cd_true=cd_true, cl_pred=cl_pred, cl_true=cl_true,
               cd_rel_err=abs(cd_pred - cd_true) / (abs(cd_true) + 1e-8),
               cl_rel_err=abs(cl_pred - cl_true) / (abs(cl_true) + 1e-8))
    return out

"""DrivAerML raw cases and vehicle geometry.

A raw case is one `run_<id>.pt` dictionary (see the README for the field
list): volume coordinates, velocity, pressure and nu_t; surface
coordinates, pressure, wall shear stress, normals and face areas; the STL
vertices and faces; and the reference block with aRef, Cd and Cl.

Distances to the vehicle use Open3D's ray-casting scene over the STL.
Volume nodes lie outside the body, so their unsigned distance is the
signed distance; group centroids may fall inside and receive signed
distance with the gradient flipped. Surface normals are oriented outward
once per case by integrating the reference drag and lift.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

U_INF = 38.889      # m/s, the dataset's inflow speed (Cd/Cl reference)


@dataclass
class RawCase:
    name: str
    volume_pos: np.ndarray
    volume_velocity: np.ndarray
    volume_pressure: np.ndarray
    volume_nut: np.ndarray
    surface_pos: np.ndarray
    surface_pressure: np.ndarray
    surface_wss: np.ndarray
    surface_normal: np.ndarray
    surface_area: np.ndarray
    stl_vertices: np.ndarray
    stl_faces: np.ndarray
    a_ref: float
    cd_ref: float | None
    cl_ref: float | None


def case_path(data_root: str | Path, name: str) -> Path:
    p = Path(data_root) / f"{name}.pt"
    if not p.exists():
        raise FileNotFoundError(f"raw case {p} not found")
    return p


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _scalar(ref, *names) -> float | None:
    """A named scalar anywhere in the (nested) reference block."""
    if isinstance(ref, dict):
        for n in names:
            if n in ref:
                return float(ref[n])
        for v in ref.values():
            hit = _scalar(v, *names)
            if hit is not None:
                return hit
    elif isinstance(ref, (list, tuple)):
        for v in ref:
            hit = _scalar(v, *names)
            if hit is not None:
                return hit
    return None


def load_raw_case(path: str | Path, name: str) -> RawCase:
    pt = torch.load(str(path), map_location="cpu", weights_only=False)
    ref = pt.get("reference")
    a_ref = _scalar(ref, "aRef", "a_ref") if ref is not None else None
    if a_ref is None:
        raise KeyError(f"{path}: reference area aRef missing")
    raw = RawCase(
        name=name,
        volume_pos=_np(pt["volume_coords"]).astype(np.float32),
        volume_velocity=_np(pt["volume_velocity"]).astype(np.float32),
        volume_pressure=_np(pt["volume_pressure"]).reshape(-1).astype(np.float32),
        volume_nut=_np(pt["volume_nut"]).reshape(-1).astype(np.float32),
        surface_pos=_np(pt["surface_coords"]).astype(np.float32),
        surface_pressure=_np(pt["surface_pressure"]).reshape(-1).astype(np.float32),
        surface_wss=_np(pt["surface_wallshearstress"]).astype(np.float32),
        surface_normal=_np(pt["surface_normal"]).astype(np.float32),
        surface_area=_np(pt["surface_area"]).reshape(-1).astype(np.float32),
        stl_vertices=_np(pt["stl_vertices"]).astype(np.float32),
        stl_faces=_np(pt["stl_faces"]).astype(np.int32),
        a_ref=float(a_ref), cd_ref=_scalar(ref, "Cd", "cd"), cl_ref=_scalar(ref, "Cl", "cl"))
    bad = [k for k in ("volume_pos", "volume_velocity", "volume_pressure", "volume_nut", "surface_pos",
                       "surface_pressure", "surface_wss", "surface_normal", "surface_area")
           if not np.isfinite(getattr(raw, k)).all()]
    if bad:
        raise RuntimeError(f"{name}: non-finite values in {bad}")
    return raw


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def crop_mask(points: np.ndarray, lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)
    return np.all((points >= lo) & (points <= hi), axis=1)


def _scene(vertices: np.ndarray, faces: np.ndarray):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(np.asarray(vertices, dtype=np.float32))
    mesh.triangle.indices = o3d.core.Tensor(np.asarray(faces, dtype=np.int32))
    scene.add_triangles(mesh)
    return scene


def compute_sdf_disp(points: np.ndarray, vertices, faces, chunk_size: int) -> np.ndarray:
    """q - closest surface point, metres; its norm is the distance of a
    fluid point and its direction the distance gradient."""
    import open3d as o3d

    points = np.asarray(points, dtype=np.float32)
    scene = _scene(vertices, faces)
    disp = np.empty((len(points), 3), dtype=np.float32)
    for lo in range(0, len(points), chunk_size):
        q = o3d.core.Tensor(points[lo:lo + chunk_size])
        closest = scene.compute_closest_points(q)["points"].numpy().astype(np.float32)
        disp[lo:lo + chunk_size] = points[lo:lo + chunk_size] - closest
    return disp


def compute_sdf_scalar_grad(points: np.ndarray, vertices, faces, chunk_size: int):
    """Signed distance and unit gradient at arbitrary points (centroids may
    lie inside the body). -> (sdf [N] metres, grad [N, 3])."""
    import open3d as o3d

    points = np.asarray(points, dtype=np.float32)
    scene = _scene(vertices, faces)
    sdf = np.empty(len(points), dtype=np.float32)
    grad = np.empty((len(points), 3), dtype=np.float32)
    for lo in range(0, len(points), chunk_size):
        q = o3d.core.Tensor(points[lo:lo + chunk_size])
        closest = scene.compute_closest_points(q)["points"].numpy().astype(np.float32)
        signed = scene.compute_signed_distance(q).numpy().astype(np.float32)
        disp = points[lo:lo + chunk_size] - closest
        unit = disp / np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-12)
        sdf[lo:lo + chunk_size] = signed
        grad[lo:lo + chunk_size] = unit * np.sign(signed)[:, None]
    return sdf, grad


def integrate_cdcl(pressure, wall_shear, normal, area, a_ref: float, weight: float = 1.0) -> dict:
    """Drag and lift coefficients: pressure force sum(p n A), shear force
    -sum(tau A), 2 F / (U^2 A_ref) with kinematic pressure (rho = 1).
    `weight` = n_surface / n_sampled for a uniform surface sample."""
    pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
    normal = np.asarray(normal, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64).reshape(-1)
    pforce = (normal.T * (pressure * area)).sum(axis=1) * weight
    sforce = -(np.asarray(wall_shear, dtype=np.float64).T * area).sum(axis=1) * weight
    force = pforce + sforce
    qref = 0.5 * U_INF ** 2 * float(a_ref)
    return {"cd": float(force[0] / qref), "cl": float(force[2] / qref)}


def resolve_normal_orientation(raw: RawCase, tol: float) -> tuple[np.ndarray, dict]:
    """Outward normals: the STL orientation is kept when the integrated
    Cd/Cl match the reference within tol, flipped when the flipped ones
    match better. -> (normals, {"cd", "cl", "flipped", "reference_error"})."""
    normal = raw.surface_normal.astype(np.float32)
    base = integrate_cdcl(raw.surface_pressure, raw.surface_wss, normal, raw.surface_area, raw.a_ref)
    if raw.cd_ref is None and raw.cl_ref is None:
        return normal, {**base, "flipped": False, "reference_error": None}

    def err(m):
        e = 0.0
        if raw.cd_ref is not None:
            e = max(e, abs(m["cd"] - raw.cd_ref))
        if raw.cl_ref is not None:
            e = max(e, abs(m["cl"] - raw.cl_ref))
        return e

    e0 = err(base)
    if e0 <= tol:
        return normal, {**base, "flipped": False, "reference_error": e0}
    flipped = integrate_cdcl(raw.surface_pressure, raw.surface_wss, -normal, raw.surface_area, raw.a_ref)
    e1 = err(flipped)
    if e1 < e0:
        if e1 > tol:
            print(f"[cdcl] {raw.name}: flipped normals, error {e1:.4g} > tol {tol}", flush=True)
        return (-normal).astype(np.float32), {**flipped, "flipped": True, "reference_error": e1}
    print(f"[cdcl] {raw.name}: Cd/Cl reference mismatch (error {e0:.4g}), keeping the raw normals",
          flush=True)
    return normal, {**base, "flipped": False, "reference_error": e0}

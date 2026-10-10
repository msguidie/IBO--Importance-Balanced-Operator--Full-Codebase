"""TandemFoilSet preprocessing: the shipped Part pickles of one subset ->
the cache read by every model. Per subset (own cache, statistics and
calibration):

  unpack     each Part pickle -> one raw case file
  bake       per case, physical units: the two airfoil polygons, the signed
             distance and gradient to each airfoil at every mesh point, the
             two-body directional distance profile, the curvature proxy, the
             latent layout (equal-count KD-tree, or the lattice of the
             layout ablation) with the same fields at the token positions,
             and the k nearest tokens of every point
  slots      per token, k member points with 18 features
  stats      train-split statistics (Welford) -> coef_norm.pt
  finalize   z-scored per-point (static/) and per-token (latent/) tensors
  calibrate  the RoPE domain and the Fourier SDF scale of this subset

Targets are nondimensionalised by [U, U, U^2] before the z-score; the
case condition is (U, angle of attack) z-scored; absolute coordinates get
a per-axis P5/P95 affine, difference vectors their own z-score. Cases with
non-finite values, or (racecar, the steady regime) a max Cp above the
incompressible bound by more than a discretisation margin, are skipped;
the published split already excludes them. --from-cache (layout grid)
copies every position-only field of the importance cache and rebuilds the
partition-dependent ones.
"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import os.path as osp
import shutil

import numpy as np
import torch

from ...config import Config, repo_root
from ...utils.seed import per_case_epoch_seed
from ..common.calibration import ROPE_UNITS_PER_CHORD, run_calibration, suggest_rope_base
from ..common.kdtree import gather_slots, knn_indices, latent_partition, occupancy_summary
from ..common.polygon import (combine_did, compute_did, compute_knn_curvature,
                              polygon_sdf_and_grad, polygon_vertex_normals)
from ..common.pool import atomic_torch_save, limit_threads, report, run_pool
from ..common.welford import WelfordAccumulator, welford_local
from .cases import CACHE_VERSION, cache_dir

# file patterns of the shipped pickles and the simulation domain (chords)
SUBSETS = {
    "cruise_random": dict(patterns=["cruise_randomFields_*Part*.pickle"], exclude="extrap",
                          domain_lo=(-9.6, -9.6), domain_hi=(13.6, 9.6)),
    "cruise_aoa0": dict(patterns=["cruise_Re500_aoa0_*Part*.pickle"], exclude=None,
                        domain_lo=(-9.9, -9.9), domain_hi=(13.9, 9.9)),
    "cruise_aoa5": dict(patterns=["cruise_Re500_aoa5_*Part*.pickle"], exclude=None,
                        domain_lo=(-9.9, -9.9), domain_hi=(13.9, 9.9)),
    "takeoff": dict(patterns=["takeoff_*Part*.pickle"], exclude=None,
                    domain_lo=(-9.9, -1.0), domain_hi=(13.9, 9.9)),
    "racecar": dict(patterns=["raceCar_randomFields_*Part*.pickle"], exclude=None,
                    domain_lo=(-9.6, 0.0), domain_hi=(11.5, 9.7)),
}
BOUNDARY_FRONT, BOUNDARY_REAR = 5, 6
# slot features (18): rel pos 2 | sdf1 | grad1 2 | sdf2 | grad2 2 | surface flag | did 8 | curvature
PN_RELPOS, PN_SDF1, PN_GRAD1, PN_SDF2 = slice(0, 2), 2, slice(3, 5), 5
PN_GRAD2, PN_SURF, PN_DID, PN_CURV = slice(6, 8), 8, slice(9, 17), 17

_W: dict = {}


def _init(state: dict) -> None:
    _W.clear()
    _W.update(state)
    limit_threads(1)


class QualityError(Exception):
    """A case the dataset shipped with unusable values: skipped, not failed."""


def check_quality(y_phys, umag: float, re_phys: float, cid: str, cp_bound: float,
                  cp_min_re: float) -> None:
    arr = y_phys.numpy() if torch.is_tensor(y_phys) else np.asarray(y_phys)
    if not np.isfinite(arr).all():
        raise QualityError(f"{cid}: non-finite target values")
    if float(re_phys) >= cp_min_re and umag > 0:
        cp_max = float(arr[:, 2].max()) / (0.5 * umag * umag)
        if cp_max > cp_bound:
            raise QualityError(f"{cid}: Cp_max={cp_max:.4f} > {cp_bound} at Re={re_phys:.3g} "
                               f"(steady solution beyond the incompressible bound)")


def chain_surface_polygon(boundary, edge_index, code: int, pos):
    """The surface nodes of one airfoil (boundary == code) ordered into a
    closed polygon along the mesh edges. -> (node indices [P], polygon [P, 2])."""
    idx = (boundary == code).nonzero().flatten()
    if idx.numel() < 3:
        raise ValueError(f"boundary code {code}: only {idx.numel()} nodes")
    idset = set(idx.tolist())
    m = torch.isin(edge_index[0], idx) & torch.isin(edge_index[1], idx)
    adj = {}
    for a, b in edge_index[:, m].t().tolist():
        if a != b:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    bad = [n for n in idset if len(adj.get(n, ())) != 2]
    if bad:
        raise ValueError(f"boundary code {code}: {len(bad)} nodes without two surface neighbours")
    start = idx[0].item()
    loop, prev, cur = [start], None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev][0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(idset):
            raise ValueError(f"boundary code {code}: loop overrun")
    if len(loop) != len(idset):
        raise ValueError(f"boundary code {code}: chained {len(loop)} of {len(idset)} nodes")
    loop_t = torch.tensor(loop, dtype=torch.long)
    return loop_t, pos[loop_t].clone()


# ---------------------------------------------------------------------------
# unpack
# ---------------------------------------------------------------------------

def select_subset_files(data_dir: str, subset: str) -> list[str]:
    sc = SUBSETS[subset]
    files = []
    for pat in sc["patterns"]:
        files.extend(glob.glob(osp.join(data_dir, pat)))
    if sc["exclude"]:
        files = [f for f in files if sc["exclude"] not in osp.basename(f)]
    ive = [f for f in files if "_ive_" in osp.basename(f)]
    mgn = [f for f in files if "_mgn_" in osp.basename(f)]
    chosen = ive if ive else mgn
    if not chosen:
        raise SystemExit(f"[{subset}] no Part pickles found in {data_dir}")
    return sorted(chosen)


def unpack_subset(files: list[str], subset: str, unpack_dir: str) -> list[str]:
    os.makedirs(unpack_dir, exist_ok=True)
    case_ids = []
    for pi, f in enumerate(files, start=1):
        print(f"[unpack] {osp.basename(f)} ...", flush=True)
        d = torch.load(f, map_location="cpu", weights_only=False)
        for i, case in enumerate(d):
            cid = f"{subset}.p{pi}.{i:04d}"
            torch.save(case, osp.join(unpack_dir, f"{cid}.pt"))
            case_ids.append(cid)
        del d
        gc.collect()
    print(f"[unpack] {len(case_ids)} cases from {len(files)} files", flush=True)
    return case_ids


# ---------------------------------------------------------------------------
# bake (per case, physical units)
# ---------------------------------------------------------------------------

def _partition(pos_np: np.ndarray, w: dict) -> dict:
    return latent_partition(pos_np, w["n_leaves"], w["latent_grid"] == "structured",
                            w["domain_lo"], w["domain_hi"], w["grid_dims"])


def _token_fields(centroids: np.ndarray, poly1, poly2, w: dict) -> dict:
    c = torch.from_numpy(centroids)
    s1, g1 = polygon_sdf_and_grad(c, poly1)
    s2, g2 = polygon_sdf_and_grad(c, poly2)
    bins, dmax = w["did_bins"], w["did_dmax"]
    did = combine_did(compute_did(centroids, poly1.numpy(), bins, dmax),
                      compute_did(centroids, poly2.numpy(), bins, dmax), dmax)
    return {"leaf_pos_geo": c, "leaf_sdf1_raw": s1, "leaf_sdf_grad1": g1, "leaf_sdf2_raw": s2,
            "leaf_sdf_grad2": g2, "leaf_did_raw": torch.from_numpy(did)}


def _bake_case(cid: str):
    w = _W
    out_path = osp.join(w["wip_dir"], f"{cid}.pt")
    if osp.exists(out_path):
        cached = torch.load(out_path, map_location="cpu", weights_only=False)
        if cached.get("version") == CACHE_VERSION and cached.get("latent_grid") == w["latent_grid"]:
            check_quality(cached["y_phys"], cached["umag_phys"], cached["re_phys"], cid,
                          w["cp_bound"], w["cp_min_re"])
            return ("ok", cid, None, int(cached["n_points"]))
    raw = torch.load(osp.join(w["unpack_dir"], f"{cid}.pt"), map_location="cpu", weights_only=False)
    pos = raw["pos"].float()
    boundary = raw["boundary"].to(torch.uint8)
    y = raw["y"].float()
    flow = raw["flowState"]
    umag, aoa = float(flow["Umag"]), float(raw["AoA"][0])
    re, nu = float(flow["Re"]), float(flow["nu"])
    if not (np.isfinite(umag) and umag > 0 and np.isfinite(aoa)):
        raise QualityError(f"{cid}: flow state Umag={umag} AoA={aoa}")
    if not torch.isfinite(pos).all():
        raise QualityError(f"{cid}: non-finite positions")
    check_quality(y, umag, re, cid, w["cp_bound"], w["cp_min_re"])

    loop1, poly1 = chain_surface_polygon(boundary, raw["edge_index"], BOUNDARY_FRONT, pos)
    loop2, poly2 = chain_surface_polygon(boundary, raw["edge_index"], BOUNDARY_REAR, pos)
    sdf1, grad1 = polygon_sdf_and_grad(pos, poly1)
    sdf2, grad2 = polygon_sdf_and_grad(pos, poly2)
    sdf1[loop1], grad1[loop1] = 0.0, polygon_vertex_normals(poly1)   # wall rows: exact zero, outward normal
    sdf2[loop2], grad2[loop2] = 0.0, polygon_vertex_normals(poly2)
    surf = (boundary == BOUNDARY_FRONT) | (boundary == BOUNDARY_REAR)
    pos_np = pos.numpy()
    n = int(pos_np.shape[0])

    part = _partition(pos_np, w)
    bins, dmax = w["did_bins"], w["did_dmax"]
    mesh_did = combine_did(compute_did(pos_np, poly1.numpy(), bins, dmax),
                           compute_did(pos_np, poly2.numpy(), bins, dmax), dmax)
    # curvature from the gradient field of the nearer airfoil
    grad_dom = torch.where((sdf1.abs() <= sdf2.abs())[:, None], grad1, grad2)
    mesh_curv = compute_knn_curvature(pos_np, pos_np, grad_dom.numpy(), k=w["curvature_k"])
    assert w["n_leaves"] <= 65536, "token indices are stored as uint16"
    idw = knn_indices(pos_np, part["centroids"], w["idw_k"])

    wip = {"version": CACHE_VERSION, "case_id": cid, "n_points": n, "pos": pos,
           "boundary": boundary, "surf": surf, "sdf1_raw": sdf1, "sdf_grad1": grad1,
           "sdf2_raw": sdf2, "sdf_grad2": grad2, "y_phys": y,
           "idw_indices": torch.from_numpy(idw.astype(np.uint16)),
           "tree_perm": torch.from_numpy(part["perm"]),
           "tree_leaf_offsets": torch.from_numpy(part["leaf_offsets"]),
           "tree_split_axes": torch.from_numpy(part["split_axes"]),
           "tree_split_values": torch.from_numpy(part["split_values"]),
           "latent_grid": w["latent_grid"],
           "cell_ids": torch.from_numpy(part["cell_ids"].astype(np.int32)),
           "n_out_of_box": int(part["n_out_of_box"]),
           "mesh_did_raw": torch.from_numpy(mesh_did), "mesh_curv_raw": torch.from_numpy(mesh_curv),
           "umag_phys": umag, "aoa_deg": aoa, "re_phys": re, "nu_phys": nu,
           "poly1": poly1, "poly2": poly2}
    wip.update(_token_fields(part["centroids"], poly1, poly2, w))
    atomic_torch_save(wip, out_path)
    return ("ok", cid, None, n)


def _bake_worker(cid: str):
    try:
        return _bake_case(cid)
    except QualityError as exc:
        return ("skipped", cid, str(exc), None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", cid, str(exc), None)


def _releaf_worker(cid: str):
    """Layout grid from the importance cache: every position-only field is
    copied, only the partition-dependent ones are rebuilt."""
    try:
        w = _W
        out_path = osp.join(w["wip_dir"], f"{cid}.pt")
        if osp.exists(out_path):
            cached = torch.load(out_path, map_location="cpu", weights_only=False)
            if cached.get("version") == CACHE_VERSION and cached.get("latent_grid") == "structured" \
                    and int(cached["tree_leaf_offsets"].numel()) == w["n_leaves"] + 1:
                return ("ok", cid, None, int(cached["n_points"]))
        src = torch.load(osp.join(w["src_wip_dir"], f"{cid}.pt"), map_location="cpu", weights_only=False)
        if src.get("version") != CACHE_VERSION:
            raise RuntimeError(f"source cache version {src.get('version')!r} != {CACHE_VERSION}")
        pos_np = src["pos"].numpy()
        part = _partition(pos_np, w)
        wip = dict(src)
        wip.update({"idw_indices": torch.from_numpy(knn_indices(pos_np, part["centroids"], w["idw_k"]).astype(np.uint16)),
                    "tree_perm": torch.from_numpy(part["perm"]),
                    "tree_leaf_offsets": torch.from_numpy(part["leaf_offsets"]),
                    "tree_split_axes": torch.from_numpy(part["split_axes"]),
                    "tree_split_values": torch.from_numpy(part["split_values"]),
                    "latent_grid": "structured",
                    "cell_ids": torch.from_numpy(part["cell_ids"].astype(np.int32)),
                    "n_out_of_box": int(part["n_out_of_box"])})
        wip.update(_token_fields(part["centroids"], src["poly1"], src["poly2"], w))
        atomic_torch_save(wip, out_path)
        return ("ok", cid, None, int(src["n_points"]))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", cid, str(exc), None)


# ---------------------------------------------------------------------------
# slots (k member points per token, 18 features, physical units)
# ---------------------------------------------------------------------------

def _slots_worker(cid: str):
    try:
        w = _W
        out_path = osp.join(w["pn_dir"], f"{cid}.pt")
        if osp.exists(out_path):
            cached = torch.load(out_path, map_location="cpu", weights_only=False)
            if cached.get("version") == CACHE_VERSION and int(cached.get("k", -1)) == int(w["k"]) \
                    and cached.get("latent_grid") == w["latent_grid"]:
                return ("ok", cid, None, None)
        wip = torch.load(osp.join(w["wip_dir"], f"{cid}.pt"), map_location="cpu", weights_only=False)
        pos = wip["pos"].numpy()
        part = {"centroids": wip["leaf_pos_geo"].numpy(), "perm": wip["tree_perm"].numpy(),
                "leaf_offsets": wip["tree_leaf_offsets"].numpy(),
                "cell_ids": wip["cell_ids"].numpy().astype(np.int64)}
        rel_pos, valid, nbr, counts = gather_slots(pos, part, w["k"], per_case_epoch_seed(cid, 0),
                                                   w["latent_grid"] == "structured")
        safe = np.where(valid, nbr, 0)

        def pick1(arr):
            return np.where(valid, arr[safe], 0.0)[..., None].astype(np.float32)

        def pick2(arr):
            v = arr[safe].astype(np.float32)
            v[~valid] = 0.0
            return v

        pn_input = np.concatenate([
            rel_pos, pick1(wip["sdf1_raw"].numpy()), pick2(wip["sdf_grad1"].numpy()),
            pick1(wip["sdf2_raw"].numpy()), pick2(wip["sdf_grad2"].numpy()),
            pick1(wip["surf"].numpy().astype(np.float32)), pick2(wip["mesh_did_raw"].numpy()),
            pick1(wip["mesh_curv_raw"].numpy())], axis=-1).astype(np.float32)
        atomic_torch_save({"version": CACHE_VERSION, "case_id": cid, "k": int(w["k"]),
                           "latent_grid": w["latent_grid"], "pn_input": torch.from_numpy(pn_input),
                           "pn_mask": torch.from_numpy(valid)}, out_path)
        return ("ok", cid, None, counts)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", cid, str(exc), None)


# ---------------------------------------------------------------------------
# train-split statistics
# ---------------------------------------------------------------------------

def _stats_worker(cid: str):
    """Per-case local Welford triples in a fixed order (targets nondim,
    targets raw, slot offsets, query sdf1, query sdf2, token sdf1, token
    sdf2, query did, token did, query curvature), the condition, and a
    bounded position sample for the coordinate percentiles."""
    w = _W
    wip = torch.load(osp.join(w["wip_dir"], f"{cid}.pt"), map_location="cpu", weights_only=False)
    pn = torch.load(osp.join(w["pn_dir"], f"{cid}.pt"), map_location="cpu", weights_only=False)
    u = float(wip["umag_phys"])
    y = wip["y_phys"].numpy().astype(np.float64)
    mask = pn["pn_mask"].numpy()
    arrays = (y / np.array([u, u, u * u]), y, pn["pn_input"].numpy()[..., PN_RELPOS][mask],
              wip["sdf1_raw"].numpy()[:, None], wip["sdf2_raw"].numpy()[:, None],
              wip["leaf_sdf1_raw"].numpy()[:, None], wip["leaf_sdf2_raw"].numpy()[:, None],
              wip["mesh_did_raw"].numpy(), wip["leaf_did_raw"].numpy(),
              wip["mesh_curv_raw"].numpy()[:, None])
    triples = tuple(welford_local(a) for a in arrays)
    pos = wip["pos"].numpy()
    rng = np.random.default_rng(per_case_epoch_seed(cid, 1))
    k = w["coord_sample"]
    sample = pos[rng.choice(len(pos), size=k, replace=False)] if len(pos) > k else pos
    return triples, u, float(wip["aoa_deg"]), sample.astype(np.float32)


def compute_coef_norm(state: dict, train: list[str], subset: str, workers: int, pn_k: int,
                      pctl: tuple[float, float], did_bins: int) -> dict:
    widths = (3, 3, 2, 1, 1, 1, 1, did_bins, did_bins, 1)
    accs = [WelfordAccumulator(c) for c in widths]
    results = run_pool(train, _stats_worker, _init, (state,), workers, f"{subset}:stats", ordered=True)
    umags, aoas, samples = [], [], []
    for triples, u, aoa, sample in results:
        for acc, (n, mean, m2) in zip(accs, triples):
            acc.merge_local(n, mean, m2)
        umags.append(u)
        aoas.append(aoa)
        samples.append(sample)
    (t_m, t_s), (p_m, p_s), (rp_m, rp_s), (q1_m, q1_s), (q2_m, q2_s), (l1_m, l1_s), (l2_m, l2_s), \
        (qd_m, qd_s), (ld_m, ld_s), (qc_m, qc_s) = [acc.finalize() for acc in accs]
    pooled = np.concatenate(samples, axis=0).astype(np.float64)

    def t(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    out = {"version": CACHE_VERSION, "subset": subset,
           "mean_out": t(t_m), "std_out": t(t_s),                       # training space
           "mean_out_protocol": t(p_m), "std_out_protocol": t(p_s),     # reporting space (raw fields)
           "query_sdf1_mean": float(q1_m[0]), "query_sdf1_std": float(q1_s[0]),
           "query_sdf2_mean": float(q2_m[0]), "query_sdf2_std": float(q2_s[0]),
           "latent_sdf1_mean": float(l1_m[0]), "latent_sdf1_std": float(l1_s[0]),
           "latent_sdf2_mean": float(l2_m[0]), "latent_sdf2_std": float(l2_s[0]),
           "relpos_mean": t(rp_m), "relpos_std": t(rp_s),
           "query_did_mean": t(qd_m), "query_did_std": t(qd_s),
           "latent_did_mean": t(ld_m), "latent_did_std": t(ld_s),
           "query_curv_mean": float(qc_m[0]), "query_curv_std": float(qc_s[0]),
           "mean_cond": t([np.mean(umags), np.mean(aoas)]), "std_cond": t([np.std(umags), np.std(aoas)]),
           "coord_p5": t(np.percentile(pooled, pctl[0], axis=0)),
           "coord_p95": t(np.percentile(pooled, pctl[1], axis=0)),
           "coord_pctl_lo": float(pctl[0]), "coord_pctl_hi": float(pctl[1]),
           "n_train_cases": len(train), "idw_k": int(state["idw_k"]), "pn_k": int(pn_k)}
    for k, v in out.items():
        if (torch.is_tensor(v) and not torch.isfinite(v).all()) or (isinstance(v, float) and not math.isfinite(v)):
            raise RuntimeError(f"coef_norm: {k} is not finite")
    return out


# ---------------------------------------------------------------------------
# finalize: wip + statistics -> static/ and latent/
# ---------------------------------------------------------------------------

def coord_affine(x, p5, p95) -> np.ndarray:
    """(x - p5) / (p95 - p5) per axis, no clamping; model-input coordinates only."""
    span = np.maximum(np.asarray(p95, dtype=np.float64) - np.asarray(p5, dtype=np.float64), 1e-8)
    return ((np.asarray(x, dtype=np.float64) - np.asarray(p5)) / span).astype(np.float32)


def _finalize_worker(cid: str):
    try:
        w = _W
        cn = w["coef"]
        static_path = osp.join(w["static_dir"], f"{cid}.pt")
        latent_path = osp.join(w["latent_dir"], f"{cid}.pt")
        if osp.exists(static_path) and osp.exists(latent_path):
            return ("ok", cid, None)
        wip = torch.load(osp.join(w["wip_dir"], f"{cid}.pt"), map_location="cpu", weights_only=False)
        pn = torch.load(osp.join(w["pn_dir"], f"{cid}.pt"), map_location="cpu", weights_only=False)

        def z(x, mean, std):
            return ((np.asarray(x, dtype=np.float64) - mean) / np.maximum(std, 1e-8)).astype(np.float32)

        def t64(key):
            return cn[key].numpy().astype(np.float64)

        q1 = (cn["query_sdf1_mean"], cn["query_sdf1_std"])
        q2 = (cn["query_sdf2_mean"], cn["query_sdf2_std"])
        qd, ld = (t64("query_did_mean"), t64("query_did_std")), (t64("latent_did_mean"), t64("latent_did_std"))
        qc = (cn["query_curv_mean"], cn["query_curv_std"])
        p5, p95 = t64("coord_p5"), t64("coord_p95")
        pos = wip["pos"].numpy()
        u = float(wip["umag_phys"])
        y_nondim = wip["y_phys"].numpy().astype(np.float64) / np.array([u, u, u * u])
        static = {"version": CACHE_VERSION, "case_id": cid, "n_points": int(wip["n_points"]),
                  "norm_pos": torch.from_numpy(coord_affine(pos, p5, p95)),
                  "query_pos_geo": torch.from_numpy(pos.astype(np.float32)),
                  "sdf1": torch.from_numpy(z(wip["sdf1_raw"].numpy(), *q1)), "sdf_grad1": wip["sdf_grad1"].float(),
                  "sdf2": torch.from_numpy(z(wip["sdf2_raw"].numpy(), *q2)), "sdf_grad2": wip["sdf_grad2"].float(),
                  "did": torch.from_numpy(z(wip["mesh_did_raw"].numpy(), *qd)),
                  "curvature": torch.from_numpy(z(wip["mesh_curv_raw"].numpy(), *qc)),
                  "target": torch.from_numpy(z(y_nondim, t64("mean_out"), t64("std_out"))),
                  "surf": wip["surf"].bool(), "boundary": wip["boundary"],
                  "idw_indices": wip["idw_indices"],
                  "cond": torch.from_numpy(z([u, float(wip["aoa_deg"])], t64("mean_cond"), t64("std_cond"))),
                  "umag_phys": u, "aoa_deg": float(wip["aoa_deg"]), "re_phys": float(wip["re_phys"]),
                  "nu_phys": float(wip["nu_phys"]), "poly1": wip["poly1"].float(), "poly2": wip["poly2"].float()}
        atomic_torch_save(static, static_path)

        pn_in = pn["pn_input"].numpy().copy()
        mask = pn["pn_mask"].numpy()
        rp_m, rp_s = t64("relpos_mean"), np.maximum(t64("relpos_std"), 1e-8)
        for a in (0, 1):                      # offsets: their own z-score, valid slots only
            pn_in[..., a][mask] = (pn_in[..., a][mask] - rp_m[a]) / rp_s[a]
        pn_in[..., PN_SDF1][mask] = z(pn_in[..., PN_SDF1][mask], *q1)
        pn_in[..., PN_SDF2][mask] = z(pn_in[..., PN_SDF2][mask], *q2)
        pn_in[..., PN_DID][mask] = z(pn_in[..., PN_DID][mask], *qd)
        pn_in[..., PN_CURV][mask] = z(pn_in[..., PN_CURV][mask], *qc)
        # gradients (unit vectors) and the surface flag stay raw
        lpg = wip["leaf_pos_geo"].numpy()
        latent = {"version": CACHE_VERSION, "case_id": cid,
                  "pn_input": torch.from_numpy(pn_in.astype(np.float32)), "pn_mask": torch.from_numpy(mask.copy()),
                  "leaf_norm_pos": torch.from_numpy(coord_affine(lpg, p5, p95)),
                  "leaf_pos_geo": torch.from_numpy(lpg.astype(np.float32)),
                  "leaf_sdf1": torch.from_numpy(z(wip["leaf_sdf1_raw"].numpy(), cn["latent_sdf1_mean"], cn["latent_sdf1_std"])),
                  "leaf_sdf_grad1": wip["leaf_sdf_grad1"].float(),
                  "leaf_sdf2": torch.from_numpy(z(wip["leaf_sdf2_raw"].numpy(), cn["latent_sdf2_mean"], cn["latent_sdf2_std"])),
                  "leaf_sdf_grad2": wip["leaf_sdf_grad2"].float(),
                  "leaf_did": torch.from_numpy(z(wip["leaf_did_raw"].numpy(), *ld)),
                  "tree_split_axes": wip["tree_split_axes"], "tree_split_values": wip["tree_split_values"]}
        atomic_torch_save(latent, latent_path)
        return ("ok", cid, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", cid, str(exc))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def published_split(subset: str) -> dict:
    with open(repo_root() / "datasets" / "tandemfoilset" / "splits.json", encoding="utf-8") as f:
        return json.load(f)["subsets"][subset]


def run_preprocess(cfg: Config, from_cache: bool = False) -> None:
    pp, subset = cfg.tandemfoilset, cfg.run.subset
    if subset not in SUBSETS:
        raise SystemExit(f"unknown subset {subset!r}; subsets: {', '.join(SUBSETS)}")
    sc = SUBSETS[subset]
    n_leaves = int(pp.n_leaves)
    depth = int(round(math.log2(n_leaves)))
    assert 2 ** depth == n_leaves, "n_leaves must be a power of two"
    structured = cfg.run.layout == "grid"
    grid_dims = tuple(int(v) for v in pp.grid_dims)
    if structured:
        assert grid_dims[0] * grid_dims[1] == n_leaves, f"grid_dims {grid_dims} must multiply to {n_leaves}"
    cache = cache_dir(cfg)
    wip_dir, pn_dir = str(cache / "wip"), str(cache / "wip_pn")
    static_dir, latent_dir = str(cache / "static"), str(cache / "latent")
    unpack_dir = str(cache / "_unpacked")
    for d in (wip_dir, pn_dir, static_dir, latent_dir):
        os.makedirs(d, exist_ok=True)
    workers = int(pp.n_workers)
    state = dict(unpack_dir=unpack_dir, wip_dir=wip_dir, pn_dir=pn_dir, n_leaves=n_leaves, depth=depth,
                 idw_k=int(pp.idw_k), latent_grid="structured" if structured else "adaptive",
                 grid_dims=grid_dims, domain_lo=tuple(sc["domain_lo"]), domain_hi=tuple(sc["domain_hi"]),
                 did_bins=int(pp.did_bins), did_dmax=float(pp.did_dmax), curvature_k=int(pp.curvature_k),
                 cp_bound=float(pp.cp_gate), cp_min_re=float(pp.cp_gate_min_re),
                 coord_sample=int(pp.coord_pctl_sample))
    print(f"\n=== [{subset}] -> {cache} (layout {cfg.run.layout}, {n_leaves} tokens) ===", flush=True)

    if from_cache:
        if not structured:
            raise SystemExit("--from-cache rebuilds the grid layout from the importance cache: add --layout grid")
        src = cache_dir(cfg, "importance")
        for p in (src / "wip", src / "manifest.json", src / "kdtree.pt"):
            if not p.exists():
                raise SystemExit(f"source cache incomplete: {p} missing (bake the importance layout first)")
        with open(src / "manifest.json", encoding="utf-8") as f:
            man = json.load(f)
        case_ids = sorted(set(man["train"]) | set(man["val"]) | set(man["test"]))
        src_meta = torch.load(str(src / "kdtree.pt"), map_location="cpu", weights_only=False)
        if int(src_meta["n_leaves"]) != n_leaves or int(src_meta["idw_k"]) != int(pp.idw_k):
            raise SystemExit(f"source cache has {src_meta['n_leaves']} tokens / idw_k {src_meta['idw_k']}")
        pn_k = int(src_meta["pn_k"])          # the same slots per token as the source
        results = run_pool(case_ids, _releaf_worker, _init, (dict(state, src_wip_dir=str(src / "wip")),),
                           workers, f"{subset}:releaf")
        report(results, "releaf")
        ok_ids = sorted(r[1] for r in results if r[0] == "ok")
        if set(ok_ids) != set(case_ids):
            raise SystemExit(f"[{subset}] releaf failed on {len(set(case_ids) - set(ok_ids))} cases")
    else:
        files = select_subset_files(cfg.data.data_root, subset)
        print(f"[{subset}] {len(files)} pickles: " + ", ".join(osp.basename(f) for f in files), flush=True)
        case_ids = sorted(osp.splitext(osp.basename(p))[0] for p in glob.glob(osp.join(unpack_dir, "*.pt")))
        if not case_ids:
            existing = sorted(osp.splitext(osp.basename(p))[0] for p in glob.glob(osp.join(wip_dir, "*.pt")))
            if existing:
                probe = torch.load(osp.join(wip_dir, f"{existing[0]}.pt"), map_location="cpu", weights_only=False)
                if probe.get("version") == CACHE_VERSION and probe.get("latent_grid") == state["latent_grid"]:
                    case_ids = existing
                    print(f"[unpack] reusing {len(case_ids)} baked cases", flush=True)
            if not case_ids:
                case_ids = unpack_subset(files, subset, unpack_dir)
        results = run_pool(case_ids, _bake_worker, _init, (state,), workers, f"{subset}:bake")
        report(results, "bake")
        ok = [(r[1], r[3]) for r in results if r[0] == "ok"]
        ok_ids = sorted(c for c, _ in ok)
        if not ok_ids:
            raise SystemExit("bake produced no cases")
        # slots per token = mean token occupancy over the subset
        pn_k = max(1, int(round(sum(n for _, n in ok) / (len(ok) * n_leaves))))
        print(f"[bake] mean token occupancy -> {pn_k} slots per token", flush=True)

    lo, hi = state["domain_lo"], state["domain_hi"]
    rope_dims = tuple(int(d) for d in cfg.ibo.rope_dims)
    rope_scales = [float(round(ROPE_UNITS_PER_CHORD * (hi[a] - lo[a]))) for a in range(2)]
    meta = {"version": CACHE_VERSION, "subset": subset, "depth": depth, "n_leaves": n_leaves,
            "did_bins": int(pp.did_bins), "did_dmax": float(pp.did_dmax), "idw_k": int(pp.idw_k),
            "pn_k": int(pn_k), "latent_grid": state["latent_grid"], "grid_dims": list(grid_dims),
            "grid_domain_lo": list(lo), "grid_domain_hi": list(hi), "rope_dims": list(rope_dims),
            "rope_scales": rope_scales,
            "rope_base": [round(suggest_rope_base(rope_scales[a], rope_dims[a]), 3) for a in range(2)],
            "rope_domain_lo": list(lo), "rope_domain_hi": list(hi)}
    atomic_torch_save(meta, cache / "kdtree.pt")

    split = published_split(subset)
    wanted = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    missing = sorted(set(wanted) - set(ok_ids))
    if missing:
        raise SystemExit(f"[{subset}] {len(missing)} cases of the published split were not baked "
                         f"(e.g. {missing[:5]}); check the data directory and the failures above")
    with open(cache / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({k: list(split[k]) for k in ("train", "val", "test")}, f, indent=1)

    results = run_pool(wanted, _slots_worker, _init, (dict(state, k=pn_k),), workers, f"{subset}:slots")
    failed, _ = report(results, "slots")
    if failed:
        raise SystemExit(f"[{subset}] slot gathering failed on {len(failed)} cases")
    if structured:
        occupancy_summary([r[3] for r in results if r[3] is not None], pn_k, f"{subset}:grid")
    shutil.rmtree(unpack_dir, ignore_errors=True)

    coef = compute_coef_norm(state, list(split["train"]), subset, workers, pn_k,
                             tuple(float(v) for v in pp.coord_pctl), int(pp.did_bins))
    atomic_torch_save(coef, cache / "coef_norm.pt")
    results = run_pool(wanted, _finalize_worker, _init,
                       (dict(coef=coef, wip_dir=wip_dir, pn_dir=pn_dir, static_dir=static_dir,
                             latent_dir=latent_dir),), workers, f"{subset}:finalize")
    failed, _ = report(results, "finalize")
    if failed:
        raise SystemExit(f"[{subset}] finalize failed on {len(failed)} cases")
    run_calibration(cache / "static", cache / "latent", cache / "calibration_report.json",
                    {k: list(split[k]) for k in ("train", "val", "test")},
                    int(pp.calibration_sample_cases), rope_dims, int(pp.idw_k),
                    lambda static, sel: np.abs(np.concatenate([static["sdf1"].numpy()[sel],
                                                               static["sdf2"].numpy()[sel]])),
                    subset)
    print(f"[{subset}] done -> {cache} ({len(wanted)} cases)", flush=True)

"""GAOT's graphs on TandemFoilSet (derived cache), baked once per case over
the finished cache:

    <cache>/gaot/gaot_meta.pt   the latent grid (grid_h x grid_w cell centres
                                in the affine coordinate space), the radius
    <cache>/gaot/<case>.pt      encoder CSR (points within the radius of each
                                grid node), decoder CSR (grid nodes within
                                the radius of each mesh point, nearest node
                                when empty), the cosine-softmax edge
                                attention and the geometric embedding
                                features of both graphs

The grid spans the affine image of the subset's physical domain. The
radius is radius_factor x the larger cell side, so every mesh point sees
at least one node; grid nodes with no point inside the radius stay empty
(single-scale GAOT on a graded mesh). Edge attention and the geometric
features are pure functions of the fixed coordinates, so computing them
here is identical to computing them per step.
"""
from __future__ import annotations

import os
import os.path as osp

import numpy as np
import torch
from scipy.spatial import cKDTree

from ...models.gaot.agno import cosine_attention_scores, segment_softmax
from ...models.gaot.gemb import compute_statistical_features
from ..common.pool import atomic_torch_save, limit_threads, report, run_pool

GAOT_BAKE_VERSION = 1
GAOT_IN_CHANNELS = 18       # sdf1, grad1 (2), sdf2, grad2 (2), surface flag, did (8), curvature, cond (2)
_W: dict = {}


def gaot_pndata(case: dict) -> torch.Tensor:
    """[N, 18] per-point input: the z-scored geometry channels + the case
    condition broadcast to every point."""
    n = int(case["n_points"])
    return torch.cat([case["sdf1"].float()[:, None], case["sdf_grad1"].float(),
                      case["sdf2"].float()[:, None], case["sdf_grad2"].float(),
                      case["surf"].float()[:, None], case["did"].float(),
                      case["curvature"].float()[:, None],
                      case["cond"].float().view(1, 2).expand(n, 2)], dim=-1).contiguous()


def _affine(x, p5, p95):
    span = np.maximum(np.asarray(p95, np.float64) - np.asarray(p5, np.float64), 1e-8)
    return (np.asarray(x, np.float64) - np.asarray(p5)) / span


def build_grid(cache: str, grid_h: int, grid_w: int, radius_factor: float) -> dict:
    kdtree = torch.load(osp.join(cache, "kdtree.pt"), map_location="cpu", weights_only=False)
    cn = torch.load(osp.join(cache, "coef_norm.pt"), map_location="cpu", weights_only=False)
    p5, p95 = cn["coord_p5"].numpy(), cn["coord_p95"].numpy()
    lo = _affine(kdtree["rope_domain_lo"], p5, p95)
    hi = _affine(kdtree["rope_domain_hi"], p5, p95)
    cell = (hi - lo) / np.array([grid_h, grid_w], np.float64)
    xs = lo[0] + (np.arange(grid_h) + 0.5) * cell[0]
    ys = lo[1] + (np.arange(grid_w) + 0.5) * cell[1]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return {"version": GAOT_BAKE_VERSION, "subset": str(kdtree.get("subset")),
            "grid_h": int(grid_h), "grid_w": int(grid_w),
            "grid": torch.from_numpy(grid.astype(np.float32)),
            "radius": float(radius_factor * cell.max()), "radius_factor": float(radius_factor),
            "bounds_lo": torch.from_numpy(lo.astype(np.float32)),
            "bounds_hi": torch.from_numpy(hi.astype(np.float32))}


def _ball_csr(tree: cKDTree, queries: np.ndarray, radius: float):
    lists = tree.query_ball_point(queries, r=radius)
    counts = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
    splits = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum(counts, out=splits[1:])
    if splits[-1] == 0:
        return np.zeros(0, dtype=np.int64), splits
    return np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if len(l)]), splits


def _alpha(src_coords, qry_coords, index, splits):
    counts = splits[1:] - splits[:-1]
    self_c = torch.repeat_interleave(qry_coords, counts, dim=0)
    return segment_softmax(cosine_attention_scores(self_c, src_coords[index]), splits)


def _init_worker(cache, gaot_dir, grid_np, radius):
    _W.clear()
    _W.update(cache=cache, gaot_dir=gaot_dir, grid=np.asarray(grid_np, np.float64), radius=float(radius))
    limit_threads(1)


def _bake_case(cid: str):
    try:
        out_path = osp.join(_W["gaot_dir"], f"{cid}.pt")
        if osp.exists(out_path):
            cached = torch.load(out_path, map_location="cpu", weights_only=False)
            if cached.get("version") == GAOT_BAKE_VERSION:
                return ("ok", cid, cached.get("stats"))
        static = torch.load(osp.join(_W["cache"], "static", f"{cid}.pt"), map_location="cpu",
                            weights_only=True)
        pos = static["norm_pos"].numpy().astype(np.float64)
        grid, r = _W["grid"], _W["radius"]
        n = pos.shape[0]
        enc_index, enc_splits = _ball_csr(cKDTree(pos), grid, r)
        grid_tree = cKDTree(grid)
        dec_index, dec_splits = _ball_csr(grid_tree, pos, r)
        counts = dec_splits[1:] - dec_splits[:-1]
        n_fallback = int((counts == 0).sum())
        if n_fallback:                      # nearest node for a point no node reaches
            empty = np.where(counts == 0)[0]
            _, nearest = grid_tree.query(pos[empty], k=1)
            counts2 = counts.copy()
            counts2[empty] = 1
            new_splits = np.zeros_like(dec_splits)
            np.cumsum(counts2, out=new_splits[1:])
            new_index = np.zeros(int(new_splits[-1]), dtype=np.int64)
            old_rows = np.repeat(np.arange(n), counts)
            within = np.arange(int(dec_splits[-1])) - np.repeat(dec_splits[:-1], counts)
            new_index[new_splits[old_rows] + within] = dec_index
            new_index[new_splits[empty]] = np.atleast_1d(nearest)
            dec_index, dec_splits = new_index, new_splits
        grid_t = torch.from_numpy(grid.astype(np.float32))
        pos_t = torch.from_numpy(pos.astype(np.float32))
        ei, es = torch.from_numpy(enc_index), torch.from_numpy(enc_splits)
        di, ds = torch.from_numpy(dec_index), torch.from_numpy(dec_splits)
        stats = {"n_points": int(n), "enc_edges": int(enc_splits[-1]), "dec_edges": int(dec_splits[-1]),
                 "empty_tokens": int(((enc_splits[1:] - enc_splits[:-1]) == 0).sum()),
                 "dec_fallback_rows": n_fallback}
        atomic_torch_save({
            "version": GAOT_BAKE_VERSION, "case_id": cid,
            "enc_index": ei.to(torch.int32), "enc_splits": es.to(torch.int32),
            "enc_alpha": _alpha(pos_t, grid_t, ei, es).float(),
            "enc_geo": compute_statistical_features(pos_t, grid_t, ei, es).float(),
            "dec_index": di.to(torch.int32), "dec_splits": ds.to(torch.int32),
            "dec_alpha": _alpha(grid_t, pos_t, di, ds).float(),
            "dec_geo": compute_statistical_features(grid_t, pos_t, di, ds).float(),
            "stats": stats}, out_path)
        return ("ok", cid, stats)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", cid, str(exc))


def bake_gaot(cache, names: list[str], grid_h: int, grid_w: int, radius_factor: float,
              workers: int) -> None:
    cache = str(cache)
    gaot_dir = osp.join(cache, "gaot")
    os.makedirs(gaot_dir, exist_ok=True)
    meta = build_grid(cache, grid_h, grid_w, radius_factor)
    atomic_torch_save(meta, osp.join(gaot_dir, "gaot_meta.pt"))
    print(f"[gaot] {meta['subset']}: grid {grid_h}x{grid_w}, radius {meta['radius']:.5f}, "
          f"{len(names)} cases -> {gaot_dir}", flush=True)
    results = run_pool(sorted(names), _bake_case, _init_worker,
                       (cache, gaot_dir, meta["grid"].numpy(), meta["radius"]), workers,
                       f"{meta['subset']}:gaot")
    failed, _ = report(results, "gaot")
    oks = [m for s, _, m in results if s == "ok" and isinstance(m, dict)]
    if oks:
        empty = np.mean([m["empty_tokens"] for m in oks])
        print(f"[gaot] mean {np.mean([m['enc_edges'] for m in oks]):.0f} encoder edges per case; "
              f"{empty:.0f}/{grid_h * grid_w} empty grid nodes on average; "
              f"{sum(m['dec_fallback_rows'] for m in oks)} nearest-node fallback rows", flush=True)
    if failed:
        raise SystemExit(f"[gaot] {len(failed)} cases failed")


def load_gaot_meta(cache) -> dict:
    path = osp.join(str(cache), "gaot", "gaot_meta.pt")
    if not osp.exists(path):
        raise SystemExit(f"{path} missing: GAOT needs its graph bake. Build it once:\n"
                         f"  python -m ibo derive --dataset tandemfoilset --model gaot --subset <subset>")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_gaot_graphs(cache, name: str) -> dict:
    g = torch.load(osp.join(str(cache), "gaot", f"{name}.pt"), map_location="cpu", weights_only=False)
    return {"enc_index": g["enc_index"].long(), "enc_splits": g["enc_splits"].long(),
            "enc_alpha": g["enc_alpha"].float(), "enc_geo": g["enc_geo"].float(),
            "dec_index": g["dec_index"].long(), "dec_splits": g["dec_splits"].long(),
            "dec_alpha": g["dec_alpha"].float(), "dec_geo": g["dec_geo"].float()}

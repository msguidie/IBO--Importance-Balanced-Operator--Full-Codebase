"""SpiderSolver's surface regions on TandemFoilSet (derived cache).

Per airfoil body (front / rear) the FIRST train case's polygon is the
template; spectral clustering partitions it into clusters_per_body
regions; every case's body is aligned to the template by entropic optimal
transport (Sinkhorn, reg 0.8, 25 iterations) followed by a kNN-10
majority vote, and every mesh point inherits the region of its nearest
surface point (front 0..K-1, rear K..2K-1). The median train-split
surface fraction fixes SpiderSolver's density-matched surface quota.

    <cache>/derived/spider_v1/<case>.pt   {vol_cluster_id int8 [N]}
    <cache>/derived/spider_v1/_meta.json  {clusters_per_body, template_case, surf_ratio_median}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..common.pool import limit_threads, run_pool

SPIDER_DERIVED_VERSION = "spider_v1"
_W: dict = {}


def derived_dir(cache: Path) -> Path:
    return Path(cache) / "derived" / SPIDER_DERIVED_VERSION


def sinkhorn_plan(a, b, M, reg: float = 0.8, num_iter: int = 25) -> np.ndarray:
    K = np.exp(-M / reg)
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(num_iter):
        u = a / np.maximum(K @ v, 1e-300)
        v = b / np.maximum(K.T @ u, 1e-300)
    return (u[:, None] * K) * v[None, :]


def assign_body_labels(template, template_labels, points, knn_vote: int = 10) -> np.ndarray:
    """Template -> points transport plan, argmax per point, kNN majority vote."""
    a = np.ones(len(template)) / len(template)
    b = np.ones(len(points)) / len(points)
    M = np.linalg.norm(template[:, None, :] - points[None, :, :], axis=-1)
    labels = template_labels[np.argmax(sinkhorn_plan(a, b, M), axis=0)]
    k = min(knn_vote + 1, len(points))
    _, nbr = cKDTree(points).query(points, k=k)
    nbr = nbr.reshape(len(points), -1)
    neigh = labels[nbr[:, 1:]] if nbr.shape[1] > 1 else labels[nbr]
    out = np.empty(len(points), dtype=np.int64)
    for i in range(len(points)):
        vals, counts = np.unique(neigh[i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def _init_worker(cache, out_dir, templates, template_labels, clusters_per_body):
    _W.clear()
    _W.update(cache=str(cache), out_dir=str(out_dir), templates=templates,
              template_labels=template_labels, K=int(clusters_per_body))
    limit_threads(1)


def _derive_case(cid: str):
    try:
        static = torch.load(os.path.join(_W["cache"], "static", f"{cid}.pt"),
                            map_location="cpu", weights_only=True)
        n = int(static["n_points"])
        ratio = float(static["surf"].bool().sum()) / n
        dst = Path(_W["out_dir"]) / f"{cid}.pt"
        if dst.exists():
            return ("ok", cid, ratio)
        pos = static["query_pos_geo"].numpy().astype(np.float64)
        boundary = static["boundary"].numpy()
        labels_full = np.zeros(n, dtype=np.int64)
        surf_rows_all = []
        for body_i, code in enumerate((5, 6)):
            rows = np.nonzero(boundary == code)[0]
            if len(rows) == 0:
                continue
            labels_full[rows] = assign_body_labels(_W["templates"][code], _W["template_labels"][code],
                                                   pos[rows]) + body_i * _W["K"]
            surf_rows_all.append(rows)
        surf_rows = np.concatenate(surf_rows_all)
        _, nearest = cKDTree(pos[surf_rows]).query(pos, k=1)
        vol_cluster = labels_full[surf_rows[nearest]]
        vol_cluster[surf_rows] = labels_full[surf_rows]
        tmp = f"{dst}.tmp{os.getpid()}"
        torch.save({"version": SPIDER_DERIVED_VERSION, "case_id": cid, "n_points": n,
                    "vol_cluster_id": torch.from_numpy(vol_cluster.astype(np.int8))}, tmp)
        Path(tmp).replace(dst)
        return ("ok", cid, ratio)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", cid, f"{type(exc).__name__}: {exc}")


def build_spider_derived(cache: Path, train: list[str], everyone: list[str],
                         clusters_per_body: int, workers: int) -> None:
    from sklearn.cluster import SpectralClustering

    cache = Path(cache)
    out = derived_dir(cache)
    out.mkdir(parents=True, exist_ok=True)
    template_case = sorted(train)[0]
    tc = torch.load(str(cache / "static" / f"{template_case}.pt"), map_location="cpu", weights_only=True)
    templates, template_labels = {}, {}
    for code, key in ((5, "poly1"), (6, "poly2")):
        pts = tc[key].numpy().astype(np.float64)
        sc = SpectralClustering(n_clusters=min(clusters_per_body, max(2, len(pts) - 1)),
                                affinity="nearest_neighbors", random_state=42)
        templates[code] = pts
        template_labels[code] = sc.fit_predict(pts)
    print(f"[spider-derive] {cache.name}: template {template_case}, {clusters_per_body} clusters "
          f"per body, {len(everyone)} cases", flush=True)
    results = run_pool(sorted(everyone), _derive_case, _init_worker,
                       (cache, out, templates, template_labels, clusters_per_body), workers,
                       f"{cache.name}:spider")
    failed = [r for r in results if r[0] == "failed"]
    if failed:
        raise SystemExit(f"[spider-derive] {len(failed)} cases failed, e.g. {failed[:3]}")
    train_set = set(train)
    ratios = [r[2] for r in results if r[1] in train_set]
    meta = {"version": SPIDER_DERIVED_VERSION, "clusters_per_body": int(clusters_per_body),
            "template_case": template_case, "surf_ratio_median": float(np.median(ratios)),
            "labels": "front 0..K-1, rear K..2K-1; volume rows inherit the nearest surface point"}
    tmp = out / "_meta.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    tmp.replace(out / "_meta.json")
    print(f"[spider-derive] done -> {out} (surf_ratio_median={meta['surf_ratio_median']:.5f})", flush=True)


def load_spider_meta(cache: Path) -> dict:
    p = derived_dir(cache) / "_meta.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: SpiderSolver needs its derived cache. Build it once:\n"
                         f"  python -m ibo derive --dataset tandemfoilset --model spider --subset <subset>")
    with open(p, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("version") != SPIDER_DERIVED_VERSION:
        raise SystemExit(f"{p}: version {meta.get('version')!r} != {SPIDER_DERIVED_VERSION}; rebuild")
    return meta


def load_spider_labels(cache: Path, cid: str, n_points: int) -> torch.Tensor:
    p = derived_dir(cache) / f"{cid}.pt"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `python -m ibo derive --dataset tandemfoilset "
                         f"--model spider --subset <subset>`")
    d = torch.load(str(p), map_location="cpu", weights_only=True)
    if int(d.get("n_points", -1)) != int(n_points):
        raise RuntimeError(f"{p}: n_points mismatch, stale derived cache")
    return d["vol_cluster_id"]

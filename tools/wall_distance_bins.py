#!/usr/bin/env python
"""Error by wall distance for one evaluated UrbanWind run (the layout
ablation of the paper): per-channel relative L2 and R_k in the near / mid /
far thirds of the volume points.

Per test case the volume points are sorted by distance to the nearest
building and split into three equal-count thirds; surface points are left
out. The thirds depend on the geometry alone, so every run gets the same
bins.

  Ux Uy Uz p nut  physical relative L2: the bin's error norm over the truth
                  norm of all volume points of the case, so
                  near^2 + mid^2 + far^2 = all^2; "all" is checked against
                  the run's own stored metrics
  R_k (m)         per volume point, the mean distance to its k nearest
                  tokens (k = the run's ibo.knn)

Values are averaged over the test cases with a saved prediction:

    python tools/wall_distance_bins.py --run-dir runs/urbanwind_ibo_s42

The run needs an evaluation with --save-predictions. Prints only.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from ibo.config import load_config_file  # noqa: E402
from ibo.datasets.urbanwind.estate import (VOL_CHANNEL_NAMES, cache_dir,  # noqa: E402
                                           norm_stats_path)
from ibo.datasets.urbanwind.metrics import inverse_transform_vol, load_norm_stats  # noqa: E402

BINS = ("near", "mid", "far", "all")
RK_CHUNK = 500_000


def load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        return torch.load(str(path), map_location="cpu", weights_only=True)


def thirds(v: np.ndarray, parts) -> list:
    return [float(np.mean(v[p])) for p in parts] + [float(np.mean(v))]


def estate_task(job: dict) -> dict:
    """Every predicted direction of one estate."""
    est, run, stats = job["estate"], job["run"], job["stats"]
    try:
        torch.set_num_threads(1)
        pack = load(Path(run["cache"]) / f"{est}.pt")
        nv = int(pack["N_vol"])
        div = run["div"]
        pos = pack["query_pos_norm"][:nv].double().numpy() * div
        wall = np.abs(pack["query_sdf_norm"][:nv].double().numpy()) * div
        zph = pack["query_z_phys"][:nv].double().numpy()
        parts = np.array_split(np.argsort(wall, kind="stable"), 3)
        edges = [float(wall[p[-1]]) for p in parts[:2]]
        tokens = pack["leaf_pos_norm"].double().numpy() * div
        k = int(min(run["k"], len(tokens)))
        tree = cKDTree(tokens)
        rk = np.empty(nv, dtype=np.float64)
        for lo in range(0, nv, RK_CHUNK):
            d, _ = tree.query(pos[lo:lo + RK_CHUNK], k=k)
            rk[lo:lo + RK_CHUNK] = np.asarray(d).reshape(-1, k).mean(axis=1)
        rk = thirds(rk, parts)

        eval_dir = Path(run["eval_dir"])
        recs = []
        for d in job["dirs"]:
            case = f"{est}_{d}"
            true = inverse_transform_vol(pack[f"target_vol_{d}"][:nv].float().numpy(), zph, stats)
            pz = load(eval_dir / "predictions" / f"{case}.pt")["pred_vol_z"].float().numpy()
            if pz.shape != true.shape:
                raise RuntimeError(f"{case}: prediction {pz.shape} against truth {true.shape}")
            e2 = (inverse_transform_vol(pz, zph, stats) - true) ** 2
            den = np.maximum(np.sqrt((true ** 2).sum(axis=0)), 1e-12)
            rl2 = {ch: [float(np.sqrt(e2[p, i].sum())) / den[i] for p in parts]
                   + [float(np.sqrt(e2[:, i].sum())) / den[i]] for i, ch in enumerate(VOL_CHANNEL_NAMES)}
            stored = json.loads((eval_dir / "metrics" / f"{case}.json").read_text(encoding="utf-8"))
            dev = max((abs(rl2[ch][3] - float(stored[ch])) / abs(float(stored[ch]))
                       for ch in VOL_CHANNEL_NAMES if float(stored[ch]) != 0.0), default=0.0)
            recs.append({"rl2": rl2, "rk": rk, "edges": edges, "dev": dev})
        return {"ok": True, "estate": est, "recs": recs}
    except Exception:
        return {"ok": False, "estate": est, "err": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="an evaluated run directory (config_used.yaml, eval_<split>/)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--cache-root", default=None, help="override data.cache_root of the run config")
    ap.add_argument("--workers", type=int, default=None, help="estates in parallel (default: CPU count)")
    args = ap.parse_args()

    rd = Path(args.run_dir).expanduser().resolve()
    cfg = load_config_file(rd / "config_used.yaml")
    if cfg.run.dataset != "urbanwind":
        raise SystemExit("this tool reads UrbanWind runs")
    if args.cache_root:
        cfg.data.cache_root = str(Path(args.cache_root).expanduser())
    eval_dir = rd / f"eval_{args.split}"
    run = {"eval_dir": str(eval_dir), "cache": str(cache_dir(cfg)), "k": int(cfg.ibo.knn),
           "div": float(cfg.urbanwind.coord_divisor)}
    stats = load_norm_stats(norm_stats_path(cfg))
    pdir = eval_dir / "predictions"
    cases = sorted(p.stem for p in pdir.glob("*.pt") if (eval_dir / "metrics" / f"{p.stem}.json").exists())
    if not cases:
        raise SystemExit(f"{pdir}: no saved predictions (evaluate with --save-predictions)")
    by_est: dict = {}
    for c in cases:
        e, _, d = c.rpartition("_")
        by_est.setdefault(e, []).append(d)
    n_workers = max(1, min(args.workers or os.cpu_count() or 1, len(by_est)))
    print(f"[bins] run {rd.name}: {cfg.run.model}, layout {cfg.run.layout}, k={run['k']}, "
          f"cache {Path(run['cache']).name}; {len(cases)} cases, {len(by_est)} estates, "
          f"{n_workers} workers", flush=True)

    jobs = [{"estate": e, "dirs": sorted(ds), "run": run, "stats": stats} for e, ds in sorted(by_est.items())]
    recs, fails = [], []
    t0 = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for n, res in enumerate(pool.imap_unordered(estate_task, jobs), 1):
            if res["ok"]:
                recs += res["recs"]
            else:
                fails.append(res["estate"])
                print(f"[FAIL] {res['estate']}\n{res['err']}", flush=True)
            if n % 10 == 0 or n == len(jobs):
                print(f"[bins] {n}/{len(jobs)} estates ({time.time() - t0:.0f} s)", flush=True)
    if not recs:
        raise SystemExit("every estate failed (tracebacks above)")

    e0 = float(np.mean([r["edges"][0] for r in recs]))
    e1 = float(np.mean([r["edges"][1] for r in recs]))
    print(f"\n[bins] {rd.name}: {len(recs)} cases; thirds split at {e0:.2f} m / {e1:.2f} m "
          f"to the nearest building (case mean)")
    print(f"  {'':8s}" + "".join(f"{b:>10s}" for b in BINS))
    for ch in VOL_CHANNEL_NAMES:
        v = np.mean([r["rl2"][ch] for r in recs], axis=0)
        print(f"  {ch:8s}" + "".join(f"{x:10.4f}" for x in v))
    v = np.mean([r["rk"] for r in recs], axis=0)
    print(f"  {'Rk (m)':8s}" + "".join(f"{x:10.3f}" for x in v))
    print(f"  check: 'all' vs the run's own metrics, max rel dev {max(r['dev'] for r in recs):.1e}")
    if fails:
        print(f"  FAILED estates: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

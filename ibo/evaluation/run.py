"""Evaluation of a finished run, driven by its config_used.yaml.

Every evaluation point of every case in the chosen split is predicted
exactly once (the recipe's infer_case contract); the dataset computes its
own protocol metrics per case and aggregates them. Optional: predictions on
disk, the mean per-case latency on one GPU, and the metered FLOPs of one
full-field prediction. Per-case metrics are resumable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..datasets import get_dataset
from ..registry import get_recipe
from ..training.checkpoint import load_swa_weights
from ..training.ddp import all_reduce_sum, barrier, cleanup_ddp, init_ddp
from ..training.swa import atomic_save
from .flops import FlopMeter


def _mean_over_draws(draws: list[dict]) -> dict:
    """Element-wise mean of per-draw metric dicts (nested, numeric leaves)."""
    out = {}
    for k, v in draws[0].items():
        if isinstance(v, dict):
            out[k] = _mean_over_draws([d[k] for d in draws])
        elif isinstance(v, (int, float)):
            out[k] = float(np.mean([d[k] for d in draws]))
        else:
            out[k] = v
    return out


def evaluate(cfg: Config, run_dir: str | Path) -> None:
    rank, world, local = init_ddp()
    run_dir = Path(run_dir)
    ev = cfg.evaluation
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    dataset = get_dataset(cfg.run.dataset)
    recipe = get_recipe(cfg.run.model)
    spec = dataset.spec
    out_dir = run_dir / f"eval_{ev.split}"
    names = dataset.splits(cfg)[ev.split]
    my_names = names[rank::world]
    if rank == 0:
        (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
        if ev.save_predictions:
            (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
        print(f"[eval] {cfg.run.dataset} x {cfg.run.model} split={ev.split}: "
              f"{len(names)} cases on {world} rank(s)", flush=True)
    barrier()

    model = recipe.build_model(cfg, spec).to(device)
    swa_meta = load_swa_weights(run_dir / "swa_model.pt", model)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_draws = int(recipe.infer_draws(cfg))

    flops = None
    if ev.flops and rank == 0 and my_names:
        # metered eagerly, before any compilation, on this rank's first case
        for _, case in recipe.stream_cases(cfg, dataset, my_names[:1]):
            with FlopMeter(model) as meter, torch.inference_mode():
                recipe.infer_case(model, case, cfg, device, use_cuda)
            manual = recipe.manual_infer_flops(cfg, spec, int(case["n_vol"]), int(case["n_surf"]))
            flops = {"case": case["name"], "n_vol": int(case["n_vol"]),
                     "n_surf": int(case["n_surf"]), **meter.breakdown()}
            flops["manual"] = manual
            flops["total"] = meter.linear + meter.norm + meter.activation + sum(manual.values())
            del case
    infer_label = recipe.prepare_infer_model(model, cfg, use_cuda)

    for name, case in recipe.stream_cases(cfg, dataset, my_names):
        mpath = out_dir / "metrics" / f"{name}.json"
        if mpath.exists():
            del case
            continue
        draws = []
        for d in range(n_draws):
            recipe.infer_draw = d
            with torch.inference_mode():
                pv, ps = recipe.infer_case(model, case, cfg, device, use_cuda)
            assert pv.shape == (int(case["n_vol"]), len(spec.vol_channels)) \
                and ps.shape == (int(case["n_surf"]), len(spec.surf_channels)), \
                f"{name}: producer returned {pv.shape} / {ps.shape}"
            assert np.isfinite(pv).all() and np.isfinite(ps).all(), f"{name}: non-finite predictions"
            draws.append(dataset.case_metrics(cfg, case, pv, ps))
            if d == 0 and ev.save_predictions:
                atomic_save({"name": name, "pred_vol_z": torch.from_numpy(pv),
                             "pred_surf_z": torch.from_numpy(ps)},
                            out_dir / "predictions" / f"{name}.pt")
        recipe.infer_draw = 0
        metrics = _mean_over_draws(draws) if n_draws > 1 else draws[0]
        metrics["name"] = name
        if n_draws > 1:
            metrics["draws"] = draws
        tmp = mpath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=1)
        tmp.replace(mpath)
        print(f"[eval rank {rank}] {name}: "
              + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()
                         if isinstance(v, (int, float))), flush=True)
        del case
    barrier()

    latency = None
    if ev.latency:
        for _, case in recipe.stream_cases(cfg, dataset, my_names[:1]):  # untimed warm-up
            with torch.inference_mode():
                recipe.infer_case(model, case, cfg, device, use_cuda)
            del case
        t_inf = 0.0
        for _, case in recipe.stream_cases(cfg, dataset, my_names):
            if use_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                recipe.infer_case(model, case, cfg, device, use_cuda)
            if use_cuda:
                torch.cuda.synchronize(device)
            t_inf += time.perf_counter() - t0
            del case
        wall = all_reduce_sum(torch.tensor(float(t_inf), dtype=torch.float64, device=device))
        latency = {"device": torch.cuda.get_device_name(device) if use_cuda else "cpu",
                   "inference": infer_label, "n_cases": len(names),
                   "s_per_case": float(wall.item()) / max(len(names), 1)}
    barrier()

    if rank == 0:
        rows = []
        for name in names:
            with open(out_dir / "metrics" / f"{name}.json", encoding="utf-8") as f:
                rows.append(json.load(f))
        result = {"dataset": cfg.run.dataset, "model": cfg.run.model,
                  "subset": cfg.run.subset, "layout": cfg.run.layout,
                  "seed": cfg.run.seed, "split": ev.split, "n_cases": len(rows),
                  "params": int(n_params), "swa": swa_meta,
                  "metrics": dataset.aggregate(cfg, rows)}
        if latency is not None:
            result["latency"] = latency
        if flops is not None:
            result["flops"] = flops
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        line = " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                          for k, v in result["metrics"].items())
        print(f"[eval] {cfg.run.model}: {line}", flush=True)
        if latency is not None:
            print(f"[eval] latency {latency['s_per_case']:.3f} s/case on {latency['device']}",
                  flush=True)
        if flops is not None:
            print(f"[eval] FLOPs per case {flops['total'] / 1e12:.2f} TFLOP "
                  f"({flops['case']})", flush=True)
    barrier()
    cleanup_ddp()

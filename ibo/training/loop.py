"""The trainer: one loop for every dataset and model.

Everything model-specific arrives through the Recipe and everything
dataset-specific through the Dataset. The loop owns what is identical
across runs: DDP (compile first, then wrap), bf16 autocast without a
GradScaler, gradient clipping, gradient accumulation to the paper's
effective batch, the scheduler stepped at the recipe's granularity, one
checkpoint per epoch, resume, the walltime stop at an epoch boundary, and
the disk-based SWA over the last swa_fraction of the epochs. A finished run
is three files: swa_model.pt, config_used.yaml, training_log.jsonl.
"""
from __future__ import annotations

import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import Config, resolve_effective_batch, save_config
from ..recipes.base import Recipe, cpu_defloat_bf16
from ..utils.env import quiet_inductor_perf_hints
from ..utils.seed import seed_everything, train_seed_salt
from .checkpoint import (find_latest_trainer_state, load_full_checkpoint,
                         load_state_strict, save_epoch)
from .data import CudaPrefetcher, build_dataloader
from .ddp import all_reduce_sum, barrier, cleanup_ddp, init_ddp, is_distributed
from .swa import atomic_save, average_epoch_weights


def train(cfg: Config, recipe: Recipe, dataset, auto_resume: bool = False,
          max_hours: float | None = None) -> None:
    t_wall = time.time()
    rank, world, local = init_ddp()
    resolve_effective_batch(cfg, world)
    run_dir = Path(cfg.data.run_dir)
    swa_path = run_dir / "swa_model.pt"
    if swa_path.exists():
        if auto_resume:
            if rank == 0:
                print(f"[train] {run_dir} is complete, nothing to do", flush=True)
            cleanup_ddp()
            return
        raise SystemExit(f"{run_dir} already holds swa_model.pt; use a fresh "
                         f"--run-dir or --auto-resume")

    seed_everything(cfg.run.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    tc = cfg.training
    accum = cfg.run.grad_accum
    spec = dataset.spec

    names = dataset.splits(cfg)["train"]
    shard = dataset.shard(names, rank, world)
    recipe.ensure_derived(cfg, dataset, shard)
    train_ds, collate_fn, residency = recipe.build_train_dataset(
        cfg, dataset, shard, train_seed_salt(cfg.run.seed))
    micro_per_epoch = (len(train_ds) // tc.cases_per_gpu) // accum * accum
    opt_steps_per_epoch = micro_per_epoch // accum
    assert opt_steps_per_epoch > 0, "shard too small for one optimizer step"
    if rank == 0:
        print(f"[train] {cfg.run.dataset} x {cfg.run.model}"
              + (f" ({cfg.run.subset})" if cfg.run.subset else "")
              + f" layout={cfg.run.layout} seed={cfg.run.seed}: "
              f"{len(shard)} train cases/rank, {opt_steps_per_epoch} optimizer "
              f"steps/epoch x {tc.epochs} epochs, effective batch "
              f"{cfg.run.effective_batch} = {tc.cases_per_gpu} x world {world} "
              f"x accum {accum}", flush=True)

    model = recipe.build_model(cfg, spec).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[train] {n_params / 1e6:.2f}M trainable parameters", flush=True)

    quiet_inductor_perf_hints()
    compile_mode, fullgraph = recipe.compile_plan(cfg)
    if compile_mode == "reduce-overhead" and tc.activation_checkpointing:
        # CUDA-Graph capture with checkpoint recompute corrupts backward
        compile_mode = "default"
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.optimize_ddp = not recipe.disable_ddp_optimizer
    if compile_mode and use_cuda:
        model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph,
                              dynamic=False)
    if is_distributed():
        model = DDP(model, device_ids=[local] if use_cuda else None,
                    static_graph=True, gradient_as_bucket_view=True,
                    find_unused_parameters=False, broadcast_buffers=False)

    optimizer, scheduler, sched_gran = recipe.build_optimizer(cfg, model, opt_steps_per_epoch)
    assert sched_gran in {"step", "epoch"}
    swa_start = tc.swa_start
    identity = {"dataset": cfg.run.dataset, "model": cfg.run.model,
                "subset": cfg.run.subset, "layout": cfg.run.layout,
                "world_size": world, "cases_per_gpu": tc.cases_per_gpu,
                "grad_accum": accum}

    start_epoch = 1
    if auto_resume:
        found = find_latest_trainer_state(run_dir)
        if found is not None:
            last = load_full_checkpoint(found[1], model, optimizer, scheduler,
                                        run_dir, identity)
            expected = last * opt_steps_per_epoch if sched_gran == "step" else last
            if int(scheduler.last_epoch) != expected:
                raise RuntimeError(f"resume scheduler position {scheduler.last_epoch} "
                                   f"!= expected {expected}; schedule geometry drifted")
            start_epoch = last + 1
            if rank == 0:
                print(f"[train] resumed after epoch {last}", flush=True)

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config_used.yaml")
    log_path = run_dir / "training_log.jsonl"
    loader, sampler = build_dataloader(train_ds, tc.cases_per_gpu, tc.num_workers,
                                       tc.prefetch_factor, cfg.run.seed, collate_fn,
                                       residency)

    def autocast_ctx():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda)

    def batches(epoch: int):
        sampler.set_epoch(epoch)
        it = CudaPrefetcher(loader, device) if use_cuda else iter(loader)
        for batch in it:
            if not use_cuda:
                batch = cpu_defloat_bf16(recipe.batch_to_device(batch, device))
            yield batch

    if use_cuda and start_epoch <= tc.epochs:
        # compile warm-up on one batch; its gradients are discarded
        warm = next(batches(0))
        with autocast_ctx():
            loss, _ = recipe.forward_loss(model, warm, cfg, spec)
        loss.backward()
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize(device)
        del warm, loss
        barrier()

    model.train()
    n_vol_ch, n_surf_ch = len(spec.vol_channels), len(spec.surf_channels)
    stopped_early = False
    for epoch in range(start_epoch, tc.epochs + 1):
        t0 = time.time()
        vol_sum = torch.zeros(n_vol_ch, dtype=torch.float64, device=device)
        surf_sum = torch.zeros(n_surf_ch, dtype=torch.float64, device=device)
        n_vol = torch.zeros((), dtype=torch.float64, device=device)
        n_surf = torch.zeros((), dtype=torch.float64, device=device)
        gsum = torch.zeros((), device=device)
        gmax = torch.zeros((), device=device)
        optimizer.zero_grad(set_to_none=True)
        it = batches(epoch)
        for micro in range(micro_per_epoch):
            boundary = (micro + 1) % accum == 0
            batch = next(it)
            sync = model.no_sync() if (is_distributed() and not boundary) else nullcontext()
            with sync:
                with autocast_ctx():
                    loss, br = recipe.forward_loss(model, batch, cfg, spec)
                (loss / accum).backward()
            if boundary:
                # a non-finite gradient aborts before the update reaches the weights
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm,
                                                 error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if sched_gran == "step":
                    scheduler.step()
                gsum += gnorm.detach()
                gmax = torch.maximum(gmax, gnorm.detach())
            vol_sum += br["vol_loss_sum"].double()
            surf_sum += br["surf_loss_sum"].double()
            n_vol += br["n_vol"].double()
            n_surf += br["n_surf"].double()
        del it
        if sched_gran == "epoch":
            scheduler.step()
        for t in (vol_sum, surf_sum, n_vol, n_surf):
            all_reduce_sum(t)
        bad = torch.zeros((), dtype=torch.float64, device=device)
        if not (bool(torch.isfinite(vol_sum).all()) and bool(torch.isfinite(surf_sum).all())):
            bad += 1.0
        all_reduce_sum(bad)
        if float(bad) > 0.0:
            raise RuntimeError(f"epoch {epoch}: non-finite training loss; stopping "
                               f"before this epoch is checkpointed")

        if rank == 0:
            save_epoch(run_dir, model, optimizer, scheduler, epoch, identity,
                       tc.rolling_keep, swa_start)
            vol_per = (vol_sum / n_vol.clamp(min=1)).cpu()
            surf_per = (surf_sum / n_surf.clamp(min=1)).cpu()
            lr_now = optimizer.param_groups[0]["lr"]
            dur = time.time() - t0
            print(f"epoch {epoch}: vol={float(vol_per.mean()):.6f} "
                  f"surf={float(surf_per.mean()):.6f} "
                  f"grad={float(gsum) / opt_steps_per_epoch:.3f} lr={lr_now:.3g} "
                  f"({dur:.1f}s)", flush=True)
            entry = {"epoch": epoch, "lr": lr_now,
                     "train_loss_vol": {c: float(vol_per[i]) for i, c in enumerate(spec.vol_channels)},
                     "train_loss_surf": {c: float(surf_per[i]) for i, c in enumerate(spec.surf_channels)},
                     "grad_norm_avg": float(gsum) / opt_steps_per_epoch,
                     "grad_norm_max": float(gmax), "duration_s": dur}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        if max_hours:
            stop = torch.zeros((), dtype=torch.float64, device=device)
            if rank == 0:
                elapsed = time.time() - t_wall
                if elapsed + 1.25 * (time.time() - t0) + 300.0 >= float(max_hours) * 3600.0:
                    stop += 1.0
                    print(f"[train] walltime budget reached after epoch {epoch}; "
                          f"relaunch with --auto-resume", flush=True)
            all_reduce_sum(stop)
            if float(stop) > 0.0:
                stopped_early = True
                break

    barrier()
    del optimizer, scheduler, loader
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    if stopped_early:
        cleanup_ddp()
        return

    if rank == 0:
        _finish_swa(cfg, recipe, spec, run_dir, swa_start, train_ds, collate_fn,
                    device, log_path)
    barrier()
    cleanup_ddp()


def _finish_swa(cfg, recipe, spec, run_dir: Path, swa_start: int, train_ds,
                collate_fn, device, log_path: Path) -> None:
    tc = cfg.training
    weights_dir = run_dir / "epoch_weights"
    averaged, count = average_epoch_weights(weights_dir, swa_start, tc.epochs)
    fresh = recipe.build_model(cfg, spec)
    load_state_strict(fresh, averaged)
    bn = [m for m in fresh.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if bn:
        # BatchNorm running statistics are recomputed on real batches
        # (torch.optim.swa_utils.update_bn semantics) rather than averaged
        _bn_recalibrate(fresh.to(device), recipe, cfg, spec, train_ds, collate_fn, device)
        fresh = fresh.to("cpu")
        averaged = {k: (v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu())
                    for k, v in fresh.state_dict().items()}
    payload = {"format": "swa_model", "model_state_dict": averaged,
               "swa_window": [int(swa_start), int(tc.epochs)], "swa_count": count,
               "bn_recalibrated": bool(bn)}
    atomic_save(payload, run_dir / "swa_model.pt")
    verify = recipe.build_model(cfg, spec)
    load_state_strict(verify, torch.load(str(run_dir / "swa_model.pt"), map_location="cpu",
                                         weights_only=True)["model_state_dict"])
    del fresh, verify
    for p in list(weights_dir.glob("epoch_*.pt")) + list(run_dir.glob("full_epoch*.pt")):
        p.unlink(missing_ok=True)
    try:
        weights_dir.rmdir()
    except OSError:
        pass
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "training_complete", "epochs": tc.epochs,
                            "swa_window": [swa_start, tc.epochs],
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(f"[train] complete: swa_model.pt averages epochs [{swa_start}, {tc.epochs}]",
          flush=True)


def _bn_recalibrate(model, recipe, cfg, spec, train_ds, collate_fn, device) -> None:
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = {}
    for m in bns:
        m.reset_running_stats()
        momenta[m] = m.momentum
        m.momentum = None
    model.train()
    use_cuda = device.type == "cuda"
    n = len(train_ds)
    bs = cfg.training.cases_per_gpu
    with torch.no_grad():
        for i in range(n // bs):
            batch = collate_fn([train_ds[((i * bs + j) % n, cfg.training.epochs + 1)]
                                for j in range(bs)])
            batch = recipe.batch_to_device(batch, device)
            if not use_cuda:
                batch = cpu_defloat_bf16(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                recipe.forward_loss(model, batch, cfg, spec)
    for m in bns:
        m.momentum = momenta[m]
    model.eval()

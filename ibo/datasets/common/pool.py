"""Process-pool helpers of the preprocessing pipelines. A worker returns a
record ("ok" | "failed" | "skipped", name, message, ...)."""
from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

import torch
from tqdm import tqdm


def limit_threads(n: int = 1) -> None:
    """One BLAS thread per worker process (the pool is the parallelism)."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS"):
        os.environ[key] = str(max(int(n), 1))
    torch.set_num_threads(max(int(n), 1))


def atomic_torch_save(payload, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run_pool(items: list, worker, initializer, initargs: tuple, workers: int,
             desc: str, ordered: bool = False) -> list:
    """worker(item) over `items` in `workers` processes (in-process when
    workers <= 1); `ordered` keeps the item order (bit-reproducible
    floating-point merges)."""
    items = list(items)
    workers = max(1, min(int(workers), len(items)))
    if workers <= 1:
        initializer(*initargs)
        return [worker(c) for c in tqdm(items, desc=desc)]
    with Pool(processes=workers, initializer=initializer, initargs=initargs) as pool:
        it = pool.imap(worker, items) if ordered else pool.imap_unordered(worker, items)
        return list(tqdm(it, total=len(items), desc=desc))


def report(results: list, phase: str) -> tuple[list, list]:
    """Print and return (failed, skipped) records of a phase."""
    failed = [(r[1], r[2]) for r in results if r[0] == "failed"]
    skipped = [(r[1], r[2]) for r in results if r[0] == "skipped"]
    for label, rows in (("FAILED", failed), ("SKIPPED (data quality)", skipped)):
        if rows:
            print(f"[{phase}] {len(rows)} cases {label}:", flush=True)
            for name, msg in rows[:20]:
                print(f"  {name}: {msg}", flush=True)
    return failed, skipped

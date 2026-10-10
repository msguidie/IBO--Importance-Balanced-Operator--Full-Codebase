"""Process environment hygiene. Import this module BEFORE importing torch:
the allocator / NCCL variables only take effect if set before CUDA init."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")


def limit_worker_threads():
    """Single-threaded math libraries inside multiprocessing pool workers."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def quiet_inductor_perf_hints() -> None:
    """Silence inductor's per-graph "skipping cudagraphs" hints: the IBO
    decoder's neighbour gather has an index_put(accumulate=True) backward, so
    inductor declines CUDA-Graph capture of the training graph. That affects
    speed only, and the message repeats per subgraph per rank."""
    import logging
    try:
        import torch._logging
        torch._logging.set_logs(perf_hints=logging.ERROR)
        return
    except Exception:
        pass
    for name in ("torch._inductor.utils", "torch._inductor.cudagraph_trees",
                 "torch._inductor.compile_fx"):
        logging.getLogger(name).setLevel(logging.ERROR)

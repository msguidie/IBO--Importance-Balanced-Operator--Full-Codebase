"""segment_csr — CSR-rowwise sum/mean reduction.

[PORT] The upstream file dispatches to torch_scatter when installed and
falls back to a per-row Python loop otherwise (their comment: the native
path exists "for CPU-only GitHub CI runners"). Neither is acceptable here:
torch_scatter is not a dependency here, and the loop is O(rows) Python.
This version is a single native implementation built on
``Tensor.index_add_`` — one fused CUDA kernel, no dependency — that
matches torch_scatter.segment_csr(reduce='sum'|'mean') semantics
including empty rows (sum 0; mean 0, matching torch_scatter's
zero-initialised output rather than NaN).

Only the unbatched [E, C] layout is implemented: the tandem port always
flattens the batch into a disjoint-union graph first (see
model.merge_csr), so the batched [B, E, C] path of the upstream
file is dead code here and deliberately dropped — calling with a 3-D src
raises instead of silently mis-reducing.
"""
import torch


def _row_index_from_splits(indptr: torch.Tensor) -> torch.Tensor:
    counts = indptr[1:] - indptr[:-1]
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=indptr.device), counts)


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str,
                use_scatter: bool = True):  # noqa: ARG001 (kept for API compat)
    if src.ndim not in (1, 2):
        raise ValueError(
            f"segment_csr: src must be [E] or [E, C], got shape "
            f"{tuple(src.shape)} — batch by disjoint-union merge, not by a "
            f"leading batch dim")
    if reduce not in ("sum", "mean", "max"):
        raise ValueError(f"reduce must be sum/mean/max, got {reduce!r}")

    n_rows = indptr.numel() - 1
    row = _row_index_from_splits(indptr)
    out_shape = (n_rows,) if src.ndim == 1 else (n_rows, src.shape[1])

    if reduce == "max":
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        idx = row if src.ndim == 1 else row.unsqueeze(-1).expand_as(src)
        out.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
        # empty rows keep their zero initialisation, matching torch_scatter
        return out

    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.index_add_(0, row, src)
    if reduce == "mean":
        counts = (indptr[1:] - indptr[:-1]).clamp(min=1).to(src.dtype)
        out = out / (counts if src.ndim == 1 else counts.unsqueeze(-1))
    return out

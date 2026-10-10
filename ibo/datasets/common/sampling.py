"""Deterministic per-(case, epoch) row draws shared by the 2-D datasets."""
from __future__ import annotations

import torch

from ...utils.seed import per_case_epoch_seed


def uniform_rows(key: str, n: int, n_query: int, epoch: int) -> torch.Tensor:
    """n_query of n rows, uniformly without replacement, sorted."""
    if n < n_query:
        raise RuntimeError(f"{key}: N={n} < n_query={n_query}")
    g = torch.Generator()
    g.manual_seed(per_case_epoch_seed(key, epoch))
    return torch.randperm(n, generator=g)[:n_query].sort().values


def stratified_rows(key: str, vol_rows: torch.Tensor, surf_rows: torch.Tensor, n_vol: int,
                    n_surf: int, epoch: int, tag: str) -> tuple[torch.Tensor, torch.Tensor]:
    """n_vol volume rows and n_surf surface rows, drawn independently
    (seeds key + tag + "_vol" / "_surf")."""
    if vol_rows.shape[0] < n_vol or surf_rows.shape[0] < n_surf:
        raise RuntimeError(f"{key}: N_vol={vol_rows.shape[0]} N_surf={surf_rows.shape[0]} "
                           f"below the draw ({n_vol} + {n_surf})")
    out = []
    for rows, n_s, side in ((vol_rows, n_vol, "_vol"), (surf_rows, n_surf, "_surf")):
        g = torch.Generator()
        g.manual_seed(per_case_epoch_seed(key + tag + side, epoch))
        out.append(rows[torch.randperm(rows.shape[0], generator=g)[:n_s].sort().values])
    return out[0], out[1]

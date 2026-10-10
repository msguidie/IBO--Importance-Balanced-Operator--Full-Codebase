"""Geometry of closed 2-D polygons (airfoil sections): signed distance and
its gradient, outward vertex normals, the directional distance profile
(DID: mean capped distance to the polygon in angular sectors) and a kNN
circular-variance curvature proxy."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree


def polygon_sdf_and_grad(query: torch.Tensor, polygon: torch.Tensor,
                         chunk_size: int = 20000) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed distance (negative inside, winding-number test) and its unit
    gradient at every query point; query [N, 2], polygon [P, 2]."""
    assert query.dim() == 2 and query.shape[-1] == 2
    assert polygon.dim() == 2 and polygon.shape[-1] == 2
    a = polygon
    seg = torch.roll(polygon, -1, dims=0) - a
    seg_len_sq = (seg ** 2).sum(-1) + 1e-12
    N = query.shape[0]
    sdf_out = torch.empty(N, dtype=query.dtype, device=query.device)
    grad_out = torch.empty(N, 2, dtype=query.dtype, device=query.device)
    for start in range(0, N, chunk_size):
        q = query[start:start + chunk_size]
        B = q.shape[0]
        t = ((q[:, None, :] - a[None]) * seg[None]).sum(-1) / seg_len_sq[None]
        closest_all = a[None] + t.clamp(0.0, 1.0)[..., None] * seg[None]
        dist_sq = ((q[:, None, :] - closest_all) ** 2).sum(-1)
        min_dist_sq, min_idx = dist_sq.min(dim=1)
        min_dist = min_dist_sq.sqrt()
        closest = closest_all[torch.arange(B, device=q.device), min_idx]
        a_q = polygon - q[:, None, :]
        b_q = torch.roll(polygon, -1, dims=0) - q[:, None, :]
        cross = a_q[..., 0] * b_q[..., 1] - a_q[..., 1] * b_q[..., 0]
        dot = a_q[..., 0] * b_q[..., 0] + a_q[..., 1] * b_q[..., 1]
        inside = (torch.atan2(cross, dot).sum(dim=1) / (2.0 * math.pi)).round().long().ne(0)
        direction = (q - closest) / min_dist.clamp(min=1e-7)[..., None]
        sdf_out[start:start + B] = torch.where(inside, -min_dist, min_dist)
        grad_out[start:start + B] = torch.where(inside[..., None], -direction, direction)
    return sdf_out, grad_out


def polygon_vertex_normals(polygon: torch.Tensor) -> torch.Tensor:
    """Outward unit normals at the vertices (tangent from the neighbouring
    vertices; the sign is fixed by probing a few vertices)."""
    t = torch.roll(polygon, -1, dims=0) - torch.roll(polygon, 1, dims=0)
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = torch.stack([t[:, 1], -t[:, 0]], dim=1)
    P = polygon.shape[0]
    probe_ids = torch.linspace(0, P - 1, steps=min(8, P)).long()
    s, _ = polygon_sdf_and_grad(polygon[probe_ids] + 1e-3 * n[probe_ids], polygon)
    return -n if (s < 0).float().mean().item() > 0.5 else n


def compute_knn_curvature(query_pos, ref_pos, ref_grad, k: int = 8, tree=None) -> np.ndarray:
    """Circular variance of the SDF-gradient angles over the k nearest
    reference points: near 0 at a flat wall, growing near a leading edge or
    the gap between two bodies. A query counts itself when it belongs to
    the reference set."""
    query_pos = np.asarray(query_pos, dtype=np.float32)
    ref_pos = np.asarray(ref_pos, dtype=np.float32)
    ref_grad = np.asarray(ref_grad, dtype=np.float32)
    if query_pos.shape[0] == 0 or ref_pos.shape[0] == 0:
        return np.zeros(query_pos.shape[0], dtype=np.float32)
    tree = cKDTree(ref_pos) if tree is None else tree
    _, nbr = tree.query(query_pos, k=min(int(k), ref_pos.shape[0]))
    nbr = nbr.reshape(query_pos.shape[0], -1)
    ang = np.arctan2(ref_grad[:, 1], ref_grad[:, 0])[nbr]
    R = np.sqrt(np.sin(ang).mean(axis=1) ** 2 + np.cos(ang).mean(axis=1) ** 2)
    return (1.0 - R).astype(np.float32)


def compute_did(query_pos, polygon, n_bins: int = 8, d_max: float = 5.0,
                chunk_size: int = 4096) -> np.ndarray:
    """[N, n_bins] directional distance profile: per angular sector the mean
    capped distance to the polygon vertices, blended toward d_max when the
    sector is only partly covered."""
    query_pos = np.asarray(query_pos, dtype=np.float32)
    polygon = np.asarray(polygon, dtype=np.float32)
    N = query_pos.shape[0]
    bin_width = (2.0 * math.pi) / n_bins
    out = np.full((N, n_bins), d_max, dtype=np.float32)
    if N == 0 or polygon.shape[0] == 0:
        return out
    for start in range(0, N, chunk_size):
        q = query_pos[start:start + chunk_size]
        rel = polygon[None, :, :] - q[:, None, :]
        dists_capped = np.minimum(np.linalg.norm(rel, axis=-1), d_max)
        angles = np.mod(np.arctan2(rel[..., 1], rel[..., 0]), 2.0 * math.pi)
        bin_idx = np.clip(np.floor(angles / bin_width).astype(np.int64), 0, n_bins - 1)
        rel_in_bin = angles - bin_idx.astype(np.float64) * bin_width
        for k in range(n_bins):
            mask = bin_idx == k
            count = mask.sum(axis=1)
            has = count > 0
            if not has.any():
                continue
            mean_d = (dists_capped * mask).sum(axis=1) / np.maximum(count, 1)
            span = np.where(mask, rel_in_bin, -np.inf).max(axis=1) \
                - np.where(mask, rel_in_bin, np.inf).min(axis=1)
            w = np.clip(np.where(has, span, 0.0) / bin_width, 0.0, 1.0)
            val = w * mean_d + (1.0 - w) * d_max
            out[start:start + q.shape[0], k] = np.where(has, val, d_max).astype(np.float32)
    return out


def combine_did(did1: np.ndarray, did2: np.ndarray, d_max: float = 5.0) -> np.ndarray:
    """Two-body profile: the two profiles blended with their deviation from
    d_max as weights."""
    w1, w2 = d_max - did1, d_max - did2
    s = w1 + w2
    out = np.full_like(did1, d_max)
    nz = s > 1e-12
    out[nz] = (w1[nz] * did1[nz] + w2[nz] * did2[nz]) / s[nz]
    return out

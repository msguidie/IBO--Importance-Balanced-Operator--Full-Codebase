"""Equal-count KD-tree (the paper's importance-balanced layout with unit
weights, as used on the 2-D benchmarks) and the fixed lattice of the
layout ablation.

The tree partitions point INDICES: at every level each node is split at
its median along its widest axis by one stable sort, so ties on the split
axis never produce an empty leaf and leaf sizes differ by at most one.
Membership is the (perm, leaf_offsets) pair; the recorded split planes are
for figures only, nothing descends them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class KDTree:
    depth: int
    split_axes: np.ndarray    # [2^depth - 1] int8
    split_values: np.ndarray  # [2^depth - 1] float64
    perm: np.ndarray          # [N] int64, original indices in leaf order
    leaf_offsets: np.ndarray  # [2^depth + 1] int64

    @property
    def n_leaves(self) -> int:
        return 2 ** self.depth


def build_equal_kdtree(points: np.ndarray, depth: int) -> KDTree:
    """points [N, D] with N >= 2^depth."""
    points = np.asarray(points, dtype=np.float64)
    N, n_dims = points.shape
    assert N >= 2 ** depth, f"KD-tree needs N >= 2^depth points ({N} < {2 ** depth})"
    n_internal = 2 ** depth - 1
    split_axes = np.zeros(n_internal, dtype=np.int8)
    split_values = np.zeros(n_internal, dtype=np.float64)
    perm = np.arange(N, dtype=np.int64)
    offsets = np.array([0, N], dtype=np.int64)
    node_base = 0
    for level in range(depth):
        n_nodes = len(offsets) - 1
        counts = np.diff(offsets)
        assert counts.min() >= 2, f"KD level {level}: a node has {int(counts.min())} point(s)"
        starts = offsets[:-1]
        pts_ord = points[perm]
        mins = np.stack([np.minimum.reduceat(pts_ord[:, a], starts) for a in range(n_dims)], axis=1)
        maxs = np.stack([np.maximum.reduceat(pts_ord[:, a], starts) for a in range(n_dims)], axis=1)
        axis_per_node = np.argmax(maxs - mins, axis=1)
        node_id = np.repeat(np.arange(n_nodes, dtype=np.int64), counts)
        axis_per_point = np.repeat(axis_per_node, counts)
        vals = pts_ord[np.arange(N), axis_per_point]
        # one stable sort per level: node id first, then the coordinate
        order = np.lexsort((vals, node_id))
        perm = perm[order]
        vals_sorted = vals[order]
        mids = np.clip(counts // 2, 1, counts - 1)
        left_max = vals_sorted[starts + mids - 1]
        right_min = vals_sorted[starts + mids]
        split_axes[node_base:node_base + n_nodes] = axis_per_node.astype(np.int8)
        split_values[node_base:node_base + n_nodes] = np.where(
            left_max < right_min, 0.5 * (left_max + right_min), right_min)
        new_offsets = np.empty(2 * n_nodes + 1, dtype=np.int64)
        new_offsets[0:-1:2] = starts
        new_offsets[1:-1:2] = starts + mids
        new_offsets[-1] = offsets[-1]
        offsets = new_offsets
        node_base += n_nodes
    assert np.diff(offsets).min() >= 1
    return KDTree(depth=depth, split_axes=split_axes, split_values=split_values,
                  perm=perm, leaf_offsets=offsets)


def leaf_centroids(points: np.ndarray, perm: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean of every leaf's members, float64 accumulation -> float32."""
    counts = np.diff(offsets).astype(np.float64)
    assert counts.min() >= 1, "empty leaf"
    sums = np.add.reduceat(np.asarray(points, dtype=np.float64)[perm], offsets[:-1])
    return (sums / counts[:, None]).astype(np.float32)


def gather_leaf_neighbors(positions, perm, leaf_offsets, centroids, k: int, rng_seed: int):
    """Up to k member points per leaf (a random subset above k, all members
    plus a mask below). -> (rel_pos [L, k, D], valid [L, k], index [L, k])."""
    positions = np.asarray(positions, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    L, D = centroids.shape
    rel_pos = np.zeros((L, k, D), dtype=np.float32)
    valid = np.zeros((L, k), dtype=np.bool_)
    index = np.full((L, k), -1, dtype=np.int64)
    rng = np.random.default_rng(rng_seed)
    for leaf in range(L):
        members = perm[leaf_offsets[leaf]:leaf_offsets[leaf + 1]]
        sel = members[rng.choice(len(members), size=k, replace=False)] if len(members) > k else members
        n = len(sel)
        rel_pos[leaf, :n] = positions[sel] - centroids[leaf]
        valid[leaf, :n] = True
        index[leaf, :n] = sel
    return rel_pos, valid, index


def latent_partition(pos: np.ndarray, n_leaves: int, structured: bool, domain_lo, domain_hi,
                     grid_dims) -> dict:
    """The latent layout of one case: token positions (centroids) and the
    index partition (perm, leaf_offsets); the lattice also returns the
    cell id of every point."""
    pos = np.asarray(pos)
    if structured:
        centres, lo, step = grid_geometry(domain_lo, domain_hi, grid_dims)
        assert len(centres) == n_leaves, f"lattice has {len(centres)} cells, n_leaves={n_leaves}"
        cell_ids, n_out = assign_cells(pos, lo, step, grid_dims)
        offsets = np.zeros(n_leaves + 1, dtype=np.int64)
        np.cumsum(np.bincount(cell_ids, minlength=n_leaves), out=offsets[1:])
        return dict(centroids=centres, perm=np.argsort(cell_ids, kind="stable").astype(np.int64),
                    leaf_offsets=offsets, split_axes=np.zeros(0, np.int8),
                    split_values=np.zeros(0, np.float32), cell_ids=cell_ids, n_out_of_box=n_out)
    tree = build_equal_kdtree(pos, int(round(math.log2(n_leaves))))
    return dict(centroids=leaf_centroids(pos, tree.perm, tree.leaf_offsets), perm=tree.perm,
                leaf_offsets=tree.leaf_offsets, split_axes=tree.split_axes.copy(),
                split_values=tree.split_values.astype(np.float32),
                cell_ids=np.zeros(0, np.int64), n_out_of_box=0)


def gather_slots(pos: np.ndarray, part: dict, k: int, seed: int, structured: bool):
    """k member slots per token of a partition (see gather_leaf_neighbors /
    gather_structured_neighbors). -> (rel_pos, valid, index, counts | None)."""
    if structured:
        return gather_structured_neighbors(pos, part["cell_ids"], part["centroids"], k, seed)
    rel_pos, valid, index = gather_leaf_neighbors(pos, part["perm"], part["leaf_offsets"],
                                                  part["centroids"], k, seed)
    return rel_pos, valid, index, None


def knn_indices(query_pos, centroids, k: int) -> np.ndarray:
    """[N, k] int64 nearest token indices (distances are recomputed on the GPU)."""
    assert len(centroids) >= k
    _, idx = cKDTree(np.asarray(centroids)).query(np.asarray(query_pos), k=k)
    return idx.astype(np.int64)


# ---------------------------------------------------------------------------
# fixed lattice (layout "grid"): one token per cell of a regular grid over
# the dataset's domain, every cell filled with exactly k slots
# ---------------------------------------------------------------------------

def grid_geometry(domain_lo, domain_hi, dims):
    """(centres [L, D] float32, lo [D], step [D]); cell id = ix * ny + iy."""
    lo = np.asarray(domain_lo, dtype=np.float64)
    hi = np.asarray(domain_hi, dtype=np.float64)
    dims = tuple(int(d) for d in dims)
    assert lo.shape == hi.shape == (len(dims),) and np.all(hi > lo)
    step = (hi - lo) / np.asarray(dims, dtype=np.float64)
    axes = [lo[d] + (np.arange(dims[d], dtype=np.float64) + 0.5) * step[d] for d in range(len(dims))]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32), lo, step


def assign_cells(pos, lo, step, dims) -> tuple[np.ndarray, int]:
    """(cell id per point, number of points outside the box, which are
    folded into the boundary cells)."""
    pos = np.asarray(pos, dtype=np.float64)
    dims = tuple(int(d) for d in dims)
    idx = np.empty((pos.shape[0], len(dims)), dtype=np.int64)
    out = np.zeros(pos.shape[0], dtype=bool)
    for d in range(len(dims)):
        np.floor_divide(pos[:, d] - lo[d], step[d], out=idx[:, d], casting="unsafe")
        out |= (idx[:, d] < 0) | (idx[:, d] >= dims[d])
        np.clip(idx[:, d], 0, dims[d] - 1, out=idx[:, d])
    cell = idx[:, 0]
    for d in range(1, len(dims)):
        cell = cell * dims[d] + idx[:, d]
    return cell.astype(np.int64), int(out.sum())


def structured_slots(pos, cell, centres, k: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Exactly k point indices per cell: its own members in random order
    (truncated at k), then the nearest non-member points of the cell centre.
    -> (sel [L, k], counts [L])."""
    pos = np.asarray(pos, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.int64)
    L, N = int(len(centres)), int(pos.shape[0])
    assert N >= k
    counts = np.bincount(cell, minlength=L).astype(np.int64)
    order = np.argsort(cell.astype(np.float64) + rng.random(N), kind="stable")
    offsets = np.zeros(L + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    cell_of_pos = np.repeat(np.arange(L, dtype=np.int64), counts)
    pos_in_cell = np.arange(N, dtype=np.int64) - np.repeat(offsets[:-1], counts)
    keep = pos_in_cell < k
    sel = np.zeros((L, k), dtype=np.int64)
    sel[cell_of_pos[keep], pos_in_cell[keep]] = order[keep]
    starved = np.flatnonzero(counts < k)
    if starved.size:
        need = (k - counts[starved]).astype(np.int64)
        kq = int(min(2 * k, N))              # a starved cell owns fewer than k of them
        _, cand = cKDTree(pos).query(np.asarray(centres, dtype=np.float64)[starved], k=kq, workers=1)
        cand = np.asarray(cand, dtype=np.int64).reshape(len(starved), -1)
        ok = cell[cand] != starved[:, None]
        rank = np.cumsum(ok, axis=1)
        assert (rank[:, -1] >= need).all(), "kNN top-up ran out of candidates"
        take = ok & (rank <= need[:, None])
        rows, cols = np.nonzero(take)
        sel[starved[rows], counts[starved][rows] + rank[rows, cols] - 1] = cand[rows, cols]
    return sel, counts


def gather_structured_neighbors(positions, cell_ids, centres, k: int, rng_seed: int):
    """Lattice counterpart of gather_leaf_neighbors; the mask is all True.
    -> (rel_pos [L, k, D], valid [L, k], index [L, k], counts [L])."""
    positions = np.asarray(positions, dtype=np.float32)
    centres = np.asarray(centres, dtype=np.float32)
    sel, counts = structured_slots(positions, cell_ids, centres, k, np.random.default_rng(rng_seed))
    rel_pos = (positions[sel] - centres[:, None, :]).astype(np.float32)
    return rel_pos, np.ones(sel.shape, dtype=np.bool_), sel, counts


def occupancy_summary(counts_per_case: list, k: int, tag: str) -> None:
    """How many lattice cells are empty or under-filled (and so topped up)."""
    if not counts_per_case:
        return
    L = len(counts_per_case[0])
    empty = float(np.mean([(c == 0).sum() for c in counts_per_case]))
    under = float(np.mean([((c > 0) & (c < k)).sum() for c in counts_per_case]))
    print(f"[{tag}] lattice occupancy over {len(counts_per_case)} cases, {L} cells, k={k}: "
          f"{empty:.0f} empty and {under:.0f} under-filled cells per case "
          f"({100.0 * (empty + under) / L:.1f}% of tokens topped up by nearest points)", flush=True)

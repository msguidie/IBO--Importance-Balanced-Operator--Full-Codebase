"""Group statistics of the DrivAerML union tree (cropped volume + surface
points, 15 region levels + 7 subregion levels) and the training volume
pool: one volume point per subregion."""
from __future__ import annotations

import numpy as np

from ..common.kdtree import build_equal_kdtree


def grouped_centroids(pos_ord: np.ndarray, is_surf_ord: np.ndarray, offsets: np.ndarray,
                      point_budget: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
    """[G, 3] centroids and [G] surface fractions over perm-ordered points,
    reduced in float64 over bounded point chunks."""
    G = len(offsets) - 1
    centers = np.empty((G, 3), dtype=np.float32)
    surf_ratio = np.empty(G, dtype=np.float32)
    g0 = 0
    while g0 < G:
        g1 = g0 + 1
        while g1 < G and offsets[g1 + 1] - offsets[g0] <= point_budget:
            g1 += 1
        p0, p1 = int(offsets[g0]), int(offsets[g1])
        off = (offsets[g0:g1 + 1] - p0).astype(np.int64)
        counts = np.diff(off).astype(np.float64)
        assert counts.min() >= 1, "empty group"
        inv = 1.0 / counts
        centers[g0:g1] = (np.add.reduceat(np.asarray(pos_ord[p0:p1], dtype=np.float64), off[:-1])
                          * inv[:, None]).astype(np.float32)
        surf_ratio[g0:g1] = (np.add.reduceat(np.asarray(is_surf_ord[p0:p1], dtype=np.float64), off[:-1])
                             * inv).astype(np.float32)
        g0 = g1
    return centers, surf_ratio


def pick_volume_representatives(pos: np.ndarray, is_surf: np.ndarray, perm: np.ndarray,
                                slot_offsets: np.ndarray, n_subcells: int, sub_centroids: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """One volume point per subregion, exactly n_slots picks: a uniform
    random volume member when the subregion has one, else the parent
    region's volume point nearest the subregion centroid, else the case's
    nearest volume point (so a point may represent several subregions)."""
    n_slots = len(slot_offsets) - 1
    counts = np.diff(slot_offsets)
    # volume members sort ahead of surface members, in random order
    key = np.repeat(np.arange(n_slots, dtype=np.float64), counts) + 0.5 * is_surf[perm] + 0.5 * rng.random(len(perm))
    order = np.argsort(key, kind="stable")
    rep = perm[order][slot_offsets[:-1]].copy()
    del key, order
    missing = np.flatnonzero(is_surf[rep])
    if missing.size:
        cell_offsets = slot_offsets[::n_subcells]
        cells = missing // n_subcells
        unresolved = []
        for c in np.unique(cells):
            members = perm[cell_offsets[c]:cell_offsets[c + 1]]
            vol_members = members[~is_surf[members]]
            slots = missing[cells == c]
            if vol_members.size == 0:
                unresolved.extend(int(s) for s in slots)
                continue
            d = np.linalg.norm(sub_centroids[slots][:, None, :].astype(np.float64)
                               - pos[vol_members][None, :, :].astype(np.float64), axis=-1)
            rep[slots] = vol_members[np.argmin(d, axis=1)]
        if unresolved:
            from scipy.spatial import cKDTree
            vol_idx = np.flatnonzero(~is_surf)
            slots = np.asarray(unresolved, dtype=np.int64)
            _, nn = cKDTree(pos[vol_idx]).query(sub_centroids[slots], k=1)
            rep[slots] = vol_idx[nn]
    assert not is_surf[rep].any() and len(rep) == n_slots
    return rep


def subcell_centroids_per_cell(pos_ord: np.ndarray, surf_ord: np.ndarray, offsets: np.ndarray,
                               centres: np.ndarray, n_sub: int, subcell_depth: int):
    """Layout grid: per lattice cell an equal-count tree over its own
    points gives n_sub subregion centroids; a cell with fewer points than
    subregions repeats its points, an empty cell collapses onto its centre.
    -> ([n_cells, n_sub, 3] centroids, [n_cells, n_sub] surface fractions)."""
    n_cells = len(offsets) - 1
    sub_pos = np.empty((n_cells, n_sub, 3), dtype=np.float64)
    sub_surf = np.empty((n_cells, n_sub), dtype=np.float32)
    for c in range(n_cells):
        p0, p1 = int(offsets[c]), int(offsets[c + 1])
        n = p1 - p0
        if n >= n_sub:
            pts = pos_ord[p0:p1]
            tree = build_equal_kdtree(pts, subcell_depth)
            sub_pos[c], sub_surf[c] = grouped_centroids(pts[tree.perm], surf_ord[p0:p1][tree.perm],
                                                        tree.leaf_offsets)
        elif n > 0:
            take = np.arange(n_sub) % n
            sub_pos[c] = pos_ord[p0:p1][take]
            sub_surf[c] = surf_ord[p0:p1][take]
        else:
            sub_pos[c] = centres[c]
            sub_surf[c] = 0.0
    return sub_pos, sub_surf

"""DrivAerML preprocessing: the pre-converted `run_<id>.pt` cases -> the
cache.

  bake      per case, physical units / coord_scale: outward normals and the
            reference-checked full-surface Cd/Cl, the volume crop (the
            surface is never cropped), ONE equal-count tree over all
            points (15 region levels x 7 subregion levels), subregion and
            region centroids with surface fractions and signed wall
            distances (Open3D ray casting), the 8-feature slots, the query
            pools (training: one volume point per subregion + sampled
            surface points; held out: sampled volume + every surface point
            by default), the exact wall displacement of the pooled volume
            points, the 16 nearest region centroids of every pooled point
  stats     train-split statistics (Welford) -> norm_stats.json
  finalize  z-scored static/ (query pool) and latent/ (tokens)

Layout grid reads the finished importance cache: the query pool is copied
verbatim, only the tokens change (a 32^3 lattice over the crop box, each
cell split into 128 subregions by its own equal-count tree) and the
nearest-token table is recomputed; its token statistics are its own.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ...config import Config
from ...utils.seed import per_case_epoch_seed
from ..common.kdtree import assign_cells, build_equal_kdtree, grid_geometry, knn_indices
from ..common.pool import atomic_torch_save, limit_threads, report, run_pool
from ..common.welford import WelfordAccumulator, welford_local
from .cases import SCHEMA, SURF_CACHE, VOL_CACHE, paths
from .features import grouped_centroids, pick_volume_representatives, subcell_centroids_per_cell
from .raw import (case_path, compute_sdf_disp, compute_sdf_scalar_grad, crop_mask, load_raw_case,
                  resolve_normal_orientation)

# slot features (8): rel centroid 3 | signed distance | gradient 3 | surface fraction
PN_RELPOS, PN_SIGMA, PN_GRAD, PN_SURF_RATIO = slice(0, 3), 3, slice(4, 7), 7
_W: dict = {}


def _init(state: dict) -> None:
    _W.clear()
    _W.update(state)
    limit_threads(state.get("threads", 1))


def _coord_affine(x, p5, p95) -> np.ndarray:
    span = np.maximum(np.asarray(p95, dtype=np.float64) - np.asarray(p5, dtype=np.float64), 1e-8)
    return ((np.asarray(x, dtype=np.float64) - np.asarray(p5)) / span).astype(np.float32)


# ---------------------------------------------------------------------------
# bake (importance layout)
# ---------------------------------------------------------------------------

def _bake_worker(name: str):
    try:
        w = _W
        out = Path(w["wip"]) / f"{name}.pt"
        if out.exists():
            return ("ok", name, None)
        raw = load_raw_case(case_path(w["data_root"], name), name)
        normal, cdcl = resolve_normal_orientation(raw, w["cdcl_tol"])
        n_surf_raw = int(raw.surface_pos.shape[0])
        scale = w["scale"]
        core = crop_mask(raw.volume_pos, w["crop_lo"], w["crop_hi"])
        vol_pos, vol_vel = raw.volume_pos[core], raw.volume_velocity[core]
        vol_p, vol_nut = raw.volume_pressure[core], raw.volume_nut[core]
        del core, raw.volume_pos, raw.volume_velocity, raw.volume_pressure, raw.volume_nut
        n_vol_pts, n_surf_pts = len(vol_pos), n_surf_raw
        n_cells, n_sub = w["n_cells"], w["n_subcells"]
        n_slots = n_cells * n_sub
        assert n_vol_pts + n_surf_pts >= n_slots, f"{name}: fewer points than subregions"
        rng = np.random.default_rng(per_case_epoch_seed(name, 0))

        union = np.concatenate([(vol_pos / scale).astype(np.float32),
                                (raw.surface_pos / scale).astype(np.float32)], axis=0)
        is_surf = np.zeros(len(union), dtype=bool)
        is_surf[n_vol_pts:] = True
        tree = build_equal_kdtree(union, w["tree_depth"])
        slot_off = tree.leaf_offsets
        cell_off = slot_off[::n_sub]
        pos_ord, surf_ord = union[tree.perm], is_surf[tree.perm]
        sub_centroids, sub_surf_ratio = grouped_centroids(pos_ord, surf_ord, slot_off)
        cell_pos, _ = grouped_centroids(pos_ord, surf_ord, cell_off)
        del pos_ord, surf_ord
        sub_sigma, sub_grad = compute_sdf_scalar_grad(sub_centroids * scale, raw.stl_vertices,
                                                      raw.stl_faces, w["sdf_chunk"])
        cell_sigma, cell_grad = compute_sdf_scalar_grad(cell_pos * scale, raw.stl_vertices,
                                                        raw.stl_faces, w["sdf_chunk"])
        pn_feat = np.concatenate([sub_centroids - np.repeat(cell_pos, n_sub, axis=0),
                                  (sub_sigma / scale)[:, None], sub_grad, sub_surf_ratio[:, None]],
                                 axis=1).reshape(n_cells, n_sub, 8).astype(np.float32)

        # query pools: volume rows first
        if w["train"].get(name, False):
            vol_pool = pick_volume_representatives(union, is_surf, tree.perm, slot_off, n_sub,
                                                   sub_centroids, rng)
            cap = w["train_surface_cap"]
            surf_pool = np.sort(rng.choice(n_surf_pts, size=min(cap, n_surf_pts), replace=False))
        else:
            cap_v = w["eval_volume_cap"]
            vol_pool = np.sort(rng.choice(n_vol_pts, size=min(cap_v, n_vol_pts), replace=False))
            cap_s = w["eval_surface_cap"]
            surf_pool = np.arange(n_surf_pts) if cap_s is None else \
                np.sort(rng.choice(n_surf_pts, size=min(cap_s, n_surf_pts), replace=False))
        del tree
        disp = compute_sdf_disp(vol_pos[vol_pool], raw.stl_vertices, raw.stl_faces, w["sdf_chunk"]) / scale
        dec_vol = np.concatenate([union[vol_pool], disp.astype(np.float32)], axis=1)
        dec_surf = np.concatenate([union[n_vol_pts + surf_pool], normal[surf_pool]], axis=1)
        decoder_input = np.concatenate([dec_vol, dec_surf], axis=0).astype(np.float32)
        del dec_vol, dec_surf, disp, union, is_surf
        target_vol = np.concatenate([vol_vel[vol_pool], vol_p[vol_pool][:, None], vol_nut[vol_pool][:, None]],
                                    axis=1).astype(np.float32)
        target_surf = np.concatenate([raw.surface_pressure[surf_pool][:, None], raw.surface_wss[surf_pool]],
                                     axis=1).astype(np.float32)
        assert w["n_cells"] <= 65536, "token indices are stored as uint16"
        idw = knn_indices(decoder_input[:, :3], cell_pos, w["idw_k"]).astype(np.uint16)
        atomic_torch_save({
            "case_name": name, "layout": "vol_first", "n_vol": int(len(vol_pool)),
            "n_surf": int(len(surf_pool)), "n_surf_raw": n_surf_raw,
            "decoder_input": torch.from_numpy(decoder_input), "idw_indices": torch.from_numpy(idw),
            "target_vol_raw": torch.from_numpy(target_vol), "target_surf_raw": torch.from_numpy(target_surf),
            "surface_normal": torch.from_numpy(normal[surf_pool].astype(np.float32)),
            "surface_area": torch.from_numpy(raw.surface_area[surf_pool].astype(np.float32)),
            "a_ref": float(raw.a_ref), "cd_true": float(cdcl["cd"]), "cl_true": float(cdcl["cl"]),
            "normal_flipped": bool(cdcl["flipped"]), "reference_error": cdcl["reference_error"],
            "latent": {"pn_feat": torch.from_numpy(pn_feat), "cell_pos": torch.from_numpy(cell_pos),
                       "cell_sdf": torch.from_numpy(np.concatenate(
                           [(cell_sigma / scale)[:, None], cell_grad], axis=1).astype(np.float32))}},
            out)
        return ("ok", name, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", name, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# train-split statistics
# ---------------------------------------------------------------------------

def _stats_worker(name: str):
    w = _W
    wip = torch.load(str(Path(w["wip"]) / f"{name}.pt"), map_location="cpu", weights_only=True)
    lat = wip["latent"]
    pn = lat["pn_feat"].numpy()
    n_vol = int(wip["n_vol"])
    dec = wip["decoder_input"].numpy()
    arrays = (wip["target_vol_raw"].numpy(), wip["target_surf_raw"].numpy(),
              pn[..., PN_RELPOS].reshape(-1, 3),
              np.linalg.norm(dec[:n_vol, 3:6], axis=1)[:, None],          # pooled |displacement|
              np.concatenate([pn[..., PN_SIGMA].reshape(-1), lat["cell_sdf"].numpy()[:, 0]])[:, None])
    triples = tuple(welford_local(a) for a in arrays)
    rng = np.random.default_rng(per_case_epoch_seed(name, 1))
    k = w["coord_sample"]
    pos = dec[:, :3]
    sample = pos[rng.choice(len(pos), size=k, replace=False)] if len(pos) > k else pos
    return triples, sample.astype(np.float32)


def compute_norm_stats(state: dict, train: list[str], workers: int, pctl, scale: float) -> dict:
    accs = [WelfordAccumulator(c) for c in (5, 4, 3, 1, 1)]
    results = run_pool(train, _stats_worker, _init, (state,), workers, "drivaerml:stats", ordered=True)
    samples = []
    for triples, sample in results:
        for acc, (n, mean, m2) in zip(accs, triples):
            acc.merge_local(n, mean, m2)
        samples.append(sample)
    (vm, vs), (sm, ss), (rm, rs), (qm, qs), (lm, ls) = [acc.finalize() for acc in accs]
    pooled = np.concatenate(samples, axis=0).astype(np.float64)
    return {"schema": SCHEMA, "vol_channel_names": list(VOL_CACHE), "surf_channel_names": list(SURF_CACHE),
            "vol_mean": vm.tolist(), "vol_std": vs.tolist(), "surf_mean": sm.tolist(), "surf_std": ss.tolist(),
            "subcell_relpos_mean": rm.tolist(), "subcell_relpos_std": rs.tolist(),
            "query_sdf_mean": float(qm[0]), "query_sdf_std": float(qs[0]),
            "latent_sdf_mean": float(lm[0]), "latent_sdf_std": float(ls[0]),
            "coord_p5": np.percentile(pooled, pctl[0], axis=0).tolist(),
            "coord_p95": np.percentile(pooled, pctl[1], axis=0).tolist(),
            "coord_pctl_lo": float(pctl[0]), "coord_pctl_hi": float(pctl[1]),
            "coord_scale": float(scale), "n_train_cases": len(train)}


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

def _finalize_worker(name: str):
    try:
        w = _W
        st = w["stats"]
        static_path, latent_path = Path(w["static"]) / f"{name}.pt", Path(w["latent"]) / f"{name}.pt"
        if static_path.exists() and latent_path.exists():
            return ("ok", name, None)
        wip = torch.load(str(Path(w["wip"]) / f"{name}.pt"), map_location="cpu", weights_only=True)
        lat = wip["latent"]
        n_vol = int(wip["n_vol"])
        q_m, q_s = float(st["query_sdf_mean"]), max(float(st["query_sdf_std"]), 1e-8)
        l_m, l_s = float(st["latent_sdf_mean"]), max(float(st["latent_sdf_std"]), 1e-8)
        p5, p95 = st["coord_p5"], st["coord_p95"]
        dec_raw = wip["decoder_input"].numpy()
        dec = np.empty((dec_raw.shape[0], 7), dtype=np.float32)
        dec[:, 0:3] = _coord_affine(dec_raw[:, :3], p5, p95)
        disp = dec_raw[:n_vol, 3:6]
        mag = np.linalg.norm(disp, axis=1, keepdims=True)
        dec[:n_vol, 3] = ((mag[:, 0] - q_m) / q_s).astype(np.float32)       # signed z-score of the distance
        dec[:n_vol, 4:7] = (disp / np.maximum(mag, 1e-12)).astype(np.float32)
        dec[n_vol:, 3] = np.float32((0.0 - q_m) / q_s)                      # the wall's own z-score
        dec[n_vol:, 4:7] = dec_raw[n_vol:, 3:6]                             # outward normal

        def z(x, mean, std):
            return ((x.double() - torch.tensor(mean, dtype=torch.float64))
                    / torch.tensor(std, dtype=torch.float64).clamp(min=1e-8)).float()

        atomic_torch_save({
            "case_name": name, "layout": "vol_first", "schema": SCHEMA, "n_vol": n_vol,
            "n_surf": int(wip["n_surf"]), "n_surf_raw": int(wip["n_surf_raw"]),
            "decoder_input": torch.from_numpy(dec),
            "query_pos_geo": torch.from_numpy(dec_raw[:, :3].copy()),
            "idw_indices": wip["idw_indices"].to(torch.uint16),
            "target_vol": z(wip["target_vol_raw"], st["vol_mean"], st["vol_std"]),
            "target_surf": z(wip["target_surf_raw"], st["surf_mean"], st["surf_std"]),
            "surface_normal": wip["surface_normal"].float(), "surface_area": wip["surface_area"].float(),
            "a_ref": float(wip["a_ref"]), "cd_true": float(wip["cd_true"]), "cl_true": float(wip["cl_true"])},
            static_path)

        pn = lat["pn_feat"].numpy().copy()
        rm = np.asarray(st["subcell_relpos_mean"], dtype=np.float32)
        rs = np.maximum(np.asarray(st["subcell_relpos_std"], dtype=np.float32), 1e-8)
        pn[..., PN_RELPOS] = (pn[..., PN_RELPOS] - rm) / rs         # offsets: their own z-score
        pn[..., PN_SIGMA] = (pn[..., PN_SIGMA] - l_m) / l_s
        cell_sdf = lat["cell_sdf"].numpy().copy()
        cell_sdf[:, 0] = (cell_sdf[:, 0] - l_m) / l_s
        cell_pos_geo = lat["cell_pos"].numpy()
        atomic_torch_save({"pn_feat": torch.from_numpy(pn.astype(np.float32)),
                           "cell_pos": torch.from_numpy(_coord_affine(cell_pos_geo, p5, p95)),
                           "cell_pos_geo": torch.from_numpy(cell_pos_geo.astype(np.float32)),
                           "cell_sdf": torch.from_numpy(cell_sdf.astype(np.float32))}, latent_path)
        return ("ok", name, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", name, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# layout grid: lattice tokens over the importance cache's query pool
# ---------------------------------------------------------------------------

def grid_cells(cfg: Config):
    """(centres [L, 3], lo, step) of the lattice in the scaled frame."""
    pp = cfg.drivaerml
    s = float(pp.coord_scale)
    return grid_geometry(np.asarray(pp.crop_min) / s, np.asarray(pp.crop_max) / s, pp.grid_dims)


def _grid_worker(name: str):
    try:
        w = _W
        out = Path(w["wip_grid"]) / f"{name}.pt"
        if out.exists():
            return ("ok", name, None, None)
        scale = w["scale"]
        raw = load_raw_case(case_path(w["data_root"], name), name)
        core = crop_mask(raw.volume_pos, w["crop_lo"], w["crop_hi"])
        n_vol = int(core.sum())
        pos = np.empty((n_vol + len(raw.surface_pos), 3), dtype=np.float32)
        np.divide(raw.volume_pos[core], scale, out=pos[:n_vol], casting="unsafe")
        np.divide(raw.surface_pos, scale, out=pos[n_vol:], casting="unsafe")
        stl_v, stl_f = raw.stl_vertices, raw.stl_faces
        del raw, core
        is_surf = np.zeros(len(pos), dtype=np.float32)
        is_surf[n_vol:] = 1.0
        cell, _ = assign_cells(pos, w["lo"], w["step"], w["grid_dims"])
        n_cells = len(w["centres"])
        counts = np.bincount(cell, minlength=n_cells)
        order = np.argsort(cell, kind="stable")
        pos_ord, surf_ord = pos[order], is_surf[order]
        del pos, is_surf, order, cell
        offsets = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        n_sub = w["n_subcells"]
        sub_pos, sub_surf = subcell_centroids_per_cell(pos_ord, surf_ord, offsets, w["centres"], n_sub,
                                                       int(round(math.log2(n_sub))))
        del pos_ord, surf_ord
        centres = w["centres"]
        q = np.concatenate([sub_pos.reshape(-1, 3), centres], axis=0)
        sigma, grad = compute_sdf_scalar_grad((q * scale).astype(np.float32), stl_v, stl_f, w["sdf_chunk"])
        sigma = sigma / scale
        n_tot = n_cells * n_sub
        static_old = torch.load(str(Path(w["static_src"]) / f"{name}.pt"), map_location="cpu", weights_only=True)
        idw = knn_indices(static_old["query_pos_geo"].numpy(), centres.astype(np.float32), w["idw_k"])
        atomic_torch_save({
            "rel_pos_raw": torch.from_numpy((sub_pos - centres[:, None, :]).astype(np.float32)),
            "sub_sigma_raw": torch.from_numpy(sigma[:n_tot].reshape(n_cells, n_sub).astype(np.float32)),
            "sub_grad": torch.from_numpy(grad[:n_tot].reshape(n_cells, n_sub, 3).astype(np.float32)),
            "sub_surf_ratio": torch.from_numpy(sub_surf),
            "cell_sigma_raw": torch.from_numpy(sigma[n_tot:].astype(np.float32)),
            "cell_grad": torch.from_numpy(grad[n_tot:].astype(np.float32)),
            "idw_indices": torch.from_numpy(idw.astype(np.uint16))}, out)
        return ("ok", name, None, counts)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", name, f"{type(exc).__name__}: {exc}", None)


def _grid_finalize_worker(name: str):
    try:
        w = _W
        st = w["grid_stats"]
        static_path, latent_path = Path(w["static"]) / f"{name}.pt", Path(w["latent"]) / f"{name}.pt"
        if static_path.exists() and latent_path.exists():
            return ("ok", name, None)
        g = torch.load(str(Path(w["wip_grid"]) / f"{name}.pt"), map_location="cpu", weights_only=True)
        rm = np.asarray(st["subcell_relpos_mean"], dtype=np.float32)
        rs = np.maximum(np.asarray(st["subcell_relpos_std"], dtype=np.float32), 1e-8)
        l_m, l_s = float(st["latent_sdf_mean"]), max(float(st["latent_sdf_std"]), 1e-8)
        pn = np.concatenate([(g["rel_pos_raw"].numpy() - rm) / rs,
                             ((g["sub_sigma_raw"].numpy() - l_m) / l_s)[..., None],
                             g["sub_grad"].numpy(), g["sub_surf_ratio"].numpy()[..., None]],
                            axis=-1).astype(np.float32)
        centres = w["centres"].astype(np.float32)
        atomic_torch_save({"pn_feat": torch.from_numpy(pn),
                           "cell_pos": torch.from_numpy(_coord_affine(centres, w["p5"], w["p95"])),
                           "cell_pos_geo": torch.from_numpy(centres),
                           "cell_sdf": torch.from_numpy(np.concatenate(
                               [((g["cell_sigma_raw"].numpy() - l_m) / l_s)[:, None], g["cell_grad"].numpy()],
                               axis=1).astype(np.float32))}, latent_path)
        static = dict(torch.load(str(Path(w["static_src"]) / f"{name}.pt"), map_location="cpu", weights_only=True))
        static["idw_indices"] = g["idw_indices"].to(torch.uint16)
        atomic_torch_save(static, static_path)
        return ("ok", name, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", name, f"{type(exc).__name__}: {exc}")


def run_structured(cfg: Config, names: list[str], train: list[str]) -> None:
    pp = cfg.drivaerml
    p, src = paths(cfg), paths(cfg, "importance")
    missing = [n for n in names if not (src["static"] / f"{n}.pt").exists()]
    if missing or not src["norm_stats"].exists():
        raise SystemExit("layout grid needs the finished importance cache (its query pool is reused): "
                         f"{len(missing)} cases missing under {src['static']}")
    assert int(np.prod(pp.grid_dims)) == int(pp.n_leaves), "grid_dims must multiply to n_leaves"
    centres, lo, step = grid_cells(cfg)
    for d in (p["wip_grid"], p["static"], p["latent"]):
        Path(d).mkdir(parents=True, exist_ok=True)
    with open(src["norm_stats"], encoding="utf-8") as f:
        stats = json.load(f)
    state = dict(wip_grid=str(p["wip_grid"]), static_src=str(src["static"]), data_root=cfg.data.data_root,
                 scale=float(pp.coord_scale), crop_lo=tuple(pp.crop_min), crop_hi=tuple(pp.crop_max),
                 lo=lo, step=step, grid_dims=tuple(pp.grid_dims), centres=centres.astype(np.float64),
                 n_subcells=int(pp.slots), sdf_chunk=int(pp.sdf_chunk), idw_k=int(pp.idw_k),
                 threads=int(pp.worker_threads))
    workers = int(pp.n_workers)
    results = run_pool(names, _grid_worker, _init, (state,), workers, "drivaerml:grid")
    failed, _ = report(results, "grid")
    if failed:
        raise SystemExit(f"[drivaerml] grid bake failed on {len(failed)} cases")
    counts = [r[3] for r in results if r[3] is not None]
    if counts:
        L, n_sub = len(centres), int(pp.slots)
        empty = np.mean([(c == 0).sum() for c in counts])
        under = np.mean([((c > 0) & (c < n_sub)).sum() for c in counts])
        print(f"[drivaerml:grid] {L} cells: {empty:.0f} empty and {under:.0f} under-filled per case "
              f"({100.0 * (empty + under) / L:.1f}% degenerate tokens)", flush=True)
    # token statistics of this layout (the query pool keeps the importance statistics)
    rel_acc, sdf_acc = WelfordAccumulator(3), WelfordAccumulator(1)
    for name in train:
        g = torch.load(str(p["wip_grid"] / f"{name}.pt"), map_location="cpu", weights_only=True)
        rel_acc.merge_local(*welford_local(g["rel_pos_raw"].numpy().reshape(-1, 3)))
        sdf_acc.merge_local(*welford_local(np.concatenate([g["sub_sigma_raw"].numpy().reshape(-1),
                                                           g["cell_sigma_raw"].numpy()])[:, None]))
    (rm, rs), (lm, ls) = rel_acc.finalize(), sdf_acc.finalize()
    grid_stats = {"subcell_relpos_mean": rm.tolist(), "subcell_relpos_std": rs.tolist(),
                  "latent_sdf_mean": float(lm[0]), "latent_sdf_std": float(ls[0]), "n_train_cases": len(train)}
    with open(p["latent_stats"], "w", encoding="utf-8") as f:
        json.dump(grid_stats, f, indent=1)
    results = run_pool(names, _grid_finalize_worker, _init,
                       (dict(state, grid_stats=grid_stats, static=str(p["static"]), latent=str(p["latent"]),
                             p5=stats["coord_p5"], p95=stats["coord_p95"]),), workers, "drivaerml:grid-finalize")
    failed, _ = report(results, "grid-finalize")
    if failed:
        raise SystemExit(f"[drivaerml] grid finalize failed on {len(failed)} cases")
    print(f"[drivaerml] grid layout done -> {p['cache']}", flush=True)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_preprocess(cfg: Config, splits: dict) -> None:
    pp = cfg.drivaerml
    names = list(splits["train"]) + list(splits["val"]) + list(splits["test"])
    missing = [n for n in names if not (Path(cfg.data.data_root) / f"{n}.pt").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} raw cases missing under {cfg.data.data_root} (e.g. {missing[:5]})")
    if cfg.run.layout == "grid":
        run_structured(cfg, names, list(splits["train"]))
        return
    p = paths(cfg)
    n_cells, n_sub = int(pp.n_leaves), int(pp.slots)
    depth = int(round(math.log2(n_cells))) + int(round(math.log2(n_sub)))
    assert 2 ** depth == n_cells * n_sub and n_cells <= 65536
    for d in (p["wip"], p["static"], p["latent"]):
        Path(d).mkdir(parents=True, exist_ok=True)
    state = dict(wip=str(p["wip"]), data_root=cfg.data.data_root, scale=float(pp.coord_scale),
                 crop_lo=tuple(pp.crop_min), crop_hi=tuple(pp.crop_max), n_cells=n_cells, n_subcells=n_sub,
                 tree_depth=depth, idw_k=int(pp.idw_k), sdf_chunk=int(pp.sdf_chunk),
                 cdcl_tol=float(pp.cdcl_tolerance), train_surface_cap=int(pp.train_surface_cap),
                 eval_volume_cap=int(pp.eval_volume_cap),
                 eval_surface_cap=None if pp.eval_surface_cap is None else int(pp.eval_surface_cap),
                 coord_sample=int(pp.coord_pctl_sample), threads=int(pp.worker_threads),
                 train={n: True for n in splits["train"]})
    workers = int(pp.n_workers)
    print(f"[drivaerml] {len(names)} cases -> {p['cache']}", flush=True)
    results = run_pool(names, _bake_worker, _init, (state,), workers, "drivaerml:bake")
    failed, _ = report(results, "bake")
    if failed:
        raise SystemExit(f"[drivaerml] bake failed on {len(failed)} cases")
    stats = compute_norm_stats(state, list(splits["train"]), workers,
                               tuple(float(v) for v in pp.coord_pctl), float(pp.coord_scale))
    with open(p["norm_stats"], "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    results = run_pool(names, _finalize_worker, _init,
                       (dict(state, stats=stats, static=str(p["static"]), latent=str(p["latent"])),),
                       min(workers, 16), "drivaerml:finalize")
    failed, _ = report(results, "finalize")
    if failed:
        raise SystemExit(f"[drivaerml] finalize failed on {len(failed)} cases")
    print(f"[drivaerml] done -> {p['cache']}", flush=True)

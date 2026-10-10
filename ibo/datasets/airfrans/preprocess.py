"""AirfRANS preprocessing: the official case folders -> the cache.

  manifest   the official full task: 800 train-pool + 200 test names
             (Dataset/manifest.json); 720 / 80 train / val carved from the
             pool with a fixed seed
  bake       per case, physical units: internal mesh + airfoil polygon
             (pyvista), the signed distance and gradient at every mesh point
             (outward normal on the wall), the directional distance profile,
             the curvature proxy, the latent layout (equal-count KD-tree, or
             the lattice of the layout ablation) with the same fields at the
             token positions, k member slots per token (15 features), the k
             nearest tokens of every point
  stats      train-split statistics (Welford) -> norm_stats.json
  finalize   z-scored per-point (static/) and per-token (latent/) tensors
  calibrate  the RoPE domain and the Fourier SDF scale

Targets (u_x, u_y, p, nut) are z-scored with pooled statistics; the inflow
(U cos a, U sin a) is z-scored per case; absolute coordinates get a
per-axis P5/P95 affine. The decoder's distance column is the wall-anchored
|sdf| / sigma (exactly zero on the wall), the slot distance channel the
z-scored |sdf|. --from-cache (layout grid) copies the position-only fields
of the importance bake and rebuilds the partition-dependent ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ...config import Config
from ...utils.seed import per_case_epoch_seed
from ..common.calibration import run_calibration
from ..common.kdtree import gather_slots, knn_indices, latent_partition, occupancy_summary
from ..common.polygon import compute_did, compute_knn_curvature, polygon_sdf_and_grad
from ..common.pool import atomic_torch_save, limit_threads, report, run_pool
from ..common.welford import WelfordAccumulator, welford_local
from .cases import SCHEMA, paths
from .metrics import reorganize

VOL_CHANNELS = ("u_x", "u_y", "p", "nut")
# slot features (15): rel pos 2 | sdf | grad 2 | surface flag | did 8 | curvature
PN_RELPOS, PN_SDF, PN_GRAD, PN_SURF, PN_DID, PN_CURV = slice(0, 2), 2, slice(3, 5), 5, slice(6, 14), 14
_W: dict = {}


def _init(state: dict) -> None:
    _W.clear()
    _W.update(state)
    limit_threads(state.get("threads", 1))


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def build_manifest(data_root: str, val_seed: int, n_val: int) -> dict:
    """720 / 80 / 200 from the official manifest's full_train / full_test."""
    path = Path(data_root) / "manifest.json"
    if not path.exists():
        raise SystemExit(f"official AirfRANS manifest not found at {path} (it ships with the dataset)")
    with open(path, encoding="utf-8") as f:
        official = json.load(f)
    pool, test = list(official["full_train"]), list(official["full_test"])
    assert len(pool) == 800 and len(test) == 200, "unexpected official manifest"
    perm = np.random.default_rng(val_seed).permutation(len(pool))
    n_train = len(pool) - n_val
    return {"train": [pool[i] for i in perm[:n_train]], "val": [pool[i] for i in perm[n_train:]],
            "test": test}


# ---------------------------------------------------------------------------
# bake
# ---------------------------------------------------------------------------

def parse_case_name(name: str) -> tuple[float, float]:
    """airFoil2D_SST_<Uinf>_<alpha>_... -> (Uinf, alpha in degrees)."""
    parts = name.split("_")
    return float(parts[2]), float(parts[3])


def read_raw_case(data_root: str, name: str, sdf_chunk: int) -> dict:
    import pyvista as pv

    case_dir = Path(data_root) / name
    internal = pv.read(str(case_dir / f"{name}_internal.vtu"))
    aerofoil = pv.read(str(case_dir / f"{name}_aerofoil.vtp"))
    pos = np.asarray(internal.points[:, :2], dtype=np.float32)
    airfoil_pos = np.asarray(aerofoil.points[:, :2], dtype=np.float32)
    surf = np.asarray(internal.point_data["U"][:, 0] == 0)
    Uinf, alpha = parse_case_name(name)
    a = np.deg2rad(alpha)
    uinf_raw = np.array([np.cos(a) * Uinf, np.sin(a) * Uinf], dtype=np.float32)
    target_raw = np.concatenate([np.asarray(internal.point_data["U"][:, :2], dtype=np.float32),
                                 np.asarray(internal.point_data["p"], dtype=np.float32)[:, None],
                                 np.asarray(internal.point_data["nut"], dtype=np.float32)[:, None]],
                                axis=-1)
    sdf_t, grad_t = polygon_sdf_and_grad(torch.from_numpy(pos), torch.from_numpy(airfoil_pos), sdf_chunk)
    mesh_sdf, mesh_grad = sdf_t.numpy().astype(np.float32), grad_t.numpy().astype(np.float32)
    if surf.any():        # wall rows: exact zero, the outward normal from the airfoil mesh
        normals = -np.asarray(aerofoil.point_data["Normals"][:, :2], dtype=np.float32)
        mesh_sdf[surf] = 0.0
        mesh_grad[surf] = np.ascontiguousarray(reorganize(np.asarray(aerofoil.points[:, :2]),
                                                          np.asarray(internal.points)[surf, :2], normals),
                                               dtype=np.float32)
    fields = {"pos": pos, "target_raw": target_raw, "mesh_sdf": mesh_sdf, "mesh_grad": mesh_grad,
              "uinf_raw": uinf_raw, "airfoil_pos": airfoil_pos}
    bad = [k for k, v in fields.items() if not np.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"{name}: non-finite values in {bad}")
    return {**fields, "surf": surf, "Uinf": Uinf, "alpha": alpha}


def _latent_block(w: dict, name: str, pos, surf, mesh_sdf, mesh_grad, mesh_did, mesh_curv, airfoil_pos):
    """Everything that depends on the latent partition: token positions
    with their fields, the k slots per token, the nearest tokens."""
    part = latent_partition(pos, w["n_leaves"], w["structured"], w["domain_lo"], w["domain_hi"],
                            w["grid_dims"])
    leaf_pos = part["centroids"]
    s_t, g_t = polygon_sdf_and_grad(torch.from_numpy(leaf_pos), torch.from_numpy(airfoil_pos), w["sdf_chunk"])
    leaf_did = compute_did(leaf_pos, airfoil_pos, w["did_bins"], w["did_dmax"])
    rel_pos, valid, sel, counts = gather_slots(pos, part, w["slots"], per_case_epoch_seed(name, 0),
                                               w["structured"])
    safe = np.where(valid, sel, 0)
    m3 = valid[..., None]
    pn_input = np.concatenate([
        rel_pos, np.where(valid, mesh_sdf[safe], 0.0)[..., None], np.where(m3, mesh_grad[safe], 0.0),
        np.where(valid, surf.astype(np.float32)[safe], 0.0)[..., None],
        np.where(m3, mesh_did[safe], 0.0), np.where(valid, mesh_curv[safe], 0.0)[..., None]],
        axis=-1).astype(np.float32)
    latent = {"pn_input": torch.from_numpy(pn_input), "pn_mask": torch.from_numpy(valid),
              "leaf_pos": torch.from_numpy(leaf_pos), "leaf_sdf": s_t.float(), "leaf_sdf_grad": g_t.float(),
              "leaf_did_raw": torch.from_numpy(leaf_did), "latent_grid": w["latent_grid"],
              "n_leaves": int(w["n_leaves"]), "tree_split_axes": torch.from_numpy(part["split_axes"]),
              "tree_split_values": torch.from_numpy(part["split_values"])}
    idw = torch.from_numpy(knn_indices(pos, leaf_pos, w["idw_k"]).astype(np.uint16))
    return latent, idw, counts


def _bake_worker(name: str):
    try:
        w = _W
        out = Path(w["wip"]) / f"{name}.pt"
        if out.exists():
            return ("ok", name, None, None)
        if w.get("src_wip"):                      # layout grid from the importance bake
            src = torch.load(str(Path(w["src_wip"]) / f"{name}.pt"), map_location="cpu", weights_only=True)
            if src.get("schema") != SCHEMA:
                raise RuntimeError(f"source wip schema {src.get('schema')!r}")
            raw = {k: src[k].numpy() for k in ("pos", "surf", "mesh_sdf", "mesh_grad", "mesh_did_raw",
                                                 "mesh_curv_raw", "airfoil_pos")}
            payload = dict(src)
        else:
            r = read_raw_case(w["data_root"], name, w["sdf_chunk"])
            raw = {"pos": r["pos"], "surf": r["surf"], "mesh_sdf": r["mesh_sdf"], "mesh_grad": r["mesh_grad"],
                   "mesh_did_raw": compute_did(r["pos"], r["airfoil_pos"], w["did_bins"], w["did_dmax"]),
                   "mesh_curv_raw": compute_knn_curvature(r["pos"], r["pos"], r["mesh_grad"], w["curvature_k"]),
                   "airfoil_pos": r["airfoil_pos"]}
            payload = {"case_name": name, "schema": SCHEMA, "n_points": int(r["pos"].shape[0]),
                       **{k: torch.from_numpy(v) for k, v in raw.items()},
                       "target_raw": torch.from_numpy(r["target_raw"]),
                       "uinf_raw": torch.from_numpy(r["uinf_raw"]), "Uinf": r["Uinf"], "alpha": r["alpha"]}
        latent, idw, counts = _latent_block(w, name, raw["pos"], raw["surf"], raw["mesh_sdf"],
                                            raw["mesh_grad"], raw["mesh_did_raw"], raw["mesh_curv_raw"],
                                            raw["airfoil_pos"])
        payload.update(idw_indices=idw, latent=latent)
        atomic_torch_save(payload, out)
        return ("ok", name, None, counts)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", name, f"{type(exc).__name__}: {exc}", None)


# ---------------------------------------------------------------------------
# train-split statistics
# ---------------------------------------------------------------------------

def _stats_worker(name: str):
    w = _W
    wip = torch.load(str(Path(w["wip"]) / f"{name}.pt"), map_location="cpu", weights_only=True)
    latent = wip["latent"]
    pn, mask = latent["pn_input"].numpy(), latent["pn_mask"].numpy()
    arrays = (wip["target_raw"].numpy(), pn[..., PN_RELPOS][mask],
              np.abs(wip["mesh_sdf"].numpy())[:, None],        # |distance| at every point
              latent["leaf_sdf"].numpy()[:, None],             # signed distance at the tokens
              wip["mesh_did_raw"].numpy(), latent["leaf_did_raw"].numpy(),
              wip["mesh_curv_raw"].numpy()[:, None])
    triples = tuple(welford_local(a) for a in arrays)
    pos = wip["pos"].numpy()
    rng = np.random.default_rng(per_case_epoch_seed(name, 1))
    k = w["coord_sample"]
    sample = pos[rng.choice(len(pos), size=k, replace=False)] if len(pos) > k else pos
    return triples, wip["uinf_raw"].numpy(), sample.astype(np.float32)


def compute_norm_stats(state: dict, train: list[str], workers: int, pctl, did_bins: int) -> dict:
    accs = [WelfordAccumulator(c) for c in (4, 2, 1, 1, did_bins, did_bins, 1)]
    results = run_pool(train, _stats_worker, _init, (state,), workers, "airfrans:stats", ordered=True)
    uinfs, samples = [], []
    for triples, uinf_raw, sample in results:
        for acc, (n, mean, m2) in zip(accs, triples):
            acc.merge_local(n, mean, m2)
        uinfs.append(uinf_raw)
        samples.append(sample)
    (t_m, t_s), (rp_m, rp_s), (qs_m, qs_s), (ls_m, ls_s), (qd_m, qd_s), (ld_m, ld_s), (qc_m, qc_s) = \
        [acc.finalize() for acc in accs]
    uinfs = np.stack(uinfs).astype(np.float64)
    pooled = np.concatenate(samples, axis=0).astype(np.float64)
    return {"schema": SCHEMA, "target_channel_names": list(VOL_CHANNELS),
            "target_mean": t_m.tolist(), "target_std": t_s.tolist(),
            "uinf_mean": uinfs.mean(axis=0).tolist(), "uinf_std": uinfs.std(axis=0).tolist(),
            "relpos_mean": rp_m.tolist(), "relpos_std": rp_s.tolist(),
            "query_sdf_mean": float(qs_m[0]), "query_sdf_std": float(qs_s[0]),
            "latent_sdf_mean": float(ls_m[0]), "latent_sdf_std": float(ls_s[0]),
            "query_did_mean": qd_m.tolist(), "query_did_std": qd_s.tolist(),
            "latent_did_mean": ld_m.tolist(), "latent_did_std": ld_s.tolist(),
            "query_curv_mean": float(qc_m[0]), "query_curv_std": float(qc_s[0]),
            "coord_p5": np.percentile(pooled, pctl[0], axis=0).tolist(),
            "coord_p95": np.percentile(pooled, pctl[1], axis=0).tolist(),
            "coord_pctl_lo": float(pctl[0]), "coord_pctl_hi": float(pctl[1]),
            "n_train_cases": len(train)}


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

def coord_affine(x, p5, p95) -> np.ndarray:
    span = np.maximum(np.asarray(p95, dtype=np.float64) - np.asarray(p5, dtype=np.float64), 1e-8)
    return ((np.asarray(x, dtype=np.float64) - np.asarray(p5)) / span).astype(np.float32)


def _finalize_worker(name: str):
    try:
        w = _W
        st = w["stats"]
        static_path, latent_path = Path(w["static"]) / f"{name}.pt", Path(w["latent"]) / f"{name}.pt"
        if static_path.exists() and latent_path.exists():
            return ("ok", name, None)
        wip = torch.load(str(Path(w["wip"]) / f"{name}.pt"), map_location="cpu", weights_only=True)
        lat = wip["latent"]

        def z(x, mean, std):
            return ((np.asarray(x, dtype=np.float64) - np.asarray(mean))
                    / np.maximum(np.asarray(std, dtype=np.float64), 1e-8)).astype(np.float32)

        p5, p95 = st["coord_p5"], st["coord_p95"]
        pos = wip["pos"].numpy()
        mesh_sdf, mesh_grad = wip["mesh_sdf"].numpy(), wip["mesh_grad"].numpy()
        dec = np.empty((pos.shape[0], 5), dtype=np.float32)
        dec[:, 0:2] = coord_affine(pos, p5, p95)
        dec[:, 2] = (np.abs(mesh_sdf) / max(float(st["query_sdf_std"]), 1e-8)).astype(np.float32)
        dec[:, 3:5] = mesh_grad
        static = {"case_name": name, "schema": SCHEMA, "decoder_input": torch.from_numpy(dec),
                  "did": torch.from_numpy(z(wip["mesh_did_raw"].numpy(), st["query_did_mean"], st["query_did_std"])),
                  "curvature": torch.from_numpy(z(wip["mesh_curv_raw"].numpy(), st["query_curv_mean"], st["query_curv_std"])),
                  "query_pos_geo": torch.from_numpy(pos.astype(np.float32)),
                  "idw_indices": wip["idw_indices"].to(torch.uint16),
                  "target": torch.from_numpy(z(wip["target_raw"].numpy(), st["target_mean"], st["target_std"])),
                  "is_volume": ~wip["surf"].bool(),
                  "uinf": torch.from_numpy(z(wip["uinf_raw"].numpy(), st["uinf_mean"], st["uinf_std"])),
                  "uinf_raw": wip["uinf_raw"].float(), "Uinf": float(wip["Uinf"]), "alpha": float(wip["alpha"])}
        atomic_torch_save(static, static_path)

        pn = lat["pn_input"].numpy().copy()
        mask = lat["pn_mask"].numpy()
        for a in (0, 1):            # offsets: their own z-score, valid slots only
            pn[..., a][mask] = z(pn[..., a][mask], st["relpos_mean"][a], st["relpos_std"][a])
        pn[..., PN_SDF][mask] = z(np.abs(pn[..., PN_SDF][mask]), st["query_sdf_mean"], st["query_sdf_std"])
        pn[..., PN_DID][mask] = z(pn[..., PN_DID][mask], st["query_did_mean"], st["query_did_std"])
        pn[..., PN_CURV][mask] = z(pn[..., PN_CURV][mask], st["query_curv_mean"], st["query_curv_std"])
        leaf_pos_geo = lat["leaf_pos"].numpy()
        latent = {"pn_input": torch.from_numpy(pn.astype(np.float32)), "pn_mask": torch.from_numpy(mask.copy()),
                  "leaf_pos": torch.from_numpy(coord_affine(leaf_pos_geo, p5, p95)),
                  "leaf_pos_geo": torch.from_numpy(leaf_pos_geo.astype(np.float32)),
                  "leaf_sdf": torch.from_numpy(z(lat["leaf_sdf"].numpy(), st["latent_sdf_mean"], st["latent_sdf_std"])),
                  "leaf_sdf_grad": lat["leaf_sdf_grad"].float(),
                  "leaf_did": torch.from_numpy(z(lat["leaf_did_raw"].numpy(), st["latent_did_mean"], st["latent_did_std"])),
                  "latent_grid": str(lat.get("latent_grid", "adaptive")),
                  "tree_split_axes": lat["tree_split_axes"], "tree_split_values": lat["tree_split_values"]}
        atomic_torch_save(latent, latent_path)
        return ("ok", name, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("failed", name, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_preprocess(cfg: Config, from_cache: bool = False) -> None:
    pp = cfg.airfrans
    p = paths(cfg)
    structured = cfg.run.layout == "grid"
    n_leaves = int(pp.n_leaves)
    assert 2 ** int(round(np.log2(n_leaves))) == n_leaves and n_leaves <= 65536
    if structured:
        assert pp.grid_dims[0] * pp.grid_dims[1] == n_leaves, "grid_dims must multiply to n_leaves"
    for d in (p["wip"], p["static"], p["latent"]):
        Path(d).mkdir(parents=True, exist_ok=True)
    if p["manifest"].exists():
        with open(p["manifest"], encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = build_manifest(cfg.data.data_root, int(pp.val_seed), int(pp.n_val))
        with open(p["manifest"], "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
    names = list(manifest["train"]) + list(manifest["val"]) + list(manifest["test"])
    print(f"[airfrans] {len(manifest['train'])} / {len(manifest['val'])} / {len(manifest['test'])} cases "
          f"-> {p['cache']} (layout {cfg.run.layout})", flush=True)

    state = dict(wip=str(p["wip"]), data_root=cfg.data.data_root, n_leaves=n_leaves, slots=int(pp.slots),
                 idw_k=int(pp.idw_k), structured=structured,
                 latent_grid="structured" if structured else "adaptive",
                 grid_dims=tuple(pp.grid_dims), domain_lo=tuple(pp.grid_domain_lo),
                 domain_hi=tuple(pp.grid_domain_hi), sdf_chunk=int(pp.sdf_chunk),
                 did_bins=int(pp.did_bins), did_dmax=float(pp.did_dmax), curvature_k=int(pp.curvature_k),
                 coord_sample=int(pp.coord_pctl_sample), threads=int(pp.worker_threads))
    if from_cache:
        if not structured:
            raise SystemExit("--from-cache rebuilds the grid layout from the importance bake: add --layout grid")
        src = paths(cfg, "importance")["wip"]
        missing = [n for n in names if not (src / f"{n}.pt").exists()]
        if missing:
            raise SystemExit(f"source bake incomplete: {len(missing)} cases missing under {src}")
        state["src_wip"] = str(src)
    workers = int(pp.n_workers)
    results = run_pool(names, _bake_worker, _init, (state,), workers, "airfrans:bake")
    failed, _ = report(results, "bake")
    if failed:
        raise SystemExit(f"[airfrans] bake failed on {len(failed)} cases")
    if structured:
        occupancy_summary([r[3] for r in results if r[3] is not None], int(pp.slots), "airfrans:grid")

    stats = compute_norm_stats(state, list(manifest["train"]), workers,
                               tuple(float(v) for v in pp.coord_pctl), int(pp.did_bins))
    with open(p["norm_stats"], "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    results = run_pool(names, _finalize_worker, _init,
                       (dict(state, stats=stats, static=str(p["static"]), latent=str(p["latent"])),),
                       workers, "airfrans:finalize")
    failed, _ = report(results, "finalize")
    if failed:
        raise SystemExit(f"[airfrans] finalize failed on {len(failed)} cases")
    run_calibration(p["static"], p["latent"], p["calibration"], manifest, int(pp.calibration_sample_cases),
                    cfg.ibo.rope_dims, int(pp.idw_k),
                    lambda static, sel: static["decoder_input"].numpy()[sel[static["is_volume"].numpy()[sel]], 2],
                    "airfrans")
    print(f"[airfrans] done -> {p['cache']}", flush=True)

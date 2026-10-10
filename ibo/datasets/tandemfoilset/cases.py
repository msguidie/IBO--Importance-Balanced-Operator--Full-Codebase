"""One subset's cache: per-case tensors, the native point inputs of the
baselines, the per-view training datasets and the evaluation cases.

Cache layout (<cache_root>/<subset>_L<n_leaves>[_structured]/):
  kdtree.pt        subset metadata: tokens, slots per token, domain, RoPE placeholders
  coef_norm.pt     train-split statistics (training space, reporting space, features)
  manifest.json    the split actually baked (datasets/tandemfoilset/splits.json)
  calibration_report.json   RoPE domain and Fourier scale evidence (adopted at launch)
  static/<case>.pt per-point tensors   latent/<case>.pt per-token tensors
  derived/spider_v1/, gaot/            derived caches of the two baselines

Rows of a case are mixed volume / surface in mesh order; the evaluation
cases are re-ordered volume first (the producer contract).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

from ...config import Config
from ...models.erwin.balltree import build_tree, knn_edges, restore_index
from ...models.gaot import gather_csr_rows, merge_csr
from ...models.spider import onion_shell_ids
from ..common.sampling import stratified_rows, uniform_rows
from .gaot_derive import gaot_pndata, load_gaot_graphs
from .spider_derive import load_spider_labels

CACHE_VERSION = 13          # file schema of the caches the paper's runs used
UNIFIED_REF = 8             # Transolver / SpiderSolver: distances to an 8 x 8 grid over the domain
INNER_FRACTION = 0.10       # SpiderSolver: innermost tenth of the sampled volume rows
P = 2                       # column of p in (u, v, p): the surface target


def cache_dir(cfg: Config, layout: str | None = None) -> Path:
    layout = cfg.run.layout if layout is None else layout
    suffix = "_structured" if layout == "grid" else ""
    return Path(cfg.data.cache_root) / f"{cfg.run.subset}_L{cfg.tandemfoilset.n_leaves}{suffix}"


def _load(path: Path, what: str, weights_only: bool = False) -> dict:
    if not path.exists():
        raise SystemExit(f"{what} missing: {path}. Run `python -m ibo preprocess --dataset "
                         f"tandemfoilset --subset <subset>` first")
    d = torch.load(str(path), map_location="cpu", weights_only=weights_only)
    if d.get("version") != CACHE_VERSION:
        raise SystemExit(f"{path}: cache version {d.get('version')!r} != {CACHE_VERSION}; "
                         f"delete and rebuild the cache")
    return d


def load_meta(cache: Path) -> dict:
    return _load(cache / "kdtree.pt", "cache metadata")


def load_coef_norm(cache: Path) -> dict:
    return _load(cache / "coef_norm.pt", "normalisation statistics")


def load_case(cache: Path, name: str) -> dict:
    static = _load(cache / "static" / f"{name}.pt", "case", weights_only=True)
    latent = _load(cache / "latent" / f"{name}.pt", "case", weights_only=True)
    surf = static["surf"].bool()
    case = {
        "name": name, "n_points": int(static["n_points"]),
        "umag": float(static["umag_phys"]), "aoa_deg": float(static["aoa_deg"]),
        "re": float(static["re_phys"]), "nu": float(static["nu_phys"]),
        "cond": static["cond"].float(),
        "norm_pos": static["norm_pos"].float(), "query_pos_geo": static["query_pos_geo"].float(),
        "sdf1": static["sdf1"].float(), "sdf_grad1": static["sdf_grad1"].float(),
        "sdf2": static["sdf2"].float(), "sdf_grad2": static["sdf_grad2"].float(),
        "did": static["did"].float(), "curvature": static["curvature"].float(),
        "idw_indices": static["idw_indices"].to(torch.int32),     # uint16 on disk
        "target": static["target"].float(), "surf": surf, "is_volume": ~surf,
        "boundary": static["boundary"], "poly1": static["poly1"].float(),
        "poly2": static["poly2"].float(),
    }
    for k in ("pn_input", "pn_mask", "leaf_norm_pos", "leaf_pos_geo", "leaf_sdf1",
              "leaf_sdf_grad1", "leaf_sdf2", "leaf_sdf_grad2", "leaf_did"):
        case[k] = latent[k]
    return case


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def latent_stats(case: dict) -> torch.Tensor:
    """[L, 16] per-token descriptor: the two signed distances and their
    gradients at the token position, the position, the distance profile."""
    return torch.cat([case["leaf_sdf1"].float()[:, None], case["leaf_sdf_grad1"].float(),
                      case["leaf_sdf2"].float()[:, None], case["leaf_sdf_grad2"].float(),
                      case["leaf_norm_pos"].float(), case["leaf_did"].float()], dim=-1)


def query_pos_sdf(case: dict, sdf_scale: float, sdf_clip: float | None) -> torch.Tensor:
    """[N, 6] Fourier group of the query self-embedding: the position and,
    per airfoil, gradient x |z-scored distance| (clipped, scaled)."""
    za1, za2 = case["sdf1"].abs(), case["sdf2"].abs()
    if sdf_clip is not None:
        za1, za2 = za1.clamp(max=sdf_clip), za2.clamp(max=sdf_clip)
    return torch.cat([case["norm_pos"], case["sdf_grad1"] * (za1 * sdf_scale)[:, None],
                      case["sdf_grad2"] * (za2 * sdf_scale)[:, None]], dim=-1)


def native7(case: dict, coef: dict) -> torch.Tensor:
    """[N, 7] the AirfRANS-convention input of Transolver / SpiderSolver:
    position (affine), inflow vector / mean train speed, nearest-wall
    signed distance (z-scored with the front-airfoil statistics), the own
    outward normal on surface rows (zero elsewhere)."""
    n = case["n_points"]
    q1_m, q1_s = float(coef["query_sdf1_mean"]), max(float(coef["query_sdf1_std"]), 1e-8)
    q2_m, q2_s = float(coef["query_sdf2_mean"]), max(float(coef["query_sdf2_std"]), 1e-8)
    raw1 = case["sdf1"].double() * q1_s + q1_m
    raw2 = case["sdf2"].double() * q2_s + q2_m
    sdf_min_z = ((torch.minimum(raw1, raw2) - q1_m) / q1_s).float()
    u_ref = max(abs(float(coef["mean_cond"][0])), 1e-8)
    aoa = math.radians(case["aoa_deg"])
    uvec = torch.tensor([case["umag"] * math.cos(aoa) / u_ref,
                         case["umag"] * math.sin(aoa) / u_ref], dtype=torch.float32)
    normals = torch.zeros(n, 2, dtype=torch.float32)
    front, rear = case["boundary"] == 5, case["boundary"] == 6
    normals[front] = case["sdf_grad1"][front]
    normals[rear] = case["sdf_grad2"][rear]
    return torch.cat([case["norm_pos"], uvec.expand(n, 2), sdf_min_z[:, None], normals],
                     dim=-1).contiguous()


def wall_distance(case: dict, coef: dict) -> torch.Tensor:
    """[N] distance to the nearest airfoil, physical units."""
    raw1 = (case["sdf1"].double() * max(float(coef["query_sdf1_std"]), 1e-8)
            + float(coef["query_sdf1_mean"])).abs()
    raw2 = (case["sdf2"].double() * max(float(coef["query_sdf2_std"]), 1e-8)
            + float(coef["query_sdf2_mean"])).abs()
    return torch.minimum(raw1, raw2).float()


def unified_grid(meta: dict, ref: int = UNIFIED_REF) -> torch.Tensor:
    """[ref^2, 2] grid over the subset's physical domain."""
    lo, hi = meta["rope_domain_lo"], meta["rope_domain_hi"]
    xs = torch.linspace(float(lo[0]), float(hi[0]), ref, dtype=torch.float64)
    ys = torch.linspace(float(lo[1]), float(hi[1]), ref, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).float()


def unified_features(pos_geo: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """[N, ref^2] raw distances of every point to the grid."""
    return torch.cdist(pos_geo, grid.to(pos_geo.device))


def erwin_graph(pos: torch.Tensor, ec) -> dict:
    """Ball tree, restore index and kNN edges of one sample, built on the CPU."""
    ti, tm, rot = build_tree(pos, ec.strides, ec.ball_sizes, ec.rotate)
    return {"tree_idx": ti, "tree_mask": tm, "tree_idx_rot": rot, "restore_idx": restore_index(ti, tm),
            "edge_index": knn_edges(pos, int(ec.knn)) if ec.mp_steps > 0 else None}


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def spider_rows(c: dict, vsel: torch.Tensor, ssel: torch.Tensor, grid: torch.Tensor, sc):
    """(x [Nv + Ns, 71], vol_gid [Nv], inner_idx) of a volume-first row set."""
    rows = torch.cat([vsel, ssel])
    x = torch.cat([c["x7"][rows], unified_features(c["pos_geo"][rows], grid)], dim=-1)
    wall = c["wall"][vsel]
    gid = onion_shell_ids(wall, int(sc.onion_num)) * int(sc.n_clusters) + c["cluster"][vsel]
    inner = torch.topk(wall, min(int(sc.n_inner), int(vsel.shape[0])), largest=False).indices.sort().values
    return x, gid, inner


class TandemTrainSet(TorchDataset):
    """This rank's training cases, resident in RAM; __getitem__((item,
    epoch)) draws the case's sample of that epoch. mode: ibo | transolver |
    spider | erwin | abupt | gaot."""

    def __init__(self, cfg: Config, names: list[str], mode: str, seed_salt: str):
        self.cfg, self.mode, self.salt = cfg, mode, str(seed_salt)
        self.n_query = int(cfg.training.n_query)
        cache = cache_dir(cfg)
        coef = load_coef_norm(cache)
        self.grid = unified_grid(load_meta(cache)) if mode in ("transolver", "spider") else None
        print(f"[data] loading {len(names)} cases into RAM ({mode} view)...", flush=True)
        self.cases = [self._prepare(load_case(cache, n), cache, coef) for n in names]
        gb = sum(v.numel() * v.element_size() for c in self.cases for v in c.values()
                 if isinstance(v, torch.Tensor)) / 1024 ** 3
        print(f"[data] resident: {gb:.1f} GiB ({len(self.cases)} cases)", flush=True)

    def _prepare(self, case: dict, cache: Path, coef: dict) -> dict:
        c = {"name": case["name"], "n_points": case["n_points"], "cond": case["cond"],
             "target": case["target"], "is_volume": case["is_volume"]}
        m = self.mode
        if m == "ibo":
            ib = self.cfg.ibo
            c.update(latent_slots=case["pn_input"], latent_mask=case["pn_mask"],
                     latent_stats=latent_stats(case), latent_pos_geo=case["leaf_pos_geo"].float(),
                     query_pos_geo=case["query_pos_geo"],
                     pos_sdf=query_pos_sdf(case, ib.fourier_sdf_scale, ib.fourier_sdf_clip),
                     did=case["did"], curvature=case["curvature"][:, None],
                     idw_indices=case["idw_indices"])
        elif m in ("transolver", "spider"):
            c.update(x7=native7(case, coef), pos_geo=case["query_pos_geo"])
            if m == "spider":
                c.update(wall=wall_distance(case, coef),
                         cluster=load_spider_labels(cache, case["name"], case["n_points"]).long(),
                         vol_rows=case["is_volume"].nonzero().flatten(),
                         surf_rows=case["surf"].nonzero().flatten())
        elif m == "erwin":
            c["norm_pos"] = case["norm_pos"]
        elif m == "abupt":
            c.update(pos_geo=case["query_pos_geo"], vol_rows=case["is_volume"].nonzero().flatten(),
                     surf_rows=case["surf"].nonzero().flatten())
        elif m == "gaot":
            c.update(pndata=gaot_pndata(case), x_coords=case["norm_pos"])
            c.update(load_gaot_graphs(cache, case["name"]))
        else:
            raise ValueError(f"tandemfoilset: unknown view {m!r}")
        return c

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, item) -> dict:
        idx, epoch = item
        c = self.cases[idx]
        return getattr(self, f"_{self.mode}")(c, int(epoch))

    def _ibo(self, c, epoch):
        sel = uniform_rows(c["name"] + self.salt, c["n_points"], self.n_query, epoch)
        return {"latent_slots": c["latent_slots"], "latent_mask": c["latent_mask"],
                "latent_stats": c["latent_stats"], "latent_pos_geo": c["latent_pos_geo"],
                "cond": c["cond"], "query_pos_geo": c["query_pos_geo"][sel],
                "query_feats": (c["pos_sdf"][sel], c["did"][sel], c["curvature"][sel]),
                "query_knn": c["idw_indices"][sel], "target_vol": c["target"][sel],
                "target_surf": c["target"][sel, P:P + 1], "is_volume": c["is_volume"][sel]}

    def _transolver(self, c, epoch):
        sel = uniform_rows(c["name"], c["n_points"], self.n_query, epoch)
        x = torch.cat([c["x7"][sel], unified_features(c["pos_geo"][sel], self.grid)], dim=-1)
        return {"x": x, "target_vol": c["target"][sel], "target_surf": c["target"][sel, P:P + 1],
                "is_volume": c["is_volume"][sel]}

    def _spider(self, c, epoch):
        sc = self.cfg.spider
        quota = int(sc.surf_quota)
        vsel, ssel = stratified_rows(c["name"], c["vol_rows"], c["surf_rows"], self.n_query - quota,
                                     quota, epoch, "__spider")
        x, gid, inner = spider_rows(c, vsel, ssel, self.grid, sc)
        return {"x": x, "vol_gid": gid, "inner_idx": inner, "surf_region_id": None,
                "target_vol": c["target"][vsel], "target_surf": c["target"][ssel, P:P + 1]}

    def _erwin(self, c, epoch):
        sel = uniform_rows(c["name"], c["n_points"], self.n_query, epoch)
        pos = c["norm_pos"][sel]
        out = {"pos": pos, "extra": c["cond"].view(1, 2).expand(pos.shape[0], 2).contiguous(),
               "target_vol": c["target"][sel], "target_surf": c["target"][sel, P:P + 1],
               "is_volume": c["is_volume"][sel]}
        out.update(erwin_graph(pos, self.cfg.erwin))
        return out

    def _abupt(self, c, epoch):
        ac = self.cfg.abupt
        vsel, ssel = stratified_rows(c["name"] + self.salt, c["vol_rows"], c["surf_rows"],
                                     int(ac.vol_anchors), int(ac.surf_anchors), epoch, "__abupt")
        return {"surface_position": c["pos_geo"][ssel], "volume_position": c["pos_geo"][vsel],
                "cond": c["cond"], "target_surf": c["target"][ssel, P:P + 1],
                "target_vol": c["target"][vsel]}

    def _gaot(self, c, epoch):
        sel = uniform_rows(c["name"], c["n_points"], self.n_query, epoch)
        idx, spl, alp = gather_csr_rows(c["dec_index"], c["dec_splits"], c["dec_alpha"], sel)
        return {"n_points": c["n_points"], "pndata": c["pndata"], "x_coords": c["x_coords"],
                "enc_index": c["enc_index"], "enc_splits": c["enc_splits"],
                "enc_alpha": c["enc_alpha"], "enc_geo": c["enc_geo"],
                "dec_index": idx, "dec_splits": spl, "dec_alpha": alp, "dec_geo": c["dec_geo"][sel],
                "query_coords": c["x_coords"][sel], "target_vol": c["target"][sel],
                "target_surf": c["target"][sel, P:P + 1], "is_volume": c["is_volume"][sel]}


def collate_gaot(samples: list[dict]) -> dict:
    """Disjoint-union merge of B per-case graphs: the encoder gathers from
    the concatenated clouds into B x n_tokens grid rows, the decoder from
    those rows into the sampled points."""
    n_tokens = int(samples[0]["enc_splits"].shape[0]) - 1
    node_offsets, off = [], 0
    for s in samples:
        node_offsets.append(off)
        off += int(s["n_points"])
    enc_index, enc_splits, enc_alpha = merge_csr(
        [s["enc_index"] for s in samples], [s["enc_splits"] for s in samples],
        [s["enc_alpha"] for s in samples], node_offsets)
    dec_index, dec_splits, dec_alpha = merge_csr(
        [s["dec_index"] for s in samples], [s["dec_splits"] for s in samples],
        [s["dec_alpha"] for s in samples], [b * n_tokens for b in range(len(samples))])
    return {"batch_size": len(samples),
            "pndata": torch.cat([s["pndata"] for s in samples]),
            "x_coords": torch.cat([s["x_coords"] for s in samples]),
            "enc_index": enc_index, "enc_splits": enc_splits, "enc_alpha": enc_alpha,
            "enc_geo": torch.stack([s["enc_geo"] for s in samples]),
            "query_coords": torch.cat([s["query_coords"] for s in samples]),
            "dec_index": dec_index, "dec_splits": dec_splits, "dec_alpha": dec_alpha,
            "dec_geo": torch.cat([s["dec_geo"] for s in samples]),
            "target_vol": torch.stack([s["target_vol"] for s in samples]),
            "target_surf": torch.stack([s["target_surf"] for s in samples]),
            "is_volume": torch.stack([s["is_volume"] for s in samples])}


# ---------------------------------------------------------------------------
# evaluation cases (every point once, volume rows first)
# ---------------------------------------------------------------------------

def eval_case(cfg: Config, cache: Path, name: str, view: str, coef: dict, meta: dict,
              grid: torch.Tensor | None) -> dict:
    case = load_case(cache, name)
    vol_rows = case["is_volume"].nonzero().flatten()
    surf_rows = case["surf"].nonzero().flatten()
    perm = torch.cat([vol_rows, surf_rows])
    n_vol, n_surf = int(vol_rows.shape[0]), int(surf_rows.shape[0])
    out = {"name": name, "n_vol": n_vol, "n_surf": n_surf,
           "target_vol": case["target"][vol_rows], "target_surf": case["target"][surf_rows, P:P + 1],
           "meta": {"umag": case["umag"], "aoa_deg": case["aoa_deg"], "re": case["re"]}}
    model = cfg.run.model
    if view == "ibo":
        ib = cfg.ibo
        out.update(latent_slots=case["pn_input"], latent_mask=case["pn_mask"],
                   latent_stats=latent_stats(case), latent_pos_geo=case["leaf_pos_geo"].float(),
                   cond=case["cond"], query_pos_geo=case["query_pos_geo"][perm],
                   query_feats=(query_pos_sdf(case, ib.fourier_sdf_scale, ib.fourier_sdf_clip)[perm],
                                case["did"][perm], case["curvature"][perm, None]),
                   query_knn=case["idw_indices"][perm],
                   is_volume=torch.cat([torch.ones(n_vol, dtype=torch.bool),
                                        torch.zeros(n_surf, dtype=torch.bool)]))
    elif view == "points":
        out["x"] = torch.cat([native7(case, coef)[perm],
                              unified_features(case["query_pos_geo"][perm], grid)], dim=-1)
        if model == "spider":
            labels = load_spider_labels(cache, name, case["n_points"]).long()
            out["spider"] = {"wall": wall_distance(case, coef)[vol_rows],
                             "vol_cluster": labels[vol_rows], "surf_region": None}
    elif view == "erwin":
        out.update(pos=case["norm_pos"][perm],
                   extra=case["cond"].view(1, 2).expand(n_vol + n_surf, 2).contiguous())
    elif view == "abupt":
        out.update(pos=case["query_pos_geo"][perm], cond=case["cond"])
    elif view == "gaot":
        g = load_gaot_graphs(cache, name)
        idx, spl, alp = gather_csr_rows(g["dec_index"], g["dec_splits"], g["dec_alpha"], perm)
        out["gaot"] = {"batch_size": 1, "pndata": gaot_pndata(case), "x_coords": case["norm_pos"],
                       "enc_index": g["enc_index"], "enc_splits": g["enc_splits"],
                       "enc_alpha": g["enc_alpha"], "enc_geo": g["enc_geo"].unsqueeze(0),
                       "query_coords": case["norm_pos"][perm], "dec_index": idx,
                       "dec_splits": spl, "dec_alpha": alp, "dec_geo": g["dec_geo"][perm]}
    else:
        raise ValueError(f"tandemfoilset: unknown view {view!r}")
    return out

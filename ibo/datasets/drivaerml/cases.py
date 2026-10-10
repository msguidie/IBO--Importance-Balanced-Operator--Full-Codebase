"""The DrivAerML cache: per-case tensors, the IBO training view and the
evaluation cases.

Layout under <cache_root>:
  wip/<case>.pt                       physical-unit intermediates
  norm_stats.json                     train-split statistics
  cache/C<n>_SC<slots>_idw<k>_fp32_sdf7d[_structured]/{static,latent}/<case>.pt
  (grid layout: ... /latent_stats.json, its own token statistics)

A static file holds the case's query pool, volume rows first: training
cases carry one volume point per subregion (2^22) plus up to five million
surface points, held-out cases a uniform sample of the volume and (by
default) every surface point. decoder_input is [x, y, z (affine), signed
z-scored wall distance, unit direction (outward normal on the wall)].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ...config import Config
from ..common.sampling import uniform_rows

SCHEMA = "sdf7d"
VOL_CACHE = ("Ux", "Uy", "Uz", "p", "nut")              # channels stored in the cache
SURF_CACHE = ("p_surf", "wss_x", "wss_y", "wss_z")


def paths(cfg: Config, layout: str | None = None) -> dict:
    pp = cfg.drivaerml
    layout = cfg.run.layout if layout is None else layout
    suffix = "_structured" if layout == "grid" else ""
    root = Path(cfg.data.cache_root)
    cache = root / "cache" / f"C{pp.n_leaves}_SC{pp.slots}_idw{pp.idw_k}_fp32_{SCHEMA}{suffix}"
    return {"root": root, "wip": root / "wip", "wip_grid": root / "wip_structured_grid",
            "norm_stats": root / "norm_stats.json", "cache": cache,
            "static": cache / "static", "latent": cache / "latent",
            "latent_stats": cache / "latent_stats.json"}


def load_norm_stats(path: Path) -> dict:
    if not Path(path).exists():
        raise SystemExit(f"{path} missing: run `python -m ibo preprocess --dataset drivaerml` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def coord_affine(x: torch.Tensor, p5, p95) -> torch.Tensor:
    """(x - p5) / (p95 - p5) per axis in float64, as the bake wrote it."""
    p5 = torch.as_tensor(p5, dtype=torch.float64)
    p95 = torch.as_tensor(p95, dtype=torch.float64)
    return ((x.double() - p5) / torch.clamp(p95 - p5, min=1e-8)).float()


def load_case(p: dict, name: str, vol_idx: list[int], knn: int, stats: dict,
              surface_meta: bool = True) -> dict:
    static = torch.load(str(p["static"] / f"{name}.pt"), map_location="cpu", weights_only=True)
    latent = torch.load(str(p["latent"] / f"{name}.pt"), map_location="cpu", weights_only=True)
    if static.get("schema") != SCHEMA or static.get("layout") != "vol_first":
        raise SystemExit(f"{name}: cache schema {static.get('schema')!r}; rebuild the cache")
    idw = static["idw_indices"]
    if idw.shape[1] < knn:
        raise SystemExit(f"{name}: the cache stores {idw.shape[1]} neighbours per point, ibo.knn={knn}")
    case = {"name": name, "n_vol": int(static["n_vol"]), "n_surf": int(static["n_surf"]),
            "n_surf_raw": int(static.get("n_surf_raw", static["n_surf"])),
            "a_ref": float(static["a_ref"]), "cd_true": float(static["cd_true"]),
            "cl_true": float(static["cl_true"]),
            "query_pos_geo": static["query_pos_geo"].float(),
            "decoder_feat": static["decoder_input"][:, 3:7].contiguous(),   # z distance, direction
            "idw_indices": idw[:, :knn].contiguous(),                       # uint16
            "target_vol": static["target_vol"][:, vol_idx].contiguous().float(),
            "target_surf": static["target_surf"].float(),
            "coord_p5": stats["coord_p5"], "coord_p95": stats["coord_p95"],
            "pn_feat": latent["pn_feat"], "cell_pos": latent["cell_pos"],
            "cell_pos_geo": latent["cell_pos_geo"], "cell_sdf": latent["cell_sdf"]}
    if surface_meta:
        case["surface_normal"] = static["surface_normal"].float()
        case["surface_area"] = static["surface_area"].float()
    return case


def latent_view(case: dict, mask: torch.Tensor) -> dict:
    return {"latent_slots": case["pn_feat"], "latent_mask": mask,
            "latent_stats": torch.cat([case["cell_pos"].float(), case["cell_sdf"].float()], dim=-1),
            "latent_pos_geo": case["cell_pos_geo"].float(), "cond": None}


def query_feats(case: dict, sel, sdf_scale: float, sdf_clip: float | None):
    """(signed wall distance [n, 1], direction [n, 3], affine position [n, 3])
    of the rows `sel` (None = all)."""
    feat = case["decoder_feat"] if sel is None else case["decoder_feat"][sel]
    pos = case["query_pos_geo"] if sel is None else case["query_pos_geo"][sel]
    z = feat[:, 0:1].float()
    if sdf_clip is not None:
        z = z.clamp(max=sdf_clip)
    return (z * sdf_scale, feat[:, 1:4].float(), coord_affine(pos, case["coord_p5"], case["coord_p95"]))


def knn_rows(case: dict, sel) -> torch.Tensor:
    """uint16 neighbour table rows -> int32 (gathered through numpy)."""
    idw = case["idw_indices"].numpy()
    return torch.from_numpy((idw if sel is None else idw[sel.numpy()]).astype(np.int32))


class DrivaermlTrainSet(TorchDataset):
    """This rank's training cases, resident in RAM (targets in fp16);
    __getitem__((item, epoch)) draws n_query rows of the case's pool."""

    def __init__(self, cfg: Config, names: list[str], vol_idx: list[int], seed_salt: str):
        self.cfg, self.salt = cfg, str(seed_salt)
        self.n_query = int(cfg.training.n_query)
        p = paths(cfg)
        stats = load_norm_stats(p["norm_stats"])
        print(f"[data] loading {len(names)} cases into RAM...", flush=True)
        self.cases = [load_case(p, n, vol_idx, int(cfg.ibo.knn), stats, surface_meta=False) for n in names]
        for c in self.cases:
            c["target_vol"], c["target_surf"] = c["target_vol"].half(), c["target_surf"].half()
        pn = self.cases[0]["pn_feat"]
        self.mask = torch.ones(pn.shape[0], pn.shape[1], dtype=torch.bool)   # every slot filled
        gb = sum(v.numel() * v.element_size() for c in self.cases for v in c.values()
                 if isinstance(v, torch.Tensor)) / 1024 ** 3
        print(f"[data] resident: {gb:.1f} GiB ({len(self.cases)} cases)", flush=True)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, item) -> dict:
        idx, epoch = item
        c = self.cases[idx]
        ib = self.cfg.ibo
        n_vol = c["n_vol"]
        sel = uniform_rows(c["name"] + self.salt, n_vol + c["n_surf"], self.n_query, int(epoch))
        is_volume = sel < n_vol
        target_vol = torch.zeros(self.n_query, c["target_vol"].shape[1], dtype=torch.float32)
        target_surf = torch.zeros(self.n_query, c["target_surf"].shape[1], dtype=torch.float32)
        target_vol[is_volume] = c["target_vol"][sel[is_volume]].float()
        target_surf[~is_volume] = c["target_surf"][sel[~is_volume] - n_vol].float()
        out = latent_view(c, self.mask)
        out.update(query_pos_geo=c["query_pos_geo"][sel],
                   query_feats=query_feats(c, sel, ib.fourier_sdf_scale, ib.fourier_sdf_clip),
                   query_knn=knn_rows(c, sel), target_vol=target_vol, target_surf=target_surf,
                   is_volume=is_volume)
        return out


def eval_case(cfg: Config, p: dict, name: str, vol_idx: list[int], stats: dict) -> dict:
    c = load_case(p, name, vol_idx, int(cfg.ibo.knn), stats, surface_meta=True)
    ib = cfg.ibo
    n_vol, n_surf = c["n_vol"], c["n_surf"]
    pn = c["pn_feat"]
    out = {"name": name, "n_vol": n_vol, "n_surf": n_surf,
           "target_vol": c["target_vol"], "target_surf": c["target_surf"],
           "meta": {"a_ref": c["a_ref"], "cd_true": c["cd_true"], "cl_true": c["cl_true"],
                    "n_surf_raw": c["n_surf_raw"], "surface_normal": c["surface_normal"],
                    "surface_area": c["surface_area"]}}
    out.update(latent_view(c, torch.ones(pn.shape[0], pn.shape[1], dtype=torch.bool)))
    out.update(query_pos_geo=c["query_pos_geo"],
               query_feats=query_feats(c, None, ib.fourier_sdf_scale, ib.fourier_sdf_clip),
               query_knn=knn_rows(c, None),
               is_volume=torch.cat([torch.ones(n_vol, dtype=torch.bool),
                                    torch.zeros(n_surf, dtype=torch.bool)]))
    return out

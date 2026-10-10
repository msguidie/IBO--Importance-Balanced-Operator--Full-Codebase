"""The AirfRANS cache: per-case tensors, the training views (IBO, AB-UPT)
and the evaluation cases.

Layout under <cache_root> (the suffix is "_structured" for layout grid):
  manifest.json                         720 / 80 / 200 case names
  wip<suffix>/<case>.pt                 physical-unit intermediates
  norm_stats<suffix>.json               train-split statistics
  calibration_report<suffix>.json       RoPE domain and Fourier scale (adopted at launch)
  cache/L<n>_K<slots>_idw<k>_did15d<suffix>/{static,latent}/<case>.pt

Rows are in mesh order with an is_volume mask; evaluation cases are
re-ordered volume first.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

from ...config import Config
from ..common.sampling import stratified_rows, uniform_rows

SCHEMA = "did15d"
P = 2                       # column of p in (u_x, u_y, p, nut): the surface target


def paths(cfg: Config, layout: str | None = None) -> dict:
    pp = cfg.airfrans
    layout = cfg.run.layout if layout is None else layout
    suffix = "_structured" if layout == "grid" else ""
    root = Path(cfg.data.cache_root)
    key = f"L{pp.n_leaves}_K{pp.slots}_idw{pp.idw_k}_{SCHEMA}{suffix}"
    return {"root": root, "manifest": root / "manifest.json", "wip": root / f"wip{suffix}",
            "norm_stats": root / f"norm_stats{suffix}.json",
            "calibration": root / f"calibration_report{suffix}.json",
            "cache": root / "cache" / key, "static": root / "cache" / key / "static",
            "latent": root / "cache" / key / "latent"}


def load_norm_stats(path: Path) -> dict:
    if not Path(path).exists():
        raise SystemExit(f"{path} missing: run `python -m ibo preprocess --dataset airfrans` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_case(p: dict, name: str) -> dict:
    static = torch.load(str(p["static"] / f"{name}.pt"), map_location="cpu", weights_only=True)
    latent = torch.load(str(p["latent"] / f"{name}.pt"), map_location="cpu", weights_only=True)
    if static.get("schema") != SCHEMA:
        raise SystemExit(f"{name}: cache schema {static.get('schema')!r} != {SCHEMA!r}; rebuild the cache")
    case = {"name": name, "n_points": int(static["decoder_input"].shape[0]),
            "Uinf": float(static["Uinf"]), "alpha": float(static["alpha"]),
            "cond": static["uinf"].float(),
            "decoder_input": static["decoder_input"].float(),     # x, y (affine), |sdf|/sigma, dir
            "did": static["did"].float(), "curvature": static["curvature"].float(),
            "query_pos_geo": static["query_pos_geo"].float(),
            "idw_indices": static["idw_indices"].to(torch.int32),  # uint16 on disk
            "target": static["target"].float(), "is_volume": static["is_volume"].bool()}
    for k in ("pn_input", "pn_mask", "leaf_pos", "leaf_pos_geo", "leaf_sdf", "leaf_sdf_grad", "leaf_did"):
        case[k] = latent[k]
    return case


def latent_stats(case: dict) -> torch.Tensor:
    """[L, 13] per-token descriptor: position (affine), signed distance and
    its gradient at the token, the distance profile."""
    return torch.cat([case["leaf_pos"].float(), case["leaf_sdf"].float()[:, None],
                      case["leaf_sdf_grad"].float(), case["leaf_did"].float()], dim=-1)


def query_pos_sdf(case: dict, sdf_scale: float, sdf_clip: float | None) -> torch.Tensor:
    """[N, 4] Fourier group of the query self-embedding: the affine position
    and direction x |scaled wall distance| (exactly zero on the wall)."""
    dec = case["decoder_input"]
    mag = dec[:, 2].abs()
    if sdf_clip is not None:
        mag = mag.clamp(max=sdf_clip)
    return torch.cat([dec[:, :2], dec[:, 3:5] * (mag * sdf_scale)[:, None]], dim=-1)


class AirfransTrainSet(TorchDataset):
    """This rank's training cases, resident in RAM; __getitem__((item,
    epoch)) draws the case's sample of that epoch. mode: ibo | abupt."""

    def __init__(self, cfg: Config, names: list[str], mode: str, seed_salt: str):
        self.cfg, self.mode, self.salt = cfg, mode, str(seed_salt)
        self.n_query = int(cfg.training.n_query)
        p = paths(cfg)
        print(f"[data] loading {len(names)} cases into RAM ({mode} view)...", flush=True)
        self.cases = [self._prepare(load_case(p, n)) for n in names]
        gb = sum(v.numel() * v.element_size() for c in self.cases for v in c.values()
                 if isinstance(v, torch.Tensor)) / 1024 ** 3
        print(f"[data] resident: {gb:.1f} GiB ({len(self.cases)} cases)", flush=True)

    def _prepare(self, case: dict) -> dict:
        c = {"name": case["name"], "n_points": case["n_points"], "cond": case["cond"],
             "target": case["target"], "is_volume": case["is_volume"]}
        if self.mode == "ibo":
            ib = self.cfg.ibo
            c.update(latent_slots=case["pn_input"], latent_mask=case["pn_mask"],
                     latent_stats=latent_stats(case), latent_pos_geo=case["leaf_pos_geo"].float(),
                     query_pos_geo=case["query_pos_geo"],
                     pos_sdf=query_pos_sdf(case, ib.fourier_sdf_scale, ib.fourier_sdf_clip),
                     did=case["did"], curvature=case["curvature"][:, None],
                     idw_indices=case["idw_indices"])
        elif self.mode == "abupt":
            c.update(pos_geo=case["query_pos_geo"], vol_rows=case["is_volume"].nonzero().flatten(),
                     surf_rows=(~case["is_volume"]).nonzero().flatten())
        else:
            raise ValueError(f"airfrans: unknown view {self.mode!r}")
        return c

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, item) -> dict:
        idx, epoch = item
        c = self.cases[idx]
        if self.mode == "ibo":
            sel = uniform_rows(c["name"] + self.salt, c["n_points"], self.n_query, int(epoch))
            return {"latent_slots": c["latent_slots"], "latent_mask": c["latent_mask"],
                    "latent_stats": c["latent_stats"], "latent_pos_geo": c["latent_pos_geo"],
                    "cond": c["cond"], "query_pos_geo": c["query_pos_geo"][sel],
                    "query_feats": (c["pos_sdf"][sel], c["did"][sel], c["curvature"][sel]),
                    "query_knn": c["idw_indices"][sel], "target_vol": c["target"][sel],
                    "target_surf": c["target"][sel, P:P + 1], "is_volume": c["is_volume"][sel]}
        ac = self.cfg.abupt
        vsel, ssel = stratified_rows(c["name"] + self.salt, c["vol_rows"], c["surf_rows"],
                                     int(ac.vol_anchors), int(ac.surf_anchors), int(epoch), "__abupt")
        return {"surface_position": c["pos_geo"][ssel], "volume_position": c["pos_geo"][vsel],
                "cond": c["cond"], "target_surf": c["target"][ssel, P:P + 1],
                "target_vol": c["target"][vsel]}


def eval_case(cfg: Config, p: dict, name: str, view: str) -> dict:
    """Every point once, volume rows first; meta.perm maps the re-ordered
    rows back to mesh order for the coefficient integration."""
    case = load_case(p, name)
    vol_rows = case["is_volume"].nonzero().flatten()
    surf_rows = (~case["is_volume"]).nonzero().flatten()
    perm = torch.cat([vol_rows, surf_rows])
    n_vol, n_surf = int(vol_rows.shape[0]), int(surf_rows.shape[0])
    out = {"name": name, "n_vol": n_vol, "n_surf": n_surf,
           "target_vol": case["target"][vol_rows], "target_surf": case["target"][surf_rows, P:P + 1],
           "meta": {"Uinf": case["Uinf"], "alpha": case["alpha"], "perm": perm}}
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
    elif view == "abupt":
        out.update(pos=case["query_pos_geo"][perm], cond=case["cond"])
    else:
        raise ValueError(f"airfrans: unknown view {view!r}")
    return out

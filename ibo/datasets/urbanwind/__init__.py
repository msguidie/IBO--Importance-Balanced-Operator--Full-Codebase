"""UrbanWind: steady RANS over residential estates, four wind directions per
estate, in the frame where the mean wind blows toward -y. Raw cases are
`<estate>_<dir>.pt` files (keys volume_pos, volume_fields [Ux, Uy, Uz, p,
nut], surface_pos, surface_fields [p, wss_xyz], surface_normals,
volume_sdf, volume_sdf_grad); preprocessing writes one pack per estate
(geometry once, targets per direction) into the cache. Splits are per
estate (datasets/urbanwind/splits.json); the normalisation constants are
the paper's (datasets/urbanwind/norm_stats.json).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ...config import Config, repo_root
from ...training.data import collate_single, collate_tree
from ..base import Dataset, DatasetSpec, FeatureGroup
from .abupt_estate import RamEstateDatasetABUPT
from .estate import (SURF_CHANNEL_NAMES, VOL_CHANNEL_NAMES, RamEstateDataset,
                     cache_dir, estate_of, load_estate, materialize,
                     norm_stats_path, read_json, stream_full_estate_cases)
from .metrics import (case_metrics, inverse_transform_surf, inverse_transform_vol,
                      load_norm_stats)
from .pointcloud import (PointcloudDataset, attach_spider_derived, build_spider_derived,
                         spider_derived_dir, surf_region_classes)

SPEC = DatasetSpec(
    name="urbanwind", dim=3,
    vol_channels=list(VOL_CHANNEL_NAMES), surf_channels=list(SURF_CHANNEL_NAMES),
    cond_dim=5,                   # morphology conditioners (adaLN); IBO only
    cond_to_baselines=False,
    slot_feat_dim=13,             # rel pos 3, sdf, sdf direction 3, ray support 6
    stats_dim=22,                 # 19 leaf statistics + the token position
    query_groups=[
        FeatureGroup("xy", 4, "fourier", n_freqs=8),      # x, y, dir_x, dir_y
        FeatureGroup("z", 3, "fourier", n_freqs=6),       # log(max(z, 0.1 m)), dir_z, sdf
        FeatureGroup("support", 6, "plain"),              # six axis-aligned ray distances
        FeatureGroup("point_type", 1, "embed", n_classes=2, embed_dim=4),
        FeatureGroup("region", 1, "embed", n_classes=5, embed_dim=4),
        FeatureGroup("edge", 1, "embed", n_classes=10, embed_dim=4),
    ],
    pair_fourier_freqs=(8, 8, 6), pair_unit_dir=True, pair_log_dbar=True,
    native_point_dim=7,           # pos 3, sdf, sdf direction 3
    loss_vol_weights=(2.0, 1.0, 2.0, 1.0, 1.0),
    metric_names=[*VOL_CHANNEL_NAMES, *SURF_CHANNEL_NAMES],
)


def _query_feats(pos, sdf, sdf_dir, support, dim1, dim2, dim3):
    """The decoder's feature groups from the pack fields (fp32)."""
    pos, d = pos.float(), sdf_dir.float()
    z_log = torch.log(torch.clamp(pos[:, 2:3] * 550.0, min=0.1))
    xy = torch.cat([pos[:, 0:1], pos[:, 1:2], d[:, 0:1], d[:, 1:2]], dim=-1)
    zf = torch.cat([z_log, d[:, 2:3], sdf.float()[:, None]], dim=-1)
    return (xy, zf, support.float(), dim1, dim2, dim3)


def _ibo_view(s: dict) -> dict:
    """Pack sample / case -> the IBO contract keys."""
    return {
        "latent_slots": s["pn_points_feat"], "latent_mask": s["pn_mask"],
        "latent_stats": torch.cat([s["leaf_stats"].float(), s["leaf_pos_norm"].float()], dim=-1),
        "latent_pos_geo": s["leaf_pos_norm"].float(), "cond": s["global_morph_norm"],
        "query_pos_geo": s["query_pos_norm"].float(),
        "query_feats": _query_feats(s["query_pos_norm"], s["query_sdf_norm"], s["query_sdf_dir"],
                                    s["query_support_norm"], s["region_dim1"],
                                    s["region_dim2"], s["region_dim3"]),
        "query_knn": s["idw_indices"], "target_vol": s["target_vol"],
        "target_surf": s["target_surf"], "is_volume": s["is_volume"],
    }


class _Mapped(TorchDataset):
    def __init__(self, base, fn):
        self.base, self.fn = base, fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        return self.fn(self.base[item])


def _points_x(case: dict) -> torch.Tensor:
    return torch.cat([case["query_pos_norm"].float(), case["query_sdf_norm"].float().unsqueeze(-1),
                      case["query_sdf_dir"].float()], dim=-1)


class UrbanWind(Dataset):
    spec = SPEC

    def __init__(self):
        self._stats = None

    # ---- data on disk -----------------------------------------------------
    def cache_dir(self, cfg: Config) -> Path:
        return cache_dir(cfg)

    def norm_stats_source(self, cfg: Config) -> Path:
        return Path(cfg.urbanwind.norm_stats) if cfg.urbanwind.norm_stats \
            else repo_root() / "datasets" / "urbanwind" / "norm_stats.json"

    def preprocess(self, cfg: Config, from_cache: bool = False) -> None:
        from .preprocess import run_estate_preprocess, run_releaf
        if from_cache:
            run_releaf(cfg, cache_dir(cfg, "importance"), workers=cfg.urbanwind.n_workers)
        else:
            run_estate_preprocess(cfg, self.norm_stats_source(cfg))

    def splits(self, cfg: Config) -> dict[str, list[str]]:
        """Estate splits expanded to case names `<estate>_<dir>`, using the
        cache's direction lists (every direction when no cache exists)."""
        estates = super().splits(cfg)
        index_path = cache_dir(cfg) / "_index.json"
        dirs = {e: v["dirs"] for e, v in read_json(index_path).items()} if index_path.exists() else {}
        return {k: [f"{e}_{d}" for e in v for d in dirs.get(e, ("N", "E", "S", "W"))]
                for k, v in estates.items()}

    def derive(self, cfg: Config, model: str, names: list[str]) -> None:
        if model != "spider":
            raise SystemExit(f"urbanwind: no derived preprocessing for {model}")
        build_spider_derived(cfg, sorted({estate_of(n) for n in names}),
                             workers=cfg.spider.derive_workers or cfg.urbanwind.n_workers)

    def ensure_derived(self, cfg: Config, model: str, names: list[str]) -> None:
        if model != "spider":
            return
        out = spider_derived_dir(cfg)
        missing = [e for e in sorted({estate_of(n) for n in names}) if not (out / f"{e}.pt").exists()]
        if missing:
            raise SystemExit(f"spider derived cache incomplete under {out}: {len(missing)} estates "
                             f"missing (e.g. {missing[:3]}). Build it once:\n  python -m ibo derive "
                             f"--dataset urbanwind --model spider")

    # ---- training ---------------------------------------------------------
    def shard(self, names: list[str], rank: int, world: int) -> list[str]:
        """Whole estates per rank (one pack holds all directions): estates
        go largest-first to the least loaded rank, then every rank is cut
        to the same case count, because the per-rank step count must agree
        across ranks (an estate may hold fewer than four clean directions)."""
        by_estate: dict[str, list[str]] = {}
        for n in names:
            by_estate.setdefault(estate_of(n), []).append(n)
        loads, mine = [0] * world, []
        for e, cases in sorted(by_estate.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            r = min(range(world), key=lambda i: (loads[i], i))
            loads[r] += len(cases)
            if r == rank:
                mine.extend(cases)
        return mine[:min(loads)]

    def build_train_dataset(self, cfg: Config, names: list[str], view: str, seed_salt: str):
        estates, dirs = [], {}
        for n in names:
            e, d = estate_of(n), n.rsplit("_", 1)[1]
            if e not in dirs:
                estates.append(e)
                dirs[e] = []
            dirs[e].append(d)
        if view == "ibo":
            base = RamEstateDataset(cfg, estates, cfg.training.n_query, dirs_by_estate=dirs,
                                    seed_salt=seed_salt)
            return _Mapped(base, _ibo_view), collate_tree, "ram"
        if view == "abupt":
            return RamEstateDatasetABUPT(cfg, estates, dirs_by_estate=dirs, seed_salt=seed_salt), \
                collate_tree, "ram"
        if view == "points":
            mode = "spider" if cfg.run.model == "spider" else "transolver"
            return PointcloudDataset(cfg, estates, mode, dirs_by_estate=dirs), collate_single, "ram"
        if view == "erwin":
            return PointcloudDataset(cfg, estates, "erwin", dirs_by_estate=dirs), collate_single, "ram"
        raise ValueError(f"urbanwind: unknown view {view!r}")

    # ---- evaluation -------------------------------------------------------
    def stream_eval_cases(self, cfg: Config, names: list[str], view: str):
        with_idw = view == "ibo"
        if view == "points" and cfg.run.model == "spider":
            yield from self._stream_spider(cfg, names)
            return
        for name, case in stream_full_estate_cases(cfg, names, with_idw=with_idw):
            yield name, self._case_view(case, view, name)

    def _stream_spider(self, cfg, names):
        grouped = {}
        for n in names:
            grouped.setdefault(estate_of(n), []).append(n.rsplit("_", 1)[1])
        for est, dirs in grouped.items():
            pack = load_estate(cfg, est, resident_mode=False, build_idw_tree=False)
            attach_spider_derived(cfg, pack)
            nearest = pack["_spider_nearest_surf"].long()
            for d in dirs:
                case = materialize(pack, d, keep_z_phys=True, with_idw=False)
                out = self._case_view(case, "points", f"{est}_{d}")
                surf_region = surf_region_classes(pack, d)
                out["spider"] = {"surf_region": surf_region, "vol_cluster": surf_region[nearest],
                                 "wall": out["x"][:out["n_vol"], 3].abs()}
                yield f"{est}_{d}", out
            del pack

    def _case_view(self, case: dict, view: str, name: str) -> dict:
        n_vol, n_surf = int(case["N_vol"]), int(case["N_surf"])
        out = {"name": name, "n_vol": n_vol, "n_surf": n_surf,
               "target_vol": case["target_vol"], "target_surf": case["target_surf"],
               "meta": {"z_phys": case["query_z_phys"]}}
        if view == "ibo":
            s = dict(case)
            s["is_volume"] = torch.cat([torch.ones(n_vol, dtype=torch.bool),
                                        torch.zeros(n_surf, dtype=torch.bool)])
            v = _ibo_view(s)
            v.pop("target_vol"), v.pop("target_surf")
            out.update(v)
        elif view == "points":
            out["x"] = _points_x(case)
        elif view in ("erwin", "abupt"):
            out["pos"] = case["query_pos_norm"].float()
            out["cond"] = None
        else:
            raise ValueError(f"urbanwind: unknown view {view!r}")
        return out

    def _norm_stats(self, cfg: Config) -> dict:
        if self._stats is None:
            self._stats = load_norm_stats(norm_stats_path(cfg))
        return self._stats

    def case_metrics(self, cfg: Config, case: dict, pred_vol_z, pred_surf_z) -> dict:
        stats = self._norm_stats(cfg)
        n_vol = case["n_vol"]
        true_vol_z = case["target_vol"].float().numpy()
        true_surf_z = case["target_surf"].float().numpy()
        z_phys = case["meta"]["z_phys"].numpy()
        z_vol, z_surf = z_phys[:n_vol], z_phys[n_vol:]
        m = case_metrics(pred_vol_z, true_vol_z, pred_surf_z, true_surf_z,
                         inverse_transform_vol(pred_vol_z, z_vol, stats),
                         inverse_transform_vol(true_vol_z, z_vol, stats),
                         inverse_transform_surf(pred_surf_z, z_surf, stats),
                         inverse_transform_surf(true_surf_z, z_surf, stats))
        rl2 = m["relative_l2_physical"]
        flat = {**rl2["volume_channels"], **rl2["surface_channels"]}
        flat["zscore_mse"] = m["zscore_mse"]["total"]
        return flat

    def aggregate(self, cfg: Config, rows: list[dict]) -> dict:
        """Relative L2 per channel, mean over the test cases."""
        return {k: float(np.mean([r[k] for r in rows])) for k in SPEC.metric_names}


DATASET = UrbanWind()

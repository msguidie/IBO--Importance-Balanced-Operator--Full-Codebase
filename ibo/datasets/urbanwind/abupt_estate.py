"""AB-UPT view of the estate packs: one sample = one estate x one wind
direction, with vol_anchors volume rows and surf_anchors surface rows drawn
independently without replacement, seeded by
("<estate>_<dir>" + seed_salt + "__abupt_vol" / "__abupt_surf", epoch).
Rows are gathered in the canonical frame and rotated into the direction's
frame; positions stay in the metres / 550 frame (the affine to AB-UPT's
non-negative coordinates is part of the model stem)."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ...config import Config
from ...utils.seed import per_case_epoch_seed
from .estate import load_estate, rot_points

# pack fields no AB-UPT sample reads: the IBO latent layout and geometry
_DROP_KEYS = ("leaf_pos_norm", "leaf_stats", "pn_points_feat", "pn_mask", "_idw_k",
              "query_sdf_norm", "query_sdf_dir_ang", "query_support_norm",
              "region_dim2", "region_dim3")


def check_abupt_coord_bounds(pos_norm: torch.Tensor, label: str) -> None:
    """The fixed affine (pos + [1, 1, 0]) * 500 needs xy in [-1, 1] and z >= 0
    after the division by 550 m; checked outside the compiled graph."""
    xy, z = pos_norm[..., :2], pos_norm[..., 2]
    if bool((xy.amin() < -1.0) or (xy.amax() > 1.0) or (z.amin() < 0.0)):
        raise RuntimeError(f"{label}: positions outside the UrbanWind domain (xy in "
                           f"[-1, 1], z >= 0 after /550); the AB-UPT affine would "
                           f"produce negative coordinates")


def load_resident_pack_abupt(cfg: Config, estate: str, keep_dirs: list[str] | None) -> dict:
    pack = load_estate(cfg, estate, resident_mode=True, build_idw_tree=False)
    all_dirs = list(pack["dirs"])
    kept = all_dirs if keep_dirs is None else [d for d in all_dirs if d in keep_dirs]
    for d in all_dirs:
        pack.pop(f"global_morph_norm_{d}", None)
        pack.pop(f"surf_dim2_{d}", None)
        pack.pop(f"surf_dim3_{d}", None)
        if d not in kept:
            pack.pop(f"target_vol_{d}", None)
            pack.pop(f"target_surf_{d}", None)
    pack["dirs"] = kept
    for key in _DROP_KEYS:
        pack.pop(key, None)
    check_abupt_coord_bounds(pack["query_pos_norm"], estate)
    for d in kept:
        pack[f"target_vol_{d}"] = pack[f"target_vol_{d}"].half()
        pack[f"target_surf_{d}"] = pack[f"target_surf_{d}"].half()
    return pack


def sample_estate_anchors(pack: dict, direction: str, vol_anchors: int,
                          surf_anchors: int, epoch: int, seed_salt: str = "") -> dict:
    k = int(pack["rot_code"][direction])
    n_vol, n_surf = int(pack["N_vol"]), int(pack["N_surf"])
    if n_vol < vol_anchors or n_surf < surf_anchors:
        raise RuntimeError(f"{pack['estate']}_{direction}: N_vol={n_vol} N_surf={n_surf} "
                           f"< anchors ({vol_anchors}, {surf_anchors})")
    seed_base = f"{pack['estate']}_{direction}" + seed_salt
    g_vol = torch.Generator()
    g_vol.manual_seed(per_case_epoch_seed(seed_base + "__abupt_vol", epoch))
    vol_local = torch.randperm(n_vol, generator=g_vol)[:vol_anchors].sort().values
    g_surf = torch.Generator()
    g_surf.manual_seed(per_case_epoch_seed(seed_base + "__abupt_surf", epoch))
    surf_local = torch.randperm(n_surf, generator=g_surf)[:surf_anchors].sort().values
    return {
        "surface_position": rot_points(pack["query_pos_norm"][surf_local + n_vol], k),
        "volume_position": rot_points(pack["query_pos_norm"][vol_local], k),
        "target_surf": pack[f"target_surf_{direction}"][surf_local],
        "target_vol": pack[f"target_vol_{direction}"][vol_local],
    }


class RamEstateDatasetABUPT(Dataset):
    """RAM-resident AB-UPT training dataset: a flat (estate, direction) index."""

    def __init__(self, cfg: Config, estate_names: list[str],
                 dirs_by_estate: dict[str, list[str]] | None = None, seed_salt: str = ""):
        self.seed_salt = str(seed_salt)
        self.vol_anchors = int(cfg.abupt.vol_anchors)
        self.surf_anchors = int(cfg.abupt.surf_anchors)
        print(f"[data] loading {len(estate_names)} estate packs into RAM (AB-UPT view)...",
              flush=True)
        self.packs = [load_resident_pack_abupt(
            cfg, e, keep_dirs=None if dirs_by_estate is None else dirs_by_estate.get(e))
            for e in estate_names]
        self.cases = [(pi, d) for pi, p in enumerate(self.packs) for d in p["dirs"]]
        self.case_names = [f"{self.packs[pi]['estate']}_{d}" for pi, d in self.cases]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, item) -> dict:
        flat_idx, epoch = item
        pi, d = self.cases[flat_idx]
        return sample_estate_anchors(self.packs[pi], d, self.vol_anchors, self.surf_anchors,
                                     epoch, seed_salt=self.seed_salt)

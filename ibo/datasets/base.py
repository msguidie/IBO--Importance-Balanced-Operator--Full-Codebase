"""The dataset contract: what every benchmark package provides to the
model layer and to the training / evaluation drivers.

A package exports `DATASET`, an instance of a Dataset subclass whose `spec`
describes the tensors it produces. The model layer builds networks from the
spec; the drivers never look inside a dataset beyond this interface.

Training sample (view "ibo"), batched by the loader:
  latent_slots   [B, L, S, F]   per-token member slots (slot encoder input)
  latent_mask    [B, L, S]      valid slots (all True when every slot is filled)
  latent_stats   [B, L, T]      per-token descriptor appended to the pooled slots
                                (position, wall distance, ...; the paper's region descriptor)
  latent_pos_geo [B, L, d]      token positions in the frame used for rotary embeddings
                                and neighbour distances (UrbanWind: metres / 550)
  cond           [B, C]         case vector (C = spec.cond_dim, may be 0)
  query_pos_geo  [B, N, d]      query positions in the same frame
  query_feats    tuple, one tensor per spec.query_groups item, in order:
                                [B, N, k_i] floats for plain / fourier groups,
                                [B, N] integers for embed groups
  query_knn      [B, N, k]      indices of the k nearest tokens
  target_vol     [B, N, Cv]     z-scored volume targets (zero on surface rows)
  target_surf    [B, N, Cs]     z-scored surface targets (zero on volume rows)
  is_volume      [B, N]         bool
Other views (points, abupt, erwin, gaot) carry the keys their recipe reads.

Inference case (stream_eval_cases): the same keys without the batch axis,
over every evaluation point of the case, plus `name`, `n_vol`, `n_surf`
and `meta` (whatever case_metrics needs: physical heights, reference
coefficients, raw paths).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config, repo_root


@dataclass
class FeatureGroup:
    """One query feature group of the IBO decoder's self-embedding."""
    key: str
    dim: int
    encode: str = "plain"      # plain | fourier | embed
    n_freqs: int = 0           # fourier bands (encode == fourier)
    n_classes: int = 0         # embed
    embed_dim: int = 0         # embed


@dataclass
class DatasetSpec:
    name: str
    dim: int
    vol_channels: list[str]
    surf_channels: list[str]
    cond_dim: int = 0
    cond_to_baselines: bool = True    # AB-UPT / Erwin also read the condition (2-D benchmarks)
    slot_feat_dim: int = 0            # F: features per member slot
    stats_dim: int = 0                # T: per-token descriptor width
    query_groups: list[FeatureGroup] = field(default_factory=list)
    # IBO pair embedding: Fourier bands per axis of the normalised offset,
    # plus the unit direction and log mean distance when set
    pair_fourier_freqs: tuple[int, ...] = ()
    pair_unit_dir: bool = True
    pair_log_dbar: bool = True
    native_point_dim: int = 0         # Transolver / SpiderSolver input width
    loss_vol_weights: tuple[float, ...] | None = None
    metric_names: list[str] = field(default_factory=list)


def check_ibo_layout(cfg: Config, n_tokens: int, cache_knn: int, n_slots: int | None = None) -> None:
    """The ibo section must describe the cache it reads (the preprocess
    section names the cache; the model section sizes the network)."""
    m = cfg.ibo
    bad = []
    if int(m.n_tokens) != int(n_tokens):
        bad.append(f"ibo.n_tokens={m.n_tokens}, the cache has {n_tokens} tokens")
    if int(m.cache_knn) != int(cache_knn):
        bad.append(f"ibo.cache_knn={m.cache_knn}, the cache stores {cache_knn} neighbours")
    if n_slots is not None and int(m.n_slots) != int(n_slots):
        bad.append(f"ibo.n_slots={m.n_slots}, the cache has {n_slots} slots per token")
    if bad:
        raise SystemExit("config / cache mismatch: " + "; ".join(bad))


class Dataset:
    spec: DatasetSpec          # the supported models and layouts are registry.MATRIX / LAYOUTS

    # ---- data on disk -----------------------------------------------------
    def splits_path(self) -> Path:
        return repo_root() / "datasets" / self.spec.name / "splits.json"

    def splits(self, cfg: Config) -> dict[str, list[str]]:
        """{"train": [...], "val": [...], "test": [...]} of case names."""
        with open(self.splits_path(), encoding="utf-8") as f:
            d = json.load(f)
        return {k: list(d[k]) for k in ("train", "val", "test")}

    def cache_dir(self, cfg: Config) -> Path:
        raise NotImplementedError

    def preprocess(self, cfg: Config, from_cache: bool = False) -> None:
        raise NotImplementedError

    def calibration(self, cfg: Config) -> dict:
        """Dotted config keys adopted from the cache (per-subset RoPE domain
        and Fourier scale on the 2-D benchmarks); {} when not applicable."""
        return {}

    def derive(self, cfg: Config, model: str, names: list[str]) -> None:
        raise SystemExit(f"{self.spec.name}: no derived preprocessing for {model}")

    def ensure_derived(self, cfg: Config, model: str, names: list[str]) -> None:
        """Raise with build instructions when a required derived cache is missing."""

    # ---- training ---------------------------------------------------------
    def shard(self, names: list[str], rank: int, world: int) -> list[str]:
        """This rank's training cases: round robin, equal length per rank."""
        per = len(names) // world
        return names[rank::world][:per]

    def build_train_dataset(self, cfg: Config, names: list[str], view: str,
                            seed_salt: str):
        """-> (torch Dataset indexed by (item, epoch), collate_fn, residency)."""
        raise NotImplementedError

    # ---- evaluation -------------------------------------------------------
    def stream_eval_cases(self, cfg: Config, names: list[str], view: str):
        raise NotImplementedError

    def case_metrics(self, cfg: Config, case: dict, pred_vol_z, pred_surf_z) -> dict:
        raise NotImplementedError

    def aggregate(self, cfg: Config, rows: list[dict]) -> dict:
        raise NotImplementedError

"""Per-dataset calibration written at the end of a 2-D preprocessing and
adopted at launch (Dataset.calibration): the rotary-embedding domain,
scales and bases from the exact extent of every token position, and the
decoder's Fourier SDF scale from the median |scaled distance| of a fixed
sample of cases."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from ...utils.seed import per_case_epoch_seed

_QUANTILES = (0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0)
_QUERIES_PER_CASE = 50_000
CALIBRATION_SEED = 42
ROPE_UNITS_PER_CHORD = 32.0 / 6.0     # 32 rotary units over the 6-chord AirfRANS envelope


def _q(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {f"p{p:g}": float(np.percentile(values, p)) for p in _QUANTILES}


def suggest_rope_base(scale: float, dim: int) -> float:
    """Per-axis base such that the coarsest frequency sweeps about one
    cycle over the axis span: base^(-(dim-2)/dim) * scale ~= 2 pi."""
    ratio = max(float(scale) / (2.0 * math.pi), 1.0 + 1e-6)
    return float(ratio ** (dim / (dim - 2)))


def _sample_cases(manifest: dict, total: int, seed: int) -> list[str]:
    """Half train, a quarter val, a quarter test, deterministic."""
    rng = np.random.default_rng(seed)
    quota = {"train": total - total // 2, "val": total // 4, "test": total // 2 - total // 4}
    picked = []
    for split, k in quota.items():
        names = list(manifest.get(split, []))
        k = min(k, len(names))
        if k:
            picked += [names[i] for i in sorted(rng.choice(len(names), size=k, replace=False))]
    return picked


def run_calibration(static_dir: Path, latent_dir: Path, out_path: Path, manifest: dict,
                    n_sample: int, rope_dims, idw_k: int, sdf_abs, label: str) -> dict:
    """manifest: {"train": [...], "val": [...], "test": [...]}; sdf_abs(static,
    sel) -> the |distance| values whose median sets the Fourier scale."""
    static_dir, latent_dir, out_path = Path(static_dir), Path(latent_dir), Path(out_path)
    rope_dims = tuple(int(d) for d in rope_dims)
    names_all = list(manifest["train"]) + list(manifest["val"]) + list(manifest["test"])

    pos_lo, pos_hi = np.full(2, np.inf), np.full(2, -np.inf)
    for name in names_all:
        lpg = torch.load(str(latent_dir / f"{name}.pt"), map_location="cpu",
                         weights_only=True)["leaf_pos_geo"].numpy()
        pos_lo = np.minimum(pos_lo, lpg.min(axis=0))
        pos_hi = np.maximum(pos_hi, lpg.max(axis=0))

    nn_dists, ratios, mags = [], [], []
    names = _sample_cases(manifest, n_sample, CALIBRATION_SEED)
    for name in names:
        latent = torch.load(str(latent_dir / f"{name}.pt"), map_location="cpu", weights_only=True)
        static = torch.load(str(static_dir / f"{name}.pt"), map_location="cpu", weights_only=True)
        lpg = latent["leaf_pos_geo"].numpy().astype(np.float64)
        d, _ = cKDTree(lpg).query(lpg, k=2)
        nn_dists.append(d[:, 1].astype(np.float32))
        qpg = static["query_pos_geo"].numpy()
        idw = static["idw_indices"].to(torch.int64).numpy()
        rng = np.random.default_rng(per_case_epoch_seed(name, 2))
        sel = rng.choice(len(qpg), size=min(_QUERIES_PER_CASE, len(qpg)), replace=False)
        d_ij = np.linalg.norm(qpg[sel][:, None, :].astype(np.float64) - lpg[idw[sel]], axis=-1)
        dbar = np.maximum(d_ij.mean(axis=1, keepdims=True), 1e-12)
        ratios.append((d_ij / dbar).astype(np.float32).reshape(-1))
        mags.append(np.asarray(sdf_abs(static, sel), dtype=np.float32).reshape(-1))
    nn_dists, ratios, mags = np.concatenate(nn_dists), np.concatenate(ratios), np.concatenate(mags)

    pad = 0.1 * (pos_hi - pos_lo)             # ~20% total growth around the extent
    dom_lo, dom_hi = pos_lo - pad, pos_hi + pad
    scales = np.round(ROPE_UNITS_PER_CHORD * (dom_hi - dom_lo))
    bases = [suggest_rope_base(float(scales[a]), rope_dims[a]) for a in range(2)]
    # the coarsest Fourier band is pi: scale x median in [1, 2] keeps the
    # median input within about one cycle
    med = float(np.percentile(mags, 50.0))
    sdf_scale = (1.5 / med) if med > 1e-8 else 10.0

    report = {
        "label": label, "n_cases_sampled": len(names), "cases": names,
        "n_cases_scanned_for_extent": len(names_all),
        "rope": {"leaf_nn_dist_geo": {"min": float(nn_dists.min()), **_q(nn_dists)},
                 "leaf_pos_geo_min": pos_lo.tolist(), "leaf_pos_geo_max": pos_hi.tolist(),
                 "suggested_rope_domain_lo": dom_lo.tolist(), "suggested_rope_domain_hi": dom_hi.tolist(),
                 "suggested_rope_scales": scales.tolist(),
                 "suggested_rope_base": [round(b, 3) for b in bases], "rope_dims": list(rope_dims)},
        "decoder_fourier": {"idw_k": int(idw_k), "log_idw_k": float(np.log(idw_k)),
                            "d_over_dbar": _q(ratios), "log_d_over_dbar": _q(np.log(np.maximum(ratios, 1e-12))),
                            "sdf_abs": _q(mags), "suggested_fourier_sdf_scale": round(float(sdf_scale), 3)},
    }
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    tmp.replace(out_path)
    print(f"[calibrate] {out_path}: rope domain {np.round(dom_lo, 3).tolist()} .. "
          f"{np.round(dom_hi, 3).tolist()}, scales {scales.tolist()}, base "
          f"{[round(b, 1) for b in bases]}, fourier_sdf_scale {sdf_scale:.3f}", flush=True)
    return report

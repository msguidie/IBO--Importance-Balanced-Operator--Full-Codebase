"""The support matrix: which model runs on which dataset, as in the paper."""
from __future__ import annotations

import importlib

from .config import DATASETS, MODELS

MATRIX = {
    "airfrans": ("ibo", "abupt"),
    "tandemfoilset": ("ibo", "abupt", "transolver", "spider", "erwin", "gaot"),
    "drivaerml": ("ibo",),
    "urbanwind": ("ibo", "abupt", "transolver", "spider", "erwin"),
}
LAYOUTS = {
    "airfrans": ("importance", "grid"),       # importance = equal-count tree (unit weights)
    "tandemfoilset": ("importance", "grid"),
    "drivaerml": ("importance", "grid"),
    "urbanwind": ("importance", "equal", "grid"),
}


def supported(dataset: str, model: str) -> bool:
    return model in MATRIX.get(dataset, ())


def get_recipe(name: str):
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; models: {', '.join(MODELS)}")
    return importlib.import_module(f"ibo.recipes.{name}").RECIPE


def matrix_text() -> str:
    head = "model".ljust(12) + "".join(d.ljust(15) for d in DATASETS)
    lines = [head, "-" * len(head)]
    for m in MODELS:
        lines.append(m.ljust(12) + "".join(
            ("yes" if supported(d, m) else "-").ljust(15) for d in DATASETS))
    lines.append("")
    lines.append("layouts: " + "; ".join(f"{d}: {', '.join(v)}" for d, v in LAYOUTS.items()))
    return "\n".join(lines)


def check_pair(dataset: str, model: str, layout: str) -> None:
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}; datasets: {', '.join(DATASETS)}")
    if not supported(dataset, model):
        raise SystemExit(f"{model} is not run on {dataset} in the paper\n\n{matrix_text()}")
    if layout not in LAYOUTS[dataset]:
        raise SystemExit(f"layout {layout!r} is not available on {dataset}: "
                         f"{', '.join(LAYOUTS[dataset])}")
    if layout != "importance" and model != "ibo":
        raise SystemExit("--layout applies to --model ibo only")

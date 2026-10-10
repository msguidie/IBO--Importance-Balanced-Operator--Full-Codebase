"""Dataset layer: one package per benchmark, each exporting DATASET.

    from ibo.datasets import get_dataset
    ds = get_dataset("urbanwind")
"""
from __future__ import annotations

import importlib


def get_dataset(name: str):
    return importlib.import_module(f"ibo.datasets.{name}").DATASET

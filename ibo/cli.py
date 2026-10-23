"""Command line: preprocessing, training and evaluation for every dataset
and model in the paper.

  python -m ibo list
  python -m ibo preprocess --dataset urbanwind --data-root data/urbanwind
  torchrun --standalone --nproc_per_node 4 -m ibo train --dataset urbanwind --model ibo
  torchrun --standalone --nproc_per_node 4 -m ibo eval --run-dir runs/urbanwind_ibo_s42

Configuration: configs/<dataset>/<model>.yaml holds the paper setting of
each pair; --set section.key=value overrides any key. The resolved config
is written to the run directory and `eval` reads it back from there.
"""
from ibo.utils import env  # noqa: F401  (must precede torch import)

import argparse
import sys
from pathlib import Path

from .config import (TANDEM_SUBSETS, load_config, load_config_file,
                     resolve_paths, set_dotted, set_lr)
from .datasets import get_dataset
from .registry import check_pair, get_recipe, matrix_text


def _subsets(args) -> list:
    if args.dataset != "tandemfoilset":
        return [None]
    if not args.subset:
        raise SystemExit(f"tandemfoilset needs --subset (one of {', '.join(TANDEM_SUBSETS)}, or all)")
    return list(TANDEM_SUBSETS) if args.subset == "all" else [args.subset]


def _load(args, subset):
    cfg = load_config(args.dataset, args.model, subset=subset, layout=args.layout,
                      config_path=args.config, sets=args.set)
    check_pair(cfg.run.dataset, cfg.run.model, cfg.run.layout)
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.cache_root:
        cfg.data.cache_root = args.cache_root
    return cfg


def _apply_calibration(cfg, dataset, pinned: set) -> None:
    """Per-subset calibrated values from the cache win over the yaml unless
    the user pinned the key with --set."""
    for key, value in dataset.calibration(cfg).items():
        if key in pinned:
            continue
        set_dotted(cfg, key, value)
        cfg.run.calibration[key] = value


def cmd_list(args) -> int:
    print(matrix_text())
    return 0


def cmd_preprocess(args) -> int:
    for subset in _subsets(args):
        cfg = _load(args, subset)
        resolve_paths(cfg)
        if args.workers is not None:
            cfg.preprocess.n_workers = int(args.workers)
        get_dataset(cfg.run.dataset).preprocess(cfg, from_cache=args.from_cache)
    return 0


def cmd_derive(args) -> int:
    for subset in _subsets(args):
        cfg = _load(args, subset)
        resolve_paths(cfg)
        dataset = get_dataset(cfg.run.dataset)
        names = dataset.splits(cfg)
        dataset.derive(cfg, cfg.run.model, names["train"] + names["val"] + names["test"])
    return 0


def cmd_train(args) -> int:
    from .training.loop import train

    subsets = _subsets(args)
    for subset in subsets:
        cfg = _load(args, subset)
        if args.seed is not None:
            cfg.run.seed = int(args.seed)
        if args.run_dir:
            cfg.data.run_dir = args.run_dir
        resolve_paths(cfg, subset_in_run_dir=len(subsets) == 1)
        if len(subsets) > 1:
            cfg.data.run_dir = str(Path(cfg.data.run_dir) / subset)
        if args.epochs is not None:
            cfg.training.epochs = int(args.epochs)
        if args.batch_size is not None:
            cfg.training.cases_per_gpu = int(args.batch_size)
        if args.n_query is not None:
            cfg.training.n_query = int(args.n_query)
        if args.lr is not None:
            set_lr(cfg, args.lr)
        dataset = get_dataset(cfg.run.dataset)
        _apply_calibration(cfg, dataset, {s.split("=")[0].strip() for s in args.set})
        cfg.validate()
        recipe = get_recipe(cfg.run.model)
        recipe.apply(cfg, dataset.spec)
        train(cfg, recipe, dataset, auto_resume=args.auto_resume, max_hours=args.max_hours)
    return 0


def cmd_eval(args) -> int:
    from .evaluation.run import evaluate

    run_dir = Path(args.run_dir)
    used = sorted(run_dir.glob("*/config_used.yaml")) if not (run_dir / "config_used.yaml").exists() \
        else [run_dir / "config_used.yaml"]
    if not used:
        raise SystemExit(f"{run_dir}: no config_used.yaml (a finished run, or a "
                         f"directory of per-subset runs)")
    for path in used:
        cfg = load_config_file(path)
        for item in args.set:
            key, _, value = item.partition("=")
            import yaml
            set_dotted(cfg, key.strip(), yaml.safe_load(value))
        if args.data_root:
            cfg.data.data_root = args.data_root
        if args.cache_root:
            cfg.data.cache_root = args.cache_root
        cfg.data.run_dir = str(path.parent)
        cfg.evaluation.split = args.split
        cfg.evaluation.save_predictions = bool(args.save_predictions)
        cfg.evaluation.latency = bool(args.latency)
        cfg.evaluation.flops = bool(args.flops)
        evaluate(cfg, path.parent)
    return 0


def _common(p, with_model: bool):
    p.add_argument("--dataset", required=True)
    if with_model:
        p.add_argument("--model", default="ibo")
    p.add_argument("--subset", default=None, help="tandemfoilset: a subset name or all")
    p.add_argument("--layout", default=None, choices=["importance", "equal", "grid"],
                   help="latent layout of the IBO cache (default importance)")
    p.add_argument("--config", default=None, help="yaml instead of configs/<dataset>/<model>.yaml")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-root", default=None)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m ibo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="the dataset x model support matrix").set_defaults(fn=cmd_list)

    p = sub.add_parser("preprocess", help="raw data -> cache")
    _common(p, with_model=False)
    p.set_defaults(model="ibo")
    p.add_argument("--from-cache", action="store_true",
                   help="rebuild only the latent layout of an existing cache (another --layout)")
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(fn=cmd_preprocess)

    d = sub.add_parser("derive", help="model-specific derived caches (spider, gaot)")
    _common(d, with_model=True)
    d.set_defaults(fn=cmd_derive)

    t = sub.add_parser("train", help="train one model (torchrun for several GPUs)")
    _common(t, with_model=True)
    t.add_argument("--run-dir", default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--lr", type=float, default=None, help="peak learning rate of the selected model")
    t.add_argument("--batch-size", type=int, default=None, help="cases per GPU")
    t.add_argument("--n-query", type=int, default=None)
    t.add_argument("--auto-resume", action="store_true")
    t.add_argument("--max-hours", type=float, default=None,
                   help="stop cleanly at the first epoch boundary past this walltime")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="evaluate a finished run from its config_used.yaml")
    e.add_argument("--run-dir", required=True, help="run directory (or a directory of per-subset runs)")
    e.add_argument("--split", default="test", choices=["test", "val"])
    e.add_argument("--save-predictions", action="store_true")
    e.add_argument("--latency", action="store_true")
    e.add_argument("--flops", action="store_true")
    e.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    e.add_argument("--data-root", default=None)
    e.add_argument("--cache-root", default=None)
    e.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

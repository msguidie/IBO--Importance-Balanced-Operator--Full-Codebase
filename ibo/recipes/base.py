"""Recipe: the one interface through which the trainer and the evaluation
driver run every model. A recipe binds a model builder, its data view, the
training forward + loss, the optimizer and scheduler, the full-case
inference producer (every point exactly once) and the closed-form FLOPs of
what the hook meter cannot see. The trainer has no per-model branches.
"""
from __future__ import annotations

import torch

from ..training.data import move_to_device


class Recipe:
    name: str = ""
    view: str = "ibo"                      # the dataset view this model reads
    disable_ddp_optimizer: bool = False    # dynamo's DDP graph splitting off
    infer_draw: int = 0                    # stochastic producers: current draw

    def apply(self, cfg, spec) -> None:
        """Recipe-specific config resolution before launch (optional)."""

    # ---- data -------------------------------------------------------------
    def build_train_dataset(self, cfg, dataset, names, seed_salt: str):
        return dataset.build_train_dataset(cfg, names, self.view, seed_salt)

    def batch_to_device(self, batch, device):
        return move_to_device(batch, device)

    # ---- model ------------------------------------------------------------
    def build_model(self, cfg, spec) -> torch.nn.Module:
        raise NotImplementedError

    def compile_plan(self, cfg) -> tuple[str, bool]:
        """(mode, fullgraph); mode "" disables torch.compile for training."""
        return (cfg.training.compile_mode if cfg.training.compile else "", True)

    def forward_loss(self, model, batch, cfg, spec):
        raise NotImplementedError

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        """-> (optimizer, scheduler, granularity in {"step", "epoch"})."""
        raise NotImplementedError

    # ---- inference --------------------------------------------------------
    def stream_cases(self, cfg, dataset, names):
        return dataset.stream_eval_cases(cfg, names, self.view)

    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> str:
        """Optionally compile the inference entry points; returns a label."""
        return "eager"

    def infer_draws(self, cfg) -> int:
        return 1

    def infer_case(self, model, case: dict, cfg, device, use_cuda: bool):
        """-> (pred_vol_z [N_vol, Cv], pred_surf_z [N_surf, Cs]) fp32 numpy,
        original row order, every point exactly once."""
        raise NotImplementedError

    def manual_infer_flops(self, cfg, spec, n_vol: int, n_surf: int) -> dict[str, float]:
        return {}

    # ---- derived preprocessing --------------------------------------------
    def ensure_derived(self, cfg, dataset, names: list[str]) -> None:
        """Raise with build instructions if a required derived cache is missing."""


def cpu_defloat_bf16(batch):
    """CPU runs: bf16 tensors -> fp32."""
    if isinstance(batch, torch.Tensor):
        return batch.float() if batch.is_floating_point() \
            and batch.dtype == torch.bfloat16 else batch
    if isinstance(batch, dict):
        return {k: cpu_defloat_bf16(v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(cpu_defloat_bf16(v) for v in batch)
    return batch

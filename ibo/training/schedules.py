"""Learning-rate schedules, each stepped at the recipe's own granularity
("step" = once per optimizer update, "epoch" = once per epoch).

build_warmup_cosine_fraction   IBO: linear warmup (a step count or a fraction
                               of the run) then cosine to a fraction of the
                               peak lr.
build_warmup_cosine            linear warmup then cosine to an absolute floor
                               (AB-UPT, Erwin; warmup 0 = plain cosine).
build_onecycle                 torch OneCycleLR (Transolver, SpiderSolver).
GaotMixScheduler               GAOT's three-phase schedule, per epoch.
"""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


def build_warmup_cosine_fraction(optimizer, total_steps: int,
                                 warmup_fraction: float,
                                 min_lr_fraction: float,
                                 warmup_steps: int | None = None):
    """Multiplier on the peak lr: pure linear ramp (first step = 1/warmup,
    no floor during the ramp), then cosine down to min_lr_fraction."""
    warmup = max(int(warmup_steps if warmup_steps is not None
                     else warmup_fraction * total_steps), 1)
    warmup = min(warmup, max(total_steps - 1, 1))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_fraction + (1 - min_lr_fraction) * 0.5 \
            * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_warmup_cosine(optimizer, total_units: int, peak_lr: float,
                        end_lr: float, warmup_fraction: float = 0.0):
    """Linear warmup over warmup_fraction of total_units, then cosine from
    peak_lr to the absolute end_lr at the final unit."""
    total_units = int(total_units)
    warmup = int(round(warmup_fraction * total_units))
    warmup = min(warmup, max(total_units - 1, 0))
    floor = end_lr / peak_lr

    def lr_lambda(u: int) -> float:
        if warmup > 0 and u < warmup:
            return max(u, 1) / warmup
        progress = (u - warmup) / max(total_units - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_onecycle(optimizer, total_opt_steps: int, max_lr: float,
                   pct_start: float = 0.3, div_factor: float = 25.0,
                   final_div_factor: float = 1e4):
    """OneCycleLR over optimizer steps, as in the reference Transolver /
    SpiderSolver code. Stepping past total_steps raises, which catches a
    resumed run whose step count drifted."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=int(total_opt_steps),
        pct_start=pct_start, div_factor=div_factor,
        final_div_factor=final_div_factor, anneal_strategy="cos")


class GaotMixScheduler(_LRScheduler):
    """GAOT's published schedule (its src/utils/optimizers.py), stepped once
    per epoch: linear warmup initial_lr -> max_lr over int(0.02 * epochs),
    cosine max_lr -> min_lr over int(0.90 * epochs), geometric decay
    min_lr -> final_lr over the remainder. The zero-length guards (a phase
    borrowing one epoch from the cosine phase) are the upstream ones."""

    def __init__(self, optimizer, total_epochs: int, initial_lr: float,
                 max_lr: float, min_lr: float, final_lr: float,
                 warmup_fraction: float = 0.02,
                 cosine_fraction: float = 0.90, last_epoch: int = -1):
        warmup = int(warmup_fraction * total_epochs)
        cosine = int(cosine_fraction * total_epochs)
        exp_decay = total_epochs - warmup - cosine
        if warmup == 0:
            warmup = 1
            cosine -= 1
        if exp_decay == 0:
            exp_decay = 1
            cosine -= 1
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup)
        self.cosine_epochs = max(1, int(cosine))
        self.exp_decay_epochs = int(exp_decay)
        self.initial_lr = float(initial_lr)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.final_lr = float(final_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
                e / max(1, self.warmup_epochs - 1))
        elif e < self.warmup_epochs + self.cosine_epochs:
            t = e - self.warmup_epochs
            ratio = (1 + math.cos(math.pi * t / self.cosine_epochs)) / 2
            lr = self.min_lr + (self.max_lr - self.min_lr) * ratio
        else:
            t = e - self.warmup_epochs - self.cosine_epochs
            steps = max(1, self.exp_decay_epochs - 1)
            lr = self.min_lr * ((self.final_lr / self.min_lr) ** (t / steps))
        return [lr for _ in self.optimizer.param_groups]

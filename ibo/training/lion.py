"""Lion optimizer (Chen et al. 2023, "Symbolic Discovery of Optimization
Algorithms").

    update = sign(beta1 * m + (1 - beta1) * g)
    p     <- p - lr * (update + weight_decay * p)     # decoupled decay
    m     <- beta2 * m + (1 - beta2) * g

The sign update moves every parameter by lr per step regardless of the
gradient magnitude, so lr is absolute (no batch scaling) and 3-10x smaller
than an AdamW lr; the effective decay is lr * weight_decay.
"""
from __future__ import annotations

import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        assert lr > 0.0
        assert all(0.0 <= b < 1.0 for b in betas)
        assert weight_decay >= 0.0
        super().__init__(params, dict(lr=lr, betas=betas,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                update = m.mul(beta1).add_(g, alpha=1.0 - beta1).sign_()
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)
        return loss

"""FLOPs accounting for one full-field prediction.

The Linear-dominant part of a forward is counted exactly by forward hooks
over the real modules during one eager pass (nn.Linear: 2*rows*in*out +
rows*out; norms 5/element; activations 4/element; embeddings are gathers,
0). What hooks cannot see (SDPA, einsum contractions, segment reductions,
RoPE / Fourier trigonometry) is added by each recipe's closed forms from
the same shapes the pass ran (Recipe.manual_infer_flops).

Conventions: 1 MAC = 2 FLOPs; SDPA(q, kv, H, d) = 4*H*q*kv*d + 3*H*q*kv;
einsum over c: 2 * out_elems * c; segment mean: rows*C + groups*C;
sin/cos: 1 each; gathers, sorts and tree searches: 0.
"""
from __future__ import annotations

import torch.nn as nn

FLOPS_CONVENTIONS = {
    "mac": 2, "sdpa": "4*H*q*kv*d + 3*H*q*kv", "norm_per_elem": 5,
    "act_per_elem": 4, "trig_per_elem": 1, "gather_index_ops": 0,
}


class FlopMeter:
    """Forward-hook meter over one eager pass. Softmax modules are not
    hooked: every softmax of these models sits inside a manual attention
    term already."""

    _ACT_TYPES = (nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.Sigmoid,
                  nn.Tanh, nn.ELU)

    def __init__(self, model: nn.Module):
        self.model = model
        self.linear = self.norm = self.activation = 0.0
        self._handles = []
        norm_types = [nn.LayerNorm, nn.BatchNorm1d]
        if hasattr(nn, "RMSNorm"):
            norm_types.append(nn.RMSNorm)
        self._norm_types = tuple(norm_types)

    def _hook_linear(self, mod, inputs, output):
        rows = output.numel() / output.shape[-1]
        self.linear += 2.0 * rows * mod.in_features * mod.out_features
        if mod.bias is not None:
            self.linear += rows * mod.out_features

    def _hook_conv(self, mod, inputs, output):
        k = 1.0
        for s in mod.kernel_size:
            k *= s
        self.linear += output.numel() * 2.0 * (mod.in_channels / mod.groups) * k
        if mod.bias is not None:
            self.linear += output.numel()

    def _hook_norm(self, mod, inputs, output):
        self.norm += 5.0 * output.numel()

    def _hook_act(self, mod, inputs, output):
        self.activation += 4.0 * output.numel()

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._hook_linear))
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self._handles.append(m.register_forward_hook(self._hook_conv))
            elif isinstance(m, self._norm_types):
                self._handles.append(m.register_forward_hook(self._hook_norm))
            elif isinstance(m, self._ACT_TYPES):
                self._handles.append(m.register_forward_hook(self._hook_act))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    @property
    def total(self) -> float:
        return self.linear + self.norm + self.activation

    def breakdown(self) -> dict:
        return {"linear": self.linear, "norm": self.norm, "activation": self.activation,
                "conventions": FLOPS_CONVENTIONS}


def sdpa_flops(q_rows: float, kv_rows: float, heads: float, head_dim: float) -> float:
    return 4.0 * heads * q_rows * kv_rows * head_dim + 3.0 * heads * q_rows * kv_rows


def einsum_flops(out_elems: float, contracted: float) -> float:
    return 2.0 * out_elems * contracted


def segment_mean_flops(rows: float, channels: float, groups: float) -> float:
    return rows * channels + groups * channels

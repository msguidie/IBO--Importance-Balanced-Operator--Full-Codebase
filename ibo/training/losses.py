"""The training losses, in fp32, with one breakdown contract for the epoch
accumulators: vol_loss_sum [C_vol], surf_loss_sum [C_surf], n_vol, n_surf.

compute_loss         mixed rows: zero-padded per-side targets plus an
                     is_volume mask (ibo, transolver, erwin).
compute_loss_branch  separate volume and surface branches (abupt, spider).

reduction "group": loss = weighted mean over volume channels + mean over
surface channels (UrbanWind, TandemFoilSet, AirfRANS). reduction "flat":
loss = mean over all volume and surface channels together (DrivAerML).
"""
from __future__ import annotations

import torch


def _weighted_vol(vol_per: torch.Tensor, vol_weights) -> torch.Tensor:
    if vol_weights is None:
        return vol_per.mean()
    w = torch.as_tensor(vol_weights, dtype=vol_per.dtype, device=vol_per.device)
    return (vol_per * w).sum() / w.sum().clamp(min=1e-12)


def _combine(vol_per, surf_per, vol_weights, reduction: str):
    if reduction == "flat":
        return torch.cat([vol_per, surf_per]).mean()
    return _weighted_vol(vol_per, vol_weights) + surf_per.mean()


def compute_loss(pred_vol, pred_surf, target_vol, target_surf, is_volume,
                 vol_weights=None, reduction: str = "group"):
    """Masked per-channel MSE over mixed volume / surface rows."""
    vol_mask = is_volume[..., None].float()
    surf_mask = 1.0 - vol_mask
    dv = (pred_vol.float() - target_vol.float()) ** 2 * vol_mask
    ds = (pred_surf.float() - target_surf.float()) ** 2 * surf_mask
    vol_loss_sum = dv.sum(dim=(0, 1))
    surf_loss_sum = ds.sum(dim=(0, 1))
    n_vol = vol_mask.sum()
    n_surf = surf_mask.sum()
    loss = _combine(vol_loss_sum / n_vol.clamp(min=1),
                    surf_loss_sum / n_surf.clamp(min=1), vol_weights, reduction)
    breakdown = {"vol_loss_sum": vol_loss_sum.detach(),
                 "surf_loss_sum": surf_loss_sum.detach(),
                 "n_vol": n_vol.detach(), "n_surf": n_surf.detach()}
    return loss, breakdown


def compute_loss_branch(pred_vol, pred_surf, target_vol, target_surf,
                        vol_weights=None, reduction: str = "group"):
    """Branch-separated MSE: no mask, no padded rows, static row counts."""
    dv = (pred_vol.float() - target_vol.float()) ** 2
    ds = (pred_surf.float() - target_surf.float()) ** 2
    vol_loss_sum = dv.sum(dim=(0, 1))
    surf_loss_sum = ds.sum(dim=(0, 1))
    n_vol = dv.new_tensor(float(dv.shape[0] * dv.shape[1]))
    n_surf = ds.new_tensor(float(ds.shape[0] * ds.shape[1]))
    loss = _combine(vol_loss_sum / n_vol.clamp(min=1),
                    surf_loss_sum / n_surf.clamp(min=1), vol_weights, reduction)
    breakdown = {"vol_loss_sum": vol_loss_sum.detach(),
                 "surf_loss_sum": surf_loss_sum.detach(),
                 "n_vol": n_vol.detach(), "n_surf": n_surf.detach()}
    return loss, breakdown

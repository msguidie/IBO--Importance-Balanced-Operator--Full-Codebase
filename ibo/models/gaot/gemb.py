"""Geometric embedding (statistical method) — from GAOT
src/model/layers/gemb.py.

The statistical features of a query node are pure functions of the FIXED
per-case geometry (its neighbor coordinates), so [PORT] the module is
split in two:

  * compute_statistical_features(...) — the feature computation, run ONCE
    per case by the derive step (datasets/tandemfoilset/gaot_derive.py). Native torch ops replace
    the upstream torch_scatter calls (scatter_mean/sum via index_add_);
    the math — neighbor count, mean/var of distances, centroid offset,
    covariance eigenvalues, then a mean/std normalisation ACROSS query
    nodes — is upstream's exactly, including the per-case (not per-batch)
    normalisation population.
  * GeoEmbedMLP — the learnable head (Linear-ReLU-Linear-ReLU, upstream's
    'statistical' branch), applied to the precomputed features at train
    time. This is the ONLY part with parameters, so precomputing the
    features changes nothing numerically.

The upstream 'pointnet' embedding method is not vendored (GAOT's default
is 'statistical').
"""
import torch
import torch.nn as nn


def _scatter_mean(src: torch.Tensor, row: torch.Tensor, n_rows: int
                  ) -> torch.Tensor:
    shape = (n_rows,) if src.ndim == 1 else (n_rows, src.shape[1])
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    out.index_add_(0, row, src)
    cnt = torch.zeros(n_rows, dtype=src.dtype, device=src.device)
    cnt.index_add_(0, row, torch.ones_like(row, dtype=src.dtype))
    cnt = cnt.clamp(min=1.0)
    return out / (cnt if src.ndim == 1 else cnt.unsqueeze(-1))


def compute_statistical_features(input_geom: torch.Tensor,
                                 latent_queries: torch.Tensor,
                                 neighbors_index: torch.Tensor,
                                 neighbors_row_splits: torch.Tensor
                                 ) -> torch.Tensor:
    """[M, 3 + 2*d] normalised statistical geometry features (fp32).

    Per query node: [N_i, D_avg, D_var, Delta (d), PCA eigvals (d)],
    zeroed where the node has no neighbors, then standardised across the
    M query nodes (std floored at 1e-6 -> 1.0), exactly as upstream."""
    input_geom = input_geom.float()
    latent_queries = latent_queries.float()
    m = latent_queries.shape[0]
    d = latent_queries.shape[1]
    device = latent_queries.device

    counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
    row = torch.repeat_interleave(
        torch.arange(m, device=device), counts)                    # [E]
    nbr = input_geom[neighbors_index]                              # [E, d]
    qry = latent_queries[row]                                      # [E, d]

    n_i = counts.float()
    has = n_i > 0

    dist = torch.norm(nbr - qry, dim=1)                            # [E]
    d_avg = _scatter_mean(dist, row, m)                            # [M]
    e_x2 = _scatter_mean(dist ** 2, row, m)
    d_var = (e_x2 - d_avg ** 2).clamp(min=0.0)

    centroid = _scatter_mean(nbr, row, m)                          # [M, d]
    delta = centroid - latent_queries

    centered = nbr - centroid[row]                                 # [E, d]
    cov_flat = (centered.unsqueeze(2) * centered.unsqueeze(1)
                ).reshape(-1, d * d)                               # [E, d*d]
    cov_sum = torch.zeros(m, d * d, device=device)
    cov_sum.index_add_(0, row, cov_flat)
    cov = (cov_sum / n_i.clamp(min=1.0).unsqueeze(-1)).reshape(m, d, d)

    pca = torch.zeros(m, d, device=device)
    if bool(has.any()):
        eig = torch.linalg.eigvalsh(cov[has])                      # ascending
        pca[has] = eig.flip(dims=[1])                              # descending

    feats = torch.cat([n_i.unsqueeze(1), d_avg.unsqueeze(1),
                       d_var.unsqueeze(1), delta, pca], dim=1)
    feats[~has] = 0.0

    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return ((feats - mean) / std).float()


class GeoEmbedMLP(nn.Module):
    """Upstream GeometricEmbedding's 'statistical' head: features -> MLP.
    input_dim is the coordinate dim d; feature dim = 3 + 2*d."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        num_features = 3 + 2 * input_dim
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.ReLU())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

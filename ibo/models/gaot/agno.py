"""AGNO — Attentional Graph Neural Operator, from GAOT
src/model/layers/agno.py.

Computes the attentionally-weighted kernel integral transform
    out(x) = sum_{y in A(x)} alpha(x, y) * k(x, y) * f(y)
over a CSR neighborhood graph: per-edge kernel MLP k, cosine-similarity
attention alpha normalised by a per-row segment softmax, segment_csr
aggregation.

[PORT] deltas vs upstream (marked inline):
  * segment_csr is the native index_add_ implementation (no torch_scatter).
  * forward() takes an optional precomputed per-edge ``attention_weights``
    — with cosine attention alpha depends ONLY on the two endpoint
    coordinates, which are fixed per case, so the bake stage computes it
    once and training skips the softmax entirely. None -> original path.
  * Only the pieces the tandem port exercises are kept: transform_type
    'linear', cosine attention, unbatched [n, C] features (batching is by
    disjoint-union graph merge upstream of this module). The dot_product
    branch is dropped because its projections are LEARNABLE — it could
    not be precomputed and is not GAOT's default anyway; asking for it
    raises instead of silently differing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import LinearChannelMLP
from .segment_csr import segment_csr


def segment_softmax(scores: torch.Tensor, splits: torch.Tensor,
                    ) -> torch.Tensor:
    """Row-wise softmax over CSR segments — verbatim upstream math."""
    max_values = segment_csr(scores, splits, reduce="max")
    counts = splits[1:] - splits[:-1]
    scores = scores - max_values.repeat_interleave(counts, dim=0)
    exp_scores = torch.exp(scores)
    sum_exp = segment_csr(exp_scores, splits, reduce="sum")
    return exp_scores / sum_exp.repeat_interleave(counts, dim=0).clamp_min(
        torch.finfo(scores.dtype).tiny)


def cosine_attention_scores(query_coords: torch.Tensor,
                            key_coords: torch.Tensor) -> torch.Tensor:
    """alpha-scores before the softmax: cosine similarity of the raw
    coordinate vectors (upstream's exact formula — NOT of displacement
    vectors)."""
    q = F.normalize(query_coords, p=2, dim=-1)
    k = F.normalize(key_coords, p=2, dim=-1)
    return (q * k).sum(dim=-1)


class AGNO(nn.Module):
    def __init__(self, channel_mlp_layers, transform_type="linear",
                 use_attn=True, attention_type="cosine",
                 channel_mlp_non_linearity=F.gelu):
        super().__init__()
        if transform_type != "linear":
            raise ValueError(
                f"[PORT] only transform_type='linear' is vendored "
                f"(GAOT's default); got {transform_type!r}")
        if use_attn and attention_type != "cosine":
            raise ValueError(
                f"[PORT] only cosine attention is vendored (GAOT's "
                f"default; dot_product has learnable projections and "
                f"cannot be precomputed); got {attention_type!r}")
        self.use_attn = bool(use_attn)
        self.channel_mlp = LinearChannelMLP(
            layers=channel_mlp_layers,
            non_linearity=channel_mlp_non_linearity)

    def forward(self, y_coords: torch.Tensor, x_coords: torch.Tensor,
                f_y: torch.Tensor, neighbors_index: torch.Tensor,
                neighbors_row_splits: torch.Tensor,
                attention_weights: torch.Tensor | None = None
                ) -> torch.Tensor:
        """One kernel integral transform over a CSR graph.

        y_coords [n, d]   source point coordinates
        x_coords [m, d]   query point coordinates (CSR rows)
        f_y      [n, C]   features on the source points
        neighbors_index [E] int64      edge -> source row
        neighbors_row_splits [m+1]     CSR row pointers
        attention_weights [E] or None  precomputed softmaxed alpha
        returns  [m, C]
        """
        counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        rep_coords = y_coords[neighbors_index]                    # [E, d]
        self_coords = torch.repeat_interleave(x_coords, counts, dim=0)

        if self.use_attn:
            if attention_weights is None:
                scores = cosine_attention_scores(self_coords, rep_coords)
                attention_weights = segment_softmax(
                    scores, neighbors_row_splits)
            alpha = attention_weights
        else:
            alpha = None

        agg = torch.cat([rep_coords, self_coords], dim=-1)        # [E, 2d]
        kernel = self.channel_mlp(agg)                            # [E, C]
        edge_feat = kernel * f_y[neighbors_index]                 # [E, C]
        if alpha is not None:
            edge_feat = edge_feat * alpha.unsqueeze(-1)
            reduction = "sum"     # softmax weights already normalise the row
        else:
            reduction = "mean"    # upstream: mean when attention is off
        return segment_csr(edge_feat, neighbors_row_splits, reduce=reduction)

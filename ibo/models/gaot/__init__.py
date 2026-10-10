"""GAOT (Wen et al., 2025), layers ported from https://github.com/camlab-ethz/GAOT
(no licence file published upstream at the time of porting; see
THIRD_PARTY_NOTICES.md). Two kinds of edits, both marked [PORT] at the
call site: dependency removal (torch_scatter, torch_cluster,
rotary_embedding_torch, omegaconf) and an optional precomputed per-edge
attention tensor for the cosine attention, which is a pure function of
the fixed coordinates. Passing None reproduces the original computation.
"""
from .agno import AGNO, segment_softmax
from .attn import AttentionConfig, Transformer, TransformerConfig
from .gemb import GeoEmbedMLP, compute_statistical_features
from .mlp import ChannelMLP, LinearChannelMLP
from .model import GAOT, build_gaot, gather_csr_rows, merge_csr
from .segment_csr import segment_csr

__all__ = ["AGNO", "segment_softmax", "Transformer", "TransformerConfig",
           "AttentionConfig", "GeoEmbedMLP", "compute_statistical_features",
           "ChannelMLP", "LinearChannelMLP", "segment_csr", "GAOT", "build_gaot",
           "gather_csr_rows", "merge_csr"]

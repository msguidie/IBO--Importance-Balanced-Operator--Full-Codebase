"""Equivalence test of the vectorised IBO decoder against a per-neighbour
loop implementation of the same math, on the same module and weights.

  fp32: max|new - ref| / max|ref| < 1e-5   (CPU, always run)
  bf16: < 5e-3 under autocast on CUDA      (skipped without CUDA)

plus a gradient check (every decoder parameter receives a finite
gradient). The feature groups are UrbanWind's; N and L are shrunk so the
test runs on a CPU in seconds, B = 2 exercises the batched gather.

    python -m pytest tests/        or        python tests/test_decoder_equivalence.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibo.datasets.base import FeatureGroup  # noqa: E402
from ibo.models.ibo.decoder import Decoder  # noqa: E402

B, N, L, K = 2, 4096, 512, 16
D_LATENT, D_H, H, R = 384, 384, 6, 8
GROUPS = [FeatureGroup("xy", 4, "fourier", n_freqs=8), FeatureGroup("z", 3, "fourier", n_freqs=6),
          FeatureGroup("support", 6, "plain"),
          FeatureGroup("point_type", 1, "embed", n_classes=2, embed_dim=4),
          FeatureGroup("region", 1, "embed", n_classes=5, embed_dim=4),
          FeatureGroup("edge", 1, "embed", n_classes=10, embed_dim=4)]


def make_decoder(register_attn: bool) -> Decoder:
    return Decoder(D_LATENT, D_H, H, 3, GROUPS, (8, 8, 6), True, True, cond_dim=0,
                   cond_concat=False, stages=2, n_register=R, register_attn=register_attn,
                   head_hidden=256, n_vol=5, n_surf=1, lambda_clip=None, log_ratio_clip=None)


def reference_forward(dec, encoder_out, vit_out, query_pos_geo, feats, knn, latent_pos_geo,
                      is_volume, cond=None):
    """The decoder's math as a loop over the K neighbours."""
    knn = knn.long()
    B_, N_, K_ = knn.shape
    dtype = vit_out.dtype
    batch_idx = torch.arange(B_, device=knn.device)[:, None]
    R_, hd = dec.n_register, dec.head_dim
    vit_cells = vit_out[:, :-R_, :] if R_ > 0 else vit_out
    vit_regs = vit_out[:, -R_:, :] if R_ > 0 else None

    qpg, cpg = query_pos_geo.float(), latent_pos_geo.float()
    deltas, dists = [], []
    for j in range(K_):
        delta = qpg - cpg[batch_idx, knn[:, :, j]]
        d = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        deltas.append(delta)
        dists.append(d)
    dbar = torch.stack(dists, dim=2).mean(dim=2).clamp(min=1e-6)
    log_dbar = torch.log(dbar)

    parts = dec.query_embed(feats, dtype) + [log_dbar.to(dtype)]
    if dec.cond_concat:
        parts.append(cond.to(dtype)[:, None, :].expand(B_, N_, -1))
    q_self = torch.cat(parts, dim=-1)
    mask = is_volume.to(dtype)[..., None]
    h = mask * F.silu(dec.self_proj1_vol(q_self)) + (1.0 - mask) * F.silu(dec.self_proj1_surf(q_self))
    h = dec.self_proj2(h) + dec.type_emb(is_volume.long())

    p_list = []
    for j in range(K_):
        delta, d = deltas[j], dists[j]
        rel = delta / dbar
        pp = [f(rel[..., a:a + 1]).to(dtype) for a, f in enumerate(dec.pair_fourier)]
        if dec.pair_unit_dir:
            pp.append((delta / d).to(dtype))
        pp.append(torch.log(d / dbar).to(dtype))
        if dec.pair_log_dbar:
            pp.append(log_dbar.to(dtype))
        p_list.append(F.silu(dec.pair_proj(torch.cat(pp, dim=-1))))

    for si, st in enumerate(dec.stages):
        src = st.src_norm(encoder_out if si < len(dec.stages) - 1 else vit_cells)
        k_tab = st.k_head_norm(dec._split(st.k_proj(src)))
        v_tab = st.v_proj(src)
        h_normed = st.h_norm(h)
        scores, values = [], []
        for j in range(K_):
            ij = knn[:, :, j]
            q = st.q_head_norm(dec._split(st.q_proj(h_normed + p_list[j])))
            kj = k_tab[batch_idx, ij]
            vj = v_tab[batch_idx, ij] + st.gamma * p_list[j]
            score = (q.float() * kj.float()).sum(-1) / (hd ** 0.5)
            score = score - st.lam[None, None, :] * torch.log(dists[j] / dbar)
            scores.append(score)
            values.append(dec._split(vj))
        alpha = F.softmax(torch.stack(scores, dim=2), dim=2).to(dtype)
        agg = sum(alpha[:, :, j, :, None] * values[j] for j in range(K_))
        if st.gate_reg is not None:
            reg_src = st.src_norm(vit_regs)
            k_reg = st.k_head_norm(dec._split(st.k_proj(reg_src)))
            v_reg = dec._split(st.v_proj(reg_src))
            q_reg = st.q_head_norm(dec._split(st.q_proj(h_normed)))
            reg_scores = torch.einsum("bnhd,brhd->bnrh", q_reg.float(), k_reg.float()) / (hd ** 0.5)
            alpha_reg = F.softmax(reg_scores, dim=2).to(dtype)
            agg = agg + st.gate_reg[None, None, :, None] * torch.einsum("bnrh,brhd->bnhd", alpha_reg, v_reg)
        h = h + st.out_proj(agg.reshape(B_, N_, dec.d_h))
        h = h + st.ffn(st.ffn_norm(h))
    pred = dec.pred(F.silu(dec.head(h)))
    return pred[..., :dec.n_vol], pred[..., dec.n_vol:]


def make_inputs(device=torch.device("cpu"), dtype=torch.float32):
    g = torch.Generator().manual_seed(7)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dtype)

    is_volume = (torch.arange(N)[None, :].expand(B, N) % 2 == 0).to(device)   # both row types
    feats = (rand(B, N, 4) * 0.3, rand(B, N, 3), rand(B, N, 6),
             (~is_volume).long(), torch.randint(0, 5, (B, N), generator=g).to(device),
             torch.randint(0, 10, (B, N), generator=g).to(device))
    return dict(encoder_out=rand(B, L, D_LATENT), vit_out=rand(B, L + R, D_LATENT),
                query_pos_geo=rand(B, N, 3) * 0.3, feats=feats,
                knn=torch.randint(0, L, (B, N, K), generator=g).to(device),
                latent_pos_geo=rand(B, L, 3) * 0.3, is_volume=is_volume)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-12))


def test_vectorized_forward_matches_loop_reference_fp32():
    torch.manual_seed(0)
    dec = make_decoder(register_attn=True).eval()
    inputs = make_inputs()
    with torch.no_grad():
        ref_vol, ref_surf = reference_forward(dec, **inputs)
        new_vol, new_surf = dec(**inputs, cond=None)
    ev, es = rel_err(new_vol, ref_vol), rel_err(new_surf, ref_surf)
    assert ev < 1e-5, f"fp32 gate failed (volume): rel={ev:.3e}"
    assert es < 1e-5, f"fp32 gate failed (surface): rel={es:.3e}"
    assert math.isfinite(float(new_vol.sum())) and math.isfinite(float(new_surf.sum()))


def test_vectorized_forward_matches_loop_reference_no_register_path():
    torch.manual_seed(1)
    dec = make_decoder(register_attn=False).eval()
    inputs = make_inputs()
    with torch.no_grad():
        ref_vol, ref_surf = reference_forward(dec, **inputs)
        new_vol, new_surf = dec(**inputs, cond=None)
    assert rel_err(new_vol, ref_vol) < 1e-5
    assert rel_err(new_surf, ref_surf) < 1e-5


def test_vectorized_backward_all_params_get_finite_grads():
    torch.manual_seed(2)
    dec = make_decoder(register_attn=True).train()
    pv, ps = dec(**make_inputs(), cond=None)
    (pv.float().pow(2).mean() + ps.float().pow(2).mean()).backward()
    missing = [n for n, p in dec.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing[:5]}"
    bad = [n for n, p in dec.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad[:5]}"


def test_vectorized_forward_matches_loop_reference_bf16_cuda():
    if not torch.cuda.is_available():
        print("  [skip] bf16 gate needs CUDA")
        return
    device = torch.device("cuda:0")
    torch.manual_seed(3)
    dec = make_decoder(register_attn=True).to(device).eval()
    inputs = make_inputs(device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_vol, ref_surf = reference_forward(dec, **inputs)
        new_vol, new_surf = dec(**inputs, cond=None)
    ev, es = rel_err(new_vol, ref_vol), rel_err(new_surf, ref_surf)
    assert ev < 5e-3, f"bf16 gate failed (volume): rel={ev:.3e}"
    assert es < 5e-3, f"bf16 gate failed (surface): rel={es:.3e}"


if __name__ == "__main__":
    test_vectorized_forward_matches_loop_reference_fp32()
    test_vectorized_forward_matches_loop_reference_no_register_path()
    test_vectorized_backward_all_params_get_finite_grads()
    test_vectorized_forward_matches_loop_reference_bf16_cuda()
    print("decoder equivalence: PASS")

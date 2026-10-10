"""Numerical equivalence of each optimised baseline implementation against
its reference at fp64: AB-UPT's anchor K/V cache, chunked Transolver-3,
SpiderSolver's attention rewrite, Erwin's ball tree, restore index and
grouped mean.

    python -m pytest tests/        or        python tests/test_gates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibo.models.abupt import AnchoredBranchedUPT  # noqa: E402
from ibo.models.erwin.balltree import build_tree, knn_edges, restore_index  # noqa: E402
from ibo.models.erwin.erwin import grouped_mean, scatter_mean  # noqa: E402
from ibo.models.spider import SpiderAttention  # noqa: E402
from ibo.models.transolver import Transolver  # noqa: E402

TOL = 1e-10


def dense_spider_attention(attn, x, n_vol, vol_gid, inner_idx, surf_region_id):
    """The original SpiderSolver computation (dense one-hot sector means,
    explicit softmax attention) on SpiderAttention's weights."""
    B, N, C = x.shape
    assert B == 1
    H, D = attn.heads, attn.dim_head
    scale = D ** -0.5
    n_surf = N - n_vol
    G = attn.n_vol_tokens

    def heads(t):
        b, T, _ = t.shape
        return t.view(b, T, H, D).permute(0, 2, 1, 3)

    def unheads(t):
        b, h, T, d = t.shape
        return t.permute(0, 2, 1, 3).reshape(b, T, h * d)

    def attend(q, k, v):
        return torch.matmul((torch.matmul(q, k.transpose(-1, -2)) * scale).softmax(dim=-1), v)

    fx_mid = attn.in_project_fx(x)
    vol_fx, surf_fx = fx_mid[:, :n_vol], fx_mid[:, n_vol:]
    inner_onehot = torch.zeros(inner_idx.shape[0], n_vol, dtype=x.dtype, device=x.device)
    inner_onehot[torch.arange(inner_idx.shape[0]), inner_idx] = 1.0
    inner_fx = torch.matmul(inner_onehot, vol_fx[0]).unsqueeze(0)
    lap_h = heads(torch.cat([inner_fx, surf_fx], dim=1))
    out1 = unheads(attend(attn.to_q_first_lap(lap_h), attn.to_k_first_lap(lap_h), attn.to_v_first_lap(lap_h)))
    n_inner = inner_fx.shape[1]
    coarse_inner, coarse_surf = out1[:, :n_inner], out1[:, n_inner:]

    onehot_vol = torch.zeros(G, n_vol, dtype=x.dtype, device=x.device)
    onehot_vol[vol_gid, torch.arange(n_vol)] = 1.0
    vol_tok = torch.matmul(onehot_vol, vol_fx[0]) / float(n_vol)
    if attn.surf_token_mode == "single":
        surf_tok = surf_fx[0].mean(dim=0, keepdim=True)
    else:
        onehot_surf = torch.zeros(attn.n_surf_tokens, n_surf, dtype=x.dtype, device=x.device)
        onehot_surf[surf_region_id, torch.arange(n_surf)] = 1.0
        surf_tok = torch.matmul(onehot_surf, surf_fx[0]) / float(n_surf)
    tokens = torch.cat([surf_tok, vol_tok], dim=0).unsqueeze(0)
    if attn.surf_token_mode == "per_region":
        tokens = attn.norm(tokens.permute(0, 2, 1)).permute(0, 2, 1)
    tok_h = heads(tokens)
    out_tok = unheads(attend(attn.to_q(tok_h), attn.to_k(tok_h), attn.to_v(tok_h)))[0]
    tok_surf, tok_vol = out_tok[:attn.n_surf_tokens], out_tok[attn.n_surf_tokens:]

    x_mid = attn.in_project_x(x)
    vol_x, surf_x = x_mid[:, :n_vol], x_mid[:, n_vol:]
    feature_surf = attn.weight_surf(surf_x) @ tok_surf.unsqueeze(0)
    feature_velo = attn.weight_velo_velo(vol_x) @ tok_vol.unsqueeze(0)
    feature_velo_car = attn.weight_velo_car(surf_x) @ tok_vol.unsqueeze(0)
    feature_surf = (feature_surf + feature_velo_car) / 2.0 + coarse_surf
    feature_velo = feature_velo + torch.matmul(inner_onehot.transpose(0, 1), coarse_inner[0]).unsqueeze(0)
    return attn.to_out(torch.cat([feature_velo, feature_surf], dim=1))


def test_abupt_kv_cache():
    """Cached anchor-K/V decode == forward_with_queries, with unequal branch
    lengths; 3-D without and 2-D with a case condition."""
    for ndim, cond_dim in ((3, 0), (2, 2)):
        torch.manual_seed(0)
        m = AnchoredBranchedUPT(ndim=ndim, dim=24, num_heads=3, shared_blocks="sscscs",
                                num_surface_blocks=2, num_volume_blocks=2, output_dim_surface=1,
                                output_dim_volume=5, pos_offset=(1.0,) * ndim, pos_scale=10.0,
                                cond_dim=cond_dim).double().eval()
        ns_a, nv_a, ns_q, nv_q = 16, 20, 7, 11
        sp = torch.rand(1, ns_a + ns_q, ndim, dtype=torch.float64)
        vp = torch.rand(1, nv_a + nv_q, ndim, dtype=torch.float64)
        cond = torch.rand(1, cond_dim, dtype=torch.float64) if cond_dim else None
        with torch.no_grad():
            ps_ref, pv_ref = m.forward_with_queries(sp, vp, num_surface_anchors=ns_a,
                                                    num_volume_anchors=nv_a, cond=cond)
            ps_a, pv_a, cache = m.encode_anchor_cache(sp[:, :ns_a], vp[:, :nv_a], cond=cond)
            ps_q, pv_q = m.decode_queries_cached(cache, sp[:, ns_a:], vp[:, nv_a:], cond=cond)
            ps_o, pv_o = m(sp[:, :ns_a], vp[:, :nv_a], cond)
        assert (ps_ref[:, :ns_a] - ps_a).abs().max() < TOL
        assert (pv_ref[:, :nv_a] - pv_a).abs().max() < TOL
        assert (ps_ref[:, ns_a:] - ps_q).abs().max() < TOL
        assert (pv_ref[:, nv_a:] - pv_q).abs().max() < TOL
        assert (ps_o - ps_a).abs().max() == 0 and (pv_o - pv_a).abs().max() == 0
    print("gate abupt_kv_cache: PASS")


def test_transolver_chunks():
    """Fused forward == plain-block reference; multi-chunk global aggregation
    == single chunk, forward and gradients; cached slice weights == recompute."""
    torch.manual_seed(1)
    m = Transolver(in_dim=7, n_hidden=32, n_layers=3, n_heads=4, slice_num=8, mlp_ratio=2,
                   out_dim_vol=5).double().eval()
    x = torch.rand(1, 300, 7, dtype=torch.float64)
    with torch.no_grad():
        fx = m.embed(x)
        for b in m.blocks:
            fx = b(fx)
        h = m.ln_out(fx)
        pv_ref, ps_ref = m.vol_head(h), m.surf_head(h)
        pv1, ps1 = m(x)
        outs = m.forward_chunks([x[:, :101], x[:, 101:207], x[:, 207:]])
        pv_m = torch.cat([o[0] for o in outs], dim=1)
        ps_m = torch.cat([o[1] for o in outs], dim=1)
    assert (pv_ref - pv1).abs().max() < TOL
    assert (pv_m - pv1).abs().max() < TOL and (ps_m - ps1).abs().max() < TOL

    m2 = Transolver(in_dim=7, n_hidden=32, n_layers=2, n_heads=4, slice_num=8, mlp_ratio=2,
                    out_dim_vol=5).double()
    x2 = torch.rand(1, 200, 7, dtype=torch.float64)
    pv, ps = m2(x2)
    (pv.square().mean() + ps.square().mean()).backward()
    g1 = {n: p.grad.clone() for n, p in m2.named_parameters() if p.grad is not None}
    m2.zero_grad()
    outs = m2.forward_chunks([x2[:, :77], x2[:, 77:]])
    pv2 = torch.cat([o[0] for o in outs], 1)
    ps2 = torch.cat([o[1] for o in outs], 1)
    (pv2.square().mean() + ps2.square().mean()).backward()
    gd = max((g1[n] - p.grad).abs().max().item() for n, p in m2.named_parameters() if p.grad is not None)
    assert gd < 1e-9

    m3 = Transolver(in_dim=7, n_hidden=32, n_layers=3, n_heads=4, slice_num=8, mlp_ratio=2,
                    out_dim_vol=3).double().eval()
    with torch.no_grad():
        for p_ in m3.parameters():
            p_.add_(0.01 * torch.randn_like(p_))
        x3 = torch.rand(1, 257, 7, dtype=torch.float64)
        (pv_c, ps_c), = m3.forward_chunks([x3], cache_weights=True)
        (pv_r, ps_r), = m3.forward_chunks([x3], cache_weights=False)
    assert (pv_c - pv_r).abs().max().item() == 0.0 and (ps_c - ps_r).abs().max().item() == 0.0
    print("gate transolver_chunks: PASS")


def test_spider_rewrite():
    """Segment-sum / SDPA rewrite == the dense reference, both surface-token
    modes, forward, input gradients and weight gradients."""
    torch.manual_seed(3)
    for mode, K in (("per_region", 6), ("single", 4)):
        attn = SpiderAttention(dim=32, heads=4, dropout=0.0, onion_num=5, n_clusters=K,
                               surf_token_mode=mode).double().eval()
        n_vol, n_surf, n_inner = 400, 60, 40
        x = torch.rand(1, n_vol + n_surf, 32, dtype=torch.float64, requires_grad=True)
        gid = torch.randint(0, 5 * K, (n_vol,))
        inner = torch.randperm(n_vol)[:n_inner]
        reg = torch.randint(0, K, (n_surf,)) if mode == "per_region" else None
        y1 = attn(x, n_vol, gid, inner, reg)
        y2 = dense_spider_attention(attn, x, n_vol, gid, inner, reg)
        assert (y1 - y2).abs().max() < TOL
        g1 = torch.autograd.grad(y1.square().mean(), x)[0]
        x2 = x.detach().clone().requires_grad_(True)
        g2 = torch.autograd.grad(dense_spider_attention(attn, x2, n_vol, gid, inner, reg).square().mean(), x2)[0]
        assert (g1 - g2).abs().max() < TOL
        attn.zero_grad()
        attn(x.detach(), n_vol, gid, inner, reg).square().mean().backward()
        gw1 = {n: p.grad.clone() for n, p in attn.named_parameters() if p.grad is not None}
        attn.zero_grad()
        dense_spider_attention(attn, x.detach(), n_vol, gid, inner, reg).square().mean().backward()
        gwd = max((gw1[n] - p.grad).abs().max().item() for n, p in attn.named_parameters()
                  if p.grad is not None and n in gw1)
        assert gwd < TOL
    print("gate spider_rewrite: PASS")


def test_balltree_contract():
    torch.manual_seed(5)
    for N in (1000, 1024, 777):
        pos = torch.rand(N, 3)
        ti, tm, rot = build_tree(pos, strides=[2, 2], ball_sizes=[64, 64, 64], rotate=45)
        P = ti.shape[0]
        assert P % 64 == 0 and P >= N
        assert tm.sum().item() == N
        assert torch.equal(ti[tm].sort().values, torch.arange(N))
        scr = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        assert torch.equal(scr[ti][tm][torch.argsort(ti[tm])].squeeze(1).long(), torch.arange(N))
        for r, sz in zip(rot, [P, P // 2, P // 4]):
            assert r.shape[0] == sz and torch.equal(r.sort().values, torch.arange(sz))
    ti, tm, rot = build_tree(torch.rand(500, 2), strides=[1], ball_sizes=[32, 32], rotate=45)
    assert torch.equal(ti[tm].sort().values, torch.arange(500))
    ei = knn_edges(torch.rand(50, 3), 4)
    assert ei.shape == (2, 200) and (ei[1].bincount(minlength=50) == 4).all()
    print("gate balltree_contract: PASS")


def test_restore_index_contract():
    torch.manual_seed(11)
    for N, dim in ((1000, 2), (1024, 2), (777, 2), (513, 2), (257, 3), (999, 3)):
        pos = torch.rand(N, dim)
        ti, tm, rot = build_tree(pos, strides=[2, 2], ball_sizes=[64, 64, 64], rotate=45)
        ri = restore_index(ti, tm)
        P = int(ti.shape[0])
        assert ri.dtype == torch.int64 and ri.shape == (N,)
        assert int(ri.min()) >= 0 and int(ri.max()) < P
        assert torch.equal(ri.sort().values, torch.nonzero(tm).squeeze(1))
        assert torch.equal(ti[ri], torch.arange(N))
        x = torch.randn(P, 5, dtype=torch.float64)
        assert torch.equal(x[tm][torch.argsort(ti[tm])], x.index_select(0, ri)), (N, dim)
    print("gate restore_index_contract: PASS")


def test_mpnn_grouped_mean():
    torch.manual_seed(17)
    for n, k, dim, c in ((1000, 8, 2, 16), (777, 8, 3, 8), (256, 4, 2, 5), (32, 16, 3, 3)):
        ei = knn_edges(torch.rand(n, dim), k)
        col = ei[1]
        assert ei.shape == (2, n * k)
        assert torch.equal(col, torch.arange(n).repeat_interleave(k))
        src = torch.randn(n * k, c, dtype=torch.float64, requires_grad=True)
        ref, new = scatter_mean(src, col, n), grouped_mean(src, n)
        assert (ref - new).abs().max().item() < 1e-13, (n, k)
        gout = torch.randn_like(ref)
        g_ref = torch.autograd.grad(ref, src, grad_outputs=gout, retain_graph=True)[0]
        g_new = torch.autograd.grad(new, src, grad_outputs=gout)[0]
        assert (g_ref - g_new).abs().max().item() < 1e-13, (n, k)
    print("gate mpnn_grouped_mean: PASS")


def test_tree_threading_bitwise():
    torch.manual_seed(23)
    for n, dim, strides, balls in ((3000, 2, [1], [256, 256]), (5000, 3, [2, 2], [64, 64, 64]),
                                   (777, 2, [2], [128, 128])):
        pos = torch.rand(n, dim)
        a = build_tree(pos, strides, balls, 45, workers=1)
        b = build_tree(pos, strides, balls, 45, workers=16)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and len(a[2]) == len(b[2])
        for ra, rb in zip(a[2], b[2]):
            assert torch.equal(ra, rb)
        assert torch.equal(knn_edges(pos, 8, workers=1), knn_edges(pos, 8, workers=16))
    print("gate tree_threading_bitwise: PASS")


if __name__ == "__main__":
    test_abupt_kv_cache()
    test_transolver_chunks()
    test_spider_rewrite()
    test_balltree_contract()
    test_restore_index_contract()
    test_mpnn_grouped_mean()
    test_tree_threading_bitwise()
    print("ALL GATES PASS")

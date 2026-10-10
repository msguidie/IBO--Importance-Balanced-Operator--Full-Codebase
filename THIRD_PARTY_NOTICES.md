# Third-party notices

The baselines of the paper are ports of their authors' public code. Each
file below states its source and modifications in its header; the licence
of the source applies to that file.

| File | Source | Licence | Modifications |
|---|---|---|---|
| `ibo/models/transolver.py` | Transolver-3, https://github.com/thuml/Transolver (Zhou et al., 2026) | MIT | einops removed; one shared backbone with volume and surface heads; static-shape chunked inference with the pass-1 slice weights reused |
| `ibo/models/spider.py` | SpiderSolver, https://github.com/Kai-Qi/SpiderSolver (Qi et al., 2025) | MIT | compile-safe attention (segment sums, scaled dot-product attention, static row layout); surface tokens per region (3-D) or one mean token (2-D) |
| `ibo/models/abupt.py` | AB-UPT, https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers (Alkin et al., 2025) | Emmi AI Non-Production License (ENPL), `LICENSES/ENPL-AB-UPT.txt` | geometry branch not ported; einops removed; rotary embedding in real arithmetic; the coordinate affine moved into the input stem; a case-condition MLP added to the stem for the 2-D benchmarks; anchor keys and values cached for full-field inference. This file is a modified version of the licensed work (ENPL, section 3). |
| `ibo/models/erwin/erwin.py`, `ibo/models/erwin/balltree.py` | Erwin, https://github.com/maxxxzdn/erwin (Zhdanov et al., 2025) | no licence file published upstream at the time of porting | ball tree rebuilt in numpy / torch in place of the `balltree-erwin` extension; a static restore index replaces the dynamic-shape unpadding; scatter-free grouped mean; separate volume and surface heads |
| `ibo/models/gaot/` | GAOT, https://github.com/camlab-ethz/GAOT (Wen et al., 2025) | no licence file published upstream at the time of porting | torch_scatter, torch_cluster, rotary_embedding_torch and omegaconf removed; optional precomputed per-edge attention (a pure function of the fixed coordinates); edits marked `[PORT]` |
| `ibo/datasets/airfrans/metrics.py` (`reorganize`, `WallShearStress`, `Compute_coefficients`) | AirfRANS, https://github.com/Extrality/AirfRANS (Bonnet et al., 2022) | MIT | the lift and drag integration on the predicted fields, trimmed to the coefficients |

The ENPL permits research, testing and evaluation and does not permit
production use of the AB-UPT port or of weights trained with it; see the
licence text. Erwin and GAOT are used with attribution; their authors have
been asked to state a licence (issues on the repositories above).

The training optimizer `ibo/training/lion.py` implements Lion (Chen et al.,
2023). Everything else in this repository is original work under the MIT
License of `LICENSE`.

"""Configuration: the single source of every tunable.

Dataclass defaults are the UrbanWind x IBO setting of the paper; every
other (dataset, model) pair is a yaml under configs/<dataset>/<model>.yaml
holding only the keys that differ. Precedence, low to high: dataclass
defaults < configs/<dataset>/<model>.yaml (or --config) < the `subsets:`
block for the selected subset < --set a.b=c < named CLI flags. The final
config is written to <run_dir>/config_used.yaml and evaluation rebuilds the
model from that file alone.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

DATASETS = ("airfrans", "tandemfoilset", "drivaerml", "urbanwind")
MODELS = ("ibo", "abupt", "transolver", "spider", "erwin", "gaot")
LAYOUTS = ("importance", "equal", "grid")
TANDEM_SUBSETS = ("cruise_aoa0", "cruise_aoa5", "cruise_random", "racecar",
                  "takeoff")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tuple(v, cast):
    return None if v is None else tuple(cast(x) for x in v)


@dataclass
class RunConfig:
    dataset: str = "urbanwind"
    model: str = "ibo"
    subset: str | None = None          # tandemfoilset only
    layout: str = "importance"         # importance | equal | grid
    seed: int = 42                     # training seed (utils.seed.BASE_SEED = paper)
    world_size: int = 0                # detected at launch
    effective_batch: int = 0           # resolved at launch (see training)
    grad_accum: int = 1
    # keys adopted from the cache's calibration.json at launch (2-D datasets
    # calibrate RoPE domains and Fourier scales per subset), for the record
    calibration: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    data_root: str | None = None       # raw data; default data/<dataset>
    cache_root: str | None = None      # preprocessed cache; default cache/<dataset>
    run_dir: str | None = None         # default runs/<dataset>_<model>[_<subset>]_s<seed>

    def __post_init__(self):
        for k in ("data_root", "cache_root", "run_dir"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, str(Path(os.path.expanduser(str(v)))))


@dataclass
class IBOConfig:
    """The IBO model. Switch defaults are the UrbanWind values; the 2-D and
    DrivAerML configs set their own."""
    # latent layout (baked into the cache)
    n_tokens: int = 16384              # L, a power of two
    n_slots: int = 32                  # members per token seen by the slot encoder
    cache_knn: int = 16                # neighbours stored per query point
    knn: int = 8                       # neighbours the decoder reads (<= cache_knn)
    # processor
    latent_dim: int = 384
    head_dim: int = 32                 # heads = latent_dim // head_dim
    n_blocks: int = 12
    n_register: int = 0
    long_skips: bool = False           # gated skips between mirrored blocks
    dropout: float = 0.1
    # conditioning on the case vector (dataset spec: 5 morphology scalars,
    # 2 inflow scalars, or none)
    adaln: str = "trunk"               # none | trunk (shared MLP + per-block head) | zero (per-block zero-init linear)
    adaln_hidden: int = 64
    encoder_film: bool = False         # FiLM on the token projection
    decoder_cond_concat: bool = False  # cond appended to the query self-embedding
    # slot encoder
    pn_hidden: int = 64
    pn_out: int = 256
    pn_act: str = "relu"               # relu | silu
    pn_fp32_first: bool = False        # first slot linear in fp32
    token_proj_layers: int = 1
    # linear initialisation of the token projection and the processor:
    # "default" (PyTorch's) or "trunc_normal" (N(0, 0.02), zero bias)
    init: str = "default"
    # decoder
    decoder_dh: int = 192
    decoder_heads: int = 6
    decoder_stages: int = 1            # 1: processor tokens; 2: encoder tokens then processor tokens
    decoder_register_attn: bool = False
    head_hidden: int = 256
    fourier_sdf_scale: float = 1.0
    fourier_sdf_clip: float | None = None
    idw_lambda_clip: float | None = None
    idw_log_ratio_clip: float | None = None
    # rotary embeddings per axis; domain None = positions already normalised
    rope_dims: tuple[int, ...] = (12, 12, 8)
    rope_base: tuple[float, ...] = (450.0, 450.0, 64.0)
    rope_scales: tuple[float, ...] = (500.0, 500.0, 500.0)
    rope_domain_lo: tuple[float, ...] | None = None
    rope_domain_hi: tuple[float, ...] | None = None

    def __post_init__(self):
        self.rope_dims = _tuple(self.rope_dims, int)
        self.rope_base = _tuple(self.rope_base, float)
        self.rope_scales = _tuple(self.rope_scales, float)
        self.rope_domain_lo = _tuple(self.rope_domain_lo, float)
        self.rope_domain_hi = _tuple(self.rope_domain_hi, float)

    @property
    def num_heads(self) -> int:
        return self.latent_dim // self.head_dim

    @property
    def ffn_hidden(self) -> int:
        return 4 * self.latent_dim

    @property
    def n_levels(self) -> int:
        return int(round(math.log2(self.n_tokens)))

    def validate(self):
        assert self.latent_dim % self.head_dim == 0
        assert sum(self.rope_dims) == self.head_dim, \
            f"rope_dims {self.rope_dims} must sum to head_dim {self.head_dim}"
        assert all(d % 2 == 0 for d in self.rope_dims)
        assert len(self.rope_base) == len(self.rope_dims) == len(self.rope_scales)
        assert self.decoder_dh % self.decoder_heads == 0
        assert 0 < self.knn <= self.cache_knn
        assert 2 ** self.n_levels == self.n_tokens, "n_tokens must be a power of 2"
        assert self.adaln in {"none", "trunk", "zero"}
        assert self.decoder_stages in {1, 2}
        assert self.token_proj_layers in {1, 2}
        assert self.pn_act in {"relu", "silu"}
        assert self.init in {"default", "trunc_normal"}


@dataclass
class AbuptConfig:
    """AB-UPT without the geometry branch: shared trunk + surface and
    volume branches. Lion, 5% warmup, cosine to an absolute floor."""
    hidden_dim: int = 192
    num_heads: int = 3
    shared_blocks: str = "sscscs"      # s = split attention, c = cross attention
    num_surface_blocks: int = 6
    num_volume_blocks: int = 6
    vol_anchors: int = 16384
    surf_anchors: int = 16384
    # fixed affine to AB-UPT's non-negative coordinates
    pos_offset: tuple[float, ...] = (1.0, 1.0, 0.0)
    pos_scale: float = 500.0
    infer_chunk: int = 65536           # query rows per branch per decode chunk
    infer_draws: int = 5               # independent anchor draws per test case
    lr: float = 5e-5
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    end_lr: float = 1e-6

    def __post_init__(self):
        self.pos_offset = _tuple(self.pos_offset, float)

    def validate(self):
        assert self.hidden_dim % self.num_heads == 0
        assert set(self.shared_blocks) <= {"s", "c"}


@dataclass
class TransolverConfig:
    """Transolver-3 (chunked physics attention), one backbone with volume
    and surface heads."""
    n_layers: int = 16
    n_hidden: int = 256
    n_heads: int = 8
    slice_num: int = 64
    mlp_ratio: int = 2
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.01
    schedule: str = "cosine_epoch"     # cosine_epoch (AdamW, per epoch) | onecycle (Adam, per step)
    min_lr: float = 1e-5
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    infer_chunk: int = 128000

    def validate(self):
        assert self.n_hidden % self.n_heads == 0
        assert self.schedule in {"cosine_epoch", "onecycle"}


@dataclass
class SpiderConfig:
    """SpiderSolver. Surface tokens: one per region (3-D, regions by normal
    direction) or clusters per body (2-D, template clustering)."""
    n_layers: int = 8
    n_hidden: int = 256
    n_heads: int = 8
    mlp_ratio: int = 2
    dropout: float = 0.0
    onion_num: int = 10
    # surface tokens: one per surface region (3-D, regions by normal
    # direction) or one mean token (2-D, clusters from a template)
    surf_token_mode: str = "per_region"   # per_region | single
    n_clusters: int = 6                # surface regions (3-D) or clusters over all bodies (2-D)
    n_inner: int = 500                 # innermost volume participants of the coarse attention
    surf_quota: int = 2048             # surface rows per training sample
    lr: float = 1e-3
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1000.0
    infer_chunk: int = 128000
    derive_workers: int | None = None

    def validate(self):
        assert self.n_hidden % self.n_heads == 0
        assert self.surf_token_mode in {"per_region", "single"}


@dataclass
class ErwinConfig:
    """Erwin ball-tree transformer; trees and kNN graphs built on the CPU in
    the loader."""
    c_in: int = 64
    c_hidden: tuple[int, ...] = (64, 128, 256, 512)
    ball_sizes: tuple[int, ...] = (256, 256, 256, 256)
    enc_num_heads: tuple[int, ...] = (2, 4, 8, 16)
    enc_depths: tuple[int, ...] = (2, 2, 6, 2)
    dec_num_heads: tuple[int, ...] = (2, 4, 8)
    dec_depths: tuple[int, ...] = (2, 2, 2)
    strides: tuple[int, ...] = (2, 2, 2)
    rotate: int = 45
    mp_steps: int = 3
    knn: int = 8
    mlp_ratio: int = 4
    rff_sigma: float = 1.0
    dimensionality: int = 3            # set from the dataset at launch
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.0
    end_lr: float = 1e-6
    infer_chunk: int = 128000

    def __post_init__(self):
        for f in ("c_hidden", "ball_sizes", "enc_num_heads", "enc_depths",
                  "dec_num_heads", "dec_depths", "strides"):
            setattr(self, f, _tuple(getattr(self, f), int))

    def validate(self):
        assert len(self.enc_num_heads) == len(self.enc_depths) == \
            len(self.ball_sizes) == len(self.c_hidden)
        assert len(self.dec_num_heads) == len(self.dec_depths) == \
            len(self.strides) == len(self.c_hidden) - 1


@dataclass
class GaotConfig:
    """GAOT (MAGNO encoder/decoder around a ViT), its published optimizer
    and three-phase per-epoch schedule."""
    latent_size: int = 384
    grid_h: int = 64
    grid_w: int = 64
    patch_size: int = 1
    vit_hidden: int = 384
    vit_layers: int = 5
    num_heads: int = 6
    num_kv_heads: int = 6
    ffn_multiplier: int = 4
    atten_dropout: float = 0.0
    positional_embedding: str = "absolute"
    use_long_range_skip: bool = True
    kernel_hidden: int = 64
    kernel_layers: int = 3
    radius_factor: float = 1.05
    lr: float = 8e-4
    weight_decay: float = 1e-5
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    final_lr: float = 5e-5
    warmup_fraction: float = 0.02
    cosine_fraction: float = 0.90

    def validate(self):
        assert self.vit_hidden % self.num_heads == 0


@dataclass
class TrainingConfig:
    epochs: int = 400
    cases_per_gpu: int = 1
    # the paper's effective batch; kept through gradient accumulation when
    # world_size * cases_per_gpu divides it, else the run uses
    # world_size * cases_per_gpu and records it in run.effective_batch
    effective_batch: int = 4
    n_query: int = 128000              # query points per case and step
    # IBO's optimizer (Lion); baselines carry their own numbers above
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    warmup_fraction: float = 0.05
    warmup_steps: int | None = 8000    # wins over warmup_fraction when set
    min_lr_fraction: float = 0.10
    max_grad_norm: float = 1.0
    loss_vol_weights: tuple[float, ...] | None = (2.0, 1.0, 2.0, 1.0, 1.0)
    loss_reduction: str = "group"      # group | flat (see training/losses.py)
    swa_fraction: float = 0.25         # average the last quarter of the epochs
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    activation_checkpointing: bool = False
    num_workers: int = 8
    prefetch_factor: int = 4
    rolling_keep: int = 3              # resumable epochs kept on disk

    def __post_init__(self):
        self.betas = _tuple(self.betas, float)
        self.loss_vol_weights = _tuple(self.loss_vol_weights, float)

    @property
    def swa_start(self) -> int:
        """First epoch of the averaging window: the last ceil(swa_fraction x
        epochs) epochs (100 of 400, 100 of 398); never before epoch 1."""
        return max(1, self.epochs - math.ceil(self.swa_fraction * self.epochs) + 1)

    def validate(self):
        assert self.epochs > 0 and self.cases_per_gpu > 0 and self.n_query > 0
        assert self.lr > 0 and self.weight_decay >= 0
        assert 0 < self.swa_fraction <= 1
        assert self.loss_reduction in {"group", "flat"}


@dataclass
class EvaluationConfig:
    decode_chunk: int = 262144         # query rows per decode chunk (padded)
    split: str = "test"
    save_predictions: bool = False
    latency: bool = False
    flops: bool = False


@dataclass
class UrbanWindPreprocess:
    n_workers: int = 32
    norm_stats: str | None = None      # default datasets/urbanwind/norm_stats.json
    coord_divisor: float = 550.0       # positions are divided by 550 m
    # building footprints from the CFD surface points
    dbscan_eps: float = 5.0            # metres
    dbscan_min_samples: int = 10
    wall_nz_threshold: float = 0.3     # |n_z| below this -> wall point
    concave_hull_ratio: float = 0.3
    dp_tol: float = 0.5                # Douglas-Peucker tolerance, metres
    min_edge_collapse: float = 1.0     # metres
    wind_direction: tuple[float, float] = (0.0, -1.0)
    wind_face_dot_threshold: float = 0.3
    threshold_edge: float = 0.5        # metres
    near_building_sdf_threshold: float = 10.0
    near_ground_z_threshold: float = 4.0
    invsqrt_floor: float = 2.0         # metres: w = max(s, floor)^(-1/2)
    grid_dims: tuple[int, int, int] = (32, 32, 16)
    pn_surf_first: bool = True         # surface points take slots first
    support_max: float = 550.0         # ray support cap, metres
    geom_chunk: int = 16384
    anomaly_concordance_threshold: float = 5.0
    anomaly_extreme_threshold: float = 8.0
    sdf_qc: bool = True                # log 2.5D vs raw distance agreement

    def __post_init__(self):
        self.wind_direction = _tuple(self.wind_direction, float)
        self.grid_dims = _tuple(self.grid_dims, int)


@dataclass
class TandemPreprocess:
    n_workers: int = 32
    n_leaves: int = 4096               # latent tokens of the cache (= ibo.n_tokens)
    idw_k: int = 8                     # nearest tokens stored per point (= ibo.cache_knn)
    cp_gate: float = 1.10              # drop cases with max Cp above this ...
    cp_gate_min_re: float = 1e4        # ... when Re is at least this
    did_bins: int = 8
    did_dmax: float = 5.0              # chord lengths
    curvature_k: int = 8
    coord_pctl: tuple[float, float] = (5.0, 95.0)
    coord_pctl_sample: int = 50000
    calibration_sample_cases: int = 48
    grid_dims: tuple[int, int] = (64, 64)

    def __post_init__(self):
        self.coord_pctl = _tuple(self.coord_pctl, float)
        self.grid_dims = _tuple(self.grid_dims, int)


@dataclass
class AirfransPreprocess:
    n_workers: int = 32
    n_leaves: int = 4096
    slots: int = 25                    # members per token (= ibo.n_slots)
    idw_k: int = 8
    worker_threads: int = 1
    val_seed: int = 42                 # 80 validation cases drawn from the official 800
    n_val: int = 80
    sdf_chunk: int = 20000
    did_bins: int = 8
    did_dmax: float = 2.0              # chord lengths
    curvature_k: int = 8
    coord_pctl: tuple[float, float] = (5.0, 95.0)
    coord_pctl_sample: int = 50000
    calibration_sample_cases: int = 48
    grid_dims: tuple[int, int] = (64, 64)
    grid_domain_lo: tuple[float, float] = (-2.0, -1.5)
    grid_domain_hi: tuple[float, float] = (4.0, 1.5)

    def __post_init__(self):
        self.coord_pctl = _tuple(self.coord_pctl, float)
        self.grid_dims = _tuple(self.grid_dims, int)
        self.grid_domain_lo = _tuple(self.grid_domain_lo, float)
        self.grid_domain_hi = _tuple(self.grid_domain_hi, float)


@dataclass
class DrivaermlPreprocess:
    n_workers: int = 30
    n_leaves: int = 32768
    slots: int = 128
    idw_k: int = 16
    worker_threads: int = 4
    crop_min: tuple[float, float, float] = (-30.0, -8.0, -0.5)
    crop_max: tuple[float, float, float] = (60.0, 8.0, 8.0)
    coord_scale: float = 10.0          # metres / 10
    sdf_chunk: int = 262144
    cdcl_tolerance: float = 5e-3       # normal orientation check vs reference Cd/Cl
    train_surface_cap: int = 5000000   # surface points in the training pool
    eval_volume_cap: int = 5000000     # frozen evaluation pool sizes ...
    eval_surface_cap: int | None = None  # ... None keeps every surface point (exact Cd/Cl)
    coord_pctl: tuple[float, float] = (5.0, 95.0)
    coord_pctl_sample: int = 200000
    grid_dims: tuple[int, int, int] = (32, 32, 32)

    def __post_init__(self):
        self.crop_min = _tuple(self.crop_min, float)
        self.crop_max = _tuple(self.crop_max, float)
        self.coord_pctl = _tuple(self.coord_pctl, float)
        self.grid_dims = _tuple(self.grid_dims, int)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ibo: IBOConfig = field(default_factory=IBOConfig)
    abupt: AbuptConfig = field(default_factory=AbuptConfig)
    transolver: TransolverConfig = field(default_factory=TransolverConfig)
    spider: SpiderConfig = field(default_factory=SpiderConfig)
    erwin: ErwinConfig = field(default_factory=ErwinConfig)
    gaot: GaotConfig = field(default_factory=GaotConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    urbanwind: UrbanWindPreprocess = field(default_factory=UrbanWindPreprocess)
    tandemfoilset: TandemPreprocess = field(default_factory=TandemPreprocess)
    airfrans: AirfransPreprocess = field(default_factory=AirfransPreprocess)
    drivaerml: DrivaermlPreprocess = field(default_factory=DrivaermlPreprocess)
    # per-subset overrides {subset: {"section.key": value}}, applied at load
    subsets: dict = field(default_factory=dict)

    @property
    def preprocess(self):
        return getattr(self, self.run.dataset)

    @property
    def model_section(self):
        return getattr(self, self.run.model)

    def validate(self):
        assert self.run.dataset in DATASETS, f"unknown dataset {self.run.dataset!r}"
        assert self.run.model in MODELS, f"unknown model {self.run.model!r}"
        assert self.run.layout in LAYOUTS, f"unknown layout {self.run.layout!r}"
        if self.run.dataset == "tandemfoilset":
            assert self.run.subset in TANDEM_SUBSETS, \
                f"tandemfoilset needs --subset, one of {TANDEM_SUBSETS}"
        for name in ("ibo", "abupt", "transolver", "spider", "erwin", "gaot",
                     "training"):
            getattr(self, name).validate()


_SECTIONS = {f.name: f.type for f in fields(Config) if f.name != "subsets"}


# ---------------------------------------------------------------------------
# loading / saving
# ---------------------------------------------------------------------------

def _section_cls(name: str):
    return {"run": RunConfig, "data": DataConfig, "ibo": IBOConfig,
            "abupt": AbuptConfig, "transolver": TransolverConfig,
            "spider": SpiderConfig, "erwin": ErwinConfig, "gaot": GaotConfig,
            "training": TrainingConfig, "evaluation": EvaluationConfig,
            "urbanwind": UrbanWindPreprocess, "tandemfoilset": TandemPreprocess,
            "airfrans": AirfransPreprocess, "drivaerml": DrivaermlPreprocess}[name]


def _to_dict(cfg: Config) -> dict:
    import dataclasses
    out = {}
    for name in _SECTIONS:
        d = dataclasses.asdict(getattr(cfg, name))
        out[name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
    out["subsets"] = copy.deepcopy(cfg.subsets)
    return out


def _from_dict(raw: dict, strict: bool = True) -> Config:
    kwargs = {}
    for name in _SECTIONS:
        cls = _section_cls(name)
        section = raw.get(name) or {}
        known = {k: v for k, v in section.items() if k in cls.__dataclass_fields__}
        unknown = sorted(set(section) - set(known))
        if unknown and strict:
            raise KeyError(f"unknown keys in config section '{name}': {unknown}")
        kwargs[name] = cls(**known)
    cfg = Config(**kwargs)
    cfg.subsets = copy.deepcopy(raw.get("subsets") or {})
    return cfg


def save_config(cfg: Config, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False)
    Path(tmp).replace(path)


def load_config_file(path: str | Path) -> Config:
    """A complete config written by save_config (config_used.yaml)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _from_dict(raw)
    cfg.validate()
    return cfg


def set_dotted(cfg: Config, key: str, value) -> None:
    """cfg.<section>.<field> = value; unknown sections or fields raise."""
    section, _, name = key.partition(".")
    if section == "subsets":
        raise KeyError("set the subsets block in the yaml, not with --set")
    if section not in _SECTIONS:
        raise KeyError(f"unknown config section {section!r} in {key!r}")
    obj = getattr(cfg, section)
    if name not in obj.__dataclass_fields__:
        raise KeyError(f"unknown key {key!r}")
    setattr(obj, name, value)
    obj.__post_init__() if hasattr(obj, "__post_init__") else None


def parse_set(item: str) -> tuple[str, object]:
    """'section.key=value' with the value parsed as yaml."""
    key, sep, value = item.partition("=")
    if not sep:
        raise SystemExit(f"--set expects section.key=value, got {item!r}")
    return key.strip(), yaml.safe_load(value)


def default_config_path(dataset: str, model: str) -> Path:
    return repo_root() / "configs" / dataset / f"{model}.yaml"


def load_config(dataset: str, model: str, subset: str | None = None,
                layout: str | None = None, config_path: str | Path | None = None,
                sets: list[str] | tuple[str, ...] = ()) -> Config:
    """The config of one (dataset, model[, subset]) launch."""
    path = Path(config_path) if config_path else default_config_path(dataset, model)
    if not path.exists():
        raise SystemExit(f"no config for dataset {dataset!r} x model {model!r}: "
                         f"{path} (see `python -m ibo list`)")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.setdefault("run", {})
    raw["run"]["dataset"] = dataset
    raw["run"]["model"] = model
    if subset is not None:
        raw["run"]["subset"] = subset
    if layout is not None:
        raw["run"]["layout"] = layout
    cfg = _from_dict(raw)
    if cfg.run.subset and cfg.subsets.get(cfg.run.subset):
        for k, v in cfg.subsets[cfg.run.subset].items():
            set_dotted(cfg, k, v)
    for item in sets:
        set_dotted(cfg, *parse_set(item))
    cfg.validate()
    return cfg


def resolve_paths(cfg: Config, subset_in_run_dir: bool = True) -> None:
    """Default paths. A launch over every TandemFoilSet subset keeps the
    subset out of the run tag and puts one directory per subset below it."""
    d, r = cfg.data, cfg.run
    if d.data_root is None:
        d.data_root = str(Path("data") / r.dataset)
    if d.cache_root is None:
        d.cache_root = str(Path("cache") / r.dataset)
    if d.run_dir is None:
        tag = f"{r.dataset}_{r.model}" + (f"_{r.subset}" if r.subset and subset_in_run_dir else "") \
            + ("" if r.layout == "importance" else f"_{r.layout}") + f"_s{r.seed}"
        d.run_dir = str(Path("runs") / tag)


def set_lr(cfg: Config, lr: float) -> None:
    """`--lr` applies to the selected model's own optimizer section."""
    if cfg.run.model == "ibo":
        cfg.training.lr = float(lr)
    else:
        cfg.model_section.lr = float(lr)


def resolve_effective_batch(cfg: Config, world: int) -> None:
    """Keep the paper's effective batch through gradient accumulation when
    the launch geometry divides it; otherwise run at world * cases_per_gpu
    and record that."""
    tc = cfg.training
    micro = world * tc.cases_per_gpu
    if tc.effective_batch % micro == 0:
        cfg.run.grad_accum = tc.effective_batch // micro
        cfg.run.effective_batch = tc.effective_batch
    else:
        cfg.run.grad_accum = 1
        cfg.run.effective_batch = micro
        print(f"[config] effective batch {tc.effective_batch} is not a multiple "
              f"of world {world} x cases_per_gpu {tc.cases_per_gpu}; running "
              f"at effective batch {micro}", flush=True)
    cfg.run.world_size = world

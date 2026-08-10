"""SegConfig — dataclass-over-YAML config for the segmentation harness (defaults everywhere, unknown
keys dropped). Self-contained (does not import em_ssl.config), mirroring
the repo style, but a superset of the encoder-evaluation ProbeConfig: nested neck / decoder / loss specs
selected by a string ``type`` + free-form ``params`` so the registries (models/{necks,decoders,
losses}.py) can add variants without schema churn.

Component selection is the repo idiom: a string field naming a variant, dispatched by a module-level
factory dict — no decorator registry framework. The frozen-encoder + feature-tap contract is reused
verbatim from em_ssl's checkpoint_index manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ..constants import CANONICAL_NM, DEFAULT_DERIVED_ROOT, IGNORE_INDEX


def _known(cls, raw: dict | None) -> dict:
    """Keep only keys that are declared fields of ``cls`` (drop unknowns, like the em_ssl loader)."""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in (raw or {}).items() if k in allowed}


# --------------------------------------------------------------------------- #
# Nested specs
# --------------------------------------------------------------------------- #
@dataclass
class EncoderSpec:
    """The frozen base encoder + how its multi-block feature taps are read."""

    run_dir: str | None = None          # encoder run dir holding checkpoint_index.json. None -> supplied at runtime / mock.
    checkpoint_step: int | None = None  # explicit teacher/encoder step; None -> latest.
    tile_size: int = 512                # /16 -> 32x32 tokens; matches the pretraining base crop.
    feature_layers: Any = "last4"       # "last4" | "last1" | explicit list of block indices.
    apply_encoder_norm: bool = True     # apply the encoder's final LayerNorm per selected tap.
    # encoder-adaptation hook. The LoRA modes install Conv-LoRA adapters inside the encoder blocks and
    # leave the base weights frozen; last_n/full instead unfreeze base weights.
    adapt: str = "frozen"               # frozen | lora | lora_ln | cond_lora | last_n | full  (hooks/encoder_adaptation.py)
    adapt_params: dict = field(default_factory=dict)

    def resolved_layers(self, depth: int) -> list[int]:
        """Concrete 0-based block indices for this arch depth."""
        fl = self.feature_layers
        if isinstance(fl, (list, tuple)):
            return [int(i) for i in fl]
        if fl == "last1":
            return [depth - 1]
        if fl == "last4":
            return [depth - 4, depth - 3, depth - 2, depth - 1]
        raise ValueError(f"feature_layers must be 'last1', 'last4', or a list; got {fl!r}")

    @classmethod
    def from_dict(cls, d: dict | None) -> "EncoderSpec":
        return cls(**_known(cls, d))


@dataclass
class NeckSpec:
    """Per-patch tokens -> spatial multi-scale features (the neck ablation variable)."""

    type: str = "naive_1x1"  # naive_1x1 | resnet34_detail
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "NeckSpec":
        return cls(**_known(cls, d))


@dataclass
class DecoderSpec:
    """The mask head (the decoder experiment variable). See models/decoders.py for the registry keys."""

    # dense: upernet | dpt | nnunet_convnext_unet | pspnet | deeplabv3plus | unet
    # instance: panoptic_deeplab | affinity_mws ; query: mask2former_query_hf
    type: str = "upernet"
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "DecoderSpec":
        return cls(**_known(cls, d))


@dataclass
class LossTerm:
    """One additive term of the training loss (the loss-function experiment builds these up cumulatively).

    ``affinity`` and ``panoptic_instance`` are registered terms as well; ``build_loss`` appends the
    matching one automatically when an instance decoder is configured without it.
    """

    type: str = "dice_bce"  # dice_bce | dice_focal | cldice | skeleton_recall | orientation_centerline_aux
    weight: float = 1.0
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LossTerm":
        return cls(**_known(cls, d))


@dataclass
class LossSpec:
    terms: list[LossTerm] = field(default_factory=lambda: [LossTerm()])
    ignore_index: int = IGNORE_INDEX

    @classmethod
    def from_dict(cls, d: dict | None) -> "LossSpec":
        d = d or {}
        terms = [LossTerm.from_dict(t) for t in d.get("terms", [])] or [LossTerm()]
        return cls(terms=terms, ignore_index=int(d.get("ignore_index", IGNORE_INDEX)))


@dataclass
class DataSpec:
    """The canonical-scale derived dataset + the augmentation recipe (fixed across arms)."""

    data_root: str = DEFAULT_DERIVED_ROOT
    organelle: str = "er"                 # er | mito  (one head per run, per the neck experiment's per-organelle design)
    canonical_nm: float | None = None     # target nm/px; None -> CANONICAL_NM[organelle].
    group: str | None = None              # e.g. "group2_er"; None -> f"group2_{organelle}".
    bucket: str = "canonical"             # resolution view: canonical (per-organelle canonical nm/px) | native (source resolution, unresampled). Built by build_dataset --scale-mode.
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_classes: int = 2                  # background + organelle (binary-per-organelle head).
    task: str = "semantic"                # semantic | instance (mito instance metrics vs ER dense).
    min_fg_frac_keep: float = 0.10        # train crops must overlap >= this labelled fraction.
    label_fractions: list[float] = field(default_factory=lambda: [1.0])  # label-efficiency hook (adjacent to the input-scale experiment).
    # Diversity-vs-volume gate: subset the train pool along two independent axes.
    subset_frac: float = 1.0              # volume: fraction of crops kept per source (stratified by dataset).
    source_frac: float = 1.0              # diversity: fraction of distinct sources (datasets) kept.
    subset_seed: int = 0                  # seed for both train-pool subsets (nested + reproducible).
    manifest_name: str = "manifest.jsonl"  # override to a rebalanced manifest (e.g. held-out-image).
    # LOSO-CV: exclude these training sources from the train pool and score them as a held-out 'loso' fold
    # -> leave-one-source-out cross-source generalisation.
    holdout_sources: list = field(default_factory=list)
    # augmentation (fixed recipe; identical across arms — knobs exist only to disable for an ablation).
    aug_flip: bool = True
    aug_rot90: bool = True
    aug_elastic: bool = True
    aug_elastic_alpha: float = 20.0       # displacement magnitude (px, at canonical scale).
    aug_elastic_sigma: float = 6.0        # smoothing of the displacement field (px).
    aug_brightness: float = 0.15          # image-only brightness jitter (+/-).
    aug_contrast: float = 0.15            # image-only contrast jitter (+/-).
    aug_gamma: float = 0.20               # image-only gamma jitter (+/-).
    aug_noise_std: float = 0.02           # image-only additive Gaussian noise std (in normalised units).

    def resolved_canonical_nm(self) -> float:
        return float(self.canonical_nm) if self.canonical_nm is not None else float(CANONICAL_NM[self.organelle])

    def resolved_group(self) -> str:
        return self.group or f"group2_{self.organelle}"

    @classmethod
    def from_dict(cls, d: dict | None) -> "DataSpec":
        return cls(**_known(cls, d))


@dataclass
class OptimSpec:
    max_steps: int = 8000
    warmup_steps: int = 500
    batch_size: int = 8
    lr: float = 1e-3                      # decoder / neck base LR.
    weight_decay: float = 1e-4
    decoder_lr_mult: float = 1.0          # multiply lr for the decoder head (query decoders may want < 1).
    adapter_lr: float = 1e-3             # LR for every trainable encoder param (LoRA adapters, unfrozen blocks).
    grad_checkpoint: bool = False         # gradient-checkpoint heavy necks/decoders to save memory.
    grad_clip: float = 0.0                # 0 disables.
    grad_accum: int = 1                   # micro-batches per optimizer step; effective batch = batch_size * grad_accum.
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "OptimSpec":
        return cls(**_known(cls, d))


@dataclass
class ConditioningSpec:
    """Image-style conditioning. Disabled by default -> byte-identical to the base arm (no conditioning).

    Arms (``arm`` is documentation; the individual flags below drive behaviour):
      * baseline — no conditioning (``enabled=False``).
      * inferred style — image-style FiLM from the image itself, tile or dataset scope.
      * MixStyle / DSU — feature-statistic mixing (``mixstyle!='off'``, ``film=False``); no test-time code.

    The conditioning code ``s in R^style_dim`` is produced from image appearance by the inferred style
    encoder, pooled over ``style_scope``, and consumed by FiLM / conditional-GroupNorm re-injected at
    every neck+decoder norm layer.
    """

    arm: str = "baseline"
    enabled: bool = False              # master switch; when False no conditioning module is built.

    # --- style-code source ---
    # inferred (style encoder) | confident_feature (pooled-global FiLM: a per-image
    # per-organelle appearance code pooled from the encoder features of confident organelle regions).
    style_source: str = "inferred"
    style_dim: int = 64                # d_s, the conditioning-code width.
    style_stats: bool = True           # append cheap low-level image statistics to the style encoder.
    style_from_features: bool = False  # also feed the (detached) coarsest encoder tap to the style encoder.
    style_hidden: int = 64             # style-encoder conv width.
    # non-spurious-style options
    grad_reversal: float = 0.0         # DANN lambda_max (0 disables); adversary predicts adv_targets from the code.
    adv_targets: list = field(default_factory=lambda: ["dataset"])  # what the reversed adversary predicts.
    adv_hidden: int = 128
    metadata_dropout: float = 0.0      # prob of dropping the code (-> zeros) per forward for graceful degradation.

    # --- conditioning mechanism ---
    film: bool = True                  # install FiLM (conditional-GroupNorm). MixStyle/DSU set film=False (pure MixStyle).
    film_scope: str = "per_block"      # per_block (re-inject at every norm) | once (single injection point).
    condition_neck: bool = True
    condition_decoder: bool = True

    # --- style-scope pooling ---
    style_scope: str = "tile"          # tile | source (multi-tile, within-batch by source) | dataset (per-source)
    n_prototypes: int = 1              # multi-prototype at tile scope (K modes); 1 = single code.


    # --- MixStyle / DSU ---
    mixstyle: str = "off"              # off | mixstyle | dsu
    mixstyle_p: float = 0.5
    mixstyle_alpha: float = 0.1
    mixstyle_mix: str = "random"       # random | crossdomain (source-aware permutation)
    mixstyle_points: str = "neck"      # neck | neck_decoder — which stages get the mixing hook.


    # style_source='confident_feature': the appearance code is pooled from the encoder features of
    # confident organelle regions (GT foreground at train; the model's own confident prediction at test).
    confident_thresh: float = 0.7      # first-pass FG-prob threshold selecting the confident seed regions.

    # --- test-time adaptation — read by harness/tta.py, not the trainer ---
    tta: str = "off"    # off | b2_support | support | b4_b2film
    tta_steps: int = 1
    tta_lr: float = 1e-3
    tta_debias_svd: int = 64           # positional-subspace rank projected out before matching.
    # --- Support-prototype family (b2_support / support): 3 orthogonal axes ---
    # Axis 1 — seed quality:
    support_source: str = "inferred"   # inferred(_raw) | gt (oracle/ceiling) | inferred_gated | interactive | few_shot
    support_conf: float = 0.9          # inferred_gated: hard high-confidence threshold for accepting a seed.
    support_min_size: int = 32         # inferred_gated: min connected-component size (px) — reject specks.
    support_open: int = 1              # inferred_gated: morphological-opening iterations (spatial coherence).
    interactive_clicks: int = 3        # interactive support: # simulated user clicks/scribbles sampled from GT.
    interactive_radius: int = 6        # click -> disk-scribble radius (px).
    # Few-shot verified-instance ingestion: k user-verified GT instances as seeds.
    n_shots: int = 0                   # support_source='few_shot': # GT instances (connected components) accepted.
    # Seed-quality -> Dice curve: degrade the (GT) seed set in controlled precision/recall steps.
    seed_drop_frac: float = 0.0        # drop this fraction of true seed pixels (lowers seed recall).
    seed_false_frac: float = 0.0       # inject false seed pixels = this fraction of #true (lowers seed precision).
    # Axis 2 — combination mode (how support combines with the head):
    support_combine: str = "replace"   # replace(neg-ref) | residual | uncertainty_gated(priority) | early(FiLM)
    support_alpha: float = 0.5         # residual: blend strength toward support where head+support disagree.
    support_uncertain_margin: float = 0.2  # uncertainty_gated: |P(fg)-0.5| < margin => head unconfident zone.
    # Uncertainty-gated FiLM: global FiLM conditioning is sensitive to a poorly estimated code, which
    # mis-conditions every pixel. film_gate applies the conditioned pass where the base head is
    # unconfident, so a poor code cannot override the confident base head everywhere.
    film_gate: bool = False            # uncertainty-gate the FiLM-conditioned pass against the base pass.
    # Axis 3 — prototype count / aggregation (scale with cleanliness):
    n_support: int = 1                 # 1 = pooled prototype; K = K appearance-mode prototypes (cluster).
    proto_gate: bool = False           # per-prototype confidence-gating (drop noisy modes; only add K when clean).
    proto_min_frac: float = 0.05       # proto_gate: drop a mode whose cluster is < this fraction of seed pixels.
    support_seed: int = 0              # RNG seed for sampled seeds / interactive clicks.
    tta_passes: int = 2                # T recurrent passes (re-estimate the code from pass k -> pass k+1).
    tta_field: int = 1                 # spatial-field multiplier for the estimate.
    bank_key: str = "style"            # style | featstat — key the memory bank is indexed by.
    bank_granularity: str = "source"   # source | image — one bank entry per source or per training image.
    bank_k: int = 3                    # nearest-neighbour count.
    fda_beta: float = 0.01             # FDA low-frequency band fraction to swap toward the training reference.

    @classmethod
    def from_dict(cls, d: dict | None) -> "ConditioningSpec":
        return cls(**_known(cls, d))


@dataclass
class EvalSpec:
    """Sliding-window eval over the full annotated region (metrics, annotation-masked)."""

    overlap: float = 0.25
    boundary_theta_frac: float = 0.0075   # Boundary-F1 tolerance (fraction of diagonal).
    boundary_dilation_ratio: float = 0.02  # Boundary-IoU band width (fraction of diagonal).
    fg_threshold: float = 0.5
    auprc_bins: int = 256
    hd95_pct: float = 95.0
    instance_min_size: int = 16           # mito instance post-proc min object size (px).
    max_region_px: int = 0                # >0: central-crop each test region to <= this many px before
                                          # sliding-window eval, bounding cost on pathologically-huge
                                          # crops (uniform across arms -> fair ranking). 0 = full region.
    max_eval_crops: int = 0               # >0: stratified-by-dataset subsample of each eval split to
                                          # ~this many crops (bounds CPU-heavy eval for fast sweeps;
                                          # uniform across arms -> fair relative ranking). 0 = all crops.
    bootstrap_n: int = 1000
    bootstrap_ci: float = 95.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "EvalSpec":
        return cls(**_known(cls, d))


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class SegConfig:
    name: str = "arm"
    notes: str = ""
    device: str = "cuda"                  # falls back to cpu if cuda unavailable.
    amp: bool = True                      # bf16 autocast on cuda.
    num_workers: int = 4

    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    neck: NeckSpec = field(default_factory=NeckSpec)
    decoder: DecoderSpec = field(default_factory=DecoderSpec)
    loss: LossSpec = field(default_factory=LossSpec)
    data: DataSpec = field(default_factory=DataSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    cond: ConditioningSpec = field(default_factory=ConditioningSpec)  # image-style conditioning.

    # provenance (populated by the loader).
    config_path: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, raw: dict | None) -> "SegConfig":
        raw = raw or {}
        top = _known(cls, raw)  # name/notes/device/amp/num_workers/config_path
        top.pop("encoder", None)  # nested handled explicitly below
        for nested in ("neck", "decoder", "loss", "data", "optim", "eval", "cond"):
            top.pop(nested, None)
        return cls(
            **top,
            encoder=EncoderSpec.from_dict(raw.get("encoder")),
            neck=NeckSpec.from_dict(raw.get("neck")),
            decoder=DecoderSpec.from_dict(raw.get("decoder")),
            loss=LossSpec.from_dict(raw.get("loss")),
            data=DataSpec.from_dict(raw.get("data")),
            optim=OptimSpec.from_dict(raw.get("optim")),
            eval=EvalSpec.from_dict(raw.get("eval")),
            cond=ConditioningSpec.from_dict(raw.get("cond")),
        )


def load_seg_config(path: str | Path | None) -> SegConfig:
    """Load a SegConfig from YAML (or all-defaults if ``path`` is None). Unknown keys are dropped."""
    if path is None:
        return SegConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = SegConfig.from_dict(raw)
    cfg.config_path = str(path)
    return cfg

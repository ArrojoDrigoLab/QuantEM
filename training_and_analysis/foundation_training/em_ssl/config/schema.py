"""Normalized experiment-spec schema and YAML loader.

A single experiment definition, in this schema rather than DINOv3's, is the source of truth. The
runner translates it into DINOv3 configuration keys at launch.

Design notes:
  * Crop *schedule* is a list of stages so continuation runs (512 -> 768 -> 1024) are first
    class. Each stage has its own crop size and step budget, and is launched separately, as
    dependent warm-started runs (DINOv3 has no native mid-run resolution change).
  * `min_side` defaults to the largest global crop in the schedule; a parent tile too small for the
    crop is filtered out and logged rather than upsampled.
  * `mean`/`std` default to a computed tile_intensity_stats.json if present, else the EM
    corpus defaults (`EM_DEFAULT_MEAN`/`EM_DEFAULT_STD`); ImageNet statistics are never used.
  * Step-based budgets everywhere ("epoch" is arbitrary over a ~306k-tile manifest).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .. import EM_DEFAULT_MEAN, EM_DEFAULT_STD
from ..fino.factors import FinoFactorSpec, factor_from_dict

# --------------------------------------------------------------------------- #
# Sub-specs
# --------------------------------------------------------------------------- #
@dataclass
class CropStage:
    global_crops_size: int = 512
    local_crops_size: int = 112
    max_steps: int = 10000
    # Optional per-stage overrides used by continuation/high-res-adaptation stages.
    warmup_steps: int | None = None
    lr: float | None = None
    # Per-stage batch (DINOv3 multi-stage): each stage is a separate warm-started run, so a
    # low-res stage can use a big batch and a high-res stage a small one. None ⇒ train.batch_size_per_gpu.
    batch_size_per_gpu: int | None = None
    name: str | None = None

@dataclass
class ModelSpec:
    arch: str = "vit_base"  # dinov3 factory name: vit_base / vit_large / vit_huge2
    patch_size: int = 16
    in_chans: int = 1
    # Stochastic depth and register tokens, emitted into student{}. These defaults are the inherited
    # DINOv3 ssl_default values; set them explicitly per experiment. Adding registers changes the token
    # count, so a checkpoint warm-loaded into a different n_storage_tokens takes the shared weights and
    # initialises the register tokens fresh.
    drop_path_rate: float = 0.3
    n_storage_tokens: int = 0
    # RoPE coordinate augmentations, training-only, injected into student.pos_embed_rope_*: they randomize
    # the RoPE coordinate grid each step, hardening the encoder against position and resolution shift at no
    # compute cost. rescale_coords=2 is the stock DINOv3 pretraining setting; shift and jitter have no
    # upstream precedent and stay off unless an experiment enables them.
    rope_rescale_coords: float | None = 2.0   # isotropic positional zoom; log-uniform [1/x, x]
    rope_shift_coords: float | None = None    # translation
    rope_jitter_coords: float | None = None   # anisotropic per-axis scale

@dataclass
class CropsSpec:
    schedule: list[CropStage] = field(default_factory=lambda: [CropStage()])
    local_crops_number: int = 8
    global_crops_scale: tuple[float, float] = (0.32, 1.0)
    local_crops_scale: tuple[float, float] = (0.05, 0.32)
    gram_teacher_crops_size: int | None = None

@dataclass
class DataSpec:
    manifest: str | None = None
    exports_root: str | None = None
    tile_root: str | None = None
    shard_dir: str | None = None
    shard_prefix: str = "em_tiles_v0"
    stats_file: str | None = None  # tile_intensity_stats.json
    min_side: int | None = None  # default = max global crop in schedule
    mean: float | None = None
    std: float | None = None
    filter: dict[str, Any] = field(default_factory=dict)
    use_loose_files: bool = False  # debug only

@dataclass
class AugmentationSpec:
    # Mirrors transforms.EMAugmentationConfig; defaults are mild (membrane-preserving).
    brightness: float = 0.2
    contrast: float = 0.2
    color_jitter_p: float = 0.8
    gamma: float = 0.2
    gamma_p: float = 0.2
    noise_sigma_max: float = 0.04
    noise_p: float = 0.2
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.5
    blur_kernel_size: int = 9
    global1_blur_p: float = 1.0
    global2_blur_p: float = 0.1
    local_blur_p: float = 0.5
    dihedral: bool = True
    horizontal_flips: bool = True
    vertical_flips: bool = True
    rotations: bool = True
    teacher_no_color_jitter: bool = False
    # Native-resolution field of view, on by default. The stored 2048px tiles are ~4x larger than a 512
    # crop, so a plain random-resized crop would downsample every crop 2.3-4x while the downstream decoder
    # runs at native resolution. native_fov instead draws a per-crop downsample factor
    # M = 1 + (max-1)*u**bias (M=1 is 1:1 native), biased toward native by native_bias. Crop sizes are
    # absolute pixels (M*crop_size) clamped to each tile's real H/W, so small tiles cap at their native
    # size rather than being upscaled.
    native_fov: bool = True
    native_downsample_max: float = 4.0
    native_bias: float = 5.0
    native_local_downsample_max: float = 3.0
    native_overlap_room: float = 1.5

@dataclass
class OptimSpec:
    # Half stock DINOv3's 1e-3, for stability on high-contrast single-channel EM.
    lr: float = 5e-4
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    clip_grad: float = 3.0
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainSpec:
    batch_size_per_gpu: int = 64
    num_workers: int = 10
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    official_epoch_length: int = 1250  # DINOv3 schedule granularity
    compile: bool = True
    activation_checkpointing: bool = False
    param_dtype: str = "bf16"
    seed: int = 0

@dataclass
class LoggingSpec:
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str | None = None
    log_every: int = 20
    system_probe_every: int = 200
    viz_every: int = 2000  # augmentation grids

@dataclass
class CheckpointingSpec:
    period_steps: int = 2000
    keep_last: int = 3
    keep_every_steps: int = 20000
    teacher_export_every_steps: int = 2000

@dataclass
class FinoSpec:
    """Run-level FINO (metadata-guided) controls.

    Per-factor sign, weight and vocabulary live on ``ExperimentSpec.metadata_factors``; this block
    holds the run-wide knobs translated into the DINOv3 ``guide:`` block and ``optim``. An empty
    ``metadata_factors`` disables FINO, leaving a plain DINOv3 run. The gradient-reversal lambda
    ramps 0 to 1 over the run via ``lambda_schedule_type``.
    """

    lambda_schedule_type: str = "sigmoid"  # sigmoid | linear | constant
    lambda_warmup_steps: int = 0
    # Equalize gradient magnitudes between guide losses; only acts with two or more enabled factors.
    grad_norm_normalization: bool = False
    # Abort if a factor's valid fraction, from the coverage report, is below this.
    min_valid_fraction: float = 0.02
    allow_low_coverage: bool = False
    # Optional path to a fino_metadata_spec.json, for the coverage fingerprint.
    spec_file: str | None = None

# --------------------------------------------------------------------------- #
# Top-level spec
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentSpec:
    name: str = "unnamed"
    framework: str = "dinov3"
    notes: str = ""
    model: ModelSpec = field(default_factory=ModelSpec)
    crops: CropsSpec = field(default_factory=CropsSpec)
    data: DataSpec = field(default_factory=DataSpec)
    augmentation: AugmentationSpec = field(default_factory=AugmentationSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    checkpointing: CheckpointingSpec = field(default_factory=CheckpointingSpec)
    # FINO metadata-guided training (empty metadata_factors ⇒ plain baseline, no behaviour change).
    metadata_factors: list[FinoFactorSpec] = field(default_factory=list)
    fino: FinoSpec = field(default_factory=FinoSpec)
    # Raw DINOv3-schema overrides passed straight through (escape hatch for ibot/gram/etc.).
    dinov3: dict[str, Any] = field(default_factory=dict)
    # Provenance: original YAML path + raw dict.
    config_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # --- convenience ---
    @property
    def max_global_crop(self) -> int:
        return max(s.global_crops_size for s in self.crops.schedule)

    @property
    def total_steps(self) -> int:
        return sum(s.max_steps for s in self.crops.schedule)

    @property
    def effective_min_side(self) -> int:
        if self.data.min_side is not None:
            return int(self.data.min_side)
        return self.max_global_crop

    # --- FINO ---
    @property
    def enabled_factors(self) -> list[FinoFactorSpec]:
        return [f for f in self.metadata_factors if f.enabled]

    @property
    def fino_enabled(self) -> bool:
        return self.framework == "dinov3" and len(self.enabled_factors) > 0

    def validate_fino(self) -> None:
        """Validate FINO factors (allow/deny guard, vocab, sign). Raises on misconfiguration."""
        if self.metadata_factors and self.framework != "dinov3":
            raise ValueError("metadata_factors require framework: dinov3 (FINO is a DINOv3 extension).")
        if self.fino.lambda_schedule_type not in ("sigmoid", "linear", "constant"):
            raise ValueError(f"fino.lambda_schedule_type must be sigmoid|linear|constant, got {self.fino.lambda_schedule_type!r}.")
        for f in self.metadata_factors:
            f.validate()
            # Crop-scale correction needs the realized per-crop downsample, which only the native_fov
            # path computes (and shares across both global crops). Fail fast rather than silently
            # train an uncorrected label when the user asked for the correction.
            if getattr(f, "crop_scale_correction", False) and f.enabled and not self.augmentation.native_fov:
                raise ValueError(
                    f"FINO factor '{f.name}': crop_scale_correction requires augmentation.native_fov=true "
                    "(the per-crop downsample is only known and shared across both global crops in "
                    "native_fov mode)."
                )

    def resolved_mean_std(self) -> tuple[float, float]:
        mean, std = self.data.mean, self.data.std
        if (mean is None or std is None) and self.data.stats_file and Path(self.data.stats_file).exists():
            import json

            s = json.loads(Path(self.data.stats_file).read_text(encoding="utf-8"))
            mean = mean if mean is not None else s.get("mean_01")
            std = std if std is not None else s.get("std_01")
        if mean is None:
            mean = EM_DEFAULT_MEAN
        if std is None:
            std = EM_DEFAULT_STD
        return float(mean), float(std)

    def augmentation_config(self):
        """Build a transforms.EMAugmentationConfig from this spec."""
        from ..transforms import EMAugmentationConfig

        a = self.augmentation
        return EMAugmentationConfig(
            brightness=a.brightness,
            contrast=a.contrast,
            color_jitter_p=a.color_jitter_p,
            gamma=a.gamma,
            gamma_p=a.gamma_p,
            noise_sigma_max=a.noise_sigma_max,
            noise_p=a.noise_p,
            blur_sigma_min=a.blur_sigma_min,
            blur_sigma_max=a.blur_sigma_max,
            blur_kernel_size=a.blur_kernel_size,
            global1_blur_p=a.global1_blur_p,
            global2_blur_p=a.global2_blur_p,
            local_blur_p=a.local_blur_p,
            dihedral=a.dihedral,
            horizontal_flips=a.horizontal_flips,
            vertical_flips=a.vertical_flips,
            rotations=a.rotations,
            native_fov=a.native_fov,
            native_downsample_max=a.native_downsample_max,
            native_bias=a.native_bias,
            native_local_downsample_max=a.native_local_downsample_max,
            native_overlap_room=a.native_overlap_room,
        )

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self, drop={"raw"})

# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
def _coerce(dc_type, value):
    """Recursively build a dataclass instance from a (possibly partial) dict."""
    if value is None:
        return dc_type()
    if not isinstance(value, dict):
        return value
    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f for f in fields(dc_type)}
    for k, v in value.items():
        if k not in type_hints:
            # Unknown key: keep on raw-passthrough dataclasses (dinov3/extra), else ignore.
            continue
        f = type_hints[k]
        ftype = f.type
        # Handle nested dataclasses by name.
        if k == "schedule":
            kwargs[k] = [_coerce(CropStage, s) for s in (v or [])]
        elif k == "metadata_factors":
            # Build FinoFactorSpec list (fills discrete-vocab defaults; validation is deferred
            # to ExperimentSpec.validate_fino so partial configs still load).
            kwargs[k] = [s if isinstance(s, FinoFactorSpec) else factor_from_dict(dict(s)) for s in (v or [])]
        elif _is_dataclass_type(_resolve(ftype)):
            kwargs[k] = _coerce(_resolve(ftype), v)
        elif isinstance(v, list):
            kwargs[k] = tuple(v) if "tuple" in str(ftype) else v
        else:
            kwargs[k] = v
    return dc_type(**kwargs)

def _resolve(ftype):
    mapping = {
        "ModelSpec": ModelSpec,
        "CropsSpec": CropsSpec,
        "DataSpec": DataSpec,
        "AugmentationSpec": AugmentationSpec,
        "OptimSpec": OptimSpec,
        "TrainSpec": TrainSpec,
        "LoggingSpec": LoggingSpec,
        "CheckpointingSpec": CheckpointingSpec,
        "FinoSpec": FinoSpec,
    }
    if isinstance(ftype, type):
        return ftype
    for name, cls in mapping.items():
        if isinstance(ftype, str) and ftype.endswith(name):
            return cls
    return ftype

def _is_dataclass_type(t) -> bool:
    import dataclasses

    return isinstance(t, type) and dataclasses.is_dataclass(t)

def _dataclass_to_dict(obj, drop=frozenset()):
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in fields(obj):
            if f.name in drop:
                continue
            out[f.name] = _dataclass_to_dict(getattr(obj, f.name), drop)
        return out
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v, drop) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v, drop) for k, v in obj.items()}
    return obj

def load_experiment(path: str | os.PathLike) -> ExperimentSpec:
    """Load a configuration into a normalized ExperimentSpec.

    The shipped configurations under ``configs/`` are experiment definitions: what this project sets,
    with upstream DINOv3's defaults left to upstream rather than restated.

    A fully resolved DINOv3 config is also accepted, so a run's own emitted config can be replayed:
    those carry the experiment they were generated from under ``em.experiment`` — a block DINOv3
    itself ignores — and it is unwrapped here.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    embedded = (raw.get("em") or {}).get("experiment") if isinstance(raw.get("em"), dict) else None
    if isinstance(embedded, dict):
        raw = embedded
    spec = _coerce(ExperimentSpec, raw)
    spec.config_path = str(path)
    spec.raw = raw
    return spec

def resolve_data_paths(
    spec: ExperimentSpec,
    data_root: str | os.PathLike | None = None,
    manifest: str | os.PathLike | None = None,
) -> ExperimentSpec:
    """Fill data paths from CLI overrides / a data-root bundle layout.

    A "data root" bundle (the transferable directory) is expected to contain:
        <root>/shards/<prefix>-*.tar                   (or <root>/shards/<prefix>/<prefix>-*.tar)
        <root>/data_prep/ssl_manifest.filtered.jsonl   (and parquet; <root>/manifests/ also accepted)
        <root>/data_prep/tile_intensity_stats.json     (<root>/stats/ also accepted)
    Any field already set on the spec wins; CLI args win over bundle inference.
    """
    d = spec.data
    if manifest:
        d.manifest = str(manifest)
    if data_root:
        root = Path(data_root)
        if d.shard_dir is None and (root / "shards").is_dir():
            shards_root = root / "shards"
            sub = shards_root / d.shard_prefix
            if sub.is_dir() and any(sub.glob("*.tar")):
                d.shard_dir = str(sub)  # shards/<prefix>/<prefix>-*.tar layout
            elif any(shards_root.glob("*.tar")):
                d.shard_dir = str(shards_root)  # shards/<prefix>-*.tar layout
            else:
                d.shard_dir = str(sub if sub.is_dir() else shards_root)
        if d.stats_file is None:
            for cand in (root / "data_prep" / "tile_intensity_stats.json", root / "stats" / "tile_intensity_stats.json"):
                if cand.exists():
                    d.stats_file = str(cand)
                    break
        if d.manifest is None:
            for cand in (
                root / "data_prep" / "ssl_manifest.filtered.jsonl",
                root / "manifests" / "ssl_manifest.filtered.jsonl",
            ):
                if cand.exists():
                    d.manifest = str(cand)
                    break
    if d.min_side is None:
        d.min_side = spec.effective_min_side
    return spec

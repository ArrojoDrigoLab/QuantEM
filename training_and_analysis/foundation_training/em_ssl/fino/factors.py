"""EM metadata-factor layer for FINO guided training.

This is the EM-specific glue around the upstream FINO machinery (taken from the
pinned DINOv3 ``FINO`` branch: ``dinov3.train.guided_ssl_meta_arch.GuidedSSLMetaArch`` +
``dinov3.train.metadata_utils``). Upstream assumes *complete* metadata supplied by a dataset
that emits ``(image, (label, metadata_dataclass))``; this module provides:

  * the allow/deny guard — only ``log(effective_nm_per_px)``, ``modality`` and
    ``organ`` may drive a guide head; ``tissue``/``scale_band`` and all provenance ids
    (``source_id``/``dataset_id``/``asset_id``/…) and ``species_group``/``prep_context``/
    ``in_situ_status`` are diagnostics only and raise if used as objectives;
  * ``modality`` / ``organ`` vocab canonicalization into a small fixed class set;
  * ``log_effective_nm_per_px`` derivation (positivity/finiteness checks + z-score standardize);
  * ``EMTileMetadata`` — a picklable per-sample dataclass with encoded fields (so the
    upstream ``_collate_metadata`` batches them into tensors) plus a parallel ``*_valid`` mask
    per factor (the EM extension over the reference, since EM metadata is partial);
  * ``FinoTargetTransform`` (the dataset ``target_transform`` installed in FINO mode) and
    ``FinoRuntime`` (the resolved-factor holder set as ``dinov3_patch.ACTIVE_FINO``).

Pure-stdlib (no torch): per-sample encoding produces plain ``int``/``float``/``str`` so the
module is importable by the CPU data tools and unit-testable without torch. Batching to
tensors happens in the upstream collate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field as dfield, fields as dc_fields
from typing import Any

# --------------------------------------------------------------------------- #
# Allow / deny policy
# --------------------------------------------------------------------------- #
# The only manifest fields permitted to drive a FINO guide head, with their logical names.
ALLOWED_OBJECTIVES: dict[str, str] = {
    "log_effective_nm_per_px": "effective_nm_per_px",  # logical name -> source field
    "modality": "modality",
    "organ": "organ",  # discrete biological factor: organ (~9 classes). tissue is logged-only (100+, ~=dataset)
}
# Fields that are never FINO objectives — provenance/grouping/diagnostics only.
DENIED_OBJECTIVE_FIELDS: frozenset[str] = frozenset(
    {
        "scale_band",
        "source_id",
        "dataset_id",
        "asset_id",
        "experiment_id",
        "run_id",
        "normalization_hash",
        "species_group",
        "species",
        "prep_context",
        "in_situ_status",
        "tissue",          # fine cell/sample-level tissue -> logged diagnostic only (organ is the objective)
        "tissue_context",  # coarse tissue context -> `organ` is the objective
    }
)

UNKNOWN = "unknown"

# Metadata key carrying the realized per-sample global-crop downsample factor, so a crop-scale-aware
# continuous factor can correct nm/px to the crop's true resolution. Produced by
# ``EMDataAugmentationDINO``, forwarded by ``EMShardDataset.__iter__``, consumed by
# ``encode_tile_metadata`` below. Those modules use the literal rather than importing it.
CROP_GLOBAL_DOWNSAMPLE_KEY = "global_downsample"

# Sensible default vocabularies (the coverage tool emits data-derived ones to paste into a
# config; these defaults keep tests/smoke runnable and document the expected label set).
DEFAULT_MODALITY_CLASSES: list[str] = [
    "FIB-SEM",
    "SBF-SEM",
    "TEM",
    "ssTEM",
    "SEM",
    "STEM",
    "cryo-EM",
    "other",
]
DEFAULT_MODALITY_NORMALIZE: dict[str, str] = {
    "fibsem": "FIB-SEM",
    "fib": "FIB-SEM",
    "focusedionbeamsem": "FIB-SEM",
    "sbfsem": "SBF-SEM",
    "sbsem": "SBF-SEM",
    "serialblockface": "SBF-SEM",
    "serialblockfacesem": "SBF-SEM",
    "tem": "TEM",
    "sstem": "ssTEM",
    "serialsectiontem": "ssTEM",
    "sem": "SEM",
    "stem": "STEM",
    "cryoem": "cryo-EM",
    "cryo": "cryo-EM",
    "cryoelectronmicroscopy": "cryo-EM",
}
DEFAULT_ORGAN_CLASSES: list[str] = [
    # Default organ vocabulary. ``em_ssl.tools.fino_metadata_coverage`` derives the
    # corpus-specific list, which the config then pins.
    "Brain",
    "Liver",
    "Kidney",
    "Pancreas",
    "Intestine",
    "Heart",
    "Muscle",
    "Lung",
    "other",
]

def _norm_token(s: Any) -> str:
    """Normalize a label for vocab matching: lowercase, strip non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

# --------------------------------------------------------------------------- #
# Factor spec
# --------------------------------------------------------------------------- #
@dataclass
class FinoFactorSpec:
    """One metadata factor to preserve (M+) or suppress (M-) during FINO training.

    Mirrors the experiment-YAML ``metadata_factors[]`` entry and is also the resolved runtime
    spec used to (a) encode per-sample metadata and (b) build the DINOv3 ``guide:`` block.

    ``guidance``: ``positive`` (M+, grl=False) | ``negative`` (M-, grl=True) | ``disabled``.
    Discrete factors carry ``classes`` (the ordered vocab) + ``normalize_map`` (raw->canonical);
    continuous factors carry ``log_transform`` + ``standardize_mean``/``standardize_std``.
    """

    name: str = ""
    field: str = ""
    type: str = "discrete"  # "discrete" | "continuous"
    guidance: str = "positive"  # "positive" | "negative" | "disabled"
    loss_weight: float = 1.0
    method: str = "auto"  # auto -> classification(discrete)/regression(continuous)
    # discrete
    classes: list[str] = dfield(default_factory=list)
    normalize_map: dict[str, str] = dfield(default_factory=dict)
    include_unknown: bool = False
    use_bce: bool = False
    # continuous
    log_transform: bool = False
    standardize_mean: float | None = None
    standardize_std: float | None = None
    positive_only: bool = True  # require source value > 0 (effective_nm_per_px is a length)
    # Crop-scale correction, for continuous scale factors only. The guide head reads the CLS of a global
    # crop that native_fov downsampled by a per-sample factor M, so the crop's true resolution is
    # value*M. When set, the encoder applies that factor before log and standardisation, making the
    # regression target the crop's real nm/px rather than the tile's. Requires augmentation.native_fov.
    # Discrete factors are crop-invariant and ignore this.
    crop_scale_correction: bool = False
    # head hyperparameters (forwarded to the upstream Classifier/Regressor MLP)
    hidden_dim: list[int] = dfield(default_factory=lambda: [512, 512, 256])
    dropout: float = 0.5
    grl_space: str = "embedding"  # "embedding" | "prototype"
    notes: str = ""

    # -- derived ---------------------------------------------------------- #
    @property
    def enabled(self) -> bool:
        return self.guidance in ("positive", "negative")

    @property
    def grl(self) -> bool:
        return self.guidance == "negative"

    @property
    def is_continuous(self) -> bool:
        return self.type == "continuous"

    @property
    def effective_method(self) -> str:
        if self.method and self.method != "auto":
            return self.method
        return "regression" if self.is_continuous else "classification"

    @property
    def effective_classes(self) -> list[str]:
        cls = list(self.classes)
        if self.include_unknown and UNKNOWN not in cls:
            cls = cls + [UNKNOWN]
        return cls

    @property
    def n_outputs(self) -> int:
        return 1 if self.is_continuous else len(self.effective_classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.effective_classes)}

    @property
    def lookup(self) -> dict[str, str]:
        """norm-token -> canonical class, from classes + normalize_map."""
        lut: dict[str, str] = {_norm_token(c): c for c in self.effective_classes}
        for raw, canon in self.normalize_map.items():
            lut[_norm_token(raw)] = canon
        return lut

    # -- validation ------------------------------------------------------- #
    def validate(self) -> None:
        if not self.enabled:
            return
        if self.field in DENIED_OBJECTIVE_FIELDS:
            raise ValueError(
                f"FINO factor '{self.name}': field '{self.field}' is a diagnostic/provenance "
                f"field and is not permitted as a FINO objective (denied: {sorted(DENIED_OBJECTIVE_FIELDS)})."
            )
        if self.name not in ALLOWED_OBJECTIVES or ALLOWED_OBJECTIVES[self.name] != self.field:
            raise ValueError(
                f"FINO factor '{self.name}' (field '{self.field}') is not an allowed objective. "
                f"Allowed name->field: {ALLOWED_OBJECTIVES}."
            )
        if self.type not in ("discrete", "continuous"):
            raise ValueError(f"FINO factor '{self.name}': type must be discrete|continuous, got '{self.type}'.")
        if self.guidance not in ("positive", "negative", "disabled"):
            raise ValueError(f"FINO factor '{self.name}': guidance must be positive|negative|disabled.")
        if self.crop_scale_correction and not self.is_continuous:
            raise ValueError(
                f"FINO factor '{self.name}': crop_scale_correction is only valid for a continuous "
                "(scale) factor — discrete factors (modality/organ) are crop-invariant."
            )
        if self.effective_method == "prototypical":
            # The prototypical head maintains global EMA centroids and is not mask-aware, so
            # it cannot honour the per-sample missing-metadata mask the EM corpus requires. The
            # small EM vocabularies (modality ~8, organ ~9) are well served by
            # classification.
            raise ValueError(
                f"FINO factor '{self.name}': prototypical guide heads are not supported in EM FINO "
                "(they do not honour the per-sample missing-metadata mask). Use classification."
            )
        if not self.is_continuous and self.use_bce:
            # BCE-with-logits expects multi-hot [B, C] targets; EM discrete factors are
            # single-label (one class index per tile), so BCE is a misconfiguration here.
            raise ValueError(
                f"FINO factor '{self.name}': use_bce (multi-label BCE) is unsupported — EM discrete "
                "factors are single-label. Use cross-entropy (use_bce: false)."
            )
        if self.is_continuous:
            if self.standardize_std is not None and float(self.standardize_std) == 0.0:
                raise ValueError(f"FINO factor '{self.name}': standardize_std must be non-zero.")
        else:
            if not self.effective_classes:
                raise ValueError(
                    f"FINO factor '{self.name}': discrete factor needs a non-empty `classes` vocab "
                    f"(or include_unknown: true). Run em_ssl.tools.fino_metadata_coverage to derive it."
                )

    # -- encoding (per sample, pure python) ------------------------------- #
    def encode_value(self, raw: Any, downsample: float = 1.0) -> tuple[float | int, bool]:
        """Encode one raw metadata value -> (encoded, valid).

        Discrete -> (class index or -1, valid). Continuous -> (standardized value or 0.0, valid).
        ``valid=False`` means the sample is masked out of this factor's guide loss (but still
        used for ordinary SSL). ``downsample`` (>=1, the realized native_fov global-crop downsample)
        only takes effect for a continuous factor with ``crop_scale_correction`` set, where it scales
        the value to the crop's true resolution before log+standardize; ignored otherwise.
        """
        if self.is_continuous:
            return self._encode_continuous(raw, downsample)
        return self._encode_discrete(raw)

    def _encode_discrete(self, raw: Any) -> tuple[int, bool]:
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            canon = None
        else:
            canon = self.lookup.get(_norm_token(raw))
        idx_map = self.class_to_idx
        if canon is not None and canon in idx_map:
            return idx_map[canon], True
        if self.include_unknown:
            return idx_map.get(UNKNOWN, -1), idx_map.get(UNKNOWN, -1) >= 0
        return -1, False

    def _encode_continuous(self, raw: Any, downsample: float = 1.0) -> tuple[float, bool]:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return 0.0, False
        if not math.isfinite(v):
            return 0.0, False
        if self.positive_only and v <= 0.0:
            return 0.0, False
        # Crop-scale correction: the embedded crop is downsampled by `downsample` (native_fov), so its
        # true resolution is v*downsample (coarser). Apply before log+standardize so the regression
        # target is the crop's real value. log(v*M) = log(v) + log(M): an additive shift in log space.
        if self.crop_scale_correction:
            try:
                d = float(downsample)
            except (TypeError, ValueError):
                d = 1.0
            if math.isfinite(d) and d > 0.0:
                v = v * d
        x = math.log(v) if self.log_transform else v
        if self.standardize_mean is not None and self.standardize_std:
            x = (x - float(self.standardize_mean)) / float(self.standardize_std)
        return float(x), True

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

# --------------------------------------------------------------------------- #
# Per-sample metadata payload: encoded fields plus validity masks, picklable
# --------------------------------------------------------------------------- #
# Fixed-shape so the upstream `_collate_metadata` batches consistently. Each factor's field
# name matches its guide name (GuidedSSLMetaArch does `getattr(metadata, guide.name)`); the
# `<name>_valid` masks are the EM extension read by the masked guide-loss patch. Encoded
# values are always concrete (never None) so collation yields clean tensors.
@dataclass
class EMTileMetadata:
    log_effective_nm_per_px: float = 0.0
    modality: int = -1
    organ: int = -1
    log_effective_nm_per_px_valid: bool = False
    modality_valid: bool = False
    organ_valid: bool = False
    # diagnostics only (never a guide objective) — batched by _collate_metadata into lists
    source_id: str = ""
    dataset_id: str = ""

_METADATA_FIELD_NAMES = {f.name for f in dc_fields(EMTileMetadata)}

def encode_tile_metadata(meta: dict[str, Any], factors: list[FinoFactorSpec]) -> EMTileMetadata:
    """Build an :class:`EMTileMetadata` from a shard sidecar dict for the given factors.

    If the dict carries a bridged ``CROP_GLOBAL_DOWNSAMPLE_KEY`` (the realized native_fov global-crop
    downsample injected by the dataset), crop-scale-aware continuous factors use it to correct their
    value to the crop's true resolution. Absent (native_fov off / non-FINO / unit tests) -> 1.0 (no-op).
    """
    md = EMTileMetadata()
    raw_downsample = meta.get(CROP_GLOBAL_DOWNSAMPLE_KEY)
    try:
        downsample = float(raw_downsample) if raw_downsample is not None else 1.0
    except (TypeError, ValueError):
        downsample = 1.0
    for fac in factors:
        if not fac.enabled or fac.name not in _METADATA_FIELD_NAMES:
            continue
        enc, valid = fac.encode_value(meta.get(fac.field), downsample)
        setattr(md, fac.name, enc)
        setattr(md, f"{fac.name}_valid", bool(valid))
    md.source_id = str(meta.get("source_id") or "")
    md.dataset_id = str(meta.get("dataset_id") or "")
    return md

@dataclass
class FinoTargetTransform:
    """Picklable dataset ``target_transform`` for FINO mode.

    Maps a shard sidecar dict -> ``((), EMTileMetadata)`` so the EM dataset yields
    ``(crops, ((), metadata))`` — the shape the upstream FINO ``collate_data_and_cast`` expects
    (``samples_list[i][1][1]`` is the metadata dataclass). Carries the resolved factor specs so
    encoding works in dataloader workers on both fork (Linux) and spawn (Windows).
    """

    factors: list[FinoFactorSpec]

    def __call__(self, meta: dict[str, Any]) -> tuple:
        return ((), encode_tile_metadata(meta, self.factors))

# --------------------------------------------------------------------------- #
# Runtime holder (set as dinov3_patch.ACTIVE_FINO)
# --------------------------------------------------------------------------- #
@dataclass
class FinoRuntime:
    """Resolved FINO factors for one run.

    The trainer installs it as ``dinov3_patch.ACTIVE_FINO``; ``make_em_dataset`` reads it there and
    swaps in the factor-aware ``target_transform``.
    """

    factors: list[FinoFactorSpec]

    @property
    def enabled_factors(self) -> list[FinoFactorSpec]:
        return [f for f in self.factors if f.enabled]

    def target_transform(self) -> FinoTargetTransform:
        return FinoTargetTransform(self.enabled_factors)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def factor_from_dict(d: dict[str, Any]) -> FinoFactorSpec:
    """Build a FinoFactorSpec from a config dict, filling discrete-vocab defaults."""
    kw = {f.name: d[f.name] for f in dc_fields(FinoFactorSpec) if f.name in d}
    spec = FinoFactorSpec(**kw)
    # Provide default vocab for discrete factors when not given explicitly.
    if not spec.is_continuous and not spec.classes:
        if spec.field == "modality":
            spec.classes = list(DEFAULT_MODALITY_CLASSES)
            if not spec.normalize_map:
                spec.normalize_map = dict(DEFAULT_MODALITY_NORMALIZE)
        elif spec.field == "organ":
            spec.classes = list(DEFAULT_ORGAN_CLASSES)
    return spec

def factors_from_config(factor_dicts: list[Any]) -> list[FinoFactorSpec]:
    """Build + validate FinoFactorSpec list from config entries (dicts or FinoFactorSpec)."""
    out: list[FinoFactorSpec] = []
    for d in factor_dicts or []:
        spec = d if isinstance(d, FinoFactorSpec) else factor_from_dict(dict(d))
        spec.validate()
        out.append(spec)
    return out

def fino_factors_fingerprint(factors: list[FinoFactorSpec]) -> str:
    """Stable short hash of the factor specs (for checkpoint metadata / provenance)."""
    payload = json.dumps([f.to_dict() for f in factors], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

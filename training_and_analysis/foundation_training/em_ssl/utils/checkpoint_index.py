"""Checkpoint index + encoder manifest.

Every SSL run writes ``<run_dir>/checkpoint_index.json`` describing the encoder and every
saved checkpoint in a *framework-agnostic* way, so a later decoder-evaluation script can
load any released checkpoint with identical code paths and without guessing. This is the contract
``encoder_evaluation`` and ``segmentation_training`` load encoders through.

The header (`EncoderManifest`) records everything the probe needs to instantiate the
backbone and extract features:
  run_id, framework, objective, arch, patch_size, embedding_dim, depth, input_channels,
  image_mean / image_std (single-channel EM stats, not ImageNet), the crop schedule, the
  feature-extraction entry point, and which intermediate layers to read.

Each `CheckpointRecord` points at a concrete artifact, one per ``kind``:
  * ``teacher`` — DINOv3 teacher export ``eval/<step>/teacher_checkpoint.pth`` =
    torch.save({"teacher": ema_sd}) whose ``backbone.*`` keys are the feature extractor.
  * ``resume`` — DINOv3 ``ckpt/<step>/`` DCP sharded dir (student+teacher+optimizer); resume only.
  * ``encoder`` — a single published-baseline weight file, indexed by
    ``foundation_baselines.register_external_encoders`` so an external encoder loads through the
    same path. ``teacher_checkpoints()`` returns the ``teacher`` and ``encoder`` records together,
    since both are loadable feature extractors.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class EncoderManifest:
    run_id: str
    framework: str  # "dinov3" | "timm_vit"
    objective: str  # e.g. "dino+ibot+koleo", "dino+ibot+koleo+gram"
    arch: str  # vit_base / vit_large / vit_huge2
    patch_size: int
    embedding_dim: int
    depth: int
    input_channels: int = 1
    image_mean: list[float] = field(default_factory=lambda: [0.583])
    image_std: list[float] = field(default_factory=lambda: [0.244])
    crop_schedule: list[dict[str, Any]] = field(default_factory=list)
    # How a downstream probe should extract features from the saved checkpoint.
    feature_entry_point: dict[str, Any] = field(default_factory=dict)
    intermediate_layers: list[int] = field(default_factory=list)
    dataset_fingerprint_ref: str | None = None
    config_path: str | None = None
    notes: str = ""
    # FINO metadata-guided provenance (defaults ⇒ baseline run; so a later fixed-decoder eval
    # knows exactly which metadata factors this checkpoint preserved (M+) / suppressed (M-)).
    fino_enabled: bool = False
    fino_factors: list[dict[str, Any]] = field(default_factory=list)
    fino_lambda_schedule: dict[str, Any] | None = None
    fino_grad_norm_normalization: bool = False
    fino_factors_fingerprint: str | None = None
    manifest_fingerprint: str | None = None
    shard_fingerprint: str | None = None
    metadata_coverage_fingerprint: str | None = None

@dataclass
class CheckpointRecord:
    step: int
    kind: str  # "teacher" | "resume" | "encoder"
    path: str
    crop_size: int | None = None
    stage_name: str | None = None
    sha256: str | None = None
    created_unix: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

class CheckpointIndex:
    """Append-only index persisted at ``<run_dir>/checkpoint_index.json``."""

    FILENAME = "checkpoint_index.json"

    def __init__(self, run_dir: str | os.PathLike, manifest: EncoderManifest):
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        self.records: list[CheckpointRecord] = []

    @property
    def path(self) -> Path:
        return self.run_dir / self.FILENAME

    # -- mutation ----------------------------------------------------------
    def add(
        self,
        step: int,
        kind: str,
        path: str | os.PathLike,
        crop_size: int | None = None,
        stage_name: str | None = None,
        sha256: str | None = None,
        **extra: Any,
    ) -> CheckpointRecord:
        rec = CheckpointRecord(
            step=int(step),
            kind=kind,
            path=str(path),
            crop_size=crop_size,
            stage_name=stage_name,
            sha256=sha256,
            created_unix=_now(),
            extra=extra,
        )
        # Replace any existing record with the same (kind, step).
        self.records = [r for r in self.records if not (r.kind == kind and r.step == step)]
        self.records.append(rec)
        self.records.sort(key=lambda r: (r.step, r.kind))
        self.save()
        return rec

    # -- queries -----------------------------------------------------------
    def latest(self, kind: str | None = None) -> CheckpointRecord | None:
        recs = [r for r in self.records if kind is None or r.kind == kind]
        return recs[-1] if recs else None

    def teacher_checkpoints(self) -> list[CheckpointRecord]:
        return [r for r in self.records if r.kind in ("teacher", "encoder")]

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder": asdict(self.manifest),
            "checkpoints": [asdict(r) for r in self.records],
            "schema_version": 1,
        }

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
        return self.path

    @classmethod
    def load(cls, run_dir: str | os.PathLike) -> "CheckpointIndex":
        run_dir = Path(run_dir)
        data = json.loads((run_dir / cls.FILENAME).read_text(encoding="utf-8"))
        manifest = EncoderManifest(**data["encoder"])
        idx = cls(run_dir, manifest)
        idx.records = [CheckpointRecord(**r) for r in data.get("checkpoints", [])]
        return idx

def _now() -> float:
    return time.time()

def dinov3_feature_entry_point(arch: str, depth: int, patch_size: int = 16, in_chans: int = 1) -> dict[str, Any]:
    """Descriptor telling a downstream loader how to load a DINOv3 teacher checkpoint.

    ``build.kwargs`` records only ``patch_size`` / ``in_chans``. The rest of the ViT block config
    (LayerScale, register/storage tokens, untied CLS/patch norms) is reconstructed from the
    checkpoint's own keys, because the bare arch factory defaults it off (e.g. LayerScale is on in
    DINOv3 training but ``layerscale_init=None`` by default). Loaders call
    :func:`infer_dinov3_build_kwargs` on those keys to rebuild a matching backbone; calling the
    factory with only these kwargs drops ls1/ls2.gamma and yields mismatched features.
    """
    return {
        "loader": "dinov3_teacher",
        "checkpoint_key": "teacher",
        "backbone_prefix": "backbone.",
        "build": {
            "module": "dinov3.models.vision_transformer",
            "factory": arch,
            "kwargs": {"patch_size": patch_size, "in_chans": in_chans},
        },
        "forward": "get_intermediate_layers",
        "note": "torch.save({'teacher': model_ema.state_dict()}); strip 'backbone.'; rebuild via "
                "infer_dinov3_build_kwargs(backbone_sd, build['kwargs']) so LayerScale/registers match.",
    }

def infer_dinov3_build_kwargs(backbone_state_dict: dict, base_kwargs: dict | None = None) -> dict:
    """Reconstruct the DINOv3 ViT block config from a teacher backbone state dict.

    The bare arch factory (``vit_base(patch_size, **kwargs)``) uses ctor defaults that differ from the
    SSL training config — most importantly LayerScale is off by default (``layerscale_init=None``)
    but on in DINOv3 training (``ssl_default_config.yaml: layerscale: 1e-5``). Building without it drops
    the checkpoint's ``ls1/ls2.gamma`` and yields mismatched features. This infers the needed kwargs
    (LayerScale, register/storage tokens, untied CLS/patch norms) from the checkpoint's own keys, so any
    loader rebuilds a matching backbone regardless of what the index recorded. Stdlib only (reads keys +
    tensor ``.shape``; no torch import).
    """
    kw = dict(base_kwargs or {})
    keys = list(backbone_state_dict.keys())
    if "storage_tokens" in backbone_state_dict:
        try:
            kw["n_storage_tokens"] = int(backbone_state_dict["storage_tokens"].shape[1])
        except Exception:  # pragma: no cover - defensive on odd tensors
            pass
    if any(".ls1." in k or ".ls2." in k for k in keys):
        kw.setdefault("layerscale_init", 1.0e-05)  # module must exist; the loaded gamma overwrites this
    if any(k.startswith("cls_norm.") for k in keys):
        kw["untie_cls_and_patch_norms"] = True
    return kw


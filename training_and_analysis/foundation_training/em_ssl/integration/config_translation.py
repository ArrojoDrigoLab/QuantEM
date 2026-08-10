"""Translate an em_ssl ExperimentSpec (one crop stage) into DINOv3 configuration keys.

Emits the keys that differ from DINOv3's ssl_default_config.yaml, plus an ``em:`` block read by the
augmentation patch. The files under configs/ are the experiment definitions this translation reads;
``em_ssl.config.resolve`` merges its output under the upstream default to produce the complete
configuration a run sees. DINOv3's setup_config(strict_cfg=False) accepts the extra ``em:`` keys.

Step budgeting: DINOv3 has no max_steps; total steps = optim.epochs * train.OFFICIAL_EPOCH_LENGTH
and the sampler is infinite. OFFICIAL_EPOCH_LENGTH and epochs are chosen so their product ≈
the stage's max_steps, keeping schedule granularity reasonable.
"""

from __future__ import annotations

import math
from typing import Any

from ..arch import resolve_arch
from ..config.schema import CropStage, ExperimentSpec
from ..fino.factors import FinoFactorSpec, fino_factors_fingerprint

def _step_budget(max_steps: int, target_oel: int) -> tuple[int, int]:
    """Return (official_epoch_length, epochs) whose product ≈ max_steps."""
    target_oel = max(1, int(target_oel))
    if max_steps <= target_oel:
        return max(1, int(max_steps)), 1
    epochs = max(1, round(max_steps / target_oel))
    oel = max(1, max_steps // epochs)
    return oel, epochs

def em_dataset_path(spec: ExperimentSpec, stage: CropStage, seed: int) -> str:
    d = spec.data
    if d.use_loose_files:
        # Loose-file debug mode is handled by the runner, not the dataset string.
        return "EMLoose"
    root = d.shard_dir or "<shard_dir>"
    min_side = stage.global_crops_size  # never below the stage's global crop
    return (
        f"EMShards:root={root}:prefix={d.shard_prefix}:min_side={min_side}"
        f":shuffle={max(spec.train.batch_size_per_gpu * 16, 1000)}:seed={seed}:resampled=1"
    )

def _gram_enabled(spec: ExperimentSpec) -> bool:
    """Is the Gram loss on? Either separate gram-teacher crops, or ``gram.use_loss`` anchoring to the
    live EMA teacher, which needs no extra crops and so leaves gram_teacher_crops_size null."""
    if spec.crops.gram_teacher_crops_size:
        return True
    gram = spec.dinov3.get("gram") if isinstance(spec.dinov3, dict) else None
    return bool(isinstance(gram, dict) and gram.get("use_loss"))

def objective_string(spec: ExperimentSpec, gram_enabled: bool) -> str:
    # SIGReg (DINO-head bottleneck regularizer) replaces KoLeo unless koleo_too is set; read it
    # off the raw dinov3 escape-hatch block so checkpoint provenance is labelled correctly.
    sig = spec.dinov3.get("sigreg") if isinstance(spec.dinov3, dict) else None
    if isinstance(sig, dict) and sig.get("enabled"):
        reg = "sigreg+koleo" if sig.get("koleo_too") else "sigreg"
    else:
        reg = "koleo"
    base = f"dino+ibot+{reg}"
    return base + ("+gram" if gram_enabled else "")

def fino_guide_entry(f: FinoFactorSpec) -> dict[str, Any]:
    """Translate one resolved FinoFactorSpec into a DINOv3 ``guide.guides[]`` entry.

    The full entry shape is emitted (incl. proto_*/target_normalization) so the upstream
    ``GuidedSSLMetaArch.__init__`` can read every field it touches regardless of method. The
    sign is explicit: ``grl`` marks negative guidance (M-), for which the guide-loss step negates
    the lambda — reversing the gradient that reaches the backbone — and divides the loss weight by
    10 (``em_ssl.fino.meta_arch_patch``).
    """
    return {
        "name": f.name,
        "enabled": True,
        "method": f.effective_method,
        "hidden_dim": list(f.hidden_dim),
        "dropout": f.dropout,
        "loss_weight": f.loss_weight,
        "n_outputs": f.n_outputs,
        "grl": f.grl,  # positive guidance -> False (preserve); negative -> True (suppress)
        "grl_space": f.grl_space,
        "use_bce": f.use_bce,
        "output_activation": "none",  # continuous factors are z-standardised upstream of the loss
        "target_normalization": {"output_min": None, "output_max": None},
        "proto_temperature": 0.07,
        "proto_centroid_momentum": 0.999,
        "proto_phi_min": 0.01,
    }

def fino_guide_block(spec: ExperimentSpec) -> dict[str, Any]:
    """Build the DINOv3 ``guide:`` config block from the experiment's enabled factors."""
    return {
        "enabled": True,
        "guides": [fino_guide_entry(f) for f in spec.enabled_factors],
        "lambda_schedule": {
            "type": spec.fino.lambda_schedule_type,
            "warmup_iterations": int(spec.fino.lambda_warmup_steps),
        },
    }

def translate_stage(
    spec: ExperimentSpec,
    stage_index: int,
    output_dir: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the sparse DINOv3 override config for one crop stage."""
    stage = spec.crops.schedule[stage_index]
    mean, std = spec.resolved_mean_std()
    arch = resolve_arch(spec.model.arch)

    oel, epochs = _step_budget(stage.max_steps, spec.train.official_epoch_length)
    warmup_steps = stage.warmup_steps if stage.warmup_steps is not None else spec.optim.warmup_steps
    warmup_epochs = max(1, round(warmup_steps / oel)) if warmup_steps else 0
    lr = stage.lr if stage.lr is not None else spec.optim.lr

    gram_enabled = _gram_enabled(spec)

    teacher_export_every = max(1, round(spec.checkpointing.teacher_export_every_steps))

    cfg: dict[str, Any] = {
        "MODEL": {"META_ARCHITECTURE": "SSLMetaArch", "DEVICE": "cuda"},
        "compute_precision": {
            "param_dtype": spec.train.param_dtype,
            "sharding_strategy": "SHARD_GRAD_OP",  # hard-required by DINOv3
        },
        "student": {
            "arch": arch.name,
            "patch_size": spec.model.patch_size,
            "in_chans": spec.model.in_chans,
            "drop_path_rate": spec.model.drop_path_rate,
            "n_storage_tokens": spec.model.n_storage_tokens,
            **_student_rope_aug(spec.model),
        },
        "teacher": {
            "in_chans": spec.model.in_chans,
        },
        "crops": {
            "global_crops_size": stage.global_crops_size,
            "local_crops_size": stage.local_crops_size,
            "local_crops_number": spec.crops.local_crops_number,
            "global_crops_scale": list(spec.crops.global_crops_scale),
            "local_crops_scale": list(spec.crops.local_crops_scale),
            "gram_teacher_crops_size": spec.crops.gram_teacher_crops_size,
            "rgb_mean": [round(mean, 6)],
            "rgb_std": [round(std, 6)],
        },
        "ibot": {"separate_head": True},
        "gram": {"use_loss": gram_enabled},
        "train": {
            "batch_size_per_gpu": (
                stage.batch_size_per_gpu if stage.batch_size_per_gpu is not None else spec.train.batch_size_per_gpu
            ),
            "num_workers": spec.train.num_workers,
            "OFFICIAL_EPOCH_LENGTH": oel,
            "dataset_path": em_dataset_path(spec, stage, seed),
            "output_dir": output_dir,
            "compile": spec.train.compile,
            "checkpointing": spec.train.activation_checkpointing,  # activation checkpointing in DINOv3
            "seed": seed,
        },
        "optim": {
            "epochs": epochs,
            "lr": lr,
            "warmup_epochs": warmup_epochs,
            "weight_decay": spec.optim.weight_decay,
            "weight_decay_end": spec.optim.weight_decay_end,
            "min_lr": spec.optim.min_lr,
            "clip_grad": spec.optim.clip_grad,
        },
        "checkpointing": {
            "period": max(1, round(spec.checkpointing.period_steps)),
            "max_to_keep": spec.checkpointing.keep_last,
            "keep_every": max(1, round(spec.checkpointing.keep_every_steps)),
        },
        "evaluation": {
            "eval_period_iterations": teacher_export_every,
        },
        # EM-specific block consumed by em_ssl.integration.dinov3_patch.build_em_data_augmentation
        "em": {
            **spec.augmentation_config().to_dict(),
            "teacher_no_color_jitter": spec.augmentation.teacher_no_color_jitter,
            # The experiment definition this configuration was generated from. DINOv3 ignores the
            # `em:` block, and carrying it here lets the resolved file be launched directly.
            "experiment": dict(spec.raw) if spec.raw else None,
            "stage_index": stage_index,
        },
    }

    # FINO metadata-guided training: switch to GuidedSSLMetaArch + add the guide block. When no
    # factors are enabled this branch is skipped, so the config is byte-for-byte the baseline.
    if spec.fino_enabled:
        cfg["MODEL"]["META_ARCHITECTURE"] = "GuidedSSLMetaArch"
        cfg["guide"] = fino_guide_block(spec)
        cfg["optim"]["grad_norm_normalization"] = bool(spec.fino.grad_norm_normalization)

    # Deep-merge raw dinov3 escape-hatch overrides last (wins).
    if spec.dinov3:
        _deep_merge(cfg, dict(spec.dinov3))

    return cfg

def _student_rope_aug(model) -> dict:
    """DINOv3 RoPE coordinate augmentations -> ``student.pos_embed_rope_*`` (only the non-None ones, so a
    ``None`` field leaves DINOv3's own default rather than writing an explicit null). These live under
    ``student`` — disjoint from the FINO ``guide``/provenance surfaces — and are still overridable per
    experiment via the ``dinov3:`` escape hatch (deep-merged last). See ``ModelSpec`` for provenance:
    rescale=2 matches stock DINOv3 pretraining; shift and jitter have no upstream precedent."""
    out: dict[str, float] = {}
    for key, val in (
        ("pos_embed_rope_rescale_coords", getattr(model, "rope_rescale_coords", None)),
        ("pos_embed_rope_shift_coords", getattr(model, "rope_shift_coords", None)),
        ("pos_embed_rope_jitter_coords", getattr(model, "rope_jitter_coords", None)),
    ):
        if val is not None:
            out[key] = float(val)
    return out

def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def encoder_manifest_fields(spec: ExperimentSpec, stage_index: int) -> dict[str, Any]:
    """Fields for CheckpointIndex / EncoderManifest derived from the spec + arch."""
    arch = resolve_arch(spec.model.arch)
    mean, std = spec.resolved_mean_std()
    gram_enabled = _gram_enabled(spec)
    # Default intermediate layers: last 4 blocks (common for ViT feature probing).
    last4 = list(range(max(0, arch.depth - 4), arch.depth))
    # FINO factors trained into this checkpoint (sign + weight + class count) so the later
    # fixed-decoder evaluation knows exactly what was preserved/suppressed.
    fino_factors = [
        {
            "name": f.name,
            "field": f.field,
            "type": f.type,
            "guidance": f.guidance,  # positive (M+) | negative (M-)
            "grl": f.grl,
            "loss_weight": f.loss_weight,
            "method": f.effective_method,
            "n_outputs": f.n_outputs,
            "classes": list(f.effective_classes) if not f.is_continuous else None,
            "crop_scale_correction": f.crop_scale_correction,
            "standardize": (
                {"mean": f.standardize_mean, "std": f.standardize_std, "log_transform": f.log_transform}
                if f.is_continuous
                else None
            ),
        }
        for f in spec.enabled_factors
    ]
    return {
        "arch": arch.name,
        "patch_size": spec.model.patch_size,
        "embedding_dim": arch.embed_dim,
        "depth": arch.depth,
        "input_channels": spec.model.in_chans,
        "image_mean": [round(mean, 6)],
        "image_std": [round(std, 6)],
        "objective": objective_string(spec, gram_enabled),
        "intermediate_layers": last4,
        "crop_schedule": [
            {"global_crops_size": s.global_crops_size, "local_crops_size": s.local_crops_size, "max_steps": s.max_steps}
            for s in spec.crops.schedule
        ],
        # FINO provenance (None/empty for baseline runs).
        "fino_enabled": spec.fino_enabled,
        "fino_factors": fino_factors,
        "fino_lambda_schedule": (
            {"type": spec.fino.lambda_schedule_type, "warmup_iterations": spec.fino.lambda_warmup_steps}
            if spec.fino_enabled
            else None
        ),
        "fino_grad_norm_normalization": bool(spec.fino.grad_norm_normalization) if spec.fino_enabled else False,
        "fino_factors_fingerprint": fino_factors_fingerprint(spec.metadata_factors) if spec.fino_enabled else None,
    }

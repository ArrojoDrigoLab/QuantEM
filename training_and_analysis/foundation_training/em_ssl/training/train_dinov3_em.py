"""DINOv3 single-channel EM training entry (one crop stage).

This is the torchrun target: ``torchrun --nproc_per_node=G -m em_ssl.training.train_dinov3_em
--config <exp.yaml> --output-dir <run_dir> [--stage-index N] [--data-root ...] [--warm-start prev_teacher.pth]``.

It applies the EM patches, dumps the reproducible run manifest + checkpoint index, writes a
resolved DINOv3 config, then delegates to DINOv3's own ``main()``, inheriting its
FSDP2/optimizer/scheduler/resume loop unchanged. ``--dry-run`` validates the whole
setup (config build + 1-channel model on the meta device) on a CPU-only host without training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import yaml

from ..config.schema import load_experiment, resolve_data_paths
from ..fino.factors import FinoRuntime, fino_factors_fingerprint
from ..integration import config_translation as ct
from ..integration import dinov3_patch
from ..utils.checkpoint_index import CheckpointIndex, EncoderManifest, dinov3_feature_entry_point
from .run_manifest import dump_run_manifest

def _config_stage_index(path) -> int:
    """The crop stage a configuration was generated for, recorded in its ``em:`` block."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    em = raw.get("em")
    if isinstance(em, dict) and isinstance(em.get("stage_index"), int):
        return int(em["stage_index"])
    return 0

def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") in ("0", "") and os.environ.get("LOCAL_RANK", "0") in ("0", "")

def _build_overrides_yaml(spec, stage_index, run_dir, seed) -> Path:
    overrides = ct.translate_stage(spec, stage_index, output_dir=str(run_dir), seed=seed)
    path = Path(run_dir) / "resolved_dinov3_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return path

def _gather_fingerprints(spec) -> dict:
    """Best-effort manifest / shard / metadata-coverage fingerprints from the data bundle."""
    import json as _json

    from ..utils.fingerprint import shard_fingerprint
    from .run_manifest import _bundle_manifests_dir

    out = {"manifest_fingerprint": None, "shard_fingerprint": None, "metadata_coverage_fingerprint": None}
    md = _bundle_manifests_dir(spec)
    if md is not None:
        df = md / "dataset_fingerprint.json"
        if df.exists():
            try:
                d = _json.loads(df.read_text(encoding="utf-8"))
                out["manifest_fingerprint"] = d.get("tile_ids_sha256") or (d.get("manifest") or {}).get("sha256")
            except Exception:
                pass
        si = md / "shard_index.json"
        if si.exists():
            try:
                out["shard_fingerprint"] = shard_fingerprint(_json.loads(si.read_text(encoding="utf-8"))).get(
                    "shards_sha256"
                )
            except Exception:
                pass
    if spec.fino.spec_file and Path(spec.fino.spec_file).exists():
        try:
            out["metadata_coverage_fingerprint"] = _json.loads(
                Path(spec.fino.spec_file).read_text(encoding="utf-8")
            ).get("fingerprint")
        except Exception:
            pass
    return out

def _make_checkpoint_index(spec, stage_index, run_dir, run_id) -> CheckpointIndex:
    fields = ct.encoder_manifest_fields(spec, stage_index)
    fp = _gather_fingerprints(spec)
    manifest = EncoderManifest(
        run_id=run_id,
        framework="dinov3",
        objective=fields["objective"],
        arch=fields["arch"],
        patch_size=fields["patch_size"],
        embedding_dim=fields["embedding_dim"],
        depth=fields["depth"],
        input_channels=fields["input_channels"],
        image_mean=fields["image_mean"],
        image_std=fields["image_std"],
        crop_schedule=fields["crop_schedule"],
        feature_entry_point=dinov3_feature_entry_point(
            fields["arch"], fields["depth"], patch_size=fields["patch_size"], in_chans=fields["input_channels"]
        ),
        intermediate_layers=fields["intermediate_layers"],
        config_path=spec.config_path,
        notes=spec.notes,
        # FINO provenance for the later fixed-decoder evaluation.
        fino_enabled=fields["fino_enabled"],
        fino_factors=fields["fino_factors"],
        fino_lambda_schedule=fields["fino_lambda_schedule"],
        fino_grad_norm_normalization=fields["fino_grad_norm_normalization"],
        fino_factors_fingerprint=fields["fino_factors_fingerprint"],
        manifest_fingerprint=fp["manifest_fingerprint"],
        shard_fingerprint=fp["shard_fingerprint"],
        metadata_coverage_fingerprint=fp["metadata_coverage_fingerprint"],
    )
    idx = CheckpointIndex(run_dir, manifest)
    idx.save()
    return idx

def _check_fino_coverage(spec) -> None:
    """Warn loudly (and fail-fast unless overridden) on low metadata coverage for a factor.

    Reads per-factor valid fractions from the coverage spec (em_ssl.tools.fino_metadata_coverage
    writes ``valid_fraction``). When no spec_file is configured, coverage cannot be verified up
    front — the per-batch ``fino/<factor>_frac_valid`` metric still surfaces it during training.
    """
    if not spec.fino.spec_file or not Path(spec.fino.spec_file).exists():
        warnings.warn(
            "FINO: no fino.spec_file configured — metadata coverage not verified before training. "
            "Run em_ssl.tools.fino_metadata_coverage and set fino.spec_file for a fail-fast guard."
        )
        return
    cov = json.loads(Path(spec.fino.spec_file).read_text(encoding="utf-8"))
    fractions = cov.get("valid_fraction", {})
    for f in spec.enabled_factors:
        frac = fractions.get(f.name, fractions.get(f.field))
        if frac is None:
            continue
        if float(frac) < float(spec.fino.min_valid_fraction):
            msg = (
                f"FINO factor '{f.name}': valid metadata fraction {float(frac):.4f} is below "
                f"fino.min_valid_fraction={spec.fino.min_valid_fraction}."
            )
            if spec.fino.allow_low_coverage:
                warnings.warn(msg + " allow_low_coverage=true — continuing.")
            else:
                raise RuntimeError(msg + " Fix coverage or set fino.allow_low_coverage: true to override.")

def _dry_run(overrides_yaml: Path, spec) -> None:
    """Validate the resolved config builds a 1-channel model (meta device, no training).

    For FINO runs, also instantiate the upstream guide heads (CPU) to confirm the translated
    ``guide:`` block is consistent (method, n_outputs, hidden dims).
    """
    import torch
    from omegaconf import OmegaConf
    from dinov3.configs import get_default_config
    from dinov3.models import build_model_from_cfg

    dinov3_patch.apply_em_patches()
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(str(overrides_yaml)))
    with torch.device("meta"):
        student, teacher, embed_dim = build_model_from_cfg(cfg, only_teacher=False)
    cin = tuple(student.patch_embed.proj.weight.shape)[1]
    assert cin == int(cfg.student.in_chans) == 1, f"expected 1-channel stem, got C_in={cin}"
    print(
        f"[dry-run OK] arch={cfg.student.arch} patch={cfg.student.patch_size} in_chans={cin} "
        f"embed_dim={embed_dim} global_crop={cfg.crops.global_crops_size} "
        f"rgb_mean={list(cfg.crops.rgb_mean)} dataset={cfg.train.dataset_path} meta_arch={cfg.MODEL.META_ARCHITECTURE} "
        f"epochs={cfg.optim.epochs} OEL={cfg.train.OFFICIAL_EPOCH_LENGTH} -> max_iter={cfg.optim.epochs*cfg.train.OFFICIAL_EPOCH_LENGTH}"
    )
    if spec.fino_enabled:
        from dinov3.train.metadata_utils import Classifier, Regressor

        summary = []
        for g in cfg.guide.guides:
            in_dim = cfg.dino.head_n_prototypes if (g.grl and g.grl_space == "prototype") else embed_dim
            if g.method == "regression":
                Regressor(input_dim=in_dim, hidden_dim=list(g.hidden_dim), n_outputs=g.n_outputs, dropout=g.dropout)
            elif g.method == "classification":
                Classifier(input_dim=in_dim, hidden_dim=list(g.hidden_dim), num_classes=g.n_outputs, dropout=g.dropout)
            summary.append(f"{g.name}:{g.method}:{'M-' if g.grl else 'M+'}:n={g.n_outputs}:w={g.loss_weight}")
        print(
            f"[dry-run OK] FINO guide heads build: {summary} "
            f"lambda={dict(cfg.guide.lambda_schedule)} grad_norm_norm={cfg.optim.grad_norm_normalization}"
        )

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="DINOv3 single-channel EM training (one stage).")
    p.add_argument("--config", required=True, help="Configuration YAML from configs/.")
    p.add_argument("--output-dir", required=True, help="Run/stage output directory.")
    p.add_argument("--stage-index", type=int, default=None,
                   help="Which crop-schedule stage to run. Defaults to the stage the configuration "
                        "was generated for.")
    p.add_argument("--data-root", default=None, help="Data bundle root (fills shard_dir/stats).")
    p.add_argument("--manifest", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--warm-start", default=None, help="Previous-stage teacher_checkpoint.pth for continuation.")
    p.add_argument("--run-id", default=None)
    p.add_argument("--dry-run", action="store_true", help="Validate config + build model on meta device; do not train.")
    p.add_argument("opts", nargs=argparse.REMAINDER, help="Extra DINOv3 dotlist overrides.")
    args = p.parse_args(argv)

    if args.stage_index is None:
        args.stage_index = _config_stage_index(args.config)
    spec = load_experiment(args.config)
    spec = resolve_data_paths(spec, data_root=args.data_root, manifest=args.manifest)
    spec.validate_fino()
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"{spec.name}_stage{args.stage_index}"

    overrides_yaml = _build_overrides_yaml(spec, args.stage_index, run_dir, args.seed)

    # FINO runtime — built on every rank because each rank's dataloader/collate encodes metadata.
    fino_runtime = None
    if spec.fino_enabled:
        fino_runtime = FinoRuntime(list(spec.metadata_factors))
        _check_fino_coverage(spec)

    if _is_rank0():
        dump_run_manifest(
            run_dir,
            spec,
            extra={
                "run_id": run_id,
                "framework": "dinov3",
                "stage_index": args.stage_index,
                "warm_start": args.warm_start,
                "resolved_dinov3_config": str(overrides_yaml),
                "fino_enabled": spec.fino_enabled,
                "fino_factors": [f.name for f in spec.enabled_factors],
                "fino_guidance": {f.name: f.guidance for f in spec.enabled_factors},
                "fino_factors_fingerprint": (
                    fino_factors_fingerprint(spec.metadata_factors) if spec.fino_enabled else None
                ),
            },
        )
        idx = _make_checkpoint_index(spec, args.stage_index, run_dir, run_id)
        dinov3_patch.ACTIVE_CKPT_INDEX = idx

    dinov3_patch.apply_em_patches()
    dinov3_patch.set_warm_start(args.warm_start)
    if fino_runtime is not None:
        dinov3_patch.set_fino_runtime(fino_runtime)
        from ..fino.meta_arch_patch import apply_fino_grafts

        if not apply_fino_grafts():
            raise RuntimeError(
                "FINO is enabled (metadata_factors present) but the GuidedSSLMetaArch graft is "
                "unavailable — third_party/dinov3 is not the FINO branch. Run third_party/fetch_dinov3.sh "
                "(third_party/dinov3.pin is pinned to the FINO branch)."
            )

    if args.dry_run:
        _dry_run(overrides_yaml, spec)
        return

    # Opt-in bitwise determinism. This is global: an op with no deterministic kernel, such as the
    # flash-attention backward, either raises or takes a slower path, so step time is worth measuring
    # before relying on it. Unrelated to the bf16-atomic compile fallback in em_ssl.integration.dinov3_patch.
    import os

    if os.environ.get("EM_DETERMINISTIC") == "1":
        import torch

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        print("[train_dinov3_em] EM_DETERMINISTIC=1: deterministic algorithms ON "
              "(reproducibility only; for full compile use EM_COMPILE_HEADS=1).")

    # Delegate to DINOv3's main() via sys.argv (avoids its argv[1] output-dir quirk).
    from dinov3.train.train import main as dinov3_main

    new_argv = [sys.argv[0], "--config-file", str(overrides_yaml), "--output-dir", str(run_dir), "--seed", str(args.seed)]
    if args.no_resume:
        new_argv.append("--no-resume")
    if args.opts:
        opts = args.opts[1:] if args.opts and args.opts[0] == "--" else args.opts
        new_argv += list(opts)
    old_argv = sys.argv
    try:
        sys.argv = new_argv
        dinov3_main()
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    main()

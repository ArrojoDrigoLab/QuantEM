"""Entry point: batch-evaluate encoder checkpoints with the fixed decoder probe.

For each experiment run_dir: load checkpoint_index.json, pick N (default 4) evenly-spaced encoder
checkpoints, and for every (checkpoint x organelle x label-fraction) train the fixed decoder on the
encoder — frozen, or adapted when the config sets ``adapt`` — and evaluate on the test split. Emits a
tidy results table (CSV + JSON), a per-crop/per-subgroup JSON, and per-run training logs.

Usage:
    python -m encoder_evaluation.harness.run_probe \
        --run-dir <encoder run dir> [<another encoder run dir> ...] \
        --derived-root <ground-truth tiles> --config encoder_evaluation/configs/probe_upernet_frozen.yaml \
        --organelles mito er --n-checkpoints 4 --output-dir <results dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import replace
from pathlib import Path

import torch

from ..constants import DEFAULT_DERIVED_ROOT, VALID_ORGANELLES
from .config import load_probe_config
from .dataset import load_manifest, subset_fraction
from .encoder_adaptation import apply_adaptation
from .encoders import FrozenEncoder, select_checkpoints
from .evaluate import evaluate_head
from .train import train_head

_TABLE_COLS = [
    "run_dir", "run_id", "framework", "objective", "arch", "embedding_dim", "step", "crop_size",
    "context_tile", "compare_tile",
    "organelle", "decoder", "adapt", "feature_layers", "label_fraction", "n_test_crops", "n_evaluated",
    "n_excluded", "macro_dice", "macro_iou", "macro_boundary_f1", "macro_boundary_iou",
    "micro_dice", "micro_iou", "micro_boundary_f1", "micro_boundary_iou",
]

def _resolve_device(want: str) -> str:
    if want.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA unavailable; falling back to CPU.")
        return "cpu"
    return want

def run(args) -> list[dict]:
    from em_ssl.utils.checkpoint_index import CheckpointIndex

    try:
        from em_ssl.utils.logging import MetricLogger
    except Exception:
        MetricLogger = None  # logging is best-effort

    cfg = load_probe_config(args.config)
    device = _resolve_device(args.device or cfg.device)
    organelles = [o for o in args.organelles if o in VALID_ORGANELLES]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = args.fractions if args.fractions else cfg.label_fractions

    rows: list[dict] = []
    per_crop_dump: list[dict] = []

    for run_dir in args.run_dir:
        run_dir = Path(run_dir)
        try:
            index = CheckpointIndex.load(run_dir)
        except Exception as exc:
            warnings.warn(f"[skip run] {run_dir}: cannot load checkpoint_index.json ({exc!r})")
            continue
        man = index.manifest
        fe = man.feature_entry_point or {}
        man_tile = fe.get("tile_size")
        ctx_tile = getattr(args, "context_tile", None)
        # Per-encoder context window (the encoder input). An encoder takes a larger window when
        # --context-tile is set and it is sweepable, i.e. the RoPE encoders: no manifest tile, or the
        # manifest flag context_sweepable. Learned-position externals (EMCF-MAE, OmniEM, DINOv2) keep
        # their native manifest tile (512 at patch 16, 518 at patch 14); interpolating their position
        # embedding to 1024 is out of distribution, so they are not swept.
        sweepable = bool(fe.get("context_sweepable", man_tile is None))
        enc_tile = int(ctx_tile if (ctx_tile and sweepable) else (man_tile or cfg.tile_size))
        # Common compare region (decoder output and scored pixels), clamped to the context tile.
        # None -> standard probe (no token crop; decoder predicts the whole tile).
        comp = None if cfg.compare_tile is None else min(int(cfg.compare_tile), enc_tile)
        enc_cfg = replace(cfg, tile_size=enc_tile, compare_tile=comp)
        ckpts = select_checkpoints(index, n=args.n_checkpoints, steps=args.steps)
        if not ckpts:
            warnings.warn(f"[skip run] {run_dir}: no teacher/encoder checkpoints in index")
            continue
        layers = cfg.resolved_layers(man.depth)
        print(f"== {run_dir} [{man.framework}/{man.arch}] checkpoints: {[c.step for c in ckpts]}")

        for rec in ckpts:
            if not Path(rec.path).exists():
                warnings.warn(f"[skip ckpt] missing weights: {rec.path}")
                continue
            for organelle in organelles:
                train_all = load_manifest(args.derived_root, organelle, "train")
                test_recs = load_manifest(args.derived_root, organelle, "test")
                if not train_all or not test_recs:
                    warnings.warn(f"[skip] no train/test derived samples for {organelle}")
                    continue
                for frac in fractions:
                    tag = f"{man.run_id}_s{rec.step}_{organelle}_f{frac}"
                    logger = None
                    if MetricLogger is not None:
                        try:
                            logger_obj = MetricLogger(out_dir / "logs" / tag, tensorboard=False)
                            logger = lambda step, m, _l=logger_obj: _l.log(step, m)
                        except Exception:
                            logger = None
                    encoder = FrozenEncoder.from_manifest(
                        rec.path, man, enc_tile, apply_encoder_norm=enc_cfg.apply_encoder_norm,
                    )
                    encoder.compare_tile = enc_cfg.compare_tile  # central-token crop (None disables)
                    # Encoder adaptation, when the config asks for it: install LoRA adapters or unfreeze
                    # encoder base weights (see harness/encoder_adaptation.py for the modes). The encoder
                    # is rebuilt per head above, so each head adapts from the pretrained weights.
                    if str(getattr(enc_cfg, "adapt", "frozen") or "frozen").lower() != "frozen":
                        apply_adaptation(encoder, enc_cfg.adapt, enc_cfg.adapt_params or {})
                    train_recs = subset_fraction(train_all, frac, seed=enc_cfg.seed)
                    print(f"  train {tag}: {len(train_recs)}/{len(train_all)} tiles -> eval {len(test_recs)}")
                    decoder = train_head(encoder, train_recs, enc_cfg, args.derived_root, layers,
                                         device, logger=logger, tag=tag)
                    result = evaluate_head(encoder, decoder, test_recs, enc_cfg, args.derived_root,
                                           layers, device)
                    s = result["summary"]
                    row = {
                        "run_dir": str(run_dir), "run_id": man.run_id, "framework": man.framework,
                        "objective": man.objective, "arch": man.arch,
                        "embedding_dim": man.embedding_dim, "step": rec.step,
                        "crop_size": rec.crop_size, "context_tile": enc_tile,
                        "compare_tile": enc_cfg.effective_compare(),
                        "organelle": organelle, "decoder": cfg.decoder, "adapt": cfg.adapt,
                        "feature_layers": str(cfg.feature_layers), "label_fraction": frac,
                        "n_test_crops": s["n_crops"], "n_evaluated": s["n_evaluated"],
                        "n_excluded": s["n_excluded_both_empty"],
                        "macro_dice": s["macro"]["dice"], "macro_iou": s["macro"]["iou"],
                        "macro_boundary_f1": s["macro"]["boundary_f1"],
                        "macro_boundary_iou": s["macro"]["boundary_iou"],
                        "micro_dice": s["micro"]["dice"], "micro_iou": s["micro"]["iou"],
                        "micro_boundary_f1": s["micro"]["boundary_f1"],
                        "micro_boundary_iou": s["micro"]["boundary_iou"],
                    }
                    rows.append(row)
                    per_crop_dump.append({"tag": tag, **row, "per_subgroup": s["per_subgroup"],
                                          "per_crop": result["per_crop"]})
                    print(f"    -> macro Dice={_fmt(row['macro_dice'])} IoU={_fmt(row['macro_iou'])} "
                          f"bF1={_fmt(row['macro_boundary_f1'])} bIoU={_fmt(row['macro_boundary_iou'])}")
                    del encoder, decoder
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()

    _write_table(out_dir / "results.csv", rows)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "results_detail.json").write_text(json.dumps(per_crop_dump, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} result rows -> {out_dir/'results.csv'}")
    return rows

def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"

def _write_table(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_TABLE_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _TABLE_COLS})

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Batch-evaluate encoder checkpoints with the decoder probe.")
    p.add_argument("--run-dir", nargs="+", required=True, help="One or more SSL run/stage dirs.")
    p.add_argument("--derived-root", default=DEFAULT_DERIVED_ROOT)
    p.add_argument("--config", default=None, help="Probe YAML (defaults if omitted).")
    p.add_argument("--organelles", nargs="+", default=list(VALID_ORGANELLES))
    p.add_argument("--n-checkpoints", type=int, default=4)
    p.add_argument("--steps", type=int, nargs="*", default=None, help="Explicit checkpoint steps.")
    p.add_argument("--fractions", type=float, nargs="*", default=None,
                   help="Label-efficiency fractions (overrides config).")
    p.add_argument("--output-dir", required=True,
                   help="Directory the result tables, per-crop detail and per-run training logs are written to.")
    p.add_argument("--context-tile", type=int, default=None,
                   help="Input tile (context window) for encoders without a manifest tile, i.e. the "
                        "RoPE encoders. Learned-position baselines keep their native manifest tile "
                        "(512 at patch 16, 518 at patch 14). "
                        "Set e.g. 768 or 1024 to let a RoPE encoder use more surrounding context; the scored "
                        "region is unchanged (full-region sliding-window eval scores the same valid pixels).")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()

"""segmentation_training-run-seg — single-arm entrypoint: load config -> encoder -> train head -> eval.

Runs one segmentation arm end to end and writes its run dir (resolved config + provenance + trained
head + val/test metrics). Training covers the neck, the decoder and whatever encoder adaptation the
config selects; the base backbone trains only in the ``last_n`` / ``full`` modes. The experiment
runners under ``experiments/`` and the parallel evaluator (``harness.eval_parallel``) build on this;
each arm runs as an independent process pinned to one device.

Usage:
    python -m segmentation_training.harness.run_seg --config segmentation_training/configs/decoder/<arm>.yaml \
        --data-root <ground-truth tiles> --run-dir <encoder run dir> \
        --output-dir runs/segmentation_training/<arm> --device cuda:0 [--max-steps N] [--step S]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..config.schema import load_seg_config
from ..constants import DEFAULT_DERIVED_ROOT


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device


def _resolve_ckpt_path(rec, run_dir) -> str:
    """Resolve a checkpoint record's file, tolerant of a relocated run-dir.

    ``checkpoint_index.json`` stores the absolute path from the host that trained the encoder. When the
    run-dir is archived and relocated, that path does not resolve and the loader would raise a
    misleading FileNotFoundError. Fall back to the standard on-disk teacher layout under ``run_dir``
    (``eval/training_<step>/<name>``), then the record path's 3-component tail rebased onto
    ``run_dir``, then a basename search — so a transferred run-dir resolves without hand-editing the
    index."""
    p = Path(rec.path)
    if p.exists():
        return str(p)
    run_dir = Path(run_dir)
    name = p.name
    cands = [run_dir / "eval" / f"training_{rec.step}" / name]
    if len(p.parts) >= 3:
        cands.append(run_dir / Path(*p.parts[-3:]))
    for c in cands:
        if c.exists():
            return str(c)
    hits = sorted(run_dir.rglob(name))
    if hits:
        return str(hits[0])
    raise FileNotFoundError(
        f"Encoder checkpoint for step {rec.step} not found. The index's stored path {rec.path!r} is "
        f"absent (the run-dir was likely moved) and no fallback under run_dir {run_dir} matched "
        f"(tried eval/training_{rec.step}/{name}). The teacher_checkpoint.pth file must be present "
        f"under run_dir.")


def resolve_encoder(cfg, device: str):
    """Load the frozen base encoder from cfg.encoder.run_dir's checkpoint_index.json (the pretraining run on
    a real run dir, or a mock one for a CPU smoke test). Returns (encoder, checkpoint_record)."""
    from em_ssl.utils.checkpoint_index import CheckpointIndex

    from .encoders import FrozenEncoder, select_checkpoints

    if not cfg.encoder.run_dir:
        raise ValueError("cfg.encoder.run_dir is unset — point it at the pretraining run (checkpoint_index.json) "
                         "or a mock run dir. It is resolved at runtime, never hardcoded.")
    idx = CheckpointIndex.load(cfg.encoder.run_dir)
    steps = [cfg.encoder.checkpoint_step] if cfg.encoder.checkpoint_step is not None else None
    recs = select_checkpoints(idx, n=1, steps=steps)
    if not recs:
        raise ValueError(f"No loadable encoder checkpoints in {cfg.encoder.run_dir}")
    rec = recs[-1]
    ckpt_path = _resolve_ckpt_path(rec, cfg.encoder.run_dir)
    enc = FrozenEncoder.from_manifest(ckpt_path, idx.manifest, tile_size=cfg.encoder.tile_size,
                                      apply_encoder_norm=cfg.encoder.apply_encoder_norm)
    return enc, rec


# columns of the flat per-split results row (macro values); None where a metric is absent for the arm.
_METRIC_COLS = ("dice", "iou", "precision", "recall", "boundary_f1", "boundary_iou", "hd95", "cldice",
                "auprc", "pq", "sq", "rq", "ap", "vi")


def _flat_rows(cfg, split_summaries: dict) -> list[dict]:
    rows = []
    for split, summ in split_summaries.items():
        macro = summ.get("macro", {})
        row = {"arm": cfg.name, "organelle": cfg.data.organelle, "neck": cfg.neck.type,
               "decoder": cfg.decoder.type, "loss": "+".join(t.type for t in cfg.loss.terms),
               "adapt": getattr(cfg.encoder, "adapt", "frozen"),
               "canonical_nm": cfg.data.resolved_canonical_nm(), "split": split,
               "n_evaluated": summ.get("n_evaluated"), "n_excluded": summ.get("n_excluded_both_empty")}
        for k in _METRIC_COLS:
            row[k] = macro.get(k)
        rows.append(row)
    return rows


def run_arm(cfg, data_root, output_dir, device="cuda", max_steps=None, save_checkpoint=True,
            skip_eval=False, eval_only=False, eval_workers=1, eval_gpus=None) -> dict:
    from em_ssl.utils.reproducibility import seed_everything

    from .dataset import load_manifest
    from .evaluate import evaluate_head
    from .train import train_segmodel
    from ..training.run_manifest import dump_run_manifest

    device = resolve_device(device)
    if max_steps:
        cfg.optim.max_steps = int(max_steps)
    seed_everything(cfg.optim.seed)

    enc, rec = resolve_encoder(cfg, device)
    enc.to(device)
    group = cfg.data.resolved_group()
    bucket = getattr(cfg.data, "bucket", "canonical")  # canonical (per-organelle canonical nm/px) | native (source resolution)
    mname = getattr(cfg.data, "manifest_name", "manifest.jsonl")  # override for a rebalanced manifest
    train_recs = load_manifest(data_root, group, cfg.data.train_split, bucket=bucket, manifest_name=mname)
    # LOSO-CV: pull the held-out sources out of train and score them as a 'loso' fold below —
    # true leave-one-source-out cross-source generalization, measured without that source in training.
    holdout = set(str(s) for s in (getattr(cfg.data, "holdout_sources", []) or []))
    loso_recs = [r for r in train_recs if str(r.get("dataset")) in holdout] if holdout else []
    if holdout:
        train_recs = [r for r in train_recs if str(r.get("dataset")) not in holdout]
        print(f"[LOSO] held out {sorted(holdout)} from train ({len(loso_recs)} crops -> 'loso' fold).")
    # Diversity-vs-volume gate: optionally subset the train pool by #sources then #crops/source.
    # No-op at the defaults (1.0). The gate touches the train pool only, so every arm in the sweep is
    # scored on the same val/test records.
    sfrac = float(getattr(cfg.data, "source_frac", 1.0))
    vfrac = float(getattr(cfg.data, "subset_frac", 1.0))
    if sfrac < 1.0 or vfrac < 1.0:
        from .dataset import subset_fraction, subset_sources
        sseed = int(getattr(cfg.data, "subset_seed", 0))
        if sfrac < 1.0:
            train_recs = subset_sources(train_recs, sfrac, seed=sseed)
        if vfrac < 1.0:
            train_recs = subset_fraction(train_recs, vfrac, seed=sseed)
    val_recs = load_manifest(data_root, group, cfg.data.val_split, bucket=bucket, manifest_name=mname)
    test_recs = load_manifest(data_root, group, cfg.data.test_split, bucket=bucket, manifest_name=mname)

    run_dir = Path(output_dir)
    dump_run_manifest(run_dir, cfg, extra={"encoder_step": rec.step, "n_train": len(train_recs),
                                           "n_val": len(val_recs), "n_test": len(test_recs)})
    if not train_recs and not eval_only:
        raise ValueError(f"No training samples for {cfg.name} ({group}/{cfg.data.train_split})")

    eval_workers = int(eval_workers)
    # Parallel eval (segmentation_training.harness.eval_parallel): shard the per-region loop across GPU
    # workers — identical numbers, ~N x faster. Valid for non-image-style conditioning arms; image-style
    # conditioning derives dataset-scope style codes and therefore needs all records in one process.
    use_parallel = eval_workers > 1 and not getattr(getattr(cfg, "cond", None), "enabled", False)
    eval_gpu_list = [int(g) for g in eval_gpus] if eval_gpus else [0]

    if eval_only:
        # Re-evaluate from a saved head.pt (no training). Build the model once here — used for small
        # splits (serial) and as the fallback; large splits fan out to per-worker rebuilds (parallel).
        from .load_adapted import build_and_load_head
        model, _, _ = build_and_load_head(cfg, enc, run_dir / "head.pt", device=device)
        print(f"[{cfg.name}] eval-only from head.pt (parallel-capable={use_parallel}, workers={eval_workers})")
    else:
        model = train_segmodel(cfg, enc, train_recs, data_root, device, tag=cfg.name, run_dir=run_dir)
        if save_checkpoint:
            import torch
            # Trainable encoder params = LoRA adapters and/or unfrozen base blocks (the encoder-adaptation
            # experiment's last_n/full modes). Saving them generically by name keeps head.pt complete for
            # every adapt mode, covering unfrozen base weights as well as the conv_lora adapters.
            enc_trainable = {n: p.detach().cpu()
                             for n, p in model.encoder.named_parameters() if p.requires_grad}
            # image-style conditioning: persist the conditioner (style encoder + FiLM heads + adversary) + its metadata vocab so
            # the head.pt is self-contained for TTA / re-eval. None for non-image-style conditioning arms.
            cond = getattr(model, "conditioner", None)
            vocab = getattr(model, "_meta_vocab", None)
            torch.save({"neck": model.neck.state_dict(), "decoder": model.decoder.state_dict(),
                        "encoder_trainable": enc_trainable or None,
                        "adapters": (model.encoder._conv_lora.state_dict()
                                     if getattr(model.encoder, "_conv_lora", None) is not None else None),
                        "conditioner": (cond.state_dict() if cond is not None else None),
                        "meta_vocab": (vocab.to_dict() if vocab is not None else None)},
                       run_dir / "head.pt")
        if skip_eval:  # train-only mode — eval runs as a separate parallel pass (--eval-only)
            print(f"[{cfg.name}] train-only done (head.pt saved) -> {run_dir}")
            return {"arm": cfg.name, "encoder_step": rec.step, "splits": {}, "train_only": True}

    # Held-out-image split: scored automatically when the (rebalanced) manifest
    # carries test_image crops; empty/absent otherwise -> skipped.
    test_image_recs = load_manifest(data_root, group, "test_image", bucket=bucket, manifest_name=mname)
    split_summaries: dict = {}
    per_crop: dict = {}
    eval_splits = [("val", val_recs), ("test", test_recs)]
    if test_image_recs:
        eval_splits.append(("test_image", test_image_recs))
    if loso_recs:  # LOSO fold: the held-out source(s), scored by a model trained without them
        eval_splits.append(("loso", loso_recs))
    mec = int(getattr(cfg.eval, "max_eval_crops", 0) or 0)  # stratified eval subsample (0 = full set)
    for split, recs in eval_splits:
        if not recs:
            continue
        if mec and len(recs) > mec:
            from .dataset import subset_fraction
            recs = subset_fraction(recs, mec / len(recs), seed=cfg.optim.seed)
        # Parallelize only splits big enough to amortize the per-worker model rebuild (>=24 crops);
        # small splits (e.g. val) run serially on the already-built model — identical numbers either way.
        if use_parallel and len(recs) >= 24:
            from .eval_parallel import parallel_evaluate
            out = parallel_evaluate(cfg, run_dir, data_root, recs,
                                    n_workers=eval_workers, gpus=eval_gpu_list)
        else:
            out = evaluate_head(model, recs, cfg, data_root, device,
                                mean=enc.image_mean, std=enc.image_std)
        split_summaries[split] = out["summary"]
        per_crop[split] = out.get("per_crop", [])

    results = {"arm": cfg.name, "encoder_step": rec.step, "splits": split_summaries}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (run_dir / "results_per_crop.json").write_text(json.dumps(per_crop, default=str), encoding="utf-8")
    rows = _flat_rows(cfg, split_summaries)
    if rows:
        with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[{cfg.name}] done -> {run_dir}")
    return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run one segmentation arm (frozen encoder + head).")
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default=DEFAULT_DERIVED_ROOT)
    p.add_argument("--output-dir", default=None, help="Default: runs/segmentation_training/<config name>.")
    p.add_argument("--run-dir", default=None, help="Override cfg.encoder.run_dir (the pretraining run).")
    p.add_argument("--step", type=int, default=None, help="Override the encoder checkpoint step.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-max-region-px", type=int, default=None,
                   help="Central-crop test regions to <= this many px before eval (bounds cost on huge "
                        "crops; uniform -> fair ranking). 0/omit = full region. Set cfg.eval.max_region_px.")
    p.add_argument("--seed", type=int, default=None, help="Override cfg.optim.seed (repeat an arm under "
                                                          "several seeds; the seed statistics need at "
                                                          "least 3 to report a confidence interval).")
    p.add_argument("--skip-eval", action="store_true", help="Train + save head.pt, then exit without "
                   "evaluating (train-only mode; eval runs later as a separate parallel --eval-only pass).")
    p.add_argument("--eval-only", action="store_true", help="Skip training: load head.pt from --output-dir "
                   "and evaluate (re-eval a trained arm). Pair with --eval-workers/--eval-gpus for parallel eval.")
    p.add_argument("--eval-workers", type=int, default=1, help="Parallel eval workers (shard the per-region "
                   "loop across GPUs; identical numbers). 1 = serial (default). Non-image-style conditioning arms only.")
    p.add_argument("--eval-gpus", default=None, help="Comma-separated GPU ids the parallel eval may use "
                   "(e.g. '0,1,2,3'). Default: GPU 0. CUDA_VISIBLE_DEVICES must be unset when this is used.")
    args = p.parse_args(argv)

    cfg = load_seg_config(args.config)
    if args.run_dir:
        cfg.encoder.run_dir = args.run_dir
    if args.step is not None:
        cfg.encoder.checkpoint_step = args.step
    if args.eval_max_region_px is not None:
        cfg.eval.max_region_px = int(args.eval_max_region_px)
    if args.seed is not None:
        cfg.optim.seed = int(args.seed)
    out = args.output_dir or str(Path("runs") / "segmentation_training" / cfg.name)
    eval_gpus = [int(g) for g in args.eval_gpus.split(",")] if args.eval_gpus else None
    run_arm(cfg, args.data_root, out, device=args.device, max_steps=args.max_steps,
            skip_eval=args.skip_eval, eval_only=args.eval_only,
            eval_workers=args.eval_workers, eval_gpus=eval_gpus)


if __name__ == "__main__":
    main()

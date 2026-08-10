"""Input-scale runner and config generator.

Three mechanisms:
  * Scale sweep, standardised versus native. Ordinary training arms: one recipe, one
    ``--data-root`` per rescaled dataset. ``gen-configs`` writes those configs plus the map from arm
    name to data root; the arms themselves then run through ``harness.run_seg``.
  * Multi-scale test-time fusion (``multiscale``): evaluate a trained head over several rescalings
    of each test region and fuse the probability maps. No retraining.
  * Two-scale co-centered input (``two-scale``): train and evaluate a model that feeds a fine view
    and a co-centered coarse view through the shared encoder.

Config generation uses the shared adapted-base template, so every scale arm is matched to the baseline
on steps, seed, adaptation and evaluation settings and only the input scale differs.
``data.canonical_nm`` records the target scale; ``data.bucket`` is ``native`` for the native arm.

The rescaled datasets themselves are built by the segmentation dataset builder, once per target scale,
holding its geometry settings fixed so that only nm/px varies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.config_templates import base_config, with_overrides, write_config

# The reported scale grid.
#
# ER omits 1 nm/px: the entire ER test set is upsampled below 4.68 nm/px, because the only sources
# finer than that are train-only, so a 1 nm arm would measure interpolation rather than resolution.
# Mitochondria keep 4 nm/px, where a quarter of the test set is natively at or below that scale.
#
# The grid includes the 8 nm/px ER point so this generator emits the full reported set.
SCALE_GRID = {"er": [2.0, 4.0, 8.0], "mito": [4.0, 8.0, 16.0]}


def _variant_dir(scale_root: str, organelle: str, nm: float | None, native: bool) -> str:
    if native:
        return str(Path(scale_root) / f"{organelle}_native")
    return str(Path(scale_root) / f"{organelle}_{('%g' % nm).replace('.', 'p')}nm")


def _subsample_eval(recs, cfg, max_eval_crops=None):
    """Cap evaluation records the same way ``harness.run_seg`` does — stratified by dataset and seeded.

    Without a cap the custom-evaluation scale arms run over the full test split; the mitochondria
    variants are around 10,000 regions and each runs mutex watershed, which takes days.
    ``subset_fraction`` is seed-stable, so every arm at the same cap and seed sees the same regions.
    """
    from ...harness.dataset import subset_fraction
    mec = max_eval_crops if max_eval_crops is not None else int(getattr(cfg.eval, "max_eval_crops", 0) or 0)
    mec = int(mec or 0)
    if mec and len(recs) > mec:
        seed = int(getattr(cfg.optim, "seed", 0) or 0)
        recs = subset_fraction(recs, mec / len(recs), seed=seed)
        print(f"[scale] eval subsample -> {len(recs)} regions (max_eval_crops={mec}, seed={seed})", flush=True)
    return recs


def gen_scale_configs(out_dir: str | Path, scale_root: str, organelles=("er", "mito"),
                      include_native=True, max_eval_crops: int = 300) -> dict:
    """Write one config per (organelle, scale) plus a native arm; return the arm to data-root map."""
    out_dir = Path(out_dir)
    data_roots: dict[str, str] = {}
    for org in organelles:
        base = base_config(org)
        for nm in SCALE_GRID[org]:
            name = f"scale_{org}_{('%g' % nm).replace('.', 'p')}nm"
            cfg = with_overrides(base, name=name, **{"data.canonical_nm": float(nm),
                                 "eval.max_eval_crops": int(max_eval_crops), "notes":
                                 f"Scale sweep: {org} at {nm} nm/px (--data-root selects the dataset)."})
            write_config(cfg, out_dir / f"{name}.yaml")
            data_roots[name] = _variant_dir(scale_root, org, nm, native=False)
        if include_native:
            name = f"scale_{org}_native"
            cfg = with_overrides(base, name=name, **{"data.bucket": "native", "data.canonical_nm": None,
                                 "eval.max_eval_crops": int(max_eval_crops),
                                 "notes": f"Native-resolution arm: {org} at source nm/px (bucket=native)."})
            write_config(cfg, out_dir / f"{name}.yaml")
            data_roots[name] = _variant_dir(scale_root, org, None, native=True)
    (out_dir / "data_roots.json").write_text(json.dumps(data_roots, indent=2), encoding="utf-8")
    print(f"[gen] wrote {sum(len(SCALE_GRID[o]) for o in organelles) + (len(organelles) if include_native else 0)}"
          f" scale configs + data_roots.json -> {out_dir}")
    return data_roots


def run_multiscale(organelle: str, *, scales, fuse, split, data_root, device, run_dir, head,
                   config, out_dir=None, weights=None, max_eval_crops=None) -> dict:
    """Evaluate a trained head with multi-scale test-time fusion over one split; write the report."""
    from ...harness.dataset import load_manifest
    from ..common.base_model import load_adapted_base
    from ..common.eval_report import assemble_report, write_report
    from .multiscale import evaluate_multiscale

    base = load_adapted_base(organelle, head=head, config=config, run_dir=run_dir, device=device)
    group = base.cfg.data.resolved_group()
    # Infer the resolution bucket from the data-root name: a ``*_native`` variant carries native-bucket
    # regions, everything else canonical. Without this a native data root filters to zero regions.
    bucket = "native" if str(data_root).rstrip("/\\").endswith("_native") else getattr(base.cfg.data, "bucket", "canonical")
    base.cfg.data.bucket = bucket
    mname = getattr(base.cfg.data, "manifest_name", "manifest.jsonl")
    recs = load_manifest(data_root, group, split, bucket=bucket, manifest_name=mname)
    recs = _subsample_eval(recs, base.cfg, max_eval_crops)
    out = evaluate_multiscale(base.model, recs, base.cfg, data_root, base.device, base.mean, base.std,
                              scales=scales, fuse=fuse, weights=weights)
    report = assemble_report(f"multiscale_{organelle}_{fuse}", organelle, {split: out},
                             extra={"scales": list(scales), "fuse": fuse, "infer_cost": f"{len(scales)}x base"})
    if out_dir:
        write_report(report, out_dir, arm=f"multiscale_{organelle}_{fuse}")
    return report


def run_two_scale(organelle: str, *, fuse, coarse_factor, data_root, device, run_dir,
                  splits=("test_image", "test"), out_dir=None, max_steps=None, seed=0,
                  max_eval_crops=None, batch_size=None) -> dict:
    """Train then evaluate the two-scale model on ``data_root``; write the report."""
    from ...config.schema import SegConfig
    from ...harness.dataset import load_manifest
    from ...harness.run_seg import resolve_device, resolve_encoder
    from ..common.config_templates import base_config
    from ..common.eval_report import assemble_report, write_report
    from .two_scale import evaluate_two_scale, train_two_scale

    cfg = SegConfig.from_dict(base_config(organelle, name=f"twoscale_{organelle}", seed=seed))
    cfg.encoder.run_dir = str(run_dir)
    if max_steps:
        cfg.optim.max_steps = int(max_steps)
    # The two-scale model runs the encoder twice per step (fine and coarse branch) with the adapter
    # graph retained, so it needs roughly twice the activation memory of a single-scale arm. A smaller
    # batch keeps it inside a 24 GB card; gradient accumulation keeps the effective batch matched.
    if batch_size:
        cfg.optim.batch_size = int(batch_size)
    device = resolve_device(device)
    enc, _ = resolve_encoder(cfg, device); enc.to(device)
    group = cfg.data.resolved_group()
    # Honour the arm's resolution bucket and manifest name exactly as run_seg does, so a native-bucket
    # data root does not filter to zero training records under the canonical default.
    bucket = "native" if str(data_root).rstrip("/\\").endswith("_native") else getattr(cfg.data, "bucket", "canonical")
    cfg.data.bucket = bucket
    mname = getattr(cfg.data, "manifest_name", "manifest.jsonl")
    train_recs = load_manifest(data_root, group, cfg.data.train_split, bucket=bucket, manifest_name=mname)
    model = train_two_scale(cfg, enc, train_recs, data_root, device, fuse=fuse, coarse_factor=coarse_factor)
    model._fuse = fuse
    split_results = {}
    for sp in splits:
        recs = load_manifest(data_root, group, sp, bucket=bucket, manifest_name=mname)
        recs = _subsample_eval(recs, cfg, max_eval_crops)
        if recs:
            split_results[sp] = evaluate_two_scale(model, recs, cfg, data_root, device, enc.image_mean,
                                                   enc.image_std, coarse_factor=coarse_factor)
    report = assemble_report(f"twoscale_{organelle}_{fuse}_cf{coarse_factor}", organelle, split_results,
                             extra={"fuse": fuse, "coarse_factor": coarse_factor, "seed": seed})
    if out_dir:
        write_report(report, out_dir, arm=f"twoscale_{organelle}_{fuse}_cf{coarse_factor}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Input-scale experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = dict(type=int, default=300,
               help="Evaluation subsample cap (stratified and seeded); 0 evaluates the full split.")

    g = sub.add_parser("gen-configs", help="Write the scale-sweep and native configs plus the data-root map.")
    g.add_argument("--out-dir", default=str(Path(__file__).parent / "configs"))
    g.add_argument("--scale-root", required=True, help="Root holding one rescaled dataset per arm.")

    m = sub.add_parser("multiscale", help="Multi-scale test-time fusion over a trained head.")
    m.add_argument("--organelle", required=True, choices=["er", "mito"])
    m.add_argument("--data-root", required=True)
    m.add_argument("--head", required=True, help="head.pt from the trained baseline run.")
    m.add_argument("--config", required=True, help="resolved_config.yaml from the same run.")
    m.add_argument("--run-dir", required=True, help="Encoder run directory.")
    m.add_argument("--scales", nargs="+", type=float, default=[0.75, 1.0, 1.5])
    m.add_argument("--fuse", default="mean", choices=["mean", "max", "wmean"])
    m.add_argument("--split", default="test")
    m.add_argument("--device", default="cuda")
    m.add_argument("--out-dir", default=None)
    m.add_argument("--max-eval-crops", **cap)

    co = sub.add_parser("composition", help="Report the native-resolution composition of each arm's split.")
    co.add_argument("--data-roots", default=str(Path(__file__).parent / "configs" / "data_roots.json"))
    co.add_argument("--split", default="test")
    co.add_argument("--out", default=None)

    t = sub.add_parser("two-scale", help="Train and evaluate the two-scale co-input model.")
    t.add_argument("--organelle", required=True, choices=["er", "mito"])
    t.add_argument("--data-root", required=True)
    t.add_argument("--run-dir", required=True, help="Encoder run directory.")
    t.add_argument("--fuse", default="xattn", choices=["xattn", "concat"])
    t.add_argument("--coarse-factor", type=int, default=2)
    t.add_argument("--device", default="cuda")
    t.add_argument("--out-dir", default=None)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--batch-size", type=int, default=None,
                   help="Override the training batch size. The two-scale model runs the encoder twice "
                        "per step, so a 24 GB card may need a smaller batch.")
    t.add_argument("--max-eval-crops", **cap)

    a = p.parse_args(argv)
    if a.cmd == "gen-configs":
        gen_scale_configs(a.out_dir, scale_root=a.scale_root)
    elif a.cmd == "multiscale":
        r = run_multiscale(a.organelle, scales=a.scales, fuse=a.fuse, split=a.split, data_root=a.data_root,
                           device=a.device, run_dir=a.run_dir, head=a.head, config=a.config,
                           out_dir=a.out_dir, max_eval_crops=a.max_eval_crops)
        print(json.dumps(r.get("splits", {}), indent=2, default=str)[:2000])
    elif a.cmd == "composition":
        from .scale_report import main as comp_main
        comp_main(["--data-roots", a.data_roots, "--split", a.split] + (["--out", a.out] if a.out else []))
    elif a.cmd == "two-scale":
        r = run_two_scale(a.organelle, fuse=a.fuse, coarse_factor=a.coarse_factor,
                          data_root=a.data_root, device=a.device, run_dir=a.run_dir, out_dir=a.out_dir,
                          max_steps=a.max_steps, seed=a.seed, max_eval_crops=a.max_eval_crops,
                          batch_size=a.batch_size)
        print(json.dumps(r.get("splits", {}), indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()

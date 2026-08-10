"""Parallel segmentation evaluation — shard records across GPU workers, each running the exact
``evaluate_head`` per-region loop, then merge per-crop in order and aggregate once.

Comparability (the summary is numerically identical to a serial ``evaluate_head`` summary):
  * Each worker rebuilds the same model from ``head.pt`` (``build_and_load_head``) and calls the same
    ``predict_region`` + ``per_crop_metrics`` + ``dual_instance_metrics`` on its record-shard — the eval
    forward is ``@torch.no_grad`` with no dropout, so per-crop dicts are identical to serial.
  * Shards are contiguous record ranges merged in shard order, so the per-crop list is in the same order
    as the serial run; the seeded bootstrap in ``aggregate`` therefore yields identical CIs.
  * Only valid when ``cfg.cond.enabled`` is False. Image-style conditioning estimates dataset-scope
    style codes from all of a source's records, so sharding would change them, and the caller falls
    back to serial evaluation.

The eval bottleneck is per-region CPU metrics (distance transforms + skeletonize + mutex-watershed on
~7 MP regions), which is embarrassingly parallel across regions. On a many-core machine this shortens
a single-core evaluation substantially without changing any reported number.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path


def _shard_bounds(n_items: int, n_shards: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal ranges covering ``[0, n_items)`` — order-preserving when merged in order."""
    return [(round(i * n_items / n_shards), round((i + 1) * n_items / n_shards)) for i in range(n_shards)]


def _build_model(cfg, output_dir, device):
    """Rebuild the trained SegModel from ``head.pt`` (frozen encoder + neck/decoder/adapters)."""
    from .load_adapted import build_and_load_head
    from .run_seg import resolve_encoder
    enc, _ = resolve_encoder(cfg, device)
    enc.to(device)
    model, _, _ = build_and_load_head(cfg, enc, Path(output_dir) / "head.pt", device=device)
    return model, enc


def _worker(payload):
    cfg, output_dir, data_root, recs, gpu_id = payload
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # set before any CUDA init (spawn => fresh process)
    from .evaluate import evaluate_head
    from .run_seg import resolve_device
    device = resolve_device("cuda")
    model, enc = _build_model(cfg, output_dir, device)
    out = evaluate_head(model, recs, cfg, data_root, device,
                        mean=enc.image_mean, std=enc.image_std, do_aggregate=False)
    return out["per_crop"]


def parallel_evaluate(cfg, output_dir, data_root, recs, *, n_workers: int, gpus: list[int]) -> dict:
    """Evaluate ``recs`` across ``n_workers`` GPU-worker processes (round-robin over ``gpus``); merge + aggregate.

    Returns ``{"summary": <aggregate>, "per_crop": [...]}`` — identical to serial ``evaluate_head(..., do_aggregate=True)``.
    """
    from .metrics import aggregate
    n = max(1, min(int(n_workers), len(recs)))
    bounds = _shard_bounds(len(recs), n)
    payloads = [(cfg, output_dir, data_root, recs[a:b], gpus[i % len(gpus)])
                for i, (a, b) in enumerate(bounds)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n) as pool:
        shard_per_crop = pool.map(_worker, payloads)  # Pool.map preserves order => record order preserved
    per_crop = [m for shard in shard_per_crop for m in shard]
    summary = aggregate(per_crop, bootstrap_n=int(cfg.eval.bootstrap_n),
                        ci=float(cfg.eval.bootstrap_ci), seed=int(cfg.optim.seed))
    return {"summary": summary, "per_crop": per_crop}

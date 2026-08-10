"""Honest evaluation: sliding-window inference over the FULL annotated region, then metrics.

Val/test crops are evaluated at their stored resolution (EM is never resized — physical nm/px is
meaningful) by tiling into ``tile_size`` windows with overlap, blending probabilities with a cosine
(Hann) window, and scoring only valid (non-ignore) pixels. No cherry-picking, no per-tile selection.

Two speed properties (always safe — eval is deterministic, no augmentation):
  * windows are forwarded through the encoder in batches (``cfg.eval_batch_windows``), not 1 at a time;
  * one crop's window features are computed once and shared across every decoder being evaluated
    (``predict_regions`` / ``evaluate_heads``), so the encoder is forwarded once per (checkpoint,
    organelle, test set) regardless of how many decoder heads / label fractions are scored.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .dataset import load_inst, load_sample, normalize_em
from .eval_metrics import cfg_fields, crop_metrics, crop_task
from .metrics import aggregate

def _hann2d(t: int) -> np.ndarray:
    w = np.hanning(t).astype(np.float32)
    win = np.outer(w, w)
    return win + 1e-3  # floor so region corners (single-window coverage) get non-zero weight

def _window_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts

@torch.no_grad()
def predict_regions(encoder, decoders: dict, em: np.ndarray, cfg, layers, device) -> dict:
    """Foreground probability map per decoder over a full EM region, sharing one encoder pass.

    ``decoders`` maps name -> SegDecoder. Windows are batched (``cfg.eval_batch_windows``); each batch is
    forwarded through the frozen encoder once and every decoder runs on those shared features.

    Context handling: the output window, and so the blending and scoring, is the common compare region
    ``t_cmp = cfg.effective_compare()``. The encoder input is the larger context window
    ``t_ctx = cfg.tile_size`` centred on it — real surrounding EM, reflect-padded only at the true region
    edge. The encoder crops its central tokens back to the compare grid (``encoder.compare_tile``), so
    every encoder scores identical pixels while the RoPE encoders attend to more context. This reduces to
    the plain single-tile sliding window when ``t_cmp == t_ctx``."""
    t_ctx = int(cfg.tile_size)                    # encoder input (context window)
    t_cmp = int(cfg.effective_compare())          # output window == scored/blended region (<= t_ctx)
    m = max(0, (t_ctx - t_cmp) // 2)              # context margin around each output window
    H0, W0 = em.shape
    # (a) ensure the region is at least one compare window; (b) add the context margin so a t_ctx crop is
    #     valid for every output window (real EM inside the tile, reflect only past the true edge).
    ph, pw = max(t_cmp - H0, 0), max(t_cmp - W0, 0)
    em1 = np.pad(em, ((0, ph), (0, pw)), mode="reflect" if (H0 > 1 and W0 > 1) else "constant")
    H1, W1 = em1.shape
    em_pad = (np.pad(em1, ((m, m), (m, m)), mode="reflect" if (H1 > 1 and W1 > 1) else "constant")
              if m > 0 else em1)
    xnorm = normalize_em(em_pad, encoder.image_mean, encoder.image_std)
    stride = max(1, int(round(t_cmp * (1 - cfg.eval_overlap))))
    win = _hann2d(t_cmp)
    starts = [(y, x) for y in _window_starts(H1, t_cmp, stride) for x in _window_starts(W1, t_cmp, stride)]

    accs = {n: np.zeros((H1, W1), np.float32) for n in decoders}
    wsum = np.zeros((H1, W1), np.float32)
    bs = max(1, int(cfg.eval_batch_windows))
    for i in range(0, len(starts), bs):
        chunk = starts[i:i + bs]
        # output window (y,x) in em1 coords -> its t_ctx context is em_pad[y:y+t_ctx] (the +m shift makes
        # this exactly the t_cmp window expanded by the margin on all sides).
        batch = np.stack([xnorm[y:y + t_ctx, x:x + t_ctx] for (y, x) in chunk])  # [b,t_ctx,t_ctx]
        xt = torch.from_numpy(batch)[:, None].to(device)  # [b,1,t_ctx,t_ctx]
        feats = encoder.extract(xt, layers)  # central-token-cropped to the compare grid; shared across decoders
        ff = [f.float() for f in feats]
        for name, dec in decoders.items():
            probs = torch.softmax(dec(ff, out_hw=(t_cmp, t_cmp)), dim=1)[:, 1].cpu().numpy()  # [b,t_cmp,t_cmp]
            acc = accs[name]
            for j, (y, x) in enumerate(chunk):
                acc[y:y + t_cmp, x:x + t_cmp] += probs[j] * win
        for (y, x) in chunk:
            wsum[y:y + t_cmp, x:x + t_cmp] += win
    inv = 1.0 / np.maximum(wsum, 1e-6)
    return {n: (accs[n] * inv)[:H0, :W0] for n in decoders}

# Persistent metric pool. The GPU forward stays sequential — one shared encoder pass per crop —
# while the CPU-bound per-crop metrics (distance transforms, instance PQ/AP/VI) fan out during the
# next forward. Threads rather than processes: scipy.ndimage releases the GIL in its C loops, so
# threads parallelize the heavy work and share memory instead of pickling large arrays.
_EVAL_POOL = None

def _get_eval_pool(n_workers: int):
    global _EVAL_POOL
    if n_workers <= 1:
        return None
    if _EVAL_POOL is None:
        _EVAL_POOL = ThreadPoolExecutor(max_workers=n_workers)
    return _EVAL_POOL

def shutdown_eval_pool():
    """Tear the thread pool down cleanly (no atexit surprises)."""
    global _EVAL_POOL
    if _EVAL_POOL is not None:
        try:
            _EVAL_POOL.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        _EVAL_POOL = None

def evaluate_heads(encoder, decoders: dict, records, cfg, derived_root, layers, device,
                   on_crop=None) -> dict:
    """Evaluate several decoders on the same test set, sharing one encoder pass per crop.

    Per-crop metrics are computed on a thread pool (``cfg.eval_workers``) — bit-for-bit identical to
    serial, since numpy and scipy are deterministic here, just parallel across cores. Returns
    ``{name: {"summary":..., "per_crop":[...]}}``.
    """
    encoder = encoder.to(device)
    decoders = {n: d.to(device).eval() for n, d in decoders.items()}
    names = list(decoders)
    per_crop = {n: [None] * len(records) for n in names}  # by index -> stays in record order
    n_rec = len(records)
    cf = cfg_fields(cfg)
    n_workers = max(1, min(int(getattr(cfg, "eval_workers", 1) or 1), os.cpu_count() or 1))
    pool = _get_eval_pool(n_workers)

    def _predict(r):
        em, mask = load_sample(r, derived_root)
        if r.get("organelle") == "mito":
            r = {**r, "_gt_inst": load_inst(r, derived_root)}
        return predict_regions(encoder, decoders, em, cfg, layers, device), mask, r

    if pool is None:  # serial fallback (eval_workers<=1)
        for i, r in enumerate(records):
            probs, mask, r = _predict(r)
            for name in names:
                per_crop[name][i] = crop_metrics(probs[name], mask, r, cf)
            if on_crop is not None:
                on_crop(i + 1, n_rec)
    else:
        inflight = deque()
        max_inflight = 3 * n_workers  # backpressure so in-flight prediction arrays can't blow RAM
        done = [0]

        def _collect():
            i, fut = inflight.popleft()
            res = fut.result()
            for name in names:
                per_crop[name][i] = res[name]
            done[0] += 1
            if on_crop is not None:
                on_crop(done[0], n_rec)

        for i, r in enumerate(records):
            probs, mask, r = _predict(r)  # GPU forward — sequential
            inflight.append((i, pool.submit(crop_task, ({n: probs[n] for n in names}, mask, r, cf))))
            while len(inflight) > max_inflight:
                _collect()
        while inflight:
            _collect()

    return {n: {"summary": aggregate(per_crop[n], bootstrap_n=cfg.bootstrap_n, ci=cfg.bootstrap_ci,
                                     seed=cfg.seed), "per_crop": per_crop[n]} for n in names}

def evaluate_head(encoder, decoder, records, cfg, derived_root, layers, device, on_crop=None) -> dict:
    """Single-decoder convenience wrapper over :func:`evaluate_heads`."""
    return evaluate_heads(encoder, {"_": decoder}, records, cfg, derived_root, layers, device,
                          on_crop=on_crop)["_"]

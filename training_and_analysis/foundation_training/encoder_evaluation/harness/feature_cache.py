"""Opt-in frozen-feature caching for fast decoder training (``cfg.cache_train_features``).

Because the encoder is frozen, its features for a given tile are fixed. This module forwards the
encoder once per (checkpoint, organelle) over a fixed grid tiling of the train set, caches the
(concatenated multi-layer) features on the GPU when they fit there and in host RAM otherwise
(``cfg.cache_on_gpu``), and then trains every decoder + label fraction on the cache — the encoder is
never re-forwarded. For the small linear/light_conv heads, where the encoder forward dominates the
cost, this is the main speed lever.

Trade-off: the cache is a *fixed* grid tiling, so it drops the per-step image-space augmentation
(random crop / flip / intensity). That is acceptable for these low-capacity heads (a linear probe
barely augments), and it is the standard fast-linear-probe recipe. The default (non-cached) path
provides full per-step augmentation.
"""

from __future__ import annotations

import math
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..constants import IGNORE_INDEX
from .dataset import load_sample, normalize_em, read_png_L
from .decoder import build_decoder
from .evaluate import _window_starts
from .train import _lr_at, soft_dice_loss

def _enumerate_valid_tiles(records, derived_root, tile):
    """``(record_idx, y, x)`` for every valid (non all-ignore) non-overlapping tile.

    Loads masks only — no EM, no encoder forward — so the tiles to cache are picked before
    paying to compute their features. This is what lets the build forward only the kept `cap` tiles
    instead of every enumerated window.
    """
    out = []
    droot = Path(derived_root)
    for ri, r in enumerate(records):
        mask = read_png_L(droot / r["mask_path"])
        H0, W0 = mask.shape
        ph, pw = max(tile - H0, 0), max(tile - W0, 0)
        if ph or pw:
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=IGNORE_INDEX)
        H, W = mask.shape
        for y in _window_starts(H, tile, tile):
            for x in _window_starts(W, tile, tile):
                if int((mask[y:y + tile, x:x + tile] != IGNORE_INDEX).sum()) > 0:
                    out.append((ri, y, x))
    return out

@torch.no_grad()
def build_train_cache(encoder, records, cfg, derived_root, layers, device) -> dict:
    """Forward the frozen encoder over a fixed grid tiling of ``records``; cache the features.

    Tiles are grouped by record, so each selected record's EM is loaded once and only its kept tiles
    are forwarded, in batches of ``cfg.eval_batch_windows``. Returns
    ``{"feats": Tensor [N, C, grid, grid] fp16, "labels": Tensor [N, cmp, cmp] uint8,
    "datasets": list[str], "device": <where those two tensors live>}``, with ``cmp`` =
    ``cfg.effective_compare()`` and ``N`` the number of cached tiles.
    """
    encoder = encoder.to(device)
    t = int(cfg.tile_size)
    # The encoder crops its taps to the common compare region before returning them, so the cache is
    # sized from that region, not from the (possibly larger) context window it reads.
    grid = max(1, int(cfg.effective_compare()) // max(1, int(encoder.patch_size)))
    C = int(encoder.embedding_dim) * len(layers)
    per_tile_bytes = C * grid * grid * 2  # fp16 feature tile
    bs = max(1, int(cfg.eval_batch_windows))
    cap = int(cfg.cache_max_tiles) if int(getattr(cfg, "cache_max_tiles", 0)) > 0 else None
    # Hard GB ceiling regardless of tile count, so a wide encoder (ViT-L/H) cannot exhaust host or
    # GPU memory.
    gb_budget = float(getattr(cfg, "cache_max_gb", 0.0) or 0.0)
    if gb_budget > 0:
        gb_cap = max(1, int(gb_budget * 1e9 / max(per_tile_bytes, 1)))
        cap = gb_cap if cap is None else min(cap, gb_cap)
    rng = random.Random(int(cfg.seed))

    # Pass 1: enumerate valid tile positions (masks only, no GPU), then keep a uniform cap-subset.
    # A train set can tile into tens of thousands of windows, so the cap-subset is chosen from the
    # positions and only the kept tiles are forwarded through the encoder.
    positions = _enumerate_valid_tiles(records, derived_root, t)
    total = len(positions)
    if cap is not None and total > cap:
        rng.shuffle(positions)
        positions = positions[:cap]
    n_keep = len(positions)
    if n_keep == 0:
        return {"feats": torch.empty(0), "labels": torch.empty(0), "datasets": [], "device": "cpu"}
    by_rec: dict[int, list] = defaultdict(list)
    for (ri, y, x) in positions:
        by_rec[ri].append((y, x))

    # Decide where the cache lives. On the GPU, train_head_cached gathers batches already on-device --
    # no per-step CPU->GPU streaming (the dominant cached-training cost). Falls back to CPU when it
    # does not fit in VRAM (keeping ~12 GB headroom for the build/train/eval activations).
    cache_device = "cpu"
    if getattr(cfg, "cache_on_gpu", True) and str(device).startswith("cuda"):
        try:
            free, _tot = torch.cuda.mem_get_info()
            if n_keep * per_tile_bytes + 12e9 < free:
                cache_device = device
        except Exception:
            pass

    feats_t = torch.empty((n_keep, C, grid, grid), dtype=torch.float16, device=cache_device)
    cmp = int(cfg.effective_compare())
    labels_t = torch.empty((n_keep, cmp, cmp), dtype=torch.uint8, device=cache_device)
    datasets: list = [None] * n_keep
    pend_x: list[np.ndarray] = []
    pend_m: list[np.ndarray] = []
    pend_ds: list[str] = []
    write = 0

    def flush():
        nonlocal write
        if not pend_x:
            return
        xt = torch.from_numpy(np.stack(pend_x))[:, None].to(device)  # [b,1,t,t]
        fl = encoder.extract(xt, layers)
        cat = torch.cat([f.float() for f in fl], dim=1).half()  # [b,C,grid,grid] on `device`
        b = cat.shape[0]
        feats_t[write:write + b] = cat.to(cache_device)
        labels_t[write:write + b] = torch.from_numpy(np.stack(pend_m)).to(cache_device)
        datasets[write:write + b] = pend_ds
        write += b
        pend_x.clear(); pend_m.clear(); pend_ds.clear()

    # Pass 2: load each selected record's EM once, forward only its kept tiles.
    for ri, yxs in by_rec.items():
        r = records[ri]
        em, mask = load_sample(r, derived_root)
        H0, W0 = em.shape
        ph, pw = max(t - H0, 0), max(t - W0, 0)
        if ph or pw:
            em = np.pad(em, ((0, ph), (0, pw)), mode="reflect" if (H0 > 1 and W0 > 1) else "constant")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=IGNORE_INDEX)
        xn = normalize_em(em, encoder.image_mean, encoder.image_std)
        dsname = r.get("dataset", "?")
        for (y, x) in yxs:
            pend_x.append(np.ascontiguousarray(xn[y:y + t, x:x + t]))
            m = mask[y:y + t, x:x + t]
            if cmp != t:  # supervise the common compare region, as the uncached path does
                o = (t - cmp) // 2
                m = m[o:o + cmp, o:o + cmp]
            pend_m.append(np.ascontiguousarray(m.astype(np.uint8)))
            pend_ds.append(dsname)
            if len(pend_x) >= bs:
                flush()
    flush()
    gb = n_keep * per_tile_bytes / 1e9
    note = f" (subsampled from {total})" if total > n_keep else ""
    where = "GPU" if cache_device != "cpu" else "CPU"
    if gb > 1 or note:
        print(f"    [cache] {n_keep} train tiles{note}, ~{gb:.1f} GB {where} features (fp16)")
    return {"feats": feats_t, "labels": labels_t, "datasets": datasets, "device": cache_device}

def cache_subset_indices(cache: dict, frac: float, seed: int) -> list[int]:
    """Stratified-by-dataset, nested subset of cached-tile indices for a label fraction."""
    n = len(cache["feats"])
    if frac >= 1.0:
        return list(range(n))
    by_ds = defaultdict(list)
    for i, ds in enumerate(cache["datasets"]):
        by_ds[ds].append(i)
    out = []
    for ds in sorted(by_ds):
        idx = by_ds[ds]
        random.Random(f"{seed}:{ds}").shuffle(idx)
        out.extend(idx[: max(1, int(round(frac * len(idx))))])
    return out

def train_head_cached(cache, subset_idx, cfg, mode, embedding_dim, n_layers, patch_size,
                      device, logger=None, tag=""):
    """Train one decoder purely on cached features (no encoder forward).

    Batches are gathered by index from the stacked cache tensors. When the cache lives on the GPU
    (``cache['device']``), the gather + ``.to(device)`` is a no-op copy on-device, so there is no
    per-step CPU->GPU streaming, which is what keeps a cached linear probe GPU-bound.
    """
    if len(subset_idx) == 0:
        raise ValueError(f"empty feature cache for {tag}")
    decoder = build_decoder(embedding_dim=embedding_dim, n_layers=n_layers,
                            num_classes=cfg.num_classes, patch_size=patch_size, mode=mode).to(device)
    feats_t, labels_t = cache["feats"], cache["labels"]
    cdev = feats_t.device
    idx = torch.as_tensor(list(subset_idx), dtype=torch.long, device=cdev)
    opt = torch.optim.AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = bool(cfg.amp) and str(device).startswith("cuda")
    g = torch.Generator(device=cdev)
    g.manual_seed(int(cfg.seed))
    bs = int(cfg.batch_size)
    decoder.train()
    perm = idx[torch.randperm(idx.numel(), generator=g, device=cdev)]  # shuffle-without-replacement epochs
    pos = 0
    for step in range(cfg.max_steps):
        if pos + bs > perm.numel():
            perm = idx[torch.randperm(idx.numel(), generator=g, device=cdev)]
            pos = 0
        sel = perm[pos:pos + bs]
        pos += bs
        feat = feats_t[sel].to(device, non_blocking=True).float()  # on-device gather; no copy if GPU cache
        y = labels_t[sel].to(device, non_blocking=True).long()
        for gp in opt.param_groups:
            gp["lr"] = _lr_at(step, cfg.max_steps, cfg.warmup_steps, cfg.lr)
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _null()
        with ctx:
            # The cache stores the taps concatenated on the channel axis. Split them back into the
            # per-layer list the decoders take: the pyramid decoders index the taps separately, and
            # the linear head concatenates them again itself.
            cmp = int(cfg.effective_compare())
            logits = decoder(list(feat.chunk(n_layers, dim=1)), out_hw=(cmp, cmp))
            ce = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
            loss = cfg.ce_weight * ce + cfg.dice_weight * soft_dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if logger is not None and step % 50 == 0:
            logger(step, {"loss": float(loss.detach()), "lr": opt.param_groups[0]["lr"]})
    decoder.eval()
    return decoder

class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

"""Datasets over the derived manifest.

Train: map-style dataset yielding augmented random ``tile_size`` crops (focused on labelled area,
all-ignore crops resampled), normalised with the statistics the encoder declares — the EM corpus
mean/std for the encoders pretrained here, and 0/1 for the external timm baselines, which apply their
own per-channel normalization inside feature extraction. No per-tile percentile norm. Val/Test:
full-region loading for honest sliding-window evaluation (see evaluate.py).

Label-efficiency subsetting is stratified by dataset and nested across fractions (1% ⊂ 10% ⊂
50% ⊂ 100%) with a fixed seed, so the curve is monotone-comparable.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import IGNORE_INDEX
from ..dataprep.io import read_png_L

def load_manifest(derived_root: str | Path, organelle: str, split: str) -> list[dict]:
    """Read manifest.jsonl, filtered to one organelle + split."""
    path = Path(derived_root) / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Derived manifest not found: {path} (run dataprep.build_dataset)")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("organelle") == organelle and r.get("split") == split:
                out.append(r)
    return out

def subset_fraction(records: list[dict], frac: float, seed: int = 0) -> list[dict]:
    """Stratified-by-dataset, nested subset of ``records`` for a label fraction in (0,1]."""
    if frac >= 1.0:
        return list(records)
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_ds[r.get("dataset", "?")].append(r)
    out = []
    for ds in sorted(by_ds):
        rs = sorted(by_ds[ds], key=lambda r: r["sample_id"])
        # deterministic per-(seed,dataset) shuffle, independent of fraction -> nested prefixes
        random.Random(f"{seed}:{ds}").shuffle(rs)
        k = max(1, int(round(frac * len(rs)))) if rs else 0
        out.extend(rs[:k])
    return out

def load_sample(record: dict, derived_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(em uint8 HxW, mask uint8 HxW in {0,1,255}) for one manifest record."""
    root = Path(derived_root)
    em = read_png_L(root / record["em_path"])
    mask = read_png_L(root / record["mask_path"])
    return em, mask

def load_inst(record: dict, derived_root: str | Path):
    """Instance-id GT map (int32, 0=bg) for a mito record, or None if not stored."""
    rel = record.get("inst_path")
    if not rel:
        return None
    from ..dataprep.io import read_tif_u16
    p = Path(derived_root) / rel
    return read_tif_u16(p) if p.exists() else None

def normalize_em(em: np.ndarray, mean: float, std: float) -> np.ndarray:
    """uint8 [0,255] -> float32, scaled to [0,1] then (x-mean)/std. No per-tile percentile norm."""
    x = em.astype(np.float32) / 255.0
    return (x - mean) / std

class ProbeTrainDataset(Dataset):
    def __init__(self, records: list[dict], derived_root, cfg, mean: float, std: float):
        self.records = records
        self.root = Path(derived_root)
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.tile = cfg.tile_size
        # A sequential RNG, re-seeded per worker by worker_init_fn, and deliberately not re-seeded per
        # record index: the loop revisits each record many times, so a per-index seed would replay the
        # identical crop every visit. Drawing sequentially gives fresh crops while staying reproducible
        # per (seed, worker).
        self.rng = random.Random(cfg.seed)

    def reseed(self, salt: int) -> None:
        self.rng = random.Random((self.cfg.seed * 1_000_003 + salt) & 0x7FFFFFFF)

    def __len__(self) -> int:
        return len(self.records)

    def _pad_to_tile(self, em: np.ndarray, mask: np.ndarray):
        t = self.tile
        H, W = em.shape
        if H >= t and W >= t:
            return em, mask
        ph, pw = max(t - H, 0), max(t - W, 0)
        em = np.pad(em, ((0, ph), (0, pw)), mode="reflect" if (H > 1 and W > 1) else "constant")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=IGNORE_INDEX)
        return em, mask

    def _crop_containing(self, em, mask, ann_xyxy, rng: random.Random):
        """Position a tile_size window to contain the annotation bbox (centre the tile on the
        annotation centre, clamp to [0, dim-tile]), jittering only within the slack that still keeps
        the annotation inside. Guarantees edge annotations are covered; no reflect-pad of real EM."""
        t = self.tile
        em, mask = self._pad_to_tile(em, mask)  # only pads dims below tile (such tiles are already >= t)
        H, W = em.shape
        ax0, ay0, ax1, ay1 = ann_xyxy

        def _start(a0, a1, dim):
            a0 = max(0, min(int(a0), dim))
            a1 = max(a0, min(int(a1), dim))
            if dim <= t:
                return 0
            # window start must keep [a0,a1) inside [s, s+t): s in [a1-t, a0], then clamp to grid.
            lo = max(0, a1 - t)
            hi = min(dim - t, a0)
            if lo > hi:  # annotation wider than the tile -> centre the tile on the annotation centre
                c = (a0 + a1) // 2
                return int(np.clip(c - t // 2, 0, dim - t))
            return int(rng.randint(lo, hi))

        y = _start(ay0, ay1, H)
        x = _start(ax0, ax1, W)
        return em[y:y + t, x:x + t], mask[y:y + t, x:x + t]

    def _random_crop(self, em: np.ndarray, mask: np.ndarray, rng: random.Random):
        t = self.tile
        em, mask = self._pad_to_tile(em, mask)
        H, W = em.shape
        best = None
        for _ in range(8):  # try to land a crop with some labelled (non-ignore) pixels
            y = rng.randint(0, H - t)
            x = rng.randint(0, W - t)
            mc = mask[y:y + t, x:x + t]
            valid = int((mc != IGNORE_INDEX).sum())
            if valid == 0:
                continue
            fgfrac = float((mc == 1).sum()) / max(valid, 1)
            cand = (em[y:y + t, x:x + t], mc)
            if best is None:
                best = cand
            if fgfrac >= self.cfg.min_fg_frac_keep:
                return cand
        if best is not None:
            return best
        # guaranteed-valid fallback: centre a tile on a labelled pixel (never an all-ignore crop)
        ys, xs = np.where(mask != IGNORE_INDEX)
        if len(ys):
            i = rng.randrange(len(ys))
            y = int(np.clip(ys[i] - t // 2, 0, H - t))
            x = int(np.clip(xs[i] - t // 2, 0, W - t))
            return em[y:y + t, x:x + t], mask[y:y + t, x:x + t]
        return em[:t, :t], mask[:t, :t]

    def _augment(self, em, mask, rng):
        if self.cfg.flip:
            if rng.random() < 0.5:
                em, mask = em[:, ::-1], mask[:, ::-1]
            if rng.random() < 0.5:
                em, mask = em[::-1, :], mask[::-1, :]
        if self.cfg.rot90:
            k = rng.randint(0, 3)
            if k:
                em, mask = np.rot90(em, k), np.rot90(mask, k)
        em = np.ascontiguousarray(em)
        mask = np.ascontiguousarray(mask)
        if self.cfg.intensity_jitter > 0:
            j = self.cfg.intensity_jitter
            a = 1.0 + rng.uniform(-j, j)  # contrast
            b = rng.uniform(-j, j) * 255.0  # brightness (in uint8 units)
            em = np.clip(em.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return em, mask

    def __getitem__(self, idx: int):
        r = self.records[idx]
        rng = self.rng  # sequential draws -> fresh crop/aug each revisit (see __init__)
        em, mask = load_sample(r, self.root)
        # Records carrying annotation_bbox_in_tile_xyxy (even-0-padded >= tile tiles) are cropped to
        # contain the annotation; records without the field use a labelled-pixel-biased random crop.
        ann = r.get("annotation_bbox_in_tile_xyxy")
        if ann is not None:
            em, mask = self._crop_containing(em, mask, ann, rng)
        else:
            em, mask = self._random_crop(em, mask, rng)
        em, mask = self._augment(em, mask, rng)
        x = normalize_em(em, self.mean, self.std)
        x = torch.from_numpy(x)[None]  # [1, t, t]
        y = torch.from_numpy(mask.astype(np.int64))  # [t, t] in {0,1,255}
        return x, y

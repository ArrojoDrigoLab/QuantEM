"""Segmentation-training dataset + manifest reading over the derived corpus.

Unlike the encoder comparison, dataprep has already resampled the derived data to the per-organelle
canonical nm/px, so this module performs no resampling — it only crops, augments (the fixed PairedAug
recipe), and normalises. Train: map-style dataset yielding ``tile_size`` crops, positioned to contain
the record's annotation bbox where one is recorded and otherwise drawn at random over labelled area
(all-ignore crops resampled); regions smaller than the tile are 0-padded up. Val/Test use full-
region sliding-window eval (evaluate.py), not this dataset.

Records are read from ``<data_root>/manifest.jsonl`` and filtered to one group + split + one
resolution bucket (``canonical`` by default; the ``native_unscaled`` bucket, holding unknown-scale
crops emitted by the native_bucket policy, is never a training source here).

No GPU is needed: torch + numpy + (PIL/tifffile via dataprep.io) only. sklearn, skimage and BLAS matmuls
are kept out of import time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import IGNORE_INDEX
from ..dataprep.io import read_png_L
from .augment import PairedAug


def load_manifest(data_root: str | Path, group: str, split: str, bucket: str = "canonical",
                  manifest_name: str = "manifest.jsonl") -> list[dict]:
    """Read ``manifest.jsonl`` filtered to one group + split + resolution ``bucket``.

    ``group`` is e.g. ``"group2_er"``; ``split`` is a derived-dir name (``train`` | ``val`` | ``test``).
    ``bucket`` selects the resolution view: ``canonical`` (default; the per-organelle canonical nm/px)
    or ``native`` (source resolution, unresampled — emitted by ``build_dataset --scale-mode
    native|both`` and read by the arms that set ``data.bucket: native``). The unknown-scale
    ``native_unscaled`` crops are a third bucket, never a default source.
    """
    path = Path(data_root) / manifest_name
    if not path.exists():
        raise FileNotFoundError(f"Derived manifest not found: {path} (run segmentation_training.dataprep.build_dataset)")
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("group") == group and r.get("split") == split
                    and r.get("bucket") == bucket):
                out.append(r)
    return out


def normalize_em(em_uint8: np.ndarray, mean: float, std: float) -> np.ndarray:
    """uint8 [0,255] EM -> float32, scaled to [0,1] then ``(x - mean) / std``. No per-tile percentile
    norm (mean/std come from the encoder's EM corpus stats, not ImageNet)."""
    x = em_uint8.astype(np.float32) / 255.0
    return (x - mean) / std


def subset_fraction(records: list[dict], frac: float, seed: int = 0) -> list[dict]:
    """Stratified-by-dataset, nested subset of ``records`` for a label fraction in (0,1].

    Nested across fractions (1% ⊂ 10% ⊂ 50% ⊂ 100%): the per-(seed,dataset) shuffle is independent of
    ``frac`` so smaller fractions are prefixes of larger ones, keeping the label-efficiency curve
    monotone-comparable. Stratifies by ``dataset`` so every source contributes at least one crop.
    """
    import random

    if frac >= 1.0:
        return list(records)
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_ds[r.get("dataset", "?")].append(r)
    out: list[dict] = []
    for ds in sorted(by_ds):
        rs = sorted(by_ds[ds], key=lambda r: r["sample_id"])
        random.Random(f"{seed}:{ds}").shuffle(rs)
        k = max(1, int(round(frac * len(rs)))) if rs else 0
        out.extend(rs[:k])
    return out


def subset_sources(records: list[dict], frac: float, seed: int = 0) -> list[dict]:
    """Keep a random ``frac`` of the distinct sources (datasets), and every crop of the kept sources. This
    is the diversity axis of the diversity-vs-volume comparison (``subset_fraction`` supplies the volume
    axis). Deterministic per ``seed``; keeps at least one source; nested (25% sources ⊂ 50% sources) so the
    axis is monotone."""
    import random

    if frac >= 1.0:
        return list(records)
    datasets = sorted({r.get("dataset", "?") for r in records})
    random.Random(f"src:{seed}").shuffle(datasets)
    k = max(1, int(round(frac * len(datasets))))
    keep = set(datasets[:k])
    return [r for r in records if r.get("dataset", "?") in keep]


def _round_up_to(n: int, m: int = 16) -> int:
    """Round ``n`` up to a multiple of ``m`` (the encoder's patch size); a tile must be a whole number
    of patches — 16 for the ViT / DINOv3 / EMCF-MAE encoders, 14 for OmniEM ViT-L."""
    m = max(1, int(m))
    return int(((int(n) + m - 1) // m) * m)


def load_sample(record: dict, data_root: str | Path, with_inst: bool = True):
    """(em uint8 HxW, mask uint8 HxW in {0,1,255}, inst int32 HxW or None) for one record.

    ``with_inst=False`` skips reading the (possibly present) instance-id TIFF — semantic-task training
    never uses it, so loading it per sample is pure waste and, worse, makes batches non-uniform (only
    records that happen to have a stored inst get an ``inst`` key -> default-collate KeyError)."""
    root = Path(data_root)
    em = read_png_L(root / record["em_path"])
    mask = read_png_L(root / record["mask_path"])
    inst = None
    rel = record.get("inst_path")
    if with_inst and rel:
        p = root / rel
        if p.exists():
            from ..dataprep.io import read_tif_u16
            inst = read_tif_u16(p)
    return em, mask, inst


class SegTrainDataset(Dataset):
    """Map-style dataset of augmented ``tile_size`` crops for one organelle / split / label fraction.

    Per item: load em(png) + mask(png) (+ the instance-id tif on an instance task), take a tile crop —
    positioned to contain the record's ``annotation_bbox_in_tile_xyxy`` when it has one, otherwise a
    random crop that overlaps labelled pixels (>= ``cfg.data.min_fg_frac_keep`` foreground of the valid
    area, or a centre-on-a-labelled-pixel fallback after N tries so an all-ignore crop is never
    returned) — apply the fixed ``PairedAug`` (geometric on em+mask+inst, intensity on em only), then normalise.

    Returns ``{image: [1,H,W] float32, target: [H,W] long (mask, ignore=255 kept), inst: [H,W] long}``
    (``inst`` on every item of an instance task, on none of a semantic one). ``tile_size`` is rounded
    up to a multiple of the encoder's patch size.
    """

    def __init__(self, records: list[dict], data_root, cfg, mean: float, std: float, patch_size: int = 16,
                 vocab=None):
        self.records = records
        self.root = Path(data_root)
        self.cfg = cfg
        # image-style conditioning metadata (source id + categorical style fields) — emitted only when a vocab is supplied, so
        # non-image-style conditioning arms keep uniform batch keys {image, target[, inst]} and are byte-identical.
        self.vocab = vocab
        self.mean = float(mean)
        self.std = float(std)
        # Tile must be a whole number of encoder patches (16 for the ViT/DINOv3/EMCF encoders, 14 for OmniEM
        # ViT-L); round the configured tile_size up to the encoder's patch size.
        self.tile = _round_up_to(cfg.encoder.tile_size, patch_size)
        self.min_fg = float(cfg.data.min_fg_frac_keep)
        self.seed = int(cfg.optim.seed)
        # A *sequential* per-worker numpy Generator (re-seeded per worker in train.py via
        # worker_init_fn -> reseed()). Deliberately not re-seeded per record index: the training loop
        # revisits each record many times over the step budget, so a per-idx seed would replay the
        # identical crop+augmentation every visit and nullify augmentation. Sequential draws give fresh
        # crops/augmentations each revisit while staying reproducible per (seed, worker).
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.aug = PairedAug(cfg.data, self.rng)  # one PairedAug per worker; shares this generator

    def reseed(self, salt: int) -> None:
        self.rng = np.random.default_rng((self.seed * 1_000_003 + salt) & 0x7FFFFFFF)
        self.aug = PairedAug(self.cfg.data, self.rng)

    def __len__(self) -> int:
        return len(self.records)

    # -- padding + crop -----------------------------------------------------
    def _pad_to_tile(self, em, mask, inst):
        t = self.tile
        H, W = em.shape
        if H >= t and W >= t:
            return em, mask, inst
        ph, pw = max(t - H, 0), max(t - W, 0)
        # 0-pad the image (a black border rather than reflected, so no tissue is fabricated); mask/inst are
        # padded with ignore/0 so the padded border is never a spurious label.
        em = np.pad(em, ((0, ph), (0, pw)), mode="constant")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=IGNORE_INDEX)
        if inst is not None:
            inst = np.pad(inst, ((0, ph), (0, pw)), mode="constant", constant_values=0)
        return em, mask, inst

    def _crop(self, y, x, em, mask, inst):
        t = self.tile
        ec = em[y:y + t, x:x + t]
        mc = mask[y:y + t, x:x + t]
        ic = inst[y:y + t, x:x + t] if inst is not None else None
        return ec, mc, ic

    def _crop_containing(self, em, mask, inst, ann_xyxy):
        """Annotation-containing crop: position a tile_size window to contain the annotation bbox (jitter only within the
        slack that keeps the annotation inside; centre on the annotation if it is wider than the tile).
        Guarantees edge annotations are covered without reflect-padding real EM. Preserves the instance
        map + PairedAug that run downstream. ``ann_xyxy`` is in the tile's (canonical) pixel frame."""
        t = self.tile
        em, mask, inst = self._pad_to_tile(em, mask, inst)  # only pads dims below tile
        H, W = em.shape
        r = self.rng
        ax0, ay0, ax1, ay1 = ann_xyxy

        def _start(a0, a1, dim):
            a0 = max(0, min(int(a0), dim))
            a1 = max(a0, min(int(a1), dim))
            if dim <= t:
                return 0
            lo = max(0, a1 - t)
            hi = min(dim - t, a0)
            if lo > hi:  # annotation wider than the tile -> centre on its centre
                c = (a0 + a1) // 2
                return int(np.clip(c - t // 2, 0, dim - t))
            return int(r.integers(lo, hi + 1))

        y = _start(ay0, ay1, H)
        x = _start(ax0, ax1, W)
        return self._crop(y, x, em, mask, inst)

    def _random_crop(self, em, mask, inst):
        t = self.tile
        em, mask, inst = self._pad_to_tile(em, mask, inst)
        H, W = em.shape
        r = self.rng
        best = None
        for _ in range(8):  # try to land a crop with enough labelled foreground
            y = int(r.integers(0, H - t + 1))
            x = int(r.integers(0, W - t + 1))
            mc = mask[y:y + t, x:x + t]
            valid = int((mc != IGNORE_INDEX).sum())
            if valid == 0:
                continue
            if best is None:
                best = (y, x)
            fgfrac = float((mc == 1).sum()) / max(valid, 1)
            if fgfrac >= self.min_fg:
                return self._crop(y, x, em, mask, inst)
        if best is not None:
            return self._crop(best[0], best[1], em, mask, inst)
        # guaranteed-valid fallback: centre a tile on a labelled pixel (never an all-ignore crop).
        ys, xs = np.where(mask != IGNORE_INDEX)
        if len(ys):
            i = int(r.integers(0, len(ys)))
            y = int(np.clip(ys[i] - t // 2, 0, H - t))
            x = int(np.clip(xs[i] - t // 2, 0, W - t))
            return self._crop(y, x, em, mask, inst)
        return self._crop(0, 0, em, mask, inst)

    # -- item ---------------------------------------------------------------
    def __getitem__(self, idx: int):
        r = self.records[idx]
        # Whether this arm needs instance ids at all is a property of the task, not the record: gating
        # both ways keeps every batch's keys uniform (default collate raises KeyError otherwise).
        #   * semantic (any arm whose decoder predicts a binary map): ``inst`` is never emitted and the
        #     TIFF is not read.
        #   * instance (the arms on an instance decoder — affinity_mws, panoptic_deeplab,
        #     mask2former_query_hf, dodnet): ``inst`` is always emitted — stored ids when present, else
        #     connected components of the binary GT (evaluate.py's pseudo-instance policy), so instance
        #     decoders receive real per-crop training targets.
        want_inst = getattr(self.cfg.data, "task", "semantic") == "instance"
        em, mask, inst = load_sample(r, self.root, with_inst=want_inst)
        # Records may carry annotation_bbox_in_tile_xyxy (even-0-padded >= tile tiles at canonical
        # nm/px). Crop to contain the annotation; records without the field take a random crop.
        ann = r.get("annotation_bbox_in_tile_xyxy")
        if ann is not None:
            em, mask, inst = self._crop_containing(em, mask, inst, ann)
        else:
            em, mask, inst = self._random_crop(em, mask, inst)
        em, mask, inst = self.aug(em, mask, inst)  # geometric (em+mask+inst) + intensity (em)
        if want_inst and inst is None:
            from scipy import ndimage as ndi
            inst, _ = ndi.label((mask == 1) & (mask != IGNORE_INDEX))
            inst = inst.astype(np.int32)
        x = normalize_em(em, self.mean, self.std)
        out = {
            "image": torch.from_numpy(np.ascontiguousarray(x))[None],           # [1,H,W] float32
            "target": torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64)),  # [H,W] long
        }
        if want_inst and inst is not None:
            out["inst"] = torch.from_numpy(np.ascontiguousarray(inst).astype(np.int64))  # [H,W] long
        if self.vocab is not None:
            # nested dict of python ints -> default_collate stacks each field into a LongTensor[B].
            out["meta"] = {f: int(v) for f, v in self.vocab.encode(r).items()}
        return out

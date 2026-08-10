"""Head-only fine-tuning on a handful of user-annotated regions.

The implementation behind Fig. S6C, with the hard-coded scratch path and absolute model
registry removed.

Only the neck and decoder train (5.8 M parameters for QuantEM ViT-B, 33.4 M for OmniEM ViT-L). The
manuscript found this within noise of LoRA and of unfreezing the last four blocks, at a fraction of
the cost: measured 17.7 s (ViT-B) and 50.0 s (ViT-L) for 300 steps, against 107-114 s for the
heavier modes.

The ``valid`` mask is not optional. A user who annotates three mitochondria in the corner of a large
image has not asserted that the rest is background, and training as though they had actively degrades
the model. Everything outside ``valid`` becomes ``IGNORE_INDEX`` and contributes to neither the loss
nor the calibration metric — the same contract that produced the published curve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import IGNORE_INDEX

#: Defaults from the reference implementation. Steps is the only one worth exposing.
DEFAULT_STEPS = 300
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
#: A training window is kept only if at least this fraction of it is inside the annotated region.
MIN_VALID_FRACTION = 0.2
#: Threshold sweep used to calibrate on the user's own regions.
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.05, 0.96, 0.05))


@dataclass
class Example:
    """One annotated region: the image, the labels, and where the annotation is trusted."""

    image: np.ndarray  # 2-D, any dtype
    labels: np.ndarray  # 2-D, non-zero = foreground
    valid: np.ndarray  # 2-D bool; True inside the reviewed region
    name: str = ""
    pixel_size_nm: float | tuple[float, float] | None = None

    def __post_init__(self):
        self.image = np.asarray(self.image)
        self.labels = np.asarray(self.labels)
        self.valid = np.asarray(self.valid, dtype=bool)
        if not (self.image.shape[:2] == self.labels.shape[:2] == self.valid.shape[:2]):
            raise ValueError(
                f"example {self.name!r}: image {self.image.shape}, labels {self.labels.shape} and "
                f"valid {self.valid.shape} must have the same 2-D shape"
            )
        if not self.valid.any():
            raise ValueError(f"example {self.name!r}: the reviewed region is empty")


def _target(ex: Example) -> np.ndarray:
    """{0 background, 1 foreground, 255 ignore} — outside the reviewed region is ignore."""
    t = np.where(ex.labels > 0, 1, 0).astype(np.uint8)
    t[~ex.valid] = IGNORE_INDEX
    return t


def _prepare_tiles(model, examples, tile: int):
    """Windows on a ``tile // 2`` grid, keeping only those with enough annotated area.

    This is the reference's sampling rule, and it is also nnU-Net's point: without it, a user who
    annotates a small corner of a large image gets mostly-empty batches.
    """
    from .inference.normalize import normalize_em
    from .inference.predict import plan_resample
    from .inference.prepare import to_uint8
    from .inference.resample import zoom_image, zoom_labels

    spec = model.spec
    out = []
    for ex in examples:
        em, _ = to_uint8(ex.image)
        tgt = _target(ex)
        factors, _ = plan_resample(spec, em.shape, ex.pixel_size_nm, allow_extreme=True)
        if factors is not None:
            em = zoom_image(em, factors)
            tgt = zoom_labels(tgt, factors)
        xn = normalize_em(em, spec.encoder.dataset_mean, spec.encoder.dataset_std)
        h, w = em.shape
        step = max(1, tile // 2)
        for y in range(0, max(1, h - tile + 1), step):
            for x in range(0, max(1, w - tile + 1), step):
                sub_t = tgt[y : y + tile, x : x + tile]
                if sub_t.shape != (tile, tile):
                    continue
                if (sub_t != IGNORE_INDEX).sum() < MIN_VALID_FRACTION * tile * tile:
                    continue
                out.append(
                    (
                        np.ascontiguousarray(xn[y : y + tile, x : x + tile]),
                        np.ascontiguousarray(sub_t.astype(np.int64)),
                    )
                )
    return out


def _loss_fn(logits, target, ignore_index: int = IGNORE_INDEX):
    """Cross-entropy + soft Dice, both restricted to valid pixels. Reference formulation."""
    import torch
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, target[None], ignore_index=ignore_index)
    prob = torch.softmax(logits, 1)[:, 1]
    valid = (target != ignore_index).float()[None]
    g = (target == 1).float()[None]
    inter = (prob * g * valid).sum()
    denom = ((prob + g) * valid).sum()
    return ce + (1 - (2 * inter + 1) / (denom + 1))


def _masked_dice(prob: np.ndarray, target: np.ndarray, thr: float) -> float | None:
    valid = target != IGNORE_INDEX
    p = (prob >= thr) & valid
    g = (target == 1) & valid
    denom = int(p.sum() + g.sum())
    return None if denom == 0 else 2.0 * int((p & g).sum()) / denom


def finetune_head(
    model,
    examples,
    *,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = 0,
    calibrate_threshold: bool = True,
    progress=None,
    cancel=None,
) -> dict:
    """Train neck + decoder on ``examples``, in place. Returns a report.

    ``progress(done, total, seconds_per_step)`` is called every step, so a UI can show a live
    estimate measured from the first few steps rather than a guessed one.
    """
    import torch

    if not examples:
        raise ValueError("fine-tuning needs at least one annotated region")

    spec = model.spec
    tile = spec.effective_tile()
    dev = model.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tiles = _prepare_tiles(model, examples, tile)
    if not tiles:
        raise ValueError(
            f"no training window had at least {MIN_VALID_FRACTION:.0%} of its area inside a "
            f"reviewed region. The reviewed regions may be smaller than one {tile} px tile."
        )

    for p in model.module.parameters():
        p.requires_grad_(False)
    trainable = []
    for m in (model.module.neck, model.module.decoder):
        for p in m.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    def aug(im, t):
        if rng.random() < 0.5:
            im, t = im[:, ::-1].copy(), t[:, ::-1].copy()
        if rng.random() < 0.5:
            im, t = im[::-1].copy(), t[::-1].copy()
        k = int(rng.integers(4))
        return np.rot90(im, k).copy(), np.rot90(t, k).copy()

    model.module.train()
    t0 = time.perf_counter()
    losses = []
    for step in range(int(steps)):
        if cancel is not None and cancel():
            break
        im, tg = aug(*tiles[int(rng.integers(len(tiles)))])
        xt = torch.from_numpy(im)[None, None].float().to(dev)
        tt = torch.from_numpy(tg).to(dev)
        loss = _loss_fn(model.module(xt), tt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if progress is not None:
            progress(step + 1, int(steps), (time.perf_counter() - t0) / (step + 1))
    train_s = time.perf_counter() - t0
    model.module.eval()

    # len(losses), not the requested count: Stop breaks out of the loop, and a report claiming
    # 300 steps after 12 ran would be saved into the adapted head's model card and published.
    completed = len(losses)
    cancelled = completed < int(steps)
    report = {
        "steps": completed,
        "steps_requested": int(steps),
        "cancelled": cancelled,
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "train_seconds": round(train_s, 2),
        "seconds_per_step": round(train_s / max(1, len(losses)), 4),
        "n_regions": len(examples),
        "n_windows": len(tiles),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }

    if calibrate_threshold and not cancelled:
        cal = calibrate(model, examples)
        model._calibrated_threshold = cal["threshold"]
        report["calibration"] = cal
    model._adapted_from = spec.model_id
    return report


def calibrate(model, examples) -> dict:
    """Pick the Dice-maximising threshold on the given regions.

    The returned Dice is measured on the *same* regions the model just trained on, so it is
    optimistic and must be labelled as such wherever it is shown. Use :func:`cross_validate` for a
    held-out number.
    """
    preds = [
        (
            model.predict_probability(ex.image, pixel_size_nm=ex.pixel_size_nm).probability,
            _target(ex),
        )
        for ex in examples
    ]

    def mean_dice(thr):
        vals = [d for d in (_masked_dice(p, t, thr) for p, t in preds) if d is not None]
        return float(np.mean(vals)) if vals else 0.0

    scores = {float(t): mean_dice(t) for t in THRESHOLDS}
    best = max(scores, key=scores.get)
    return {
        "threshold": best,
        "dice_at_threshold": scores[best],
        "dice_at_default": mean_dice(model.spec.fg_threshold),
        "sweep": scores,
        "measured_on": "training regions (optimistic; not held out)",
    }


def cross_validate(
    load_fresh,
    examples,
    *,
    steps: int = DEFAULT_STEPS,
    progress=None,
    cancel=None,
    **kw,
) -> dict:
    """Leave-one-region-out cross-validation: an honest, held-out estimate.

    Trains ``len(examples)`` separate models, each on all regions but one, and scores the held-out
    region. Needs at least three regions to be meaningful, and costs *k* times a single fine-tune —
    which is why it is opt-in.

    ``load_fresh()`` must return a newly loaded, un-adapted model each time it is called.
    """
    n = len(examples)
    if n < 3:
        raise ValueError(f"leave-one-out cross-validation needs at least 3 regions, got {n}")

    folds = []
    for i in range(n):
        if cancel is not None and cancel():
            break
        train = [e for j, e in enumerate(examples) if j != i]
        held = examples[i]
        m = load_fresh()

        def fold_progress(done, total, sps, _i=i):
            if progress is not None:
                progress(_i, n, done, total, sps)

        finetune_head(
            m,
            train,
            steps=steps,
            calibrate_threshold=True,
            progress=fold_progress,
            cancel=cancel,
            **kw,
        )
        thr = m.threshold
        prob = m.predict_probability(held.image, pixel_size_nm=held.pixel_size_nm).probability
        d = _masked_dice(prob, _target(held), thr)
        folds.append({"held_out": held.name or f"region {i + 1}", "dice": d, "threshold": thr})
        del m

    vals = [f["dice"] for f in folds if f["dice"] is not None]
    return {
        "folds": folds,
        "mean_dice": float(np.mean(vals)) if vals else None,
        "std_dice": float(np.std(vals)) if vals else None,
        "n_folds": len(folds),
        "measured_on": "held-out regions (leave-one-out)",
    }


def save_adapted_head(model, path, *, note: str = "") -> Path:
    """Write the adapted neck + decoder, plus enough provenance to load it back."""
    import json

    from safetensors.torch import save_file

    path = Path(path)
    tensors = {}
    for prefix, mod in (("neck.", model.module.neck), ("decoder.", model.module.decoder)):
        for k, v in mod.state_dict().items():
            tensors[prefix + k] = v.contiguous().cpu()
    meta = {
        "quantem_adapted_head": "1",
        "base_model_id": model.spec.model_id,
        "arm_name": model.spec.arm_name,
        "organelle": model.spec.organelle,
        "threshold": str(model.threshold),
        "note": note,
    }
    save_file(tensors, str(path), metadata={k: str(v) for k, v in meta.items()})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_adapted_head(model, path) -> dict:
    """Load an adapted head back onto a freshly loaded base model."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata() or {}
    base = meta.get("base_model_id")
    if base and base != model.spec.model_id:
        raise ValueError(
            f"this adapted head was trained on {base!r}, but the loaded model is "
            f"{model.spec.model_id!r}."
        )
    tensors = load_file(str(path))
    neck = {k[len("neck.") :]: v for k, v in tensors.items() if k.startswith("neck.")}
    dec = {k[len("decoder.") :]: v for k, v in tensors.items() if k.startswith("decoder.")}
    model.module.neck.load_state_dict(neck, strict=True)
    model.module.decoder.load_state_dict(dec, strict=True)
    model.module.to(model.device).eval()
    if meta.get("threshold"):
        model._calibrated_threshold = float(meta["threshold"])
    model._adapted_from = base or model.spec.model_id
    return meta

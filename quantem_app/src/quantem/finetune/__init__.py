"""Guided fine-tuning: threshold calibration and decoder-head adaptation.

The manuscript's headline software claim. Two rungs, in increasing cost:

1. **Threshold calibration** (:mod:`quantem.finetune.calibrate`) — fits one
   scalar against the user's annotations. numpy only, seconds, no GPU. Always
   available.
2. **Head adaptation** — freezes the encoder and trains the neck + decoder
   (5.78 M parameters for QuantEM ViT-B). Because the backbone is frozen it runs
   forward-only through the encoder, so it is viable on CPU, though far faster
   with CUDA or MPS.

Both are fit on the user's *training* crops and only ever *scored* on held-out
ones. The split mode is reported alongside every number so a within-image score
is never mistaken for generalisation to a new image.
"""

from .adapt import (
    IGNORE,
    AdaptConfig,
    AdaptProgress,
    HeadAdaptationUnavailable,
    build_patches,
    freeze_to_head,
    head_loss,
    save_head,
    tile_for,
    torch_available,
    train_head,
)
from .calibrate import (
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    Crop,
    SweepResult,
    masked_dice,
    mean_dice,
    split_crops,
    sweep_threshold,
)

# ``job``, ``models`` and ``views`` are deliberately not imported here: they
# reach into the segmentation app (which imports this package's calibrator), and
# the job entry point is looked up lazily by quantem.jobs.handlers anyway.
__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "IGNORE",
    "AdaptConfig",
    "AdaptProgress",
    "Crop",
    "HeadAdaptationUnavailable",
    "SweepResult",
    "build_patches",
    "freeze_to_head",
    "head_loss",
    "masked_dice",
    "mean_dice",
    "save_head",
    "split_crops",
    "sweep_threshold",
    "tile_for",
    "torch_available",
    "train_head",
]

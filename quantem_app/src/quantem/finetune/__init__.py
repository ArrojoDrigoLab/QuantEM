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

The public symbols remain available from this package, but are imported only
when requested. Django registers ``quantem.finetune`` on every launch; eagerly
loading OpenCV, NumPy, and the training stack there made even ordinary viewing
pay model-training startup costs.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "IGNORE": "adapt",
    "AdaptConfig": "adapt",
    "AdaptProgress": "adapt",
    "HeadAdaptationUnavailable": "adapt",
    "build_patches": "adapt",
    "freeze_to_head": "adapt",
    "head_loss": "adapt",
    "save_head": "adapt",
    "tile_for": "adapt",
    "torch_available": "adapt",
    "train_head": "adapt",
    "DEFAULT_THRESHOLD": "calibrate",
    "DEFAULT_THRESHOLDS": "calibrate",
    "Crop": "calibrate",
    "SweepResult": "calibrate",
    "masked_dice": "calibrate",
    "mean_dice": "calibrate",
    "split_crops": "calibrate",
    "sweep_threshold": "calibrate",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


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

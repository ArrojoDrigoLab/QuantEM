"""Turning a user's own annotations into supervision for guided fine-tuning.

Everything here answers one question: *what has the user actually told us, and
where are we allowed to score a model against it?* The answer is the
completed-ROI contract — inside a completed ROI the annotation is exhaustive, so
a confirmed object is foreground and everything else is genuine background;
outside it nothing is known and the pixels are ``ignore``.

:mod:`quantem.finetune` consumes this; nothing here imports the trainer.
"""

from .extract_crops import (
    IGNORE,
    MODE_HEAD,
    MODE_THRESHOLD_ONLY,
    NO_PROBABILITY_MESSAGE,
    AnnotatedCrop,
    CompletedRoiRequired,
    CropSet,
    collect_crops,
    plan_split,
    require_crops,
)

__all__ = [
    "IGNORE",
    "MODE_HEAD",
    "MODE_THRESHOLD_ONLY",
    "NO_PROBABILITY_MESSAGE",
    "AnnotatedCrop",
    "CompletedRoiRequired",
    "CropSet",
    "collect_crops",
    "plan_split",
    "require_crops",
]

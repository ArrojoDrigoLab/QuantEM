"""Turning a user's own annotations into supervision for guided fine-tuning.

Everything here answers one question: *what has the user actually told us, and
where are we allowed to score a model against it?* The answer is the
exhaustively-annotated-region contract — inside a completed area the annotation
is exhaustive, so a confirmed object is foreground and everything else is
genuine background; outside it nothing is known and the pixels are ``ignore``.

Two records make that statement: a ``CompletedROI`` polygon, and a
``RoiSegmentationStatus`` marked complete. Both are read here.

:mod:`quantem.finetune` consumes this; nothing here imports the trainer.
"""

from .extract_crops import (
    IGNORE,
    MODE_HEAD,
    MODE_THRESHOLD_ONLY,
    NO_PROBABILITY_MESSAGE,
    NOTHING_ANNOTATED_HERE,
    NOTHING_ANNOTATED_IN_SCOPE,
    SOURCE_CONFIRMED_AREA,
    SOURCE_DONE_ROI,
    AnnotatedCrop,
    CompletedRoiRequired,
    CropSet,
    collect_crops,
    collect_crops_for_scope,
    plan_split,
    require_crops,
    require_crops_for_scope,
)

__all__ = [
    "IGNORE",
    "MODE_HEAD",
    "MODE_THRESHOLD_ONLY",
    "NOTHING_ANNOTATED_HERE",
    "NOTHING_ANNOTATED_IN_SCOPE",
    "NO_PROBABILITY_MESSAGE",
    "SOURCE_CONFIRMED_AREA",
    "SOURCE_DONE_ROI",
    "AnnotatedCrop",
    "CompletedRoiRequired",
    "CropSet",
    "collect_crops",
    "collect_crops_for_scope",
    "plan_split",
    "require_crops",
    "require_crops_for_scope",
]

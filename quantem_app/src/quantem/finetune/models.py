"""The record of one adaptation: what was fit, on what, and how well it did.

An :class:`Adapter` is deliberately a *result* record rather than a model
wrapper. Everything the UI needs to present a number honestly — which crops the
threshold was fit on, whether the held-out score is image-disjoint, the per-crop
oracle ceiling — is stored with the number itself, because these are the values
that end up in a figure caption.

The weights, when there are any, are a small ``head.pt`` (neck + decoder only)
under the user data directory. The frozen encoder is not copied: it is already in
the registry cache, addressed by digest.
"""

from __future__ import annotations

from pathlib import Path

from django.db import models

from quantem.assets.models import TimeStampedModel
from quantem.core.config import MODELS_DIR
from quantem.finetune.storage import adapter_head_path

__all__ = ["Adapter", "active_adapter_for", "adapter_head_path"]

#: Rungs of the guided fine-tuning ladder.
MODE_THRESHOLD_ONLY = "threshold_only"
MODE_HEAD = "head"
MODE_CHOICES = [
    (MODE_THRESHOLD_ONLY, "Calibrated threshold only"),
    (MODE_HEAD, "Head-only training"),
]

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_CHOICES = [
    (STATUS_PENDING, "Pending"),
    (STATUS_RUNNING, "Running"),
    (STATUS_SUCCESS, "Success"),
    (STATUS_FAILED, "Failed"),
]

#: How the held-out crops were separated from the fitted ones. Never omitted
#: from a response: "within-image" and "image-disjoint" are different claims.
SPLIT_IMAGE_DISJOINT = "image-disjoint"
SPLIT_WITHIN_IMAGE = "within-image"
SPLIT_NO_HELDOUT = "no-heldout"
SPLIT_CHOICES = [
    (SPLIT_IMAGE_DISJOINT, "Image-disjoint"),
    (SPLIT_WITHIN_IMAGE, "Within image"),
    (SPLIT_NO_HELDOUT, "No held-out data"),
]


class Adapter(TimeStampedModel):
    """One fit of a released model to the user's own annotated crops."""

    #: Segmentation the adaptation was started from. Kept so "apply this
    #: adapter" has something to apply it to; the crops themselves may come from
    #: sibling segmentations of the same organelle on other images.
    segmentation = models.ForeignKey(
        "segmentation.ImageSegmentation",
        on_delete=models.CASCADE,
        related_name="adapters",
        null=True,
        blank=True,
    )
    base_model = models.CharField(max_length=64)  # e.g. "quantem:mito"
    name = models.CharField(max_length=255, blank=True)
    status = models.CharField(
        max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING
    )
    mode = models.CharField(
        max_length=32, choices=MODE_CHOICES, default=MODE_THRESHOLD_ONLY
    )
    #: Requested hyper-parameters (steps, lr, seed, ...), as submitted.
    params = models.JSONField(default=dict, blank=True)
    #: The full threshold sweep: every threshold tried, the train curve, the
    #: held-out score at the chosen point, the per-crop oracle ceiling and which
    #: crops were fitted. See :class:`quantem.finetune.calibrate.SweepResult`.
    sweep = models.JSONField(default=dict, blank=True)
    calibrated_threshold = models.FloatField(null=True, blank=True)
    split_mode = models.CharField(
        max_length=24, choices=SPLIT_CHOICES, default=SPLIT_NO_HELDOUT
    )
    #: Relative to MODELS_DIR; empty for a threshold-only adapter, which has no
    #: weights at all.
    head_path = models.CharField(max_length=1024, blank=True)
    #: True once the saved head has been reloaded onto a fresh encoder and
    #: re-scored to the same Dice. The reference asserted this; an adapter that
    #: cannot reproduce its own number is not shippable.
    verified_reload = models.BooleanField(default=False)
    trainable_params = models.BigIntegerField(null=True, blank=True)
    train_seconds = models.FloatField(null=True, blank=True)
    error = models.TextField(blank=True)
    #: Set when the user chooses this adapter for subsequent runs. The most
    #: recently applied adapter for a segmentation is the active one.
    applied_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["segmentation", "status"]),
            models.Index(fields=["segmentation", "applied_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.name or self.base_model} ({self.mode}, {self.status})"

    @property
    def heldout_dice(self) -> float | None:
        """Held-out Dice at the calibrated threshold, or None if not held out.

        Meaningless without :attr:`split_mode` beside it, which is why every
        serializer here emits the two together.
        """
        value = (self.sweep or {}).get("heldout_dice_at_calibrated")
        return float(value) if value is not None else None

    @property
    def head_file(self) -> Path | None:
        """Absolute path to the trained head, or None for a threshold-only adapter."""
        if not self.head_path:
            return None
        path = Path(self.head_path)
        return path if path.is_absolute() else MODELS_DIR / path

    def caveats(self) -> list[str]:
        """Everything a reader must know before quoting a number from here."""
        notes = [
            "The threshold was fit on the training crops only; the held-out "
            "score is reported at that threshold, never used to choose it."
        ]
        if self.split_mode == SPLIT_WITHIN_IMAGE:
            notes.append(
                "The held-out crops come from the same image as the training "
                "crops, so this is a within-image score and does not measure "
                "generalisation to a new image."
            )
        elif self.split_mode == SPLIT_NO_HELDOUT:
            notes.append(
                "Every annotated region was used to fit the threshold, so there "
                "is no held-out score at all."
            )
        if (self.sweep or {}).get("heldout_oracle") is not None:
            notes.append(
                "The oracle is the best achievable with a threshold chosen per "
                "crop using the answers. It is a ceiling, not a target."
            )
        if self.mode == MODE_HEAD and not self.verified_reload:
            notes.append(
                "The saved head was not re-scored after reloading, so these "
                "numbers are from the in-memory model only."
            )
        return notes


def active_adapter_for(segmentation) -> Adapter | None:
    """The adapter currently applied to a segmentation, if any."""
    return (
        Adapter.objects.filter(
            segmentation=segmentation,
            status=STATUS_SUCCESS,
            applied_at__isnull=False,
        )
        .order_by("-applied_at")
        .first()
    )

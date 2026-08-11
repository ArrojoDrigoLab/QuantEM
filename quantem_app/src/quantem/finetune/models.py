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

__all__ = [
    "DEFAULT_USE_ALL_MAX_TILES",
    "TRAINING_MODES",
    "TRAINING_MODE_HOLDOUT_1",
    "TRAINING_MODE_USE_ALL",
    "Adapter",
    "active_adapter_for",
    "adapter_head_path",
]

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
#:
#: The same vocabulary answers "what was held out": by image when the scope has
#: more than one annotated image (``image-disjoint``), by tile when it has only
#: one (``within-image``), and nothing at all under *use all*
#: (``no-heldout``). That is the whole distinction, so there is no second field.
SPLIT_IMAGE_DISJOINT = "image-disjoint"
SPLIT_WITHIN_IMAGE = "within-image"
SPLIT_NO_HELDOUT = "no-heldout"
SPLIT_CHOICES = [
    (SPLIT_IMAGE_DISJOINT, "Image-disjoint"),
    (SPLIT_WITHIN_IMAGE, "Within image"),
    (SPLIT_NO_HELDOUT, "No held-out data"),
]

#: What the fine-tune does with the annotated areas it was given.
TRAINING_MODE_USE_ALL = "use_all"
TRAINING_MODE_HOLDOUT_1 = "holdout_1"
TRAINING_MODE_CHOICES = [
    (TRAINING_MODE_USE_ALL, "Use all"),
    (TRAINING_MODE_HOLDOUT_1, "Hold out one"),
]
TRAINING_MODES = (TRAINING_MODE_USE_ALL, TRAINING_MODE_HOLDOUT_1)

#: At or below this many tiles the dialog defaults to *use all*; above it, to
#: *hold out 1*.
#:
#: Owner R13 gave the two ends and not the middle: "use all at <= 3 tiles,
#: hold-out 1 at > 4 tiles". Four tiles was unstated. The round-3 contract
#: resolves it as hold-out, so the rule here is a single boundary at 3 rather
#: than a third case invented for one value. Holding out is the safer default at
#: the boundary: it costs one tile of training data and buys a number that was
#: not fitted on itself, and a user who wants the tile back can pick *use all*.
DEFAULT_USE_ALL_MAX_TILES = 3

#: Below this many tiles a cross-validated mean is a weak estimate and has to
#: say so. Four folds over four tiles is four numbers, each from one tile.
WEAK_CV_TILE_COUNT = 5


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
    #: The organelle this fine-tune is for. One at a time, always. Null on rows
    #: written before named fine-tunes existed, where the organelle is only
    #: reachable through :attr:`segmentation`.
    segmentation_type = models.ForeignKey(
        "segmentation.SegmentationType",
        on_delete=models.SET_NULL,
        related_name="adapters",
        null=True,
        blank=True,
    )
    #: The experiment every image in the scope belongs to. Null when the scope
    #: is unassigned images, which is its own bucket and not an error.
    experiment = models.ForeignKey(
        "library.Experiment",
        on_delete=models.SET_NULL,
        related_name="adapters",
        null=True,
        blank=True,
    )
    #: The images the user chose. Datasets are stored beside the assets they
    #: expanded to, not instead of them: a dataset that gains an image later must
    #: not silently change what an existing fine-tune claims it was trained on.
    scope_assets = models.ManyToManyField(
        "assets.Asset", blank=True, related_name="adapters_scoped"
    )
    scope_datasets = models.ManyToManyField(
        "library.Dataset", blank=True, related_name="adapters_scoped"
    )
    base_model = models.CharField(max_length=64)  # e.g. "quantem:mito"
    #: Identifies a fine-tune for overwrite. Unique per organelle among named
    #: rows -- the same name for mitochondria and for ER is two different
    #: fine-tunes, and that is deliberate.
    name = models.CharField(max_length=255, blank=True)
    status = models.CharField(
        max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING
    )
    #: Whether every annotated area was trained on, or one was held back.
    training_mode = models.CharField(
        max_length=16, choices=TRAINING_MODE_CHOICES, default=TRAINING_MODE_USE_ALL
    )
    #: Rotate the hold-out over every unit and report the average.
    cv_benchmark = models.BooleanField(default=False)
    #: ``{"folds": [...], "mean": {...}, "per_image": [...]}``. Per-image results
    #: are required, not optional: an average over images hides the one the model
    #: cannot do, which is the one the user needs to know about.
    cv_results = models.JSONField(default=dict, blank=True)
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
            models.Index(fields=["segmentation_type", "status"]),
        ]
        constraints = [
            # Named fine-tunes only. Every row written before this feature has a
            # blank name and a null type, and several of them legitimately
            # coexist -- a constraint that counted those would refuse to migrate
            # an existing library.
            models.UniqueConstraint(
                fields=["name", "segmentation_type"],
                condition=models.Q(segmentation_type__isnull=False)
                & ~models.Q(name=""),
                name="unique_finetune_name_per_organelle",
            )
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
        notes.extend(self.cv_caveats())
        return notes

    def cv_caveats(self) -> list[str]:
        """What a reader must know before quoting the cross-validated average.

        A mean over a handful of folds is a mean of a handful of numbers, each
        measured on one held-out area. It is the honest estimate available from
        this much data and it is not a benchmark; saying so is the difference
        between a figure caption that survives review and one that does not.
        """
        results = self.cv_results if isinstance(self.cv_results, dict) else {}
        folds = results.get("folds")
        if not isinstance(folds, list) or not folds:
            return []
        tiles = sum(
            int(fold.get("n_tiles") or 0) for fold in folds if isinstance(fold, dict)
        )
        notes = [
            f"The average below is over {len(folds)} rounds, each scored on the "
            "one area held back from it."
        ]
        if tiles < WEAK_CV_TILE_COUNT:
            notes.append(
                f"Only {tiles} training area(s) took part, so the average is a "
                "weak estimate: it varies a lot with which area happens to be "
                "held out. Read the per-image numbers rather than the mean."
            )
        return notes


def active_adapter_for(segmentation) -> Adapter | None:
    """The adapter currently applied to a segmentation, if any.

    Two ways an adapter can be applied to this segmentation, and the more
    recently applied of the two wins:

    * it was fitted **from** this segmentation, the original single-image path,
      matched on the ``segmentation`` foreign key;
    * it is a named fine-tune for this organelle whose scope includes this
      image, and the user chose to run it here. A scoped fine-tune covers many
      images and cannot point its one foreign key at all of them, so the match
      is on ``(organelle, image)``.

    Both are filtered to ``SUCCESS`` with ``applied_at`` set, so an adapter that
    is merely finished is never used until the user says to use it.
    """
    direct = (
        Adapter.objects.filter(
            segmentation=segmentation,
            status=STATUS_SUCCESS,
            applied_at__isnull=False,
        )
        .order_by("-applied_at")
        .first()
    )
    asset_id = getattr(segmentation, "asset_id", None)
    type_id = getattr(segmentation, "segmentation_type_id", None)
    scoped = None
    if asset_id is not None and type_id is not None:
        scoped = (
            Adapter.objects.filter(
                segmentation_type_id=type_id,
                scope_assets__id=asset_id,
                status=STATUS_SUCCESS,
                applied_at__isnull=False,
            )
            .order_by("-applied_at")
            .first()
        )
    candidates = [a for a in (direct, scoped) if a is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.applied_at)

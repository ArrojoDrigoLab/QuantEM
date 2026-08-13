"""Experiments and datasets.

``Experiment`` is the outer grouping -- one preparation, one cohort, one
sitting at the microscope. ``Dataset`` is a named subset of one experiment.
Every active image points at one experiment and belongs to any number of that
experiment's datasets. An import without an explicit experiment receives its
own experiment named after the image.

The one rule worth enforcing in code is that an image's datasets cannot
disagree with its experiment -- see :func:`validate_asset_grouping`. Without it
"every image in this dataset" and "every image in this experiment" can return
overlapping-but-inconsistent sets, and a fine-tune scoped to one experiment
would silently train on an image from another.
"""

from __future__ import annotations

from django.db import IntegrityError, models, transaction

from quantem.assets.models import TimeStampedModel

__all__ = [
    "Experiment",
    "Dataset",
    "create_image_experiment",
    "validate_asset_grouping",
]


class Experiment(TimeStampedModel):
    """One experiment. The boundary a fine-tune may not cross."""

    name = models.CharField(max_length=255, unique=True)
    notes = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["name", "created_at"]

    def __str__(self) -> str:
        return self.name


class Dataset(TimeStampedModel):
    """A named subset of one experiment's images."""

    experiment = models.ForeignKey(
        Experiment,
        on_delete=models.CASCADE,
        related_name="datasets",
    )
    name = models.CharField(max_length=255)
    notes = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["name", "created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["experiment", "name"],
                name="unique_dataset_name_per_experiment",
            )
        ]

    def __str__(self) -> str:
        return f"{self.experiment.name} / {self.name}"


def create_image_experiment(display_name: str) -> Experiment:
    """Create the unique experiment owned by one otherwise-unfiled image.

    Experiment names are globally unique. The image name is used exactly when
    available; duplicate names receive `` (2)``, `` (3)``, and so on. Each
    create attempt has its own savepoint so concurrent imports that choose the
    same candidate can retry cleanly after the uniqueness constraint wins.
    """
    base = str(display_name or "").strip() or "Untitled image"
    for number in range(1, 100_000):
        suffix = "" if number == 1 else f" ({number})"
        candidate = f"{base[: max(1, 255 - len(suffix))]}{suffix}"
        try:
            with transaction.atomic():
                return Experiment.objects.create(name=candidate)
        except IntegrityError:
            continue
    raise RuntimeError("Could not allocate a unique experiment name for this image.")


def validate_asset_grouping(asset) -> None:
    """Raise if ``asset``'s datasets do not all belong to its experiment.

    Called from the serializer and from anywhere that assigns datasets in bulk.
    Not a database constraint: SQLite cannot express "every row in this m2m
    agrees with a column on the other side", and a signal on ``m2m_changed``
    would fire during fixture loading and migrations where the two sides are
    legitimately half-written.
    """
    from django.core.exceptions import ValidationError

    dataset_experiments = {dataset.experiment_id for dataset in asset.datasets.all()}
    if not dataset_experiments:
        return
    if asset.experiment_id is None:
        raise ValidationError(
            "This image is in a dataset but has no experiment. Choose the "
            "experiment the dataset belongs to, or remove the image from it."
        )
    stray = dataset_experiments - {asset.experiment_id}
    if stray:
        raise ValidationError(
            "This image is in a dataset that belongs to a different "
            "experiment. An image can only be in datasets from its own "
            "experiment."
        )

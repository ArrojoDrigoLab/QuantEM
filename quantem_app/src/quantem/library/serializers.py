"""Read and write shapes for experiments and datasets.

Serialized by hand, like ``assets/serializers.py`` and for the same reason: an
experiment payload is the row plus counts joined from two other tables, and the
counts are the whole point of the payload. A dropdown that cannot say "12
images" is a dropdown the user has to guess at.

The counts are of **active** images only. A tombstoned asset is not in the
library any more, and an experiment reading "3 images" over two visible cards
is the kind of small lie that makes a user stop trusting the screen.
"""

from __future__ import annotations

from typing import Any

from rest_framework import serializers

from quantem.library.models import Dataset, Experiment

__all__ = [
    "DatasetWriteSerializer",
    "ExperimentWriteSerializer",
    "serialize_dataset",
    "serialize_experiment",
]


def _isoformat(value) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


def _active_asset_count(manager) -> int:
    """How many images in this group are still in the library."""
    from quantem.assets.models import Asset

    return manager.filter(lifecycle_status=Asset.LIFECYCLE_ACTIVE).count()


def serialize_dataset(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": str(dataset.id),
        "experiment": str(dataset.experiment_id),
        "name": dataset.name,
        "notes": dataset.notes,
        "asset_count": _active_asset_count(dataset.assets),
        "created_at": _isoformat(dataset.created_at),
        "updated_at": _isoformat(dataset.updated_at),
    }


def serialize_experiment(experiment: Experiment) -> dict[str, Any]:
    """One experiment with its datasets nested.

    Nested rather than fetched separately because the library is a desktop
    library -- tens of experiments, not thousands -- and every screen that
    wants one wants the other in the same breath: the filter, the import form's
    two pickers, and the scope tree a fine-tune is chosen with.

    ``ungrouped_asset_count`` is images in the experiment and in none of its
    datasets. It is a real bucket, not a rounding error: an experiment whose
    images have not been split into datasets yet is the normal state right
    after an import.
    """
    datasets = list(experiment.datasets.all())
    dataset_payloads = [serialize_dataset(dataset) for dataset in datasets]
    asset_count = _active_asset_count(experiment.assets)
    grouped_ids: set = set()
    for dataset in datasets:
        grouped_ids.update(
            dataset.assets.filter(lifecycle_status="ACTIVE").values_list("id", flat=True)
        )
    return {
        "id": str(experiment.id),
        "name": experiment.name,
        "notes": experiment.notes,
        "datasets": dataset_payloads,
        "asset_count": asset_count,
        "ungrouped_asset_count": max(asset_count - len(grouped_ids), 0),
        "created_at": _isoformat(experiment.created_at),
        "updated_at": _isoformat(experiment.updated_at),
    }


class ExperimentWriteSerializer(serializers.Serializer):
    """Validates a create or an update of one experiment."""

    name = serializers.CharField(max_length=255, required=True, trim_whitespace=True)
    notes = serializers.CharField(
        required=False, allow_blank=True, default="", trim_whitespace=False
    )

    def validate_name(self, value: str) -> str:
        name = value.strip()
        if not name:
            raise serializers.ValidationError("An experiment needs a name. Type one and try again.")
        clashes = Experiment.objects.filter(name__iexact=name)
        if self.instance is not None:
            clashes = clashes.exclude(id=self.instance.id)
        if clashes.exists():
            raise serializers.ValidationError(
                "There is already an experiment with that name. Pick a "
                "different one, or use the existing experiment."
            )
        return name


class DatasetWriteSerializer(serializers.Serializer):
    """Validates a create or an update of one dataset.

    ``experiment`` is required on create and refused on update: moving a
    dataset between experiments would silently take every image in it with it,
    or leave the images contradicting their own experiment. Neither is
    something a rename box should be able to do, so the move is simply not
    offered.
    """

    experiment = serializers.UUIDField(required=False)
    name = serializers.CharField(max_length=255, required=True, trim_whitespace=True)
    notes = serializers.CharField(
        required=False, allow_blank=True, default="", trim_whitespace=False
    )

    def validate_name(self, value: str) -> str:
        name = value.strip()
        if not name:
            raise serializers.ValidationError("A dataset needs a name. Type one and try again.")
        return name

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        experiment_id = attrs.get("experiment")
        if self.instance is None:
            if not experiment_id:
                raise serializers.ValidationError(
                    "A dataset lives inside an experiment. Choose one, then name the dataset."
                )
            if not Experiment.objects.filter(id=experiment_id).exists():
                raise serializers.ValidationError(
                    "That experiment is no longer in the library. Refresh and pick another one."
                )
            experiment_filter = {"experiment_id": experiment_id}
        else:
            experiment_filter = {"experiment_id": self.instance.experiment_id}

        clashes = Dataset.objects.filter(name__iexact=attrs["name"], **experiment_filter)
        if self.instance is not None:
            clashes = clashes.exclude(id=self.instance.id)
        if clashes.exists():
            raise serializers.ValidationError(
                "That experiment already has a dataset with this name. Pick a "
                "different one, or use the existing dataset."
            )
        return attrs

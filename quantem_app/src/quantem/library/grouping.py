"""Putting images into experiments and datasets, and taking them out again.

Every write that touches :attr:`Asset.experiment` or :attr:`Asset.datasets`
goes through :func:`apply_grouping`, so the one rule the models document --
an image's datasets all belong to its experiment -- is enforced in exactly one
place rather than at each call site. The import door, the library's bulk
assignment and the tests all use it.

Three states, not two
---------------------

Each of ``experiment`` and ``datasets`` is a *tri-state*: absent means "leave
this alone", ``None``/``[]`` means "clear it", and a value means "set it". A
two-state parameter cannot express "move these forty images into an experiment
and do not touch which datasets they are in", which is the commonest thing the
library asks for.

Moving an image to another experiment
-------------------------------------

An image that sits in "Liver 24h" and is then moved to another experiment
cannot stay in "Liver 24h": a dataset belongs to exactly one experiment, so the
membership would be a contradiction rather than a preference. Refusing the move
was the other option and it is worse -- it makes the user go and empty the
datasets by hand before they are allowed to do the thing they asked for, for no
gain, since the outcome is the same either way.

So the move **drops the dataset memberships that the new experiment cannot
hold**, counts them, and returns the count so the caller can say so before and
after. Nothing else about the image changes, and memberships in datasets that
*do* belong to the new experiment survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction

from quantem.library.models import (
    Dataset,
    Experiment,
    create_image_experiment,
    validate_asset_grouping,
)

__all__ = [
    "UNSET",
    "GroupingOutcome",
    "apply_grouping",
    "resolve_dataset",
    "resolve_experiment",
]


class _Unset:
    """Sentinel for "the caller did not mention this field"."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"

    def __bool__(self) -> bool:
        return False


#: "Leave this field exactly as it is." Distinct from ``None``, which clears it.
UNSET = _Unset()


@dataclass
class GroupingOutcome:
    """What one grouping write actually did."""

    #: Images whose experiment or dataset membership came out different.
    assets_changed: int = 0
    #: Dataset memberships dropped because the image left their experiment.
    dataset_links_dropped: int = 0
    #: Images that lost at least one dataset for that reason.
    assets_moved_out_of_datasets: int = 0
    #: Names of the datasets those images were taken out of, for the copy.
    datasets_left: list[str] = field(default_factory=list)


def resolve_experiment(
    *,
    experiment_id: Any = None,
    experiment_name: Any = None,
) -> Experiment | None:
    """Find or create the experiment an id or a typed name refers to.

    An id must exist -- a client holding one that does not is out of date, and
    silently creating a row under that id would invent an experiment nobody
    named. A typed name is matched case-insensitively first, so typing "Fasted
    cohort" twice does not produce two experiments that look identical in a
    dropdown.

    Returns ``None`` when neither was given. Import and assignment services
    interpret that as "create an experiment from each image name".
    """
    if experiment_id:
        try:
            return Experiment.objects.get(id=experiment_id)
        except (Experiment.DoesNotExist, ValueError, TypeError) as exc:
            raise ValidationError(
                "That experiment is no longer in the library. Refresh and pick "
                "another one, or type a new name."
            ) from exc

    name = str(experiment_name or "").strip()
    if not name:
        return None
    existing = Experiment.objects.filter(name__iexact=name).first()
    if existing is not None:
        return existing
    return Experiment.objects.create(name=name)


def resolve_dataset(
    *,
    experiment: Experiment | None,
    dataset_id: Any = None,
    dataset_name: Any = None,
) -> Dataset | None:
    """Find or create one dataset, inside ``experiment``.

    A dataset cannot exist outside an experiment, so naming one without also
    naming an experiment is refused here rather than left to produce a
    confusing failure further in.
    """
    if dataset_id:
        try:
            dataset = Dataset.objects.select_related("experiment").get(id=dataset_id)
        except (Dataset.DoesNotExist, ValueError, TypeError) as exc:
            raise ValidationError(
                "That dataset is no longer in the library. Refresh and pick "
                "another one, or type a new name."
            ) from exc
        if experiment is not None and dataset.experiment_id != experiment.id:
            raise ValidationError(
                f"The dataset “{dataset.name}” belongs to a different "
                "experiment. Pick a dataset from the experiment you chose, or "
                "type a new name to create one there."
            )
        return dataset

    name = str(dataset_name or "").strip()
    if not name:
        return None
    if experiment is None:
        raise ValidationError(
            "A dataset lives inside an experiment. Choose or name an "
            "experiment as well, then this dataset can be created in it."
        )
    existing = Dataset.objects.filter(experiment=experiment, name__iexact=name).first()
    if existing is not None:
        return existing
    return Dataset.objects.create(experiment=experiment, name=name)


def apply_grouping(
    assets,
    *,
    experiment: Experiment | None | _Unset = UNSET,
    datasets: list[Dataset] | None | _Unset = UNSET,
    datasets_mode: str = "replace",
) -> GroupingOutcome:
    """Set the experiment and datasets of every asset in ``assets``.

    Raises :class:`~django.core.exceptions.ValidationError` with a sentence fit
    for a person if the result would break the grouping rule; the whole write
    is rolled back, so a bulk assignment never half-lands.

    ``datasets_mode`` is ``"replace"`` (the membership becomes exactly this
    set) or ``"add"`` (these are added to whatever is already there). Replace
    is the default because that is what a picker showing the current value
    means when the user changes it.
    """
    if datasets_mode not in {"replace", "add"}:  # pragma: no cover - caller bug
        raise ValueError(f"Unknown datasets mode: {datasets_mode!r}")

    outcome = GroupingOutcome()
    requested_datasets = [] if datasets is None else datasets
    left_dataset_names: set[str] = set()

    with transaction.atomic():
        for asset in assets:
            before_experiment = asset.experiment_id
            before_datasets = {dataset.id for dataset in asset.datasets.all()}

            if not isinstance(experiment, _Unset):
                # ``None`` means "give each image its own experiment", not an
                # unassigned state. A bulk selection deliberately receives
                # one experiment per image, each named from that image.
                asset.experiment = (
                    experiment
                    if experiment is not None
                    else create_image_experiment(asset.display_name)
                )
                asset.save(update_fields=["experiment", "updated_at"])

            # Whatever the experiment now is, memberships that contradict it
            # cannot survive. This is the move rule in the module docstring,
            # and it runs before the requested datasets are applied so a move
            # and an assignment in the same request behave like a move followed
            # by an assignment.
            stray = [
                dataset
                for dataset in asset.datasets.all()
                if dataset.experiment_id != asset.experiment_id
            ]
            if stray:
                asset.datasets.remove(*stray)
                outcome.dataset_links_dropped += len(stray)
                outcome.assets_moved_out_of_datasets += 1
                left_dataset_names.update(dataset.name for dataset in stray)

            if not isinstance(datasets, _Unset):
                if datasets_mode == "replace":
                    asset.datasets.set(requested_datasets)
                elif requested_datasets:
                    asset.datasets.add(*requested_datasets)

            validate_asset_grouping(asset)

            after_datasets = {dataset.id for dataset in asset.datasets.all()}
            if asset.experiment_id != before_experiment or after_datasets != before_datasets:
                outcome.assets_changed += 1

    outcome.datasets_left = sorted(left_dataset_names)
    return outcome

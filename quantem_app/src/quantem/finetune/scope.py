"""Which images a fine-tune is over, how much work that is, and how it is split.

A fine-tune used to be scoped to one segmentation and, through it, to every
annotation of the same organelle anywhere in the library. That is the wrong unit
for a question a microscopist actually asks -- *fit this to my fasted cohort* --
and it silently mixed conditions. This module is the replacement: an explicit
set of images, inside one experiment, with the counts the dialog shows and the
fold plan the run follows.

Three things live here and nowhere else, because each of them is read in more
than one place and must not be answered twice:

* **the scope**, resolved from datasets plus individual images, with the
  same-experiment rule applied to the result;
* **the tile count**, which is what ``build_patches`` will cut, computed through
  the trainer's own window rule rather than a second estimate of it;
* **the fold plan**, which decides how many training-plus-evaluation rounds the
  run will do -- and therefore the denominator the progress bar counts to, which
  the queue asks for at enqueue time, before the worker exists.

Nothing here loads an image or a model. The dialog answers immediately.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from quantem.finetune.adapt import (
    AdaptConfig,
    iter_windows,
    masks_to_model_scale,
    pad_masks_to_tile,
    tile_for,
)
from quantem.inference import resample
from quantem.inference.specs import MODEL_SPECS
from quantem.segmentation.services.adapt import AnnotatedCrop

logger = logging.getLogger(__name__)

__all__ = [
    "ResolvedScope",
    "TrainingFold",
    "count_tiles",
    "default_training_mode",
    "plan_folds",
    "planned_round_count",
    "planned_training_units",
    "resolve_scope",
    "tiles_for_crop",
]


# ---------------------------------------------------------------------------
# The scope
# ---------------------------------------------------------------------------


#: The one selectable group that is not an experiment. Images with no experiment
#: are their own bucket, and the same-experiment rule applies to them as a group
#: -- so an unassigned image cannot be mixed with an experiment's image either.
UNASSIGNED = "unassigned"


@dataclass
class ResolvedScope:
    """The images a run will use, and whether the selection is allowed at all."""

    asset_ids: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    #: The one experiment every image belongs to, or None for unassigned images.
    experiment_id: str | None = None
    experiment_name: str = ""
    #: Plain-language reasons the selection cannot be run. Empty means it can.
    blockers: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return not self.blockers and bool(self.asset_ids)


def resolve_scope(
    *,
    asset_ids: Sequence[str] | None = None,
    dataset_ids: Sequence[str] | None = None,
) -> ResolvedScope:
    """Expand datasets to their images, union with the chosen images, and check.

    The same-experiment rule (owner R13) is enforced here rather than at each
    call site, because it is a property of the selection and every entry point
    -- preview, start, apply -- has to agree about it.
    """
    from quantem.assets.models import Asset  # noqa: PLC0415 -- Django app registry
    from quantem.library.models import Dataset  # noqa: PLC0415

    asset_ids = [str(value) for value in (asset_ids or [])]
    dataset_ids = [str(value) for value in (dataset_ids or [])]
    scope = ResolvedScope(dataset_ids=dataset_ids)

    known_datasets = set()
    if dataset_ids:
        known_datasets = {
            str(value)
            for value in Dataset.objects.filter(id__in=dataset_ids).values_list(
                "id", flat=True
            )
        }
        missing = [value for value in dataset_ids if value not in known_datasets]
        if missing:
            scope.blockers.append(
                "One of the groups you chose is no longer in the library. Close "
                "this and choose again."
            )

    resolved: set[str] = set()
    if known_datasets:
        resolved.update(
            str(value)
            for value in Asset.objects.filter(
                datasets__id__in=known_datasets, lifecycle_status="ACTIVE"
            ).values_list("id", flat=True)
        )
    if asset_ids:
        found = {
            str(value)
            for value in Asset.objects.filter(
                id__in=asset_ids, lifecycle_status="ACTIVE"
            ).values_list("id", flat=True)
        }
        if len(found) != len(set(asset_ids)):
            scope.blockers.append(
                "One of the images you chose is no longer in the library. Close "
                "this and choose again."
            )
        resolved.update(found)

    scope.asset_ids = sorted(resolved)
    if not scope.asset_ids:
        scope.blockers.append("Choose at least one image to fine-tune on.")
        return scope

    experiments = {
        (str(value) if value else UNASSIGNED)
        for value in Asset.objects.filter(id__in=scope.asset_ids).values_list(
            "experiment_id", flat=True
        )
    }
    if len(experiments) > 1:
        scope.blockers.append(
            "The images you chose are from more than one experiment. A fine-tune "
            "covers one experiment at a time, so pick images from just one of "
            "them."
        )
        return scope

    only = next(iter(experiments))
    if only != UNASSIGNED:
        from quantem.library.models import Experiment  # noqa: PLC0415

        scope.experiment_id = only
        scope.experiment_name = (
            Experiment.objects.filter(id=only)
            .values_list("name", flat=True)
            .first()
            or ""
        )
    return scope


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------


def tiles_for_crop(
    crop: AnnotatedCrop,
    base_model: str,
    *,
    config: AdaptConfig = AdaptConfig(),
) -> int:
    """How many training windows one annotated area is worth for this pack.

    Exactly what :func:`quantem.finetune.adapt.build_patches` will produce: the
    same resample, the same padding, the same window rule -- reached through
    :func:`quantem.finetune.adapt.iter_windows`, which both call. Needs the label
    masks only, never the image, so the dialog can answer before anything is
    decoded off disk.
    """
    spec = MODEL_SPECS.get(base_model)
    if spec is None:
        return 0
    tile = tile_for(spec.tile_size, spec.patch_size)
    context = resample.plan_resample(
        crop.valid.shape[:2], crop.pixel_size_nm, spec.canonical_nm
    )
    gt, valid = masks_to_model_scale(crop.gt, crop.valid, context)
    _gt, valid = pad_masks_to_tile(gt, valid, tile)
    return sum(1 for _ in iter_windows(valid, tile, config=config))


def count_tiles(
    crops: Sequence[AnnotatedCrop],
    base_model: str,
    *,
    config: AdaptConfig = AdaptConfig(),
) -> int:
    return sum(tiles_for_crop(crop, base_model, config=config) for crop in crops)


def default_training_mode(tile_count: int) -> str:
    """*Use all* at or below the boundary, *hold out 1* above it.

    The boundary itself is :data:`quantem.finetune.models
    .DEFAULT_USE_ALL_MAX_TILES`, where the reasoning for its value lives.
    """
    from quantem.finetune.models import (  # noqa: PLC0415 -- Django app registry
        DEFAULT_USE_ALL_MAX_TILES,
        TRAINING_MODE_HOLDOUT_1,
        TRAINING_MODE_USE_ALL,
    )

    return (
        TRAINING_MODE_USE_ALL
        if int(tile_count) <= DEFAULT_USE_ALL_MAX_TILES
        else TRAINING_MODE_HOLDOUT_1
    )


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingFold:
    """One training-plus-evaluation round."""

    index: int
    train: list[AnnotatedCrop]
    heldout: list[AnnotatedCrop]
    #: The asset held out, when the split is by image. None when it is by tile,
    #: because then the held-out area's image is also in the training set and
    #: naming it would imply a generalisation claim the split cannot support.
    held_out_asset_id: str | None


def _hold_out_units(
    crops: Sequence[AnnotatedCrop],
) -> tuple[list[list[AnnotatedCrop]], str]:
    """Group the crops into the things that can be held out, and say which.

    Hold out by **image** whenever the scope has more than one annotated image:
    that is the only split that measures generalisation to an image the model
    has not seen. With a single annotated image the units are individual tiles
    and the resulting score is within-image, which is a weaker claim and is
    recorded as such in ``split_mode`` rather than presented as the same number.
    """
    from quantem.finetune.models import (  # noqa: PLC0415 -- Django app registry
        SPLIT_IMAGE_DISJOINT,
        SPLIT_NO_HELDOUT,
        SPLIT_WITHIN_IMAGE,
    )

    crops = list(crops)
    by_image: dict[str, list[AnnotatedCrop]] = {}
    for crop in crops:
        by_image.setdefault(crop.image_key, []).append(crop)

    if len(by_image) >= 2:
        return [by_image[key] for key in sorted(by_image)], SPLIT_IMAGE_DISJOINT
    if len(crops) >= 2:
        return [[crop] for crop in crops], SPLIT_WITHIN_IMAGE
    # One area, so there is nothing to hold back that still leaves something to
    # train on. Reported honestly rather than fudged into a one-crop "split".
    return [], SPLIT_NO_HELDOUT


def plan_folds(
    crops: Sequence[AnnotatedCrop],
    *,
    training_mode: str,
    cv_benchmark: bool = False,
) -> tuple[list[TrainingFold], str]:
    """The rounds this run will do, and the ``split_mode`` they add up to.

    * *use all* is one round over everything, with nothing held back.
    * *hold out 1* holds one unit back and is one round.
    * *hold out 1* with cross-validation rotates the hold-out over every unit,
      so every one is held out exactly once.

    Falls back to a single no-hold-out round when the data cannot support a
    split at all, so a run never fails over a choice the dialog offered.
    """
    from quantem.finetune.models import (  # noqa: PLC0415 -- Django app registry
        SPLIT_IMAGE_DISJOINT,
        SPLIT_NO_HELDOUT,
        TRAINING_MODE_HOLDOUT_1,
    )

    crops = list(crops)
    everything = [TrainingFold(0, crops, [], None)]
    if training_mode != TRAINING_MODE_HOLDOUT_1 or len(crops) < 2:
        return everything, SPLIT_NO_HELDOUT

    units, split_mode = _hold_out_units(crops)
    if not units:
        return everything, SPLIT_NO_HELDOUT

    chosen = units if cv_benchmark else units[:1]
    folds: list[TrainingFold] = []
    for index, held in enumerate(chosen):
        held_names = {crop.name for crop in held}
        train = [crop for crop in crops if crop.name not in held_names]
        if not train:
            continue
        # Only a by-image split names an image. Under a by-tile split the held
        # out area's image is also in the training set, so attributing the score
        # to that image would read as a generalisation claim it cannot support.
        held_out_asset_id = (
            held[0].image_key if split_mode == SPLIT_IMAGE_DISJOINT else None
        )
        folds.append(
            TrainingFold(
                index=index,
                train=train,
                heldout=list(held),
                held_out_asset_id=held_out_asset_id,
            )
        )
    if not folds:
        return everything, SPLIT_NO_HELDOUT
    return folds, split_mode


# ---------------------------------------------------------------------------
# The progress denominator
# ---------------------------------------------------------------------------


def planned_round_count(payload: dict) -> int:
    """How many training-plus-evaluation rounds this payload will run.

    One for *use all*, one for a plain hold-out, and one per held-out unit under
    cross-validation. The queue needs this at enqueue time, before a worker has
    touched the data, so the view that started the run records what it worked
    out (``planned_rounds``) and this only recomputes when that is absent -- an
    older payload, or a job enqueued by something other than the dialog.
    """
    recorded = payload.get("planned_rounds")
    try:
        if recorded is not None and int(recorded) > 0:
            return int(recorded)
    except (TypeError, ValueError):
        pass

    from quantem.finetune.models import (  # noqa: PLC0415 -- Django app registry
        TRAINING_MODE_HOLDOUT_1,
    )

    training_mode = str(payload.get("training_mode") or payload.get("mode") or "")
    if training_mode != TRAINING_MODE_HOLDOUT_1 or not payload.get("cv_benchmark"):
        return 1

    crops = _payload_crops(payload)
    if not crops:
        return 1
    units, _split_mode = _hold_out_units(crops)
    return max(1, len(units))


def is_scoped_payload(payload: dict) -> bool:
    """Whether this fine-tune payload is the scoped kind.

    The **same** discriminator ``quantem.finetune.job.adapter_job`` uses to pick
    a run path, and it has to stay the same one: only the scoped path writes
    steps, so a payload planned as counting steps that then runs the old path
    would publish a denominator nothing ever counts against -- and
    ``overall_percent`` prefers the unit percentage over the coarse one, so the
    Improve panel's bar would sit at 0 % for the whole run.
    """
    return bool(payload.get("asset_ids")) and bool(payload.get("segmentation_type_id"))


def planned_training_units(payload: dict) -> int | None:
    """``steps x rounds``: the one number a fine-tune's progress bar counts to.

    A single monotone total covers both halves of what the owner asked the bar
    to show -- how far through the steps, and how far through the rounds -- and
    a monotone total is what makes an ETA meaningful. Returns None for the older
    single-segmentation payload, which counts no units at all; that reads as
    "this job does not count units", which is what it has always said.
    """
    if not is_scoped_payload(payload):
        return None
    steps = payload.get("steps")
    try:
        steps = int(steps) if steps is not None else AdaptConfig.steps
    except (TypeError, ValueError):
        steps = AdaptConfig.steps
    if steps <= 0:
        return None
    return steps * planned_round_count(payload)


def _payload_crops(payload: dict) -> list[AnnotatedCrop]:
    """The annotated areas a scoped payload resolves to, labels only.

    Guarded: a denominator that cannot be worked out is reported as no
    denominator by the caller, never as a wrong one, and never as a failed
    enqueue.
    """
    from quantem.segmentation.services.adapt import (  # noqa: PLC0415
        collect_crops_for_scope,
    )

    if not is_scoped_payload(payload):
        return []
    try:
        return list(
            collect_crops_for_scope(
                str(payload["segmentation_type_id"]),
                [str(value) for value in payload["asset_ids"]],
            ).crops
        )
    except Exception:  # pragma: no cover - a planning read must never break enqueue
        logger.debug("could not read the scope's crops for planning", exc_info=True)
        return []

"""How much work a run will be, known before the run starts.

Why this exists
---------------
``progress_units_total`` used to be written by the run itself, from inside the
tiling loop's first callback. Everything downstream inherited that timing:

* a queued run could not say how big it was, so the run panel read
  ``waiting to start`` with no number while the user waited;
* worse, the whole-image rollup
  (:func:`quantem.jobs.serializers.aggregate_batch_progress`) could only count
  runs that had already begun. Press Run on three organelles and the wave
  reported the *first* one as the whole of the work: measured on this machine,
  three runs over one image (56 + 6 + 56 = 118 tiles) reported
  ``units_done 19, units_reachable 19, percent 100.0, runs_total 1`` and put
  **"Everything on montage16real 100% · 25 of 25 tiles"** on screen while a
  third run had not started and would fail.

The tiling plan does not need the model or the image data. It is fully
determined by the region shape, the pack's canonical nm/px, its tile size and
its patch size -- all of which are known the moment the job is enqueued. So it
is computed here, at enqueue, and the wave carries its whole denominator from
the moment the last run joins it.

The same number, not a second opinion
-------------------------------------
The estimate a queued run publishes has to be the number the loop will count
to, or the denominator moves under the user when the run starts. So this module
does not reimplement the arithmetic: it calls
``quantem.seg_core.db.inference._estimate_model_tile_count`` -- the exact
function the run calls -- with the same segmenter and the same region shape.
``test_tile_plan.py`` pins that equality on a real segmenter, and the import is
guarded so that an install without the inference stack (or a rename in a module
this app does not own) degrades to "no plan", which is where this started, and
never to a wrong plan.

Nothing here may raise. An enqueue that fails because progress could not be
estimated would be a progress feature breaking the thing it reports on.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["planned_units_for"]


def planned_units_for(job_type: str, payload: dict | None) -> tuple[int, str] | None:
    """``(count, unit)`` of work this job will do, or None when unknowable.

    Two units, and which one a job counts is a property of the job type:
    a segmentation run walks **tiles**, a fine-tune takes **steps**. They are
    never added together -- see ``quantem.jobs.models.BATCH_ROLLUP_JOB_TYPES``.

    None is the honest answer for a job type that counts neither, a segmentation
    that has been deleted between the click and the enqueue, an image with no
    readable rendition yet, and a pack this build does not know. The caller
    leaves the unit columns null, which reads as "this job does not count
    units" -- the same thing they said before this module existed.
    """
    from quantem.jobs.constants import (  # noqa: PLC0415 -- module-level cycle
        JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    )
    from quantem.jobs.models import (  # noqa: PLC0415 -- module-level cycle
        UNIT_PROGRESS_JOB_TYPES,
        UNIT_STEP,
        UNIT_TILE,
    )

    if job_type not in UNIT_PROGRESS_JOB_TYPES:
        return None
    if job_type == JOB_TYPE_TRAIN_ORGANELLE_ADAPTER:
        try:
            steps = _estimate_training_steps(payload or {})
        except Exception:
            logger.debug("could not plan steps for a fine-tune", exc_info=True)
            return None
        if not steps or steps <= 0:
            return None
        return int(steps), UNIT_STEP
    try:
        tiles = _estimate_tiles(payload or {})
    except Exception:
        # Enqueue must not fail because a denominator could not be worked out.
        logger.debug("could not plan tiles for a %s job", job_type, exc_info=True)
        return None
    if not tiles or tiles <= 0:
        return None
    return int(tiles), UNIT_TILE


def _estimate_training_steps(payload: dict) -> int | None:
    """``steps x rounds`` for a fine-tune, the one number its bar counts to.

    A *round* is one training pass plus the evaluation that follows it. There is
    one round for *use all*, one for a plain hold-out, and one per held-out unit
    when cross-validation is on. Multiplying gives a single monotone total that
    accounts for both halves of what the owner asked the bar to show -- the step
    count and the number of training-plus-inference rounds -- and, because it is
    monotone, gives the ETA for free.

    Delegated to ``quantem.finetune``, which owns the fold-planning rule that
    the run itself will follow. Importing it lazily keeps this module usable on
    an install where guided fine-tuning is not present.
    """
    from quantem.finetune.scope import planned_training_units  # noqa: PLC0415

    return planned_training_units(payload)


def _estimate_tiles(payload: dict) -> int | None:
    """The tiling plan for this payload, or None.

    Deliberately assembled the same way ``_run_segmentation`` assembles it:
    the segmentation row names the organelle, the payload names the family, and
    the asset's own pixel size decides the resample the plan is laid out on. A
    plan computed at a different scale from the one the run uses would be a
    denominator that changes the moment the run starts.

    A **multi-organelle payload** (``legs``) plans every leg and adds them up:
    one job, one denominator, and it is on the row from the moment the job is
    queued. If any single leg cannot be planned -- the commonest reason being a
    model that has not finished downloading yet -- the whole job reports *no*
    plan rather than a short one. A denominator missing one organelle's tiles is
    worse than no denominator: the bar reaches 100 % with an organelle still to
    run, which is precisely the defect the wave rollup was rewritten to remove.
    """
    legs = payload.get("legs")
    if isinstance(legs, list):
        if not legs:
            return None
        total = 0
        for leg in legs:
            if not isinstance(leg, dict):
                return None
            tiles = _estimate_tiles({**payload, **leg, "legs": None})
            if not tiles:
                return None
            total += int(tiles)
        return total or None
    from quantem.assets.asset_openable import get_asset_openable  # noqa: PLC0415
    from quantem.assets.models import ImageROI  # noqa: PLC0415
    from quantem.seg_core.db.inference import (  # noqa: PLC0415
        _estimate_model_tile_count,
    )
    from quantem.seg_core.registry import get_segmenter_or_none  # noqa: PLC0415
    from quantem.segmentation.models import ImageSegmentation  # noqa: PLC0415
    from quantem.segmentation.source_models import (  # noqa: PLC0415
        normalize_source_model,
        resolve_segmenter_internal_name,
    )

    segmentation_id = str(payload.get("segmentation_id") or "").strip()
    if not segmentation_id:
        return None
    segmentation = (
        ImageSegmentation.objects.select_related("asset", "segmentation_type")
        .filter(id=segmentation_id)
        .first()
    )
    if segmentation is None or segmentation.asset_id is None:
        return None

    shape = _region_shape(payload, segmentation, get_asset_openable, ImageROI)
    if shape is None:
        return None

    source_model = normalize_source_model(payload.get("source_model"))
    segmenter_internal_name = resolve_segmenter_internal_name(
        segmentation_type_internal_name=segmentation.segmentation_type.internal_name,
        source_model=source_model,
    )
    # Only the two kwargs that change the window layout. ``instance_params``,
    # the adapter and the threshold do not: an adapter swaps the head's weights,
    # not the grid it is walked over.
    segmenter = get_segmenter_or_none(
        segmenter_internal_name,
        source_model=source_model,
        pixel_size_nm=_asset_pixel_size_nm(segmentation),
    )
    if segmenter is None:
        return None
    return _estimate_model_tile_count(segmenter, shape)


def _region_shape(payload, segmentation, get_asset_openable, ImageROI):
    """``(height, width)`` the run will tile over, in native pixels.

    An ROI run tiles the crop, a full run tiles the image. The image's shape is
    read through the same openable the run reads it through, because a
    rendition's stored size is what inference sees and the asset's logical size
    is not always the same number.
    """
    roi_id = str(payload.get("roi_id") or "").strip()
    if roi_id:
        roi = ImageROI.objects.filter(id=roi_id).first()
        if roi is not None:
            return (int(roi.height), int(roi.width))
    openable = get_asset_openable(segmentation.asset, require=False)
    if openable is None:
        return None
    height, width = int(openable.height), int(openable.width)
    if height <= 0 or width <= 0:
        return None
    return (height, width)


def _asset_pixel_size_nm(segmentation) -> float | None:
    """The asset's nm/px, or None when the image is uncalibrated.

    Same rule as ``quantem.segmentation.organelle_tasks._asset_pixel_size_nm``:
    a missing or non-positive value means "not calibrated", and the model then
    runs at native scale -- which is a different tile count, so the plan has to
    agree with the run about it.
    """
    asset = getattr(segmentation, "asset", None)
    value = getattr(asset, "pixel_size_nm", None) if asset is not None else None
    try:
        size = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return size if size and size > 0 else None

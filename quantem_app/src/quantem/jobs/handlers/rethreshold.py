"""Re-derive the objects at a new include level, without running the model.

This is the worker behind the include-level dial. It runs
:func:`~quantem.seg_core.db.inference.replay_stored_probability_map`, which
thresholds probability bytes already on disk, and hands the result to the same
:func:`~quantem.seg_core.db.extraction.extract_and_save_segments` a real run
uses. No model is loaded, no forward pass happens, and nothing is downloaded:
on the images this application is for it is a few seconds of arithmetic, which
is why it sits on the interactive queue.

**The objects it produces are the objects a fresh run at that level would have
produced**, exactly and not approximately -- same polygons, same centroids,
same measured areas. That is a structural property of the ordering rather than
a hope, and ``segmentation/tests/test_threshold_replay.py`` is the acceptance
test for it. It is what makes the dial honest: a control that produced
*nearly* the same objects as a re-run would quietly change a scientist's
candidate set every time they touched it, and nothing on screen would say so.

Failure has two shapes and they must not be conflated
-----------------------------------------------------
Both end in "the model has to run again", and a user who is told only that
learns nothing about whether this will keep happening. The sentence differs:

* **Nothing was ever stored.** The run predates map storage, the image was over
  the size ceiling, or the file was reclaimed to save disk. Running once
  rebuilds it and the dial works from then on.
* **Bytes exist but are from an older build.** Their provenance markers are not
  today's, so re-thresholding them would give objects that do *not* match a run
  at the same level -- the one thing the dial promises. They are refused rather
  than reinterpreted, and the sentence says the stored result is an old one.

Both arrive as
:class:`~quantem.seg_core.db.inference.StoredMapUnavailable` carrying the
sentence for its own case; this module lets it through with the error code the
client renders a "Run inference again" control for, and never rewrites the
wording.

Registered eagerly from ``quantem.jobs.handlers.__init__``. A submodule left to
lazy import drops its job type out of the registry -- silently at boot, and
visibly only when a queued row of that type fails in a frozen build.
"""

from __future__ import annotations

import logging

from quantem.assets.models import ImageROI
from quantem.jobs.constants import JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL
from quantem.jobs.registry import job_handler
from quantem.jobs.reporter import CancelToken, JobReporter
from quantem.seg_core.db.extraction import extract_and_save_segments, resolve_min_area
from quantem.seg_core.db.inference import replay_stored_probability_map
from quantem.seg_core.registry import get_segmenter_or_none
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.organelle_tasks import zero_object_outcome
from quantem.segmentation.run_identity import (
    RUN_SCOPE_FULL,
    RUN_SCOPE_PATCH,
    run_identity_from_segmenter,
)
from quantem.segmentation.source_models import (
    normalize_source_model,
    resolve_segmenter_internal_name,
)

logger = logging.getLogger(__name__)

#: The include level is a probability, so it lives in the closed unit interval.
#: Validated in the serializer that accepts it from a client; re-checked here
#: because a job payload can also be written by an internal caller, and a
#: nonsense level would reach the segmenter's threshold setter unexamined.
INCLUDE_LEVEL_MIN = 0.0
INCLUDE_LEVEL_MAX = 1.0


def _asset_pixel_size_nm(segmentation: ImageSegmentation) -> float | None:
    asset = getattr(segmentation, "asset", None)
    return getattr(asset, "pixel_size_nm", None) if asset is not None else None


def _parse_include_level(raw: object) -> float:
    """The dial position this job was queued for, or a legible refusal.

    ``ValueError`` rather than a silent clamp: a level outside the range is a
    caller bug, and quietly running at 1.0 instead would write a candidate set
    the user did not ask for and could not distinguish from one they did.
    """
    if raw is None or raw == "":
        raise ValueError("This task was queued without an include level to use.")
    try:
        level = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("The include level has to be a number between 0 and 1.") from exc
    if level != level:  # NaN
        raise ValueError("The include level has to be a number between 0 and 1.")
    if not (INCLUDE_LEVEL_MIN <= level <= INCLUDE_LEVEL_MAX):
        raise ValueError("The include level has to be a number between 0 and 1.")
    return level


@job_handler(JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL)
def handle_reextract_at_include_level(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    """Threshold the stored probability map again and rewrite the objects.

    ``payload`` carries ``segmentation_id`` -- required, and not merely by
    convention: :data:`~quantem.jobs.constants.ACTIVE_SEGMENTATION_JOB_TYPES`
    includes this type, and the failure reconcilers read that exact key to
    release a segmentation whose worker died. A payload without it leaves the
    image showing as still running with nothing to clear it.

    Also accepts ``include_level`` (required), ``source_model`` and ``roi_id``.
    """
    cancel.check_cancelled()

    segmentation_id = str(payload.get("segmentation_id") or "").strip()
    if not segmentation_id:
        raise ValueError("This task was queued without an image to work on.")

    segmentation = (
        ImageSegmentation.objects.select_related("asset", "segmentation_type", "config")
        .filter(id=segmentation_id)
        .first()
    )
    if segmentation is None:
        raise ValueError("The image this task was queued for is no longer here.")
    if segmentation.asset_id is None:
        raise ValueError("The image this task was queued for has no picture to read.")

    include_level = _parse_include_level(payload.get("include_level"))

    roi = None
    roi_id = str(payload.get("roi_id") or "").strip()
    if roi_id:
        roi = ImageROI.objects.filter(id=roi_id).first()
        if roi is None:
            raise ValueError("The region this task was queued for is no longer here.")

    source_model = normalize_source_model(payload.get("source_model"))
    segmenter_internal_name = resolve_segmenter_internal_name(
        segmentation_type_internal_name=segmentation.segmentation_type.internal_name,
        source_model=source_model,
    )
    segmenter = get_segmenter_or_none(segmenter_internal_name)
    if segmenter is None:
        # The stored map belongs to a model, and re-thresholding it needs that
        # model's own extraction settings -- its area floor, its closing radius.
        # Without them this is not the same operation the run performed.
        raise ValueError(
            "The model that produced this result is not available on this "
            "computer, so its objects cannot be redone at a new include level."
        )

    cancel.check_cancelled()
    reporter.update(progress=5.0, message="reading the stored result")

    # ``StoredMapUnavailable`` is deliberately not caught. Its sentence is the
    # one its raiser chose for its own case -- the two cases say different
    # things, see the module docstring -- and it inherits ``ProbabilityMapMissing``
    # so ``classify_exception`` gives the failure the code the client renders a
    # "Run inference again" control for. Catching it here could only make the
    # message vaguer and drop the code.
    stored_metadata: dict[str, object] = {}
    result, image_array = replay_stored_probability_map(
        segmenter,
        segmentation,
        threshold=include_level,
        roi=roi,
        on_detail=lambda message: reporter.log("info", message),
        on_stored_metadata=stored_metadata.update,
    )

    cancel.check_cancelled()
    reporter.update(progress=40.0, message="finding the objects at the new level")

    area_floor = resolve_min_area(segmenter, None)
    stored_finished_at = stored_metadata.get("run_finished_at")
    run_identity = run_identity_from_segmenter(
        segmenter,
        run_id=str(
            stored_metadata.get("run_id")
            or getattr(reporter, "job_id", "")
            or "include-level"
        ),
        pack_id_fallback=source_model or segmenter.name,
        native_pixel_size_nm=_asset_pixel_size_nm(segmentation),
        min_area=area_floor,
        finished_at=str(stored_finished_at) if stored_finished_at else None,
        scope=RUN_SCOPE_PATCH if roi is not None else RUN_SCOPE_FULL,
        include_level=include_level,
    )
    if "adapter_id" in stored_metadata:
        adapter_id = stored_metadata.get("adapter_id")
        run_identity["adapter_id"] = str(adapter_id) if adapter_id else None
    if "device" in stored_metadata:
        device = stored_metadata.get("device")
        run_identity["device"] = str(device) if device else None

    segment_count = extract_and_save_segments(
        segmenter,
        segmentation,
        result,
        image_array,
        roi,
        min_area=area_floor,
        on_detail=lambda message: reporter.log("info", message),
        run_identity=run_identity,
        include_level=include_level,
    )

    # Written after the objects, never before: until the extraction succeeded
    # the objects on screen were still found at the old level, and a field
    # claiming otherwise would be read by every screen that shows the dial.
    segmentation.include_level = include_level
    segmentation.save(update_fields=["include_level", "updated_at"])

    if segment_count > 0:
        message = f"completed: {segment_count} objects at this include level"
        outcome: dict = {"segment_count": segment_count, "found_objects": True}
    else:
        zero_message, next_steps = zero_object_outcome(segmentation)
        message = zero_message
        outcome = {
            "segment_count": 0,
            "found_objects": False,
            "next_steps": next_steps,
        }

    reporter.update(progress=100.0, message=message)
    logger.info(
        "Re-extracted segmentation %s at include level %s: %d object(s)",
        segmentation.id,
        include_level,
        segment_count,
    )
    return {
        "segmentation_id": str(segmentation.id),
        "segmentation_type": segmentation.segmentation_type.internal_name,
        "include_level": include_level,
        "roi_id": str(roi.id) if roi is not None else None,
        "source_model": source_model or None,
        **outcome,
    }

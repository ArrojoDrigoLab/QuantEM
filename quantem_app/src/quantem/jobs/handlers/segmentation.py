"""Job handlers that run, re-measure or re-render a segmentation."""

from quantem.jobs.constants import (
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_REFRESH_SEGMENT_FEATURES,
    JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
)
from quantem.jobs.handlers.common import _as_bool
from quantem.jobs.registry import job_handler
from quantem.jobs.reporter import CancelToken, JobReporter
from quantem.seg_core.registry import get_segmenter_or_none
from quantem.segmentation.features.measure import MEASURED_MARKER_KEY
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import (
    NO_OBJECTS_MESSAGE,
    run_segmentation_for_image_task,
    run_segmentation_full_task,
    run_segmentation_roi_task,
    zero_object_outcome,
)
from quantem.segmentation.overlay_ngff import run_overlay_rebuild_job
from quantem.segmentation.source_models import (
    normalize_source_model,
    resolve_segmenter_internal_name,
)
from quantem.segmentation.tasks import compute_segment_features_task


def _segmentation_run_outcome(
    segment_count: int,
    *,
    segmentation: ImageSegmentation | None = None,
) -> tuple[str, dict]:
    """Terminal job message and result fields for a finished segmentation run.

    A run that found nothing used to end at "Candidates ready" with a plain
    success and no way to tell it apart from a worker that died quietly. The
    count and the actionable next step are reported here so the queue UI shows
    them without having to go and count objects itself.

    ``segmentation`` is what makes the zero case honest: over a proofread image
    a run is *expected* to create nothing, and "Lower the detection threshold
    and run again" is then advice to change a setting the user should not touch.
    See :func:`quantem.segmentation.organelle_tasks.zero_object_outcome`.
    Without it, the caller gets the nothing-is-labelled wording.
    """
    if segment_count > 0:
        return (
            f"completed: {segment_count} objects found",
            {"segment_count": segment_count, "found_objects": True},
        )
    if segmentation is None:
        message = NO_OBJECTS_MESSAGE
        next_steps = [
            "Lower the detection threshold and run again.",
            "Check that the selected model is trained for this organelle.",
        ]
    else:
        message, next_steps = zero_object_outcome(segmentation)
    return (
        message,
        {
            "segment_count": 0,
            "found_objects": False,
            "next_steps": next_steps,
        },
    )


def _validate_segmentation_payload(payload: dict) -> ImageSegmentation:
    segmentation_id = str(payload.get("segmentation_id") or "").strip()
    payload_segmentation_type = str(payload.get("segmentation_type") or "").strip()
    if not segmentation_id:
        raise ValueError("payload.segmentation_id is required")
    if not payload_segmentation_type:
        raise ValueError("payload.segmentation_type is required")

    segmentation = (
        ImageSegmentation.objects.select_related("asset", "segmentation_type", "config")
        .filter(id=segmentation_id)
        .first()
    )
    if segmentation is None:
        raise ValueError(f"Segmentation {segmentation_id} not found")
    if segmentation.asset_id is None:
        raise ValueError(f"Segmentation {segmentation_id} is missing asset_id")

    internal_name = segmentation.segmentation_type.internal_name
    if payload_segmentation_type != internal_name:
        raise ValueError(
            "payload.segmentation_type must match segmentation internal_name "
            f"(expected {internal_name}, got {payload_segmentation_type})"
        )

    source_model = normalize_source_model(payload.get("source_model"))
    segmenter_internal_name = resolve_segmenter_internal_name(
        segmentation_type_internal_name=internal_name,
        source_model=source_model,
    )
    if get_segmenter_or_none(segmenter_internal_name) is None:
        raise ValueError(f"No segmenter registered for type: {segmenter_internal_name}")

    return segmentation


@job_handler(JOB_TYPE_RUN_SEGMENTATION_ROI)
def handle_run_segmentation_roi_task(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    cancel.check_cancelled()
    segmentation = _validate_segmentation_payload(payload)
    roi_id = payload.get("roi_id")
    force_recompute_prob_maps = _as_bool(
        payload.get("force_recompute_prob_maps"),
        default=False,
    )
    reporter.update(progress=5.0, message="running ROI segmentation")
    segment_count = run_segmentation_roi_task(
        segmentation_id=str(segmentation.id),
        segmentation_type=segmentation.segmentation_type.internal_name,
        roi_id=str(roi_id) if roi_id else None,
        source_model=normalize_source_model(payload.get("source_model")) or None,
        force_recompute_prob_maps=force_recompute_prob_maps,
        reporter=reporter,
    )
    message, outcome = _segmentation_run_outcome(
        segment_count, segmentation=segmentation
    )
    reporter.update(progress=100.0, message=f"ROI segmentation {message}")
    return {
        "segmentation_id": str(segmentation.id),
        "segmentation_type": segmentation.segmentation_type.internal_name,
        "roi_id": str(roi_id) if roi_id else None,
        "source_model": normalize_source_model(payload.get("source_model")) or None,
        **outcome,
    }


@job_handler(JOB_TYPE_RUN_SEGMENTATION_FULL)
def handle_run_segmentation_full_task(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    cancel.check_cancelled()
    segmentation = _validate_segmentation_payload(payload)
    force_recompute_prob_maps = _as_bool(
        payload.get("force_recompute_prob_maps"),
        default=False,
    )
    reporter.update(progress=5.0, message="running full-image segmentation")
    segment_count = run_segmentation_full_task(
        segmentation_id=str(segmentation.id),
        segmentation_type=segmentation.segmentation_type.internal_name,
        source_model=normalize_source_model(payload.get("source_model")) or None,
        force_recompute_prob_maps=force_recompute_prob_maps,
        reporter=reporter,
    )
    message, outcome = _segmentation_run_outcome(
        segment_count, segmentation=segmentation
    )
    reporter.update(progress=100.0, message=f"full-image segmentation {message}")
    return {
        "segmentation_id": str(segmentation.id),
        "segmentation_type": segmentation.segmentation_type.internal_name,
        "source_model": normalize_source_model(payload.get("source_model")) or None,
        **outcome,
    }


@job_handler(JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE)
def handle_run_segmentation_for_image(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    """One run over one image, covering every organelle the user ticked.

    The payload carries the image and a list of ``legs``; each leg is one
    organelle's segmentation and the model family to run it with. Validation is
    the single-run validation applied leg by leg, so a bad request is refused
    here rather than half-way through a twenty-minute run.
    """
    cancel.check_cancelled()
    asset_id = str(payload.get("asset_id") or "").strip()
    if not asset_id:
        raise ValueError("payload.asset_id is required")
    raw_legs = payload.get("legs")
    if not isinstance(raw_legs, list) or not raw_legs:
        raise ValueError("payload.legs must list at least one organelle")

    legs = []
    for raw in raw_legs:
        if not isinstance(raw, dict):
            raise ValueError("payload.legs entries must be objects")
        segmentation = _validate_segmentation_payload(raw)
        if str(segmentation.asset_id) != asset_id:
            raise ValueError(
                "Every organelle in one run must belong to the same image."
            )
        legs.append(
            {
                "segmentation_id": str(segmentation.id),
                "source_model": normalize_source_model(raw.get("source_model")) or None,
            }
        )

    force_recompute_prob_maps = _as_bool(
        payload.get("force_recompute_prob_maps"),
        default=False,
    )
    reporter.update(progress=5.0, message="starting the run")
    outcome = run_segmentation_for_image_task(
        asset_id=asset_id,
        legs=legs,
        force_recompute_prob_maps=force_recompute_prob_maps,
        reporter=reporter,
        cancel=cancel,
    )
    found = [
        item for item in outcome["organelles"] if int(item.get("segment_count") or 0)
    ]
    if found:
        message = "completed: " + ", ".join(
            f"{item['name']} {item['segment_count']} objects" for item in found
        )
    else:
        message = "completed: no objects found"
    reporter.update(progress=100.0, message=message)
    return outcome


def _unmeasured_segment_ids(segmentation_id: str) -> list[str]:
    """Objects in this segmentation that carry no stored measurements.

    ``features["area"]`` is the marker: every successful measurement writes it,
    and ``regionprops`` cannot produce any of the other shape keys without it,
    so its absence is the one reliable "never measured".

    Read in Python rather than as a JSON query because ``features`` is a plain
    JSON column on SQLite and the key-path lookups differ by backend; this runs
    in a background job over one indexed filter, and the answer is normally an
    empty list.
    """
    rows = SegmentObject.objects.filter(segmentation_id=segmentation_id).values_list(
        "id", "features"
    )
    return [
        str(segment_id)
        for segment_id, features in rows.iterator(chunk_size=1000)
        if not isinstance(features, dict) or MEASURED_MARKER_KEY not in features
    ]


@job_handler(JOB_TYPE_REFRESH_SEGMENT_FEATURES)
def handle_refresh_segment_features(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    """Re-measure the objects an edit invalidated.

    ``segment_ids`` are the outlines that moved. ``recompute_features`` marks an
    edit that changed the confirmed set instead -- a label flip -- and asks for a
    sweep of this segmentation for objects that have never been measured at all,
    because those are the ones that reach ``objects.csv`` as blank columns once
    they join the analysed population.

    The flag used to be written into the payload by both label-change call sites
    and read by nothing here, so every flip queued a job that looped zero times
    and then reported *"segment feature refresh complete"* at 100%. The result
    now says which objects it looked at and how many needed work, so "complete"
    over nothing is distinguishable from "complete" over something.
    """
    cancel.check_cancelled()
    segmentation_id = str(payload.get("segmentation_id") or "").strip()
    raw_segment_ids = payload.get("segment_ids") or []
    if not isinstance(raw_segment_ids, list):
        raw_segment_ids = []
    segment_ids = [str(item).strip() for item in raw_segment_ids if str(item).strip()]
    recompute_features = _as_bool(payload.get("recompute_features"))

    if not segment_ids and not segmentation_id:
        # Named the task and its payload fields; this text becomes the failed
        # job's message and is read in Tasks & Queues (I-12, internal-name).
        raise ValueError(
            "This measurement task was queued without anything to measure."
        )

    swept = False
    if not segment_ids and recompute_features:
        if not segmentation_id:
            raise ValueError(
                "Re-measuring every object needs a segmentation to sweep, and "
                "this task was queued without one."
            )
        reporter.update(progress=5.0, message="checking for unmeasured objects")
        segment_ids = _unmeasured_segment_ids(segmentation_id)
        swept = True

    total = len(segment_ids)
    for index, segment_id in enumerate(segment_ids):
        cancel.check_cancelled()
        compute_segment_features_task(segment_id)
        progress = 10.0 + (80.0 * (index + 1) / total)
        reporter.update(
            progress=progress,
            message=f"refreshing segment features ({index + 1}/{total})",
        )

    if total:
        message = f"measured {total} object(s)"
    elif swept:
        message = "every object in this segmentation is already measured"
    else:
        message = "nothing to refresh"
    reporter.update(progress=100.0, message=message)
    return {
        "segmentation_id": segmentation_id,
        "segment_count": total,
        "swept_segmentation": swept,
        "refreshed_ids": segment_ids,
    }


@job_handler(JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY)
def handle_rebuild_segmentation_overlay(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    cancel.check_cancelled()
    segmentation_id = str(payload.get("segmentation_id") or "").strip()
    if not segmentation_id:
        raise ValueError("payload.segmentation_id is required")

    mode = str(payload.get("mode") or "full").strip().lower()
    if mode not in {"partial", "full"}:
        raise ValueError("payload.mode must be partial or full")

    segmentation = (
        ImageSegmentation.objects.select_related("asset", "segmentation_type")
        .filter(id=segmentation_id)
        .first()
    )
    if segmentation is None:
        raise ValueError(f"Segmentation {segmentation_id} not found")

    source_model = normalize_source_model(payload.get("source_model"))
    reporter.update(progress=5.0, message=f"rebuilding overlay ({mode})")
    state = run_overlay_rebuild_job(
        segmentation,
        mode=mode,
        source_model=source_model or None,
    )
    reporter.update(progress=100.0, message="overlay rebuild complete")
    return {
        "segmentation_id": str(segmentation.id),
        "mode": mode,
        "bundle_version": int(state.bundle_version),
        "applied_revision": int(state.applied_revision),
        "desired_revision": int(state.desired_revision),
        "status": state.status,
        "source_model": source_model or None,
    }

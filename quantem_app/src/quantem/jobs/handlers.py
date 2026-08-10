import logging

from quantem.assets.models import Asset
from quantem.assets.preprocess_status import set_stage
from quantem.assets.tasks import (
    encode_asset_full_to_png_task,
    ensure_ngff_for_asset_task,
    ensure_roi_for_asset_task,
)
from quantem.assets.volume_tasks import encode_asset_volume_to_ome_tiff_task
from quantem.jobs.constants import (
    JOB_TYPE_ENSURE_IMAGE_NGFF,
    JOB_TYPE_INSTALL_MODEL_PACK,
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_REFRESH_SEGMENT_FEATURES,
    JOB_TYPE_RUN_ANALYSIS,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    QUEUE_P3_ROI,
)
from quantem.jobs.models import Job
from quantem.jobs.registry import job_handler
from quantem.jobs.reporter import CancelToken, JobReporter
from quantem.seg_core.registry import get_segmenter_or_none
from quantem.segmentation.features.measure import MEASURED_MARKER_KEY
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationConfig,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import (
    NO_OBJECTS_MESSAGE,
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
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_lipid_droplet_type,
    get_or_create_mitochondria_type,
    get_or_create_nucleus_type,
)

logger = logging.getLogger(__name__)

# Organelles the upload wizard can request, in the order they are reported back.
# Each entry is (payload flag, reporting name, segmentation-type factory).
_UPLOAD_ORGANELLE_CHOICES = (
    ("segment_mito", "mitochondria", get_or_create_mitochondria_type),
    ("segment_er", "er", get_or_create_er_type),
    ("segment_nucleus", "nucleus", get_or_create_nucleus_type),
    ("segment_ld", "lipid_droplet", get_or_create_lipid_droplet_type),
)

VOLUME_SEGMENTATION_SKIP_REASON = "volume_segmentation_unsupported"
VOLUME_SEGMENTATION_SKIP_MESSAGE = (
    "This is a 3D volume. QuantEM imports and displays volumes, but its "
    "segmentation models are 2D and it will not fabricate a 3D result, so the "
    "organelles selected for this upload were not segmented."
)


def _asset_for_payload(payload: dict) -> Asset | None:
    asset_id = str(payload.get("asset_id") or "").strip()
    if asset_id:
        return Asset.objects.filter(id=asset_id).first()
    return None


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


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


def _requested_upload_organelles(payload: dict) -> list[tuple[str, object]]:
    """Organelles this upload asked for, as (reporting name, type factory).

    Reads the payload only — no segmentation types are created — so a caller can
    tell the user what it is refusing to do before touching the database.
    """
    return [
        (name, factory)
        for flag, name, factory in _UPLOAD_ORGANELLE_CHOICES
        if _as_bool(payload.get(flag))
    ]


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


@job_handler(JOB_TYPE_ENSURE_IMAGE_NGFF)
def handle_ensure_image_ngff(
    payload: dict,
    reporter: JobReporter,
    cancel: CancelToken,
) -> dict:
    cancel.check_cancelled()
    asset = _asset_for_payload(payload)
    if asset is None:
        raise ValueError("payload.asset_id is required")
    reporter.update(progress=5.0, message="building image NGFF")
    ensure_ngff_for_asset_task(str(asset.id))
    reporter.update(progress=100.0, message="image NGFF ready")
    return {"asset_id": str(asset.id)}


@job_handler(JOB_TYPE_UPLOAD_IMAGE_PIPELINE)
def handle_upload_image_pipeline(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    cancel.check_cancelled()
    asset = _asset_for_payload(payload)
    if asset is None:
        raise ValueError("payload.asset_id is required")

    requested_organelles = _requested_upload_organelles(payload)

    # 3D volumes follow a dedicated encode -> NGFF path and skip the 2D ROI and
    # feature stages. This is a designed refusal, not an oversight, and it is
    # reported back on the job rather than silently dropping the user's
    # organelle selection.
    if asset.logical_depth and int(asset.logical_depth) > 1:
        return _run_volume_upload_pipeline(
            asset,
            [name for name, _ in requested_organelles],
            reporter,
            cancel,
        )

    segmentations: list[tuple[ImageSegmentation, SegmentationConfig]] = []
    for _, type_factory in requested_organelles:
        seg_instance, _ = ImageSegmentation.objects.get_or_create(
            asset=asset,
            segmentation_type=type_factory(),
        )
        config, _ = SegmentationConfig.objects.get_or_create(segmentation=seg_instance)
        segmentations.append((seg_instance, config))

    reporter.update(progress=5.0, message="creating ROI")
    roi = ensure_roi_for_asset_task(str(asset.id))
    roi_id = str(roi.id) if roi else None
    if roi:
        roi_segmentations = [seg_instance for seg_instance, _ in segmentations]
        if roi_segmentations:
            roi.segmentations.add(*roi_segmentations)

    cancel.check_cancelled()

    for seg_instance, _ in segmentations:
        segmentation_type_internal_name = seg_instance.segmentation_type.internal_name
        Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            payload={
                "segmentation_id": str(seg_instance.id),
                "segmentation_type": segmentation_type_internal_name,
                "roi_id": roi_id,
                "asset_id": str(asset.id),
            },
            priority="high",
            resource_class="gpu",
            queue_name=QUEUE_P3_ROI,
            tags=[
                f"asset:{asset.id}",
                f"segmentation:{seg_instance.id}",
                f"segmentation_type:{segmentation_type_internal_name}",
            ],
        )

    reporter.update(progress=55.0, message="encoding full image")
    cancel.check_cancelled()
    encode_asset_full_to_png_task(str(asset.id))
    cancel.check_cancelled()
    reporter.update(progress=80.0, message="building NGFF")
    ensure_ngff_for_asset_task(str(asset.id))

    asset.refresh_from_db()
    if asset.preprocess_stage == "FAILED":
        reporter.update(progress=100.0, message="preprocessing failed")
        return {"asset_id": str(asset.id), "status": "failed"}
    if asset.preprocess_stage != "CANCELLED":
        set_stage(asset, "DONE", progress=100.0)

    reporter.update(progress=100.0, message="upload pipeline complete")
    return {"asset_id": str(asset.id)}


def _run_volume_upload_pipeline(
    asset: Asset,
    requested_organelles: list[str],
    reporter: JobReporter,
    cancel: CancelToken,
) -> dict:
    """Encode a 3D volume to its canonical OME-TIFF and build its NGFF.

    Segmentation is refused for volumes: every shipped model is 2D. When the
    upload asked for organelles anyway, the refusal and the organelles it
    applies to are written into the job result and the job log so the UI can
    tell the user instead of leaving them waiting for overlays that will never
    appear.
    """

    if requested_organelles:
        logger.info(
            "Skipping segmentation for volume asset %s (requested: %s).",
            asset.id,
            ", ".join(requested_organelles),
        )
        reporter.log("warning", VOLUME_SEGMENTATION_SKIP_MESSAGE)

    reporter.update(progress=5.0, message="encoding volume")
    cancel.check_cancelled()
    encode_asset_volume_to_ome_tiff_task(str(asset.id))
    cancel.check_cancelled()

    reporter.update(progress=70.0, message="building 3D NGFF")
    ensure_ngff_for_asset_task(str(asset.id))

    result: dict = {"asset_id": str(asset.id), "logical_depth": int(asset.logical_depth)}
    if requested_organelles:
        result["segmentation_skipped"] = True
        result["segmentation_skipped_reason"] = VOLUME_SEGMENTATION_SKIP_REASON
        result["segmentation_skipped_message"] = VOLUME_SEGMENTATION_SKIP_MESSAGE
        result["segmentation_skipped_types"] = list(requested_organelles)

    asset.refresh_from_db()
    if asset.preprocess_stage == "FAILED":
        reporter.update(progress=100.0, message="volume preprocessing failed")
        result["status"] = "failed"
        return result
    if asset.preprocess_stage != "CANCELLED":
        set_stage(asset, "DONE", progress=100.0)

    if requested_organelles:
        reporter.update(
            progress=100.0,
            message="volume imported; segmentation skipped (3D not supported)",
        )
    else:
        reporter.update(progress=100.0, message="volume upload pipeline complete")
    return result


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
        raise ValueError(
            "refresh_segment_features requires segmentation_id or segment_ids."
        )

    swept = False
    if not segment_ids and recompute_features:
        if not segmentation_id:
            raise ValueError(
                "refresh_segment_features with recompute_features requires "
                "segmentation_id."
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


@job_handler(JOB_TYPE_TRAIN_ORGANELLE_ADAPTER)
def handle_train_organelle_adapter(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    """Adapt a released model's head to the user's own annotated crops.

    The payload is passed through untouched; ``segmentation_id`` and
    ``base_model`` are validated here because a job that fails three minutes in
    on a missing key is worse than one that never starts.
    """
    cancel.check_cancelled()
    segmentation_id = str(payload.get("segmentation_id") or "").strip()
    if not segmentation_id:
        raise ValueError("payload.segmentation_id is required")
    base_model = str(payload.get("base_model") or "").strip()
    if not base_model:
        raise ValueError("payload.base_model is required")

    # Imported lazily: adapter training pulls in torch, and every other job type
    # would pay that import cost at module load.
    from quantem.finetune.adapter_job import train_organelle_adapter_job

    reporter.update(progress=1.0, message="adapting model to your data")
    return train_organelle_adapter_job(
        payload=payload,
        reporter=reporter,
        cancel=cancel,
    )


@job_handler(JOB_TYPE_INSTALL_MODEL_PACK)
def handle_install_model_pack(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    """Download a model pack from the QuantEM Hugging Face repository and install it.

    The whole pipeline -- fetch, digest verification, conversion to the pack
    format, TorchScript export, atomic promote -- lives in
    :mod:`quantem.registry.hf_install`; this handler is the progress and
    cancellation seam. Download bytes are known up front, so the bar reports a
    real fraction: 2-80% is the download, the rest is verify/convert/export.

    Cancellation is honoured between progress samples. The one thing that
    cannot be interrupted mid-flight is the byte transfer itself; an abandoned
    transfer finishes into huggingface_hub's content-addressed cache (where a
    retry reuses it) and never becomes an installed pack.
    """
    cancel.check_cancelled()
    pack_id = str(payload.get("pack_id") or "").strip()
    if not pack_id:
        raise ValueError("payload.pack_id is required")
    force = _as_bool(payload.get("force"))

    # Imported lazily: the registry's HF path pulls in huggingface_hub, and at
    # export time torch; every other job type must not pay those imports.
    from quantem.registry import catalogue
    from quantem.registry.hf_install import install_pack_from_hf

    reporter.update(progress=1.0, message=f"contacting the model repository for {pack_id}")

    def on_bytes(done: int, total: int) -> None:
        if total > 0:
            reporter.update(
                progress=2.0 + 78.0 * (done / total),
                message=f"downloading {pack_id}: {done / 1e6:.0f} of {total / 1e6:.0f} MB",
                # Raw counts too, so the Models screen's active_install block
                # can show real bytes without parsing the message back apart.
                current_bytes=done,
                total_bytes=total,
            )

    def on_status(message: str) -> None:
        reporter.update(message=message)

    installed = install_pack_from_hf(
        pack_id,
        force=force,
        on_status=on_status,
        on_bytes=on_bytes,
        cancel_check=cancel.check_cancelled,
    )

    entry = catalogue.pack_entry(pack_id)
    summary = f"installed {pack_id}" + ("" if entry["runnable"] else " (not runnable here)")
    reporter.update(progress=100.0, message=summary)
    return {
        "pack_id": pack_id,
        "source": "huggingface",
        "revision": installed.revision,
        "downloaded_bytes": installed.downloaded_bytes,
        "bytes_written": installed.bytes_written,
        "reused_blobs": installed.reused_blobs,
        "exported": installed.exported,
        **({"export_error": installed.export_error} if installed.export_error else {}),
        "runnable": entry["runnable"],
        "reason": entry["reason"],
        "encoder_tier": entry["encoder_tier"],
    }


@job_handler(JOB_TYPE_RUN_ANALYSIS)
def handle_run_analysis(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    """Run a quantitative analysis and write its export bundle.

    The payload is passed through untouched; ``analysis_run_id`` identifies both
    the run record and its export directory.
    """
    cancel.check_cancelled()
    analysis_run_id = str(payload.get("analysis_run_id") or "").strip()
    if not analysis_run_id:
        raise ValueError("payload.analysis_run_id is required")

    # Imported lazily: the analysis suite reaches back into segmentation and
    # assets, and importing it here at module load would make the handler
    # registry depend on the whole graph.
    from quantem.analysis.run_job import run_analysis_job

    reporter.update(progress=1.0, message="running analysis")
    return run_analysis_job(
        payload=payload,
        reporter=reporter,
        cancel=cancel,
    )

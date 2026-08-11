"""Job handlers for importing an image: NGFF, renditions, ROI, volumes."""

import logging

from quantem.assets.models import Asset
from quantem.assets.preprocess_status import set_stage
from quantem.assets.tasks import (
    ensure_ngff_for_asset_task,
    ensure_roi_for_asset_task,
    prepare_asset_renditions_task,
)
from quantem.assets.volume_tasks import encode_asset_volume_to_ome_tiff_task
from quantem.jobs.constants import (
    JOB_TYPE_ENSURE_IMAGE_NGFF,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    QUEUE_P3_ROI,
)
from quantem.jobs.handlers.common import _as_bool, _asset_for_payload
from quantem.jobs.models import Job
from quantem.jobs.registry import job_handler
from quantem.jobs.reporter import CancelToken, JobReporter
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationConfig,
)
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

    # Viewing first, then ROI. The ROI stage decodes the whole source image
    # twice (preview scoring, then the crop) and used to run ahead of the
    # pyramid, so a 475 MP import spent its first ~6 s -- and a 3 460 MP import
    # ~28 s -- on work the viewer does not need before it could show anything
    # at all. Nothing downstream is worse off: the ROI crop now comes out of
    # the finished pyramid instead of a full PNG decode, and the segmentation
    # jobs it feeds are queued for the GPU either way.
    reporter.update(progress=5.0, message="preparing image for viewing")
    cancel.check_cancelled()
    prepare_asset_renditions_task(str(asset.id))
    cancel.check_cancelled()

    asset.refresh_from_db()
    if asset.preprocess_stage == "FAILED":
        reporter.update(progress=100.0, message="preprocessing failed")
        return {"asset_id": str(asset.id), "status": "failed"}

    reporter.update(progress=70.0, message="creating ROI")
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

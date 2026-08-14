"""Action endpoints for ROI and full-image segmentation jobs."""

from __future__ import annotations

import logging

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.models import ImageROI
from quantem.assets.roi_state import get_active_roi_for_asset
from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    QUEUE_P3_ROI,
    QUEUE_P4_FULL,
)
from quantem.jobs.models import Job
from quantem.seg_core.registry import get_segmenter_or_none
from quantem.segmentation.instance_params import supports_instance_params
from quantem.segmentation.models import ImageSegmentation, SegmentationConfig
from quantem.segmentation.serializers import (
    SegmentationConfigResponseSerializer,
    SegmentationInstanceParamsPatchSerializer,
)
from quantem.segmentation.source_models import (
    get_source_model_definition,
    normalize_source_model,
    resolve_segmenter_internal_name,
    source_models_for_organelle,
)
from quantem.segmentation.status_reconcile import reconcile_segmentation_status

from .shared import (
    _ORGANELLE_ACTION_JOB_TYPES,
    active_segmentation_job,
    blocking_job_response_payload,
    completion_lock_response,
    get_segmentation_target_image,
)

logger = logging.getLogger(__name__)


def _resolve_resource_class(
    segmentation: ImageSegmentation,
    *,
    source_model: str | None = None,
) -> str:
    segmenter_internal_name = resolve_segmenter_internal_name(
        segmentation_type_internal_name=segmentation.segmentation_type.internal_name,
        source_model=source_model,
    )
    segmenter = get_segmenter_or_none(
        segmenter_internal_name,
        source_model=source_model,
    )
    if segmenter is None:
        return "gpu"
    return segmenter.job_resource_class


def _invalid_source_model_response(
    segmentation: ImageSegmentation, source_model: str
) -> Response | None:
    """A 400 naming the valid pack ids, or None when ``source_model`` is usable.

    ``source_model`` is already normalised; empty means "use the organelle's
    default" and is always fine. Anything else must be one of the released
    packs *for this segmentation's organelle*: accepting a free string here
    202'd the run and failed minutes later inside the worker with
    ``ValueError: No segmenter registered for type: ...`` -- an error about an
    internal name the user never typed, on a run they were told was queued.
    """
    if not source_model:
        return None
    definition = get_source_model_definition(source_model)
    organelle = segmentation.segmentation_type.internal_name
    if definition is not None and definition.organelle_internal_name == organelle:
        return None
    valid = [d.value for d in source_models_for_organelle(organelle)]
    return Response(
        {
            "detail": (
                f"{source_model!r} is not a model that can run this "
                "segmentation. Valid source_model values here: "
                f"{', '.join(valid) if valid else '(none)'}. Omit source_model "
                "to use the default."
            ),
            "valid_source_models": valid,
        },
        status=status.HTTP_400_BAD_REQUEST,
    )


def _invalidate_segment_tiles(segmentation_id: str) -> None:
    del segmentation_id
    return None


def _build_config_response(segmentation: ImageSegmentation, config: SegmentationConfig) -> dict:
    supports_params = supports_instance_params(segmentation.segmentation_type.internal_name)
    return {
        "supports_instance_params": supports_params,
        "instance_params": config.get_instance_params() if supports_params else None,
    }


class SegmentationConfigView(APIView):
    """Read/update segmentation-level configuration."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("segmentation_type"),
            id=seg_id,
        )
        config, _ = SegmentationConfig.objects.get_or_create(segmentation=segmentation)
        payload = _build_config_response(segmentation, config)
        serializer = SegmentationConfigResponseSerializer(payload)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("segmentation_type"),
            id=seg_id,
        )
        # These are the settings the next run uses -- the detection threshold
        # above all. ``LOCKED_DETAIL`` promises a locked segmentation's
        # measurements are final, and every endpoint that could start that run
        # already refuses; leaving this one open let a caller change the
        # threshold on a finished image and get a 200, so the recorded settings
        # no longer matched the objects they produced.
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        if not supports_instance_params(segmentation.segmentation_type.internal_name):
            return Response(
                {"detail": ("instance_params are only supported for organelle segmentations.")},
                status=status.HTTP_400_BAD_REQUEST,
            )

        config, _ = SegmentationConfig.objects.get_or_create(segmentation=segmentation)

        payload = request.data or {}
        if (
            isinstance(payload, dict)
            and "instance_params" in payload
            and isinstance(payload["instance_params"], dict)
        ):
            payload = payload["instance_params"]

        serializer = SegmentationInstanceParamsPatchSerializer(
            data=payload,
            partial=True,
        )
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        params = config.get_instance_params()
        params.update(serializer.to_instance_params_update())
        config.instance_params = params
        config.save(update_fields=["instance_params"])

        response_payload = _build_config_response(segmentation, config)
        response_serializer = SegmentationConfigResponseSerializer(response_payload)
        return Response(response_serializer.data, status=status.HTTP_200_OK)


class OrganelleRerunRoiView(APIView):
    """Queue a segmentation rerun over the active or explicit ROI."""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked

        # The user is telling us they believe nothing is running. If the stage
        # says otherwise because a worker died mid-run, correct it now rather
        # than leaving a phantom "Running..." over the run they are starting.
        reconcile_segmentation_status(segmentation)
        blocking_job = active_segmentation_job(
            segmentation,
            job_types=_ORGANELLE_ACTION_JOB_TYPES,
        )
        if blocking_job is not None:
            return Response(
                blocking_job_response_payload(blocking_job),
                status=status.HTTP_409_CONFLICT,
            )

        get_segmentation_target_image(segmentation)
        source_model = normalize_source_model(request.data.get("source_model"))
        invalid = _invalid_source_model_response(segmentation, source_model)
        if invalid is not None:
            return invalid
        explicit_roi_id = request.data.get("roi_id") if request.data else None
        if explicit_roi_id:
            roi = ImageROI.objects.filter(asset=segmentation.asset, id=explicit_roi_id).first()
            if roi is None:
                return Response(
                    {"detail": "ROI not found for the segmentation asset."},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            roi = get_active_roi_for_asset(segmentation.asset)
            if roi is None:
                return Response(
                    {"detail": "No active ROI found for this segmentation."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        payload = {
            "segmentation_id": str(segmentation.id),
            "segmentation_type": segmentation.segmentation_type.internal_name,
            "roi_id": str(roi.id),
            "asset_id": str(segmentation.asset_id),
        }
        if source_model:
            payload["source_model"] = source_model

        job = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            payload=payload,
            priority="high",
            resource_class=_resolve_resource_class(segmentation, source_model=source_model),
            queue_name=QUEUE_P3_ROI,
            tags=[f"segmentation:{seg_id}"],
        )
        _invalidate_segment_tiles(str(segmentation.id))
        return Response(
            {"job_id": str(job.id), "roi_id": str(roi.id)},
            status=status.HTTP_202_ACCEPTED,
        )


class OrganelleApplyFullImageView(APIView):
    """Queue a manual full-image segmentation run."""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        source_model = normalize_source_model(request.data.get("source_model"))
        # Refused here, with the valid ids, rather than 202'd and failed
        # minutes later inside the worker (adversarial round 13, finding 4).
        invalid = _invalid_source_model_response(segmentation, source_model)
        if invalid is not None:
            return invalid

        if not SegmentationConfig.objects.filter(segmentation=segmentation).exists():
            return Response(
                {"detail": "Not an organelle segmentation (no config)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # The user is telling us they believe nothing is running. If the stage
        # says otherwise because a worker died mid-run, correct it now rather
        # than leaving a phantom "Running..." over the run they are starting.
        reconcile_segmentation_status(segmentation)
        blocking_job = active_segmentation_job(
            segmentation,
            job_types=_ORGANELLE_ACTION_JOB_TYPES,
        )
        if blocking_job is not None:
            return Response(
                blocking_job_response_payload(blocking_job),
                status=status.HTTP_409_CONFLICT,
            )

        payload = {
            "segmentation_id": str(seg_id),
            "segmentation_type": segmentation.segmentation_type.internal_name,
            "asset_id": str(segmentation.asset_id),
        }
        if source_model:
            payload["source_model"] = source_model

        job = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload=payload,
            priority="default",
            resource_class=_resolve_resource_class(segmentation, source_model=source_model),
            queue_name=QUEUE_P4_FULL,
            max_attempts=1,
            tags=[f"segmentation:{seg_id}"],
        )
        _invalidate_segment_tiles(str(segmentation.id))
        return Response({"job_id": str(job.id)}, status=status.HTTP_202_ACCEPTED)

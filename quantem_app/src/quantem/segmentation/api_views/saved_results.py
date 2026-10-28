"""Compare retained predictions and choose the version used for measurement."""

from io import BytesIO
from uuid import UUID

import numpy as np
from django.db import transaction
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.urls import path
from PIL import Image
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.jobs.constants import (
    ACTIVE_SEGMENTATION_JOB_TYPES,
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
)
from quantem.jobs.models import Job
from quantem.segmentation.api_views.shared import active_segmentation_job
from quantem.segmentation.models import ImageSegmentation, SavedSegmentationResult
from quantem.segmentation.prob_maps.io import resolve_probability_map_path
from quantem.segmentation.prob_maps.preview import ensure_probability_preview


def serialize_result(result):
    return {
        "id": str(result.id),
        "name": result.name,
        "source_model": result.source_model,
        "adapter_id": result.adapter_id,
        "threshold": result.threshold,
        "is_canonical": result.is_canonical,
        "created_at": result.created_at,
        "object_count": result.object_count,
        "available": bool(result.probability_map_id),
        "preview_url": f"/api/segmentations/{result.segmentation_id}/results/{result.id}/preview/",
    }


class SavedResultsView(APIView):
    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, pk=seg_id)
        return Response(
            {
                "results": [
                    serialize_result(row)
                    for row in segmentation.saved_results.filter(deleted_at=None)
                ]
            }
        )

    @transaction.atomic
    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation.objects.select_for_update(), pk=seg_id)
        try:
            result_id = UUID(str(request.data.get("result_id", "")))
            raw_ids = request.data.get("delete_result_ids", [])
            if not isinstance(raw_ids, list):
                raise ValueError
            delete_ids = list({UUID(str(value)) for value in raw_ids})
        except (TypeError, ValueError):
            return Response(
                {"detail": "Choose a saved result and valid results to delete."}, status=400
            )
        result = get_object_or_404(segmentation.saved_results, pk=result_id, deleted_at=None)
        if not result.probability_map_id:
            return Response(
                {"detail": "This prediction is unavailable. Run the model again."}, status=409
            )
        if result_id in delete_ids or segmentation.saved_results.filter(
            id__in=delete_ids, deleted_at=None
        ).count() != len(delete_ids):
            return Response(
                {"detail": "Only other results on this image can be deleted."}, status=400
            )
        if active_segmentation_job(segmentation, job_types=ACTIVE_SEGMENTATION_JOB_TYPES):
            return Response({"detail": "Wait for the image's current task to finish."}, status=409)
        job = Job.enqueue(
            job_type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
            payload={
                "segmentation_id": str(segmentation.pk),
                "include_level": result.threshold,
                "saved_result_id": str(result.pk),
                "source_model": result.source_model,
                "delete_result_ids": [str(value) for value in delete_ids],
            },
            resource_class="cpu",
            priority="high",
            max_attempts=1,
            tags=[f"segmentation:{segmentation.pk}"],
        )
        return Response({"job_id": str(job.id)}, status=202)


class SavedResultPreviewView(APIView):
    def get(self, request, seg_id, result_id):
        result = get_object_or_404(
            SavedSegmentationResult.objects.select_related("probability_map"),
            id=result_id,
            segmentation_id=seg_id,
            deleted_at=None,
        )
        if result.probability_map is None:
            return Response({"detail": "This prediction is unavailable."}, status=404)
        try:
            preview = ensure_probability_preview(
                resolve_probability_map_path(result.probability_map)
            )
            with Image.open(preview) as image:
                values = np.asarray(image.convert("L"))
            rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
            from quantem.inference.resample import binarize_quantized

            # Each comparison pane controls opacity, so foreground pixels must
            # be opaque here to avoid multiplying two transparency settings.
            rgba[binarize_quantized(values, result.threshold)] = [6, 182, 212, 255]
            buffer = BytesIO()
            Image.fromarray(rgba).save(buffer, format="PNG")
            return HttpResponse(buffer.getvalue(), content_type="image/png")
        except (OSError, ValueError):
            return Response({"detail": "The stored prediction could not be read."}, status=404)


urlpatterns = [
    path("segmentations/<uuid:seg_id>/results/", SavedResultsView.as_view()),
    path(
        "segmentations/<uuid:seg_id>/results/<uuid:result_id>/preview/",
        SavedResultPreviewView.as_view(),
    ),
]

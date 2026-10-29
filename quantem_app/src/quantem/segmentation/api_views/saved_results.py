"""Compare retained predictions and choose the version used for measurement."""

import hashlib
import json
import os
from uuid import UUID, uuid4

import numpy as np
from django.db import transaction
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404
from django.urls import path
from PIL import Image
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.canonical_decode import decode_canonical_plane
from quantem.jobs.constants import (
    ACTIVE_SEGMENTATION_JOB_TYPES,
    JOB_DEFAULTS,
    JOB_TYPE_BUILD_RESULT_COMPARISON,
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
)
from quantem.jobs.models import Job
from quantem.segmentation.api_views.shared import active_segmentation_job
from quantem.segmentation.models import ImageSegmentation, SavedSegmentationResult
from quantem.segmentation.prob_maps.io import resolve_probability_map_path
from quantem.segmentation.prob_maps.preview import (
    ensure_probability_preview,
    result_preview_cache_path,
)


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
            source_path = resolve_probability_map_path(result.probability_map)
            preview = ensure_probability_preview(source_path)
            stat = preview.stat()
            key = hashlib.sha256(
                f"preview-v1:{preview}:{stat.st_mtime_ns}:{stat.st_size}:{result.threshold}".encode()
            ).hexdigest()[:24]
            etag = f'"{key}"'
            if request.headers.get("If-None-Match") == etag:
                response = HttpResponse(status=304)
            else:
                cache_dir = result_preview_cache_path(source_path)
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached = cache_dir / f"{result.id}.{key}.png"
                if not cached.exists():
                    values = decode_canonical_plane(preview).array
                    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
                    from quantem.inference.resample import binarize_quantized

                    rgba[binarize_quantized(values, result.threshold)] = [6, 182, 212, 255]
                    temporary = cached.with_suffix(f".{uuid4().hex}.tmp")
                    try:
                        Image.fromarray(rgba).save(temporary, format="PNG")
                        try:
                            os.replace(temporary, cached)
                        except OSError:
                            # Another request may have published and opened the
                            # same immutable PNG while this one was encoding.
                            if not cached.exists():
                                raise
                    finally:
                        temporary.unlink(missing_ok=True)
                response = FileResponse(cached.open("rb"), content_type="image/png")
            response["ETag"] = etag
            response["Cache-Control"] = "private, no-cache"
            response["X-Overlay-Preview-Kind"] = "thresholded-probability"
            return response
        except (OSError, ValueError):
            return Response({"detail": "The stored prediction could not be read."}, status=404)


class SavedResultOverlayView(APIView):
    """Request detailed postprocessing explicitly; polling only reads status."""

    @transaction.atomic
    def post(self, request, seg_id, result_id):
        return self._status(request, seg_id, result_id, enqueue=True)

    def get(self, request, seg_id, result_id):
        return self._status(request, seg_id, result_id, enqueue=False)

    def _status(self, request, seg_id, result_id, *, enqueue):
        from quantem.segmentation.result_overlays import (
            comparison_directory,
            comparison_identity,
            comparison_is_ready,
        )

        rows = SavedSegmentationResult.objects.select_related(
            "probability_map", "segmentation__asset", "segmentation__segmentation_type"
        )
        if enqueue:
            rows = rows.select_for_update(of=("self",))
        result = get_object_or_404(rows, id=result_id, segmentation_id=seg_id, deleted_at=None)
        if result.probability_map is None:
            return Response({"detail": "The saved prediction is unavailable."}, status=404)
        try:
            # The initial request computes the scientific identity once. Polls
            # pin that immutable key instead of repeating four aggregates over
            # a potentially very large segmentation every 1.5 seconds.
            key = request.query_params.get("revision") if not enqueue else None
            key = key or comparison_identity(result)
            directory = comparison_directory(result, key)
            ready = comparison_is_ready(directory, key)
        except (OSError, ValueError):
            return Response({"detail": "The stored prediction could not be read."}, status=404)
        base = f"/api/segmentations/{seg_id}/results/{result_id}/overlay/{key}"
        if ready:
            return Response({"status": "READY", "ngff_url": base, "revision": key})
        jobs = Job.objects.filter(
            type=JOB_TYPE_BUILD_RESULT_COMPARISON,
            payload_json__result_id=str(result_id),
            payload_json__comparison_key=key,
        )
        job = jobs.order_by("-created_at").first()
        active = job is not None and job.status in {"PENDING", "RETRY", "RUNNING"}
        if enqueue and not active:
            job = Job.enqueue(
                job_type=JOB_TYPE_BUILD_RESULT_COMPARISON,
                payload={
                    "result_id": str(result_id),
                    "comparison_key": key,
                    "segmentation_id": str(seg_id),
                },
                max_attempts=1,
                tags=[f"segmentation:{seg_id}"],
                **JOB_DEFAULTS[JOB_TYPE_BUILD_RESULT_COMPARISON],
            )
            active = True
        return Response(
            {
                "status": "BUILDING" if active else "FAILED" if job is not None else "MISSING",
                "job_id": str(job.id) if job else None,
                "message": job.message if job else "",
                "revision": key,
            }
        )


def saved_result_overlay_file(request, seg_id, result_id, key, ngff_path):
    from quantem.segmentation.api_views.overlay import _sparse_chunk_bytes
    from quantem.segmentation.overlay_ngff.store import (
        parse_overlay_chunk_path,
        stored_chunk_shape,
    )
    from quantem.segmentation.result_overlays import comparison_directory

    result = get_object_or_404(
        SavedSegmentationResult.objects.select_related("probability_map", "segmentation__asset"),
        pk=result_id,
        segmentation_id=seg_id,
        deleted_at=None,
    )
    try:
        if result.probability_map is None:
            raise ValueError
        directory = comparison_directory(result, key)
        root = (directory / "labels.zarr").resolve()
        target = (root / ngff_path).resolve()
        target.relative_to(root)
        snapshot = json.loads((directory / "ready.json").read_text(encoding="utf-8"))
        if snapshot["key"] != key or not (root / ".zgroup").is_file():
            raise ValueError
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise Http404("This comparison generation is unavailable.") from exc
    if target.is_file():
        content_type = (
            "application/json" if target.name.startswith(".z") else "application/octet-stream"
        )
        response = FileResponse(target.open("rb"), content_type=content_type)
    elif target.name == ".zattrs" and target.parent.is_dir():
        response = HttpResponse("{}", content_type="application/json")
    else:
        chunk = parse_overlay_chunk_path(ngff_path)
        shape = stored_chunk_shape(root, *chunk) if chunk else None
        if shape is None:
            raise Http404("Comparison tile not found")
        response = HttpResponse(
            _sparse_chunk_bytes(chunk[0], shape), content_type="application/octet-stream"
        )
    response["Cache-Control"] = "private, max-age=31536000, immutable"
    return response


urlpatterns = [
    path(
        "segmentations/<uuid:seg_id>/results/<uuid:result_id>/overlay/",
        SavedResultOverlayView.as_view(),
    ),
    path(
        "segmentations/<uuid:seg_id>/results/<uuid:result_id>/overlay/<str:key>/<path:ngff_path>",
        saved_result_overlay_file,
    ),
    path("segmentations/<uuid:seg_id>/results/", SavedResultsView.as_view()),
    path(
        "segmentations/<uuid:seg_id>/results/<uuid:result_id>/preview/",
        SavedResultPreviewView.as_view(),
    ),
]

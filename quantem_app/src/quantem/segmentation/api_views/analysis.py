"""Quantitative-analysis endpoints.

The views live in the segmentation package because that is where every other
``/api/segmentations/<id>/...`` endpoint lives; the model, serializers, loaders
and routes are in :mod:`quantem.analysis`.

Everything long-running goes through the job queue: an analysis over a
full-resolution image rasterises several masks and runs a Monte-Carlo null, and
a desktop UI that blocks on that looks broken.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.analysis.models import AnalysisRun
from quantem.analysis.serializers import (
    AnalysisRunCreateSerializer,
    AnalysisRunSerializer,
    AnalysisRunSummarySerializer,
)
from quantem.jobs.constants import JOB_DEFAULTS, JOB_TYPE_RUN_ANALYSIS
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation

logger = logging.getLogger(__name__)

#: Content types for the files :func:`quantem.analysis.service.write_bundle`
#: writes. ``mimetypes`` alone answers ".csv" differently on different Windows
#: installs (the registry decides), and the frontend keys off this header.
EXPORT_CONTENT_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
}


def _segmentation_or_404(seg_id) -> ImageSegmentation:
    return get_object_or_404(
        ImageSegmentation.objects.select_related("asset", "segmentation_type"),
        id=seg_id,
    )


class SegmentationAnalysisView(APIView):
    """Start an analysis of a segmentation, or list the ones already run."""

    def get(self, request, seg_id):
        segmentation = _segmentation_or_404(seg_id)
        runs = AnalysisRun.objects.filter(segmentation=segmentation)
        return Response(
            AnalysisRunSummarySerializer(runs, many=True).data,
            status=status.HTTP_200_OK,
        )

    def post(self, request, seg_id):
        segmentation = _segmentation_or_404(seg_id)
        serializer = AnalysisRunCreateSerializer(
            data=request.data, context={"segmentation": segmentation}
        )
        if not serializer.is_valid():
            return Response(
                {"error": _first_error(serializer.errors)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        params = serializer.validated_data
        run = AnalysisRun.objects.create(
            segmentation=segmentation,
            params=params,
            group=params.get("group", ""),
        )

        defaults = JOB_DEFAULTS[JOB_TYPE_RUN_ANALYSIS]
        job = Job.enqueue(
            job_type=JOB_TYPE_RUN_ANALYSIS,
            payload={
                "analysis_run_id": str(run.id),
                # Carried so the queue screen can name the image and organelle
                # this job belongs to (quantem.jobs.views._collect_context_maps).
                "segmentation_id": str(segmentation.id),
                "asset_id": str(segmentation.asset_id) if segmentation.asset_id else None,
            },
            priority=defaults["priority"],
            resource_class=defaults["resource_class"],
            queue_name=defaults["queue_name"],
            tags=[
                f"segmentation:{segmentation.id}",
                f"analysis_run:{run.id}",
            ],
        )
        return Response(
            {"job_id": str(job.id), "analysis_run_id": str(run.id)},
            status=status.HTTP_202_ACCEPTED,
        )


class AnalysisRunDetailView(APIView):
    """One analysis run and everything it measured."""

    def get(self, request, run_id):
        run = get_object_or_404(AnalysisRun.objects.select_related("segmentation"), id=run_id)
        return Response(AnalysisRunSerializer(run).data, status=status.HTTP_200_OK)


class AnalysisRunExportView(APIView):
    """Download one file from a run's export bundle.

    The bundle is the reportable artefact -- ``objects.csv``,
    ``image_summary.csv`` and the ``manifest.json`` that records what produced
    them -- so this is a download, not a preview: ``Content-Disposition:
    attachment`` always.

    ``name`` is resolved *inside* the run's own export directory and rejected if
    it escapes, which is what keeps a crafted name from turning a loopback
    endpoint into a whole-filesystem read.
    """

    def get(self, request, run_id, name):
        run = get_object_or_404(AnalysisRun, id=run_id)
        base = run.export_path
        if base is None:
            return Response(
                {"error": "This analysis has not written an export bundle yet."},
                status=status.HTTP_404_NOT_FOUND,
            )

        resolved_base = Path(base).resolve()
        try:
            candidate = (resolved_base / name).resolve()
        except (OSError, ValueError):
            candidate = None
        if candidate is None or not _is_within(candidate, resolved_base):
            logger.warning("Refused export path %r outside run %s's directory.", name, run.id)
            return Response(
                {"error": "That file is not part of this analysis's export bundle."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not candidate.is_file():
            return Response(
                {"error": f"The export bundle has no file called {name!r}."},
                status=status.HTTP_404_NOT_FOUND,
            )

        content_type = (
            EXPORT_CONTENT_TYPES.get(candidate.suffix.lower())
            or mimetypes.guess_type(candidate.name)[0]
            or "application/octet-stream"
        )
        return FileResponse(
            candidate.open("rb"),
            as_attachment=True,
            filename=candidate.name,
            content_type=content_type,
        )


def _is_within(candidate: Path, base: Path) -> bool:
    """True when ``candidate`` is ``base`` itself or something under it."""
    try:
        return candidate == base or candidate.is_relative_to(base)
    except (OSError, ValueError):  # pragma: no cover - unresolvable path
        return False


def _first_error(errors) -> str:
    """One human-readable sentence out of a DRF error structure.

    The contract says errors are ``{"error": "<sentence>"}``; DRF's nested
    dict-of-lists is not that.
    """
    if isinstance(errors, dict):
        for key, value in errors.items():
            message = _first_error(value)
            if key in {"non_field_errors", "detail"}:
                return message
            return f"{key}: {message}"
    if isinstance(errors, list) and errors:
        return _first_error(errors[0])
    return str(errors)

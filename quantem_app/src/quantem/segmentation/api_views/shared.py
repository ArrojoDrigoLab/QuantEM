"""Shared helpers for segmentation API views."""

from __future__ import annotations

import logging
import os

from django.http import Http404
from rest_framework import status
from rest_framework.response import Response

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.models import ImageROI
from quantem.assets.roi_state import get_active_roi_for_asset
from quantem.assets.utils import create_roi_image_from_image
from quantem.jobs.constants import ACTIVE_SEGMENTATION_JOB_TYPES
from quantem.jobs.models import Job
from quantem.segmentation.completion import is_locked, locked_payload
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.roi_selection import select_roi_for_image

logger = logging.getLogger(__name__)

_ACTIVE_JOB_STATUSES = frozenset({"PENDING", "RUNNING", "RETRY"})
_ORGANELLE_ACTION_JOB_TYPES = ACTIVE_SEGMENTATION_JOB_TYPES


def get_or_create_roi_image(image) -> ImageROI:
    asset = getattr(image, "asset", None)
    existing = get_active_roi_for_asset(asset)
    if existing:
        return existing

    roi_min_size = int(os.environ.get("ROI_MIN_IMAGE_SIZE", "6000"))
    roi_size = int(os.environ.get("ROI_SIZE", "3000"))
    if image.width >= roi_min_size and image.height >= roi_min_size:
        roi_result = select_roi_for_image(image, roi_size=roi_size)
        return create_roi_image_from_image(
            image,
            x=roi_result.x,
            y=roi_result.y,
            width=roi_result.width,
            height=roi_result.height,
            source="AUTO",
            is_active=True,
        )

    return create_roi_image_from_image(
        image,
        x=0,
        y=0,
        width=image.width,
        height=image.height,
        source="AUTO",
        is_active=True,
    )


def get_segmentation_target_image(segmentation: ImageSegmentation):
    if segmentation.asset_id:
        try:
            return get_asset_openable(segmentation.asset)
        except Http404 as exc:
            raise ValueError("Segmentation asset has no local full/subset rendition") from exc
    raise ValueError("Segmentation has no target asset")

def active_segmentation_job(
    segmentation: ImageSegmentation,
    *,
    job_types: frozenset[str] | None = None,
) -> Job | None:
    """The job currently holding this segmentation, or None.

    Returns the job rather than a bare bool so the 409 that refuses a new run
    can name it. "A task is already queued or running" with no job id and no way
    out is the message a user gets when a worker has died, and it is a dead end:
    they cannot cancel what they cannot identify.
    """
    jobs = Job.objects.filter(
        status__in=_ACTIVE_JOB_STATUSES,
        payload_json__segmentation_id=str(segmentation.id),
    )
    if job_types is not None:
        jobs = jobs.filter(type__in=job_types)
    # A RUNNING job is the one actually holding the segmentation and the one the
    # user has to clear, so name it ahead of anything merely queued behind it.
    return (
        jobs.filter(status="RUNNING").order_by("created_at").first()
        or jobs.order_by("created_at").first()
    )


def has_active_segmentation_jobs(
    segmentation: ImageSegmentation,
    *,
    job_types: frozenset[str] | None = None,
) -> bool:
    return active_segmentation_job(segmentation, job_types=job_types) is not None


def completion_lock_response(*segmentations: ImageSegmentation) -> Response | None:
    """A 409 for the first locked segmentation, or ``None`` if none is locked.

    Every endpoint that changes a segmentation's objects, its labels, or starts
    a run over it calls this first. Refused with 409 rather than 403 because the
    request is not forbidden -- it conflicts with a state the user chose and can
    undo, and the body says how.
    """
    for segmentation in segmentations:
        if segmentation is not None and is_locked(segmentation):
            return Response(
                locked_payload(segmentation),
                status=status.HTTP_409_CONFLICT,
            )
    return None


def blocking_job_response_payload(job: Job) -> dict:
    """Body for the 409 that refuses a run because ``job`` holds the segmentation."""
    if job.status == "RUNNING":
        how_to_clear = (
            f"Cancel it (POST /api/jobs/{job.id}/cancel/) and run again. If its "
            "worker is already gone the cancel takes effect within a few seconds."
        )
    else:
        how_to_clear = (
            f"Wait for it, or remove it from the queue "
            f"(DELETE /api/jobs/{job.id}/)."
        )
    return {
        "detail": (
            f"A ROI/full segmentation task is already {job.status.lower()} for "
            f"this segmentation (job {job.id}). {how_to_clear}"
        ),
        "job_id": str(job.id),
        "job_type": job.type,
        "job_status": job.status,
    }

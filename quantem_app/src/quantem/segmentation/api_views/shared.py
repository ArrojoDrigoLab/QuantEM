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
from quantem.jobs.constants import ACTIVE_SEGMENTATION_JOB_TYPES, JOB_TYPE_LABELS
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

    roi_min_size = int(os.environ.get("ROI_MIN_IMAGE_SIZE", "512"))
    roi_size = int(os.environ.get("ROI_SIZE", "512"))
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
    held = (
        jobs.filter(status="RUNNING").order_by("created_at").first()
        or jobs.order_by("created_at").first()
    )
    return held or _multi_organelle_job_holding(segmentation, job_types=job_types)


def _multi_organelle_job_holding(
    segmentation: ImageSegmentation,
    *,
    job_types: frozenset[str] | None = None,
) -> Job | None:
    """A one-run-per-image job that has this segmentation among its organelles.

    The query above cannot see one: that job's payload names an *image* and
    lists its organelles under ``legs``, so there is no ``segmentation_id`` key
    to match on. Without this, pressing Run on the labeling screen while the
    image-wide run was mid-way through the same organelle queued a second pass
    over it -- two runs writing candidates into one segmentation.

    Matched in Python rather than with a JSON containment lookup because SQLite,
    which is what a desktop install runs on, does not support one. The set it
    scans is the open jobs for this one image: on a single-user desktop with a
    one-slot pool that is a handful of rows, not a scan.
    """
    asset_id = getattr(segmentation, "asset_id", None)
    if not asset_id:
        return None
    from quantem.jobs.constants import (  # noqa: PLC0415 -- keeps the import list flat
        JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
    )

    if job_types is not None and JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE not in job_types:
        return None
    candidates = Job.objects.filter(
        status__in=_ACTIVE_JOB_STATUSES,
        type=JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
        payload_json__asset_id=str(asset_id),
    ).order_by("created_at")
    wanted = str(segmentation.id)
    holding = [
        job
        for job in candidates
        if any(
            isinstance(leg, dict) and str(leg.get("segmentation_id") or "") == wanted
            for leg in (job.payload_json or {}).get("legs") or []
        )
    ]
    if not holding:
        return None
    running = [job for job in holding if job.status == "RUNNING"]
    return (running or holding)[0]


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
    """Body for the 409 that refuses a run because ``job`` holds the segmentation.

    ``detail`` is read by a biologist, so it names the task the way the Tasks &
    Queues panel names it and points at the control in that panel. It used to
    say *"Cancel it (POST /api/jobs/<id>/cancel/)"*, which is invariant I-12's
    exact failure: an HTTP verb and an API route handed to someone who has no
    way to issue either, in place of the button that does the job. The job id
    stays in the payload as its own field for clients; it is not a sentence.
    """
    label = JOB_TYPE_LABELS.get(job.type)
    # Quoted only when it is the panel's own wording, so the fallback reads as a
    # sentence rather than as a name nothing on screen uses.
    task = f'"{label}"' if label else "A segmentation run"
    if job.status == "RUNNING":
        detail = (
            f"{task} is already running on this segmentation. Cancel it in "
            "Tasks & Queues, then start the new run. If its worker has already "
            "stopped, the cancellation takes effect within a few seconds."
        )
    else:
        detail = (
            f"{task} is already waiting in the queue for this segmentation. "
            "Wait for it to finish, or remove it in Tasks & Queues, then start "
            "the new run."
        )
    return {
        "detail": detail,
        "job_id": str(job.id),
        "job_type": job.type,
        "job_status": job.status,
    }


def delete_blocked_response_payload(job: Job) -> dict:
    """Body for the 409 that refuses a *delete* because ``job`` holds it.

    The same shape and the same vocabulary as
    :func:`blocking_job_response_payload`, because it is the same situation
    seen from the delete dialog rather than from the run button. It is written
    out separately only because the way out is different: there is nothing to
    start afterwards, the user came here to remove something.

    This is the one that shipped broken. Until 2026-08-10 the delete refusal
    composed its own sentence and it read, verbatim, in the confirm dialog::

        This segmentation cannot be deleted while a run_segmentation_full_task
        job is running on it (job 04a18666-11de-4c39-8fd2-c25a67b7d6c9). Cancel
        it (POST /api/jobs/04a18666-.../cancel/) and delete again once it has
        stopped.

    Four of invariant I-12's classes in one sentence -- an internal task name, a
    raw job id, an HTTP verb and an API route -- two clicks from the viewer, in
    front of someone who has no way to issue a request and no screen on which
    that id appears. The machine-readable half of all four is still in the
    payload, in fields, which is where a client reads them from anyway.
    """
    label = JOB_TYPE_LABELS.get(job.type)
    task = f'"{label}"' if label else "A run"
    if job.status == "RUNNING":
        detail = (
            f"{task} is running on this segmentation right now. Stop it in "
            "Tasks & Queues, then delete this segmentation. If its worker has "
            "already stopped, the cancellation takes effect within a few seconds."
        )
    else:
        detail = (
            f"{task} is waiting in the queue for this segmentation. Wait for it "
            "to finish, or remove it in Tasks & Queues, then delete this "
            "segmentation."
        )
    return {
        "detail": detail,
        "job_id": str(job.id),
        "job_type": job.type,
        "job_status": job.status,
    }

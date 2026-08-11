"""Queueing the per-object morphometrics refresh."""

from __future__ import annotations

import os

from quantem.jobs.constants import JOB_TYPE_REFRESH_SEGMENT_FEATURES, QUEUE_P1_INTERACTIVE
from quantem.jobs.models import Job


def _segment_feature_refresh_triggers_enabled() -> bool:
    raw = str(os.environ.get("QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS", "0")).strip()
    return raw.lower() not in {"", "0", "false", "no", "off"}


def _pending_sweep_job(segmentation_id: str) -> Job | None:
    """A queued sweep for this segmentation that has not started yet, if any.

    A proofreading session is hundreds of label flips, and each one asks for the
    same whole-segmentation sweep. Queueing one job per click would have the
    interactive queue working through a few hundred identical scans; one pending
    job covers every flip made before it runs.
    """
    return (
        Job.objects.filter(
            type=JOB_TYPE_REFRESH_SEGMENT_FEATURES,
            status="PENDING",
            payload_json__segmentation_id=str(segmentation_id),
            payload_json__segment_ids=[],
        )
        .order_by("-created_at")
        .first()
    )


def _enqueue_segment_feature_refresh(
    *,
    segmentation_id: str,
    segment_ids: list[str],
    recompute_features: bool,
) -> Job | None:
    """Queue a per-object morphometrics refresh.

    ``segment_ids`` names the objects an edit changed the outline of; each is
    re-measured.

    ``recompute_features`` marks edits that changed the *confirmed set* rather
    than any one outline -- a label flip, a confirm, an exclude. No individual
    object needs re-measuring for those: the geometry did not move. What changes
    is which objects the analysis aggregates over, and an object that has never
    been measured (drawn before measure-on-create existed, or one whose
    measurement failed) contributes blank columns to ``objects.csv`` the moment
    it joins that population. So the flag asks the handler to sweep this
    segmentation for unmeasured objects and measure them; normally it finds
    none, and says so.

    It used to be written into the payload and read by nothing:
    ``jobs/handlers.py`` looped over ``segment_ids`` alone, so every label flip
    queued a job that looped zero times and reported *"segment feature refresh
    complete"* at 100%.
    """
    if not _segment_feature_refresh_triggers_enabled():
        return None

    deduped_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in segment_ids:
        normalized = str(raw_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_ids.append(normalized)

    if not deduped_ids and not recompute_features:
        return None

    if not deduped_ids:
        existing = _pending_sweep_job(segmentation_id)
        if existing is not None:
            return existing

    return Job.enqueue(
        job_type=JOB_TYPE_REFRESH_SEGMENT_FEATURES,
        payload={
            "segmentation_id": str(segmentation_id),
            "segment_ids": deduped_ids,
            "recompute_features": bool(recompute_features),
        },
        priority="high",
        resource_class="cpu",
        queue_name=QUEUE_P1_INTERACTIVE,
        tags=[
            f"segmentation:{segmentation_id}",
            "interactive:segment-feature-refresh",
        ],
    )

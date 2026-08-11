"""Repairing a segmentation status that no run is behind any more.

``ImageSegmentation.status_stage`` is written by the run that is in progress. If
that run's worker disappears -- the app is killed, the machine sleeps, the
process tree is torn down mid-inference -- the last thing it wrote stays there.
The labeling screen then shows *"Run full-image segmentation, 40%"* with a
disabled ``Running…`` pill for a run with no process behind it, and every new
run on that segmentation is refused with a 409 naming a job nobody is executing.

A user hit exactly this: they killed the server tree mid-run, and the ghost
persisted for 4m46s -- the job heartbeat's staleness window -- before the
reaper failed the job.

This module is the segmentation-side half of that. When something asks about a
segmentation whose stage says "running" and **no job of any status holds it**,
there is no future in which that stage becomes anything else, so it is corrected
to what the data actually supports: candidates if the segmentation holds
objects, unstarted if it does not.

Two things it deliberately does not do:

* It does not touch a segmentation with a live job. A PENDING, RUNNING or RETRY
  job means the queue still intends to run this; the stage is that job's to
  write.
* It does not fire immediately. A run driven outside the queue -- the CLI, a
  test -- legitimately writes a running stage with no ``Job`` row, so a stage
  is only considered abandoned once it has gone :func:`stale_after_seconds`
  without an update. The status callback in
  :mod:`quantem.segmentation.organelle_tasks` saves at least every half second
  while a run is alive, so a live run never goes quiet for that long.

The other half is not here: the queue should reap orphaned jobs **at startup**
rather than waiting out a heartbeat interval sized for a multi-process world,
because a desktop app that has just started knows for certain that no worker it
owns is alive. That belongs to :mod:`quantem.jobs.scheduler`.
"""

from __future__ import annotations

import logging
import os

from django.utils import timezone

from .models import ImageSegmentation, SegmentObject

logger = logging.getLogger(__name__)

#: Stages that mean "a run is working on this right now". Anything else is a
#: resting state that survives a dead worker perfectly well.
RUNNING_STAGES: frozenset[str] = frozenset(
    {
        "RUNNING_INFERENCE",
        "EXTRACTING_CANDIDATES",
        "UPDATING",
        "COMPUTING_FEATURES",
    }
)

DEFAULT_STALE_STATUS_SECONDS = 60

#: What a user is told when their run's worker vanished. It says the run did not
#: finish rather than that anything is wrong with the objects, because objects
#: from an earlier successful run are still there and are still correct.
ABANDONED_RUN_MESSAGE = (
    "The last run stopped before it finished (its worker is no longer running, "
    "usually because the application was closed or restarted mid-run). Nothing "
    "already saved was lost. Run it again when you are ready."
)


def stale_after_seconds() -> int:
    try:
        value = int(
            str(
                os.environ.get(
                    "QUANTEM_SEGMENTATION_STALE_STATUS_SECONDS",
                    DEFAULT_STALE_STATUS_SECONDS,
                )
            ).strip()
        )
    except (TypeError, ValueError):
        return DEFAULT_STALE_STATUS_SECONDS
    return max(value, 0)


def _has_active_job(segmentation: ImageSegmentation) -> bool:
    # Imported here rather than at module scope: this module is reached from the
    # models/serializer side, and a top-level import of the API view helpers
    # would make that a cycle.
    from .api_views.shared import active_segmentation_job  # noqa: PLC0415

    return active_segmentation_job(segmentation) is not None


def reconcile_segmentation_status(segmentation: ImageSegmentation) -> bool:
    """Correct a running stage that no job is behind. True if anything changed.

    Safe to call on every read: it is one indexed job query, and it returns
    immediately for the overwhelmingly common case of a segmentation that is not
    claiming to be mid-run.
    """
    if segmentation.status_stage not in RUNNING_STAGES:
        return False

    grace = stale_after_seconds()
    updated_at = segmentation.updated_at
    if updated_at is not None:
        age = (timezone.now() - updated_at).total_seconds()
        if age < grace:
            return False

    if _has_active_job(segmentation):
        return False

    has_objects = SegmentObject.objects.filter(segmentation=segmentation).exists()
    previous_stage = segmentation.status_stage
    segmentation.status_stage = "CANDIDATES_READY" if has_objects else "UNSTARTED"
    segmentation.status_progress = 100.0 if has_objects else 0.0
    segmentation.status_error = ABANDONED_RUN_MESSAGE
    segmentation.save(update_fields=["status_stage", "status_progress", "status_error"])
    logger.info(
        "Segmentation %s was left at %s by a run with no job behind it; corrected to %s.",
        segmentation.id,
        previous_stage,
        segmentation.status_stage,
    )
    return True


def reconcile_segmentation_statuses(segmentations) -> list[ImageSegmentation]:
    """:func:`reconcile_segmentation_status` over an iterable. Returns the list."""
    materialized = list(segmentations)
    for segmentation in materialized:
        try:
            reconcile_segmentation_status(segmentation)
        except Exception:
            # A read endpoint must not fail because a repair did.
            logger.warning(
                "Could not reconcile status for segmentation %s",
                getattr(segmentation, "id", None),
                exc_info=True,
            )
    return materialized


__all__ = [
    "ABANDONED_RUN_MESSAGE",
    "DEFAULT_STALE_STATUS_SECONDS",
    "RUNNING_STAGES",
    "reconcile_segmentation_status",
    "reconcile_segmentation_statuses",
    "stale_after_seconds",
]

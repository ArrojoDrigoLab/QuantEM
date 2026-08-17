"""Overlay manifest build and ensure helpers."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlencode

from quantem.segmentation.models import ImageSegmentation, SegmentationOverlayState

from .constants import OVERLAY_ARRAY_KEYS, OVERLAY_CHUNK_SIZE
from .dimensions import segmentation_dimensions
from .paths import (
    get_or_create_overlay_state,
    get_overlay_active_bundle_path,
    get_overlay_debug_manifest_path,
    normalize_overlay_source_model,
)
from .store import _is_valid_label_store, _level_shapes

logger = logging.getLogger(__name__)

#: How many times the *manifest poll* may re-queue a rebuild that keeps failing
#: before it stops asking and reports the failure instead. Counted since the
#: last successful build, so a bundle that builds once starts over.
#:
#: This is the bound on the loop the viewer used to be stuck in: a rebuild that
#: cannot succeed (the spawned pool child could not import its own module) left
#: no valid store, the poll saw no store and no live job, re-queued, and the
#: user watched "Overlay updating..." for as long as they were willing to. Three
#: is enough to ride out a genuinely transient failure (a rename losing to a
#: virus scanner, a worker killed by a low-memory moment) and small enough that
#: nobody stares at a spinner for a minute of doomed rebuilds.
MANIFEST_REQUEUE_FAILURE_LIMIT = 3

#: What the overlay state records when the user cancels a rebuild.
#:
#: Cancellation is recorded through the same ``FAILED`` + reason shape a build
#: failure uses, because that shape is the only one both ends of the app treat
#: as *terminal*: ``ensure_overlay_manifest`` stops re-queueing (see
#: :func:`_cancelled_since_last_success`) and the client stops polling
#: (``overlayManifestStatus.overlayBuildFailed``). Written as DIRTY instead, a
#: cancelled job came straight back: the poll saw pending work with no live job
#: 1.5 s later and enqueued the identical rebuild, so Cancel visibly undid
#: itself and a long build could not be stopped at all.
#:
#: The string is rendered verbatim to the user under a "could not be rebuilt"
#: heading, so it has to say the two things that heading does not: that nothing
#: is broken, and how to start the update again. It deliberately does not
#: promise a previous overlay exists -- the very first build of a segmentation
#: can be cancelled too, and the card that renders this already distinguishes a
#: stale bundle from one that never built.
OVERLAY_CANCELLED_MESSAGE = (
    "Display update cancelled before it finished, so any overlay on screen is "
    "still the previous version. Use Retry overlay build, or make any edit, to "
    "start it updating again."
)


def _failed_rebuilds_since_last_success(
    segmentation: ImageSegmentation,
    state: SegmentationOverlayState,
) -> tuple[int, str]:
    """``(count, newest message)`` for this bundle's failed rebuilds.

    Only failures *after* the last successful build count: a bundle that built
    successfully has spent its history, and the next failure starts a fresh
    budget.
    """
    from .mutations import overlay_jobs_for_bundle

    jobs = overlay_jobs_for_bundle(
        str(segmentation.id),
        source_model=state.candidate_source_model,
    ).filter(status="FAILED")
    if state.last_built_at is not None:
        jobs = jobs.filter(created_at__gt=state.last_built_at)
    newest = jobs.order_by("-created_at").first()
    return jobs.count(), str(getattr(newest, "message", "") or "")


def _cancelled_since_last_success(
    segmentation: ImageSegmentation,
    state: SegmentationOverlayState,
) -> bool:
    """Was the last thing that happened to this bundle a user cancellation?

    The brake on the re-queue loop for cancelled builds, and the reason it is
    asked *here* rather than trusted from the state row. The worker's own
    ``except JobCancelledError`` arm records the stop
    (``mutations.run_overlay_rebuild_job``), but it does not always run: when
    the runner gives up waiting it terminates the worker process outright, and
    then nothing writes anything -- the state is left mid-build, looking exactly
    like pending work nobody is building, which is the shape this endpoint
    answers by enqueueing another job. The queue row is the one record of the
    cancellation that survives either path.

    Only the *newest* job counts, and only since the last successful build. A
    cancellation the user has already moved past -- they edited again, the
    follow-up ran, the bundle built -- must not keep braking; and a bundle whose
    latest attempt failed rather than being cancelled belongs to the failure
    budget above, not here.
    """
    from .mutations import overlay_jobs_for_bundle

    jobs = overlay_jobs_for_bundle(
        str(segmentation.id),
        source_model=state.candidate_source_model,
    )
    if state.last_built_at is not None:
        jobs = jobs.filter(created_at__gt=state.last_built_at)
    newest = jobs.order_by("-created_at", "-id").first()
    return newest is not None and newest.status == "CANCELLED"


def _record_manifest_failure(
    segmentation: ImageSegmentation,
    state: SegmentationOverlayState,
    *,
    reason: str,
) -> None:
    if state.status == SegmentationOverlayState.STATUS_FAILED and state.last_error == reason:
        return
    state.status = SegmentationOverlayState.STATUS_FAILED
    state.last_error = reason
    state.save(update_fields=["status", "last_error", "updated_at"])


def _try_queue_overlay_rebuild(
    segmentation: ImageSegmentation,
    *,
    mode: str,
    source_model: str | None,
) -> None:
    """Best-effort overlay-rebuild enqueue.

    A failure to enqueue (e.g. job-backend/schema issue) must never turn the
    manifest endpoint into a 500: the viewer can still render a previously built
    bundle or the aggregate fallback. The enqueue runs inside a savepoint so a
    failure rolls back cleanly without poisoning a surrounding request
    transaction (ATOMIC_REQUESTS). Log and continue instead of propagating.
    """
    from django.db import transaction

    from .mutations import queue_overlay_rebuild

    try:
        with transaction.atomic():
            queue_overlay_rebuild(segmentation, mode=mode, source_model=source_model)
    except Exception:
        logger.warning(
            "Failed to enqueue overlay rebuild for segmentation %s (source_model=%r)",
            segmentation.id,
            source_model,
            exc_info=True,
        )


def _write_debug_manifest(
    segmentation: ImageSegmentation,
    state: SegmentationOverlayState,
) -> None:
    """Drop a copy of the manifest on disk, beside the bundle, for debugging.

    Best effort, and it has to be: this is a *debug* artifact, and it is
    written on the read path. Measured, with a stray file sitting where the
    overlay directory belongs: ``mkdir(parents=True)`` raised
    ``FileExistsError: [WinError 183]`` and every ``GET .../overlay-manifest/``
    returned HTTP 500 with an empty body -- so the viewer could not even be
    told what was wrong, which is the same silence in a different costume.
    Log it and carry on; the response the caller needs does not depend on it.
    """
    try:
        manifest_path = get_overlay_debug_manifest_path(
            str(segmentation.id),
            state.candidate_source_model,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = build_overlay_manifest(segmentation, state)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        logger.warning(
            "Could not write the overlay debug manifest for segmentation %s "
            "(source_model=%r); continuing.",
            segmentation.id,
            state.candidate_source_model,
            exc_info=True,
        )


def build_overlay_manifest(
    segmentation: ImageSegmentation,
    state: SegmentationOverlayState,
) -> dict[str, Any]:
    # The manifest is the viewer's cheap status probe as well as its rendering
    # contract.  Query the one bundle-scoped queue row here so every open
    # viewer/labelling layer can tell whether what it is showing is being
    # replaced, including the time a job spends queued behind analysis.
    from .mutations import ACTIVE_OVERLAY_JOB_STATUSES, overlay_jobs_for_bundle

    update_job = (
        overlay_jobs_for_bundle(
            str(segmentation.id),
            source_model=state.candidate_source_model,
        )
        .filter(status__in=ACTIVE_OVERLAY_JOB_STATUSES)
        .order_by("-created_at")
        .values(
            "id",
            "status",
            "progress",
            "message",
            "progress_units_done",
            "progress_units_total",
            "progress_unit_label",
        )
        .first()
    )
    if update_job is not None:
        update_job["id"] = str(update_job["id"])
    active_path: str | None = None
    source_query = (
        f"?{urlencode({'source_model': state.candidate_source_model})}"
        if state.candidate_source_model
        else ""
    )
    width, height = segmentation_dimensions(segmentation)
    if state.bundle_version > 0 and _is_valid_label_store(
        get_overlay_active_bundle_path(state),
        width=width,
        height=height,
    ):
        active_path = f"/segmentation-overlays/{segmentation.id}.zarr{source_query}"
    lut_url = f"/api/segmentations/{segmentation.id}/overlay-lut/{source_query}"
    return {
        "status": state.status,
        # Always present, empty when there is nothing wrong. A ``FAILED`` status
        # with no reason beside it is the state this package exists to abolish:
        # the client must be able to say *what* went wrong without a second
        # request, and there is no second endpoint that would tell it.
        "last_error": state.last_error or "",
        "ngff_url": active_path,
        "lut_url": lut_url,
        "arrays": list(OVERLAY_ARRAY_KEYS),
        "label_dtype": "uint32",
        "overlay_kind": (
            "binary_mask"
            if segmentation.segmentation_type.measurement_mode == "global"
            else "object_ids"
        ),
        "pickable": segmentation.segmentation_type.measurement_mode != "global",
        "source_model": state.candidate_source_model or None,
        "display_role": "model" if state.candidate_source_model else "confirmed",
        # Overlay work maintains a display cache. The canonical geometry/state
        # is in the database before this job is queued, so analysis never needs
        # to wait for this flag to clear.
        "data_ready": True,
        "update_job": update_job,
        "bundle_version": state.bundle_version,
        "applied_revision": state.applied_revision,
        "desired_revision": state.desired_revision,
        "lut_revision": state.lut_revision,
        "chunk_size": [OVERLAY_CHUNK_SIZE, OVERLAY_CHUNK_SIZE],
        "level_count": len(_level_shapes(width, height)),
        "width": width,
        "height": height,
    }


def ensure_overlay_manifest(
    segmentation: ImageSegmentation,
    source_model: str | None = None,
) -> dict[str, Any]:
    from .mutations import _overlay_job_exists

    state = get_or_create_overlay_state(segmentation, source_model)
    width, height = segmentation_dimensions(segmentation)
    current_valid = state.bundle_version > 0 and _is_valid_label_store(
        get_overlay_active_bundle_path(state),
        width=width,
        height=height,
    )
    build_failed = state.status == SegmentationOverlayState.STATUS_FAILED and bool(state.last_error)
    if current_valid:
        has_pending_work = bool(state.pending_full_rebuild) or (
            int(state.desired_revision) > int(state.applied_revision)
        )
        overlay_job_active = _overlay_job_exists(
            str(segmentation.id),
            source_model=state.candidate_source_model,
        )
        if (
            has_pending_work
            and not overlay_job_active
            and not build_failed
            and _cancelled_since_last_success(segmentation, state)
        ):
            # The user cancelled the update that would have refreshed this
            # bundle. Record it as a stop, on the same terminal footing as a
            # failure, so the branch below serves the stale bundle instead of
            # enqueueing the very job they just cancelled. Gated on
            # ``has_pending_work`` so a job cancelled after it had already
            # finished writing does not drag a settled bundle out of READY.
            _record_manifest_failure(segmentation, state, reason=OVERLAY_CANCELLED_MESSAGE)
            build_failed = True
        if build_failed and not overlay_job_active:
            # The bundle on disk is usable but out of date, and the update that
            # would have refreshed it failed or was cancelled. Serve the stale
            # bundle -- seeing yesterday's objects beats seeing none -- and keep
            # the stop visible rather than resetting it to BUILDING and asking
            # again. The user's own next edit, or the rebuild button, clears it
            # and retries; see ``mutations._register_overlay_mutation_one``.
            manifest = build_overlay_manifest(segmentation, state)
            _write_debug_manifest(segmentation, state)
            return manifest
        if has_pending_work and not overlay_job_active:
            _try_queue_overlay_rebuild(
                segmentation,
                mode="full" if state.pending_full_rebuild else "partial",
                source_model=state.candidate_source_model,
            )
            overlay_job_active = _overlay_job_exists(
                str(segmentation.id),
                source_model=state.candidate_source_model,
            )
        next_status = (
            SegmentationOverlayState.STATUS_READY
            if not has_pending_work
            else (
                SegmentationOverlayState.STATUS_BUILDING
                if overlay_job_active
                else SegmentationOverlayState.STATUS_DIRTY
            )
        )
        if state.status != next_status or state.last_error:
            state.status = next_status
            state.last_error = ""
            state.save(update_fields=["status", "last_error", "updated_at"])
        manifest = build_overlay_manifest(segmentation, state)
        _write_debug_manifest(segmentation, state)
        return manifest

    overlay_job_active = _overlay_job_exists(
        str(segmentation.id),
        source_model=state.candidate_source_model,
    )
    if not overlay_job_active:
        # The bundle is missing/invalid and nothing is currently building it.
        # Recover even when a previous attempt left the state stuck in BUILDING
        # with no live job (e.g. a queued rebuild that was dropped before it
        # ran): without this, the manifest endpoint would keep reporting
        # BUILDING with no ngff_url forever and the viewer would poll a phantom
        # build. Re-scheduling here is safe because queue_overlay_rebuild is a
        # no-op when an active job already exists.
        #
        # But recovery is not unconditional, and that is the fix this package
        # carries. A rebuild that *cannot* succeed left exactly this shape --
        # no store, no live job -- so the endpoint re-queued it on every poll
        # and the viewer said "Overlay updating..." until the user gave up.
        # Two brakes, either of which stops the loop:
        #   1. the build recorded its own failure on the state (the ordinary
        #      case: ``run_overlay_rebuild_job`` writes FAILED + the reason);
        #   2. the job died without recording anything -- killed worker, lost
        #      queue -- and the count of failed jobs since the last successful
        #      build has run out of budget.
        #   3. the user cancelled it. A first-ever build has no bundle to fall
        #      back on, so without this the cancel spent no failure budget and
        #      the build the user stopped was re-queued on every poll, forever.
        if not build_failed and _cancelled_since_last_success(segmentation, state):
            _record_manifest_failure(segmentation, state, reason=OVERLAY_CANCELLED_MESSAGE)
            build_failed = True
        failure_count, failure_message = _failed_rebuilds_since_last_success(segmentation, state)
        out_of_budget = failure_count >= MANIFEST_REQUEUE_FAILURE_LIMIT
        if out_of_budget and not build_failed:
            _record_manifest_failure(
                segmentation,
                state,
                reason=(
                    f"Overlay build failed {failure_count} times and was not "
                    f"retried again. Last failure: "
                    f"{failure_message or 'the rebuild worker stopped without a message'}"
                ),
            )
        if not build_failed and not out_of_budget:
            if (
                state.status != SegmentationOverlayState.STATUS_BUILDING
                or not state.pending_full_rebuild
                or state.last_error
            ):
                state.status = SegmentationOverlayState.STATUS_BUILDING
                state.pending_full_rebuild = True
                state.last_error = ""
                state.save(
                    update_fields=[
                        "status",
                        "pending_full_rebuild",
                        "last_error",
                        "updated_at",
                    ]
                )
            _try_queue_overlay_rebuild(
                segmentation,
                mode="full",
                source_model=state.candidate_source_model,
            )

    # Display fallback: a per-source bundle that has not been built yet would
    # otherwise serve a null ngff_url (nothing to show) while the build is
    # pending -- which on large images can take a long time. If the aggregate
    # (all-sources) bundle is already valid, serve it so the viewer renders
    # overlays immediately. The per-source build stays queued above and takes
    # over for subsequent requests once it is ready.
    if normalize_overlay_source_model(source_model):
        aggregate_state = get_or_create_overlay_state(segmentation, None)
        if aggregate_state.bundle_version > 0 and _is_valid_label_store(
            get_overlay_active_bundle_path(aggregate_state),
            width=width,
            height=height,
        ):
            _write_debug_manifest(segmentation, state)
            fallback = build_overlay_manifest(segmentation, aggregate_state)
            # Draw the aggregate, but do not let it swallow the news that this
            # model's own bundle failed: the two are different pictures, and a
            # user comparing models has to know they are looking at the other
            # one.
            state.refresh_from_db(fields=["status", "last_error"])
            if state.status == SegmentationOverlayState.STATUS_FAILED and state.last_error:
                fallback["last_error"] = state.last_error
            return fallback

    manifest = build_overlay_manifest(segmentation, state)
    _write_debug_manifest(segmentation, state)
    return manifest

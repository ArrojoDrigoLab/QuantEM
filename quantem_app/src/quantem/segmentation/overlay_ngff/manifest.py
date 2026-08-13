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
        if build_failed and not overlay_job_active:
            # The bundle on disk is usable but out of date, and the update that
            # would have refreshed it failed. Serve the stale bundle -- seeing
            # yesterday's objects beats seeing none -- and keep the failure
            # visible rather than resetting it to BUILDING and asking again.
            # The user's own next edit, or the rebuild button, clears it and
            # retries; see ``mutations._register_overlay_mutation_one``.
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

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
    manifest_path = get_overlay_debug_manifest_path(
        str(segmentation.id),
        state.candidate_source_model,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_overlay_manifest(segmentation, state)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
        "ngff_url": active_path,
        "lut_url": lut_url,
        "arrays": list(OVERLAY_ARRAY_KEYS),
        "label_dtype": "uint32",
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
    if current_valid:
        has_pending_work = bool(state.pending_full_rebuild) or (
            int(state.desired_revision) > int(state.applied_revision)
        )
        overlay_job_active = _overlay_job_exists(
            str(segmentation.id),
            source_model=state.candidate_source_model,
        )
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
            return build_overlay_manifest(segmentation, aggregate_state)

    manifest = build_overlay_manifest(segmentation, state)
    _write_debug_manifest(segmentation, state)
    return manifest

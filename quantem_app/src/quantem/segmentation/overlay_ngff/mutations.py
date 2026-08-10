"""Overlay rebuild policy, mutation tracking, and job execution."""

from __future__ import annotations

import logging
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from django.db import transaction
from django.db.models import F, Q
from django.utils import timezone

from quantem.jobs.constants import (
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    QUEUE_P1_INTERACTIVE,
)
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, SegmentationOverlayState
from quantem.segmentation.services.spatial_lookup import (
    bbox_intersects_filter,
    make_bbox,
)

from . import labels_lut
from . import render as render_module
from .constants import (
    ACTIVE_OVERLAY_JOB_STATUSES,
    ASYNC_PARTIAL_MAX_IMAGE_COVERAGE,
    ASYNC_PARTIAL_MAX_LEVEL0_CHUNKS,
    BORDER_ARRAY_KEY,
    LABELS_ARRAY_KEY,
    MACRO_TILE_SIZE,
    OVERLAY_ARRAY_KEYS,
    OVERLAY_BORDER_WIDTH,
    PYRAMID_BLOCK_SIZE,
    RASTER_POOL_MIN_OBJECTS,
    RASTER_PROCESS_POOL_MAX,
    SYNC_PARTIAL_MAX_CHANGED_PIXELS,
    SYNC_PARTIAL_MAX_LEVEL0_CHUNKS,
)
from .dimensions import segmentation_dimensions
from .dirty import (
    DirtyBBox,
    _dirty_run_payload,
    _merge_dirty_runs_to_bbox,
    dirty_bbox_to_chunk_coords,
    full_image_dirty_bbox,
)
from .manifest import _write_debug_manifest
from .paths import (
    OverlayStoreError,
    _close_overlay_arrays,
    _remove_tree,
    get_or_create_overlay_state,
    get_overlay_active_bundle_path,
    get_overlay_stage_bundle_path,
    get_overlay_version_dir,
    normalize_overlay_source_model,
)
from .store import (
    _create_empty_label_store,
    _is_valid_label_store,
    _level_shapes,
    _open_label_arrays,
)

logger = logging.getLogger(__name__)


def _overlay_job_exists(segmentation_id: str, source_model: str | None = None) -> bool:
    """Is a rebuild already queued or running **for this bundle**?

    Per bundle, not per segmentation, and that is the point. A segmentation has
    an *aggregate* overlay (every object, ``source_model`` unset) and one
    *per-source* overlay for each model that produced objects in it, each with
    its own :class:`SegmentationOverlayState`, its own zarr store and its own
    revision counters. Opening a nucleus segmentation asks the manifest endpoint
    for both -- ``ensure_overlay_manifest`` serves the aggregate as the display
    fallback while the per-source bundle builds -- so two rebuild jobs is the
    correct number, and collapsing them would leave one of the two bundles
    permanently stale.

    They *look* like duplicate work in the queue only because both render under
    the one label ``"Rebuild segmentation overlay"``
    (``jobs.constants.JOB_TYPE_LABELS``) with nothing on the row to say which
    bundle each is for. The distinguishing value is on the job already, in
    ``payload_json["source_model"]`` and in the ``source_model:<name>`` tag,
    both of which ``JobSerializer`` already returns.

    Genuine duplicates -- the same bundle queued twice -- are what this filter
    prevents, and ``tests.test_overlay_ngff`` pins both halves: repeated
    requests for one bundle enqueue once, and the aggregate and the per-source
    bundle each get their own job. ``source_model`` is normalised (trimmed,
    lower-cased) before it is matched, so a differently-cased name reuses the
    job rather than starting a second one.
    """
    normalized_source_model = normalize_overlay_source_model(source_model)
    qs = Job.objects.filter(
        type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
        status__in=ACTIVE_OVERLAY_JOB_STATUSES,
        payload_json__segmentation_id=str(segmentation_id),
    )
    if normalized_source_model:
        qs = qs.filter(payload_json__source_model=normalized_source_model)
    else:
        qs = qs.filter(
            Q(payload_json__source_model__isnull=True)
            | Q(payload_json__source_model="")
        )
    return qs.exists()


def queue_overlay_rebuild(
    segmentation: ImageSegmentation,
    *,
    mode: str,
    source_model: str | None = None,
) -> Job | None:
    normalized_source_model = normalize_overlay_source_model(source_model)
    if _overlay_job_exists(str(segmentation.id), source_model=normalized_source_model):
        return None
    payload = {
        "segmentation_id": str(segmentation.id),
        "mode": mode,
    }
    if normalized_source_model:
        payload["source_model"] = normalized_source_model
    tags = [
        f"segmentation:{segmentation.id}",
        "interactive:overlay-rebuild",
    ]
    if normalized_source_model:
        tags.append(f"source_model:{normalized_source_model}")
    return Job.enqueue(
        job_type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
        payload=payload,
        priority="high",
        resource_class="cpu",
        queue_name=QUEUE_P1_INTERACTIVE,
        tags=tags,
    )


def _set_overlay_state(
    state: SegmentationOverlayState,
    *,
    status_value: str,
    applied_revision: int | None = None,
    desired_revision: int | None = None,
    bundle_version: int | None = None,
    pending_full_rebuild: bool | None = None,
    dirty_chunk_runs: list[dict[str, Any]] | None = None,
    last_error: str | None = None,
    last_built_at=None,
) -> None:
    fields = ["status", "updated_at"]
    state.status = status_value
    if applied_revision is not None:
        state.applied_revision = applied_revision
        fields.append("applied_revision")
    if desired_revision is not None:
        state.desired_revision = desired_revision
        fields.append("desired_revision")
    if bundle_version is not None:
        state.bundle_version = bundle_version
        fields.append("bundle_version")
    if pending_full_rebuild is not None:
        state.pending_full_rebuild = pending_full_rebuild
        fields.append("pending_full_rebuild")
    if dirty_chunk_runs is not None:
        state.dirty_chunk_runs = dirty_chunk_runs
        fields.append("dirty_chunk_runs")
    if last_error is not None:
        state.last_error = last_error
        fields.append("last_error")
    if last_built_at is not None:
        state.last_built_at = last_built_at
        fields.append("last_built_at")
    state.save(update_fields=fields)


def _remaining_dirty_runs_after_revision(
    runs: list[dict[str, Any]],
    *,
    applied_revision: int,
) -> list[dict[str, Any]]:
    remaining_runs: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        try:
            run_revision = int(run.get("revision", -1))
        except (TypeError, ValueError):
            remaining_runs.append(run)
            continue
        if run_revision > int(applied_revision):
            remaining_runs.append(run)
    return remaining_runs


def _finalize_overlay_rebuild_state(
    segmentation: ImageSegmentation,
    *,
    source_model: str | None = None,
    applied_revision: int,
    bundle_version: int | None = None,
    last_built_at=None,
) -> SegmentationOverlayState:
    with transaction.atomic():
        state = (
            SegmentationOverlayState.objects.select_for_update()
            .select_related("segmentation")
            .get(
                segmentation=segmentation,
                candidate_source_model=normalize_overlay_source_model(source_model),
            )
        )
        final_applied_revision = max(int(state.applied_revision), int(applied_revision))
        remaining_runs = _remaining_dirty_runs_after_revision(
            list(state.dirty_chunk_runs or []),
            applied_revision=final_applied_revision,
        )
        final_desired_revision = max(
            int(state.desired_revision),
            int(final_applied_revision),
        )
        pending_full_rebuild = bool(
            state.pending_full_rebuild and final_desired_revision > final_applied_revision
        )
        has_pending_work = (
            pending_full_rebuild
            or bool(remaining_runs)
            or final_desired_revision > final_applied_revision
        )
        _set_overlay_state(
            state,
            status_value=(
                SegmentationOverlayState.STATUS_DIRTY
                if has_pending_work
                else SegmentationOverlayState.STATUS_READY
            ),
            applied_revision=final_applied_revision,
            desired_revision=final_desired_revision,
            bundle_version=bundle_version,
            pending_full_rebuild=pending_full_rebuild,
            dirty_chunk_runs=remaining_runs,
            last_error="",
            last_built_at=last_built_at,
        )
    return state


def _object_only_fields() -> list[str]:
    return [
        "id",
        "geometry_wkb",
        "label_state",
        "refined",
        "status",
        "source_model",
    ]


def _build_draw_ops(
    objects,
    *,
    label_map: dict[Any, int],
) -> list[dict[str, Any]]:
    draw_ops: list[dict[str, Any]] = []
    for obj in objects:
        label = label_map.get(obj.id)
        if not label:
            continue
        rings = render_module.geometry_to_rings(obj.geometry)
        if not rings:
            continue
        priority, _state, _color = labels_lut.resolve_object_style(obj)
        area = 0.0
        x_min = y_min = None
        x_max = y_max = None
        for exterior, _holes in rings:
            # floor/ceil, not int(): ring coordinates carry their fractional
            # part now, and int() truncates towards zero, which would shrink the
            # box on the negative side. The box only decides which tiles an
            # object is handed to and which of two objects is painted first, so
            # it must never be tighter than the object.
            ex_x0 = math.floor(exterior[:, 0].min())
            ex_x1 = math.ceil(exterior[:, 0].max())
            ex_y0 = math.floor(exterior[:, 1].min())
            ex_y1 = math.ceil(exterior[:, 1].max())
            area += float((ex_x1 - ex_x0) * (ex_y1 - ex_y0))
            x_min = ex_x0 if x_min is None else min(x_min, ex_x0)
            y_min = ex_y0 if y_min is None else min(y_min, ex_y0)
            x_max = ex_x1 if x_max is None else max(x_max, ex_x1)
            y_max = ex_y1 if y_max is None else max(y_max, ex_y1)
        draw_ops.append(
            {
                "label": int(label),
                "priority": int(priority),
                "area": area,
                "rings": rings,
                "bbox": (x_min, y_min, x_max, y_max),
            }
        )
    return draw_ops


def _bbox_intersects(bbox, x0: int, y0: int, x1: int, y1: int) -> bool:
    bx0, by0, bx1, by1 = bbox
    return not (bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1)


def _macro_tile_payloads(
    draw_ops: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    region_x0: int = 0,
    region_y0: int = 0,
    region_x1: int | None = None,
    region_y1: int | None = None,
) -> list[dict[str, Any]]:
    region_x1 = width if region_x1 is None else region_x1
    region_y1 = height if region_y1 is None else region_y1
    halo = OVERLAY_BORDER_WIDTH
    payloads: list[dict[str, Any]] = []
    y = region_y0
    while y < region_y1:
        iy1 = min(region_y1, y + MACRO_TILE_SIZE)
        x = region_x0
        while x < region_x1:
            ix1 = min(region_x1, x + MACRO_TILE_SIZE)
            rx0 = max(0, x - halo)
            ry0 = max(0, y - halo)
            rx1 = min(width, ix1 + halo)
            ry1 = min(height, iy1 + halo)
            tile_ops = [op for op in draw_ops if _bbox_intersects(op["bbox"], rx0, ry0, rx1, ry1)]
            if tile_ops:
                payloads.append(
                    {
                        "region": (rx0, ry0, rx1, ry1),
                        "interior": (x, y, ix1, iy1),
                        "draw_ops": tile_ops,
                        "border_width": OVERLAY_BORDER_WIDTH,
                    }
                )
            x += MACRO_TILE_SIZE
        y += MACRO_TILE_SIZE
    return payloads


def _write_tile_result(arrays, result) -> None:
    interior_x0, interior_y0, labels_crop, border_crop = result
    crop_h, crop_w = labels_crop.shape
    if crop_h == 0 or crop_w == 0:
        return
    arrays[LABELS_ARRAY_KEY][0][
        interior_y0 : interior_y0 + crop_h, interior_x0 : interior_x0 + crop_w
    ] = labels_crop
    arrays[BORDER_ARRAY_KEY][0][
        interior_y0 : interior_y0 + crop_h, interior_x0 : interior_x0 + crop_w
    ] = border_crop


def _rasterize_level0(arrays, payloads: list[dict[str, Any]], *, use_pool: bool) -> None:
    if not payloads:
        return
    if use_pool and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=RASTER_PROCESS_POOL_MAX) as executor:
            for result in executor.map(render_module.rasterize_tile_worker, payloads):
                _write_tile_result(arrays, result)
    else:
        for payload in payloads:
            _write_tile_result(arrays, render_module.rasterize_tile_worker(payload))


def _build_pyramid(stage_root, *, width: int, height: int, use_pool: bool) -> None:
    """Build the pyramid in large blocks, parallelised across a process pool.

    Runs after the parent has written + closed level 0. Each parent level is
    processed in large parent-pixel blocks (vs per-256px chunk -- far fewer zarr
    calls), all-background blocks are skipped (no zero-chunks written for the
    mostly-empty gigapixel), and the non-empty blocks of a level are downsampled
    concurrently by workers that open the staged store by path. Levels are
    sequential (a level reads the one below), so each level is a pool barrier.
    """
    level_shapes = _level_shapes(width, height)
    if len(level_shapes) <= 1:
        return
    executor = ProcessPoolExecutor(max_workers=RASTER_PROCESS_POOL_MAX) if use_pool else None
    try:
        for array_key in OVERLAY_ARRAY_KEYS:
            for level_idx in range(1, len(level_shapes)):
                parent_height, parent_width = level_shapes[level_idx]
                tasks: list[tuple[str, str, int, tuple[int, int, int, int]]] = []
                for block_y0 in range(0, parent_height, PYRAMID_BLOCK_SIZE):
                    block_y1 = min(parent_height, block_y0 + PYRAMID_BLOCK_SIZE)
                    for block_x0 in range(0, parent_width, PYRAMID_BLOCK_SIZE):
                        block_x1 = min(parent_width, block_x0 + PYRAMID_BLOCK_SIZE)
                        tasks.append(
                            (
                                str(stage_root),
                                array_key,
                                level_idx,
                                (block_y0, block_y1, block_x0, block_x1),
                            )
                        )
                if executor is not None and len(tasks) > 1:
                    list(executor.map(render_module.downsample_block_worker, tasks))
                else:
                    for task in tasks:
                        render_module.downsample_block_worker(task)
    finally:
        if executor is not None:
            executor.shutdown()


def rebuild_overlay_full(
    segmentation: ImageSegmentation,
    *,
    source_model: str | None = None,
    desired_revision: int | None = None,
) -> SegmentationOverlayState:
    normalized_source_model = normalize_overlay_source_model(source_model)
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    next_bundle_version = max(1, int(state.bundle_version) + 1)
    stage_root = get_overlay_stage_bundle_path(
        str(segmentation.id),
        next_bundle_version,
        normalized_source_model,
    )
    if stage_root.parent.exists():
        _remove_tree(stage_root.parent)
    _set_overlay_state(
        state,
        status_value=SegmentationOverlayState.STATUS_BUILDING,
        last_error="",
    )

    arrays = _create_empty_label_store(segmentation, stage_root)
    width, height = segmentation_dimensions(segmentation)
    queryset = labels_lut.bundle_queryset(segmentation, normalized_source_model)
    objects = list(queryset.only(*_object_only_fields()))
    # Compact 1..N renumber on every full rebuild keeps max label ~ live count.
    assignments = [(idx + 1, obj.id) for idx, obj in enumerate(objects)]
    label_map = {obj.id: idx + 1 for idx, obj in enumerate(objects)}
    use_pool = len(objects) >= RASTER_POOL_MIN_OBJECTS
    try:
        draw_ops = _build_draw_ops(objects, label_map=label_map)
        payloads = _macro_tile_payloads(draw_ops, width=width, height=height)
        _rasterize_level0(arrays, payloads, use_pool=use_pool)
    finally:
        # Close level-0 handles before the pyramid: its workers re-open the
        # staged store by path, so the parent must not hold the arrays open.
        _close_overlay_arrays(arrays)
    _build_pyramid(stage_root, width=width, height=height, use_pool=use_pool)

    if not _is_valid_label_store(stage_root, width=width, height=height):
        raise OverlayStoreError(f"Generated overlay store is invalid at {stage_root}")

    version_dir = get_overlay_version_dir(
        str(segmentation.id),
        next_bundle_version,
        normalized_source_model,
    )
    if version_dir.exists():
        _remove_tree(version_dir)
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage_root.parent), str(version_dir))

    labels_lut.replace_bundle_labels(state, assignments=assignments)

    applied_revision = desired_revision if desired_revision is not None else state.desired_revision
    now = timezone.now()
    state = _finalize_overlay_rebuild_state(
        segmentation,
        source_model=normalized_source_model,
        applied_revision=int(applied_revision),
        bundle_version=next_bundle_version,
        last_built_at=now,
    )
    # A full rebuild renumbers labels, so clients must refetch the LUT.
    SegmentationOverlayState.objects.filter(pk=state.pk).update(
        lut_revision=F("lut_revision") + 1
    )
    state.refresh_from_db(fields=["lut_revision"])
    _write_debug_manifest(segmentation, state)
    return state


def apply_partial_overlay_update(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox,
    desired_revision: int,
    source_model: str | None = None,
) -> SegmentationOverlayState:
    normalized_source_model = normalize_overlay_source_model(source_model)
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    chunk_coords = dirty_bbox_to_chunk_coords(dirty_bbox)
    if not chunk_coords:
        _set_overlay_state(
            state,
            status_value=SegmentationOverlayState.STATUS_READY,
            applied_revision=desired_revision,
            desired_revision=desired_revision,
            last_error="",
        )
        _write_debug_manifest(segmentation, state)
        return state

    width, height = segmentation_dimensions(segmentation)
    halo = OVERLAY_BORDER_WIDTH
    interior_x0 = max(0, dirty_bbox.x_min)
    interior_y0 = max(0, dirty_bbox.y_min)
    interior_x1 = min(width, dirty_bbox.x_max)
    interior_y1 = min(height, dirty_bbox.y_max)
    region_x0 = max(0, interior_x0 - halo)
    region_y0 = max(0, interior_y0 - halo)
    region_x1 = min(width, interior_x1 + halo)
    region_y1 = min(height, interior_y1 + halo)

    # ``bbox__intersects=region_box`` becomes a numeric range filter on the
    # indexed bbox columns -- an axis-aligned rectangle needs no shapely refine.
    region_box = make_bbox(region_x0, region_y0, region_x1, region_y1)
    queryset = labels_lut.bundle_queryset(segmentation, normalized_source_model)
    objects = list(
        queryset.filter(bbox_intersects_filter(region_box)).only(*_object_only_fields())
    )
    label_map = labels_lut.existing_label_map(state)
    new_objects = [obj.id for obj in objects if obj.id not in label_map]
    if new_objects:
        label_map.update(labels_lut.allocate_labels(state, new_objects=new_objects))

    arrays = _open_label_arrays(state)
    try:
        draw_ops = _build_draw_ops(objects, label_map=label_map)
        result = render_module.rasterize_tile_worker(
            {
                "region": (region_x0, region_y0, region_x1, region_y1),
                "interior": (interior_x0, interior_y0, interior_x1, interior_y1),
                "draw_ops": draw_ops,
                "border_width": OVERLAY_BORDER_WIDTH,
            }
        )
        _write_tile_result(arrays, result)

        child_coords = chunk_coords
        for level_idx in range(1, len(arrays[LABELS_ARRAY_KEY])):
            parent_coords = {(chunk_x // 2, chunk_y // 2) for chunk_x, chunk_y in child_coords}
            for array_key in OVERLAY_ARRAY_KEYS:
                for chunk_x, chunk_y in sorted(parent_coords):
                    render_module.write_parent_chunk(
                        arrays[array_key][level_idx - 1],
                        arrays[array_key][level_idx],
                        chunk_x=chunk_x,
                        chunk_y=chunk_y,
                        kind=array_key,
                    )
            child_coords = parent_coords
    finally:
        _close_overlay_arrays(arrays)

    state = _finalize_overlay_rebuild_state(
        segmentation,
        source_model=normalized_source_model,
        applied_revision=int(desired_revision),
        last_built_at=timezone.now(),
    )
    # Geometry edits can add/remove labels, so clients must refetch the LUT.
    SegmentationOverlayState.objects.filter(pk=state.pk).update(
        lut_revision=F("lut_revision") + 1
    )
    state.refresh_from_db(fields=["lut_revision"])
    _write_debug_manifest(segmentation, state)
    return state


def overlay_rebuild_policy(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox | None,
    force_full: bool = False,
    source_model: str | None = None,
) -> str:
    state = get_or_create_overlay_state(segmentation, source_model)
    if force_full:
        return "async_full"
    if dirty_bbox is None:
        return "async_full"
    width, height = segmentation_dimensions(segmentation)
    if state.bundle_version <= 0 or not _is_valid_label_store(
        get_overlay_active_bundle_path(state),
        width=width,
        height=height,
    ):
        return "async_full"
    if state.pending_full_rebuild or state.applied_revision != state.desired_revision:
        return "async_partial"
    chunk_count = len(dirty_bbox_to_chunk_coords(dirty_bbox))
    if (
        chunk_count <= SYNC_PARTIAL_MAX_LEVEL0_CHUNKS
        and dirty_bbox.area <= SYNC_PARTIAL_MAX_CHANGED_PIXELS
    ):
        return "sync_partial"

    image_area = max(1, width * height)
    if (
        chunk_count <= ASYNC_PARTIAL_MAX_LEVEL0_CHUNKS
        and (dirty_bbox.area / image_area) <= ASYNC_PARTIAL_MAX_IMAGE_COVERAGE
    ):
        return "async_partial"
    return "async_full"


def build_overlay_mutation_response(
    state: SegmentationOverlayState,
    *,
    sync_applied: bool,
    rebuild_mode: str,
) -> dict[str, Any]:
    return {
        "desired_revision": int(state.desired_revision),
        "applied_revision": int(state.applied_revision),
        "lut_revision": int(state.lut_revision),
        "bundle_version": int(state.bundle_version),
        "sync_applied": bool(sync_applied),
        "rebuild_mode": rebuild_mode,
        "source_model": state.candidate_source_model or None,
    }


def register_state_mutation(
    segmentation: ImageSegmentation,
    *,
    source_model: str | None = None,
) -> dict[str, Any]:
    """Record a *state-only* change (confirm / reject / recolour / show-hide).

    State lives entirely in the render-time LUT, so this bumps ``lut_revision``
    on every bundle of the segmentation -- with zero raster work, no job, and no
    bundle-version change. The client refetches the (cheap) LUT and recolours
    instantly. A state change to a confirmed/manual object can affect every
    per-source bundle's LUT, so all rows are bumped.
    """
    normalized_source_model = normalize_overlay_source_model(source_model)
    get_or_create_overlay_state(segmentation, normalized_source_model)
    SegmentationOverlayState.objects.filter(segmentation=segmentation).update(
        lut_revision=F("lut_revision") + 1,
        updated_at=timezone.now(),
    )
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    return build_overlay_mutation_response(
        state,
        sync_applied=True,
        rebuild_mode="metadata_only",
    )


def _existing_bundle_source_models(segmentation: ImageSegmentation) -> list[str]:
    source_models = list(
        SegmentationOverlayState.objects.filter(segmentation=segmentation)
        .values_list("candidate_source_model", flat=True)
        .distinct()
    )
    if "" not in source_models:
        source_models.append("")
    return source_models


def _register_overlay_mutation_one(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox | None,
    force_full_rebuild: bool = False,
    source_model: str | None = None,
) -> dict[str, Any]:
    normalized_source_model = normalize_overlay_source_model(source_model)
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    desired_revision = int(state.desired_revision) + 1
    rebuild_mode = overlay_rebuild_policy(
        segmentation,
        dirty_bbox=dirty_bbox,
        force_full=force_full_rebuild,
        source_model=normalized_source_model,
    )

    if rebuild_mode == "sync_partial" and dirty_bbox is not None:
        state.desired_revision = desired_revision
        state.save(update_fields=["desired_revision", "updated_at"])
        try:
            state = apply_partial_overlay_update(
                segmentation,
                dirty_bbox=dirty_bbox,
                desired_revision=desired_revision,
                source_model=normalized_source_model,
            )
            return build_overlay_mutation_response(
                state,
                sync_applied=True,
                rebuild_mode=rebuild_mode,
            )
        except OverlayStoreError:
            rebuild_mode = "async_full"
            state.refresh_from_db()
            desired_revision = int(state.desired_revision) + 1

    dirty_runs = list(state.dirty_chunk_runs or [])
    if dirty_bbox is not None and rebuild_mode == "async_partial":
        dirty_runs.append(_dirty_run_payload(revision=desired_revision, dirty_bbox=dirty_bbox))
    pending_full_rebuild = rebuild_mode == "async_full" or force_full_rebuild
    status_value = (
        SegmentationOverlayState.STATUS_BUILDING
        if pending_full_rebuild
        and _overlay_job_exists(
            str(segmentation.id),
            source_model=normalized_source_model,
        )
        else SegmentationOverlayState.STATUS_DIRTY
    )
    _set_overlay_state(
        state,
        status_value=status_value,
        desired_revision=desired_revision,
        pending_full_rebuild=pending_full_rebuild,
        dirty_chunk_runs=dirty_runs,
        last_error="",
    )
    queue_overlay_rebuild(
        segmentation,
        mode="full" if pending_full_rebuild else "partial",
        source_model=normalized_source_model,
    )
    _write_debug_manifest(segmentation, state)
    return build_overlay_mutation_response(
        state,
        sync_applied=False,
        rebuild_mode=rebuild_mode,
    )


def register_overlay_mutation(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox | None,
    force_full_rebuild: bool = False,
    source_model: str | None = None,
) -> dict[str, Any]:
    """Register a geometry edit scoped to one source bundle (+ the aggregate).

    Use :func:`register_overlay_mutation_all_bundles` instead when the edited
    object is confirmed/manual and therefore a member of every bundle.
    """
    normalized_source_model = normalize_overlay_source_model(source_model)
    if normalized_source_model:
        _register_overlay_mutation_one(
            segmentation,
            dirty_bbox=dirty_bbox,
            force_full_rebuild=force_full_rebuild,
            source_model="",
        )
        return _register_overlay_mutation_one(
            segmentation,
            dirty_bbox=dirty_bbox,
            force_full_rebuild=force_full_rebuild,
            source_model=normalized_source_model,
        )

    return _register_overlay_mutation_one(
        segmentation,
        dirty_bbox=dirty_bbox,
        force_full_rebuild=force_full_rebuild,
        source_model="",
    )


def register_overlay_mutation_all_bundles(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox | None,
    force_full_rebuild: bool = False,
    source_model: str | None = None,
) -> dict[str, Any]:
    """Fan a geometry edit to every existing bundle of the segmentation.

    Confirmed/manual objects belong to all per-source bundles (the membership
    rule), so an edit to one must dirty them all.
    """
    normalized_source_model = normalize_overlay_source_model(source_model)
    target_source_models = _existing_bundle_source_models(segmentation)
    if normalized_source_model and normalized_source_model not in target_source_models:
        target_source_models.append(normalized_source_model)
    responses: dict[str, dict[str, Any]] = {}
    for candidate_source_model in target_source_models:
        responses[candidate_source_model] = _register_overlay_mutation_one(
            segmentation,
            dirty_bbox=dirty_bbox,
            force_full_rebuild=force_full_rebuild,
            source_model=candidate_source_model,
        )
    return (
        responses.get(normalized_source_model)
        or responses.get("")
        or next(iter(responses.values()))
    )


def queue_full_overlay_rebuild(
    segmentation: ImageSegmentation,
    *,
    source_model: str | None = None,
) -> dict[str, Any]:
    return register_overlay_mutation(
        segmentation,
        dirty_bbox=full_image_dirty_bbox(segmentation),
        force_full_rebuild=True,
        source_model=source_model,
    )


def run_overlay_rebuild_job(
    segmentation: ImageSegmentation,
    *,
    mode: str,
    source_model: str | None = None,
) -> SegmentationOverlayState:
    normalized_source_model = normalize_overlay_source_model(source_model)
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    _set_overlay_state(
        state,
        status_value=SegmentationOverlayState.STATUS_BUILDING,
        last_error="",
    )
    try:
        if mode == "full" or state.pending_full_rebuild:
            return rebuild_overlay_full(
                segmentation,
                source_model=normalized_source_model,
                desired_revision=int(state.desired_revision),
            )

        dirty_bbox = _merge_dirty_runs_to_bbox(segmentation, list(state.dirty_chunk_runs or []))
        if dirty_bbox is None:
            _set_overlay_state(
                state,
                status_value=SegmentationOverlayState.STATUS_READY,
                applied_revision=int(state.desired_revision),
                dirty_chunk_runs=[],
                pending_full_rebuild=False,
                last_error="",
                last_built_at=timezone.now(),
            )
            _write_debug_manifest(segmentation, state)
            return state

        return apply_partial_overlay_update(
            segmentation,
            dirty_bbox=dirty_bbox,
            desired_revision=int(state.desired_revision),
            source_model=normalized_source_model,
        )
    except Exception as exc:
        logger.error(
            "Overlay rebuild failed for segmentation %s: %s",
            segmentation.id,
            exc,
            exc_info=True,
        )
        _set_overlay_state(
            state,
            status_value=SegmentationOverlayState.STATUS_FAILED,
            last_error=str(exc),
        )
        _write_debug_manifest(segmentation, state)
        raise

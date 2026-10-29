"""Overlay rebuild policy, mutation tracking, and job execution."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import wraps
from typing import Any

import numpy as np
from django.db import transaction
from django.db.models import F, Q
from django.utils import timezone

from quantem.core.performance import timed_stage
from quantem.jobs.constants import (
    JOB_DEFAULTS,
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
)
from quantem.jobs.models import Job
from quantem.jobs.pool import django_pool_initializer
from quantem.jobs.reporter import JobCancelledError
from quantem.segmentation.global_masks import load_global_mask
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationOverlayLabel,
    SegmentationOverlayState,
)
from quantem.segmentation.services.spatial_lookup import (
    bbox_intersects_filter,
    make_bbox,
)
from quantem.segmentation.source_models import SOURCE_MODEL_MANUAL

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
    OVERLAY_CHUNK_SIZE,
    PYRAMID_BLOCK_SIZE,
    RASTER_POOL_MIN_OBJECTS,
    RASTER_POOL_MIN_PYRAMID_BLOCKS,
    RASTER_POOL_WINDOW_MULTIPLIER,
    SYNC_PARTIAL_MAX_CHANGED_PIXELS,
    SYNC_PARTIAL_MAX_LEVEL0_CHUNKS,
    raster_process_pool_size,
)
from .dimensions import segmentation_dimensions
from .dirty import (
    DirtyBBox,
    _dirty_run_payload,
    dirty_bbox_to_chunk_coords,
    dirty_runs_to_chunk_coords,
    full_image_dirty_bbox,
)
from .failure_text import describe_failure
from .manifest import OVERLAY_CANCELLED_MESSAGE, _write_debug_manifest
from .paths import (
    OverlayStoreError,
    _close_overlay_arrays,
    _remove_tree,
    get_or_create_overlay_state,
    get_overlay_active_bundle_path,
    get_overlay_root,
    get_overlay_stage_bundle_path,
    get_overlay_version_dir,
    normalize_overlay_source_model,
)
from .snapshots import clone_store, read_snapshot, schedule_snapshot_prune, write_snapshot
from .store import (
    _create_empty_label_store,
    _is_valid_label_store,
    _level_shapes,
    _open_label_arrays,
    _retry_on_windows_lock,
)

logger = logging.getLogger(__name__)


class OverlayRenderPoolError(RuntimeError):
    """The rasterisation worker pool died before it produced the overlay.

    Distinct from :class:`~.paths.OverlayStoreError`, which means "the store on
    disk is unusable" and has a downgrade path (a failed sync-partial retries as
    a full async rebuild). A dead pool has no such path -- retrying it in the
    same interpreter reproduces it -- so it must reach the job as a failure with
    the underlying cause attached, never as a silent re-queue.
    """


def _raise_broken_pool(
    exc: BrokenProcessPool,
    *,
    stage: str,
    task_count: int,
    worker_count: int,
) -> None:
    """Re-raise a dead pool as something that names what died and why.

    ``BrokenProcessPool`` says only that "a process ... was terminated
    abruptly": not which pool, not what it was doing, and -- because the child
    dies during *its own start-up import*, before any task -- not the
    ImportError that actually killed it. Everything specific has to be added
    here or the user is handed a sentence with no content in it.

    ``worker_count`` is passed in rather than read from the constant because the
    pool is sized from the machine profile now: on a laptop the sentence has to
    say 2, not the 4 this module used to hard-code.
    """
    raise OverlayRenderPoolError(
        f"Overlay {stage} failed: the {worker_count} background "
        f"rendering workers stopped before finishing {task_count} tiles. "
        f"Underlying error: {exc}"
    ) from exc


#: Name of the per-bundle write lock, kept in the bundle's own root so it
#: survives a bundle-version bump and is never inside a tree that gets removed.
BUNDLE_WRITE_LOCK_FILENAME = ".write.lock"
#: How long a writer waits for the bundle before giving up. A partial update is
#: ~100 ms of work, so this is two orders of magnitude of headroom; it exists to
#: turn a stuck neighbour into a stated failure rather than a hung request.
BUNDLE_WRITE_LOCK_TIMEOUT_SECONDS = 30.0
_BUNDLE_WRITE_LOCK_POLL_SECONDS = 0.01

_bundle_thread_locks: dict[str, threading.Lock] = {}
_bundle_thread_locks_guard = threading.Lock()


def _bundle_thread_lock(key: str) -> threading.Lock:
    with _bundle_thread_locks_guard:
        lock = _bundle_thread_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _bundle_thread_locks[key] = lock
        return lock


def _try_lock_file(handle) -> bool:
    """Take an exclusive OS lock on ``handle``, or report that it is taken.

    Both primitives used here are released by the kernel when the holding
    process exits, which is the property that matters: a job worker killed
    mid-write must not leave a bundle permanently unwritable. A lock implemented
    as "create a file, delete it afterwards" does not have it.

    On Windows the lock is a *byte range* taken at the handle's current
    position, so the seek is not tidiness: lock and unlock must name the same
    byte or the unlock silently does nothing.
    """
    if os.name == "nt":
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        with contextlib.suppress(OSError):
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    with contextlib.suppress(OSError):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def bundle_write_lock(
    segmentation_id: str,
    source_model: str | None = None,
    *,
    timeout: float = BUNDLE_WRITE_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Serialize full and partial snapshot writers for one bundle.

    Both paths choose the next version, stage pixels and publish label mappings
    under this lock. Threads and worker processes share the OS file lock; the
    kernel releases it if a writer exits. Request transactions use a nonblocking
    attempt so they cannot hold a database row while waiting for a job that
    needs that row to publish.
    """
    bundle_root = get_overlay_root(str(segmentation_id), source_model)
    key = str(bundle_root)
    lock_path = bundle_root / BUNDLE_WRITE_LOCK_FILENAME
    deadline = time.monotonic() + max(0.0, timeout)

    thread_lock = _bundle_thread_lock(key)
    if not thread_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise OverlayStoreError(
            "Another change to this overlay is still being written. "
            "Wait for it to finish and try again."
        )
    try:
        bundle_root.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as handle:
            while True:
                if _try_lock_file(handle):
                    break
                if time.monotonic() >= deadline:
                    raise OverlayStoreError(
                        "Another change to this overlay is still being written. "
                        "Wait for it to finish and try again."
                    )
                time.sleep(_BUNDLE_WRITE_LOCK_POLL_SECONDS)
            try:
                yield
            finally:
                _unlock_file(handle)
    finally:
        thread_lock.release()


def overlay_jobs_for_bundle(segmentation_id: str, source_model: str | None = None):
    """Every rebuild job for one bundle, in no particular order.

    Bundle, not segmentation: a segmentation has one confirmed-display overlay
    (``source_model`` unset) and one per-source preview overlay for each model
    that produced objects in it, each with its own state row, store and revisions.
    See :func:`_overlay_job_exists` for why two jobs in the queue is the correct
    number rather than a duplicate.
    """
    normalized_source_model = normalize_overlay_source_model(source_model)
    qs = Job.objects.filter(
        type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
        payload_json__segmentation_id=str(segmentation_id),
    )
    if normalized_source_model:
        return qs.filter(payload_json__source_model=normalized_source_model)
    return qs.filter(Q(payload_json__source_model__isnull=True) | Q(payload_json__source_model=""))


def _overlay_job_exists(segmentation_id: str, source_model: str | None = None) -> bool:
    """Is a rebuild already queued or running **for this bundle**?

    Per bundle, not per segmentation, and that is the point. A segmentation has
    a *confirmed-display* overlay (an all-object raster rendered through a
    confirmed-only LUT, ``source_model`` unset) and one *per-source* preview
    overlay for each model that produced objects in it, each with
    its own :class:`SegmentationOverlayState`, its own zarr store and its own
    revision counters. Opening a nucleus segmentation asks the manifest endpoint
    for both -- ``ensure_overlay_manifest`` may serve the confirmed raster as a
    temporary display fallback while the per-source bundle builds -- so two rebuild jobs is the
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
    return (
        overlay_jobs_for_bundle(segmentation_id, source_model)
        .filter(status__in=ACTIVE_OVERLAY_JOB_STATUSES)
        .exists()
    )


@transaction.atomic
def queue_overlay_rebuild(
    segmentation: ImageSegmentation,
    *,
    mode: str,
    source_model: str | None = None,
) -> Job | None:
    normalized_source_model = normalize_overlay_source_model(source_model)
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    # UPDATE also takes the write lock on SQLite, where select_for_update is a
    # no-op. The existence check and enqueue must be one serialized operation.
    SegmentationOverlayState.objects.filter(pk=state.pk).update(lut_revision=F("lut_revision"))
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
        "background:overlay-rebuild",
    ]
    if normalized_source_model:
        tags.append(f"source_model:{normalized_source_model}")
    defaults = JOB_DEFAULTS[JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY]
    return Job.enqueue(
        job_type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
        payload=payload,
        priority=defaults["priority"],
        resource_class=defaults["resource_class"],
        queue_name=defaults["queue_name"],
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
    columns = math.ceil((region_x1 - region_x0) / MACRO_TILE_SIZE)
    rows = math.ceil((region_y1 - region_y0) / MACRO_TILE_SIZE)
    # Assign each outline once to the macro tiles it can touch. Scanning all
    # objects for every tile made sparse gigapixel images O(objects * tiles).
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for op in draw_ops:
        x0, y0, x1, y1 = op["bbox"]
        first_x = max(0, math.ceil((x0 - halo - region_x0) / MACRO_TILE_SIZE) - 1)
        last_x = min(columns - 1, math.floor((x1 + halo - region_x0) / MACRO_TILE_SIZE))
        first_y = max(0, math.ceil((y0 - halo - region_y0) / MACRO_TILE_SIZE) - 1)
        last_y = min(rows - 1, math.floor((y1 + halo - region_y0) / MACRO_TILE_SIZE))
        for cy in range(first_y, last_y + 1):
            for cx in range(first_x, last_x + 1):
                buckets.setdefault((cy, cx), []).append(op)
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
            candidates = buckets.get(
                ((y - region_y0) // MACRO_TILE_SIZE, (x - region_x0) // MACRO_TILE_SIZE), ()
            )
            tile_ops = [op for op in candidates if _bbox_intersects(op["bbox"], rx0, ry0, rx1, ry1)]
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
    """Write one rasterised tile's crops into level 0, retrying a locked file.

    The retry is the same bounded one the store metadata writes already use, for
    the same reason: zarr publishes a chunk with ``tmp.replace(target)``, and on
    Windows that raises ``PermissionError`` while *anything* holds a transient
    handle on the target -- an antivirus scanner or the search indexer opening a
    just-written file is enough, and both do it routinely. POSIX rename cannot
    fail that way, so upstream does not retry. Windows is a shipping platform.

    This is the second half of the concurrency fix and not a substitute for the
    first: :func:`bundle_write_lock` stops two writers colliding *by design*,
    and this absorbs the collisions no lock of ours can see.
    """
    interior_x0, interior_y0, labels_crop, border_crop = result
    crop_h, crop_w = labels_crop.shape
    if crop_h == 0 or crop_w == 0:
        return
    labels_level0 = arrays[LABELS_ARRAY_KEY][0]
    border_level0 = arrays[BORDER_ARRAY_KEY][0]
    y_slice = slice(interior_y0, interior_y0 + crop_h)
    x_slice = slice(interior_x0, interior_x0 + crop_w)
    _retry_on_windows_lock(lambda: labels_level0.__setitem__((y_slice, x_slice), labels_crop))
    _retry_on_windows_lock(lambda: border_level0.__setitem__((y_slice, x_slice), border_crop))


def _rasterize_level0(
    arrays,
    payloads: list[dict[str, Any]],
    *,
    use_pool: bool,
    on_progress=None,
    cancel_check=None,
) -> None:
    """Rasterise every macro tile and write it, holding a bounded number at once.

    The parent's memory here is not set by how many objects there are but by how
    many finished tiles it is holding, and that used to be "all of them".
    ``executor.map(worker, payloads)`` submits the whole list up front; the
    workers return a 2048^2 ``uint32`` labels crop plus a 2048^2 ``uint8`` border
    crop (21 MB a tile) faster than :func:`_write_tile_result` can push 64
    compressed chunks into zarr, so completed results queue up in the parent
    until the consumer catches up. Measured: 2 051 MB on a 419 MP canvas, and
    linear in canvas area -- about 16 GB at 3 224 MP -- with the object count
    making no difference (2 000 objects cost the same as 20 000).

    So submission is windowed: at most ``RASTER_POOL_WINDOW_MULTIPLIER x
    workers`` tasks are outstanding, and the loop consumes them **in submission
    order**, exactly as ``map`` did. Order matters for one reason: it makes the
    written bytes provably unchanged rather than argued to be unchanged. (Tile
    interiors are disjoint and 2048 is a multiple of the 256 chunk size, so no
    two tiles share a chunk and any completion order would in fact produce the
    same store -- but the FIFO costs nothing and removes the argument.)
    """
    if not payloads:
        return
    if not (use_pool and len(payloads) > 1):
        for index, payload in enumerate(payloads, start=1):
            if cancel_check is not None:
                cancel_check()
            _write_tile_result(arrays, render_module.rasterize_tile_worker(payload))
            if on_progress is not None:
                on_progress("raster", index, len(payloads))
        return

    workers = raster_process_pool_size()
    window = max(2, workers * RASTER_POOL_WINDOW_MULTIPLIER)
    # ``initializer`` is not optional: the child unpickles
    # ``render_module.rasterize_tile_worker`` by importing
    # ``quantem.segmentation.overlay_ngff.render``, whose package __init__
    # reaches Django models. Without django.setup() first, every worker dies
    # on that import and the pool comes back broken. See quantem.jobs.pool.
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=django_pool_initializer,
    ) as executor:
        outstanding: deque[Future] = deque()
        completed = 0
        try:
            for payload in payloads:
                outstanding.append(executor.submit(render_module.rasterize_tile_worker, payload))
                if len(outstanding) >= window:
                    if cancel_check is not None:
                        cancel_check()
                    _write_tile_result(arrays, outstanding.popleft().result())
                    completed += 1
                    if on_progress is not None:
                        on_progress("raster", completed, len(payloads))
            while outstanding:
                if cancel_check is not None:
                    cancel_check()
                _write_tile_result(arrays, outstanding.popleft().result())
                completed += 1
                if on_progress is not None:
                    on_progress("raster", completed, len(payloads))
        except BrokenProcessPool as exc:
            _raise_broken_pool(
                exc,
                stage="rasterisation",
                task_count=len(payloads),
                worker_count=workers,
            )
        finally:
            # Do not make a failure wait out the tail of the queue: whatever is
            # still unstarted when we leave here is not going to be written.
            while outstanding:
                outstanding.popleft().cancel()


def _pyramid_level_blocks(
    level_shapes: list[tuple[int, int]],
    content_bboxes: list[tuple[int, int, int, int]] | None,
) -> dict[int, list[tuple[int, int, int, int]]]:
    """Map each pyramid level to the blocks that can hold content.

    A parent block is worth visiting only if some level-0 geometry projects into
    it; every other block is background, and background downsamples to
    background. Enumerating them all instead is what made the pyramid pass cost
    scale with *canvas area* rather than with annotated area -- ~2 100 blocks per
    array on a 165 231 x 153 701 asset, each of them a zarr group open and a
    4096-square read that found nothing.

    **Why the bboxes bound the raster.** Level-0 pixels only ever land inside a
    draw op's bbox. :func:`~quantem.seg_core.rasterize.paint_ring` paints only
    ``ceil(min) .. ceil(max) - 1`` of the ring it is given (``ring_window``),
    which sits inside the ``floor(min) .. ceil(max)`` box
    :func:`_build_draw_ops` records; holes paint background, which can only
    remove pixels; and the baked border is ``diff & foreground`` re-masked to
    ``foreground`` after dilation (:func:`~.render._compute_border`), so it
    cannot reach past the object either. Then by induction: level 1 reads a
    fully written level 0, and each level's block set covers all the content of
    the level below.

    Level 1's blocks come from the bboxes directly, padded a pixel each way so
    halving rounding can never drop an edge. Above that a child block index
    ``i`` spans parent block ``i // 2`` -- ``PYRAMID_BLOCK_SIZE`` is constant
    while the level extent halves -- so each level's set is the previous one
    halved. That is the same child-to-parent walk
    :func:`apply_partial_overlay_update` already does over chunk coordinates.

    ``content_bboxes=None`` means "unknown, visit everything" and preserves the
    old behaviour exactly; an empty list means "no content", which correctly
    visits nothing.
    """
    blocks_by_level: dict[int, list[tuple[int, int, int, int]]] = {}
    if len(level_shapes) <= 1:
        return blocks_by_level

    # ``None`` and ``[]`` must not collapse into each other here: the first has
    # to fall through to the visit-everything branch below, the second has to
    # produce an empty index set. Hence the explicit ``is not None``.
    indices: set[tuple[int, int]] | None = None
    if content_bboxes is not None:
        indices = set()
        for x_min, y_min, x_max, y_max in content_bboxes:
            # Level-0 -> level-1 pixels (child p feeds parent p // 2), padded a
            # pixel each way.
            level1_y0 = max(0, y_min // 2 - 1)
            level1_x0 = max(0, x_min // 2 - 1)
            level1_y1 = max(0, y_max // 2 + 1)
            level1_x1 = max(0, x_max // 2 + 1)
            for block_y in range(
                level1_y0 // PYRAMID_BLOCK_SIZE, level1_y1 // PYRAMID_BLOCK_SIZE + 1
            ):
                for block_x in range(
                    level1_x0 // PYRAMID_BLOCK_SIZE, level1_x1 // PYRAMID_BLOCK_SIZE + 1
                ):
                    indices.add((block_y, block_x))

    for level_idx in range(1, len(level_shapes)):
        parent_height, parent_width = level_shapes[level_idx]
        block_rows = max(1, math.ceil(parent_height / PYRAMID_BLOCK_SIZE))
        block_cols = max(1, math.ceil(parent_width / PYRAMID_BLOCK_SIZE))
        if indices is None:
            level_indices = {
                (block_y, block_x) for block_y in range(block_rows) for block_x in range(block_cols)
            }
        else:
            if level_idx > 1:
                indices = {(block_y // 2, block_x // 2) for block_y, block_x in indices}
            # A bbox that runs off the image, or the pixel of padding above,
            # can name a block this level does not have.
            level_indices = {
                (block_y, block_x)
                for block_y, block_x in indices
                if block_y < block_rows and block_x < block_cols
            }
        blocks: list[tuple[int, int, int, int]] = []
        for block_y, block_x in sorted(level_indices):
            block_y0 = block_y * PYRAMID_BLOCK_SIZE
            block_x0 = block_x * PYRAMID_BLOCK_SIZE
            blocks.append(
                (
                    block_y0,
                    min(parent_height, block_y0 + PYRAMID_BLOCK_SIZE),
                    block_x0,
                    min(parent_width, block_x0 + PYRAMID_BLOCK_SIZE),
                )
            )
        blocks_by_level[level_idx] = blocks
    return blocks_by_level


def _build_pyramid(
    stage_root,
    *,
    width: int,
    height: int,
    content_bboxes: list[tuple[int, int, int, int]] | None = None,
    on_progress=None,
    cancel_check=None,
) -> None:
    """Build the pyramid in large blocks over the level-0 content region.

    Runs after the parent has written + closed level 0. Each parent level is
    processed in large parent-pixel blocks (vs per-256px chunk -- far fewer zarr
    calls), restricted to the blocks ``content_bboxes`` can reach
    (:func:`_pyramid_level_blocks`), and all-background blocks among those are
    still skipped so no zero-chunks are written. Cost therefore scales with how
    much of the canvas is annotated rather than with how big the canvas is: a
    freshly created segmentation of a gigapixel asset visits a handful of blocks
    instead of every block of every level.

    Whether to spawn a pool is decided from that block count, because a block is
    the unit of work here. It used to inherit ``use_pool`` from the *object*
    count, which had the sign backwards in the worst case -- a huge canvas with
    few objects, which is exactly a segmentation someone has just started, took
    the fully sequential path over thousands of blocks.

    Levels are sequential (a level reads the one below), so each level is a pool
    barrier.
    """
    level_shapes = _level_shapes(width, height)
    blocks_by_level = _pyramid_level_blocks(level_shapes, content_bboxes)
    total_blocks = sum(len(blocks) for blocks in blocks_by_level.values()) * len(OVERLAY_ARRAY_KEYS)
    if not total_blocks:
        return

    if total_blocks < RASTER_POOL_MIN_PYRAMID_BLOCKS:
        # Few blocks: one store handle for all of them, rather than a group
        # open per block. Levels ascend within each array, so a level always
        # reads one that is already finished.
        group = render_module.open_staged_group(str(stage_root))
        completed = 0
        try:
            for array_key in OVERLAY_ARRAY_KEYS:
                for level_idx in sorted(blocks_by_level):
                    for block in blocks_by_level[level_idx]:
                        if cancel_check is not None:
                            cancel_check()
                        render_module.downsample_block(group, array_key, level_idx, block)
                        completed += 1
                        if on_progress is not None:
                            on_progress("pyramid", completed, total_blocks)
        finally:
            # Before the caller moves the staged bundle, not after.
            render_module.close_staged_group(group)
        return

    # Same contract as _rasterize_level0: the child imports
    # ``overlay_ngff.render`` to unpickle ``downsample_block_worker``, so it
    # needs a loaded app registry before its first task.
    #
    # No submission window here, deliberately. ``downsample_block_worker``
    # opens the staged store by path and returns ``None``: the tasks are small
    # tuples and the results are nothing, so ``map`` accumulates nothing in the
    # parent. The backlog fixed in _rasterize_level0 is specifically a backlog
    # of *returned arrays*.
    workers = raster_process_pool_size()
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=django_pool_initializer,
    )
    try:
        completed = 0
        for array_key in OVERLAY_ARRAY_KEYS:
            for level_idx in sorted(blocks_by_level):
                tasks: list[tuple[str, str, int, tuple[int, int, int, int]]] = [
                    (str(stage_root), array_key, level_idx, block)
                    for block in blocks_by_level[level_idx]
                ]
                if len(tasks) > 1:
                    try:
                        for _result in executor.map(render_module.downsample_block_worker, tasks):
                            if cancel_check is not None:
                                cancel_check()
                            completed += 1
                            if on_progress is not None:
                                on_progress("pyramid", completed, total_blocks)
                    except BrokenProcessPool as exc:
                        _raise_broken_pool(
                            exc,
                            stage="pyramid build",
                            task_count=len(tasks),
                            worker_count=workers,
                        )
                else:
                    for task in tasks:
                        if cancel_check is not None:
                            cancel_check()
                        render_module.downsample_block_worker(task)
                        completed += 1
                        if on_progress is not None:
                            on_progress("pyramid", completed, total_blocks)
    finally:
        executor.shutdown()


def _serialized_build(function):
    @wraps(function)
    def build(segmentation, *, source_model=None, **kwargs):
        timeout = kwargs.pop("_lock_timeout", BUNDLE_WRITE_LOCK_TIMEOUT_SECONDS)
        with bundle_write_lock(str(segmentation.id), source_model, timeout=timeout):
            return function(segmentation, source_model=source_model, **kwargs)

    return build


def _publish_snapshot(
    segmentation,
    state,
    stage_root,
    *,
    version,
    revision,
    assignments,
    new_assignments=None,
    max_label=None,
    geometry_revision=None,
):
    """Publish pixels and their label map as one database-visible generation."""
    # Another serialized writer may have completed while this job waited for
    # the file lock. Never publish a token older than the starting snapshot.
    revision = max(int(revision), int(state.applied_revision))
    write_snapshot(
        stage_root,
        bundle_version=version,
        applied_revision=revision,
        assignments=assignments,
        max_label=max_label,
        geometry_revision=geometry_revision,
    )
    version_dir = get_overlay_version_dir(
        str(segmentation.id), version, state.candidate_source_model
    )
    if version_dir.exists():
        # An earlier attempt may have moved its files and died before commit.
        _remove_tree(version_dir)
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage_root.parent), str(version_dir))
    with transaction.atomic():
        if new_assignments is None:
            labels_lut.replace_bundle_labels(state, assignments=assignments)
        else:
            # Old readers use immutable labels.json. Reclaim obsolete live
            # index rows on every partial publication without touching theirs.
            labels_lut.remove_obsolete_labels(state, assignments)
            # Existing identities do not change on a partial build. Avoid
            # deleting/reinserting every object's row for a one-tile edit.
            SegmentationOverlayLabel.objects.bulk_create(
                [
                    SegmentationOverlayLabel(
                        overlay_state=state, label=label, object_uuid=object_id
                    )
                    for label, object_id in new_assignments
                ],
                batch_size=2000,
            )
        state = _finalize_overlay_rebuild_state(
            segmentation,
            source_model=state.candidate_source_model,
            applied_revision=revision,
            bundle_version=version,
            last_built_at=timezone.now(),
        )
        SegmentationOverlayState.objects.filter(pk=state.pk).update(
            lut_revision=F("lut_revision") + 1
        )
        state.refresh_from_db(fields=["lut_revision"])

    # A synchronous edit can publish several generations inside one outer
    # transaction. Its old database-visible snapshot must survive a rollback.
    def after_commit():
        schedule_snapshot_prune(str(segmentation.id), state.candidate_source_model, version)
        _write_debug_manifest(segmentation, state)

    transaction.on_commit(after_commit, robust=True)
    return state


@_serialized_build
@timed_stage("overlay_full")
def rebuild_overlay_full(
    segmentation: ImageSegmentation,
    *,
    source_model: str | None = None,
    desired_revision: int | None = None,
    on_progress=None,
    cancel_check=None,
) -> SegmentationOverlayState:
    geometry_revision = ImageSegmentation.objects.values_list("geometry_revision", flat=True).get(
        pk=segmentation.pk
    )
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
    is_global = segmentation.segmentation_type.measurement_mode == "global"
    queryset = labels_lut.bundle_queryset(segmentation, normalized_source_model)
    objects = [] if is_global else list(queryset.only(*_object_only_fields()))
    if on_progress is not None:
        on_progress("objects", len(objects), len(objects))
    # Global overlays use label 1 only and intentionally have no label->object
    # assignment. Object overlays retain their compact 1..N identity map.
    assignments = [] if is_global else [(idx + 1, obj.id) for idx, obj in enumerate(objects)]
    label_map = {} if is_global else {obj.id: idx + 1 for idx, obj in enumerate(objects)}
    content_bboxes: list[tuple[int, int, int, int]] = []
    try:
        if is_global:
            mask = load_global_mask(segmentation)
            total = math.ceil(width / MACRO_TILE_SIZE) * math.ceil(height / MACRO_TILE_SIZE)
            done = 0
            for y0 in range(0, height, MACRO_TILE_SIZE):
                for x0 in range(0, width, MACRO_TILE_SIZE):
                    if cancel_check is not None:
                        cancel_check()
                    x1, y1 = min(width, x0 + MACRO_TILE_SIZE), min(height, y0 + MACRO_TILE_SIZE)
                    rx0, ry0 = max(0, x0 - OVERLAY_BORDER_WIDTH), max(0, y0 - OVERLAY_BORDER_WIDTH)
                    rx1, ry1 = (
                        min(width, x1 + OVERLAY_BORDER_WIDTH),
                        min(height, y1 + OVERLAY_BORDER_WIDTH),
                    )
                    block = mask[ry0:ry1, rx0:rx1]
                    if block.any():
                        labels = block.astype(np.uint32)
                        border = render_module._compute_border(labels, width=OVERLAY_BORDER_WIDTH)
                        interior = (slice(y0 - ry0, y1 - ry0), slice(x0 - rx0, x1 - rx0))
                        _write_tile_result(arrays, (x0, y0, labels[interior], border[interior]))
                    done += 1
                    if on_progress is not None:
                        on_progress("raster", done, total)
            occupied_rows = np.flatnonzero(mask.any(axis=1))
            occupied_cols = np.flatnonzero(mask.any(axis=0))
            if occupied_rows.size and occupied_cols.size:
                content_bboxes = [
                    (
                        int(occupied_cols[0]),
                        int(occupied_rows[0]),
                        int(occupied_cols[-1]) + 1,
                        int(occupied_rows[-1]) + 1,
                    )
                ]
        else:
            draw_ops = _build_draw_ops(objects, label_map=label_map)
            # Level-0 pixels only ever land inside a draw op's bbox, so these bound
            # the region the pyramid has to visit. See _pyramid_level_blocks.
            content_bboxes = [op["bbox"] for op in draw_ops]
            payloads = _macro_tile_payloads(draw_ops, width=width, height=height)
            # Level 0 alone is gated on the object count: that is the stage whose
            # cost scales with draw ops. The pyramid decides for itself, from the
            # number of blocks it will actually visit.
            _rasterize_level0(
                arrays,
                payloads,
                use_pool=len(objects) >= RASTER_POOL_MIN_OBJECTS,
                on_progress=on_progress,
                cancel_check=cancel_check,
            )
    finally:
        # Close level-0 handles before the pyramid: it re-opens the staged store
        # by path, so the parent must not hold the arrays open.
        _close_overlay_arrays(arrays)
    _build_pyramid(
        stage_root,
        width=width,
        height=height,
        content_bboxes=content_bboxes,
        on_progress=on_progress,
        cancel_check=cancel_check,
    )

    if cancel_check is not None:
        cancel_check()

    if not _is_valid_label_store(stage_root, width=width, height=height):
        raise OverlayStoreError(f"Generated overlay store is invalid at {stage_root}")

    applied_revision = desired_revision if desired_revision is not None else state.desired_revision
    return _publish_snapshot(
        segmentation,
        state,
        stage_root,
        version=next_bundle_version,
        revision=int(applied_revision),
        assignments=assignments,
        geometry_revision=geometry_revision,
    )


@_serialized_build
@timed_stage("overlay_partial")
def apply_partial_overlay_update(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox | None,
    desired_revision: int,
    source_model: str | None = None,
    on_progress=None,
    cancel_check=None,
    dirty_chunks: set[tuple[int, int]] | None = None,
) -> SegmentationOverlayState:
    geometry_revision = ImageSegmentation.objects.values_list("geometry_revision", flat=True).get(
        pk=segmentation.pk
    )
    normalized_source_model = normalize_overlay_source_model(source_model)
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    chunk_coords = (
        dirty_chunks
        if dirty_chunks is not None
        else (dirty_bbox_to_chunk_coords(dirty_bbox) if dirty_bbox is not None else set())
    )
    width, height = segmentation_dimensions(segmentation)
    halo = OVERLAY_BORDER_WIDTH
    version = int(state.bundle_version) + 1
    stage_root = get_overlay_stage_bundle_path(
        str(segmentation.id), version, normalized_source_model
    )
    parent_levels = []
    coords = chunk_coords
    for _ in range(1, len(_level_shapes(width, height))):
        coords = {(x // 2, y // 2) for x, y in coords}
        parent_levels.append(coords)
    replaced_chunks = {
        f"{key}/{level}/{y}.{x}"
        for level, coordinates in enumerate([chunk_coords, *parent_levels])
        for key in OVERLAY_ARRAY_KEYS
        for x, y in coordinates
    }
    # Never expose half-written levels or mutate bytes under an immutable URL.
    clone_store(get_overlay_active_bundle_path(state), stage_root, replaced_chunks=replaced_chunks)
    arrays = _open_label_arrays(state, store_root=stage_root)
    queryset = labels_lut.bundle_queryset(segmentation, normalized_source_model)
    label_map = labels_lut.existing_label_map(state, current_only=True)
    _, previous_snapshot = read_snapshot(
        str(segmentation.id),
        normalized_source_model,
        f"{state.bundle_version}-{state.applied_revision}",
        include_labels=False,
        lease=False,
    )
    # A partial repaint cannot certify unchanged chunks against a different
    # canonical geometry generation. Retain certification for LUT-only work;
    # otherwise analysis rasterizes vectors until a full build certifies them.
    if previous_snapshot.get("geometry_revision") != geometry_revision:
        geometry_revision = None
    # Never recycle a deleted label into unchanged pixels. The small snapshot
    # watermark survives row reclamation; full builds can safely renumber.
    high_watermark = previous_snapshot.get("max_label")
    if high_watermark is None:  # Earlier format-7 snapshots.
        high_watermark = max(labels_lut.existing_label_map(state).values(), default=0)
    next_label = int(high_watermark) + 1
    first_new_label = next_label
    total = len(chunk_coords) + sum(map(len, parent_levels)) * len(OVERLAY_ARRAY_KEYS)
    completed = 0
    try:
        # Each raster allocation is at most one chunk plus its border halo.
        # Disjoint changes stay disjoint, including when many edits coalesce.
        for chunk_x, chunk_y in sorted(chunk_coords):
            if cancel_check is not None:
                cancel_check()
            x0, y0 = chunk_x * OVERLAY_CHUNK_SIZE, chunk_y * OVERLAY_CHUNK_SIZE
            x1, y1 = min(width, x0 + OVERLAY_CHUNK_SIZE), min(height, y0 + OVERLAY_CHUNK_SIZE)
            region = (
                max(0, x0 - halo),
                max(0, y0 - halo),
                min(width, x1 + halo),
                min(height, y1 + halo),
            )
            objects = list(
                queryset.filter(bbox_intersects_filter(make_bbox(*region))).only(
                    *_object_only_fields()
                )
            )
            for obj in objects:
                if obj.id not in label_map:
                    label_map[obj.id] = next_label
                    next_label += 1
            draw_ops = _build_draw_ops(objects, label_map=label_map)
            result = render_module.rasterize_tile_worker(
                {
                    "region": region,
                    "interior": (x0, y0, x1, y1),
                    "draw_ops": draw_ops,
                    "border_width": OVERLAY_BORDER_WIDTH,
                }
            )
            _write_tile_result(arrays, result)
            completed += 1
            if on_progress is not None:
                on_progress("partial", completed, total)
        for level_idx, parent_coords in enumerate(parent_levels, start=1):
            for array_key in OVERLAY_ARRAY_KEYS:
                for chunk_x, chunk_y in sorted(parent_coords):
                    if cancel_check is not None:
                        cancel_check()
                    _retry_on_windows_lock(
                        lambda array_key=array_key, level_idx=level_idx, chunk_x=chunk_x, chunk_y=chunk_y: (
                            render_module.write_parent_chunk(
                                arrays[array_key][level_idx - 1],
                                arrays[array_key][level_idx],
                                chunk_x=chunk_x,
                                chunk_y=chunk_y,
                                kind=array_key,
                            )
                        )
                    )
                    completed += 1
                    if on_progress is not None:
                        on_progress("partial", completed, total)
    finally:
        _close_overlay_arrays(arrays)
    if cancel_check is not None:
        cancel_check()
    return _publish_snapshot(
        segmentation,
        state,
        stage_root,
        version=version,
        revision=int(desired_revision),
        assignments=[(label, object_id) for object_id, label in label_map.items()],
        max_label=next_label - 1,
        geometry_revision=geometry_revision,
        new_assignments=[
            (label, object_id) for object_id, label in label_map.items() if label >= first_new_label
        ],
    )


def overlay_rebuild_policy(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox | None,
    force_full: bool = False,
    source_model: str | None = None,
    allow_sync_partial: bool = True,
) -> str:
    """Choose how an edit reaches the overlay: now, or on the rebuild queue.

    ``allow_sync_partial=False`` takes the synchronous branch off the table for
    this caller. It is not a performance hint -- it is what makes an answer cost
    a database write instead of a disk round-trip, and callers that pass it are
    saying "this edit is one of a stream, do not make the person wait for the
    picture". The edit is still registered at the same revision and still
    rebuilt; only *when* the pixels are painted changes. See
    :class:`~quantem.segmentation.api_views.segments.labels` for the callers.
    """
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
    chunk_count = len(dirty_bbox_to_chunk_coords(dirty_bbox))
    if (
        allow_sync_partial
        # A tiny changed area still clones the *entire* pyramid and rewrites
        # its identity map. Bound those costs before entering a request-side
        # transaction. Large/dense bundles use the coalescing background queue.
        and sum(
            math.ceil(h / OVERLAY_CHUNK_SIZE) * math.ceil(w / OVERLAY_CHUNK_SIZE)
            for h, w in _level_shapes(width, height)
        )
        * len(OVERLAY_ARRAY_KEYS)
        <= 512
        and not SegmentationOverlayLabel.objects.filter(overlay_state=state).values_list(
            "pk", flat=True
        )[5000:5001]
        and state.status != SegmentationOverlayState.STATUS_BUILDING
        and not state.pending_full_rebuild
        and state.applied_revision == state.desired_revision
        and chunk_count <= SYNC_PARTIAL_MAX_LEVEL0_CHUNKS
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
    confirmed = (
        state
        if not state.candidate_source_model
        else get_or_create_overlay_state(state.segmentation, None)
    )
    return {
        "desired_revision": int(state.desired_revision),
        "applied_revision": int(state.applied_revision),
        "lut_revision": int(state.lut_revision),
        "bundle_version": int(state.bundle_version),
        "sync_applied": bool(sync_applied),
        "rebuild_mode": rebuild_mode,
        "source_model": state.candidate_source_model or None,
        "confirmed_display_desired_revision": int(confirmed.desired_revision),
        "confirmed_display_lut_revision": int(confirmed.lut_revision),
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


def register_label_state_mutation(segmentation, objects, *, source_model=None):
    """Recolour answers immediately; repaint only where object ownership can change.

    One integer raster can store only one winner at an overlapping pixel. Label
    state changes paint priority, so an overlapping answer needs a bounded
    asynchronous repaint in addition to its immediate palette update.
    """
    from .dirty import merge_dirty_bboxes

    response = register_state_mutation(segmentation, source_model=source_model)
    contested = []
    sources = set()
    live = labels_lut.bundle_queryset(segmentation, None)
    for obj in objects:
        geometry = obj.geometry
        if geometry is None or geometry.is_empty:
            continue
        neighbors = live.exclude(pk=obj.pk).filter(bbox_intersects_filter(geometry.envelope))
        if any(
            geometry.intersection(other.geometry).area > 0
            for other in neighbors.only("geometry_wkb")
        ):
            contested.append(geometry)
            sources.add(obj.source_model)
    if contested:
        response = register_overlay_mutation_all_bundles(
            segmentation,
            dirty_bbox=merge_dirty_bboxes(segmentation, contested),
            source_model=source_model,
            source_models=sources,
            allow_sync_partial=False,
        )
    return response


def _existing_bundle_source_models(segmentation: ImageSegmentation) -> list[str]:
    source_models = list(
        SegmentationOverlayState.objects.filter(segmentation=segmentation)
        .values_list("candidate_source_model", flat=True)
        .distinct()
    )
    if "" not in source_models:
        source_models.append("")
    return source_models


def _bundles_containing_source_model(
    segmentation: ImageSegmentation,
    normalized_source_model: str,
) -> list[str]:
    """Every bundle whose membership rule can admit an object from this source.

    A model's own outline lives in the confirmed display and in that one model's
    bundle. A **hand-drawn** outline lives in the confirmed display *and in
    every named bundle*: ``overlay_bundle_source_filter`` keeps manual objects
    in each model's raster because the candidate layer reads that raster and is
    the only layer that draws an unconfirmed outline. Dirtying only the
    source-less bundle for a manual edit would therefore leave the drawn or
    reshaped outline stale in exactly the layer that has to show it -- the
    stale-pixels half of the same R13 failure the membership rule fixes.

    An unrecorded source is treated the same way, and that is not caution for
    its own sake: the drawing endpoint registers a freshly created candidate
    without naming a source model at all, and the object it just created is
    hand-drawn. Callers that do know a model name never reach this branch.

    The confirmed display comes first in the returned order so a synchronous
    partial paints the raster the right pane reads before the model rasters,
    and the named bundles are sorted so the fan-out is deterministic.
    """
    if normalized_source_model and normalized_source_model != SOURCE_MODEL_MANUAL:
        return ["", normalized_source_model]
    return [
        "",
        *sorted(name for name in _existing_bundle_source_models(segmentation) if name),
    ]


@transaction.atomic
def _register_overlay_mutation_one(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox | None,
    force_full_rebuild: bool = False,
    source_model: str | None = None,
    allow_sync_partial: bool = True,
) -> dict[str, Any]:
    normalized_source_model = normalize_overlay_source_model(source_model)
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    # Serialize revision allocation explicitly. A no-op UPDATE also acquires a
    # PostgreSQL row lock, but obscures the concurrency contract and generates
    # needless WAL for every edit.
    state = SegmentationOverlayState.objects.select_for_update().get(pk=state.pk)
    desired_revision = int(state.desired_revision) + 1
    rebuild_mode = overlay_rebuild_policy(
        segmentation,
        dirty_bbox=dirty_bbox,
        force_full=force_full_rebuild,
        source_model=normalized_source_model,
        allow_sync_partial=allow_sync_partial,
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
                _lock_timeout=0,
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
    # A full rebuild that is already owed is not discharged by the next
    # incremental edit, so the flag is carried forward rather than recomputed
    # from this edit's mode. `overlay_rebuild_policy` returns "async_partial"
    # *because* the flag is set, so recomputing would clear it, and the
    # follow-up job would repaint only this edit's bbox before
    # `_finalize_overlay_rebuild_state` marked the bundle READY -- over
    # whatever geometry the full rebuild existed to rasterise (a completed
    # extraction, or the rebuild button). Analysis's settled-raster reuse
    # trusts a READY bundle, so that geometry would go missing from exported
    # measurements with nothing to say so.
    pending_full_rebuild = (
        rebuild_mode == "async_full" or force_full_rebuild or bool(state.pending_full_rebuild)
    )
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
    allow_sync_partial: bool = True,
) -> dict[str, Any]:
    """Register a geometry edit for the confirmed display and its source.

    A model-produced outline lives in the source-less confirmed-display raster
    and its own model raster. A hand-drawn one lives in the confirmed display
    and in every model raster, because that is where the candidate layer reads
    an unconfirmed outline from. State controls visibility, so a later
    confirmation does not move pixels between bundles.

    ``allow_sync_partial=False`` defers the raster to the rebuild queue; see
    :func:`overlay_rebuild_policy`.
    """
    normalized_source_model = normalize_overlay_source_model(source_model)
    responses: dict[str, dict[str, Any]] = {}
    for candidate_source_model in _bundles_containing_source_model(
        segmentation, normalized_source_model
    ):
        responses[candidate_source_model] = _register_overlay_mutation_one(
            segmentation,
            dirty_bbox=dirty_bbox,
            force_full_rebuild=force_full_rebuild,
            source_model=candidate_source_model,
            allow_sync_partial=allow_sync_partial,
        )
    # The named model's own counters when the caller named one; otherwise the
    # confirmed display's, which is the bundle every caller of this function has
    # in common. Never "whichever ran last": for a manual edit that is an
    # arbitrary model bundle whose revisions the caller cannot interpret.
    if normalized_source_model and normalized_source_model != SOURCE_MODEL_MANUAL:
        return responses[normalized_source_model]
    return responses[""]


def register_overlay_mutation_all_bundles(
    segmentation: ImageSegmentation,
    *,
    dirty_bbox: DirtyBBox | None,
    force_full_rebuild: bool = False,
    source_model: str | None = None,
    source_models: Iterable[str | None] | None = None,
    allow_sync_partial: bool = True,
) -> dict[str, Any]:
    """Register a geometry edit on every bundle that can actually contain it.

    When ``source_model`` is a model name this is the confirmed display plus
    that one model bundle. When it is ``manual`` -- or absent, which the drawing
    endpoint leaves it -- it is every bundle, because a hand-drawn object is a
    member of every one of them (see
    :func:`_bundles_containing_source_model`). ``source_models`` handles the
    rarer merge that changes an existing object owned by a second model.

    ``allow_sync_partial=False`` defers the raster to the rebuild queue; see
    :func:`overlay_rebuild_policy`. It matters most here: this function writes
    *every* bundle, so a synchronous answer paid the raster cost once per
    bundle, and an image segmented by two models therefore cost twice as much
    per keystroke as one segmented by one.
    """
    normalized_source_model = normalize_overlay_source_model(source_model)
    if source_models is not None:
        # The union over the sources involved, not the sources themselves: a
        # merge that touches a hand-drawn object has to dirty every model
        # bundle, because that object is in all of them.
        named_bundles: set[str] = set()
        for candidate in source_models:
            named_bundles.update(
                name
                for name in _bundles_containing_source_model(
                    segmentation, normalize_overlay_source_model(candidate)
                )
                if name
            )
        target_source_models = ["", *sorted(named_bundles)]
    else:
        target_source_models = _bundles_containing_source_model(
            segmentation, normalized_source_model
        )
    responses: dict[str, dict[str, Any]] = {}
    for candidate_source_model in target_source_models:
        responses[candidate_source_model] = _register_overlay_mutation_one(
            segmentation,
            dirty_bbox=dirty_bbox,
            force_full_rebuild=force_full_rebuild,
            source_model=candidate_source_model,
            allow_sync_partial=allow_sync_partial,
        )
    # ``manual`` is never the reported bundle even when a legacy state row of
    # that name exists: it is a source, not a display, and its counters mean
    # nothing to the caller. A hand-drawn edit reports the confirmed display.
    named_response = (
        responses.get(normalized_source_model)
        if normalized_source_model and normalized_source_model != SOURCE_MODEL_MANUAL
        else None
    )
    response = named_response or responses.get("") or next(iter(responses.values()))
    confirmed_response = responses.get("")
    if confirmed_response is not None:
        # Named model and confirmed-display bundles have independent counters.
        # Consumers waiting for the right pane must compare against this one,
        # not against whichever named response was selected above.
        response["confirmed_display_desired_revision"] = confirmed_response["desired_revision"]
    return response


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


def queue_overlay_followup(payload: dict) -> None:
    """Retry owed work after a successful job has released its queue claim."""
    try:
        state = (
            SegmentationOverlayState.objects.select_related("segmentation")
            .filter(
                segmentation_id=payload.get("segmentation_id"),
                candidate_source_model=normalize_overlay_source_model(payload.get("source_model")),
            )
            .first()
        )
        if state is None or state.status == SegmentationOverlayState.STATUS_FAILED:
            return
        if state.pending_full_rebuild or state.desired_revision > state.applied_revision:
            queue_overlay_rebuild(
                state.segmentation,
                source_model=state.candidate_source_model,
                mode="full" if state.pending_full_rebuild else "partial",
            )
    except Exception:
        # Publication already succeeded. A queue/database outage here must not
        # retroactively fail that job; manifest polling can recover the tail.
        logger.warning("Could not enqueue the remaining overlay edits", exc_info=True)


def run_overlay_rebuild_job(
    segmentation: ImageSegmentation,
    *,
    mode: str,
    source_model: str | None = None,
    on_progress=None,
    cancel_check=None,
) -> SegmentationOverlayState:
    normalized_source_model = normalize_overlay_source_model(source_model)
    state = get_or_create_overlay_state(segmentation, normalized_source_model)
    _set_overlay_state(
        state,
        status_value=SegmentationOverlayState.STATUS_BUILDING,
        last_error="",
    )
    try:
        if (
            mode == "full"
            or state.pending_full_rebuild
            or segmentation.segmentation_type.measurement_mode == "global"
        ):
            return rebuild_overlay_full(
                segmentation,
                source_model=normalized_source_model,
                desired_revision=int(state.desired_revision),
                on_progress=on_progress,
                cancel_check=cancel_check,
            )

        dirty_chunks = dirty_runs_to_chunk_coords(segmentation, list(state.dirty_chunk_runs or []))
        if not dirty_chunks:
            # A revision without recorded regions cannot be acknowledged under
            # the previous snapshot's token. Reconstruct it conservatively.
            return rebuild_overlay_full(
                segmentation,
                source_model=normalized_source_model,
                desired_revision=int(state.desired_revision),
                on_progress=on_progress,
                cancel_check=cancel_check,
            )

        width, height = segmentation_dimensions(segmentation)
        total_chunks = math.ceil(width / OVERLAY_CHUNK_SIZE) * math.ceil(
            height / OVERLAY_CHUNK_SIZE
        )
        if (
            len(dirty_chunks) > ASYNC_PARTIAL_MAX_LEVEL0_CHUNKS
            or len(dirty_chunks) / total_chunks > ASYNC_PARTIAL_MAX_IMAGE_COVERAGE
        ):
            return rebuild_overlay_full(
                segmentation,
                source_model=normalized_source_model,
                desired_revision=int(state.desired_revision),
                on_progress=on_progress,
                cancel_check=cancel_check,
            )

        return apply_partial_overlay_update(
            segmentation,
            dirty_bbox=None,
            dirty_chunks=dirty_chunks,
            desired_revision=int(state.desired_revision),
            source_model=normalized_source_model,
            on_progress=on_progress,
            cancel_check=cancel_check,
        )
    except JobCancelledError:
        # Cancellation is not a corrupt overlay: the old active bundle stays on
        # disk and the canonical geometry is in the database, so all that is
        # lost is the repaint. But it does have to be recorded as a *stop*.
        # Written as DIRTY, as this arm first was, the manifest poll read
        # "pending work, no live job" 1.5 s later and enqueued the identical
        # rebuild -- Cancel undid itself, and a long build on a large image
        # could not be stopped while a labelling screen was open. FAILED plus a
        # reason is the one state both ends already treat as terminal, and the
        # user's next edit clears it (see `_register_overlay_mutation_one`), so
        # the pause self-heals exactly the way a build failure does.
        _set_overlay_state(
            state,
            status_value=SegmentationOverlayState.STATUS_FAILED,
            # Raise this flag here, never clear it. `state` was read before the
            # build began, so writing `False` from it would wipe a full rebuild
            # requested *while* the build ran -- and the `async_full` branch of
            # `_register_overlay_mutation_one` records that request in this flag
            # alone, with no dirty run to fall back on. The follow-up would then
            # silently downgrade to a partial and leave the bundle claiming
            # READY over geometry that was never rasterised, which Analysis's
            # settled-raster reuse trusts and under-counts. `_set_overlay_state`
            # leaves a field alone when it is passed None, so None here means
            # "whatever the row says now, which is newer than what I hold".
            pending_full_rebuild=(True if (mode == "full" or state.pending_full_rebuild) else None),
            last_error=OVERLAY_CANCELLED_MESSAGE,
        )
        _write_debug_manifest(segmentation, state)
        raise
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
            # Not `str(exc)`. This string is served on the manifest and rendered
            # verbatim on the labeling and viewer screens, and `str(OSError)`
            # `repr()`s the filename -- so the one actionable thing in it, the
            # path, arrived with every backslash doubled and could not be pasted
            # into Explorer. See `failure_text`.
            last_error=describe_failure(exc),
        )
        _write_debug_manifest(segmentation, state)
        raise

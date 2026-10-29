"""Exact, bounded-memory post-processing for very large probability maps.

This module is the out-of-core implementation of the ordinary pipeline in
``quantem.inference.postprocess``::

    threshold -> binary closing -> fill holes -> label -> minimum area

It deliberately changes *where* arrays live, not what the operations mean.
Stripes span the complete image width, which makes component ordering the same
row-major ordering as ``skimage.measure.label``.  Closing reads a two-radius
halo (dilation followed by erosion has twice the footprint reach), hole filling
labels 4-connected background and retains only background connected to the
image border, and foreground uses 8-connectivity.

Only the labels of surviving objects are compacted.  Label integers are an
internal transport detail and are not persisted; survivor order remains the
same.  Compaction keeps ``regionprops`` proportional to the number of objects
that will actually be published rather than to every discarded speck.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import closing, disk

from quantem.inference.resample import PROB_LEVELS, threshold_level

# Peak working arrays used by scipy/skimage are several times the input stripe.
# Keeping the source stripe near 64 MiB leaves comfortable headroom while still
# giving the morphology implementation enough rows to run efficiently.
DEFAULT_STRIPE_TARGET_BYTES = 64 * 1024 * 1024

# The ordinary path creates at least a bool mask plus an integer label image.
# Route before those resident allocations become multi-gigabyte.  Zarr-backed
# maps always take this path regardless of size (``np.asarray`` would defeat
# their storage contract).
DEFAULT_OUT_OF_CORE_MIN_PIXELS = 256 * 1024 * 1024

_LABEL_DTYPES = (np.dtype(np.uint32), np.dtype(np.uint64))
_PARENT_BLOCK = 4 * 1024 * 1024
_STALE_WORKSPACE_SECONDS = 24 * 60 * 60


class ComponentLabelOverflow(RuntimeError):
    """A provisional label store needs the next wider integer dtype."""


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) can terminate a process on Windows.
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            # Only ERROR_INVALID_PARAMETER establishes that the PID is gone.
            return ctypes.get_last_error() != 87
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _cleanup_stale_workspaces(workspace_root: Path) -> None:
    """Remove abandoned stores without touching live concurrent extraction."""
    now = time.time()
    for candidate in workspace_root.glob("postprocess-*"):
        if not candidate.is_dir():
            continue
        try:
            age = now - candidate.stat().st_mtime
        except OSError:
            continue
        if age < _STALE_WORKSPACE_SECONDS:
            continue
        pid = 0
        try:
            payload = json.loads((candidate / "owner.json").read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        if _process_is_alive(pid):
            continue
        shutil.rmtree(candidate, ignore_errors=True)


def is_plain_ndarray(array: Any) -> bool:
    """Whether ``array`` is already a resident NumPy array or memmap."""
    return isinstance(array, np.ndarray)


def should_use_out_of_core(
    array: Any,
    *,
    min_pixels: int = DEFAULT_OUT_OF_CORE_MIN_PIXELS,
) -> bool:
    """Return whether post-processing ``array`` should use disk-backed state."""
    shape = getattr(array, "shape", ())
    if len(shape) != 2:
        return False
    if not is_plain_ndarray(array):
        return True
    return int(shape[0]) * int(shape[1]) >= int(min_pixels)


def _stripe_rows(width: int, target_bytes: int) -> int:
    return max(1, int(target_bytes) // max(int(width), 1))


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    try:
        array.flush()
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _create_memmap(
    path: Path,
    *,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    fill: int | bool | None = None,
) -> np.memmap:
    array = np.memmap(str(path), dtype=dtype, mode="w+", shape=shape)
    if fill is not None:
        array[...] = fill
        array.flush()
    return array


def _threshold_block(source: Any, rows: slice, level: int) -> np.ndarray:
    block = np.asarray(source[rows, :], dtype=np.uint8)
    if level <= 0:
        return np.ones(block.shape, dtype=bool)
    if level > PROB_LEVELS:
        return np.zeros(block.shape, dtype=bool)
    return block >= np.uint8(level)


def _write_closed_mask(
    source: Any,
    mask: np.memmap,
    *,
    threshold: float,
    close_radius: int,
    stripe_rows: int,
) -> None:
    height, _width = mask.shape
    level = threshold_level(threshold)
    radius = max(int(close_radius), 0)
    halo = radius * 2
    footprint = disk(radius) if radius else None

    for y0 in range(0, height, stripe_rows):
        y1 = min(height, y0 + stripe_rows)
        read0 = max(0, y0 - halo)
        read1 = min(height, y1 + halo)
        block = _threshold_block(source, slice(read0, read1), level)
        if footprint is not None and block.any():
            block = closing(block, footprint)
        core0 = y0 - read0
        mask[y0:y1] = block[core0 : core0 + (y1 - y0)]
    mask.flush()


def _label_stripes(
    mask: np.memmap,
    labels: np.memmap,
    *,
    foreground: bool,
    connectivity: int,
    stripe_rows: int,
) -> int:
    """Write provisional labels and return their count.

    Each stripe spans the complete width, so scipy's row-major local labels and
    top-to-bottom stripe order together preserve global row-major first
    occurrence.  Components crossing stripe seams are reconciled separately.
    """
    structure = ndi.generate_binary_structure(2, connectivity)
    total = 0
    limit = int(np.iinfo(labels.dtype).max)
    height = int(mask.shape[0])
    for y0 in range(0, height, stripe_rows):
        y1 = min(height, y0 + stripe_rows)
        binary = np.asarray(mask[y0:y1], dtype=bool)
        if not foreground:
            binary = ~binary
        local, count = ndi.label(binary, structure=structure)
        count = int(count)
        if total + count > limit:
            raise ComponentLabelOverflow(
                f"{total + count} provisional components exceed {labels.dtype}"
            )
        local = local.astype(labels.dtype, copy=False)
        if total and count:
            nonzero = local != 0
            local[nonzero] += total
        labels[y0:y1] = local
        total += count
    labels.flush()
    return total


def _find(parent: np.memmap, value: int) -> int:
    current = int(value)
    while True:
        next_value = int(parent[current])
        if next_value == current:
            return current
        grandparent = int(parent[next_value])
        parent[current] = grandparent
        current = next_value


def _union(parent: np.memmap, left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left == root_right:
        return
    # The smaller provisional id has the earlier row-major first pixel. Keeping
    # it as the root preserves the reference component order.
    if root_left < root_right:
        parent[root_right] = root_left
    else:
        parent[root_left] = root_right


def _union_pair_arrays(parent: np.memmap, left: np.ndarray, right: np.ndarray) -> None:
    valid = (left != 0) & (right != 0) & (left != right)
    if not np.any(valid):
        return
    pairs = np.column_stack((left[valid], right[valid]))
    pairs.sort(axis=1)
    for first, second in np.unique(pairs, axis=0):
        _union(parent, int(first), int(second))


def _reconcile_stripe_seams(
    labels: np.memmap,
    parent: np.memmap,
    *,
    connectivity: int,
    stripe_rows: int,
) -> None:
    height, width = (int(labels.shape[0]), int(labels.shape[1]))
    for boundary in range(stripe_rows, height, stripe_rows):
        upper = np.asarray(labels[boundary - 1, :]).copy()
        lower = np.asarray(labels[boundary, :]).copy()
        _union_pair_arrays(parent, upper, lower)
        if connectivity >= 2 and width > 1:
            _union_pair_arrays(parent, upper[:-1], lower[1:])
            _union_pair_arrays(parent, upper[1:], lower[:-1])
    parent.flush()


def _compress_parents(parent: np.memmap) -> None:
    size = int(parent.shape[0])
    for start in range(0, size, _PARENT_BLOCK):
        stop = min(size, start + _PARENT_BLOCK)
        roots = np.asarray(parent[start:stop]).copy()
        while True:
            next_roots = np.asarray(parent[roots])
            if np.array_equal(next_roots, roots):
                break
            roots = next_roots
        parent[start:stop] = roots
    parent.flush()


def _build_parent_store(
    path: Path,
    labels: np.memmap,
    *,
    component_count: int,
    connectivity: int,
    stripe_rows: int,
) -> np.memmap:
    parent = _create_memmap(
        path,
        dtype=labels.dtype,
        shape=(component_count + 1,),
    )
    for start in range(0, component_count + 1, _PARENT_BLOCK):
        stop = min(component_count + 1, start + _PARENT_BLOCK)
        parent[start:stop] = np.arange(start, stop, dtype=labels.dtype)
    _reconcile_stripe_seams(
        labels,
        parent,
        connectivity=connectivity,
        stripe_rows=stripe_rows,
    )
    _compress_parents(parent)
    return parent


def _mark_exterior_background(
    labels: np.memmap,
    parent: np.memmap,
    exterior: np.memmap,
    *,
    stripe_rows: int,
) -> None:
    height = int(labels.shape[0])
    border_ids = [
        np.asarray(labels[0, :]).copy(),
        np.asarray(labels[height - 1, :]).copy(),
    ]
    for y0 in range(0, height, stripe_rows):
        y1 = min(height, y0 + stripe_rows)
        border_ids.append(np.asarray(labels[y0:y1, 0]).copy())
        border_ids.append(np.asarray(labels[y0:y1, -1]).copy())
    ids = np.unique(np.concatenate(border_ids))
    ids = ids[ids != 0]
    if ids.size:
        exterior[np.asarray(parent[ids])] = True
    exterior.flush()


def _fill_holes(
    mask: np.memmap,
    workspace: Path,
    *,
    stripe_rows: int,
    label_dtype: np.dtype,
) -> None:
    labels_path = workspace / "labels.dat"
    parent_path = workspace / "parents.dat"
    exterior_path = workspace / "exterior.dat"
    labels: np.memmap | None = None
    parent: np.memmap | None = None
    exterior: np.memmap | None = None
    try:
        labels = _create_memmap(labels_path, dtype=label_dtype, shape=mask.shape)
        count = _label_stripes(
            mask,
            labels,
            foreground=False,
            connectivity=1,
            stripe_rows=stripe_rows,
        )
        if count == 0:
            return
        parent = _build_parent_store(
            parent_path,
            labels,
            component_count=count,
            connectivity=1,
            stripe_rows=stripe_rows,
        )
        exterior = _create_memmap(
            exterior_path,
            dtype=bool,
            shape=(count + 1,),
            fill=False,
        )
        _mark_exterior_background(
            labels,
            parent,
            exterior,
            stripe_rows=stripe_rows,
        )

        height = int(mask.shape[0])
        for y0 in range(0, height, stripe_rows):
            y1 = min(height, y0 + stripe_rows)
            ids = np.asarray(labels[y0:y1]).copy()
            background = ids != 0
            if np.any(background):
                roots = np.asarray(parent[ids[background]])
                fill = ~np.asarray(exterior[roots], dtype=bool)
                block = np.asarray(mask[y0:y1]).copy()
                positions = np.flatnonzero(background)
                block.reshape(-1)[positions[fill]] = True
                mask[y0:y1] = block
        mask.flush()
    finally:
        _close_memmap(exterior)
        _close_memmap(parent)
        _close_memmap(labels)
        exterior_path.unlink(missing_ok=True)
        parent_path.unlink(missing_ok=True)
        labels_path.unlink(missing_ok=True)


def _component_areas(
    labels: np.memmap,
    parent: np.memmap,
    areas: np.memmap,
    *,
    stripe_rows: int,
) -> None:
    height = int(labels.shape[0])
    for y0 in range(0, height, stripe_rows):
        y1 = min(height, y0 + stripe_rows)
        ids, counts = np.unique(np.asarray(labels[y0:y1]), return_counts=True)
        nonzero = ids != 0
        if not np.any(nonzero):
            continue
        roots = np.asarray(parent[ids[nonzero]])
        np.add.at(areas, roots, counts[nonzero].astype(np.uint64, copy=False))
    areas.flush()


def _survivor_mapping(
    parent: np.memmap,
    areas: np.memmap,
    mapping: np.memmap,
    *,
    min_area: int,
) -> int:
    next_label = 0
    size = int(parent.shape[0])
    floor = max(int(min_area), 0)
    for start in range(1, size, _PARENT_BLOCK):
        stop = min(size, start + _PARENT_BLOCK)
        ids = np.arange(start, stop, dtype=parent.dtype)
        roots = np.asarray(parent[start:stop]) == ids
        if not np.any(roots):
            continue
        root_ids = ids[roots]
        if floor > 1:
            root_ids = root_ids[np.asarray(areas[root_ids]) >= floor]
        if not root_ids.size:
            continue
        first = next_label + 1
        next_label += int(root_ids.size)
        mapping[root_ids] = np.arange(first, next_label + 1, dtype=mapping.dtype)
    mapping.flush()
    return next_label


def _relabel_survivors(
    labels: np.memmap,
    parent: np.memmap,
    mapping: np.memmap,
    *,
    stripe_rows: int,
) -> None:
    height = int(labels.shape[0])
    for y0 in range(0, height, stripe_rows):
        y1 = min(height, y0 + stripe_rows)
        block = np.asarray(labels[y0:y1]).copy()
        nonzero = block != 0
        if np.any(nonzero):
            roots = np.asarray(parent[block[nonzero]])
            block[nonzero] = np.asarray(mapping[roots])
        labels[y0:y1] = block
    labels.flush()


def _label_foreground(
    mask: np.memmap,
    workspace: Path,
    *,
    min_area: int,
    stripe_rows: int,
    label_dtype: np.dtype,
) -> tuple[np.memmap, int]:
    labels_path = workspace / "labels.dat"
    parent_path = workspace / "parents.dat"
    areas_path = workspace / "areas.dat"
    mapping_path = workspace / "mapping.dat"
    labels = _create_memmap(labels_path, dtype=label_dtype, shape=mask.shape)
    parent: np.memmap | None = None
    areas: np.memmap | None = None
    mapping: np.memmap | None = None
    try:
        count = _label_stripes(
            mask,
            labels,
            foreground=True,
            connectivity=2,
            stripe_rows=stripe_rows,
        )
        if count == 0:
            return labels, 0
        parent = _build_parent_store(
            parent_path,
            labels,
            component_count=count,
            connectivity=2,
            stripe_rows=stripe_rows,
        )
        areas = _create_memmap(
            areas_path,
            dtype=np.uint64,
            shape=(count + 1,),
            fill=0,
        )
        _component_areas(
            labels,
            parent,
            areas,
            stripe_rows=stripe_rows,
        )
        mapping = _create_memmap(
            mapping_path,
            dtype=label_dtype,
            shape=(count + 1,),
            fill=0,
        )
        survivor_count = _survivor_mapping(
            parent,
            areas,
            mapping,
            min_area=min_area,
        )
        _relabel_survivors(
            labels,
            parent,
            mapping,
            stripe_rows=stripe_rows,
        )
        return labels, survivor_count
    except Exception:
        _close_memmap(labels)
        raise
    finally:
        _close_memmap(mapping)
        _close_memmap(areas)
        _close_memmap(parent)
        mapping_path.unlink(missing_ok=True)
        areas_path.unlink(missing_ok=True)
        parent_path.unlink(missing_ok=True)


def _run_with_dtype(
    source: Any,
    workspace: Path,
    *,
    threshold: float,
    close_radius: int,
    min_area: int,
    stripe_rows: int,
    label_dtype: np.dtype,
) -> tuple[np.memmap, int]:
    shape = (int(source.shape[0]), int(source.shape[1]))
    mask = _create_memmap(workspace / "mask.dat", dtype=bool, shape=shape)
    try:
        _write_closed_mask(
            source,
            mask,
            threshold=threshold,
            close_radius=close_radius,
            stripe_rows=stripe_rows,
        )
        # ``close_and_fill`` treats radius zero as a complete no-op, including
        # skipping hole filling. Preserve that established (if slightly
        # surprising) contract exactly.
        if int(close_radius) > 0 and np.any(mask):
            _fill_holes(
                mask,
                workspace,
                stripe_rows=stripe_rows,
                label_dtype=label_dtype,
            )
        return _label_foreground(
            mask,
            workspace,
            min_area=min_area,
            stripe_rows=stripe_rows,
            label_dtype=label_dtype,
        )
    finally:
        _close_memmap(mask)
        (workspace / "mask.dat").unlink(missing_ok=True)


@contextmanager
def postprocess_probability_source(
    source: Any,
    *,
    threshold: float,
    close_radius: int,
    min_area: int,
    workspace_root: Path,
    stripe_target_bytes: int = DEFAULT_STRIPE_TARGET_BYTES,
) -> Iterator[np.memmap]:
    """Yield exact component labels while keeping image-sized state on disk.

    ``source`` must be a two-dimensional uint8 array-like object supporting
    NumPy slice reads (NumPy arrays, memmaps and Zarr arrays all do).  The
    workspace is always removed on normal completion or an exception.  A
    uint32 label store is attempted first; only a pathologically fragmented
    mask that exceeds it is repeated with uint64 labels.
    """
    shape = getattr(source, "shape", ())
    dtype = np.dtype(getattr(source, "dtype", np.dtype("O")))
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D probability map, got shape {shape}")
    if dtype != np.dtype(np.uint8):
        raise TypeError(f"Stored probability map must be uint8, got {dtype}")

    workspace_root.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_workspaces(workspace_root)
    workspace = workspace_root / f"postprocess-{uuid4().hex}"
    workspace.mkdir(parents=False, exist_ok=False)
    (workspace / "owner.json").write_text(
        json.dumps({"pid": os.getpid(), "created_at": time.time()}),
        encoding="utf-8",
    )
    labels: np.memmap | None = None
    stripe_rows = _stripe_rows(int(shape[1]), int(stripe_target_bytes))
    try:
        last_overflow: ComponentLabelOverflow | None = None
        for label_dtype in _LABEL_DTYPES:
            try:
                labels, _count = _run_with_dtype(
                    source,
                    workspace,
                    threshold=threshold,
                    close_radius=close_radius,
                    min_area=min_area,
                    stripe_rows=stripe_rows,
                    label_dtype=label_dtype,
                )
                last_overflow = None
                break
            except ComponentLabelOverflow as exc:
                last_overflow = exc
                for name in ("labels.dat", "parents.dat", "areas.dat", "mapping.dat"):
                    (workspace / name).unlink(missing_ok=True)
        if labels is None:
            assert last_overflow is not None
            raise last_overflow
        yield labels
    finally:
        _close_memmap(labels)
        shutil.rmtree(workspace, ignore_errors=True)


__all__ = [
    "DEFAULT_OUT_OF_CORE_MIN_PIXELS",
    "DEFAULT_STRIPE_TARGET_BYTES",
    "postprocess_probability_source",
    "should_use_out_of_core",
]

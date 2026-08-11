"""ID-map rasterisation and non-averaging multiscale downsampling.

These label maps are read twice over: the viewer draws them, and
:func:`quantem.analysis.loaders.segmentation_mask` and the compartment masks
beside it *count* them. So they follow the app's one pixel convention -- a pixel
belongs to an object when its centre is inside the outline, defined and
implemented in :mod:`quantem.seg_core.rasterize` -- and the number of pixels a
compartment covers here is the area of the polygons that made it.

They used to be painted with :func:`cv2.fillPoly` on ``rint``-ed coordinates,
which rounds each vertex to a pixel centre and then paints *both* boundaries of
every span. An object spanning *s* pixels covered *s+1*, so three 20 px squares
reported ``areas_px.mito = 3 x 441`` instead of ``3 x 400``. That does not
cancel in ``area_fraction_*``: the numerator is small objects, which inflate
proportionally more than the whole-tissue denominator, so the fraction came out
~9% high. ``enrichment``, being a ratio of two fractions, largely did cancel --
which is why this survived so long.

Rendering is split so the CPU-heavy parts can run in a process pool with cheap
pickling:

* :func:`geometry_to_rings` (parent, needs shapely) converts an object geometry
  to ring coordinate arrays.
* :func:`rasterize_region` / :func:`rasterize_tile_worker` (pure numpy) paint
  dense labels and bake the border mask from label adjacency. Workers receive
  only numpy arrays, never ORM objects.
* :func:`mode_downsample_2x2` / :func:`max_downsample_2x2` /
  :func:`write_parent_chunk` build the pyramid without ever averaging label ids.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import zarr
from shapely.geometry.base import BaseGeometry

from quantem.seg_core.rasterize import paint_ring
from quantem.segmentation.geometry import extract_polygons

from .constants import (
    LABELS_ARRAY_KEY,
    OVERLAY_BORDER_WIDTH,
    OVERLAY_CHUNK_SIZE,
)

# A draw op is a plain dict (picklable): {label, priority, area, rings}
# where rings = [(exterior_f32[N,2], [hole_f32[M,2], ...]), ...] in absolute
# image pixel coordinates.


def geometry_to_rings(geometry: BaseGeometry | None) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Ring coordinate arrays for one geometry, at their true positions.

    float32, not rounded-to-int32: the fill decides on half-pixel boundaries,
    and rounding a vertex to the nearest pixel centre first throws away the
    information it needs. float32 keeps every coordinate and every half-pixel
    exactly up to 8.4 million, well past any image dimension, at the pickling
    cost these arrays were made int32 for.
    """
    rings: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for polygon in extract_polygons(geometry):
        try:
            exterior_coords = list(polygon.exterior.coords)
            interior_rings = [list(ring.coords) for ring in polygon.interiors]
        except Exception:
            continue
        if not exterior_coords:
            continue
        exterior = np.asarray(exterior_coords, dtype=np.float32)
        if exterior.ndim != 2 or exterior.shape[0] < 3:
            continue
        holes = [np.asarray(ring, dtype=np.float32) for ring in interior_rings if len(ring) >= 3]
        rings.append((exterior[:, :2], [hole[:, :2] for hole in holes]))
    return rings


def _compute_border(labels: np.ndarray, *, width: int) -> np.ndarray:
    """Foreground-side 1..width px border at every label boundary."""
    diff = np.zeros(labels.shape, dtype=bool)
    if labels.shape[1] > 1:
        horizontal = labels[:, 1:] != labels[:, :-1]
        diff[:, 1:] |= horizontal
        diff[:, :-1] |= horizontal
    if labels.shape[0] > 1:
        vertical = labels[1:, :] != labels[:-1, :]
        diff[1:, :] |= vertical
        diff[:-1, :] |= vertical
    foreground = labels != 0
    border = diff & foreground
    if width > 1:
        kernel = np.ones((width, width), dtype=np.uint8)
        dilated = cv2.dilate(border.astype(np.uint8), kernel)
        border = (dilated > 0) & foreground
    return border.astype(np.uint8)


def rasterize_region(
    draw_ops: list[dict[str, Any]],
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    border_width: int = OVERLAY_BORDER_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise dense labels + border mask for the region ``[x0,x1) x [y0,y1)``.

    ``draw_ops`` are sorted by ascending priority so higher-priority objects are
    painted last (and win contested pixels); within a tier larger objects are
    painted first so small objects survive.

    A pixel is an object's when its centre is inside the outline
    (:mod:`quantem.seg_core.rasterize`), so the pixels counted here are the area
    of the polygon. Holes are punched back to background exactly as before, and
    each ring costs only its own bounding box, not a pass over the region.
    """
    height = max(0, y1 - y0)
    width = max(0, x1 - x0)
    labels = np.zeros((height, width), dtype=np.int32)
    if height == 0 or width == 0:
        return labels, labels.astype(np.uint8)

    ordered = sorted(draw_ops, key=lambda op: (op["priority"], -op["area"]))
    for op in ordered:
        label = int(op["label"])
        for exterior, holes in op["rings"]:
            paint_ring(labels, exterior, label, x0=x0, y0=y0)
            for hole in holes:
                paint_ring(labels, hole, 0, x0=x0, y0=y0)

    border = _compute_border(labels, width=border_width)
    return labels, border


def rasterize_tile_worker(payload: dict[str, Any]) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Process-pool entry point. Rasterise a halo-expanded region, return the
    interior crop ``(interior_x0, interior_y0, labels_uint32, border_uint8)``.
    """
    region = payload["region"]
    interior = payload["interior"]
    labels, border = rasterize_region(
        payload["draw_ops"],
        x0=region[0],
        y0=region[1],
        x1=region[2],
        y1=region[3],
        border_width=payload["border_width"],
    )
    iy0 = interior[1] - region[1]
    ix0 = interior[0] - region[0]
    iy1 = iy0 + (interior[3] - interior[1])
    ix1 = ix0 + (interior[2] - interior[0])
    labels_crop = np.ascontiguousarray(labels[iy0:iy1, ix0:ix1]).astype(np.uint32)
    border_crop = np.ascontiguousarray(border[iy0:iy1, ix0:ix1])
    return interior[0], interior[1], labels_crop, border_crop


def _pad_to_even(arr: np.ndarray) -> np.ndarray:
    height, width = arr.shape
    pad_y = height % 2
    pad_x = width % 2
    if pad_y or pad_x:
        arr = np.pad(arr, ((0, pad_y), (0, pad_x)), mode="edge")
    return arr


def mode_downsample_2x2(arr: np.ndarray) -> np.ndarray:
    """2x2 winner-take-all downsample for label ids (never averages).

    Per output pixel: the most frequent non-zero child id wins; ties go to the
    smaller id. Background (0) only survives if all four children are 0.
    """
    arr = _pad_to_even(arr.astype(np.int32, copy=False))
    children = [arr[0::2, 0::2], arr[0::2, 1::2], arr[1::2, 0::2], arr[1::2, 1::2]]
    best_val = np.zeros_like(children[0])
    best_cnt = np.zeros_like(children[0])
    for candidate in children:
        count = np.zeros_like(children[0])
        for other in children:
            count += ((other == candidate) & (other != 0)).astype(count.dtype)
        is_foreground = candidate != 0
        take = is_foreground & (
            (count > best_cnt) | ((count == best_cnt) & ((best_val == 0) | (candidate < best_val)))
        )
        best_val = np.where(take, candidate, best_val)
        best_cnt = np.where(take, count, best_cnt)
    return best_val


def max_downsample_2x2(arr: np.ndarray) -> np.ndarray:
    """2x2 max-pool for the border mask (a block is border if any child is)."""
    arr = _pad_to_even(arr.astype(np.uint8, copy=False))
    children = [arr[0::2, 0::2], arr[0::2, 1::2], arr[1::2, 0::2], arr[1::2, 1::2]]
    return np.maximum(np.maximum(children[0], children[1]), np.maximum(children[2], children[3]))


def write_parent_chunk(
    child_array: zarr.Array,
    parent_array: zarr.Array,
    *,
    chunk_x: int,
    chunk_y: int,
    kind: str,
) -> None:
    parent_height, parent_width = parent_array.shape
    py0 = chunk_y * OVERLAY_CHUNK_SIZE
    px0 = chunk_x * OVERLAY_CHUNK_SIZE
    if py0 >= parent_height or px0 >= parent_width:
        return
    py1 = min(parent_height, py0 + OVERLAY_CHUNK_SIZE)
    px1 = min(parent_width, px0 + OVERLAY_CHUNK_SIZE)
    cy0, cx0 = py0 * 2, px0 * 2
    cy1 = min(child_array.shape[0], py1 * 2)
    cx1 = min(child_array.shape[1], px1 * 2)
    source = np.asarray(child_array[cy0:cy1, cx0:cx1])
    if kind == LABELS_ARRAY_KEY:
        downsampled = mode_downsample_2x2(source)
    else:
        downsampled = max_downsample_2x2(source)
    target_height = py1 - py0
    target_width = px1 - px0
    parent_array[py0:py1, px0:px1] = downsampled[:target_height, :target_width].astype(
        parent_array.dtype
    )


def downsample_block(
    group: zarr.Group,
    array_key: str,
    parent_level: int,
    block: tuple[int, int, int, int],
) -> None:
    """Downsample one parent block of an **already-open** staged store.

    Reads the child block, skips it if all background (so no zero-chunks are
    written for the mostly-empty gigapixel), else writes the mode/max-pooled
    parent block. Blocks within a level are disjoint and chunk-aligned, so
    concurrent callers never touch the same chunk file.

    Taking the open group rather than a path is what lets the in-process pyramid
    path open the store once and reuse the handle across every level and array,
    instead of paying a group open per block.
    """
    block_y0, block_y1, block_x0, block_x1 = block
    child = group[array_key][str(parent_level - 1)]
    parent = group[array_key][str(parent_level)]
    source = np.asarray(
        child[
            block_y0 * 2 : min(child.shape[0], block_y1 * 2),
            block_x0 * 2 : min(child.shape[1], block_x1 * 2),
        ]
    )
    if not source.any():
        return
    if array_key == LABELS_ARRAY_KEY:
        downsampled = mode_downsample_2x2(source)
    else:
        downsampled = max_downsample_2x2(source)
    parent[block_y0:block_y1, block_x0:block_x1] = downsampled[
        : block_y1 - block_y0, : block_x1 - block_x0
    ].astype(parent.dtype)


def open_staged_group(stage_root: str) -> zarr.Group:
    """Open a staged overlay store for read/write by path."""
    return zarr.open_group(str(stage_root), mode="a", zarr_format=2)


def close_staged_group(group: zarr.Group | None) -> None:
    """Release a staged store handle.

    Not tidiness: the caller moves the staging directory onto the published
    bundle path as soon as the pyramid is done, and on Windows a directory with
    an open handle under it will not move.
    """
    close = getattr(getattr(group, "store", None), "close", None)
    if callable(close):
        close()


def downsample_block_worker(task: tuple[str, str, int, tuple[int, int, int, int]]) -> None:
    """Process-pool entry point: open the staged zarr by path, then downsample.

    The parent process has already closed its handles, so each worker opens the
    store itself. Re-opening per block is the price of process isolation; the
    in-process path calls :func:`downsample_block` with one shared handle.
    """
    stage_root, array_key, parent_level, block = task
    downsample_block(open_staged_group(stage_root), array_key, parent_level, block)


def iter_level0_chunks(width: int, height: int) -> list[tuple[int, int]]:
    max_chunk_x = max(0, math.ceil(width / OVERLAY_CHUNK_SIZE))
    max_chunk_y = max(0, math.ceil(height / OVERLAY_CHUNK_SIZE))
    return [(chunk_x, chunk_y) for chunk_y in range(max_chunk_y) for chunk_x in range(max_chunk_x)]


def chunk_bounds(
    chunk_x: int,
    chunk_y: int,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x_min = chunk_x * OVERLAY_CHUNK_SIZE
    y_min = chunk_y * OVERLAY_CHUNK_SIZE
    return (
        x_min,
        y_min,
        min(width, x_min + OVERLAY_CHUNK_SIZE),
        min(height, y_min + OVERLAY_CHUNK_SIZE),
    )

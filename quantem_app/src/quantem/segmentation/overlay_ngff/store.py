"""Zarr ID-map overlay store: creation, validation, open, chunk serving.

Layout (one bundle)::

    labels.zarr/                 group  {.zattrs: overlay_format_version, labels:[...]}
      labels/                    group  {.zattrs: multiscales (y,x) + image-label}
        0/ 1/ 2/ ...             uint32 pyramid levels, chunks (256, 256)
      border/                    group  {.zattrs: multiscales (y,x)}
        0/ 1/ 2/ ...             uint8 pyramid levels, chunks (256, 256)
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

from quantem.segmentation.models import ImageSegmentation, SegmentationOverlayState

from .constants import (
    BORDER_ARRAY_KEY,
    BORDER_STORE_DTYPE,
    LABEL_STORE_DTYPE,
    LABELS_ARRAY_KEY,
    OVERLAY_ARRAY_KEYS,
    OVERLAY_CHUNK_SIZE,
    OVERLAY_FORMAT_VERSION,
)
from .dimensions import segmentation_dimensions
from .paths import OverlayStoreError, _remove_tree, get_overlay_active_bundle_path

_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

_ARRAY_DTYPES: dict[str, Any] = {
    LABELS_ARRAY_KEY: LABEL_STORE_DTYPE,
    BORDER_ARRAY_KEY: BORDER_STORE_DTYPE,
}


def _level_shapes(width: int, height: int) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = [(height, width)]
    while shapes[-1][0] > OVERLAY_CHUNK_SIZE or shapes[-1][1] > OVERLAY_CHUNK_SIZE:
        prev_height, prev_width = shapes[-1]
        shapes.append((max(1, math.ceil(prev_height / 2)), max(1, math.ceil(prev_width / 2))))
    return shapes


def parse_overlay_chunk_path(relative_path: str) -> tuple[str, int, tuple[int, int]] | None:
    """Parse ``<array>/<level>/<cy>.<cx>`` -> ``(array_key, level, (cy, cx))``."""
    parts = str(relative_path).split("/")
    if len(parts) != 3:
        return None
    array_key, level_token, chunk_token = parts
    if array_key not in OVERLAY_ARRAY_KEYS or not level_token.isdigit():
        return None
    chunk_parts = chunk_token.split(".")
    if len(chunk_parts) != 2 or not all(part.isdigit() for part in chunk_parts):
        return None
    chunk_y, chunk_x = (int(part) for part in chunk_parts)
    return array_key, int(level_token), (chunk_y, chunk_x)


def get_overlay_chunk_shape(
    segmentation: ImageSegmentation,
    *,
    level: int,
    chunk_coords: tuple[int, int],
) -> tuple[int, int] | None:
    """Return the (height, width) of a 2D chunk, clamped to the level bounds."""
    width, height = segmentation_dimensions(segmentation)
    level_shapes = _level_shapes(width, height)
    if level < 0 or level >= len(level_shapes):
        return None
    chunk_y, chunk_x = chunk_coords
    level_height, level_width = level_shapes[level]
    y_min = chunk_y * OVERLAY_CHUNK_SIZE
    x_min = chunk_x * OVERLAY_CHUNK_SIZE
    if y_min >= level_height or x_min >= level_width:
        return None
    return (
        min(OVERLAY_CHUNK_SIZE, level_height - y_min),
        min(OVERLAY_CHUNK_SIZE, level_width - x_min),
    )


def encode_zero_chunk(array_key: str, chunk_shape: tuple[int, int]) -> bytes:
    """Encode an all-background chunk for a missing (sparse) chunk request.

    Must use the same dtype + codec the array was created with so the client
    decodes it correctly.
    """
    dtype = _ARRAY_DTYPES.get(array_key)
    if dtype is None:
        raise OverlayStoreError(f"Unknown overlay array: {array_key!r}")
    return bytes(_COMPRESSOR.encode(np.zeros(chunk_shape, dtype=dtype)))


def _write_group_metadata(store_root: Path) -> None:
    zarr_root = zarr.open_group(str(store_root), mode="a", zarr_format=2)
    zarr_root.attrs["overlay_format_version"] = OVERLAY_FORMAT_VERSION
    zarr_root.attrs["labels"] = list(OVERLAY_ARRAY_KEYS)


def _retry_on_windows_lock(op: Callable[[], Any], *, attempts: int = 5) -> Any:
    """Run ``op``, retrying briefly on a Windows sharing violation.

    zarr writes metadata through ``tmp.replace(target)``. On Windows that raises
    ``PermissionError [WinError 5]`` if anything holds a transient handle on the
    target -- antivirus and the search indexer both do this routinely on a
    freshly written file. POSIX rename has no such failure mode, so upstream does
    not retry. Windows is a shipping platform for QuantEM, and an overlay rebuild
    failing because a virus scanner blinked is not acceptable, so retry here.
    """
    delay = 0.05
    for attempt in range(attempts):
        try:
            return op()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return None  # unreachable


def _write_array_group_metadata(
    array_group: zarr.Group,
    *,
    name: str,
    shapes: list[tuple[int, int]],
    is_label: bool,
) -> None:
    datasets_metadata: list[dict[str, Any]] = []
    for level_idx, _ in enumerate(shapes):
        datasets_metadata.append(
            {
                "path": str(level_idx),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [2**level_idx, 2**level_idx]}
                ],
            }
        )
    multiscales = [
        {
            "name": name,
            "version": "0.4",
            "axes": [
                {"name": "y", "type": "space", "unit": "pixel"},
                {"name": "x", "type": "space", "unit": "pixel"},
            ],
            "datasets": datasets_metadata,
        }
    ]

    def _write() -> None:
        array_group.attrs["multiscales"] = multiscales
        if is_label:
            array_group.attrs["image-label"] = {"version": "0.4"}

    _retry_on_windows_lock(_write)


def _create_array_pyramid(
    zarr_root: zarr.Group,
    *,
    array_key: str,
    level_shapes: list[tuple[int, int]],
) -> list[zarr.Array]:
    group = zarr_root.create_group(array_key)
    arrays: list[zarr.Array] = []
    for level_idx, (height, width) in enumerate(level_shapes):
        arrays.append(
            group.create_array(
                str(level_idx),
                shape=(height, width),
                chunks=(
                    min(OVERLAY_CHUNK_SIZE, height),
                    min(OVERLAY_CHUNK_SIZE, width),
                ),
                dtype=_ARRAY_DTYPES[array_key],
                compressor=_COMPRESSOR,
                overwrite=True,
                fill_value=0,
            )
        )
    _write_array_group_metadata(
        group,
        name=array_key,
        shapes=level_shapes,
        is_label=array_key == LABELS_ARRAY_KEY,
    )
    return arrays


def _create_empty_label_store(
    segmentation: ImageSegmentation,
    store_root: Path,
) -> dict[str, list[zarr.Array]]:
    if store_root.exists():
        _remove_tree(store_root)
    store_root.parent.mkdir(parents=True, exist_ok=True)
    zarr_root = zarr.open_group(str(store_root), mode="w", zarr_format=2)
    width, height = segmentation_dimensions(segmentation)
    level_shapes = _level_shapes(width, height)
    arrays = {
        array_key: _create_array_pyramid(zarr_root, array_key=array_key, level_shapes=level_shapes)
        for array_key in OVERLAY_ARRAY_KEYS
    }
    _write_group_metadata(store_root)
    return arrays


def _is_valid_label_store(
    store_root: Path,
    *,
    width: int,
    height: int,
) -> bool:
    attrs_path = store_root / ".zattrs"
    zgroup_path = store_root / ".zgroup"
    if not attrs_path.exists() or not zgroup_path.exists():
        return False
    try:
        attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(attrs.get("overlay_format_version", 0)) != OVERLAY_FORMAT_VERSION:
        return False
    for array_key in OVERLAY_ARRAY_KEYS:
        try:
            first_array = zarr.open_array(str(store_root / array_key / "0"), mode="r")
        except Exception:
            return False
        if tuple(first_array.shape) != (height, width):
            return False
        if np.dtype(first_array.dtype) != np.dtype(_ARRAY_DTYPES[array_key]):
            return False
    return True


def _open_label_arrays(
    state: SegmentationOverlayState,
) -> dict[str, list[zarr.Array]]:
    if state.bundle_version <= 0:
        raise OverlayStoreError("overlay bundle_version is not set")
    store_root = get_overlay_active_bundle_path(state)
    width, height = segmentation_dimensions(state.segmentation)
    if not _is_valid_label_store(store_root, width=width, height=height):
        raise OverlayStoreError(f"Invalid overlay store at {store_root}")
    try:
        zarr_root = zarr.open_group(str(store_root), mode="a", zarr_format=2)
    except Exception as exc:
        raise OverlayStoreError(f"Failed to open overlay store: {exc}") from exc
    level_count = len(_level_shapes(width, height))
    arrays: dict[str, list[zarr.Array]] = {}
    for array_key in OVERLAY_ARRAY_KEYS:
        group = zarr_root[array_key]
        arrays[array_key] = [group[str(level_idx)] for level_idx in range(level_count)]
    return arrays

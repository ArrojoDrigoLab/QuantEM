"""Browser-sized previews of authoritative probability maps.

The stored PNG or chunked Zarr remains full resolution because replaying a
threshold must use every source pixel. A browser preview has a different
constraint: common canvas and GPU texture limits are below the dimensions of
large EM montages, and decoding a gigapixel grayscale image into RGBA can
require several gigabytes.

This module writes a small sidecar used only for display. New inference runs
produce it directly from the already-resident uint8 array; older maps get one
lazily on their first preview request.
"""

from __future__ import annotations

import math
import os
import struct
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import zarr
from PIL import Image

from quantem.assets.canonical_decode import decode_canonical_array

# These are application-generated microscopy rasters, not untrusted uploads.
Image.MAX_IMAGE_PIXELS = None

MAX_PROBABILITY_PREVIEW_DIMENSION = 4096

_PREVIEW_WRITE_LOCK = threading.Lock()

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PNG_HEADER_BYTES = 24


def probability_map_size(source_path: Path) -> tuple[int, int]:
    """Read ``(width, height)`` without decoding the probability pixels."""
    if source_path.is_dir():
        array = zarr.open_array(str(source_path), mode="r")
        if len(array.shape) != 2:
            raise ValueError(f"Probability map is not two-dimensional: {source_path}")
        return int(array.shape[1]), int(array.shape[0])
    with source_path.open("rb") as source:
        header = source.read(_PNG_HEADER_BYTES)
    if (
        len(header) != _PNG_HEADER_BYTES
        or not header.startswith(_PNG_MAGIC)
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"Probability map is not a valid PNG: {source_path}")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1 or height < 1:
        raise ValueError(f"Probability map has invalid dimensions {width}x{height}: {source_path}")
    return width, height


def probability_preview_path(
    source_path: Path,
    *,
    max_dimension: int = MAX_PROBABILITY_PREVIEW_DIMENSION,
) -> Path:
    """Sidecar path for one authoritative map and preview-size contract."""
    stem = source_path.name if source_path.suffix.lower() == ".zarr" else source_path.stem
    return source_path.with_name(f"{stem}_preview_{max_dimension}.png")


def result_preview_cache_path(source_path: Path) -> Path:
    """Owned derived images, reclaimed with the last reference to this map."""
    return source_path.with_name(f"{source_path.name}.result-previews")


def _preview_shape(width: int, height: int, max_dimension: int) -> tuple[int, int, int]:
    factor = max(1, math.ceil(max(width, height) / max_dimension))
    return math.ceil(width / factor), math.ceil(height / factor), factor


def _atomic_save_grayscale(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}-{uuid4().hex}.tmp{destination.suffix}")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_probability_preview(
    source_path: Path,
    probability_uint8: np.ndarray,
    *,
    max_dimension: int = MAX_PROBABILITY_PREVIEW_DIMENSION,
) -> Path:
    """Write a nearest-sampled preview without changing probability values."""
    shape = getattr(probability_uint8, "shape", ())
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D probability map, got shape {shape}")
    height, width = int(shape[0]), int(shape[1])
    _, _, factor = _preview_shape(width, height, max_dimension)
    if factor == 1 and source_path.is_file():
        return source_path

    destination = probability_preview_path(source_path, max_dimension=max_dimension)
    sampled = np.ascontiguousarray(
        probability_uint8[::factor, ::factor],
        dtype=np.uint8,
    )
    _atomic_save_grayscale(Image.fromarray(sampled, mode="L"), destination)
    return destination


def ensure_probability_preview(
    source_path: Path,
    *,
    max_dimension: int = MAX_PROBABILITY_PREVIEW_DIMENSION,
) -> Path:
    """Return a browser-safe map, creating a cached sidecar when necessary."""
    width, height = probability_map_size(source_path)
    _, _, factor = _preview_shape(
        width,
        height,
        max_dimension,
    )
    if factor == 1 and source_path.is_file():
        return source_path

    destination = probability_preview_path(source_path, max_dimension=max_dimension)
    try:
        if destination.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
            return destination
    except FileNotFoundError:
        pass

    with _PREVIEW_WRITE_LOCK:
        try:
            if destination.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
                return destination
        except FileNotFoundError:
            pass
        probability_uint8 = (
            zarr.open_array(str(source_path), mode="r")
            if source_path.is_dir()
            else decode_canonical_array(source_path)
        )
        sampled = np.ascontiguousarray(probability_uint8[::factor, ::factor], dtype=np.uint8)
        _atomic_save_grayscale(Image.fromarray(sampled, mode="L"), destination)
    return destination

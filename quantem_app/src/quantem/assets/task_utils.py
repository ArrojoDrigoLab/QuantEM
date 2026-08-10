"""
Utility functions for preprocessing tasks.

This module provides centralized utilities for loading images: full arrays,
downsampled previews, and ROI windows from either the canonical PNG or the
NGFF pyramid.
"""

import logging
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

try:  # QuantEM port: pyvips is an optional accelerator (`pip install quantem[vips]`).
    import pyvips  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only without libvips installed
    # Both call sites below already wrap their pyvips path in `except Exception`
    # and fall back to Pillow, so a None here degrades cleanly rather than failing.
    pyvips = None  # type: ignore[assignment]
import zarr

# Disable decompression bomb check for large SEM/TEM images
Image.MAX_IMAGE_PIXELS = None

from .asset_openable import get_asset_ngff_path
from .file_paths import get_file_absolute_path
from .ngff import _is_valid_ngff_store, get_ngff_root_path

logger = logging.getLogger(__name__)


def load_image_array(image) -> tuple[np.ndarray, float]:
    """
    Load PNG image as numpy array.

    Args:
        image: asset-backed openable

    Returns:
        Tuple of (image_array, load_time_seconds)
        image_array: 2D numpy array (uint8, grayscale)
        load_time_seconds: Time taken to load the image

    Raises:
        FileNotFoundError: If PNG file doesn't exist
        ValueError: If image cannot be loaded or converted
    """
    load_start = time.time()
    file_path = get_file_absolute_path(image)

    pil_image = Image.open(file_path)
    # Convert to grayscale if needed
    if pil_image.mode != "L":
        pil_image = pil_image.convert("L")

    image_array = np.array(pil_image, dtype=np.uint8)
    load_elapsed = time.time() - load_start

    logger.info(
        f"Loaded image array: shape={image_array.shape}, dtype={image_array.dtype} "
        f"(took {load_elapsed:.2f} seconds)"
    )

    return image_array, load_elapsed


def _vips_to_numpy(vips_image) -> np.ndarray:
    """Convert a pyvips image to a numpy array."""
    height = vips_image.height
    width = vips_image.width
    bands = vips_image.bands
    memory = vips_image.write_to_memory()
    array = np.frombuffer(memory, dtype=np.uint8)
    if bands == 1:
        return array.reshape(height, width)
    return array.reshape(height, width, bands)


def load_image_preview_array(image, max_size: int = 1024) -> np.ndarray:
    """
    Load a downsampled preview of the PNG image as a grayscale numpy array.

    Args:
        image: asset-backed openable
        max_size: Maximum dimension (width/height) of preview

    Returns:
        2D uint8 numpy array
    """
    file_path = get_file_absolute_path(image)
    try:
        vips_image = pyvips.Image.new_from_file(str(file_path), access="sequential")
        if vips_image.bands > 1:
            vips_image = vips_image.extract_band(0)
        scale = min(max_size / vips_image.width, max_size / vips_image.height, 1.0)
        if scale < 1.0:
            vips_image = vips_image.resize(scale)
        return _vips_to_numpy(vips_image)
    except Exception:
        pil_image = Image.open(file_path)
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        pil_image.thumbnail((max_size, max_size))
        return np.array(pil_image, dtype=np.uint8)


def load_image_roi_array(image, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Load a ROI window from the PNG image as a grayscale numpy array.

    Args:
        image: asset-backed openable
        x: Left coordinate in full-res pixels
        y: Top coordinate in full-res pixels
        width: ROI width in pixels
        height: ROI height in pixels

    Returns:
        2D uint8 numpy array of shape (height, width)
    """
    file_path = get_file_absolute_path(image)
    try:
        vips_image = pyvips.Image.new_from_file(str(file_path), access="sequential")
        if vips_image.bands > 1:
            vips_image = vips_image.extract_band(0)
        roi = vips_image.crop(x, y, width, height)
        return _vips_to_numpy(roi)
    except Exception:
        pil_image = Image.open(file_path)
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        roi = pil_image.crop((x, y, x + width, y + height))
        return np.array(roi, dtype=np.uint8)


def load_image_ngff_level0_roi_array(
    image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Load a ROI window from the image OME-NGFF level-0 pyramid.

    Raises:
        FileNotFoundError: If the NGFF store or level-0 array is unavailable.
        ValueError: If the level-0 array does not match the expected shape.
    """
    asset = getattr(image, "asset", None)
    ngff_root = get_asset_ngff_path(asset) if asset is not None else None
    if ngff_root is None:
        ngff_root = get_ngff_root_path(image)
    level0 = _get_cached_ngff_level0_array(ngff_root)
    roi = np.asarray(level0[0, y : y + height, x : x + width], dtype=np.uint8)
    return roi


@lru_cache(maxsize=16)
def _open_ngff_level0_array_cached(
    ngff_root_str: str,
    store_version_ns: int,
):
    ngff_root = Path(ngff_root_str)
    if not _is_valid_ngff_store(ngff_root):
        raise FileNotFoundError("NGFF store unavailable")
    level0 = zarr.open_array(str(ngff_root / "0"), mode="r")
    if len(level0.shape) != 3 or int(level0.shape[0]) != 1:
        raise ValueError(f"Unexpected NGFF level-0 shape: {level0.shape}")
    return level0


def _get_cached_ngff_level0_array(ngff_root: Path):
    if not ngff_root.exists():
        raise FileNotFoundError("NGFF store unavailable")
    level0_meta = ngff_root / "0" / ".zarray"
    if not level0_meta.exists():
        raise FileNotFoundError("NGFF store unavailable")
    return _open_ngff_level0_array_cached(
        str(ngff_root),
        int(level0_meta.stat().st_mtime_ns),
    )

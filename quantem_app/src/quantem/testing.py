"""Shared fixtures for the QuantEM test suite.

Two deliberate properties:

1. **The TIFF fixture is generated, not shipped.** :func:`write_test_tiff`
   synthesises the image at test time, so the suite runs on a clean checkout and
   in CI with no large binary fixtures to download.
2. **It matches the QuantEM schema.** ``pixel_size_nm`` is set here, because
   nothing measurable works without it.

Import as ``from quantem.testing import ...``; a top-level ``tests`` package is
not importable from an installed distribution.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import numpy as np
import tifffile
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image

from quantem.assets.asset_openable import AssetOpenable, get_asset_openable
from quantem.assets.models import Asset, Rendition
from quantem.assets.utils import (
    convert_tiff_to_png,
    create_roi_image_from_image,
    extract_tiff_metadata,
)
from quantem.core.config import DATA_DIR, IMAGES_DIR, PROB_MAPS_DIR, STORAGE_DIR
from quantem.core.local_storage import normalize_stored_path_value
from quantem.segmentation.models import ImageSegmentation, ProbabilityMap
from quantem.segmentation.utils import get_or_create_mitochondria_type

#: Pixel size stamped on generated fixtures, and the scale the Figure-4
#: pipeline assumes.
TEST_PIXEL_SIZE_NM = 5.0


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------


def make_em_like_array(
    width: int = 512, height: int = 512, *, seed: int = 0
) -> np.ndarray:
    """A deterministic 8-bit greyscale image with EM-ish texture and blobs.

    Not a simulation — just structure enough that thresholding, connected
    components and contour extraction have something non-degenerate to act on.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    background = 110 + 18 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    background += rng.normal(0.0, 6.0, size=(height, width))

    # A few dark elliptical "organelles" with brighter rims.
    for cx, cy, rx, ry in (
        (width * 0.28, height * 0.30, width * 0.10, height * 0.07),
        (width * 0.62, height * 0.45, width * 0.08, height * 0.11),
        (width * 0.40, height * 0.72, width * 0.13, height * 0.06),
    ):
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        background[d <= 1.0] -= 55
        background[(d > 1.0) & (d <= 1.35)] += 30

    return np.clip(background, 0, 255).astype(np.uint8)


def write_test_tiff(
    path: Path, *, width: int = 512, height: int = 512, seed: int = 0
) -> Path:
    """Write a synthetic single-page greyscale TIFF carrying a resolution tag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = make_em_like_array(width, height, seed=seed)
    # resolution in pixels/cm so extract_tiff_metadata can derive nm/px
    px_per_cm = 1e7 / TEST_PIXEL_SIZE_NM
    tifffile.imwrite(
        str(path),
        array,
        photometric="minisblack",
        resolution=(px_per_cm, px_per_cm),
        resolutionunit="CENTIMETER",
    )
    return path


def test_tiff_bytes(*, width: int = 512, height: int = 512, seed: int = 0) -> bytes:
    """Bytes of a synthetic TIFF, for upload-endpoint tests."""
    tmp = STORAGE_DIR / "tmp" / f"fixture_{uuid4().hex}.tif"
    try:
        write_test_tiff(tmp, width=width, height=height, seed=seed)
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def build_test_upload_file(filename: str | None = None) -> SimpleUploadedFile:
    return SimpleUploadedFile(
        filename or "quantem_fixture.tif", test_tiff_bytes(), content_type="image/tiff"
    )


# ---------------------------------------------------------------------------
# Asset construction
# ---------------------------------------------------------------------------


def create_image_from_test_tiff(
    display_name: str = "Test Image", *, width: int = 1024, height: int = 1024
) -> AssetOpenable:
    """Create an asset from a generated TIFF, converted to canonical PNG.

    Exercises the real ``extract_tiff_metadata`` -> ``convert_tiff_to_png`` path,
    which is the point of using a TIFF here rather than writing a PNG directly.
    """
    source = STORAGE_DIR / "tmp" / f"fixture_{uuid4().hex}.tif"
    write_test_tiff(source, width=width, height=height)
    try:
        metadata = extract_tiff_metadata(source)
        target = IMAGES_DIR / f"{source.stem}_{uuid4().hex}.png"
        png_path = convert_tiff_to_png(source, metadata, target)
        return _create_asset_openable(
            original_filename=source.name,
            display_name=display_name,
            abs_path=png_path,
            width=int(metadata.get("width") or 0),
            height=int(metadata.get("height") or 0),
            channels=int(metadata.get("channels") or 1),
            bit_depth=int(metadata.get("bit_depth") or 8),
            pixel_size_nm=metadata.get("pixel_size_nm") or TEST_PIXEL_SIZE_NM,
        )
    finally:
        source.unlink(missing_ok=True)


def create_small_test_image(
    display_name: str = "Test Image",
    *,
    width: int = 256,
    height: int = 256,
    textured: bool = False,
) -> AssetOpenable:
    """Create a PNG-backed asset directly, skipping the TIFF conversion path."""
    abs_path = IMAGES_DIR / f"{uuid4().hex}.png"
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        make_em_like_array(width, height)
        if textured
        else np.zeros((height, width), dtype=np.uint8)
    )
    Image.fromarray(data, mode="L").save(abs_path)
    return _create_asset_openable(
        original_filename=abs_path.name,
        display_name=display_name,
        abs_path=abs_path,
        width=width,
        height=height,
        channels=1,
        bit_depth=8,
        pixel_size_nm=TEST_PIXEL_SIZE_NM,
    )


def _create_asset_openable(
    *,
    original_filename: str,
    display_name: str,
    abs_path: Path,
    width: int,
    height: int,
    channels: int,
    bit_depth: int,
    pixel_size_nm: float | None = None,
) -> AssetOpenable:
    relative_path = normalize_stored_path_value(abs_path, relative_to=DATA_DIR)
    asset = Asset.objects.create(
        display_name=display_name,
        original_filename=original_filename,
        logical_width=width,
        logical_height=height,
        channels=channels,
        bit_depth=bit_depth,
        pixel_size_nm=pixel_size_nm,
        preprocess_stage="DONE",
        preprocess_progress=100.0,
    )
    Rendition.objects.create(
        asset=asset,
        type=Rendition.TYPE_FULL,
        storage_root="DATA_DIR",
        stored_path=relative_path,
        path_exists=abs_path.exists(),
        is_directory=False,
        stored_width=width,
        stored_height=height,
        stored_channels=channels,
        stored_bit_depth=bit_depth,
        metadata={"display_name": display_name},
    )
    return get_asset_openable(asset)


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------


def create_mitochondria_segmentation(image) -> ImageSegmentation:
    seg_type = get_or_create_mitochondria_type()
    asset = getattr(image, "asset", image)
    segmentation, _ = ImageSegmentation.objects.get_or_create(
        asset=asset, segmentation_type=seg_type
    )
    return segmentation


def create_roi(segmentation: ImageSegmentation, width: int, height: int):
    target_image = get_asset_openable(segmentation.asset)
    return create_roi_image_from_image(
        target_image, x=0, y=0, width=width, height=height, source="AUTO"
    )


def make_prob_map(shape: tuple[int, int]) -> np.ndarray:
    """A probability map with two well-separated high-confidence blobs."""
    prob_map = np.zeros(shape, dtype=np.float32)
    h, w = shape
    cx, cy = w // 3, h // 3
    prob_map[cy : cy + 32, cx : cx + 32] = 0.95
    prob_map[h // 2 : h // 2 + 16, w // 2 : w // 2 + 16] = 0.9
    return prob_map


def write_prob_map_png(
    segmentation: ImageSegmentation, prob_map: np.ndarray, *, name: str
) -> ProbabilityMap:
    prob_maps_dir = PROB_MAPS_DIR / str(segmentation.id)
    prob_maps_dir.mkdir(parents=True, exist_ok=True)
    output_path = prob_maps_dir / f"{name}_{segmentation.id}.png"
    as_uint8 = np.rint(np.clip(prob_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(as_uint8, mode="L").save(str(output_path), "PNG", compress_level=6)
    return ProbabilityMap.objects.create(
        segmentation=segmentation,
        name=name,
        file_path=normalize_stored_path_value(output_path, relative_to=STORAGE_DIR),
        channel_index=0,
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def set_env(overrides: dict[str, str]) -> Iterator[None]:
    original: dict[str, str | None] = {}
    for key, value in overrides.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, prev in original.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def limit_prob_map_candidates():
    return set_env(
        {
            "PROB_MAP_MAX_CANDIDATES_PER_COMPONENT": "1",
            "PROB_MAP_MAX_PROMPT_RUNS": "1",
            "PROB_MAP_MAX_POS_POINTS": "1",
            "PROB_MAP_MAX_NEG_POINTS": "1",
            "PROB_MAP_MIN_COMPONENT_AREA": "50",
            "PROB_MAP_MIN_MASK_AREA": "50",
        }
    )

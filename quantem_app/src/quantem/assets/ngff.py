"""
OME-NGFF (OME-Zarr) generation helpers for image viewing.

Generated assets are stored under storage/data/tmp/ngff/<image_id>.zarr.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc
from PIL import Image

try:
    import pyvips
except ImportError:
    pyvips = None

from quantem.core.config import NGFF_TMP_DIR

from .file_paths import get_file_absolute_path

logger = logging.getLogger(__name__)

# Disable decompression bomb checks for large microscopy imagery.
Image.MAX_IMAGE_PIXELS = None

NGFF_CHUNK_SIZE = 1024  # 1024^2 chunks: ~16x fewer files than 256 -> much faster NGFF writes (esp. Windows small-file I/O)
NGFF_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
NGFF_THUMBNAIL_TARGET_MAX_SIDE = 256


def get_ngff_paths(image) -> tuple[Path, Path]:
    """
    Return (ngff_root_dir, ngff_root_attrs_path) for an image.
    """
    ngff_root = NGFF_TMP_DIR / f"{image.id}.zarr"
    attrs_path = ngff_root / ".zattrs"
    return ngff_root, attrs_path


def get_ngff_root_path(image) -> Path:
    """
    Return the root directory for an image's NGFF zarr store.
    """
    root, _ = get_ngff_paths(image)
    return root


def _is_valid_ngff_store(ngff_root: Path) -> bool:
    """
    Validate that a zarr store has Viv-compatible OME-NGFF multiscale metadata.
    """
    attrs_path = ngff_root / ".zattrs"
    zgroup_path = ngff_root / ".zgroup"
    if not attrs_path.exists() or not zgroup_path.exists():
        return False

    try:
        attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or len(multiscales) == 0:
        return False

    first_scale = multiscales[0]
    if not isinstance(first_scale, dict):
        return False
    datasets = first_scale.get("datasets")
    if not isinstance(datasets, list) or len(datasets) == 0:
        return False

    for dataset in datasets:
        if not isinstance(dataset, dict):
            return False
        dataset_path = dataset.get("path")
        if not isinstance(dataset_path, str) or not dataset_path:
            return False
        zarray_path = ngff_root / dataset_path / ".zarray"
        if not zarray_path.exists():
            return False

    return True


def render_lowest_resolution_ngff_png_from_root(
    ngff_root: Path,
    *,
    attrs_path: Path | None = None,
) -> bytes:
    """Render a small dashboard preview from an existing NGFF root path."""

    attrs_path = attrs_path or (ngff_root / ".zattrs")
    if not ngff_root.exists() or not _is_valid_ngff_store(ngff_root):
        raise FileNotFoundError(f"Valid NGFF store not found at {ngff_root}")

    attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    datasets = attrs["multiscales"][0]["datasets"]
    dataset_path = _select_ngff_thumbnail_dataset_path(ngff_root, datasets)
    array = zarr.open_array(str(ngff_root / dataset_path), mode="r")
    data = np.asarray(array[:])
    data = _select_ngff_preview_plane(data)
    data = _normalize_ngff_preview_plane(data)

    preview = Image.fromarray(data, mode="L")
    output = BytesIO()
    preview.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _select_ngff_thumbnail_dataset_path(
    ngff_root: Path,
    datasets: list[dict[str, Any]],
) -> str:
    selected_path = datasets[0]["path"]
    for dataset in datasets:
        dataset_path = dataset["path"]
        array = zarr.open_array(str(ngff_root / dataset_path), mode="r")
        height, width = _ngff_array_spatial_shape(array.shape)
        if max(height, width) < NGFF_THUMBNAIL_TARGET_MAX_SIDE:
            break
        selected_path = dataset_path
    return selected_path


def _ngff_array_spatial_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    # All generated stores keep y, x as the trailing two axes (2D [c,y,x] and
    # 3D [c,z,y,x] alike), so the spatial shape is always the last two dims.
    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])
    raise ValueError(f"Unsupported NGFF preview shape: {shape}")


def _select_ngff_preview_plane(data: np.ndarray) -> np.ndarray:
    # Collapse leading (channel, z) axes down to a single 2D y/x plane. The trailing
    # two axes are always spatial (stores are [c, y, x] or [c, z, y, x]); at each
    # leading axis take the slice carrying the most signal rather than a fixed
    # index. A positional pick landed on the blank leading plane of small z-subsets
    # (e.g. the 3-plane decimation, where plane 0 is empty padding and the real data
    # sits in planes 1-2) and rendered an all-black dashboard preview.
    while data.ndim > 2:
        data = _highest_signal_slice(data)
    return data


def _highest_signal_slice(arr: np.ndarray) -> np.ndarray:
    # Pick the slice along the leading axis with the greatest intensity spread, so
    # blank padding slices are skipped. If no slice has any spread (every slice is
    # uniform), prefer a non-zero constant slice over a blank one. Fall back to the
    # geometric middle only when every slice is identically blank (e.g. an entirely
    # empty store), preserving prior behaviour for that case.
    values = np.asarray(arr, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    spreads = flat.std(axis=1)
    if np.any(spreads > 0.0):
        return arr[int(np.argmax(spreads))]
    magnitudes = np.abs(flat).mean(axis=1)
    if np.any(magnitudes > 0.0):
        return arr[int(np.argmax(magnitudes))]
    return arr[arr.shape[0] // 2]


def _normalize_ngff_preview_plane(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)

    values = np.asarray(data, dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)

    low, high = np.percentile(finite_values, [1, 99])
    if high <= low:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
    if high <= low:
        fill_value = 0 if high <= 0 else 127
        return np.full(data.shape, fill_value, dtype=np.uint8)

    normalized = (values - low) * (255.0 / (high - low))
    return np.nan_to_num(np.clip(normalized, 0, 255), nan=0).astype(np.uint8)


def _level_shapes(height: int, width: int) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = [(height, width)]
    while min(shapes[-1]) > 1:
        prev_height, prev_width = shapes[-1]
        next_height = max(1, math.ceil(prev_height / 2))
        next_width = max(1, math.ceil(prev_width / 2))
        if (next_height, next_width) == (prev_height, prev_width):
            break
        shapes.append((next_height, next_width))
    return shapes


def _iter_level_chunks(width: int, height: int) -> Iterator[tuple[int, int]]:
    max_chunk_x = max(0, math.ceil(width / NGFF_CHUNK_SIZE))
    max_chunk_y = max(0, math.ceil(height / NGFF_CHUNK_SIZE))
    for chunk_y in range(max_chunk_y):
        for chunk_x in range(max_chunk_x):
            yield chunk_x, chunk_y


def _chunk_bounds(
    chunk_x: int,
    chunk_y: int,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x_min = chunk_x * NGFF_CHUNK_SIZE
    y_min = chunk_y * NGFF_CHUNK_SIZE
    return (
        x_min,
        y_min,
        min(width, x_min + NGFF_CHUNK_SIZE),
        min(height, y_min + NGFF_CHUNK_SIZE),
    )


def _write_multiscale_metadata(
    image,
    zarr_root,
    level_shapes: list[tuple[int, int]],
) -> None:
    datasets_metadata: list[dict[str, Any]] = []
    for level_idx, _shape in enumerate(level_shapes):
        datasets_metadata.append(
            {
                "path": str(level_idx),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 2**level_idx, 2**level_idx]}
                ],
            }
        )

    zarr_root.attrs["multiscales"] = [
        {
            "name": str(image.id),
            "version": "0.4",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "pixel"},
                {"name": "x", "type": "space", "unit": "pixel"},
            ],
            "datasets": datasets_metadata,
        }
    ]
    zarr_root.attrs["omero"] = {
        "name": image.display_name,
        "rdefs": {"model": "greyscale"},
        "channels": [
            {
                "label": "intensity",
                "color": "FFFFFF",
                "window": {
                    "start": 0,
                    "end": 255,
                    "min": 0,
                    "max": 255,
                },
                "active": True,
            }
        ],
    }


def _create_empty_store(
    image,
    ngff_root: Path,
) -> list[zarr.Array]:
    if ngff_root.exists():
        shutil.rmtree(ngff_root)

    level_shapes = _level_shapes(int(image.height), int(image.width))
    ngff_root.parent.mkdir(parents=True, exist_ok=True)
    zarr_root = zarr.open_group(str(ngff_root), mode="w", zarr_format=2)
    arrays: list[zarr.Array] = []
    for level_idx, (height, width) in enumerate(level_shapes):
        arrays.append(
            zarr_root.create_array(
                str(level_idx),
                shape=(1, height, width),
                chunks=(
                    1,
                    min(NGFF_CHUNK_SIZE, height),
                    min(NGFF_CHUNK_SIZE, width),
                ),
                dtype=np.uint8,
                compressor=NGFF_COMPRESSOR,
                overwrite=True,
                fill_value=0,
            )
        )
    _write_multiscale_metadata(image, zarr_root, level_shapes)
    return arrays


def _vips_to_numpy(vips_image) -> np.ndarray:
    memory = vips_image.write_to_memory()
    array = np.frombuffer(memory, dtype=np.uint8)
    return array.reshape(vips_image.height, vips_image.width)


class _ChunkedImageReader:
    def __init__(self, source_path: Path):
        self.backend = "pyvips" if pyvips is not None else "pil"
        self._pil_image = None
        self._vips_image = None

        if pyvips is not None:
            image = pyvips.Image.new_from_file(str(source_path), access="random")
            if image.bands > 1:
                image = image.extract_band(0)
            if image.format != "uchar":
                image = image.cast("uchar")
            self._vips_image = image
            return

        pil_image = Image.open(source_path)
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        self._pil_image = pil_image

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        if self._vips_image is not None:
            region = self._vips_image.crop(x, y, width, height)
            return _vips_to_numpy(region)
        assert self._pil_image is not None
        region = self._pil_image.crop((x, y, x + width, y + height))
        return np.asarray(region, dtype=np.uint8)

    def close(self) -> None:
        if self._pil_image is not None:
            self._pil_image.close()
            self._pil_image = None


def _write_level0_from_png(
    image,
    *,
    source_path: Path,
    level0_array: zarr.Array,
) -> None:
    level_height = int(level0_array.shape[1])
    level_width = int(level0_array.shape[2])
    source_open_start = time.time()
    reader = _ChunkedImageReader(source_path)
    source_open_elapsed = time.time() - source_open_start

    read_seconds = 0.0
    write_seconds = 0.0
    chunk_count = 0
    level_start = time.time()
    try:
        for chunk_x, chunk_y in _iter_level_chunks(level_width, level_height):
            x_min, y_min, x_max, y_max = _chunk_bounds(
                chunk_x,
                chunk_y,
                width=level_width,
                height=level_height,
            )
            chunk_width = x_max - x_min
            chunk_height = y_max - y_min

            read_start = time.time()
            chunk = reader.read_region(x_min, y_min, chunk_width, chunk_height)
            read_seconds += time.time() - read_start

            write_start = time.time()
            level0_array[0, y_min:y_max, x_min:x_max] = chunk
            write_seconds += time.time() - write_start
            chunk_count += 1
    finally:
        reader.close()

    level_elapsed = time.time() - level_start
    logger.info(
        "Image %s: NGFF level 0 completed in %.2fs across %d chunks (source_open=%.2fs read=%.2fs zarr_write=%.2fs backend=%s)",
        image.id,
        level_elapsed,
        chunk_count,
        source_open_elapsed,
        read_seconds,
        write_seconds,
        reader.backend,
    )


def _downsample_region(
    source_region: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    if target_height <= 0 or target_width <= 0:
        return np.zeros((0, 0), dtype=np.uint8)

    expected_height = max(1, target_height * 2)
    expected_width = max(1, target_width * 2)
    pad_y = max(0, expected_height - source_region.shape[0])
    pad_x = max(0, expected_width - source_region.shape[1])
    if pad_y or pad_x:
        source_region = np.pad(
            source_region,
            ((0, pad_y), (0, pad_x)),
            mode="edge",
        )
    source_region = source_region[:expected_height, :expected_width]
    reshaped = source_region.reshape(target_height, 2, target_width, 2)
    return np.rint(reshaped.mean(axis=(1, 3))).astype(np.uint8)


def _write_downsampled_level(
    image,
    *,
    level_idx: int,
    child_array: zarr.Array,
    parent_array: zarr.Array,
) -> None:
    child_height = int(child_array.shape[1])
    child_width = int(child_array.shape[2])
    parent_height = int(parent_array.shape[1])
    parent_width = int(parent_array.shape[2])

    read_seconds = 0.0
    downsample_seconds = 0.0
    write_seconds = 0.0
    chunk_count = 0
    level_start = time.time()

    for chunk_x, chunk_y in _iter_level_chunks(parent_width, parent_height):
        x_min, y_min, x_max, y_max = _chunk_bounds(
            chunk_x,
            chunk_y,
            width=parent_width,
            height=parent_height,
        )
        target_width = x_max - x_min
        target_height = y_max - y_min

        child_x_min = x_min * 2
        child_y_min = y_min * 2
        child_x_max = min(child_width, x_max * 2)
        child_y_max = min(child_height, y_max * 2)

        read_start = time.time()
        source_region = np.asarray(
            child_array[0, child_y_min:child_y_max, child_x_min:child_x_max],
            dtype=np.uint8,
        )
        read_seconds += time.time() - read_start

        downsample_start = time.time()
        parent_chunk = _downsample_region(
            source_region,
            target_height=target_height,
            target_width=target_width,
        )
        downsample_seconds += time.time() - downsample_start

        write_start = time.time()
        parent_array[0, y_min:y_max, x_min:x_max] = parent_chunk
        write_seconds += time.time() - write_start
        chunk_count += 1

    level_elapsed = time.time() - level_start
    logger.info(
        "Image %s: NGFF level %s completed in %.2fs across %d chunks (read=%.2fs downsample=%.2fs zarr_write=%.2fs)",
        image.id,
        level_idx,
        level_elapsed,
        chunk_count,
        read_seconds,
        downsample_seconds,
        write_seconds,
    )


def _write_multiscale_metadata_3d(
    image,
    zarr_root,
    level_shapes: list[tuple[int, int]],
    z_scale: float,
) -> None:
    """OME-NGFF metadata for a 4D [c, z, y, x] volume store.

    z is never downsampled, so its scale is constant across pyramid levels while
    the xy scale doubles per level. ``z_scale`` carries the physical anisotropy
    (effective z spacing / xy spacing) so distances render correctly.
    """

    datasets_metadata: list[dict[str, Any]] = []
    for level_idx, _shape in enumerate(level_shapes):
        datasets_metadata.append(
            {
                "path": str(level_idx),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1, float(z_scale), 2**level_idx, 2**level_idx],
                    }
                ],
            }
        )

    zarr_root.attrs["multiscales"] = [
        {
            "name": str(image.id),
            "version": "0.4",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "pixel"},
                {"name": "y", "type": "space", "unit": "pixel"},
                {"name": "x", "type": "space", "unit": "pixel"},
            ],
            "datasets": datasets_metadata,
        }
    ]
    zarr_root.attrs["omero"] = {
        "name": image.display_name,
        "rdefs": {"model": "greyscale"},
        "channels": [
            {
                "label": "intensity",
                "color": "FFFFFF",
                "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                "active": True,
            }
        ],
    }


def _create_empty_store_3d(
    image,
    ngff_root: Path,
    *,
    depth: int,
    z_scale: float,
) -> list[zarr.Array]:
    if ngff_root.exists():
        shutil.rmtree(ngff_root)

    level_shapes = _level_shapes(int(image.height), int(image.width))
    ngff_root.parent.mkdir(parents=True, exist_ok=True)
    zarr_root = zarr.open_group(str(ngff_root), mode="w", zarr_format=2)
    arrays: list[zarr.Array] = []
    for level_idx, (height, width) in enumerate(level_shapes):
        arrays.append(
            zarr_root.create_array(
                str(level_idx),
                shape=(1, depth, height, width),
                chunks=(
                    1,
                    1,
                    min(NGFF_CHUNK_SIZE, height),
                    min(NGFF_CHUNK_SIZE, width),
                ),
                dtype=np.uint8,
                compressor=NGFF_COMPRESSOR,
                overwrite=True,
                fill_value=0,
            )
        )
    _write_multiscale_metadata_3d(image, zarr_root, level_shapes, z_scale)
    return arrays


def _volume_z_scale(image) -> float:
    """Physical z anisotropy (effective z spacing / xy spacing), default 1.0."""

    voxel = (image.volume_metadata or {}).get("voxel_size_nm") or [None, None, None]
    z_nm = voxel[0] if len(voxel) > 0 else None
    x_nm = voxel[2] if len(voxel) > 2 else None
    try:
        if z_nm and x_nm and float(x_nm) > 0:
            return float(z_nm) / float(x_nm)
    except (TypeError, ValueError):
        pass
    return 1.0


def _regenerate_ngff_for_volume(
    image,
    source_path: Path,
    ngff_root: Path,
    attrs_path: Path,
) -> Path:
    """Build a 3D [c, z, y, x] NGFF store from the canonical OME-TIFF volume."""

    from .volume_readers import read_volume_source

    depth = int(image.stored_depth or 0)
    if depth < 1:
        raise RuntimeError(
            f"Cannot generate volume NGFF for image {image.id}: stored_depth is unset"
        )

    z_scale = _volume_z_scale(image)
    total_start = time.time()
    logger.info(
        "Image %s: regenerating 3D NGFF from canonical volume %s (depth=%d, z_scale=%.3f)",
        image.id,
        source_path,
        depth,
        z_scale,
    )
    arrays = _create_empty_store_3d(image, ngff_root, depth=depth, z_scale=z_scale)

    with read_volume_source(source_path) as source:
        for z in range(depth):
            plane = np.asarray(source.read_plane(z), dtype=np.uint8)
            arrays[0][0, z] = plane

    for level_idx in range(1, len(arrays)):
        finer = arrays[level_idx - 1]
        coarser = arrays[level_idx]
        target_height = int(coarser.shape[2])
        target_width = int(coarser.shape[3])
        for z in range(depth):
            source_plane = np.asarray(finer[0, z], dtype=np.uint8)
            coarser[0, z] = _downsample_region(
                source_plane,
                target_height=target_height,
                target_width=target_width,
            )

    if not attrs_path.exists():
        raise RuntimeError(
            f"Volume NGFF generation completed but .zattrs not found at {attrs_path}"
        )
    logger.info(
        "Image %s: 3D NGFF regeneration finished in %.2fs",
        image.id,
        time.time() - total_start,
    )
    return ngff_root


def regenerate_ngff_for_image(image) -> Path:
    """
    Force regeneration of the NGFF zarr store for an image.
    """
    total_start = time.time()
    ngff_root, attrs_path = get_ngff_paths(image)
    source_path = get_file_absolute_path(image)
    if source_path is None:
        raise RuntimeError(f"Cannot generate NGFF for image {image.id}: missing file path")
    if image.has_stored_z_stack:
        return _regenerate_ngff_for_volume(image, source_path, ngff_root, attrs_path)
    if source_path.suffix.lower() != ".png":
        raise RuntimeError(
            "Cannot generate NGFF before TIFF encoding completes for image "
            f"{image.id}: current source is {source_path}"
        )

    logger.info(
        "Image %s: regenerating NGFF from canonical PNG %s",
        image.id,
        source_path,
    )
    store_start = time.time()
    arrays = _create_empty_store(image, ngff_root)
    logger.info(
        "Image %s: initialized NGFF store with %d levels in %.2fs",
        image.id,
        len(arrays),
        time.time() - store_start,
    )

    _write_level0_from_png(
        image,
        source_path=source_path,
        level0_array=arrays[0],
    )
    for level_idx in range(1, len(arrays)):
        _write_downsampled_level(
            image,
            level_idx=level_idx,
            child_array=arrays[level_idx - 1],
            parent_array=arrays[level_idx],
        )

    if not attrs_path.exists():
        raise RuntimeError(
            f"NGFF generation completed but .zattrs not found at {attrs_path}"
        )
    logger.info(
        "Image %s: NGFF regeneration finished in %.2fs",
        image.id,
        time.time() - total_start,
    )
    return ngff_root


def ensure_ngff_for_image(image) -> Path:
    """
    Ensure an NGFF zarr store exists for an image.

    Single-node: the store is either already on this machine's disk or it has to
    be built here. There is no shared "master" storage root to pull it from.
    """
    ngff_root, _ = get_ngff_paths(image)
    if _is_valid_ngff_store(ngff_root):
        return ngff_root
    return regenerate_ngff_for_image(image)

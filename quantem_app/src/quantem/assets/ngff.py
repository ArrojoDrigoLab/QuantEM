"""
OME-NGFF (OME-Zarr) pyramid builder.

A pyramid is written into an **immutable generation directory**,
``storage/data/tmp/ngff/<image_id>.zarr/gen-<hex>/``, and becomes live by one
database ``UPDATE`` in :mod:`quantem.assets.pyramid_authority`. Nothing in this
module renames, moves or overwrites a directory a reader might be inside, and
nothing here decides whether a store may be read -- that is the authority's
single job, and this module has no opinion to add.

Two properties of the write are load-bearing and are checked before a
generation is sealed:

* **dense chunks** (``write_empty_chunks=True``). zarr elides an all-fill chunk
  by default, so a genuinely blank EM tile is indistinguishable on disk from a
  chunk that was never written -- which makes both "count the chunk files" and
  the strict store unsound. MEASURED cost on a 4096^2 plane with one all-fill
  chunk of 16: 880 extra bytes on a 15.7 MB store, and no write-time cost.
* **an exact chunk count**: ``chunk_count == prod(ceil(shape / chunks))`` per
  level, verified against the files actually on disk. Sound only because the
  writes are dense.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc
from PIL import Image

from .canonical_decode import DECODER_VERSION, CanonicalPlane
from .pyramid_authority import (
    BuildTicket,
    Unavailable,
    asset_generation_dir,
    discard_generation,
    publish,
    release_owner_lock,
    request_build,
    seal_generation,
    write_owner_for_ticket,
)

logger = logging.getLogger(__name__)

# Disable decompression bomb checks for large microscopy imagery.
Image.MAX_IMAGE_PIXELS = None

NGFF_CHUNK_SIZE = 1024  # 1024^2 chunks: ~16x fewer files than 256 -> much faster NGFF writes (esp. Windows small-file I/O)

# Blosc settings for the pyramid. These stores are uint8 grayscale, and they are
# a rebuildable cache under TMP_DIR, not an archive -- so the trade is write
# latency against a little disk, and byte-shuffling buys nothing on a
# single-byte dtype. MEASURED on the 475 MP EM plane below (whole-level writes,
# 1024^2 chunks, 28-core workstation):
#
#   zstd clevel=5 shuffle=BITSHUFFLE   3.74 s   343 MB   (the previous setting)
#   zstd clevel=3 shuffle=NOSHUFFLE    3.72 s   225 MB
#   zstd clevel=1 shuffle=NOSHUFFLE    3.10 s   254 MB
#   lz4  clevel=5 shuffle=NOSHUFFLE    3.17 s   421 MB
#
# clevel=1/NOSHUFFLE is both the fastest and 26 % smaller than what it replaces:
# BITSHUFFLE was actively hurting the ratio here by interleaving bit planes of
# data that has no multi-byte structure. Only the Blosc *parameters* change --
# the container is still Blosc, which every zarr reader in the stack (including
# the viewer's numcodecs.js) decodes without knowing or caring which cname,
# clevel or shuffle mode produced the block. Existing stores keep their own
# settings in their .zarray and stay readable.
NGFF_COMPRESSOR = Blosc(cname="zstd", clevel=1, shuffle=Blosc.NOSHUFFLE)
NGFF_THUMBNAIL_TARGET_MAX_SIDE = 256

#: Every level is written dense. See the module docstring: without this a
#: genuinely blank tile has no chunk file, which makes the strict store raise on
#: correct data and makes the chunk-count invariant unsound.
NGFF_ARRAY_CONFIG = {"write_empty_chunks": True}


def get_ngff_root_path(image) -> Path:
    """The directory holding this image's generations (not a store itself)."""

    asset = getattr(image, "asset", None)
    return asset_generation_dir((asset or image).id)


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
    level_shapes = _level_shapes(int(image.height), int(image.width))
    ngff_root.parent.mkdir(parents=True, exist_ok=True)
    # ``mode="a"``, not ``"w"``: a generation directory is brand new and is
    # never reused, so there is nothing to wipe -- and wiping it would delete
    # the ownership tag and the lock handle the sweeper reads to tell a live
    # build from debris.
    zarr_root = zarr.open_group(str(ngff_root), mode="a", zarr_format=2)
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
                config=NGFF_ARRAY_CONFIG,
            )
        )
    _write_multiscale_metadata(image, zarr_root, level_shapes)
    return arrays


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
    if source_region.dtype == np.uint8:
        return _box_mean_uint8(source_region)
    reshaped = source_region.reshape(target_height, 2, target_width, 2)
    return np.rint(reshaped.mean(axis=(1, 3))).astype(np.uint8)


def _box_mean_uint8(region: np.ndarray) -> np.ndarray:
    """2x2 box mean of an even-sized uint8 region, rounding halves to even.

    Exactly ``np.rint(region.reshape(h, 2, w, 2).mean(axis=(1, 3)))`` -- the
    expression this replaces -- but 3x quicker on a 475 MP plane (3.72 s ->
    1.19 s MEASURED), because it never materialises the float64 4-D view.

    The identity is not approximate. Four uint8 values sum to at most 1020, so
    the uint16 accumulator cannot overflow and the sum is exact; 1020 is well
    inside float32's exactly-representable integers and multiplying by 0.25 is
    a power-of-two scale, so the quotient is exact too; ``np.rint`` then sees
    the same value it saw before and rounds it the same way. Checked
    exhaustively over all 1021 possible sums, and on the real 475 MP EM plane.
    """

    totals = region[0::2, 0::2].astype(np.uint16)
    totals += region[1::2, 0::2]
    totals += region[0::2, 1::2]
    totals += region[1::2, 1::2]
    return np.rint(totals.astype(np.float32) * 0.25).astype(np.uint8)


def _downsample_plane(
    source_plane: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Half-resolution copy of ``source_plane``, tile by tile.

    The tiling and the per-tile arithmetic are exactly what the previous
    zarr-sourced implementation used -- same ``_chunk_bounds``, same clamping of
    the finer region, same ``_downsample_region`` -- so the output is
    bit-identical to the pyramid this replaced. What changed is only where the
    finer level is read from: the array still in memory, not a decompressed
    round-trip through the store that was just written.
    """

    out = np.empty((target_height, target_width), dtype=np.uint8)
    source_height, source_width = source_plane.shape
    for chunk_x, chunk_y in _iter_level_chunks(target_width, target_height):
        x_min, y_min, x_max, y_max = _chunk_bounds(
            chunk_x,
            chunk_y,
            width=target_width,
            height=target_height,
        )
        source_region = source_plane[
            y_min * 2 : min(source_height, y_max * 2),
            x_min * 2 : min(source_width, x_max * 2),
        ]
        out[y_min:y_max, x_min:x_max] = _downsample_region(
            source_region,
            target_height=y_max - y_min,
            target_width=x_max - x_min,
        )
    return out


def _write_levels_from_plane(
    image,
    plane: np.ndarray,
    arrays: list[zarr.Array],
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
    """Fill every pyramid level from one in-memory plane.

    Each level is written with a single assignment covering the whole level.
    That matters far more than it looks: zarr encodes the chunks of one
    assignment concurrently, so a whole-level write compresses on every core
    while the old chunk-at-a-time loop compressed on one. MEASURED on the
    475 MP plane, identical codec and chunking: 13.53 s per-chunk vs 3.74 s
    whole-level.
    """

    level_count = len(arrays)
    current = plane
    for level_idx, level_array in enumerate(arrays):
        level_start = time.time()
        if level_idx > 0:
            current = _downsample_plane(
                current,
                target_height=int(level_array.shape[1]),
                target_width=int(level_array.shape[2]),
            )
        downsample_elapsed = time.time() - level_start

        write_start = time.time()
        level_array[0] = current
        write_elapsed = time.time() - write_start

        logger.info(
            "Image %s: NGFF level %s %sx%s completed in %.2fs (downsample=%.2fs zarr_write=%.2fs)",
            image.id,
            level_idx,
            int(level_array.shape[2]),
            int(level_array.shape[1]),
            time.time() - level_start,
            downsample_elapsed,
            write_elapsed,
        )
        if progress_callback is not None:
            progress_callback(
                (level_idx + 1) / level_count,
                f"pyramid level {level_idx + 1}/{level_count}",
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
    level_shapes = _level_shapes(int(image.height), int(image.width))
    ngff_root.parent.mkdir(parents=True, exist_ok=True)
    # ``mode="a"``, not ``"w"``: a generation directory is brand new and is
    # never reused, so there is nothing to wipe -- and wiping it would delete
    # the ownership tag and the lock handle the sweeper reads to tell a live
    # build from debris.
    zarr_root = zarr.open_group(str(ngff_root), mode="a", zarr_format=2)
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
                config=NGFF_ARRAY_CONFIG,
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


# ---------------------------------------------------------------------------
# Building a generation
# ---------------------------------------------------------------------------


class PyramidBuildRefused(RuntimeError):
    """The authority declined to issue a ticket, or the build was superseded.

    Carries the :class:`~quantem.assets.pyramid_authority.Unavailable` so the
    caller can answer for each reason instead of re-deriving one.
    """

    def __init__(self, unavailable: Unavailable) -> None:
        super().__init__(f"{unavailable.reason.value}: {unavailable.detail}")
        self.unavailable = unavailable


def _expected_chunk_count(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    total = 1
    for extent, chunk in zip(shape, chunks, strict=True):
        total *= max(1, math.ceil(int(extent) / int(chunk)))
    return total


def _count_chunk_files(level_dir: Path) -> int:
    count = 0
    try:
        for entry in level_dir.iterdir():
            name = entry.name
            if name.startswith("."):
                continue
            if all(part.isdigit() for part in name.split(".") if part != ""):
                count += 1
    except OSError:
        return -1
    return count


def _manifest_for(root: Path, arrays: list[zarr.Array], *, source_fingerprint: str) -> dict:
    """The description a reader is entitled to trust, and its proof.

    Every level's chunk count is compared with the files actually on disk. With
    dense writes the two are equal by construction, so a mismatch means the
    build did not finish -- and a generation that cannot prove it finished is
    never published.
    """

    levels: list[dict[str, Any]] = []
    for index, array in enumerate(arrays):
        shape = tuple(int(value) for value in array.shape)
        chunks = tuple(int(value) for value in array.chunks)
        expected = _expected_chunk_count(shape, chunks)
        found = _count_chunk_files(root / str(index))
        if found != expected:
            raise RuntimeError(
                f"pyramid level {index} has {found} chunk files but its geometry "
                f"{shape}/{chunks} requires {expected}; the build did not finish"
            )
        levels.append(
            {
                "path": str(index),
                "shape": list(shape),
                "chunks": list(chunks),
                "chunk_count": expected,
            }
        )
    return {
        "levels": levels,
        "dense": True,
        "chunk_size": NGFF_CHUNK_SIZE,
        "decoder_version": DECODER_VERSION,
        "source_fingerprint": source_fingerprint,
        "built_at": time.time(),
    }


def build_pyramid(
    ticket: BuildTicket,
    image,
    plane,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Write every level of one generation and seal it. Returns the manifest.

    ``plane`` is a :class:`~quantem.assets.canonical_decode.CanonicalPlane` or
    the array out of one. Nothing here opens the source file: the only decode
    in the tree is :mod:`quantem.assets.canonical_decode`, and the builder
    takes its output rather than a path -- which is what leaves the
    ``.png``-suffix test and the four saturating decodes with nowhere to live.
    """

    fingerprint = ""
    if isinstance(plane, CanonicalPlane):
        fingerprint = plane.source_fingerprint
        plane = plane.array
    plane = np.asarray(plane)
    if plane.ndim != 2:
        raise ValueError(f"NGFF source plane must be 2D, got shape {plane.shape}")
    if plane.dtype != np.uint8:
        raise ValueError(f"NGFF source plane must be uint8, got {plane.dtype}")
    expected = (int(image.height), int(image.width))
    if plane.shape != expected:
        raise ValueError(
            f"NGFF source plane {plane.shape} does not match the recorded "
            f"geometry {expected} for image {image.id}"
        )

    started = time.time()
    arrays = _create_empty_store(image, ticket.root)
    # zarr's ``mode="w"`` may clear the directory, and the ownership tag is
    # what lets the sweeper tell a live build from debris, so it is written
    # again here -- before the first chunk, which is the property that matters.
    write_owner_for_ticket(ticket)
    logger.info(
        "Image %s: generation %s initialised with %d levels in %.2fs",
        image.id,
        ticket.generation_id,
        len(arrays),
        time.time() - started,
    )
    _write_levels_from_plane(image, plane, arrays, progress_callback=progress_callback)
    manifest = _manifest_for(ticket.root, arrays, source_fingerprint=fingerprint)
    seal_generation(ticket.root, manifest)
    logger.info(
        "Image %s: generation %s written and sealed in %.2fs",
        image.id,
        ticket.generation_id,
        time.time() - started,
    )
    return manifest


def build_volume_pyramid(ticket: BuildTicket, image, source_path: Path) -> dict:
    """Write a 4D [c, z, y, x] generation from the canonical OME-TIFF volume."""

    from .volume_readers import read_volume_source

    depth = int(image.stored_depth or 0)
    if depth < 1:
        raise RuntimeError(
            f"Cannot generate volume NGFF for image {image.id}: stored_depth is unset"
        )
    z_scale = _volume_z_scale(image)
    started = time.time()
    arrays = _create_empty_store_3d(image, ticket.root, depth=depth, z_scale=z_scale)
    write_owner_for_ticket(ticket)

    with read_volume_source(source_path) as source:
        for z in range(depth):
            arrays[0][0, z] = np.asarray(source.read_plane(z), dtype=np.uint8)

    for level_idx in range(1, len(arrays)):
        finer = arrays[level_idx - 1]
        coarser = arrays[level_idx]
        target_height = int(coarser.shape[2])
        target_width = int(coarser.shape[3])
        for z in range(depth):
            coarser[0, z] = _downsample_region(
                np.asarray(finer[0, z], dtype=np.uint8),
                target_height=target_height,
                target_width=target_width,
            )

    manifest = _manifest_for(ticket.root, arrays, source_fingerprint="")
    manifest["volume"] = True
    manifest["z_scale"] = z_scale
    seal_generation(ticket.root, manifest)
    logger.info(
        "Image %s: 3D generation %s written in %.2fs",
        image.id,
        ticket.generation_id,
        time.time() - started,
    )
    return manifest


def build_and_publish(
    image,
    plane=None,
    *,
    volume_source: Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    """Build a generation for ``image`` and make it live, or say why not.

    Returns the published generation's root. Raises
    :class:`PyramidBuildRefused` when the authority declines a ticket (a
    terminal import) or when the compare-and-swap finds the build stale --
    neither of which is an error the *user* caused, so a caller servicing a
    background job turns it into a successful no-op result rather than a job
    failure. See :func:`quantem.assets.tasks.ensure_ngff_for_asset_task`.
    """

    from .pyramid_authority import Reason

    asset = getattr(image, "asset", None) or image
    ticket = request_build(asset, decoder_version=DECODER_VERSION)
    if isinstance(ticket, Unavailable):
        raise PyramidBuildRefused(ticket)

    published = False
    try:
        if volume_source is not None:
            manifest = build_volume_pyramid(ticket, image, volume_source)
        else:
            manifest = build_pyramid(ticket, image, plane, progress_callback=progress_callback)
        published = publish(ticket, manifest)
    finally:
        if not published:
            discard_generation(ticket)
        else:
            release_owner_lock(ticket.root)
    if not published:
        raise PyramidBuildRefused(
            Unavailable(
                Reason.SUPERSEDED,
                "another attempt superseded this build before it could be published",
            )
        )
    return ticket.root


def regenerate_ngff_from_plane(
    image,
    plane,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    """Build and publish one generation from an already-decoded plane.

    Kept under its historical name because it is still the import path's verb.
    What changed is underneath: there is no build-in-a-sibling-and-rename
    dance, no ``.building``/``.superseded``/``withdrawn`` directory and no
    publish window at all. The generation is written under its final, immutable
    name and becomes live by one database ``UPDATE``.
    """

    return build_and_publish(image, plane, progress_callback=progress_callback)

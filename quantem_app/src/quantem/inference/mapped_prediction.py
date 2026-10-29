"""Windowed inference with disk-backed resampling and native probabilities.

OpenCV resizes into mapped destinations so its global interpolation coordinates
and rounding are identical to the in-memory path. Hann blending emits bands;
neither the padded image nor a full float probability field lives on the heap.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr
from numpy.typing import DTypeLike

from . import engine, out_of_core, resample, tiling

STREAMING_MIN_PIXELS = 64 * 1024 * 1024
STRIPE_BYTES = 16 * 1024 * 1024


class PyramidPlane:
    """Two-dimensional slice interface over an immutable canonical pyramid."""

    ndim = 2
    dtype = np.dtype(np.uint8)

    def __init__(self, array: zarr.Array) -> None:
        self.array = array
        self.shape = tuple(array.shape[-2:])

    def __getitem__(self, selection: tuple[slice, slice]) -> np.ndarray:
        return np.asarray(self.array[(0, *selection)], dtype=np.uint8)


def predict_native(
    model: engine.LoadedModel,
    source: Any,
    *,
    pixel_size_nm: float | None,
    overlap: float,
    on_progress: Callable[[float], None] | None = None,
    on_tile: tiling.TileCounter | None = None,
) -> resample.NativeProbabilityMap:
    """Return a temporary chunked native map, owned until its last reader exits."""
    from quantem.core.config import TMP_DIR

    root = TMP_DIR / "inference"
    root.mkdir(parents=True, exist_ok=True)
    context = resample.plan_resample(source.shape, pixel_size_nm, model.spec.canonical_nm)
    out_of_core._cleanup_stale_workspaces(root, prefix="prediction-", min_age=60)
    workspace = tempfile.TemporaryDirectory(prefix="prediction-", dir=root)
    directory = Path(workspace.name)
    (directory / "owner.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    maps: list[np.memmap] = []

    def mapped(name: str, shape: tuple[int, int], dtype: DTypeLike) -> np.memmap:
        # Anonymous/delete-on-close mappings do not strand full float planes
        # if the runner terminates a cancelled worker. Only the chunked output
        # directory needs dead-owner reclamation on the next prediction.
        with tempfile.TemporaryFile(prefix=name, dir=directory) as stream:
            array = np.memmap(stream, mode="w+", shape=shape, dtype=dtype)
        maps.append(array)
        return array

    height, width = context.native_shape
    mh, mw = context.model_shape
    rows = max(1, STRIPE_BYTES // max(width, 1))
    native = None
    try:
        # Resampling can make the model grid much larger than the source.
        # Reject before filling the data volume with partial scratch planes.
        native_pixels, model_pixels = height * width, mh * mw
        scratch_bytes = (
            native_pixels if context.is_identity else 6 * native_pixels + 5 * model_pixels
        )
        required = int(scratch_bytes * 1.1) + 64 * 1024**2
        available = shutil.disk_usage(directory).free
        if required > available * 0.9:
            raise ValueError(
                f"This image needs about {required / 1024**3:.1f} GiB of temporary disk space "
                f"for inference; {available / 1024**3:.1f} GiB is available in the data folder."
            )
        scaled = source
        if not context.is_identity:
            # Preserve exact INTER_AREA/INTER_LINEAR arithmetic, including
            # non-integer factors, without allocating resident full planes.
            original = mapped("source.u8", (height, width), np.uint8)
            for y in range(0, height, rows):
                original[y : y + rows] = source[y : y + rows, :]
            scaled = mapped("scaled.u8", (mh, mw), np.uint8)
            cv2.resize(
                original,
                (mw, mh),
                dst=scaled,
                interpolation=cv2.INTER_LINEAR if context.upsamples else cv2.INTER_AREA,
            )

        native = zarr.open_array(
            str(directory / "native.zarr"),
            mode="w",
            zarr_format=2,
            shape=(height, width),
            chunks=(1024, 1024),
            dtype="u1",
        )
        probability = None if context.is_identity else mapped("model.f32", (mh, mw), np.float32)
        plan = tiling.plan_tiles(
            tiling.padded_shape((mh, mw), model.spec.tile_size, model.spec.patch_size),
            model.spec.tile_size,
            overlap,
        )

        def read_tile(tile: tiling.Tile) -> np.ndarray:
            ys, xs = tile.slices
            tile_image = np.zeros((model.spec.tile_size, model.spec.tile_size), dtype=np.uint8)
            crop = scaled[ys.start : min(ys.stop, mh), xs.start : min(xs.stop, mw)]
            tile_image[: crop.shape[0], : crop.shape[1]] = crop
            return tile_image

        def write_band(y: int, band: np.ndarray) -> None:
            stop = min(mh, y + len(band))
            if stop <= y:
                return
            values = band[: stop - y, :mw]
            if probability is None:
                native[y:stop, :] = resample.quantize_probability(values)
            else:
                probability[y:stop, :] = values

        engine.predict_region_streaming(
            model, plan, read_tile, write_band, on_progress=on_progress, on_tile=on_tile
        )
        if probability is not None:
            restored = mapped("native.f32", (height, width), np.float32)
            cv2.resize(
                probability,
                (width, height),
                dst=restored,
                interpolation=resample.probability_interpolation(context),
            )
            for y in range(0, height, max(1, rows // 4)):
                stop = min(height, y + max(1, rows // 4))
                native[y:stop, :] = resample.quantize_probability(restored[y:stop])
        result = resample.NativeProbabilityMap(
            data=native,
            interpolation=resample.interpolation_name(resample.probability_interpolation(context)),
            back_factor=float(context.back_factor),
        )
    except BaseException:
        for array in maps:
            array._mmap.close()
        workspace.cleanup()
        raise
    for array in maps:
        array._mmap.close()
    # The returned Zarr array is held by both the segmenter and InferenceResult.
    # Cleanup must wait for both, rather than invalidating a previous result
    # when the same segmenter starts its next prediction.
    weakref.finalize(native, workspace.cleanup)
    return result

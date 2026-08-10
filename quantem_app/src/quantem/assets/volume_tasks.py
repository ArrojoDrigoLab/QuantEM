"""Canonical 3D-volume encoding: decimate z, window to 8-bit, write OME-TIFF.

This is the 3D analogue of ``assets/tasks.py:encode_asset_full_to_png``. A
staged source volume (any format understood by :mod:`quantem.assets.volume_readers`) is:

  1. opened and probed for geometry + voxel metadata,
  2. reduced along z to a representative subset of planes (see
     :func:`select_z_planes`) -- xy planes are always kept in full,
  3. windowed to 8-bit grayscale using a single global intensity window so
     brightness is consistent across z, and
  4. written as a single multi-page **OME-TIFF** (the canonical archival file),
     with effective voxel sizes recorded in the OME-XML.

The kept source-plane indices, the resolved sampling spec and the source
provenance are written back onto the canonical :class:`Rendition` row, and the
effective voxel sizes onto :class:`Asset` (``pixel_size_nm`` /
``pixel_size_nm_z``), so the viewer can label slices by true physical depth, the
NGFF step can scale z correctly, and analyses have real units.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from quantem.core.config import DATA_DIR, IMAGES_DIR
from quantem.core.local_storage import normalize_stored_path_value

from .asset_openable import get_asset_openable
from .models import Asset, Rendition
from .preprocess_status import set_stage
from .utils import _native_max_for_bit_depth
from .volume_readers import read_volume_source

logger = logging.getLogger(__name__)

# Default number of z-planes to keep when no explicit sampling is requested.
DEFAULT_TARGET_PLANES = 25


# --------------------------------------------------------------------------- #
# Z sampling
# --------------------------------------------------------------------------- #
def select_z_planes(
    source_depth: int, z_sampling: dict | None
) -> tuple[list[int], dict[str, Any]]:
    """Choose which source z-planes to keep and return ``(indices, resolved)``.

    Modes:
      * ``count``   (default): keep ~``target`` evenly-spaced planes by deriving
        a uniform ``stride = round(source_depth / target)``. This yields uniform
        spacing (so a single z-scale is exact) and ~target planes.
      * ``all`` / ``full``: keep every plane.
      * ``stride``: keep every ``stride``-th plane.
      * ``explicit``: keep exactly the listed plane indices.

    ``resolved`` echoes the effective spec including the derived ``stride`` and
    the ``source_depth`` / ``stored_depth`` so it can be persisted verbatim.
    """

    source_depth = int(source_depth)
    if source_depth <= 0:
        return [0], {"mode": "all", "source_depth": 0, "stored_depth": 1, "stride": 1}

    spec = dict(z_sampling or {})
    mode = str(spec.get("mode") or "count").lower()

    if mode in ("all", "full"):
        indices = list(range(source_depth))
        stride = 1
        resolved = {"mode": "all"}
    elif mode == "stride":
        stride = max(1, int(spec.get("stride") or 1))
        indices = list(range(0, source_depth, stride))
        resolved = {"mode": "stride", "stride": stride}
    elif mode == "explicit":
        raw = spec.get("planes") or []
        indices = sorted({int(p) for p in raw if 0 <= int(p) < source_depth})
        if not indices:
            indices = [0]
        stride = None
        resolved = {"mode": "explicit"}
    else:  # count (default)
        target = int(spec.get("target") or DEFAULT_TARGET_PLANES)
        target = max(1, target)
        stride = max(1, round(source_depth / target))
        indices = list(range(0, source_depth, stride))
        resolved = {"mode": "count", "target": target, "stride": stride}

    resolved.update(
        {
            "source_depth": source_depth,
            "stored_depth": len(indices),
        }
    )
    if stride is not None and "stride" not in resolved:
        resolved["stride"] = stride
    return indices, resolved


def _effective_z_step_factor(indices: list[int]) -> float:
    """Mean gap between kept plane indices (the z decimation factor)."""

    if len(indices) < 2:
        return 1.0
    diffs = np.diff(np.asarray(indices, dtype=np.float64))
    return float(np.mean(diffs)) if diffs.size else 1.0


# --------------------------------------------------------------------------- #
# 8-bit conversion (plain native bit-depth, no windowing or stretching)
# --------------------------------------------------------------------------- #
def _native_window(bit_depth: int) -> tuple[float, float]:
    """Native-range window [0, 2**bit_depth - 1] for plain 8-bit conversion."""
    return 0.0, _native_max_for_bit_depth(bit_depth)


def _apply_window_uint8(plane: np.ndarray, low: float, high: float) -> np.ndarray:
    """Scale a native-dtype plane into 8-bit grayscale with a fixed window."""

    if high <= low:
        return np.zeros(plane.shape, dtype=np.uint8)
    scaled = (np.asarray(plane, dtype=np.float32) - low) * (255.0 / (high - low))
    return np.nan_to_num(np.clip(scaled, 0, 255), nan=0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Encoding entry points
# --------------------------------------------------------------------------- #
def encode_asset_volume_to_ome_tiff_task(asset_id: str) -> None:
    encode_asset_volume_to_ome_tiff(asset_id)


def encode_asset_volume_to_ome_tiff(asset_id: str) -> None:
    """Convert an asset source volume rendition into canonical 8-bit OME-TIFF."""

    try:
        asset = Asset.objects.get(id=asset_id)
    except Asset.DoesNotExist:
        logger.warning("Asset %s not found for volume encode; skipping.", asset_id)
        return
    if asset.preprocess_stage == "CANCELLED":
        return

    set_stage(asset, "ENCODING", progress=0.0, error="")

    try:
        openable = get_asset_openable(asset)
    except Exception as exc:
        set_stage(asset, "FAILED", progress=0.0, error=f"Missing source volume: {exc}")
        return

    source_path = openable.path
    if not source_path or not Path(source_path).exists():
        set_stage(asset, "FAILED", progress=0.0, error="Missing source volume for upload.")
        return

    stem = (asset.original_filename or asset.display_name or str(asset.id)).split(".")[0]
    target_path = IMAGES_DIR / str(asset.id) / f"{stem}.ome.tif"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with read_volume_source(source_path) as source:
            meta = source.metadata
            indices, resolved = select_z_planes(meta.depth, openable.z_sampling)
            stored_depth = len(indices)
            height, width = meta.height, meta.width
            low, high = _native_window(meta.bit_depth)

            set_stage(asset, "ENCODING", progress=20.0, error="")
            written = {"count": 0}

            def _plane_iter():
                for z in indices:
                    plane = source.read_plane(z)
                    yield _apply_window_uint8(plane, low, high)
                    written["count"] += 1
                    if stored_depth and written["count"] % 5 == 0:
                        progress = 20.0 + 70.0 * written["count"] / stored_depth
                        set_stage(asset, "ENCODING", progress=progress, error="")

            effective_voxel = _effective_voxel_nm(meta.voxel_size_nm, indices)
            ome_metadata = _ome_metadata(effective_voxel)

            with tifffile.TiffWriter(str(target_path), bigtiff=True, ome=True) as tif:
                tif.write(
                    _plane_iter(),
                    shape=(stored_depth, height, width),
                    dtype=np.uint8,
                    photometric="minisblack",
                    compression="zlib",
                    metadata=ome_metadata,
                )

            volume_metadata = {
                "source": meta.as_dict(),
                "voxel_size_nm": list(effective_voxel),
                "stored_axes": "ZYX",
                "native_range": [low, high],
                "z_sampling": resolved,
            }
    except Exception as exc:
        logger.error("Volume encode failed for asset %s: %s", asset_id, exc, exc_info=True)
        set_stage(asset, "FAILED", progress=0.0, error=f"Volume encode failed: {exc}")
        return

    _remove_staged_source(source_path)

    asset.logical_width = int(width)
    asset.logical_height = int(height)
    asset.logical_depth = int(meta.depth)
    asset.channels = 1
    asset.bit_depth = 8
    # Fill the numeric pixel size from the source voxel metadata, but never
    # clobber a value the user typed in at upload time.
    voxel_z, _voxel_y, voxel_x = effective_voxel
    if asset.pixel_size_nm is None and voxel_x:
        asset.pixel_size_nm = float(voxel_x)
    if asset.pixel_size_nm_z is None and voxel_z:
        asset.pixel_size_nm_z = float(voxel_z)
    asset.save(
        update_fields=[
            "logical_width",
            "logical_height",
            "logical_depth",
            "channels",
            "bit_depth",
            "pixel_size_nm",
            "pixel_size_nm_z",
            "updated_at",
        ]
    )
    Rendition.objects.filter(id=openable.rendition.id).update(
        storage_root="DATA_DIR",
        stored_path=normalize_stored_path_value(target_path, relative_to=DATA_DIR),
        path_exists=target_path.exists(),
        is_directory=False,
        stored_width=int(width),
        stored_height=int(height),
        stored_depth=int(stored_depth),
        stored_channels=1,
        stored_bit_depth=8,
        z_plane_indices=[int(z) for z in indices],
        metadata={**dict(openable.rendition.metadata or {}), "volume_metadata": volume_metadata},
    )
    set_stage(asset, "ENCODING", progress=95.0, error="")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _effective_voxel_nm(source_voxel_nm, indices: list[int]):
    z, y, x = source_voxel_nm
    if z is not None:
        z = float(z) * _effective_z_step_factor(indices)
    return (z, y, x)


def _ome_metadata(effective_voxel_nm) -> dict[str, Any]:
    z, y, x = effective_voxel_nm
    metadata: dict[str, Any] = {"axes": "ZYX"}
    # OME PhysicalSize is conventionally in micrometres.
    if x is not None:
        metadata["PhysicalSizeX"] = float(x) / 1000.0
        metadata["PhysicalSizeXUnit"] = "µm"
    if y is not None:
        metadata["PhysicalSizeY"] = float(y) / 1000.0
        metadata["PhysicalSizeYUnit"] = "µm"
    if z is not None:
        metadata["PhysicalSizeZ"] = float(z) / 1000.0
        metadata["PhysicalSizeZUnit"] = "µm"
    return metadata


def _remove_staged_source(source_path: Path) -> None:
    try:
        source_path = Path(source_path)
        if source_path.is_dir():
            shutil.rmtree(source_path, ignore_errors=True)
        elif source_path.exists():
            source_path.unlink()
    except OSError as exc:
        logger.warning("Failed to remove staged volume source %s: %s", source_path, exc)

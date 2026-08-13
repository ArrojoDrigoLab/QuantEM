"""Persistence and editing for global-mode binary segmentation masks."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable

import numpy as np
from PIL import Image
from shapely.geometry.base import BaseGeometry

from quantem.assets.canonical_decode import decode_canonical_array
from quantem.core.config import GLOBAL_MASKS_DIR
from quantem.core.local_storage import storage_path, storage_relpath_for_path
from quantem.seg_core.rasterize import paint_rings
from quantem.segmentation.geometry import extract_polygons
from quantem.segmentation.models import GlobalMask, ImageSegmentation, SegmentObject


def is_global_segmentation(segmentation: ImageSegmentation) -> bool:
    return segmentation.segmentation_type.measurement_mode == "global"


def _dimensions(segmentation: ImageSegmentation) -> tuple[int, int]:
    asset = segmentation.asset
    width = int(getattr(asset, "logical_width", 0) or 0)
    height = int(getattr(asset, "logical_height", 0) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("The image dimensions are unavailable; a global mask cannot be stored.")
    return height, width


def _mask_path(segmentation: ImageSegmentation):
    return GLOBAL_MASKS_DIR / str(segmentation.id) / "mask.png"


def _rasterize_geometries(
    geometries: Iterable[BaseGeometry], *, height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for geometry in geometries:
        for polygon in extract_polygons(geometry):
            rings = [
                np.asarray(polygon.exterior.coords, dtype=np.float64),
                *(np.asarray(ring.coords, dtype=np.float64) for ring in polygon.interiors),
            ]
            paint_rings(mask, rings, 1, x0=0, y0=0)
    return mask.astype(bool, copy=False)


def save_global_mask(
    segmentation: ImageSegmentation,
    mask: np.ndarray,
    *,
    source: str,
    metadata: dict | None = None,
) -> GlobalMask:
    """Atomically replace the one binary mask belonging to ``segmentation``."""
    if not is_global_segmentation(segmentation):
        raise ValueError("Only global-mode segmentations can store a global mask.")
    height, width = _dimensions(segmentation)
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != (height, width):
        raise ValueError(
            f"Global mask shape {binary.shape} does not match image shape {(height, width)}."
        )

    path = _mask_path(segmentation)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        Image.fromarray(binary.astype(np.uint8) * 255, mode="L").save(
            temporary, format="PNG", optimize=True
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    record, _created = GlobalMask.objects.update_or_create(
        segmentation=segmentation,
        defaults={
            "file_path": storage_relpath_for_path(path),
            "width": width,
            "height": height,
            "foreground_pixels": int(np.count_nonzero(binary)),
            "source": str(source or ""),
            "metadata": dict(metadata or {}),
        },
    )
    return record


def save_global_mask_from_geometries(
    segmentation: ImageSegmentation,
    geometries: Iterable[BaseGeometry],
    *,
    source: str = "manual",
    metadata: dict | None = None,
) -> GlobalMask:
    """Replace a global mask with the union of polygonal ``geometries``.

    Analysis-mask objects use this when an object or the page is saved. Keeping
    the union here means the analysis and overlay paths continue to read the
    same compact binary mask; they never need to know how many editable objects
    produced it.
    """
    height, width = _dimensions(segmentation)
    mask = _rasterize_geometries(geometries, height=height, width=width)
    return save_global_mask(
        segmentation,
        mask,
        source=source,
        metadata=metadata,
    )


def load_global_mask(
    segmentation: ImageSegmentation,
    *,
    legacy_fallback: bool = True,
) -> np.ndarray:
    """Load the binary mask, safely reading legacy confirmed object rows.

    The fallback never writes or deletes those rows. Older installations can
    therefore reopen their confirmed global results without a lossy migration.
    """
    height, width = _dimensions(segmentation)
    try:
        record = segmentation.global_mask
    except GlobalMask.DoesNotExist:
        record = None
    if record is not None:
        data = decode_canonical_array(storage_path(record.file_path)) > 0
        if data.shape != (height, width):
            raise ValueError(
                f"Stored global mask shape {data.shape} does not match image shape "
                f"{(height, width)}."
            )
        return data
    if not legacy_fallback:
        return np.zeros((height, width), dtype=bool)
    rows = SegmentObject.objects.filter(
        segmentation=segmentation,
        superseded_at__isnull=True,
        label_state="CONFIRMED",
    )
    return _rasterize_geometries(
        (row.geometry for row in rows.iterator()), height=height, width=width
    )


def patch_global_mask(
    segmentation: ImageSegmentation,
    *,
    include: Iterable[BaseGeometry] = (),
    exclude: Iterable[BaseGeometry] = (),
    source: str = "manual",
) -> GlobalMask:
    """Apply ``union(include) - union(exclude)`` to the stored mask."""
    height, width = _dimensions(segmentation)
    result = load_global_mask(segmentation)
    current = GlobalMask.objects.filter(segmentation=segmentation).first()
    metadata = (
        dict(current.metadata) if current is not None and isinstance(current.metadata, dict) else {}
    )
    effective_source = source
    if source.startswith("manual") and current is not None:
        # A proofread model result is still a model result. Preserve the pack
        # and adapter attached to the pixels so unlocking and re-finalizing the
        # image cannot mislabel the result as purely manual after its original
        # probability map has been reclaimed.
        metadata["manually_edited"] = True
        effective_source = current.source or source
    include_mask = _rasterize_geometries(include, height=height, width=width)
    exclude_mask = _rasterize_geometries(exclude, height=height, width=width)
    result |= include_mask
    result &= ~exclude_mask
    return save_global_mask(
        segmentation,
        result,
        source=effective_source,
        metadata=metadata,
    )

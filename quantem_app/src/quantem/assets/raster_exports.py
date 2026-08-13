"""User-requested 8-bit PNG exports for images and segmentations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from io import BytesIO

import numpy as np
from PIL import Image
from shapely.geometry.base import BaseGeometry

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.task_utils import load_image_array
from quantem.seg_core.rasterize import paint_rings
from quantem.segmentation.geometry import extract_polygons
from quantem.segmentation.global_masks import load_global_mask
from quantem.segmentation.models import AnalysisMaskObject, ImageSegmentation, SegmentObject
from quantem.segmentation.type_definitions import ANALYSIS_MASK

LABEL_COUNT = 255


def export_label(index: int) -> int:
    """Return the 8-bit object label for a zero-based object index."""
    return (int(index) % LABEL_COUNT) + 1


def png_bytes(array: np.ndarray) -> bytes:
    """Encode one 2-D uint8 array as a grayscale PNG."""
    plane = np.ascontiguousarray(array, dtype=np.uint8)
    if plane.ndim != 2:
        raise ValueError(f"An export must be a 2-D plane, not shape {plane.shape}.")
    output = BytesIO()
    Image.fromarray(plane, mode="L").save(output, format="PNG", optimize=True)
    return output.getvalue()


def original_image_export(asset) -> np.ndarray:
    """The canonical, display-equivalent 8-bit EM plane."""
    plane, _elapsed = load_image_array(get_asset_openable(asset))
    return np.ascontiguousarray(plane, dtype=np.uint8)


def _geometry_rings(geometry: BaseGeometry | None):
    for polygon in extract_polygons(geometry):
        rings = [
            np.asarray(polygon.exterior.coords, dtype=np.float64),
            *(np.asarray(interior.coords, dtype=np.float64) for interior in polygon.interiors),
        ]
        if rings[0].shape[0] >= 3:
            yield polygon.bounds, rings


def labeled_geometry_export(
    geometries: Iterable[BaseGeometry | None],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize ordered geometries into labels 1..255, with later ones winning.

    Each polygon is painted only inside its own bounding box. Besides keeping
    holes transparent to earlier objects, this avoids clearing an image-sized
    scratch array once per object on large EM montages.
    """
    labels = np.zeros((height, width), dtype=np.uint8)
    for index, geometry in enumerate(geometries):
        label = export_label(index)
        for bounds, rings in _geometry_rings(geometry):
            x0 = max(0, math.floor(bounds[0]))
            y0 = max(0, math.floor(bounds[1]))
            x1 = min(width, math.ceil(bounds[2]))
            y1 = min(height, math.ceil(bounds[3]))
            if x1 <= x0 or y1 <= y0:
                continue
            scratch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            paint_rings(scratch, rings, 1, x0=x0, y0=y0)
            window = labels[y0:y1, x0:x1]
            window[scratch != 0] = label
    return labels


def _dimensions(segmentation: ImageSegmentation) -> tuple[int, int]:
    height = int(segmentation.asset.logical_height or 0)
    width = int(segmentation.asset.logical_width or 0)
    if height <= 0 or width <= 0:
        raise ValueError("The image dimensions are unavailable.")
    return height, width


def _analysis_mask_export(segmentation: ImageSegmentation) -> np.ndarray:
    height, width = _dimensions(segmentation)
    objects = AnalysisMaskObject.objects.filter(segmentation=segmentation).order_by(
        "sort_order", "created_at", "id"
    )
    if objects.exists():
        # TODO(export-v2): revisit overlapping Analysis Mask exports. An 8-bit
        # single-channel raster can encode only one object per pixel, so the
        # current explicit rule is last object in list order wins.
        return labeled_geometry_export(
            (obj.geometry for obj in objects.only("geometry_wkb").iterator()),
            height=height,
            width=width,
        )

    # Masks created before named objects existed still have useful pixels. They
    # have no separable object records, so represent the aggregate as object 1.
    return load_global_mask(segmentation).astype(np.uint8)


def _object_segmentation_export(segmentation: ImageSegmentation) -> np.ndarray:
    height, width = _dimensions(segmentation)
    objects = (
        SegmentObject.objects.filter(
            segmentation=segmentation,
            superseded_at__isnull=True,
        )
        .exclude(label_state="EXCLUDED")
        .order_by("created_at", "id")
        .only("geometry_wkb")
    )
    return labeled_geometry_export(
        (obj.geometry for obj in objects.iterator()),
        height=height,
        width=width,
    )


def segmentation_export(segmentation: ImageSegmentation) -> np.ndarray:
    """Export the current pixels of any supported segmentation as uint8."""
    if segmentation.segmentation_type.internal_name == ANALYSIS_MASK.internal_name:
        return _analysis_mask_export(segmentation)
    if (
        segmentation.segmentation_type.measurement_mode
        == segmentation.segmentation_type.MEASUREMENT_MODE_OBJECTS
    ):
        return _object_segmentation_export(segmentation)
    return load_global_mask(segmentation).astype(np.uint8) * np.uint8(255)

"""
Feature extraction helpers that avoid full-image rasterization.

The polygon is measured **as drawn**. Its coordinates are carried to the
rasteriser at full precision and the mask that comes back is the set of pixels
whose centres are inside it -- the convention
:mod:`quantem.seg_core.rasterize` defines and every mask in this app now
follows, so a shape a person drew and the same shape found by a model measure
the same.

Two things used to sit between the outline and its measurement:

* ``cv2.fillPoly``, which painted both boundaries of every span, so a polygon
  spanning *s* pixels covered *s+1* -- ``+44%`` of the area of a 5 px object,
  ``+21%`` of a 10 px one;
* a Douglas-Peucker pass at 1 px tolerance, added to speed up that
  rasterisation, which moved the boundary by up to a pixel *before* it was
  measured -- ``-5.6%`` on a 4 px-radius blob, ``-1.4%`` on a 15 px one, and
  enough on a model object's contour that re-measuring it through this module
  no longer reproduced the ``region.area`` it was extracted with.

The simplification is gone. It was a real cost as well as a real error: on a
30 px blob it took ~540 us to save ~20 us of fill.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

import cv2
import numpy as np
from shapely.geometry import Polygon

from quantem.seg_core.rasterize import fill_rings

from .geometry import compute_regionprops_features
from .intensity import compute_intensity_features


def _ring_to_array(ring: Iterable[tuple[float, ...]]) -> np.ndarray | None:
    coords: list[tuple[float, float]] = []
    for coord in ring:
        if len(coord) < 2:
            continue
        coords.append((float(coord[0]), float(coord[1])))
    if len(coords) < 3:
        return None
    # float64: the fill decides on half-pixel boundaries, and float32 stops
    # representing those exactly a good deal earlier than a large image's
    # coordinates run out.
    return np.asarray(coords, dtype=np.float64)


def _extract_polygon_rings(polygon: Polygon) -> list[np.ndarray]:
    try:
        rings = [polygon.exterior.coords, *(ring.coords for ring in polygon.interiors)]
    except Exception:
        return []

    ring_arrays: list[np.ndarray] = []
    for ring in rings:
        ring_array = _ring_to_array(ring)
        if ring_array is not None:
            ring_arrays.append(ring_array)
    return ring_arrays


def _compute_local_bbox(
    image_shape: tuple[int, int],
    ring_arrays: list[np.ndarray],
    offset_x: int,
    offset_y: int,
    pad: int,
    bbox_polygon: Polygon | None = None,
) -> tuple[int, int, int, int] | None:
    if bbox_polygon is not None:
        min_x, min_y, max_x, max_y = bbox_polygon.bounds
    else:
        if not ring_arrays:
            return None
        min_x = min(float(np.min(r[:, 0])) for r in ring_arrays)
        max_x = max(float(np.max(r[:, 0])) for r in ring_arrays)
        min_y = min(float(np.min(r[:, 1])) for r in ring_arrays)
        max_y = max(float(np.max(r[:, 1])) for r in ring_arrays)

    min_x -= offset_x
    max_x -= offset_x
    min_y -= offset_y
    max_y -= offset_y

    height, width = image_shape
    if height <= 0 or width <= 0:
        return None

    x0 = max(int(math.floor(min_x)) - pad, 0)
    y0 = max(int(math.floor(min_y)) - pad, 0)
    x1 = min(int(math.ceil(max_x)) + pad, width - 1)
    y1 = min(int(math.ceil(max_y)) + pad, height - 1)

    if x0 > x1 or y0 > y1:
        return None

    return x0, y0, x1, y1


def _rasterize_rings_to_mask(
    ring_arrays: list[np.ndarray],
    mask_shape: tuple[int, int],
    offset_x: int,
    offset_y: int,
    x0: int,
    y0: int,
    *,
    use_downsample: bool = False,
) -> np.ndarray:
    """Rasterize polygon rings to a boolean mask, in the app's pixel convention.

    A pixel belongs to the shape when its **centre** does; see
    :mod:`quantem.seg_core.rasterize` for the convention and for why
    ``cv2.fillPoly`` -- which painted both boundaries of every span and so made
    a polygon spanning *s* pixels cover *s+1* -- could not be given it.

    Very large masks (>20k px on a side) are still rasterised at reduced
    resolution and upsampled, which quantises the area to the downsample factor
    squared. It only applies to objects wider than 20,000 pixels.

    Args:
        ring_arrays: List of polygon rings (exterior + holes), in image
            coordinates.
        mask_shape: Target mask shape (height, width)
        offset_x, offset_y: Image offset coordinates
        x0, y0: Bounding box top-left corner
        use_downsample: Whether to downsample for very large masks

    Returns:
        Boolean mask of shape mask_shape
    """
    if not ring_arrays:
        return np.zeros(mask_shape, dtype=bool)

    height, width = mask_shape
    actual_downsample = 1.0

    # Determine if we should downsample for very large masks
    if use_downsample and (width > 20000 or height > 20000):
        # Calculate downsample factor to bring largest dimension to ~20k
        max_dim = max(width, height)
        actual_downsample = max_dim / 20000.0
        # Round to nearest power of 2 for cleaner upsampling
        actual_downsample = 2 ** math.ceil(math.log2(actual_downsample))
        ds_width = int(width / actual_downsample)
        ds_height = int(height / actual_downsample)
        ds_mask_shape = (ds_height, ds_width)
    else:
        ds_mask_shape = mask_shape

    ds_height, ds_width = ds_mask_shape

    def to_mask_space(ring: np.ndarray) -> np.ndarray:
        """Ring coordinates in mask space, keeping their fractional part.

        Subtracting an integer origin is exact, so the mask lands on the same
        pixel centres the image does. The coordinates are deliberately *not*
        clipped: :func:`~quantem.seg_core.rasterize.fill_rings` drops the pixels
        outside the window instead, so an object hanging off the edge of the
        image is measured on the part of it that is inside rather than on an
        outline folded flat against the border.
        """
        local = np.empty((ring.shape[0], 2), dtype=np.float64)
        local[:, 0] = (ring[:, 0] - offset_x - x0) / actual_downsample
        local[:, 1] = (ring[:, 1] - offset_y - y0) / actual_downsample
        return local

    mask = fill_rings(
        [to_mask_space(ring) for ring in ring_arrays],
        x0=0,
        y0=0,
        x1=ds_width,
        y1=ds_height,
    )

    # Upsample if downsampling was used
    if actual_downsample > 1.0:
        mask = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    return mask


def compute_segment_features(
    polygon: Polygon,
    image: np.ndarray,
    ring_pixels: int,
    *,
    image_offset: tuple[int, int] = (0, 0),
    bbox_polygon: Polygon | None = None,
    return_timing: bool = False,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """
    Compute geometry + intensity features for a polygon using a cropped window.

    The polygon is rasterised at full precision in the app's pixel convention
    (:mod:`quantem.seg_core.rasterize`): a pixel is the shape's when its centre
    is inside the outline. The mask's pixel count therefore *is* the polygon's
    area, and matches what ``regionprops`` counts for the same shape found by a
    model. Very large objects (>20k px on a side) are still rasterised at
    reduced resolution.

    All returned geometry features are in PIXELS; expressing them in physical
    units requires ``Asset.pixel_size_nm``.

    Args:
        polygon: shapely Polygon geometry
        image: Full image array
        ring_pixels: Number of pixels for outside ring computation
        image_offset: Offset coordinates for the image
        bbox_polygon: Optional precomputed bounding box polygon
        return_timing: Whether to return timing information
    """
    offset_x, offset_y = image_offset
    pad = max(int(ring_pixels), 0) + 1

    # Extract polygon rings
    t0 = time.time()
    rings = _extract_polygon_rings(polygon)
    t_extract = time.time() - t0

    # Compute bounding box
    bbox = _compute_local_bbox(
        image.shape, rings, offset_x, offset_y, pad, bbox_polygon=bbox_polygon
    )
    if bbox is None:
        return ({}, {"total": 0.0} if return_timing else None)

    x0, y0, x1, y1 = bbox
    sub_image = image[y0 : y1 + 1, x0 : x1 + 1]
    mask_height, mask_width = sub_image.shape

    # Determine if we should downsample for very large masks
    use_downsample = mask_width > 20000 or mask_height > 20000

    t0 = time.time()
    mask = _rasterize_rings_to_mask(
        rings,
        sub_image.shape,
        offset_x,
        offset_y,
        x0,
        y0,
        use_downsample=use_downsample,
    )
    t_mask = time.time() - t0

    if not np.any(mask):
        timing = (
            {
                "extract": t_extract,
                "rasterize": t_mask,
                "total": t_extract + t_mask,
            }
            if return_timing
            else None
        )
        return {}, timing

    # If downsampling was used, we need to handle feature computation carefully
    # For geometry features, the mask is already upsampled, so we can use it directly
    # For intensity features, we need to work with the original sub_image

    t0 = time.time()
    geom_features = compute_regionprops_features(mask)
    t_geom = time.time() - t0

    t0 = time.time()
    int_features = compute_intensity_features(sub_image, mask)
    t_int = time.time() - t0

    features = {
        **geom_features,
        **int_features,
    }

    if not return_timing:
        return features, None

    timing = {
        "extract": t_extract,
        "rasterize": t_mask,
        "geometry": t_geom,
        "intensity": t_int,
        "total": t_extract + t_mask + t_geom + t_int,
    }
    return features, timing

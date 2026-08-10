"""Constraint helpers for edge-refinement geometry validation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as sk_polygon


def polygon_to_mask(polygon: Polygon, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon into a binary mask with the given (height, width)."""
    if polygon is None or polygon.is_empty or getattr(polygon, "exterior", None) is None:
        return np.zeros(shape, dtype=np.uint8)
    coords = [(float(x), float(y)) for x, y in polygon.exterior.coords]
    if len(coords) < 3:
        return np.zeros(shape, dtype=np.uint8)
    rows = [y for _, y in coords]
    cols = [x for x, _ in coords]
    rr, cc = sk_polygon(rows, cols, shape=shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def polygon_satisfies_points(
    polygon: Polygon,
    include_points: Iterable[tuple[float, float]],
    exclude_points: Iterable[tuple[float, float]],
) -> bool:
    """Return True if all include points are covered and all exclude points are outside."""
    if polygon is None or polygon.is_empty:
        return False
    buffered = None
    if exclude_points:
        try:
            buffered = polygon.buffer(0.5)
        except Exception:
            buffered = None
    for x_val, y_val in include_points:
        if not polygon.covers(Point(x_val, y_val)):
            return False
    for x_val, y_val in exclude_points:
        target = buffered if buffered is not None and not buffered.is_empty else polygon
        if target.covers(Point(x_val, y_val)):
            return False
    return True


def mask_satisfies_points(
    polygon: Polygon,
    shape: tuple[int, int],
    include_points: Iterable[tuple[float, float]],
    exclude_points: Iterable[tuple[float, float]],
) -> bool:
    """Check constraints against a rasterized mask so boundary pixels count as inside."""
    if polygon is None or polygon.is_empty:
        return False
    mask = polygon_to_mask(polygon, shape)
    if mask.sum() == 0:
        return False
    for x_val, y_val in include_points:
        x_idx = int(round(x_val))
        y_idx = int(round(y_val))
        if not (0 <= y_idx < mask.shape[0] and 0 <= x_idx < mask.shape[1]):
            return False
        if mask[y_idx, x_idx] == 0:
            return False
    for x_val, y_val in exclude_points:
        x_idx = int(round(x_val))
        y_idx = int(round(y_val))
        if not (0 <= y_idx < mask.shape[0] and 0 <= x_idx < mask.shape[1]):
            continue
        if mask[y_idx, x_idx] == 1:
            return False
    return True


def constraints_satisfied(
    polygon: Polygon,
    shape: tuple[int, int],
    include_points: Iterable[tuple[float, float]],
    exclude_points: Iterable[tuple[float, float]],
) -> bool:
    """Enforce include points via geometry and exclude points via raster mask."""
    if polygon is None or polygon.is_empty:
        return False
    for x_val, y_val in include_points:
        if not polygon.covers(Point(x_val, y_val)):
            return False
    if not exclude_points:
        return True
    mask = polygon_to_mask(polygon, shape)
    if mask.sum() == 0:
        return False
    for x_val, y_val in exclude_points:
        x_idx = int(round(x_val))
        y_idx = int(round(y_val))
        if (
            0 <= y_idx < mask.shape[0]
            and 0 <= x_idx < mask.shape[1]
            and mask[y_idx, x_idx] == 1
        ):
            return False
    return True


from __future__ import annotations

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.bbox_policy import bbox_to_int_bounds, has_narrow_bbox
from quantem.segmentation.geometry import extract_polygons, normalize_polygonal_geometry


def filter_supported_confirmed_polygons(polygons: list[Polygon]) -> list[Polygon]:
    supported: list[Polygon] = []
    for polygon in polygons:
        envelope = polygon.envelope
        if has_narrow_bbox(envelope):
            continue
        x0, y0, x1, y1 = bbox_to_int_bounds(envelope)
        if (x1 - x0) <= 1 or (y1 - y0) <= 1:
            continue
        supported.append(polygon)
    return supported


def geometries_overlap(left: BaseGeometry, right: BaseGeometry) -> bool:
    try:
        return bool(left.intersects(right))
    except Exception:
        return False


def merge_polygons(polygons: list[Polygon]) -> BaseGeometry | None:
    merged: BaseGeometry | None = None
    for polygon in polygons:
        merged = polygon if merged is None else safe_union(merged, polygon)
    return normalize_polygonal_geometry(merged)


def safe_intersection(left: BaseGeometry, right: BaseGeometry) -> BaseGeometry | None:
    try:
        return normalize_polygonal_geometry(left.intersection(right))
    except Exception:
        return None


def safe_difference(left: BaseGeometry, right: BaseGeometry) -> BaseGeometry | None:
    try:
        return normalize_polygonal_geometry(left.difference(right))
    except Exception:
        return None


def safe_union(
    left: BaseGeometry | None,
    right: BaseGeometry | None,
) -> BaseGeometry | None:
    if left is None:
        return normalize_polygonal_geometry(right)
    if right is None:
        return normalize_polygonal_geometry(left)
    try:
        return normalize_polygonal_geometry(left.union(right))
    except Exception:
        return None


def geometry_area(geometry: BaseGeometry | None) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    try:
        return float(geometry.area)
    except Exception:
        return 0.0


__all__ = [
    "extract_polygons",
    "filter_supported_confirmed_polygons",
    "geometries_overlap",
    "geometry_area",
    "merge_polygons",
    "safe_difference",
    "safe_intersection",
    "safe_union",
]

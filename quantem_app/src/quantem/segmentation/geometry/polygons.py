"""Shared polygon traversal and normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


def iter_polygons(geometry: BaseGeometry | None) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    child_geoms = getattr(geometry, "geoms", None)
    if child_geoms is not None:
        polygons: list[Polygon] = []
        for child in child_geoms:
            polygons.extend(iter_polygons(child))
        return polygons
    return []


def extract_polygons(geometry: BaseGeometry | None) -> list[Polygon]:
    return list(iter_polygons(geometry))


def polygon_coords(geometry: BaseGeometry) -> list[list[list[float]]]:
    return [
        [[float(x), float(y)] for x, y, *_ in polygon.exterior.coords]
        for polygon in iter_polygons(geometry)
    ]


def normalize_polygonal_geometry(
    geometry: BaseGeometry | None,
) -> BaseGeometry | None:
    """Return a valid version of ``geometry``, or ``None`` if it cannot be repaired.

    shapely rejects some self-touching rings that GEOS tolerated, so this uses
    ``shapely.make_valid`` instead.
    """
    if geometry is None or geometry.is_empty:
        return None
    try:
        if geometry.is_valid:
            return geometry
    except Exception:
        pass
    try:
        repaired = make_valid(geometry)
    except Exception:
        return None
    if repaired is None or repaired.is_empty:
        return None
    return repaired

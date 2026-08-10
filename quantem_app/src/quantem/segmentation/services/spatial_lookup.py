"""Numeric replacements for the GeoDjango spatial ORM lookups.

QuantEM has no GeoDjango. Object geometry lives in ``geometry_wkb`` with
indexed float columns beside it (``bbox_minx``/``bbox_miny``/``bbox_maxx``/
``bbox_maxy`` and ``centroid_x``/``centroid_y``); the models expose them as
shapely ``geometry`` / ``bbox`` / ``centroid`` properties. What is *not*
expressible as a property is the query side, so every spatial lookup becomes a
numeric range filter here plus an exact shapely refine step in Python:

* ``bbox__intersects=rect``      -> :func:`bbox_intersects_filter` (exact for an
  axis-aligned rectangle; a prefilter for anything else)
* ``geometry__contains=point``   -> :func:`bbox_contains_point_filter` + shapely ``contains``
* ``centroid__intersects=poly``  -> :func:`centroid_in_bbox_filter` + shapely ``contains``
* ``Union("geometry")``          -> :func:`union_geometries`

The module depends on nothing but shapely and ``django.db.models`` so views,
services and the overlay builder can all import it.
"""

from __future__ import annotations

from collections.abc import Iterable

from django.db.models import Q
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

__all__ = [
    "bbox_contains_point_filter",
    "bbox_intersects_filter",
    "centroid_in_bbox_filter",
    "make_bbox",
    "make_point",
    "union_geometries",
]


def make_bbox(x_min: float, y_min: float, x_max: float, y_max: float) -> Polygon:
    """Axis-aligned rectangle in image pixel space."""
    return box(float(x_min), float(y_min), float(x_max), float(y_max))


def make_point(x: float, y: float) -> Point:
    return Point(float(x), float(y))


def bbox_intersects_filter(geometry: BaseGeometry | None, *, prefix: str = "") -> Q | None:
    """``bbox__intersects=geometry`` as a numeric range filter.

    ``prefix`` targets a related model (e.g. ``"segment_object__"``). Returns
    ``None`` for an empty geometry so callers can skip filtering entirely.
    """
    if geometry is None or geometry.is_empty:
        return None
    x0, y0, x1, y1 = geometry.bounds
    return Q(
        **{
            f"{prefix}bbox_maxx__gte": float(x0),
            f"{prefix}bbox_minx__lte": float(x1),
            f"{prefix}bbox_maxy__gte": float(y0),
            f"{prefix}bbox_miny__lte": float(y1),
        }
    )


def bbox_contains_point_filter(x: float, y: float, *, prefix: str = "") -> Q:
    """Rows whose bbox covers ``(x, y)``; refine with shapely ``contains``."""
    return Q(
        **{
            f"{prefix}bbox_minx__lte": float(x),
            f"{prefix}bbox_maxx__gte": float(x),
            f"{prefix}bbox_miny__lte": float(y),
            f"{prefix}bbox_maxy__gte": float(y),
        }
    )


def centroid_in_bbox_filter(geometry: BaseGeometry | None, *, prefix: str = "") -> Q | None:
    """Rows whose centroid falls in ``geometry``'s bbox.

    A prefilter only: the caller must still test ``geometry.contains(centroid)``
    to reproduce ``centroid__intersects=geometry``.
    """
    if geometry is None or geometry.is_empty:
        return None
    x0, y0, x1, y1 = geometry.bounds
    return Q(
        **{
            f"{prefix}centroid_x__gte": float(x0),
            f"{prefix}centroid_x__lte": float(x1),
            f"{prefix}centroid_y__gte": float(y0),
            f"{prefix}centroid_y__lte": float(y1),
        }
    )


def union_geometries(geometries: Iterable[BaseGeometry | None]) -> BaseGeometry | None:
    """``Union(...)`` aggregate replacement -- a shapely union of fetched rows."""
    from shapely.ops import unary_union

    parts = [
        geometry
        for geometry in geometries
        if geometry is not None and not geometry.is_empty
    ]
    if not parts:
        return None
    merged = unary_union(parts)
    if merged is None or merged.is_empty:
        return None
    return merged

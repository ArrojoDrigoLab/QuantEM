from __future__ import annotations

import math

from django.db.models import F, FloatField
from django.db.models.expressions import ExpressionWrapper
from shapely.geometry.base import BaseGeometry

MIN_OBJECT_BBOX_SPAN_PX = 1.0


def bbox_extent(bbox: BaseGeometry | None) -> tuple[float, float]:
    if bbox is None or not isinstance(bbox, BaseGeometry) or bbox.is_empty:
        return 0.0, 0.0
    min_x, min_y, max_x, max_y = bbox.bounds
    return float(max_x - min_x), float(max_y - min_y)


def geometry_bbox_extent(geometry: BaseGeometry | None) -> tuple[float, float]:
    if geometry is None or not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        return 0.0, 0.0
    return bbox_extent(geometry.envelope)


def has_narrow_bbox(
    bbox: BaseGeometry | None,
    *,
    max_span_px: float = MIN_OBJECT_BBOX_SPAN_PX,
) -> bool:
    width, height = bbox_extent(bbox)
    return width <= max_span_px or height <= max_span_px


def has_narrow_geometry_bbox(
    geometry: BaseGeometry | None,
    *,
    max_span_px: float = MIN_OBJECT_BBOX_SPAN_PX,
) -> bool:
    if geometry is None or not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        return True
    return has_narrow_bbox(geometry.envelope, max_span_px=max_span_px)


def ensure_non_narrow_bbox(
    bbox: BaseGeometry | None,
    *,
    subject: str,
    max_span_px: float = MIN_OBJECT_BBOX_SPAN_PX,
) -> None:
    width, height = bbox_extent(bbox)
    if width <= max_span_px or height <= max_span_px:
        raise ValueError(
            f"{subject} bbox is too narrow ({width:.3f}x{height:.3f}); "
            "it must span more than 1 pixel in both dimensions."
        )


def bbox_to_int_bounds(bbox: BaseGeometry) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = bbox.bounds
    return (
        int(math.floor(float(min_x))),
        int(math.floor(float(min_y))),
        int(math.ceil(float(max_x))),
        int(math.ceil(float(max_y))),
    )


def bbox_width_annotation(prefix: str = "bbox") -> ExpressionWrapper:
    """Queryset annotation for bbox width.

    Was ``RawSQL("ST_XMax(bbox) - ST_XMin(bbox)")``; the bounds are stored columns
    now, so this is plain ORM arithmetic that works on SQLite.
    """
    return ExpressionWrapper(
        F(f"{prefix}_maxx") - F(f"{prefix}_minx"),
        output_field=FloatField(),
    )


def bbox_height_annotation(prefix: str = "bbox") -> ExpressionWrapper:
    """Queryset annotation for bbox height (see :func:`bbox_width_annotation`)."""
    return ExpressionWrapper(
        F(f"{prefix}_maxy") - F(f"{prefix}_miny"),
        output_field=FloatField(),
    )

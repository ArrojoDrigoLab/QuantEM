from __future__ import annotations

from shapely.geometry.base import BaseGeometry

GEOMETRY_DETAIL_FULL = "full"
GEOMETRY_DETAIL_HOVER = "hover"
_VALID_GEOMETRY_DETAILS = {
    GEOMETRY_DETAIL_FULL,
    GEOMETRY_DETAIL_HOVER,
}
_HOVER_SIMPLIFY_MIN_POINTS = 128
_HOVER_SIMPLIFY_MAX_POINTS = 1536
_HOVER_SIMPLIFY_MAX_TOLERANCE = 6.0


def normalize_geometry_detail(raw_detail: object) -> str:
    detail = str(raw_detail or GEOMETRY_DETAIL_FULL).strip().lower()
    if detail in _VALID_GEOMETRY_DETAILS:
        return detail
    return GEOMETRY_DETAIL_FULL


def _hover_simplify_tolerance(point_count: int) -> float:
    if point_count >= 32768:
        return 4.0
    if point_count >= 8192:
        return 2.5
    if point_count >= 2048:
        return 1.5
    return 1.0


def _safe_num_points(geometry: BaseGeometry | None) -> int:
    """Total vertex count across all rings.

    ``GEOSGeometry.num_points`` counted every ring; shapely has no equivalent, so
    the exterior and interior rings are summed explicitly.
    """
    if geometry is None:
        return 0
    try:
        exterior = getattr(geometry, "exterior", None)
        if exterior is None:
            return 0
        total = len(exterior.coords)
        for interior in geometry.interiors:
            total += len(interior.coords)
        return int(total)
    except Exception:
        return 0


def simplify_geometry_for_detail(
    geometry: BaseGeometry | None,
    *,
    geometry_detail: str = GEOMETRY_DETAIL_FULL,
) -> BaseGeometry | None:
    if geometry is None:
        return None
    if normalize_geometry_detail(geometry_detail) != GEOMETRY_DETAIL_HOVER:
        return geometry

    original_point_count = _safe_num_points(geometry)
    if original_point_count < _HOVER_SIMPLIFY_MIN_POINTS:
        return geometry

    tolerance = _hover_simplify_tolerance(original_point_count)
    best_geometry = geometry
    best_point_count = original_point_count

    while tolerance <= _HOVER_SIMPLIFY_MAX_TOLERANCE:
        try:
            simplified = geometry.simplify(tolerance, preserve_topology=True)
        except Exception:
            return best_geometry
        if (
            simplified is None
            or simplified.is_empty
            or getattr(simplified, "geom_type", "") != "Polygon"
            or not getattr(simplified, "is_valid", False)
        ):
            return best_geometry

        simplified_point_count = _safe_num_points(simplified)
        if simplified_point_count < 4:
            return best_geometry
        if simplified_point_count >= best_point_count:
            return best_geometry

        best_geometry = simplified
        best_point_count = simplified_point_count
        if simplified_point_count <= _HOVER_SIMPLIFY_MAX_POINTS:
            return best_geometry
        tolerance += 1.0

    return best_geometry


def geometry_coords_from_polygon(
    geometry: BaseGeometry | None,
    *,
    geometry_detail: str = GEOMETRY_DETAIL_FULL,
) -> list[list[float]]:
    resolved_geometry = simplify_geometry_for_detail(
        geometry,
        geometry_detail=geometry_detail,
    )
    if resolved_geometry is None or getattr(resolved_geometry, "exterior", None) is None:
        return []
    try:
        return [[float(coord[0]), float(coord[1])] for coord in resolved_geometry.exterior.coords]
    except Exception:
        return []

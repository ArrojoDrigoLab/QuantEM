from __future__ import annotations

import math

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.services.spatial_lookup import bbox_intersects_filter

from .geometry import (
    extract_polygons,
    geometries_overlap,
    geometry_area,
    merge_polygons,
    safe_difference,
    safe_intersection,
    safe_union,
)
from .types import (
    HALFPLANE_PADDING_PX,
    MANUAL_CANDIDATE_OVERLAP_THRESHOLD,
    MANUAL_DELETE_ELIGIBLE_STATES,
    MIN_OVERLAP_AREA,
    _ConfirmedFamily,
)


def _fallback_axis_from_envelope(geometry: BaseGeometry) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = geometry.bounds
    width = max_x - min_x
    height = max_y - min_y
    return (1.0, 0.0) if width >= height else (0.0, 1.0)


def _major_axis_direction(geometry: BaseGeometry) -> tuple[float, float]:
    coords: list[tuple[float, float]] = []
    for polygon in extract_polygons(geometry):
        for coord in polygon.exterior.coords:
            coords.append((float(coord[0]), float(coord[1])))
    if len(coords) < 2:
        return _fallback_axis_from_envelope(geometry)

    mean_x = sum(coord[0] for coord in coords) / len(coords)
    mean_y = sum(coord[1] for coord in coords) / len(coords)
    sxx = sum((coord[0] - mean_x) ** 2 for coord in coords)
    sxy = sum((coord[0] - mean_x) * (coord[1] - mean_y) for coord in coords)
    syy = sum((coord[1] - mean_y) ** 2 for coord in coords)

    if abs(sxy) <= 1e-9 and abs(sxx - syy) <= 1e-9:
        return _fallback_axis_from_envelope(geometry)
    if abs(sxy) <= 1e-9:
        return (1.0, 0.0) if sxx >= syy else (0.0, 1.0)

    trace = sxx + syy
    determinant = (sxx * syy) - (sxy * sxy)
    discriminant = max(0.0, (trace * trace) * 0.25 - determinant)
    largest_eigenvalue = (trace * 0.5) + math.sqrt(discriminant)
    vector_x = largest_eigenvalue - syy
    vector_y = sxy
    norm = math.hypot(vector_x, vector_y)
    if norm <= 1e-9:
        return _fallback_axis_from_envelope(geometry)
    return (vector_x / norm, vector_y / norm)


def _halfplane_extent(geometry: BaseGeometry) -> float:
    min_x, min_y, max_x, max_y = geometry.bounds
    width = max_x - min_x
    height = max_y - min_y
    return max(width, height, 1.0) * 4.0 + HALFPLANE_PADDING_PX


def _build_halfplane(
    *,
    center_x: float,
    center_y: float,
    direction: tuple[float, float],
    normal: tuple[float, float],
    positive: bool,
    extent: float,
) -> Polygon:
    scale = 1.0 if positive else -1.0
    dir_x, dir_y = direction
    norm_x = normal[0] * scale
    norm_y = normal[1] * scale
    return Polygon(
        (
            (center_x + dir_x * extent, center_y + dir_y * extent),
            (center_x - dir_x * extent, center_y - dir_y * extent),
            (
                center_x - dir_x * extent + norm_x * extent * 2.0,
                center_y - dir_y * extent + norm_y * extent * 2.0,
            ),
            (
                center_x + dir_x * extent + norm_x * extent * 2.0,
                center_y + dir_y * extent + norm_y * extent * 2.0,
            ),
            (center_x + dir_x * extent, center_y + dir_y * extent),
        )
    )


def _sign_with_tolerance(value: float) -> int:
    if value > 1e-6:
        return 1
    if value < -1e-6:
        return -1
    return 0


def _anchor_side_sign(
    *,
    geometry: BaseGeometry,
    overlap_component: BaseGeometry,
    center_x: float,
    center_y: float,
    normal: tuple[float, float],
) -> int:
    anchor_source = safe_difference(geometry, overlap_component)
    anchor_geometry = anchor_source if anchor_source is not None else geometry
    anchor = anchor_geometry.centroid
    signed_distance = (float(anchor.x) - center_x) * normal[0] + (
        float(anchor.y) - center_y
    ) * normal[1]
    return _sign_with_tolerance(signed_distance)


def _split_overlap_component(
    overlap_component: Polygon,
    *,
    first_geometry: BaseGeometry,
    second_geometry: BaseGeometry,
) -> tuple[BaseGeometry | None, BaseGeometry | None]:
    center = overlap_component.centroid
    center_x = float(center.x)
    center_y = float(center.y)
    direction = _major_axis_direction(overlap_component)
    normal = (-direction[1], direction[0])

    def build_parts(
        axis_direction: tuple[float, float],
        axis_normal: tuple[float, float],
    ) -> tuple[BaseGeometry | None, BaseGeometry | None]:
        extent = _halfplane_extent(overlap_component)
        positive_half = _build_halfplane(
            center_x=center_x,
            center_y=center_y,
            direction=axis_direction,
            normal=axis_normal,
            positive=True,
            extent=extent,
        )
        negative_half = _build_halfplane(
            center_x=center_x,
            center_y=center_y,
            direction=axis_direction,
            normal=axis_normal,
            positive=False,
            extent=extent,
        )
        return (
            safe_intersection(overlap_component, positive_half),
            safe_intersection(overlap_component, negative_half),
        )

    positive_part, negative_part = build_parts(direction, normal)
    if (
        geometry_area(positive_part) <= MIN_OVERLAP_AREA
        or geometry_area(negative_part) <= MIN_OVERLAP_AREA
    ):
        fallback_direction = normal
        fallback_normal = (-fallback_direction[1], fallback_direction[0])
        direction = fallback_direction
        normal = fallback_normal
        positive_part, negative_part = build_parts(direction, normal)

    first_sign = _anchor_side_sign(
        geometry=first_geometry,
        overlap_component=overlap_component,
        center_x=center_x,
        center_y=center_y,
        normal=normal,
    )
    second_sign = _anchor_side_sign(
        geometry=second_geometry,
        overlap_component=overlap_component,
        center_x=center_x,
        center_y=center_y,
        normal=normal,
    )

    if first_sign == 0 and second_sign == 0:
        first_sign = 1
        second_sign = -1
    elif first_sign == 0:
        first_sign = -second_sign
    elif second_sign == 0 or first_sign == second_sign:
        second_sign = -first_sign

    if first_sign > 0:
        return positive_part, negative_part
    return negative_part, positive_part


def resolve_overlap_between_families(
    first_family: _ConfirmedFamily,
    second_family: _ConfirmedFamily,
) -> bool:
    first_geometry = first_family.union_geometry()
    second_geometry = second_family.union_geometry()
    if first_geometry is None or second_geometry is None:
        return False
    if not geometries_overlap(first_geometry, second_geometry):
        return False

    overlap = safe_intersection(first_geometry, second_geometry)
    if geometry_area(overlap) <= MIN_OVERLAP_AREA:
        return False

    first_result = safe_difference(first_geometry, overlap)
    second_result = safe_difference(second_geometry, overlap)

    for overlap_component in extract_polygons(overlap):
        first_piece, second_piece = _split_overlap_component(
            overlap_component,
            first_geometry=first_geometry,
            second_geometry=second_geometry,
        )
        first_result = safe_union(first_result, first_piece)
        second_result = safe_union(second_result, second_piece)

    first_family.polygons = extract_polygons(first_result)
    second_family.polygons = extract_polygons(second_result)
    first_family.dirty = True
    second_family.dirty = True
    return True


def delete_manual_overlap_candidates(
    *,
    segmentation: ImageSegmentation,
    manual_families: list[_ConfirmedFamily],
) -> tuple[int, list[BaseGeometry]]:
    manual_geometries = [
        geometry
        for family in manual_families
        for geometry in [family.union_geometry()]
        if geometry is not None
    ]
    if not manual_geometries:
        return 0, []

    manual_bounds = merge_polygons([geometry.envelope for geometry in manual_geometries])
    candidate_qs = SegmentObject.objects.filter(
        segmentation=segmentation,
        label_state__in=MANUAL_DELETE_ELIGIBLE_STATES,
    )
    bounds_filter = bbox_intersects_filter(manual_bounds)
    if bounds_filter is not None:
        candidate_qs = candidate_qs.filter(bounds_filter)
    candidates = list(candidate_qs)

    delete_ids: list[str] = []
    affected_geometries: list[BaseGeometry] = []
    for segment in candidates:
        segment_geometry = segment.geometry
        candidate_area = geometry_area(segment_geometry)
        if candidate_area <= MIN_OVERLAP_AREA:
            continue
        for manual_geometry in manual_geometries:
            if not geometries_overlap(segment_geometry, manual_geometry):
                continue
            overlap = safe_intersection(segment_geometry, manual_geometry)
            overlap_ratio = geometry_area(overlap) / candidate_area
            if overlap_ratio > MANUAL_CANDIDATE_OVERLAP_THRESHOLD:
                delete_ids.append(str(segment.id))
                affected_geometries.append(segment_geometry)
                break

    if not delete_ids:
        return 0, []

    SegmentObject.objects.filter(
        segmentation=segmentation,
        id__in=delete_ids,
    ).delete()
    return len(delete_ids), affected_geometries


__all__ = [
    "delete_manual_overlap_candidates",
    "resolve_overlap_between_families",
]

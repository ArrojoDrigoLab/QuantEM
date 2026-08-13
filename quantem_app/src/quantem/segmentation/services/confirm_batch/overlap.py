from __future__ import annotations

import heapq
import math

import numpy as np
from shapely.affinity import scale as scale_geometry
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split
from skimage.morphology import skeletonize

from quantem.seg_core.rasterize import paint_rings
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
    MANUAL_CANDIDATE_OVERLAP_THRESHOLD,
    MANUAL_DELETE_ELIGIBLE_STATES,
    MIN_OVERLAP_AREA,
    _ConfirmedFamily,
)

MANUAL_CONFIRMED_UNION_THRESHOLD = 0.70
MAX_SKELETON_RASTER_PIXELS = 4_000_000
MAX_JUNCTION_CANDIDATES = 24


class OverlapResolutionError(ValueError):
    """Raised when an overlap cannot be partitioned without losing pixels."""


def overlap_qualifies_for_union(
    new_geometry: BaseGeometry,
    existing_geometry: BaseGeometry,
) -> bool:
    """True when either object has more than 70% of itself in the overlap."""
    overlap = safe_intersection(new_geometry, existing_geometry)
    overlap_area = geometry_area(overlap)
    new_area = geometry_area(new_geometry)
    existing_area = geometry_area(existing_geometry)
    if overlap_area <= MIN_OVERLAP_AREA or min(new_area, existing_area) <= MIN_OVERLAP_AREA:
        return False
    return (
        overlap_area / new_area > MANUAL_CONFIRMED_UNION_THRESHOLD
        or overlap_area / existing_area > MANUAL_CONFIRMED_UNION_THRESHOLD
    )


def _junction_points(geometry: BaseGeometry | None) -> list[Point]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type == "LineString":
        coords = list(geometry.coords)
        return [Point(coords[0]), Point(coords[-1])] if coords else []
    points: list[Point] = []
    for part in getattr(geometry, "geoms", ()):  # MultiPoint / GeometryCollection
        points.extend(_junction_points(part))
    deduplicated: list[Point] = []
    for point in points:
        if not any(point.distance(other) <= 1e-6 for other in deduplicated):
            deduplicated.append(point)
    return deduplicated


def _farthest_pair(points: list[Point]) -> tuple[Point, Point] | None:
    best = None
    best_distance = -1.0
    for index, first in enumerate(points):
        for second in points[index + 1 :]:
            distance = float(first.distance(second))
            if distance > best_distance:
                best = (first, second)
                best_distance = distance
    return best


def _representative_junctions(points: list[Point]) -> list[Point]:
    """Bound seam search cost while retaining junctions across the component."""
    if len(points) <= MAX_JUNCTION_CANDIDATES:
        return points
    pair = _farthest_pair(points)
    if pair is None:
        return points[:1]
    selected = [pair[0], pair[1]]
    remaining = [point for point in points if point not in selected]
    while remaining and len(selected) < MAX_JUNCTION_CANDIDATES:
        next_point = max(
            remaining,
            key=lambda point: min(float(point.distance(value)) for value in selected),
        )
        selected.append(next_point)
        remaining.remove(next_point)
    return selected


def _mask_dimensions(polygon: Polygon) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = polygon.bounds
    x0 = math.floor(min_x) - 2
    y0 = math.floor(min_y) - 2
    x1 = math.ceil(max_x) + 3
    y1 = math.ceil(max_y) + 3
    return x0, y0, x1, y1


def _component_mask(polygon: Polygon) -> tuple[np.ndarray, int, int, float]:
    """Rasterize a component within a fixed memory budget.

    Large objects are reduced uniformly for seam discovery only. The resulting
    path is mapped back into image coordinates before Shapely partitions the
    original geometry, so allocation still operates on the full-resolution
    polygon rather than on an approximated raster outline.
    """
    source_polygon = polygon
    x0, y0, x1, y1 = _mask_dimensions(polygon)
    area = max(1, (x1 - x0) * (y1 - y0))
    scale_factor = min(1.0, math.sqrt(MAX_SKELETON_RASTER_PIXELS / area))
    if scale_factor < 1.0:
        polygon = scale_geometry(
            source_polygon,
            xfact=scale_factor,
            yfact=scale_factor,
            origin=(0.0, 0.0),
        )
        x0, y0, x1, y1 = _mask_dimensions(polygon)
        while (x1 - x0) * (y1 - y0) > MAX_SKELETON_RASTER_PIXELS:
            scale_factor *= 0.99
            polygon = scale_geometry(
                source_polygon,
                xfact=scale_factor,
                yfact=scale_factor,
                origin=(0.0, 0.0),
            )
            x0, y0, x1, y1 = _mask_dimensions(polygon)
    target = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.uint8)
    rings = [
        np.asarray(polygon.exterior.coords, dtype=np.float64),
        *(np.asarray(ring.coords, dtype=np.float64) for ring in polygon.interiors),
    ]
    paint_rings(target, rings, 1, x0=x0, y0=y0)
    return target.astype(bool), x0, y0, scale_factor


def _nearest_skeleton_pixel(
    point: Point, coordinates: np.ndarray, *, x0: int, y0: int
) -> tuple[int, int] | None:
    if coordinates.size == 0:
        return None
    target_row = float(point.y) - y0
    target_col = float(point.x) - x0
    distances = (coordinates[:, 0] - target_row) ** 2 + (coordinates[:, 1] - target_col) ** 2
    row, col = coordinates[int(np.argmin(distances))]
    return int(row), int(col)


def _skeleton_route(
    skeleton: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    height, width = skeleton.shape
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    distance = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current == goal:
            break
        if current_distance != distance.get(current):
            continue
        row, col = current
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = (row + dr, col + dc)
                if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                    continue
                if not skeleton[neighbor]:
                    continue
                candidate = current_distance + (math.sqrt(2.0) if dr and dc else 1.0)
                if candidate < distance.get(neighbor, math.inf):
                    distance[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
    if goal not in previous:
        return []
    route = [goal]
    while route[-1] != start:
        route.append(previous[route[-1]])
    route.reverse()
    return route


def _extended_seam(
    polygon: Polygon,
    *,
    start: Point,
    end: Point,
    via: Point | None = None,
) -> LineString:
    """A medial skeleton path crossing the overlap at its boundary junctions."""
    mask, x0, y0, scale_factor = _component_mask(polygon)
    skeleton = skeletonize(mask)
    coordinates = np.argwhere(skeleton)
    scaled_start = Point(float(start.x) * scale_factor, float(start.y) * scale_factor)
    scaled_end = Point(float(end.x) * scale_factor, float(end.y) * scale_factor)
    start_pixel = _nearest_skeleton_pixel(scaled_start, coordinates, x0=x0, y0=y0)
    end_pixel = _nearest_skeleton_pixel(scaled_end, coordinates, x0=x0, y0=y0)
    route = []
    if start_pixel is not None and end_pixel is not None:
        if via is None:
            route = _skeleton_route(skeleton, start_pixel, end_pixel)
        else:
            scaled_via = Point(float(via.x) * scale_factor, float(via.y) * scale_factor)
            via_pixel = _nearest_skeleton_pixel(scaled_via, coordinates, x0=x0, y0=y0)
            if via_pixel is not None:
                first_leg = _skeleton_route(skeleton, start_pixel, via_pixel)
                second_leg = _skeleton_route(skeleton, via_pixel, end_pixel)
                if first_leg and second_leg:
                    route = [*first_leg, *second_leg[1:]]
    middle = [
        (float(col + x0) / scale_factor, float(row + y0) / scale_factor) for row, col in route
    ]
    coordinates_xy = [(float(start.x), float(start.y)), *middle, (float(end.x), float(end.y))]
    # Crossing just beyond each boundary makes shapely.split robust to the
    # first/last skeleton pixel sitting fractionally inside the polygon.
    second = coordinates_xy[1] if len(coordinates_xy) > 2 else coordinates_xy[-1]
    penultimate = coordinates_xy[-2] if len(coordinates_xy) > 2 else coordinates_xy[0]

    def extend(origin, toward):
        dx, dy = origin[0] - toward[0], origin[1] - toward[1]
        norm = math.hypot(dx, dy) or 1.0
        return (origin[0] + dx / norm * 2.0, origin[1] + dy / norm * 2.0)

    return LineString(
        [
            extend(coordinates_xy[0], second),
            *coordinates_xy,
            extend(coordinates_xy[-1], penultimate),
        ]
    ).simplify(0.25, preserve_topology=False)


def _assign_split_parts(
    parts: list[Polygon],
    *,
    first_geometry: BaseGeometry,
    second_geometry: BaseGeometry,
    overlap: BaseGeometry,
) -> tuple[BaseGeometry | None, BaseGeometry | None]:
    """Put seam pieces into two area-balanced sides, then orient by anchors."""
    first_anchor = safe_difference(first_geometry, overlap)
    second_anchor = safe_difference(second_geometry, overlap)
    groups, _group_areas = _balanced_part_groups(parts)

    group_geometry = [merge_polygons(group) for group in groups]
    if group_geometry[0] is None or group_geometry[1] is None:
        return group_geometry[0], group_geometry[1]

    def distance(value: BaseGeometry, anchor: BaseGeometry | None) -> float:
        return value.representative_point().distance(anchor) if anchor is not None else math.inf

    direct_cost = distance(group_geometry[0], first_anchor) + distance(
        group_geometry[1], second_anchor
    )
    swapped_cost = distance(group_geometry[1], first_anchor) + distance(
        group_geometry[0], second_anchor
    )
    if swapped_cost < direct_cost:
        return group_geometry[1], group_geometry[0]
    return group_geometry[0], group_geometry[1]


def _balanced_part_groups(parts: list[Polygon]) -> tuple[list[list[Polygon]], list[float]]:
    """Partition seam pieces as evenly as practical without dropping any.

    Skeleton cuts through branched regions can yield several polygons. Greedy
    two-bin assignment is order-sensitive and can miss an exact or near-exact
    allocation even for the small component counts seen here, so enumerate the
    smaller cases and keep a bounded greedy fallback for pathological inputs.
    """
    ordered = sorted(parts, key=lambda value: float(value.area), reverse=True)
    if len(ordered) < 2:
        areas = [float(ordered[0].area) if ordered else 0.0, 0.0]
        return [ordered, []], areas

    if len(ordered) <= 18:
        total = sum(float(part.area) for part in ordered)
        best_selection = 1
        best_difference = math.inf
        # Keep the largest piece in group zero to avoid evaluating each
        # partition and its identical complement.
        for selection in range(1, 1 << (len(ordered) - 1)):
            second_area = sum(
                float(ordered[index + 1].area)
                for index in range(len(ordered) - 1)
                if selection & (1 << index)
            )
            difference = abs(total - 2.0 * second_area)
            if difference < best_difference:
                best_selection = selection
                best_difference = difference
        groups = [[ordered[0]], []]
        for index, part in enumerate(ordered[1:]):
            groups[1 if best_selection & (1 << index) else 0].append(part)
    else:
        groups = [[], []]
        totals = [0.0, 0.0]
        for part in ordered:
            target = 0 if totals[0] <= totals[1] else 1
            groups[target].append(part)
            totals[target] += float(part.area)

    return groups, [sum(float(part.area) for part in group) for group in groups]


def _split_overlap_component(
    overlap_component: Polygon,
    *,
    first_geometry: BaseGeometry,
    second_geometry: BaseGeometry,
) -> tuple[BaseGeometry | None, BaseGeometry | None]:
    junctions = _junction_points(first_geometry.boundary.intersection(second_geometry.boundary))
    relevant = [point for point in junctions if point.distance(overlap_component) <= 1e-6]
    if len(relevant) < 2:
        raise OverlapResolutionError(
            "The shared region could not be divided because two overlap junctions were not found."
        )

    # A branched overlap can have more than two boundary junctions. The
    # farthest pair is not necessarily a cut (it can follow one outside edge),
    # so try each pair from a spatially representative bounded set and retain
    # the skeleton cut closest to half.
    candidate_junctions = _representative_junctions(relevant)
    pairs = [
        (first, second)
        for index, first in enumerate(candidate_junctions)
        for second in candidate_junctions[index + 1 :]
    ]
    pairs.sort(key=lambda pair: float(pair[0].distance(pair[1])), reverse=True)
    best_parts: list[Polygon] = []
    best_balance = math.inf
    for pair in pairs:
        for via in (
            None,
            overlap_component.representative_point(),
            overlap_component.centroid,
        ):
            seam = _extended_seam(
                overlap_component,
                start=pair[0],
                end=pair[1],
                via=via,
            )
            try:
                split_parts = extract_polygons(split(overlap_component, seam))
            except Exception:
                split_parts = []
            if len(split_parts) < 2:
                continue
            _groups, totals = _balanced_part_groups(split_parts)
            balance = abs(totals[0] - totals[1])
            if balance < best_balance:
                best_balance = balance
                best_parts = split_parts
        if best_balance <= max(float(overlap_component.area) * 0.02, 1.0):
            break
    if len(best_parts) < 2:
        raise OverlapResolutionError(
            "The shared region could not be divided without losing overlap pixels."
        )
    return _assign_split_parts(
        best_parts,
        first_geometry=first_geometry,
        second_geometry=second_geometry,
        overlap=overlap_component,
    )


def resolve_overlap_between_families(
    first_family: _ConfirmedFamily,
    second_family: _ConfirmedFamily,
) -> bool:
    """Allocate every non-union overlap pixel once using a skeleton seam."""
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
    for component in extract_polygons(overlap):
        first_piece, second_piece = _split_overlap_component(
            component,
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
    SegmentObject.objects.filter(segmentation=segmentation, id__in=delete_ids).delete()
    return len(delete_ids), affected_geometries


__all__ = [
    "OverlapResolutionError",
    "delete_manual_overlap_candidates",
    "overlap_qualifies_for_union",
    "resolve_overlap_between_families",
]

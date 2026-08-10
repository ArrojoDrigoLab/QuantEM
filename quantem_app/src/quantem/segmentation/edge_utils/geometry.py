from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.validation import make_valid
from skimage.measure import find_contours


def count_points_in_mask(
    mask: np.ndarray, points: Iterable[tuple[float, float]]
) -> tuple[int, int]:
    inside = 0
    oob = 0
    for x_val, y_val in points:
        x_idx = int(round(x_val))
        y_idx = int(round(y_val))
        if 0 <= y_idx < mask.shape[0] and 0 <= x_idx < mask.shape[1]:
            if mask[y_idx, x_idx]:
                inside += 1
        else:
            oob += 1
    return inside, oob


def sample_bilinear(image: np.ndarray, x_val: float, y_val: float) -> float | None:
    height, width = image.shape
    if x_val < 0 or y_val < 0 or x_val >= width - 1 or y_val >= height - 1:
        return None
    x0 = int(np.floor(x_val))
    y0 = int(np.floor(y_val))
    dx = x_val - x0
    dy = y_val - y0
    v00 = float(image[y0, x0])
    v10 = float(image[y0, x0 + 1])
    v01 = float(image[y0 + 1, x0])
    v11 = float(image[y0 + 1, x0 + 1])
    return (
        v00 * (1.0 - dx) * (1.0 - dy)
        + v10 * dx * (1.0 - dy)
        + v01 * (1.0 - dx) * dy
        + v11 * dx * dy
    )


def resample_polyline(
    coords: list[tuple[float, float]], step: float
) -> list[tuple[float, float]]:
    sampled: list[tuple[float, float]] = []
    for idx in range(len(coords) - 1):
        x0, y0 = coords[idx]
        x1, y1 = coords[idx + 1]
        seg_len = float(np.hypot(x1 - x0, y1 - y0))
        if seg_len == 0:
            continue
        steps = max(int(seg_len // step), 1)
        for s in range(steps):
            t = (s * step) / seg_len
            if t >= 1.0:
                break
            sampled.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    if sampled and sampled[0] != sampled[-1]:
        sampled.append(sampled[0])
    return sampled


def smooth_closed_curve(
    points: list[tuple[float, float]], window: int = 7, passes: int = 1
) -> list[tuple[float, float]]:
    if len(points) < window + 2:
        return points
    if points[0] != points[-1]:
        points = points + [points[0]]
    coords = np.array(points[:-1], dtype=np.float32)
    for _ in range(passes):
        padded = np.vstack([coords[-window:], coords, coords[:window]])
        smoothed = []
        for idx in range(window, len(padded) - window):
            window_slice = padded[idx - window : idx + window + 1]
            smoothed.append(window_slice.mean(axis=0))
        coords = np.array(smoothed, dtype=np.float32)
    smoothed_points = [(float(x), float(y)) for x, y in coords]
    if smoothed_points[0] != smoothed_points[-1]:
        smoothed_points.append(smoothed_points[0])
    return smoothed_points


def _largest_polygon(geometry) -> Polygon | None:
    """Largest Polygon part of ``geometry``, or None if it has no polygonal part."""
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        return geometry
    parts = [
        part
        for part in getattr(geometry, "geoms", [])
        if isinstance(part, Polygon) and not part.is_empty
    ]
    if not parts:
        return None
    parts.sort(key=lambda poly: poly.area, reverse=True)
    return parts[0]


def polygon_from_contour(contour: np.ndarray) -> Polygon | None:
    if contour.shape[0] < 3:
        return None
    coords = [(float(pt[1]), float(pt[0])) for pt in contour]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        polygon = Polygon(coords)
    except Exception:
        return None
    if polygon.is_valid:
        return polygon
    try:
        fixed = make_valid(polygon)
    except Exception:
        return None
    return _largest_polygon(fixed)


def normalize_polygon(polygon: Polygon | MultiPolygon) -> Polygon:
    """Return the largest valid Polygon part of ``polygon``.

    shapely rejects some self-touching rings that GEOS tolerated, so this repairs
    with ``shapely.make_valid``.
    """
    if isinstance(polygon, MultiPolygon):
        largest = _largest_polygon(polygon)
        if largest is None:
            raise ValueError("Invalid polygon could not be repaired.")
        polygon = largest
    if polygon.is_valid:
        return polygon
    try:
        fixed = make_valid(polygon)
    except Exception as exc:
        raise ValueError("Invalid polygon could not be repaired.") from exc
    largest = _largest_polygon(fixed)
    if largest is None:
        raise ValueError("Invalid polygon could not be repaired.")
    return largest


def mask_to_polygon(
    mask: np.ndarray, include_points: Iterable[tuple[float, float]] | None = None
) -> Polygon | None:
    contours = find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return None

    include_points = list(include_points or [])
    contour_polys: list[tuple[Polygon, list[tuple[float, float]]]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        coords = [(float(pt[1]), float(pt[0])) for pt in contour]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
        except Exception:
            continue
        if poly.area <= 0:
            continue
        poly = normalize_polygon(poly)
        contour_polys.append((poly, coords))

    if not contour_polys:
        return None

    holes_map: dict[int, list[list[tuple[float, float]]]] = {
        idx: [] for idx in range(len(contour_polys))
    }
    outer_indices: list[int] = []
    for idx, (poly, _) in enumerate(contour_polys):
        parent_idx = None
        for jdx, (candidate, _) in enumerate(contour_polys):
            if idx == jdx:
                continue
            if candidate.contains(poly.centroid) and (
                parent_idx is None
                or candidate.area < contour_polys[parent_idx][0].area
            ):
                parent_idx = jdx
        if parent_idx is None:
            outer_indices.append(idx)
        else:
            holes_map[parent_idx].append(contour_polys[idx][1])

    result_polys: list[Polygon] = []
    for idx in outer_indices:
        outer_coords = contour_polys[idx][1]
        holes = holes_map.get(idx, [])
        try:
            poly = Polygon(outer_coords, holes)
        except Exception:
            poly = contour_polys[idx][0]
        poly = normalize_polygon(poly)
        result_polys.append(poly)

    if not result_polys:
        return None

    if include_points:
        include_pts = [Point(x_val, y_val) for x_val, y_val in include_points]
        for poly in result_polys:
            if all(poly.covers(pt) for pt in include_pts):
                return poly
        containing = [
            poly for poly in result_polys if any(poly.covers(pt) for pt in include_pts)
        ]
        if containing:
            combined = containing[0]
            for poly in containing[1:]:
                combined = normalize_polygon(combined.union(poly))
            return combined

    result_polys.sort(key=lambda poly: poly.area, reverse=True)
    return result_polys[0]


def compute_normals(
    points: list[tuple[float, float]],
    polygon: Polygon,
    mask: np.ndarray | None = None,
    signed_distance: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    normals = []
    inside_normals = []
    num_points = len(points) - 1
    height, width = (0, 0)
    if mask is not None:
        height, width = mask.shape
    step = 1.5

    def is_inside(x_val: float, y_val: float) -> bool:
        if signed_distance is not None:
            dist = sample_bilinear(signed_distance, x_val, y_val)
            if dist is not None:
                return dist >= 0.0
        if mask is not None:
            x_idx = int(round(x_val))
            y_idx = int(round(y_val))
            if 0 <= y_idx < height and 0 <= x_idx < width:
                return mask[y_idx, x_idx] == 1
            return False
        return polygon.covers(Point(x_val, y_val))

    for idx in range(num_points):
        prev_pt = points[idx - 1]
        curr_pt = points[idx]
        next_pt = points[(idx + 1) % num_points]
        dx = next_pt[0] - prev_pt[0]
        dy = next_pt[1] - prev_pt[1]
        length = float(np.hypot(dx, dy)) or 1.0
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx
        if signed_distance is not None:
            d_pos = sample_bilinear(
                signed_distance, curr_pt[0] + nx * step, curr_pt[1] + ny * step
            )
            d_neg = sample_bilinear(
                signed_distance, curr_pt[0] - nx * step, curr_pt[1] - ny * step
            )
            if d_pos is not None and d_neg is not None:
                inside_normals.append((nx, ny) if d_pos >= d_neg else (-nx, -ny))
            elif d_pos is not None:
                inside_normals.append((nx, ny) if d_pos >= 0 else (-nx, -ny))
            elif d_neg is not None:
                inside_normals.append((-nx, -ny) if d_neg >= 0 else (nx, ny))
            else:
                inside_normals.append(
                    (nx, ny)
                    if is_inside(curr_pt[0] + nx * step, curr_pt[1] + ny * step)
                    else (-nx, -ny)
                )
        else:
            if is_inside(curr_pt[0] + nx * step, curr_pt[1] + ny * step):
                inside_normals.append((nx, ny))
            elif is_inside(curr_pt[0] - nx * step, curr_pt[1] - ny * step):
                inside_normals.append((-nx, -ny))
            else:
                inside_normals.append((nx, ny))
        normals.append((nx, ny))
    return normals, inside_normals


def count_connected_components(mask: np.ndarray) -> int:
    if mask.sum() == 0:
        return 0
    _, num = label(mask.astype(np.uint8))
    return int(num)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    return distance_transform_edt(mask_bool) - distance_transform_edt(~mask_bool)

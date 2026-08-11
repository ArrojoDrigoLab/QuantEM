from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
)
from shapely.geometry import Point, Polygon
from skimage.segmentation import watershed

from quantem.segmentation.edge_constraints import (
    constraints_satisfied,
    polygon_to_mask,
)

from .geometry import count_points_in_mask, mask_to_polygon, normalize_polygon


class ConstraintEnforcementError(ValueError):
    def __init__(self, message: str, stats: dict | None = None) -> None:
        super().__init__(message)
        self.stats = stats or {}


def check_point_conflicts(
    include_points: Iterable[tuple[float, float]],
    exclude_points: Iterable[tuple[float, float]],
) -> None:
    include_pixels = {(int(round(x)), int(round(y))) for x, y in include_points}
    exclude_pixels = {(int(round(x)), int(round(y))) for x, y in exclude_points}
    if include_pixels & exclude_pixels:
        raise ValueError("Include/exclude points conflict on the same pixel.")


def enforce_constraints_watershed(
    img: np.ndarray,
    gmag: np.ndarray,
    polygon: Polygon,
    include_points: list[tuple[float, float]],
    exclude_points: list[tuple[float, float]],
    band_in: int,
    band_out: int,
) -> Polygon:
    if not include_points and not exclude_points:
        return polygon

    stats: dict = {
        "include_total": len(include_points),
        "exclude_total": len(exclude_points),
    }

    mask = polygon_to_mask(polygon, img.shape)
    stats["mask_area"] = int(mask.sum())
    if mask.sum() == 0:
        raise ConstraintEnforcementError("Empty mask for constraint enforcement.", stats)
    mask = binary_fill_holes(mask.astype(bool)).astype(bool)
    stats["mask_area_filled"] = int(mask.sum())
    stats["include_in_mask"], stats["include_oob"] = count_points_in_mask(mask, include_points)
    stats["exclude_in_mask"], stats["exclude_oob"] = count_points_in_mask(mask, exclude_points)
    if mask.sum() == 0:
        raise ConstraintEnforcementError("Empty mask for constraint enforcement.", stats)

    band_in = max(1, int(round(band_in)))
    band_out = max(1, int(round(band_out)))
    expand = max(band_in, band_out, 3)
    roi = binary_dilation(mask, iterations=expand)
    stats["roi_area"] = int(roi.sum())
    stats["roi_expand"] = int(expand)
    mask_bool = mask.astype(bool)
    signed_distance = distance_transform_edt(mask_bool) - distance_transform_edt(~mask_bool)
    band_limit = float(max(band_in, band_out))
    interior_max = float(np.max(signed_distance[mask_bool])) if mask_bool.any() else 0.0
    fg_margin = min(5.0, max(1.0, interior_max * 0.5)) if interior_max > 0 else 0.0
    seed_margin = min(5.0, max(2.0, band_limit * 0.2))
    bg_margin = max(seed_margin, band_limit * 0.7)
    fg_seed = signed_distance >= fg_margin if fg_margin > 0 else mask_bool.copy()
    bg_seed = signed_distance <= -bg_margin
    fg_seed &= roi
    bg_seed &= roi
    if not fg_seed.any():
        fg_seed = mask_bool & roi
        stats["fg_seed_fallback"] = True
    if not bg_seed.any():
        bg_seed = (~mask_bool) & roi
        stats["bg_seed_fallback"] = True
    stats["seed_margin"] = seed_margin
    stats["fg_margin"] = fg_margin
    stats["bg_margin"] = bg_margin
    stats["interior_max"] = interior_max
    stats["fg_seed_area"] = int(np.sum(fg_seed))
    stats["bg_seed_area"] = int(np.sum(bg_seed))
    stats["include_in_roi"], _ = count_points_in_mask(roi, include_points)
    stats["exclude_in_roi"], _ = count_points_in_mask(roi, exclude_points)
    if roi.sum() == 0:
        raise ConstraintEnforcementError("Constraint enforcement ROI is empty.", stats)

    inv_roi = ~roi
    _, indices = distance_transform_edt(inv_roi, return_indices=True)

    def project_point(x_val: float, y_val: float) -> tuple[int, int] | None:
        x_idx = int(round(x_val))
        y_idx = int(round(y_val))
        if not (0 <= y_idx < mask.shape[0] and 0 <= x_idx < mask.shape[1]):
            return None
        if not roi[y_idx, x_idx]:
            y_idx = int(indices[0, y_idx, x_idx])
            x_idx = int(indices[1, y_idx, x_idx])
        return y_idx, x_idx

    include_pixels: list[tuple[int, int]] = []
    exclude_pixels: list[tuple[int, int]] = []
    for x_val, y_val in include_points:
        projected = project_point(x_val, y_val)
        if projected is not None:
            include_pixels.append(projected)
    for x_val, y_val in exclude_points:
        projected = project_point(x_val, y_val)
        if projected is not None:
            exclude_pixels.append(projected)

    if set(include_pixels) & set(exclude_pixels):
        raise ConstraintEnforcementError(
            "Include/exclude points collide after ROI projection.", stats
        )

    exclude_radii = [0] if not exclude_pixels else [2, 5]
    refined_mask = None
    stats["exclude_attempts"] = len(exclude_radii)

    for radius in exclude_radii:
        markers = np.zeros_like(mask, dtype=np.int32)
        markers[fg_seed] = 1
        markers[bg_seed] = 2
        for y_idx, x_idx in include_pixels:
            markers[y_idx, x_idx] = 1

        exclude_mask = np.zeros_like(mask, dtype=bool)
        for y_idx, x_idx in exclude_pixels:
            exclude_mask[y_idx, x_idx] = True
        if radius > 0:
            exclude_mask = binary_dilation(exclude_mask, iterations=radius)
        exclude_mask &= roi
        exclude_mask[markers == 1] = False
        markers[exclude_mask] = 2
        for y_idx, x_idx in exclude_pixels:
            markers[y_idx, x_idx] = 2

        if np.all(markers == 0):
            centroid = polygon.centroid
            projected = project_point(float(centroid.x), float(centroid.y))
            if projected is not None:
                markers[projected[0], projected[1]] = 1

        stats["marker_fg"] = int(np.sum(markers == 1))
        stats["marker_bg"] = int(np.sum(markers == 2))
        stats["exclude_radius_used"] = radius
        stats["exclude_mask_area"] = int(exclude_mask.sum())

        labels = watershed(gmag, markers, mask=roi)
        candidate_mask = labels == 1
        stats["refined_area"] = int(candidate_mask.sum())
        stats["include_in_refined"], _ = count_points_in_mask(candidate_mask, include_points)
        stats["exclude_in_refined"], _ = count_points_in_mask(candidate_mask, exclude_points)
        min_area = max(10, int(mask.sum() * 0.1))
        stats["refined_min_area"] = int(min_area)

        if candidate_mask.sum() < min_area:
            continue
        if stats["include_in_refined"] == len(include_points) and stats["exclude_in_refined"] == 0:
            refined_mask = candidate_mask
            break

    if refined_mask is None:
        refined_mask = candidate_mask
    refined_polygon = mask_to_polygon(refined_mask.astype(np.uint8), include_points)
    if refined_polygon is None:
        raise ConstraintEnforcementError(
            "Failed to extract polygon from constrained segmentation.", stats
        )
    mask_satisfies = (
        stats["include_in_refined"] == len(include_points) and stats["exclude_in_refined"] == 0
    )
    stats["mask_satisfies"] = bool(mask_satisfies)
    stats["include_in_polygon"] = sum(
        1 for x_val, y_val in include_points if refined_polygon.covers(Point(x_val, y_val))
    )
    stats["exclude_in_polygon"] = sum(
        1 for x_val, y_val in exclude_points if refined_polygon.contains(Point(x_val, y_val))
    )
    stats["polygon_valid"] = bool(refined_polygon.is_valid)
    stats["polygon_holes"] = len(getattr(refined_polygon, "interiors", ()))
    stats["polygon_type"] = refined_polygon.geom_type

    if not constraints_satisfied(refined_polygon, img.shape, include_points, exclude_points):
        stats["polygon_constraint_mismatch"] = True
        if mask_satisfies:
            adjusted = refined_polygon.buffer(-0.5)
            if adjusted is not None and not adjusted.is_empty:
                adjusted = normalize_polygon(adjusted)
                if constraints_satisfied(adjusted, img.shape, include_points, exclude_points):
                    stats["polygon_adjusted"] = True
                    return adjusted

            for radius in (1, 2, 3):
                eroded = binary_erosion(refined_mask, iterations=radius)
                if eroded.sum() == 0:
                    continue
                include_inside, _ = count_points_in_mask(eroded, include_points)
                exclude_inside, _ = count_points_in_mask(eroded, exclude_points)
                if include_inside < len(include_points) or exclude_inside > 0:
                    continue
                eroded_polygon = mask_to_polygon(eroded.astype(np.uint8), include_points)
                if eroded_polygon is None:
                    continue
                if constraints_satisfied(eroded_polygon, img.shape, include_points, exclude_points):
                    stats["polygon_eroded"] = True
                    stats["erode_radius"] = radius
                    return normalize_polygon(eroded_polygon)

        raise ConstraintEnforcementError("Constraint enforcement failed to satisfy points.", stats)
    return refined_polygon

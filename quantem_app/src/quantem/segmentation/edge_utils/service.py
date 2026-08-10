from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from shapely.geometry import Polygon

from quantem.segmentation.edge_constraints import (
    constraints_satisfied,
    polygon_satisfies_points,
    polygon_to_mask,
)

from .constraints import check_point_conflicts
from .geometry import (
    compute_normals,
    count_connected_components,
    normalize_polygon,
    resample_polyline,
    sample_bilinear,
    signed_distance,
)
from .offsets import apply_offsets, compute_edge_score, run_offset_dp

logger = logging.getLogger(__name__)


def refine_mask_with_edges(
    img: np.ndarray,
    seed_polygon: Polygon,
    include_points: Iterable[tuple[float, float]],
    exclude_points: Iterable[tuple[float, float]],
    *,
    cache_key: str | None = None,
    sigma_coarse: float = 3.0,
    sigma_fine: float = 1.0,
    band_in_coarse: float = 120.0,
    band_out_coarse: float = 40.0,
    band_in_fine: float = 20.0,
    band_out_fine: float = 10.0,
    step_coarse: float = 2.0,
    step_fine: float = 1.0,
    w_g: float = 1.0,
    w_dir: float = 0.6,
    w_dark: float = 0.0,
    w_out: float = 0.2,
    conf_threshold: float = 0.2,
    dir_thresh: float = 0.6,
    pyramid_scale: float = 0.5,
    max_offset: float = 5.0,
    inward_bias_weight: float = 0.4,
    smoothness_weight: float = 0.6,
    contrast_method: str = "percentile",
    max_area_change_fraction: float = 0.20,
    return_stats: bool = False,
) -> tuple[Polygon, dict] | Polygon:
    _ = (
        cache_key,
        sigma_coarse,
        sigma_fine,
        band_in_coarse,
        band_out_coarse,
        band_in_fine,
        band_out_fine,
        step_coarse,
        step_fine,
        w_g,
        w_dir,
        w_dark,
        w_out,
        conf_threshold,
        dir_thresh,
        pyramid_scale,
    )
    check_point_conflicts(include_points, exclude_points)
    if img.ndim != 2:
        raise ValueError("image must be a 2D grayscale array.")
    if seed_polygon is None or seed_polygon.is_empty:
        raise ValueError("Invalid seed polygon.")
    seed_polygon = normalize_polygon(seed_polygon)

    include_points = list(include_points)
    exclude_points = list(exclude_points)

    stats: dict = {"edge_insufficient": False, "reverted": False}

    height, width = img.shape
    max_offset_px = max(1, int(round(max_offset)))
    max_area_change_fraction = float(max_area_change_fraction)
    pad = max_offset_px + 10

    xmin, ymin, xmax, ymax = seed_polygon.bounds
    x0 = max(int(np.floor(xmin)) - pad, 0)
    y0 = max(int(np.floor(ymin)) - pad, 0)
    x1 = min(int(np.ceil(xmax)) + pad + 1, width)
    y1 = min(int(np.ceil(ymax)) + pad + 1, height)
    if x1 <= x0 or y1 <= y0:
        stats.update({"reverted": True, "revert_reason": "empty_roi"})
        logger.info("Edge refinement reverted: empty ROI.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    patch = img[y0:y1, x0:x1].astype(np.float32)
    coords_full = list(seed_polygon.exterior.coords)
    coords_patch = [(float(x) - x0, float(y) - y0) for x, y, *_ in coords_full]

    try:
        patch_polygon = Polygon(coords_patch)
    except Exception:
        stats.update({"reverted": True, "revert_reason": "polygon_build_failed"})
        logger.info("Edge refinement reverted: patch polygon build failed.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    patch_mask = polygon_to_mask(patch_polygon, patch.shape).astype(bool)
    if patch_mask.sum() == 0:
        stats.update({"reverted": True, "revert_reason": "empty_mask"})
        logger.info("Edge refinement reverted: empty mask.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    score = compute_edge_score(patch, contrast_method=contrast_method, blur_sigma=1.0, grad_weight=0.8, dark_weight=0.2)

    points = resample_polyline(coords_patch, step=1.0)
    if len(points) < 5:
        stats.update({"reverted": True, "revert_reason": "too_few_points"})
        logger.info("Edge refinement reverted: insufficient contour points.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    signed = signed_distance(patch_mask)
    _, inside_normals = compute_normals(points, patch_polygon, mask=patch_mask, signed_distance=signed)
    outward_normals = [(-nx, -ny) for nx, ny in inside_normals]

    num_points = len(points) - 1
    t_values = np.arange(-max_offset_px, max_offset_px + 1, dtype=np.float32)
    unary = np.zeros((num_points, t_values.size), dtype=np.float32)
    stay_close_weight = 0.15
    nonzero_bias = 0.2
    inward_bias_weight = float(inward_bias_weight)
    for idx in range(num_points):
        px, py = points[idx]
        nx, ny = outward_normals[idx]
        for jdx, t in enumerate(t_values):
            sx = px + nx * float(t)
            sy = py + ny * float(t)
            sample = sample_bilinear(score, sx, sy)
            score_val = float(sample) if sample is not None else 0.0
            cost = -score_val + stay_close_weight * abs(float(t))
            if t > 0:
                cost += inward_bias_weight * float(t)
            if t != 0:
                cost += nonzero_bias
            unary[idx, jdx] = cost

    offsets = run_offset_dp(unary, t_values, smoothness_weight)
    offsets_arr = np.array(offsets, dtype=np.float32)
    moved_mask = np.abs(offsets_arr) >= 0.5
    moved_fraction = float(np.mean(moved_mask)) if moved_mask.size else 0.0
    maxed_fraction = float(np.mean(np.abs(offsets_arr) >= (max_offset_px - 1e-3)) if offsets_arr.size else 0.0)
    median_offset = float(np.median(offsets_arr)) if offsets_arr.size else 0.0

    base_scores = []
    new_scores = []
    for idx in range(num_points):
        px, py = points[idx]
        nx, ny = outward_normals[idx]
        base = sample_bilinear(score, px, py)
        shifted = sample_bilinear(score, px + nx * offsets_arr[idx], py + ny * offsets_arr[idx])
        base_scores.append(float(base) if base is not None else 0.0)
        new_scores.append(float(shifted) if shifted is not None else 0.0)
    score_improvement = float(np.mean(np.array(new_scores) - np.array(base_scores)))

    stats.update(
        {
            "moved_fraction": moved_fraction,
            "maxed_fraction": maxed_fraction,
            "median_offset": median_offset,
            "score_improvement": score_improvement,
        }
    )

    if moved_fraction < 0.05 and score_improvement < 0.01:
        stats.update({"edge_insufficient": True, "reverted": True, "revert_reason": "weak_edge"})
        logger.info("Edge refinement reverted: weak edge evidence.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    refined_points = apply_offsets(points, outward_normals, offsets)
    if len(refined_points) < 4:
        stats.update({"reverted": True, "revert_reason": "invalid_refined_points"})
        logger.info("Edge refinement reverted: invalid refined points.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    try:
        refined_patch_polygon = Polygon(refined_points)
    except Exception:
        stats.update({"reverted": True, "revert_reason": "refined_polygon_build_failed"})
        logger.info("Edge refinement reverted: polygon build failed.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    if not refined_patch_polygon.is_valid:
        stats.update({"reverted": True, "revert_reason": "self_intersection"})
        logger.info("Edge refinement reverted: self-intersection.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    area_ratio = float(refined_patch_polygon.area / seed_polygon.area) if seed_polygon.area else 1.0
    stats["area_ratio"] = area_ratio
    if abs(area_ratio - 1.0) > max_area_change_fraction:
        stats.update({"reverted": True, "revert_reason": "area_change"})
        logger.info("Edge refinement reverted: area change too large.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    if median_offset > 0.0:
        stats.update({"reverted": True, "revert_reason": "median_outward"})
        logger.info("Edge refinement reverted: median offset outward.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    if maxed_fraction > 0.35:
        stats.update({"reverted": True, "revert_reason": "max_offset_saturation"})
        logger.info("Edge refinement reverted: offset saturation.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    refined_patch_mask = polygon_to_mask(refined_patch_polygon, patch.shape).astype(bool)
    if count_connected_components(refined_patch_mask) != 1:
        stats.update({"reverted": True, "revert_reason": "multiple_components"})
        logger.info("Edge refinement reverted: multiple components.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    include_patch = [(x - x0, y - y0) for x, y in include_points]
    exclude_patch = [(x - x0, y - y0) for x, y in exclude_points]
    constraint_ok = constraints_satisfied(refined_patch_polygon, patch.shape, include_patch, exclude_patch)

    if not constraint_ok and (include_points or exclude_points):
        forced_bounds: dict[int, tuple[float | None, float | None]] = {}
        if refined_patch_mask.sum() == 0:
            stats.update({"reverted": True, "revert_reason": "empty_refined_mask"})
            logger.info("Edge refinement reverted: empty refined mask.")
            return (seed_polygon, stats) if return_stats else seed_polygon
        for x_val, y_val in include_patch:
            x_idx = int(round(x_val))
            y_idx = int(round(y_val))
            if (
                0 <= y_idx < refined_patch_mask.shape[0]
                and 0 <= x_idx < refined_patch_mask.shape[1]
                and not refined_patch_mask[y_idx, x_idx]
            ):
                    dists = [(idx, (points[idx][0] - x_val) ** 2 + (points[idx][1] - y_val) ** 2) for idx in range(num_points)]
                    nearest = min(dists, key=lambda item: item[1])[0] if dists else None
                    if nearest is not None:
                        forced_bounds[nearest] = (0.0, None)
        for x_val, y_val in exclude_patch:
            x_idx = int(round(x_val))
            y_idx = int(round(y_val))
            if (
                0 <= y_idx < refined_patch_mask.shape[0]
                and 0 <= x_idx < refined_patch_mask.shape[1]
                and refined_patch_mask[y_idx, x_idx]
            ):
                    dists = [(idx, (points[idx][0] - x_val) ** 2 + (points[idx][1] - y_val) ** 2) for idx in range(num_points)]
                    nearest = min(dists, key=lambda item: item[1])[0] if dists else None
                    if nearest is not None:
                        forced_bounds[nearest] = (None, 0.0)

        if forced_bounds:
            offsets = run_offset_dp(unary, t_values, smoothness_weight, forced_bounds)
            refined_points = apply_offsets(points, outward_normals, offsets)
            try:
                refined_patch_polygon = Polygon(refined_points)
            except Exception:
                stats.update({"reverted": True, "revert_reason": "constraint_polygon_build_failed"})
                logger.info("Edge refinement reverted: constraint polygon build failed.")
                return (seed_polygon, stats) if return_stats else seed_polygon
            refined_patch_mask = polygon_to_mask(refined_patch_polygon, patch.shape).astype(bool)
            constraint_ok = constraints_satisfied(refined_patch_polygon, patch.shape, include_patch, exclude_patch)

        if not constraint_ok:
            stats.update({"reverted": True, "revert_reason": "point_constraints"})
            logger.info("Edge refinement reverted: point constraints failed.")
            return (seed_polygon, stats) if return_stats else seed_polygon

    if moved_fraction > 0.15 and score_improvement < 0.02:
        stats.update({"reverted": True, "revert_reason": "low_score_improvement"})
        logger.info("Edge refinement reverted: low score improvement.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    refined_full_coords = [(x + x0, y + y0) for x, y in refined_points]
    try:
        refined_polygon = Polygon(refined_full_coords)
    except Exception:
        stats.update({"reverted": True, "revert_reason": "final_polygon_build_failed"})
        logger.info("Edge refinement reverted: final polygon build failed.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    if not refined_polygon.is_valid:
        stats.update({"reverted": True, "revert_reason": "final_self_intersection"})
        logger.info("Edge refinement reverted: final polygon invalid.")
        return (seed_polygon, stats) if return_stats else seed_polygon

    stats["refined"] = refined_polygon.wkt != seed_polygon.wkt
    logger.info("Edge refinement applied.", extra={"refined": stats["refined"], "score_improvement": score_improvement})

    if return_stats:
        return refined_polygon, stats
    return refined_polygon


def _refine_mask_with_edges_self_check() -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[30:70, 30:70] = 200
    coords = [(20, 20), (80, 20), (80, 80), (20, 80), (20, 20)]
    polygon = Polygon(coords)
    include = [(50, 50)]
    exclude = [(10, 10)]
    refined = refine_mask_with_edges(img, polygon, include, exclude)
    assert polygon_satisfies_points(refined, include, exclude)

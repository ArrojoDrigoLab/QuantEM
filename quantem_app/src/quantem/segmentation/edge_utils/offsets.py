from __future__ import annotations

import numpy as np
from skimage.filters import gaussian, scharr_h, scharr_v

from .geometry import sample_bilinear


def adaptive_edge_threshold(
    best_edges: np.ndarray, seg_lengths: list[float], perimeter: float
) -> tuple[float, float, float, int]:
    if best_edges.size == 0 or perimeter <= 0:
        return 0.0, 0.0, 0.0, 0

    def run_stats(edge_flags: np.ndarray) -> tuple[float, float]:
        covered = sum(seg_lengths[idx] for idx, flag in enumerate(edge_flags) if flag)
        coverage_ratio = covered / perimeter if perimeter > 0 else 0.0
        max_run = 0.0
        current = 0.0
        for idx, flag in enumerate(edge_flags):
            if flag:
                current += seg_lengths[idx]
            else:
                if current > max_run:
                    max_run = current
                current = 0.0
        if current > max_run:
            max_run = current
        if edge_flags[0] and edge_flags[-1]:
            head = 0.0
            idx = 0
            while idx < len(edge_flags) and edge_flags[idx]:
                head += seg_lengths[idx]
                idx += 1
            tail = 0.0
            idx = len(edge_flags) - 1
            while idx >= 0 and edge_flags[idx]:
                tail += seg_lengths[idx]
                idx -= 1
            max_run = max(max_run, head + tail)
        max_run_ratio = max_run / perimeter if perimeter > 0 else 0.0
        return coverage_ratio, max_run_ratio

    threshold_percentile = 40
    edge_threshold = float(np.percentile(best_edges, threshold_percentile))
    edge_flags = best_edges >= edge_threshold
    coverage, run_ratio = run_stats(edge_flags)

    for percentile in (35, 30, 25, 20, 15, 10, 5, 0):
        if coverage >= 0.6 and run_ratio >= 0.3:
            break
        edge_threshold = float(np.percentile(best_edges, percentile))
        edge_flags = best_edges >= edge_threshold
        coverage, run_ratio = run_stats(edge_flags)
        threshold_percentile = percentile

    return edge_threshold, coverage, run_ratio, threshold_percentile


def score_candidates(
    gmag: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    img: np.ndarray,
    point: tuple[float, float],
    normal: tuple[float, float],
    offsets: np.ndarray,
    gmag_max: float,
    w_g: float,
    w_dir: float,
    w_dark: float,
    w_out: float,
    dark_reference: float | None,
    edge_threshold: float,
    conf_threshold: float,
    outward_limit: float,
) -> tuple[float, float]:
    best_score = -1e9
    best_offset = 0.0
    nx, ny = normal
    for offset in offsets:
        if offset < -outward_limit:
            continue
        cand_x = point[0] + nx * offset
        cand_y = point[1] + ny * offset
        gval = sample_bilinear(gmag, cand_x, cand_y)
        gx_val = sample_bilinear(gx, cand_x, cand_y)
        gy_val = sample_bilinear(gy, cand_x, cand_y)
        if gval is None or gx_val is None or gy_val is None:
            continue
        gnorm = gval / gmag_max if gmag_max > 0 else 0.0
        if gnorm < edge_threshold:
            gnorm = 0.0
        grad_norm = float(np.hypot(gx_val, gy_val)) or 1.0
        gunit = (gx_val / grad_norm, gy_val / grad_norm)
        dir_align = abs(gunit[0] * nx + gunit[1] * ny)
        intensity = sample_bilinear(img, cand_x, cand_y)
        if intensity is None:
            continue
        score = w_g * gnorm + w_dir * dir_align
        if w_dark > 0 and dark_reference is not None:
            score -= w_dark * abs((intensity - dark_reference) / 255.0)
        if offset < 0:
            score -= w_out * abs(offset)
        if score > best_score:
            best_score = score
            best_offset = float(offset)

    if best_score < conf_threshold:
        return 0.0, best_score
    return best_offset, best_score


def coarse_offsets(
    gmag: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    img: np.ndarray,
    points: list[tuple[float, float]],
    normals: list[tuple[float, float]],
    inside_normals: list[tuple[float, float]],
    band_in: float,
    band_out: float,
    step: float,
    dir_thresh: float,
    k_mad: float,
    allow_outward: bool = True,
    direction_bias: str = "neutral",
) -> tuple[list[float], list[float]]:
    offsets = []
    confidences = []
    for idx, point in enumerate(points[:-1]):
        normal = inside_normals[idx]
        nx, ny = normal
        inward_offsets = np.arange(0, band_in + step, step)
        outward_offsets = (
            np.arange(-step, -band_out - step, -step) if band_out > 0 else np.array([])
        )
        if direction_bias == "expand":
            primary_offsets = outward_offsets
            secondary_offsets: np.ndarray = np.array([])
        elif direction_bias == "contract":
            primary_offsets = inward_offsets
            secondary_offsets = np.array([])
        else:
            primary_offsets = inward_offsets
            secondary_offsets = outward_offsets if allow_outward else np.array([])
        gvals = []
        for offset in primary_offsets:
            cand_x = point[0] + nx * offset
            cand_y = point[1] + ny * offset
            gval = sample_bilinear(gmag, cand_x, cand_y)
            gx_val = sample_bilinear(gx, cand_x, cand_y)
            gy_val = sample_bilinear(gy, cand_x, cand_y)
            if gval is None or gx_val is None or gy_val is None:
                continue
            grad_norm = float(np.hypot(gx_val, gy_val)) or 1.0
            gunit = (gx_val / grad_norm, gy_val / grad_norm)
            dir_align = abs(gunit[0] * nx + gunit[1] * ny)
            if dir_align >= dir_thresh:
                gvals.append(gval)
        if gvals:
            gvals_arr = np.array(gvals, dtype=np.float32)
            median = float(np.median(gvals_arr))
            mad = float(np.median(np.abs(gvals_arr - median))) or 1e-6
            threshold = median + k_mad * mad
        else:
            threshold = float(np.percentile(gmag, 70)) if gmag.size else 0.0

        chosen_offset = 0.0
        confidence = 0.0
        found = False
        for offset in primary_offsets:
            cand_x = point[0] + nx * offset
            cand_y = point[1] + ny * offset
            gval = sample_bilinear(gmag, cand_x, cand_y)
            gx_val = sample_bilinear(gx, cand_x, cand_y)
            gy_val = sample_bilinear(gy, cand_x, cand_y)
            if gval is None or gx_val is None or gy_val is None:
                continue
            grad_norm = float(np.hypot(gx_val, gy_val)) or 1.0
            gunit = (gx_val / grad_norm, gy_val / grad_norm)
            dir_align = abs(gunit[0] * nx + gunit[1] * ny)
            if gval >= threshold and dir_align >= dir_thresh:
                chosen_offset = float(offset)
                confidence = float(gval)
                found = True
                break

        if secondary_offsets.size and not found:
            for offset in secondary_offsets:
                cand_x = point[0] + nx * offset
                cand_y = point[1] + ny * offset
                gval = sample_bilinear(gmag, cand_x, cand_y)
                gx_val = sample_bilinear(gx, cand_x, cand_y)
                gy_val = sample_bilinear(gy, cand_x, cand_y)
                if gval is None or gx_val is None or gy_val is None:
                    continue
                grad_norm = float(np.hypot(gx_val, gy_val)) or 1.0
                gunit = (gx_val / grad_norm, gy_val / grad_norm)
                dir_align = abs(gunit[0] * nx + gunit[1] * ny)
                if gval >= threshold and dir_align >= dir_thresh:
                    chosen_offset = float(offset)
                    confidence = float(gval)
                    found = True
                    break

        offsets.append(chosen_offset)
        confidences.append(confidence)

    return offsets, confidences


def apply_offsets(
    points: list[tuple[float, float]],
    normals: list[tuple[float, float]],
    offsets: list[float],
) -> list[tuple[float, float]]:
    adjusted = []
    for idx, point in enumerate(points[:-1]):
        nx, ny = normals[idx]
        offset = offsets[idx]
        adjusted.append((point[0] + nx * offset, point[1] + ny * offset))
    if adjusted and adjusted[0] != adjusted[-1]:
        adjusted.append(adjusted[0])
    return adjusted


def percentile_stretch(
    image: np.ndarray, p_low: float = 2.0, p_high: float = 98.0
) -> np.ndarray:
    if image.size == 0:
        return image.astype(np.float32)
    low, high = np.percentile(image, [p_low, p_high])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
    stretched = (image.astype(np.float32) - float(low)) / float(high - low)
    return np.clip(stretched, 0.0, 1.0)


def compute_edge_score(
    patch: np.ndarray,
    *,
    contrast_method: str = "percentile",
    blur_sigma: float = 1.0,
    grad_weight: float = 0.8,
    dark_weight: float = 0.2,
) -> np.ndarray:
    if contrast_method == "percentile":
        norm = percentile_stretch(patch)
    else:
        norm = np.clip(patch.astype(np.float32) / 255.0, 0.0, 1.0)
    smoothed = gaussian(norm, sigma=blur_sigma, preserve_range=True).astype(np.float32)
    gx = scharr_h(smoothed).astype(np.float32)
    gy = scharr_v(smoothed).astype(np.float32)
    gmag = np.hypot(gx, gy).astype(np.float32)
    gmag_norm = gmag / (float(np.max(gmag)) + 1e-6)
    darkness = 1.0 - norm
    score = grad_weight * gmag_norm + dark_weight * darkness
    score = np.clip(score, 0.0, 1.0)
    return score.astype(np.float32)


def run_offset_dp(
    unary_cost: np.ndarray,
    t_values: np.ndarray,
    smoothness_weight: float,
    forced_bounds: dict[int, tuple[float | None, float | None]] | None = None,
) -> list[float]:
    num_points, num_offsets = unary_cost.shape
    smoothness_weight = float(smoothness_weight)
    forced_bounds = forced_bounds or {}
    cost = unary_cost.copy()
    for idx, (min_t, max_t) in forced_bounds.items():
        if 0 <= idx < num_points:
            mask = np.ones(num_offsets, dtype=bool)
            if min_t is not None:
                mask &= t_values >= min_t
            if max_t is not None:
                mask &= t_values <= max_t
            cost[idx, ~mask] = np.inf

    best_total = np.inf
    best_back = None
    best_end = 0
    best_start = 0
    for start_idx in range(num_offsets):
        if not np.isfinite(cost[0, start_idx]):
            continue
        dp = np.full(num_offsets, np.inf, dtype=np.float32)
        dp[start_idx] = cost[0, start_idx]
        back = np.zeros((num_points, num_offsets), dtype=np.int16)
        for i in range(1, num_points):
            next_dp = np.full(num_offsets, np.inf, dtype=np.float32)
            for j in range(num_offsets):
                if not np.isfinite(cost[i, j]):
                    continue
                diffs = t_values[j] - t_values
                penalty = smoothness_weight * (diffs**2)
                total = dp + penalty
                prev = int(np.argmin(total))
                next_dp[j] = cost[i, j] + total[prev]
                back[i, j] = prev
            dp = next_dp
        wrap_penalty = smoothness_weight * (t_values - t_values[start_idx]) ** 2
        totals = dp + wrap_penalty
        end_idx = int(np.argmin(totals))
        total_cost = float(totals[end_idx])
        if total_cost < best_total:
            best_total = total_cost
            best_back = back.copy()
            best_end = end_idx
            best_start = start_idx

    if best_back is None:
        return [0.0 for _ in range(num_points)]
    offsets = [0.0] * num_points
    idx = best_end
    for i in range(num_points - 1, 0, -1):
        offsets[i] = float(t_values[idx])
        idx = int(best_back[i, idx])
    offsets[0] = float(t_values[best_start])
    return offsets

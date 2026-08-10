from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy import ndimage

from .config import TileExportConfig
from .identity import normalization_config_hash


@dataclass(frozen=True)
class TileScore:
    tissue_score: float
    non_background_fraction: float
    texture_fraction: float
    gradient_fraction: float
    artifact_fraction: float
    background_fraction: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PlaneScorer:
    image_width: int
    image_height: int
    thumbnail_width: int
    thumbnail_height: int
    low: float
    high: float
    normalization_hash: str
    non_background_mask: np.ndarray
    texture_mask: np.ndarray
    gradient_mask: np.ndarray
    artifact_mask: np.ndarray
    intensity_thumbnail_uint8: np.ndarray

    def score_window(self, *, x: int, y: int, width: int, height: int) -> TileScore:
        y0, y1, x0, x1 = self._thumbnail_bounds(x=x, y=y, width=width, height=height)
        if y1 <= y0 or x1 <= x0:
            return TileScore(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, ("empty",))
        non_background = self.non_background_mask[y0:y1, x0:x1]
        texture = self.texture_mask[y0:y1, x0:x1]
        gradient = self.gradient_mask[y0:y1, x0:x1]
        artifact = self.artifact_mask[y0:y1, x0:x1]

        non_background_fraction = float(non_background.mean())
        texture_fraction = float(texture.mean())
        gradient_fraction = float(gradient.mean())
        artifact_fraction = float(artifact.mean())
        background_fraction = 1.0 - non_background_fraction
        score = (
            (0.45 * non_background_fraction)
            + (0.35 * texture_fraction)
            + (0.25 * gradient_fraction)
            - (0.30 * artifact_fraction)
        )
        score = float(np.clip(score, 0.0, 1.0))
        reasons: list[str] = []
        if background_fraction >= 0.50:
            reasons.append("background")
        if texture_fraction < 0.20:
            reasons.append("low_texture")
        if gradient_fraction < 0.15:
            reasons.append("low_gradient")
        if artifact_fraction >= 0.20:
            reasons.append("artifact")
        return TileScore(
            tissue_score=score,
            non_background_fraction=non_background_fraction,
            texture_fraction=texture_fraction,
            gradient_fraction=gradient_fraction,
            artifact_fraction=artifact_fraction,
            background_fraction=background_fraction,
            reasons=tuple(reasons),
        )

    def _thumbnail_bounds(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        scale_x = self.thumbnail_width / float(max(self.image_width, 1))
        scale_y = self.thumbnail_height / float(max(self.image_height, 1))
        x0 = int(math.floor(int(x) * scale_x))
        y0 = int(math.floor(int(y) * scale_y))
        x1 = int(math.ceil((int(x) + int(width)) * scale_x))
        y1 = int(math.ceil((int(y) + int(height)) * scale_y))
        return (
            min(max(y0, 0), self.thumbnail_height),
            min(max(y1, 0), self.thumbnail_height),
            min(max(x0, 0), self.thumbnail_width),
            min(max(x1, 0), self.thumbnail_width),
        )


def build_plane_scorer(
    thumbnail: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    config: TileExportConfig,
) -> PlaneScorer:
    values = np.asarray(thumbnail)
    if values.ndim == 3:
        values = values[..., 0]
    values_float = values.astype(np.float32, copy=False)
    finite_mask = np.isfinite(values_float)
    finite_values = values_float[finite_mask]
    if finite_values.size == 0:
        zeros = np.zeros(values_float.shape, dtype=bool)
        return PlaneScorer(
            image_width=int(image_width),
            image_height=int(image_height),
            thumbnail_width=int(values_float.shape[1]),
            thumbnail_height=int(values_float.shape[0]),
            low=0.0,
            high=255.0,
            normalization_hash=normalization_config_hash(config, low=0.0, high=255.0),
            non_background_mask=zeros,
            texture_mask=zeros,
            gradient_mask=zeros,
            artifact_mask=np.ones(values_float.shape, dtype=bool),
            intensity_thumbnail_uint8=np.zeros(values_float.shape, dtype=np.uint8),
        )

    low, high = np.percentile(finite_values, [1, 99])
    if high <= low:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
    if high <= low:
        normalized = np.zeros(values_float.shape, dtype=np.float32)
    else:
        normalized = np.clip((values_float - low) / float(high - low), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0)
    intensity_uint8 = np.rint(normalized * 255.0).astype(np.uint8)

    local_window = max(3, int(round(max(values_float.shape) / 256.0)))
    if local_window % 2 == 0:
        local_window += 1
    mean = ndimage.uniform_filter(normalized, size=local_window, mode="nearest")
    mean2 = ndimage.uniform_filter(normalized * normalized, size=local_window, mode="nearest")
    local_std = np.sqrt(np.maximum(mean2 - (mean * mean), 0.0))

    gx = ndimage.sobel(normalized, axis=1, mode="nearest")
    gy = ndimage.sobel(normalized, axis=0, mode="nearest")
    gradient = ndimage.uniform_filter(
        np.hypot(gx, gy),
        size=max(3, local_window // 2),
        mode="nearest",
    )

    near_black = intensity_uint8 <= 3
    near_white = intensity_uint8 >= 252
    constant = local_std <= max(float(np.percentile(local_std, 10)), 0.005)
    non_background = finite_mask & ~near_black & ~near_white & ~constant

    active_std = local_std[non_background]
    active_gradient = gradient[non_background]
    std_threshold = _adaptive_threshold(active_std, default=0.03)
    gradient_threshold = _adaptive_threshold(active_gradient, default=0.05)
    texture_mask = non_background & (local_std >= std_threshold)
    gradient_mask = non_background & (gradient >= gradient_threshold)
    artifact_mask = _build_artifact_mask(
        intensity_uint8,
        normalized=normalized,
        local_std=local_std,
        non_background=non_background,
    )

    return PlaneScorer(
        image_width=int(image_width),
        image_height=int(image_height),
        thumbnail_width=int(values_float.shape[1]),
        thumbnail_height=int(values_float.shape[0]),
        low=float(low),
        high=float(high),
        normalization_hash=normalization_config_hash(config, low=float(low), high=float(high)),
        non_background_mask=non_background.astype(bool, copy=False),
        texture_mask=texture_mask.astype(bool, copy=False),
        gradient_mask=gradient_mask.astype(bool, copy=False),
        artifact_mask=artifact_mask.astype(bool, copy=False),
        intensity_thumbnail_uint8=intensity_uint8,
    )


def crop_to_content(array: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Trim fully-zero border rows/columns from a 2D plane.

    Zero-padding (a specimen that does not fill its FOV, or a plane padded to
    fixed dimensions) otherwise dilutes the tissue score, because the padding
    reads as background over the scored window — tile_size is a maximum, not a
    padded requirement. Cropping to the non-zero bounding box lets the tissue
    filter run on the real data.

    Only border rows/columns that are entirely zero are removed; any row/column
    holding a single non-zero pixel is kept, so genuine (non-zero) dark EM
    background and scattered zero pixels are never cropped. Interior zeros are
    left in place (correctly scored as background).

    Returns ``(cropped, x_offset, y_offset, original_width, original_height)``;
    a no-op (offsets 0, original dims, same array) when there is no all-zero
    border or the plane is empty/all-zero.
    """
    a = np.asarray(array)
    h = int(a.shape[0]) if a.ndim >= 1 else 0
    w = int(a.shape[1]) if a.ndim >= 2 else 0
    if a.ndim < 2 or a.size == 0:
        return a, 0, 0, w, h
    nonzero = a != 0
    rows = np.flatnonzero(np.any(nonzero, axis=1))
    cols = np.flatnonzero(np.any(nonzero, axis=0))
    if rows.size == 0 or cols.size == 0:          # all-zero plane: leave as-is
        return a, 0, 0, w, h
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if x0 == 0 and y0 == 0 and x1 == w and y1 == h:  # no border to trim
        return a, 0, 0, w, h
    return a[y0:y1, x0:x1], x0, y0, w, h


def _adaptive_threshold(values: np.ndarray, *, default: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    p40 = float(np.percentile(values, 40))
    p60 = float(np.percentile(values, 60))
    return max(float(default), (p40 + p60) / 2.0)


def _build_artifact_mask(
    intensity: np.ndarray,
    *,
    normalized: np.ndarray,
    local_std: np.ndarray,
    non_background: np.ndarray,
) -> np.ndarray:
    height, width = intensity.shape
    artifact = np.zeros((height, width), dtype=bool)
    if height <= 0 or width <= 0:
        return artifact

    bottom_start = int(round(height * 0.92))
    bottom = slice(bottom_start, height)
    bottom_region = intensity[bottom, :]
    if bottom_region.size:
        near_extreme = (bottom_region <= 6) | (bottom_region >= 249)
        row_extreme = near_extreme.mean(axis=1)
        row_low_texture = local_std[bottom, :].mean(axis=1) < 0.02
        footer_rows = row_extreme > 0.50
        footer_rows |= row_low_texture & (row_extreme > 0.20)
        if np.count_nonzero(footer_rows) >= max(1, int(0.20 * footer_rows.size)):
            artifact[bottom, :] = True

    high_contrast = np.abs(ndimage.sobel(normalized, axis=1, mode="nearest")) > 0.75
    high_contrast |= np.abs(ndimage.sobel(normalized, axis=0, mode="nearest")) > 0.75
    row_density = high_contrast.mean(axis=1)
    col_density = high_contrast.mean(axis=0)
    artifact[row_density > max(0.35, float(np.percentile(row_density, 99)))] = True
    artifact[:, col_density > max(0.35, float(np.percentile(col_density, 99)))] = True

    bottom_band_start = int(round(height * 0.80))
    extreme = (intensity <= 6) | (intensity >= 249)
    labels, count = ndimage.label(extreme[bottom_band_start:, :])
    if count:
        min_area = max(32, int(0.00005 * height * width))
        # find_objects yields each component's bounding box in a single pass, so the work
        # is proportional to total component area rather than count * full-array scans.
        for label_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
            if bounds is None:
                continue
            y_slice, x_slice = bounds
            component_height = int(y_slice.stop - y_slice.start)
            component_width = int(x_slice.stop - x_slice.start)
            area = int(np.count_nonzero(labels[bounds] == label_id))
            if area == 0:
                continue
            rectangularity = area / float(max(component_height * component_width, 1))
            if (
                area >= min_area
                and rectangularity >= 0.65
                and component_width >= component_height * 3
            ):
                artifact[
                    bottom_band_start + y_slice.start : bottom_band_start + y_slice.stop,
                    x_slice.start : x_slice.stop,
                ] = True

    return artifact & (artifact | ~non_background)


def tile_status(
    score: TileScore,
    *,
    config: TileExportConfig,
) -> str:
    if score.tissue_score >= config.min_tissue_fraction:
        return "accepted"
    if score.tissue_score >= config.borderline_tissue_fraction:
        return "borderline"
    return "rejected"


def score_to_json(score: TileScore) -> dict[str, Any]:
    return {
        "tissue_score": round(score.tissue_score, 6),
        "non_background_fraction": round(score.non_background_fraction, 6),
        "texture_fraction": round(score.texture_fraction, 6),
        "gradient_fraction": round(score.gradient_fraction, 6),
        "artifact_fraction": round(score.artifact_fraction, 6),
        "background_fraction": round(score.background_fraction, 6),
        "reasons": list(score.reasons),
    }

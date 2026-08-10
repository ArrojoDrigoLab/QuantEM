"""
Probability map feature helpers for segment percentiles.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from quantem.segmentation.features.extraction import (
    _compute_local_bbox,
    _extract_polygon_rings,
    _rasterize_rings_to_mask,
)
from quantem.segmentation.models import ProbabilityMap
from quantem.segmentation.prob_maps.io import resolve_probability_map_path


def load_probability_map_float(
    prob_map: ProbabilityMap,
) -> tuple[np.ndarray, tuple[int, int]]:
    prob_file_path = resolve_probability_map_path(prob_map)
    if not prob_file_path.exists():
        raise FileNotFoundError(f"Probability map file not found: {prob_file_path}")

    pil_image = Image.open(prob_file_path)
    if pil_image.mode != "L":
        pil_image = pil_image.convert("L")
    prob_data_uint8 = np.array(pil_image, dtype=np.uint8)
    if prob_data_uint8.ndim != 2:
        raise ValueError(
            f"Expected 2D probability map, got shape {prob_data_uint8.shape}"
        )

    offset = (0, 0)
    roi_meta = prob_map.metadata.get("roi") if prob_map.metadata else None
    if roi_meta:
        roi_x = roi_meta["x"]
        roi_y = roi_meta["y"]
        roi_width = roi_meta["width"]
        roi_height = roi_meta["height"]
        if prob_data_uint8.shape != (roi_height, roi_width):
            raise ValueError(
                f"ROI probability map shape {prob_data_uint8.shape} does not match "
                f"ROI bounds ({roi_height}, {roi_width})"
            )
        offset = (roi_x, roi_y)

    prob_data = prob_data_uint8.astype(np.float32) / 255.0
    return prob_data, offset


def rasterize_polygon_in_bbox(
    polygon: Polygon,
    bbox: Polygon,
    prob_data_shape: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    height, width = prob_data_shape

    rings = _extract_polygon_rings(polygon)
    if not rings:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # The rings go to the rasteriser as drawn. They used to be run through a
    # 1 px Douglas-Peucker pass first, so the probability was averaged over an
    # outline up to a pixel away from the object's -- and then over a mask a
    # further pixel wider than that, because ``cv2.fillPoly`` painted both
    # boundaries. Both errors pulled background into the mean of a small object.
    bbox_result = _compute_local_bbox(
        prob_data_shape,
        rings,
        offset_x=offset[0],
        offset_y=offset[1],
        pad=0,
        bbox_polygon=bbox,
    )
    if bbox_result is None:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    x0, y0, x1, y1 = bbox_result
    mask_height = y1 - y0 + 1
    mask_width = x1 - x0 + 1
    use_downsample = mask_width > 20000 or mask_height > 20000

    try:
        mask = _rasterize_rings_to_mask(
            rings,
            (mask_height, mask_width),
            offset_x=offset[0],
            offset_y=offset[1],
            x0=x0,
            y0=y0,
            use_downsample=use_downsample,
        )
    except Exception:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    if not np.any(mask):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    rr_bbox, cc_bbox = np.where(mask)
    rr_full = rr_bbox + y0
    cc_full = cc_bbox + x0
    valid_mask = (
        (rr_full >= 0) & (rr_full < height) & (cc_full >= 0) & (cc_full < width)
    )
    rr_full = rr_full[valid_mask]
    cc_full = cc_full[valid_mask]
    return rr_full, cc_full


def compute_prob_map_percentiles_from_data(
    prob_data: np.ndarray,
    offset: tuple[int, int],
    polygon: Polygon,
    bbox: Polygon,
) -> dict[str, float]:
    """Mean and percentiles of the map under ``polygon``, or ``{}``.

    Empty when the polygon covers no pixel of this map -- it lies outside an
    ROI-scoped map, or rasterises to nothing. That is "there was nothing to
    measure", and the caller must store nothing.

    It used to return zeros for that case. A probability of ``0.0`` is not a
    missing value: it says the model was certain this is background, which is
    the strongest statement the number can make about an object the model in
    fact never saw.
    """
    rr, cc = rasterize_polygon_in_bbox(polygon, bbox, prob_data.shape, offset=offset)
    if len(rr) == 0:
        return {}

    values = prob_data[rr, cc]
    if values.size == 0:
        return {}

    return {
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def compute_prob_map_percentiles(
    prob_map: ProbabilityMap,
    polygon: Polygon,
    bbox: Polygon,
) -> dict[str, float]:
    prob_data, offset = load_probability_map_float(prob_map)
    return compute_prob_map_percentiles_from_data(prob_data, offset, polygon, bbox)

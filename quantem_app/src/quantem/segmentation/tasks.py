"""Synchronous segmentation helper functions used by API and DB worker flows.

Legacy probability-map pipeline orchestration and pixel-classifier training task
helpers were removed during queue unification. Keep this module focused on the
small set of helpers used by active views.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.task_utils import load_image_array, load_image_roi_array
from quantem.segmentation.features.extraction import compute_segment_features
from quantem.segmentation.features.measure import merge_measured_features
from quantem.segmentation.prob_maps.features import (
    compute_prob_map_percentiles_from_data,
    load_probability_map_float,
)

from .models import ProbabilityMap, SegmentObject

logger = logging.getLogger(__name__)
DEFAULT_OUTSIDE_RING_PIXELS = 10

#: The four statistics stored per probability map, in the order they are named.
PROB_MAP_STAT_NAMES: tuple[str, ...] = ("mean", "p10", "p50", "p90")

if TYPE_CHECKING:
    from quantem.segmentation.models import ImageSegmentation


def _segment_feature_task_enabled() -> bool:
    raw = str(os.environ.get("QUANTEM_ENABLE_SEGMENT_FEATURE_TASK", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _get_segmentation_target(segmentation: ImageSegmentation):
    if segmentation.asset_id:
        return get_asset_openable(segmentation.asset)
    raise ValueError("Segmentation has no target asset")


def prob_map_feature_keys(prob_map_id: object) -> tuple[str, ...]:
    """The four ``features`` keys one probability map owns on an object."""
    return tuple(f"prob_{prob_map_id}_{stat}" for stat in PROB_MAP_STAT_NAMES)


def _apply_prob_map_stats(
    features: dict,
    prob_map: ProbabilityMap,
    polygon,
    bbox,
) -> dict:
    """Store this map's percentiles under the object -- or store nothing.

    A map whose file has been deleted (observed for real) used to be recorded
    as ``{"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}``. In a probability
    map ``0.0`` is not a missing value: it is *the model was certain this is
    background*, the strongest claim the number can make, and the opposite of
    what happened. The same fabrication reached here from
    :func:`~quantem.segmentation.prob_maps.features.compute_prob_map_percentiles_from_data`
    when the polygon covered no pixel of the map, which is also "nothing to
    measure" rather than "measured zero".

    So the keys are written only when there is a value to write, and are
    *removed* when there is not. This task runs because a geometry changed, so
    a value left over from the previous outline describes a shape that no
    longer exists -- stale is a different kind of wrong from fabricated, not a
    lesser one. Absent means "not measured", which is the convention every
    other probability key here already follows.
    """
    keys = prob_map_feature_keys(prob_map.id)
    try:
        prob_data, offset = load_probability_map_float(prob_map)
        stats = compute_prob_map_percentiles_from_data(
            prob_data,
            offset,
            polygon,
            bbox,
        )
    except Exception as exc:
        logger.warning(
            "No probability statistics for map %s; leaving the keys unset: %s",
            prob_map.id,
            exc,
        )
        stats = {}

    if not stats:
        for key in keys:
            features.pop(key, None)
        return features

    for key, stat in zip(keys, PROB_MAP_STAT_NAMES, strict=True):
        features[key] = stats[stat]
    return features


def _load_segment_feature_window(
    *,
    target,
    bbox,
    ring_pixels: int,
) -> tuple[object, tuple[int, int]]:
    """
    Load only a local image window around the segment bbox for feature extraction.

    Falls back to full-image load if bbox/image metadata is unavailable.
    """
    image_width = int(getattr(target, "width", 0) or 0)
    image_height = int(getattr(target, "height", 0) or 0)
    if bbox is None or image_width <= 0 or image_height <= 0:
        full_image, _ = load_image_array(target)
        return full_image, (0, 0)

    try:
        min_x, min_y, max_x, max_y = bbox.bounds
    except Exception:
        full_image, _ = load_image_array(target)
        return full_image, (0, 0)

    pad = max(int(ring_pixels), 0) + 2
    x0 = max(0, int(math.floor(min(min_x, max_x))) - pad)
    y0 = max(0, int(math.floor(min(min_y, max_y))) - pad)
    x1 = min(image_width, int(math.ceil(max(min_x, max_x))) + pad)
    y1 = min(image_height, int(math.ceil(max(min_y, max_y))) + pad)
    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)

    window = load_image_roi_array(target, x0, y0, width, height)
    return window, (x0, y0)


def compute_segment_features_task(segment_id: str) -> None:
    """Re-measure one segment's geometry/intensity/probability-map features.

    **Preserving, never replacing.** What this task measures is written over the
    keys it measured; everything else the object carries survives -- see
    :func:`quantem.segmentation.features.measure.merge_measured_features`, which
    is the same rule the synchronous writer follows.
    """
    if not _segment_feature_task_enabled():
        logger.info(
            "Skipping segment feature task for %s (QUANTEM_ENABLE_SEGMENT_FEATURE_TASK disabled)",
            segment_id,
        )
        return

    try:
        segment = SegmentObject.objects.select_related(
            "segmentation", "segmentation__asset"
        ).get(id=segment_id)
    except SegmentObject.DoesNotExist:
        logger.warning("Segment %s missing for feature task", segment_id)
        return

    segmentation = segment.segmentation
    polygon = segment.geometry
    bbox = segment.bbox
    ring_pixels = DEFAULT_OUTSIDE_RING_PIXELS

    try:
        target = _get_segmentation_target(segmentation)
        image_array, image_offset = _load_segment_feature_window(
            target=target,
            bbox=bbox,
            ring_pixels=ring_pixels,
        )
        measurements, _ = compute_segment_features(
            polygon,
            image_array,
            ring_pixels,
            image_offset=image_offset,
            bbox_polygon=bbox,
        )
    except Exception as exc:
        logger.error(
            "Failed to compute features for segment %s: %s",
            segment_id,
            exc,
            exc_info=True,
        )
        measurements = {}

    features = merge_measured_features(segment.features, measurements)

    prob_maps = ProbabilityMap.objects.filter(segmentation=segmentation)
    for prob_map in prob_maps.iterator():
        features = _apply_prob_map_stats(features, prob_map, polygon, bbox)

    SegmentObject.objects.filter(id=segment_id).update(features=features)

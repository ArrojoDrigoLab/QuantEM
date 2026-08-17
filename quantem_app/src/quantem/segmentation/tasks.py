"""Synchronous segmentation helper functions used by API and DB worker flows.

Legacy probability-map pipeline orchestration and pixel-classifier training task
helpers were removed during queue unification. Keep this module focused on the
small set of helpers used by active views.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from django.db import transaction

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

#: How many objects :func:`measure_segments_batched` fetches, measures and
#: writes back per round trip.
#:
#: Preview stores geometry without measurements, so the list handed to that
#: function is routinely *every* object in the segmentation -- and because
#: Analysis outranks the background sweep, it is Analysis that pays for them,
#: with the user watching. Measured per object, the old one-at-a-time path cost
#: a SELECT of the object, a SELECT of the segmentation's probability maps and
#: a single-row committing UPDATE; on a 20 000-object image that is ~60 000
#: statements before the first mask is built. Batching turns the two SELECTs
#: into one per chunk and the UPDATEs into one ``bulk_update`` per chunk.
#:
#: 200 keeps the ``id__in`` SELECT and the ``CASE WHEN`` of ``bulk_update``
#: (~3 bound parameters per row) comfortably inside SQLite's historical
#: 999-variable ceiling, while still reducing a 20 000-object sweep to ~100
#: round trips. It also bounds how much measured-but-unwritten work a crash or
#: a cancellation can throw away.
SEGMENT_MEASUREMENT_BATCH_SIZE = 200

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
    data_cache: dict[str, tuple[object, tuple[int, int]] | None] | None = None,
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
        cache_key = str(prob_map.id)
        cached = data_cache.get(cache_key) if data_cache is not None else None
        if data_cache is not None and cache_key in data_cache and cached is None:
            raise FileNotFoundError("probability map was unavailable on first read")
        if cached is None:
            prob_data, offset = load_probability_map_float(prob_map)
            if data_cache is not None:
                data_cache[cache_key] = (prob_data, offset)
        else:
            prob_data, offset = cached
        stats = compute_prob_map_percentiles_from_data(
            prob_data,
            offset,
            polygon,
            bbox,
        )
    except Exception as exc:
        if data_cache is not None:
            data_cache[str(prob_map.id)] = None
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


def _measured_features_for(
    segment: SegmentObject,
    *,
    prob_maps: Iterable[ProbabilityMap],
    prob_data_cache: dict[str, tuple[object, tuple[int, int]] | None] | None = None,
    target_cache: dict[str, object] | None = None,
) -> dict:
    """The ``features`` dict a segment should carry after re-measurement.

    Split out of :func:`compute_segment_features_task` so the batched sweep can
    accumulate a chunk of these and write them in one statement. The measuring
    is identical either way; only the SQL around it differs, and keeping one
    body is what stops the two paths drifting into measuring different pixels.

    ``prob_maps`` is passed in rather than queried here because it belongs to
    the *segmentation*, not the object: querying it per object issued one
    identical SELECT for every row in the sweep.
    """
    segmentation = segment.segmentation
    polygon = segment.geometry
    bbox = segment.bbox
    ring_pixels = DEFAULT_OUTSIDE_RING_PIXELS

    try:
        target_key = str(segmentation.id)
        target = target_cache.get(target_key) if target_cache is not None else None
        if target is None:
            target = _get_segmentation_target(segmentation)
            if target_cache is not None:
                target_cache[target_key] = target
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
            segment.id,
            exc,
            exc_info=True,
        )
        measurements = {}

    features = merge_measured_features(segment.features, measurements)

    for prob_map in prob_maps:
        features = _apply_prob_map_stats(
            features,
            prob_map,
            polygon,
            bbox,
            data_cache=prob_data_cache,
        )
    return features


def _segmentation_prob_maps(
    segmentation: ImageSegmentation,
    cache: dict[str, list[ProbabilityMap]] | None,
) -> list[ProbabilityMap]:
    """This segmentation's probability maps, read once per sweep."""
    key = str(segmentation.id)
    if cache is not None and key in cache:
        return cache[key]
    prob_maps = list(ProbabilityMap.objects.filter(segmentation=segmentation))
    if cache is not None:
        cache[key] = prob_maps
    return prob_maps


def measure_segments_batched(
    segment_ids: Sequence[str],
    *,
    cancel_check: Callable[[], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    prob_data_cache: dict[str, tuple[object, tuple[int, int]] | None] | None = None,
    target_cache: dict[str, object] | None = None,
    batch_size: int = SEGMENT_MEASUREMENT_BATCH_SIZE,
) -> int:
    """Re-measure many segments with a bounded number of SQL round trips.

    Both callers -- the background sweep in
    ``handle_refresh_segment_features`` and the Analysis fill-in in
    ``quantem.analysis.loaders.ensure_confirmed_object_measurements`` -- used to
    loop over ids calling the single-object task, so the SQL cost scaled at
    three statements per object. That is tolerable for the handful of outlines
    an edit invalidates, but since Preview began deferring measurement the same
    loop runs over every object in the segmentation, on the Analysis job the
    user is waiting on. This function is the shared batched form: per chunk it
    issues one SELECT, at most one probability-map SELECT per segmentation, and
    one ``bulk_update``.

    Batching the SQL deliberately does not batch the two things the user
    perceives. ``on_progress(done, total)`` is called after every object, so a
    chunk of 200 is not a 200-object silence; the callers own the wording.
    ``cancel_check`` is likewise called per object, so a user who presses Cancel
    on a long sweep is not left waiting for the current chunk of 200 to finish.

    The chunk is written in a ``finally``, so a cancellation part-way through
    one still stores what it had already measured. Losing it would be
    recoverable -- the sweep recomputes which objects are unmeasured -- but a
    user who cancels a long sweep after watching it count to 12 000 should keep
    those 12 000, not the last multiple of 200.

    Returns the number of objects actually measured, which is less than
    ``len(segment_ids)`` when some of them have been deleted since the ids were
    collected.
    """
    if not _segment_feature_task_enabled():
        logger.info(
            "Skipping segment feature measurement for %d object(s) "
            "(QUANTEM_ENABLE_SEGMENT_FEATURE_TASK disabled)",
            len(segment_ids),
        )
        return 0

    ids = [str(segment_id).strip() for segment_id in segment_ids if str(segment_id).strip()]
    total = len(ids)
    if not total:
        return 0

    if prob_data_cache is None:
        prob_data_cache = {}
    if target_cache is None:
        target_cache = {}
    prob_map_cache: dict[str, list[ProbabilityMap]] = {}

    step = max(int(batch_size), 1)
    measured = 0
    for start in range(0, total, step):
        chunk = ids[start : start + step]
        if cancel_check is not None:
            cancel_check()
        # Ordered by bounding box rather than by id so the image-window reads
        # walk the pyramid in row order: neighbouring objects share chunks, and
        # chunk decompression is the part of the read that is not already
        # cached by ``resolve_pyramid``.
        segments = list(
            SegmentObject.objects.select_related("segmentation", "segmentation__asset")
            .filter(id__in=chunk)
            .order_by("bbox_miny", "bbox_minx")
        )
        found = {str(segment.id) for segment in segments}
        for missing_id in chunk:
            if missing_id not in found:
                logger.warning("Segment %s missing for feature task", missing_id)

        written: list[SegmentObject] = []
        try:
            for segment in segments:
                if cancel_check is not None:
                    cancel_check()
                segment.features = _measured_features_for(
                    segment,
                    prob_maps=_segmentation_prob_maps(segment.segmentation, prob_map_cache),
                    prob_data_cache=prob_data_cache,
                    target_cache=target_cache,
                )
                written.append(segment)
                measured += 1
                if on_progress is not None:
                    on_progress(measured, total)
        finally:
            if written:
                with transaction.atomic():
                    SegmentObject.objects.bulk_update(written, ["features"], batch_size=step)
    return measured


def compute_segment_features_task(
    segment_id: str,
    *,
    prob_data_cache: dict[str, tuple[object, tuple[int, int]] | None] | None = None,
    target_cache: dict[str, object] | None = None,
) -> None:
    """Re-measure one segment's geometry/intensity/probability-map features.

    **Preserving, never replacing.** What this task measures is written over the
    keys it measured; everything else the object carries survives -- see
    :func:`quantem.segmentation.features.measure.merge_measured_features`, which
    is the same rule the synchronous writer follows.

    A batch of one, so the single-object callers (an edit that moved one
    outline) and the whole-segmentation sweeps cannot measure differently.
    """
    measure_segments_batched(
        [segment_id],
        prob_data_cache=prob_data_cache,
        target_cache=target_cache,
    )

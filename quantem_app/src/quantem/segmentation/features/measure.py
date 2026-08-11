"""Measure a segment's polygon and store the result on the object.

A hand-drawn object used to be created with ``features = {"sam_score": 1.0}``
and nothing else, so every morphometric column in ``objects.csv`` was blank for
it while ``calibrated=True`` sat in the row beside them — a table that looks
measured and is not. The polygon is right there at create time, and the same
``regionprops`` call that measures a model-extracted object measures a drawn
one, so it is done synchronously on create and on geometry edit rather than
deferred to a queued refresh that is off by default
(``QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS`` defaults to ``0``).

**The same call is not enough; it has to be given the same mask.** It was not.
For a model object the mask is the label mask ``regionprops`` found; for a drawn
one the polygon has to be rasterised first, and ``cv2.fillPoly`` painted a mask
a half-pixel larger all round than the outline it was given. A drawn 10x10
square measured ``area 121, perimeter 40.0, circularity 0.950`` where the model
found ``area 100, perimeter 36.0, circularity 0.970`` for the identical shape --
21% apart in one column of one ``objects.csv``, with ``source_model`` the only
hint. :mod:`quantem.seg_core.rasterize` now defines the one convention both
sides use, and :mod:`quantem.segmentation.tests.test_pixel_area_convention`
pins the two paths to the same numbers.

What is produced is the union of what the two existing extractors produce:

* :func:`quantem.seg_core.extraction.build_segment_from_region` — ``area``,
  ``perimeter``, ``eccentricity``, ``solidity``, ``elongation``
* :mod:`quantem.segmentation.features.geometry` /
  :mod:`~quantem.segmentation.features.intensity` — ``major_axis_length``,
  ``minor_axis_length``, ``feret_diameter_max``, ``intensity_mean`` and the
  intensity percentiles

Everything is in PIXELS; :mod:`quantem.analysis.morphometrics` converts to
physical units and needs ``Asset.pixel_size_nm`` to do it.

``mean_prob`` is deliberately **not** invented for a drawn object. There is no
model probability behind a polygon a person traced, and writing ``0.0`` would
put a number in a paper's table that means "the model was confident this is
background".

**A measurement that fails removes the old one.** Every key this module owns
describes the outline it was measured on, so once that outline changes the
stored value describes a shape that no longer exists. When the re-measure
cannot be done -- the image was moved, deleted, or is on a share that is down --
the keys are cleared rather than left behind, which is the ruling
:func:`quantem.segmentation.tasks._apply_prob_map_stats` already makes about the
probability keys for the very same event: *stale is a different kind of wrong
from fabricated, not a lesser one*. Absent means "not measured"; a number means
"this is the measurement of this shape". Callers are told which objects came out
unmeasured (:class:`MeasurementOutcome`) so the edit does not report an
unqualified success.

The same goes for a measurement that comes back with only *some* of its keys:
:func:`merge_measured_features` clears the ones it did not produce rather than
letting them survive from the previous outline. A half-refreshed object is the
one state a reader cannot detect -- every column is populated, and some of them
describe a shape that no longer exists.

Clearing is also what makes the failure *recoverable*.
``jobs.handlers._unmeasured_segment_ids`` finds work by looking for objects with
no :data:`MEASURED_MARKER_KEY`, so an object whose measurement failed is picked
up by the next feature refresh. Leaving the parent's numbers in place made it
indistinguishable from a correctly measured one, and nothing would ever have
come back for it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shapely.geometry import Polygon

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.task_utils import load_image_array, load_image_roi_array
from quantem.segmentation.features.extraction import compute_segment_features
from quantem.segmentation.models import ImageSegmentation, SegmentObject

logger = logging.getLogger(__name__)

#: Ring width, in pixels, used when cropping the window a segment is measured
#: in. Matches ``quantem.segmentation.tasks.DEFAULT_OUTSIDE_RING_PIXELS`` so the
#: synchronous and queued paths measure the same pixels.
DEFAULT_RING_PIXELS = 10

#: Keys this module owns. Anything else already on ``features`` (the model's
#: ``*_generated`` markers, ``source_model``, stored probability statistics) is
#: preserved, because ``SegmentObject.save`` infers ``source_model`` from those
#: markers and losing them would relabel the object.
MEASUREMENT_KEYS: tuple[str, ...] = (
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "elongation",
    "major_axis_length",
    "minor_axis_length",
    "feret_diameter_max",
    "intensity_mean",
    "intensity_p10",
    "intensity_p50",
    "intensity_p90",
)

#: The key that says an object has been measured at all. Every successful pass
#: writes it -- ``regionprops`` cannot produce any of the others without it --
#: so its absence is the one reliable marker of "never measured".
MEASURED_MARKER_KEY = "area"

#: The probability of the *outline*, under the names the extractor writes it
#: under: ``mean_prob`` plus one ``mean_prob_<map>`` per named probability map
#: (:mod:`quantem.seg_core.extraction`). This module cannot recompute them --
#: the run's probability array is not kept -- so on a geometry edit they are
#: removed, exactly as ``tasks._apply_prob_map_stats`` removes ``prob_<map>_*``
#: for the same event.
OUTLINE_PROBABILITY_KEY = "mean_prob"
OUTLINE_PROBABILITY_KEY_PREFIX = "mean_prob_"

#: What an API says when an edit went through but its measurements did not. It
#: names the consequence (empty columns, not wrong ones) so a client is not left
#: to guess whether the numbers it can see are current.
UNMEASURED_DETAIL = (
    "The outlines were changed, but the image could not be read to measure "
    "them. Their area, perimeter and intensity are now empty rather than "
    "describing the previous outline. Reopen the image once it is available "
    "and re-apply, or run a feature refresh, to fill them in."
)


@dataclass(frozen=True)
class MeasurementOutcome:
    """What one call to :func:`measure_segments` actually managed to measure.

    ``unmeasured`` is the point of it. A caller that has just changed an
    object's geometry has to know whether the numbers now stored describe the
    new outline, because the alternative -- reporting the edit as a plain
    success while the morphometrics are missing -- is what let
    ``POST /segments/remove-area/`` answer ``200 {"created": 1, "updated": 1}``
    for an edit whose measurements never happened.
    """

    updated: int = 0
    unmeasured: tuple[str, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.unmeasured

    def as_payload(self) -> dict | None:
        """The API's ``measurement`` block, or None when everything measured."""
        if self.ok:
            return None
        return {
            "measured": int(self.updated),
            "unmeasured_ids": list(self.unmeasured),
            "detail": self.reason or UNMEASURED_DETAIL,
        }


def clear_measured_features(existing: object) -> dict:
    """``existing`` with every key this module measures removed.

    Used when a measurement could not be taken. The object keeps its identity
    -- ``source_model``, the ``*_generated`` markers, the ``run`` block -- and
    loses only the numbers that described a shape it no longer has.
    """
    features = dict(existing) if isinstance(existing, dict) else {}
    for key in MEASUREMENT_KEYS:
        features.pop(key, None)
    return features


def drop_outline_probability(existing: object) -> dict:
    """``existing`` without the probability measured over the previous outline.

    Called only when the geometry changed. Both halves of a cut used to carry
    the parent's ``mean_prob`` -- one number, measured over an outline twice
    their size, reported for each of them in ``objects.csv`` and used as the
    ``confidence_score`` fallback (:mod:`quantem.segmentation.confidence`). It
    is not recoverable here, so it is removed rather than reattributed.
    """
    features = dict(existing) if isinstance(existing, dict) else {}
    for key in list(features):
        name = str(key)
        if name == OUTLINE_PROBABILITY_KEY or name.startswith(OUTLINE_PROBABILITY_KEY_PREFIX):
            features.pop(key, None)
    return features


def _normalised_sam_score(raw: object) -> float | None:
    """``raw`` as a float, or None when it is not usable as a score.

    Missing is ``None``, never ``0.0``. SAM is not part of this product, so the
    only objects carrying a score are ones a caller supplied it for; a
    hand-drawn object has none, and materialising a ``0.0`` made
    ``api_views/segments/query.py`` report a human-confirmed outline as
    "confidence 0.0" rather than "no confidence".
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # drop NaN


def merge_measured_features(
    existing: object,
    measurements: dict[str, object] | None,
) -> dict:
    """Write measurements onto an object's stored ``features``. **One rule.**

    A measurement pass owns :data:`MEASUREMENT_KEYS` outright: the keys it
    produced are written, and the keys it did not produce are **cleared**.
    *Nothing else on the object is touched.* The model's ``*_generated``
    markers, ``source_model``, the ``run`` identity, the stored probability
    statistics and ``mean_prob`` all survive a re-measure, because none of them
    is a thing this pass measured.

    The clearing half is what closes the gap between this module's docstring
    ("A measurement that fails removes the old one") and what it did. Before,
    the merge only ever *added*: a pass that came back with ``{area,
    perimeter}`` and nothing else left ``intensity_mean``, ``intensity_p50`` and
    ``eccentricity`` sitting there from the shape before the edit, and both
    halves of a cut then reported the parent's ``intensity_mean = 128.0``
    against outlines the parent no longer had. Half-refreshed is the one state
    a reader cannot detect: every column is populated, and some of them describe
    a shape that no longer exists. Today's extractor is all-or-nothing off one
    mask so this cannot be reached through it, but the rule has to hold at the
    function both writers go through, not at the one caller that happens to be
    safe.

    A *re-measure* is not the same event as a *geometry edit*. When the outline
    itself changed, the caller passes ``geometry_changed=True`` to
    :func:`measure_segments` and the probability of the old outline is dropped
    by :func:`drop_outline_probability` on the way past -- see there for why.
    Keeping that rule out here means a plain refresh of an unchanged object
    (a label flip enqueues one for every object in the segmentation) can never
    destroy a probability that is still correct.

    Both feature writers go through here --
    :func:`measure_segments` and
    :func:`quantem.segmentation.tasks.compute_segment_features_task` -- and
    :mod:`quantem.segmentation.tests.test_feature_writers` pins them to it.
    They used to disagree: the queued one rebuilt ``features`` from the
    extractor's output and carried forward only ``sam_score`` and ``run``, so
    one ``POST /segments/remove-area/`` destroyed ``mean_prob`` (a column of
    ``objects.csv``, and the confidence fallback in
    ``serializers/segments.py``) and dropped ``mito_generated`` with it. The
    second loss made the first unreadable: with the marker gone,
    ``analysis.morphometrics._coverage_note`` saw the missing ``mean_prob``
    attributed to ``quantem:mito`` rather than to ``manual``, and reported a
    *destroyed measurement* as *a model that produced no probability*.

    The single exception is ``sam_score``: a stored value that is not a number
    is dropped rather than carried, because every reader of that key wants a
    score and there is no reading of ``"not-a-number"`` that is one.
    """
    features = clear_measured_features(existing)
    if "sam_score" in features:
        score = _normalised_sam_score(features["sam_score"])
        if score is None:
            features.pop("sam_score", None)
        else:
            features["sam_score"] = score
    features.update(measurements or {})
    return features


def load_measurement_window(
    target,
    bbox: Polygon | None,
    *,
    ring_pixels: int = DEFAULT_RING_PIXELS,
) -> tuple[object, tuple[int, int]]:
    """Load just the image window around ``bbox``, plus its (x, y) origin.

    Falls back to the whole image when the bbox or the image metadata is
    unusable, which is the only case where measuring is worth a full-image read.
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

    return load_image_roi_array(target, x0, y0, width, height), (x0, y0)


def measure_polygon(
    target,
    polygon: Polygon,
    *,
    bbox: Polygon | None = None,
    ring_pixels: int = DEFAULT_RING_PIXELS,
) -> dict[str, float]:
    """Shape and intensity measurements for one polygon, in pixels.

    Returns an empty dict when the polygon rasterises to nothing (a zero-area
    ring, or a shape entirely outside the image).
    """
    image_array, image_offset = load_measurement_window(
        target, bbox if bbox is not None else polygon.envelope, ring_pixels=ring_pixels
    )
    features, _ = compute_segment_features(
        polygon,
        image_array,
        ring_pixels,
        image_offset=image_offset,
        bbox_polygon=bbox if bbox is not None else polygon.envelope,
    )
    return features or {}


def _write_features(
    segment: SegmentObject,
    features: dict,
    *,
    clear_confidence: bool,
) -> None:
    """Persist ``features``, and the confidence column that goes with them."""
    updates: dict[str, object] = {"features": features}
    if clear_confidence and segment.confidence_score is not None:
        # ``confidence_score`` is written *from* ``mean_prob``
        # (``seg_core.extraction``), so it is the same measurement of the same
        # vanished outline. Dropping the feature and keeping the column would
        # only move the stale number: ``confidence.segment_confidence_score``
        # reads the column first and would keep answering 0.82 for a piece the
        # model never saw.
        updates["confidence_score"] = None
        segment.confidence_score = None
    SegmentObject.objects.filter(id=segment.id).update(**updates)
    segment.features = features


def measure_segments(
    segmentation: ImageSegmentation,
    segments: Sequence[SegmentObject] | Iterable[SegmentObject],
    *,
    geometry_changed: bool = False,
) -> MeasurementOutcome:
    """Measure each segment and persist the values onto ``features``.

    The asset is opened once for the batch. Everything that is not a
    measurement survives -- ``source_model``, the ``*_generated`` markers, the
    ``run`` block -- and only :data:`MEASUREMENT_KEYS` are replaced.

    Never raises: a measurement failure must not lose the object the user just
    drew. It does, however, **remove** the values it could not refresh, and
    reports the ids in :class:`MeasurementOutcome`, because an object silently
    keeping the previous outline's area is the failure this whole module exists
    to prevent.

    Args:
        geometry_changed: True when the caller has just reshaped these outlines.
            The probability measured over the previous outline
            (``mean_prob``/``mean_prob_<map>``, and the ``confidence_score``
            column derived from it) is then dropped as well: it cannot be
            recomputed from a polygon, and it described a different shape.
    """
    segments = [segment for segment in segments if segment is not None]
    if not segments:
        return MeasurementOutcome()

    def _all_unmeasured(reason: str) -> MeasurementOutcome:
        for segment in segments:
            features = clear_measured_features(segment.features)
            if geometry_changed:
                features = drop_outline_probability(features)
            _write_features(segment, features, clear_confidence=geometry_changed)
        return MeasurementOutcome(
            updated=0,
            unmeasured=tuple(str(segment.id) for segment in segments),
            reason=reason,
        )

    if not segmentation.asset_id:
        logger.warning(
            "Cannot measure segments for segmentation %s: no target asset",
            segmentation.id,
        )
        return _all_unmeasured(
            "This segmentation has no image behind it, so its objects cannot be measured."
        )

    try:
        target = get_asset_openable(segmentation.asset)
    except Exception:
        logger.warning(
            "Cannot measure segments for segmentation %s: image is unavailable",
            segmentation.id,
            exc_info=True,
        )
        return _all_unmeasured(UNMEASURED_DETAIL)

    updated = 0
    unmeasured: list[str] = []
    for segment in segments:
        polygon = segment.geometry
        measurements: dict[str, float] = {}
        if isinstance(polygon, Polygon) and not polygon.is_empty:
            try:
                measurements = measure_polygon(target, polygon, bbox=segment.bbox)
            except Exception:
                logger.warning("Failed to measure segment %s", segment.id, exc_info=True)
                measurements = {}

        # ``MEASURED_MARKER_KEY`` rather than a truthiness test on the dict:
        # ``regionprops`` cannot produce any other key without producing this
        # one, so its absence is what "nothing was measured" means everywhere
        # else in the app -- ``jobs.handlers._unmeasured_segment_ids`` finds
        # work by exactly this test. A dict that came back without it would
        # otherwise have counted as a success while leaving the object looking
        # never-measured to the refresh that is supposed to come back for it.
        measured = measurements.get(MEASURED_MARKER_KEY) is not None
        if measured:
            features = merge_measured_features(segment.features, measurements)
        else:
            # Nothing measurable: an empty polygon, a shape that rasterises to
            # no pixel, or a read that failed for this one object. Whatever was
            # there described the outline before the edit.
            logger.warning(
                "No measurements for segment %s; clearing the previous outline's "
                "values rather than reporting them as this shape's",
                segment.id,
            )
            features = clear_measured_features(segment.features)
            unmeasured.append(str(segment.id))

        if geometry_changed:
            features = drop_outline_probability(features)

        _write_features(segment, features, clear_confidence=geometry_changed)
        if measured:
            updated += 1

    return MeasurementOutcome(
        updated=updated,
        unmeasured=tuple(unmeasured),
        reason=UNMEASURED_DETAIL if unmeasured else "",
    )

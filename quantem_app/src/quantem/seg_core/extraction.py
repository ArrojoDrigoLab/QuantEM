"""
Shared Segment Extraction Helpers
==================================

Common logic for building ExtractedSegment from regionprops regions.
Parameterized by prob_maps dict and generated_flag so every organelle reuses
the same code.

This is where the shape measurements that the analysis suite consumes are
computed: area, perimeter, solidity, eccentricity, elongation, the ellipse axis
lengths, the Feret diameter and the intensity statistics — all from
``skimage.measure.regionprops`` and all in pixels. Converting them to physical
units is the caller's job (it needs ``pixel_size_nm``).

Two rules govern what lands in ``features``:

* **One spelling.** :data:`SEGMENT_FEATURE_KEYS` is the vocabulary. It is what
  :mod:`quantem.analysis.morphometrics` reads and what
  :data:`quantem.segmentation.features.measure.MEASUREMENT_KEYS` writes for a
  hand-drawn object, so a column of ``objects.csv`` means the same thing however
  the object was made. This module used to write ``mean_intensity`` while the
  analysis read ``intensity_mean``, and never wrote the axis lengths or the
  Feret diameter at all: ten of the twenty-seven exported columns were blank for
  every model-produced object while the measurements sat in memory and were
  dropped. ``quantem.analysis.tests.test_feature_vocabulary`` fails if the two
  ends drift apart again.
* **A measurement that was not made is absent, never zero.** ``0.0`` is a
  legitimate value for a probability and for an intensity, so writing it as a
  placeholder puts "the model was confident this is background" into a paper's
  table. Every block below omits its keys rather than filling them in.

``area`` here is ``region.area``, the pixel count of the label mask. The
polygon written beside it is the outline of exactly those pixels, and
:mod:`quantem.seg_core.rasterize` fills it back to exactly those pixels, so this
number survives a re-measure and matches what the same shape measures when a
person draws it. ``perimeter`` is ``regionprops``'s ``perimeter_crofton``
(owner ruling 2026-08-07, matching the drawn-object writer), not the walk over
the boundary pixels and so describes the region inset by half a pixel -- see
:mod:`quantem.segmentation.features.geometry` for what that means when it is
combined with ``area``.

No Django imports. No DB dependencies. The one import out of this package is
:func:`quantem.inference.resample.dequantize_probability`, which is pure numpy
and is the single definition of what a stored probability level means; see
:func:`_as_probability` for why re-spelling ``/255`` here instead would be a
second decision procedure.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.measure import find_contours

from quantem.inference.resample import dequantize_probability

from .types import ExtractedSegment

logger = logging.getLogger(__name__)

#: Shape measurements, in pixels. Written for every region.
SHAPE_FEATURE_KEYS: tuple[str, ...] = (
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "elongation",
    "major_axis_length",
    "minor_axis_length",
    "feret_diameter_max",
)

#: Intensity statistics of the pixels under the object's own mask. Absent when
#: the caller supplied no image to measure.
INTENSITY_FEATURE_KEYS: tuple[str, ...] = (
    "intensity_mean",
    "intensity_p10",
    "intensity_p50",
    "intensity_p90",
)

#: Mean foreground probability under the object's mask. Absent for a hand-drawn
#: object, which has no model behind it.
PROBABILITY_FEATURE_KEYS: tuple[str, ...] = ("mean_prob",)

#: Every measurement key this module can write, and the whole vocabulary
#: :func:`quantem.analysis.morphometrics.derive` reads. Per-model probability
#: means (``mean_prob_<model>``) are named after the caller's models and so
#: cannot be listed here; nothing in the analysis layer reads them.
SEGMENT_FEATURE_KEYS: tuple[str, ...] = (
    *SHAPE_FEATURE_KEYS,
    *INTENSITY_FEATURE_KEYS,
    *PROBABILITY_FEATURE_KEYS,
)

#: Everything above was measured before being switched on, because the gate that
#: produced the fabricated zeros was put there for performance and never
#: revisited. On scikit-image 0.26, 484 objects of ~2300 px each in a 2048x2048
#: frame: the whole function went from 4.93 to 5.38 ms/object — **9%** — to fill
#: ten columns that were blank.
#:
#: Where that goes, per object measured in isolation:
#:
#: ==================================== =========
#: mean probability (bbox crop)             17 us
#: intensity mean + p10/p50/p90            175 us
#: ellipse axis lengths                      0 us  (already computed for elongation)
#: ``feret_diameter_max``                  740 us  (in context; 3800 us cold)
#: ==================================== =========
#:
#: The Feret diameter looks expensive in isolation because 80% of it is
#: scikit-image rasterising the object's convex hull — but ``solidity``, which
#: this function already computed, needs the same hull and ``regionprops``
#: caches it. Asking for the diameter afterwards costs only the contour of that
#: hull and the pairwise distances.
#:
#: Two cheaper Feret routes were rejected. A convex hull over the object's own
#: contour disagrees with the hand-drawn path by up to 6% (2.4 px on a 40 px
#: object), which would put two different definitions in one column. Reducing
#: scikit-image's hull contour to its own hull before the pairwise distance is
#: exactly equal and saves nothing, because the cost is the hull image.


def build_segment_from_region(
    region,
    labels: np.ndarray,
    prob_maps: dict[str, np.ndarray],
    prob: np.ndarray,
    generated_flag: str,
    dx: float,
    dy: float,
    image: np.ndarray | None = None,
    measure_features: bool = True,
) -> ExtractedSegment | None:
    """Build an ExtractedSegment from a regionprops region.

    Shared by every segmenter. Computes shape features, probability means, and
    generates the features dict with a generated_flag marker.

    Args:
        region: A regionprops region object.
        labels: Label image (from measure.label or instance segmentation).
        prob_maps: Dict of named probability maps, e.g. {"DINO": arr}.
        prob: Foreground probability map the instances were extracted from.
        generated_flag: Key name for the generated marker, e.g. "er_generated".
        dx: X coordinate offset (for ROI-to-parent mapping).
        dy: Y coordinate offset.
        image: Intensity image, used when the region carries none of its own.
        measure_features: False for the geometry-first Preview path, which
            publishes outlines before the source image has been loaded. The
            probability means and ``confidence_score`` are still written -- they
            read ``prob``, not the image, and nothing downstream can recreate
            them -- but the intensity and regionprops measurements are left to
            the background job that measures the objects afterwards.

    Returns:
        ExtractedSegment or None if contour extraction fails.
    """
    # Use the region's local mask (bbox crop) to avoid full-frame equality masks.
    minr, minc, maxr, maxc = region.bbox
    local_mask = np.asarray(region.image, dtype=bool)
    # Pad so contours are recoverable even when the object fills its bbox.
    padded_mask = np.pad(local_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    contours = find_contours(padded_mask, 0.5)
    if not contours:
        return None

    contour = max(contours, key=len)

    # Convert contour to polygon coords (y, x -> x, y) with bbox + ROI offsets.
    coords = [
        (
            float(c[1] - 1.0) + float(minc) + dx,
            float(c[0] - 1.0) + float(minr) + dy,
        )
        for c in contour
    ]
    if len(coords) < 4:
        return None

    # Close the polygon
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Centroid
    centroid_y, centroid_x = region.centroid
    centroid_xy = (float(centroid_x) + dx, float(centroid_y) + dy)

    # Bounding box
    bbox_xyxy = (
        float(minc) + dx,
        float(minr) + dy,
        float(maxc) + dx,
        float(maxr) + dy,
    )

    features: dict[str, float | bool] = {}

    # --- Probability -------------------------------------------------------
    # Under the object's own mask, read out of the bbox crop rather than by
    # fancy-indexing the full frame with region.coords: same value, a third of
    # the time, no per-object coordinate array.
    #
    # This block runs *before* the geometry-first return below, and has to. The
    # mean probability is the one measurement here that needs no source image --
    # ``prob`` is the array the threshold was just applied to and is already in
    # memory -- and it is the only one nothing downstream can recreate:
    # :data:`quantem.segmentation.features.measure.MEASUREMENT_KEYS` does not
    # contain ``mean_prob``, and that module is documented as *preserving* an
    # existing value across a re-measure rather than computing one, because the
    # run's probability array is gone by then. Deferring it would therefore have
    # blanked it permanently -- the ``mean_prob`` column of ``objects.csv``, the
    # ``confidence_score`` written from it below, and with it the Uncertain
    # review mode, which selects on ``confidence_score__isnull=False``.
    for name, pmap in prob_maps.items():
        values = _masked_values(pmap, region, minr, minc, maxr, maxc)
        if values is not None:
            features[f"mean_prob_{name.lower()}"] = float(_as_probability(values).mean())

    prob_values = _masked_values(prob, region, minr, minc, maxr, maxc)
    mean_prob = float(_as_probability(prob_values).mean()) if prob_values is not None else None
    if mean_prob is not None:
        features["mean_prob"] = mean_prob

    # The mean probability under the whole object, not the probability of the
    # single centroid pixel: a ring-shaped or concave object can have background
    # at its centroid, which would report a confident detection as a doubtful
    # one. The centroid reading is kept only for the case where the probability
    # map could not be sampled at all.
    if mean_prob is not None:
        confidence_score = mean_prob
    else:
        cy = int(np.clip(round(centroid_y), 0, prob.shape[0] - 1))
        cx = int(np.clip(round(centroid_x), 0, prob.shape[1] - 1))
        confidence_score = float(_as_probability(np.asarray(prob)[cy : cy + 1, cx : cx + 1])[0, 0])

    # Geometry-first Preview deliberately stops here after the outline and the
    # probability above. The expensive intensity and regionprops measurements --
    # the ones that need the source image loaded, which is the cost this path
    # exists to avoid -- are filled by a coalesced background job after the
    # candidates are visible. ``area`` is deliberately not written:
    # :data:`quantem.segmentation.features.measure.MEASURED_MARKER_KEY` is
    # ``area``, so its absence is what tells that job the object is unmeasured.
    if not measure_features:
        features[generated_flag] = True
        return ExtractedSegment(
            polygon_coords=coords,
            centroid_xy=centroid_xy,
            bbox_xyxy=bbox_xyxy,
            area=int(region.area),
            features=features,
            confidence_score=confidence_score,
            region_mask=None,
        )

    # --- Intensity ---------------------------------------------------------
    intensity_values = _region_intensity(region, image, minr, minc, maxr, maxc)
    if intensity_values is not None:
        p10, p50, p90 = np.percentile(intensity_values, (10.0, 50.0, 90.0))
        features["intensity_mean"] = float(intensity_values.mean())
        features["intensity_p10"] = float(p10)
        features["intensity_p50"] = float(p50)
        features["intensity_p90"] = float(p90)

    # --- Shape -------------------------------------------------------------
    major = _prop(region, "axis_major_length", "major_axis_length")
    minor = _prop(region, "axis_minor_length", "minor_axis_length")
    if major is not None:
        features["major_axis_length"] = major
    if minor is not None:
        features["minor_axis_length"] = minor
    if major is not None and minor is not None:
        # max(minor, 1.0) matches quantem.segmentation.features.geometry so a
        # sub-pixel minor axis gives the same number on both paths instead of
        # one of them producing inf.
        features["elongation"] = float(major / max(minor, 1.0))

    feret = _prop(region, "feret_diameter_max")
    if feret is not None:
        features["feret_diameter_max"] = feret

    solidity = _prop(region, "solidity")
    if solidity is None:
        # A degenerate hull (a one-pixel-wide region) makes solidity undefined.
        # 1.0 is the value the hand-drawn path falls back to in the same place,
        # and unlike an intensity or a probability it is a shape ratio whose
        # only meaning here is "no concavity was measurable".
        solidity = 1.0
    features["solidity"] = solidity

    eccentricity = _prop(region, "eccentricity")
    if eccentricity is not None:
        features["eccentricity"] = eccentricity

    features["area"] = int(region.area)
    # perimeter_crofton, matching quantem.segmentation.features.geometry --
    # the drawn-object writer. The two provenances measure the same shape with
    # the same estimator or their rows are not comparable in one objects.csv,
    # which is the defect the pixel-area convention fix removed for `area`.
    # See geometry.py's call site for the numbers behind the estimator ruling.
    perimeter = _prop(region, "perimeter_crofton")
    if perimeter is None:
        perimeter = _prop(region, "perimeter")
    if perimeter is not None:
        features["perimeter"] = perimeter

    features[generated_flag] = True

    return ExtractedSegment(
        polygon_coords=coords,
        centroid_xy=centroid_xy,
        bbox_xyxy=bbox_xyxy,
        area=int(region.area),
        features=features,
        confidence_score=confidence_score,
        region_mask=None,
    )


def _masked_values(
    array: np.ndarray | None,
    region,
    minr: int,
    minc: int,
    maxr: int,
    maxc: int,
) -> np.ndarray | None:
    """Values of ``array`` under the region's mask, or None when unavailable.

    None rather than an empty array or a zero, so the caller omits the key: an
    unsampled probability map and a probability of 0.0 are different facts.
    """
    if array is None:
        return None
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] < maxr or array.shape[1] < maxc:
        return None
    values = array[minr:maxr, minc:maxc][region.image]
    return values if values.size else None


def _as_probability(values: np.ndarray) -> np.ndarray:
    """Probabilities in ``[0, 1]``, whichever form of the map was handed in.

    A model run hands in the float field
    (:meth:`~quantem.inference.segmenter.DinoOrganelleSegmenter.run_dl_inference`
    returns ``NativeProbabilityMap.as_float()``), which passes through. The
    geometry-first replay path hands in the **stored uint8 levels** instead --
    :func:`quantem.seg_core.db.inference.replay_stored_probability_map` with
    ``geometry_only=True`` skips the two image-sized float32 allocations and
    passes ``NativeProbabilityMap.data`` straight through -- and those are
    levels, not probabilities: a mean of 204 would be written into a column that
    means "the model was 0.8 sure", and the same object would export a different
    number depending on which path made it.

    Dequantising here, on the handful of values under one object's mask rather
    than on the whole frame, is what lets the geometry-first path keep this
    measurement without the allocation it exists to avoid. It uses the canonical
    :func:`~quantem.inference.resample.dequantize_probability` -- the same
    operation, in the same float32 precision, that ``as_float()`` applies to the
    whole map -- so the two paths agree bit for bit rather than approximately,
    and a v0.1.6 ``mean_prob`` is comparable with a v0.1.5 one in one
    ``objects.csv``.
    """
    return dequantize_probability(values) if values.dtype == np.uint8 else values


def _region_intensity(
    region,
    image: np.ndarray | None,
    minr: int,
    minc: int,
    maxr: int,
    maxc: int,
) -> np.ndarray | None:
    """Grey values under the object's mask, as float, or None.

    Prefers the intensity image ``regionprops`` was given; falls back to the
    ``image`` argument so a caller that forgot ``intensity_image=`` still gets
    measured intensities instead of four silently missing columns.
    """
    values: np.ndarray | None = None
    if hasattr(region, "image_intensity"):
        try:
            crop = np.asarray(region.image_intensity)
        except Exception:  # pragma: no cover - defensive
            crop = None
        if crop is not None and crop.ndim == 2:
            values = crop[region.image]
    if values is None:
        values = _masked_values(image, region, minr, minc, maxr, maxc)
    if values is None or not values.size:
        return None
    return np.asarray(values, dtype=float)


def _prop(region, name: str, legacy_name: str | None = None) -> float | None:
    """One regionprops scalar as a finite float, or None.

    ``legacy_name`` is the pre-0.26 spelling: reading it emits a
    ``FutureWarning`` and it disappears in scikit-image 2.0, so it is only tried
    when the current name is missing.
    """
    for candidate in (name, legacy_name):
        if not candidate:
            continue
        try:
            value = float(getattr(region, candidate))
        except (AttributeError, ValueError, TypeError, NotImplementedError):
            continue
        except Exception:  # pragma: no cover - degenerate region geometry
            continue
        if np.isfinite(value):
            return value
    return None

"""The measurement the geometry-first Preview path is not allowed to defer.

Preview publishes outlines before the source image has been loaded, and leaves
the intensity and ``regionprops`` measurements to a background job. The mean
foreground probability under an outline looks like one more measurement to
defer, and it is not: it reads the probability array the threshold was applied
to, which is already in memory, and **nothing downstream can recreate it**.
:data:`quantem.segmentation.features.measure.MEASUREMENT_KEYS` does not contain
``mean_prob``; that module carries an existing value across a re-measure and
says so in its own docstring, because the run's probability array is gone by
then. So does :func:`quantem.segmentation.tasks.compute_segment_features_task`,
the coalesced job Preview schedules, and so does Analysis's own fill-in.

Deferring it was therefore permanent loss, for every model object made from
that point on: the ``mean_prob`` column of ``objects.csv`` blank, the coverage
note in ``morphometrics`` blaming the model pack for a measurement that was
never attempted, ``confidence_score`` NULL on every candidate, and the Uncertain
review mode -- which selects on ``confidence_score__isnull=False`` -- empty for
every freshly previewed image.

The second rule here is **one scale**. The measured path is handed the
dequantised float field; the geometry-first replay path is handed the stored
uint8 levels themselves, because skipping the two image-sized float32
allocations is the point of it
(:func:`quantem.seg_core.db.inference.replay_stored_probability_map` with
``geometry_only=True``). A level of 204 is not a probability of 204. The two
paths have to write the *same number* for the same object or a v0.1.6 object and
a v0.1.5 object are not comparable in one ``objects.csv``, and the tests below
pin that as exact equality rather than approximate: both sides dequantise with
:func:`quantem.inference.resample.dequantize_probability`, in the same float32
precision, over the same pixels in the same order.
"""

from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase, TestCase
from skimage.measure import label as sk_label
from skimage.measure import regionprops

from quantem.inference.resample import (
    NativeProbabilityMap,
    dequantize_probability,
    quantize_probability,
)
from quantem.inference.segmenter import DL_MODEL_NAME, DinoMitoSegmenter
from quantem.seg_core.db.segment_writer import write_segments
from quantem.seg_core.extraction import build_segment_from_region
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

GENERATED_FLAG = "mito_generated"
SOURCE_MODEL = "quantem:mito"

#: Levels chosen so the mean is not a round number and a truncating or
#: mean-then-divide implementation would land somewhere else: a blob whose
#: probabilities span 0.60 to 0.96.
_BLOB_LEVELS = (153, 178, 191, 214, 229, 245)


def _stored_map(size: int = 24) -> np.ndarray:
    """A stored uint8 probability map with one clearly-foreground blob."""
    stored = np.zeros((size, size), dtype=np.uint8)
    # Background well under the threshold, so the blob is the only object and
    # the mean below is a mean over the blob's own varied levels.
    stored[:] = 12
    blob = np.array(_BLOB_LEVELS, dtype=np.uint8).reshape(2, 3)
    stored[8:10, 9:12] = blob
    stored[10:12, 9:12] = blob[::-1]
    return stored


def _only_region(mask: np.ndarray):
    labels = sk_label(mask)
    regions = regionprops(labels)
    assert len(regions) == 1, f"fixture should have one object, has {len(regions)}"
    return labels, regions[0]


class GeometryFirstProbabilityTests(SimpleTestCase):
    """``build_segment_from_region`` with ``measure_features=False``."""

    def setUp(self) -> None:
        self.stored = _stored_map()
        self.float_prob = dequantize_probability(self.stored)
        self.labels, self.region = _only_region(self.stored >= 128)
        # An intensity image only the measured path is given, so the assertions
        # below can tell "deferred" apart from "never computed".
        rng = np.random.default_rng(20260817)
        self.image = rng.integers(0, 255, size=self.stored.shape, dtype=np.uint8)

    def _measured(self):
        _, region = _only_region(self.stored >= 128)
        return build_segment_from_region(
            region,
            self.labels,
            {DL_MODEL_NAME: self.float_prob},
            self.float_prob,
            GENERATED_FLAG,
            0.0,
            0.0,
            self.image,
        )

    def _deferred(self):
        _, region = _only_region(self.stored >= 128)
        return build_segment_from_region(
            region,
            self.labels,
            {DL_MODEL_NAME: self.stored},
            self.stored,
            GENERATED_FLAG,
            0.0,
            0.0,
            None,
            measure_features=False,
        )

    def test_geometry_first_still_writes_the_probability_and_confidence(self) -> None:
        segment = self._deferred()
        assert segment is not None
        self.assertIn("mean_prob", segment.features)
        self.assertIn(f"mean_prob_{DL_MODEL_NAME.lower()}", segment.features)
        self.assertIsNotNone(segment.confidence_score)
        # confidence_score *is* mean_prob -- the whole-object reading, not the
        # centroid pixel. SegmentObjectSerializer and the Uncertain endpoint
        # both depend on that identity.
        self.assertEqual(segment.confidence_score, segment.features["mean_prob"])

    def test_geometry_first_probability_is_a_probability_not_a_stored_level(self) -> None:
        segment = self._deferred()
        assert segment is not None
        mean_prob = segment.features["mean_prob"]
        # The blob's levels are 153..245, i.e. 0.6..0.96. Reading the levels
        # straight would put ~204 in a [0, 1] column.
        self.assertGreater(mean_prob, 0.6)
        self.assertLess(mean_prob, 1.0)

    def test_both_paths_write_the_same_number_for_the_same_object(self) -> None:
        """The v0.1.5-vs-v0.1.6 comparability rule, as exact equality."""
        measured = self._measured()
        deferred = self._deferred()
        assert measured is not None and deferred is not None
        self.assertEqual(deferred.features["mean_prob"], measured.features["mean_prob"])
        self.assertEqual(
            deferred.features[f"mean_prob_{DL_MODEL_NAME.lower()}"],
            measured.features[f"mean_prob_{DL_MODEL_NAME.lower()}"],
        )
        self.assertEqual(deferred.confidence_score, measured.confidence_score)

    def test_geometry_first_still_defers_the_measurements_that_need_the_image(self) -> None:
        """The deferral itself is intact -- this is not a fix by un-deferring."""
        deferred = self._deferred()
        measured = self._measured()
        assert deferred is not None and measured is not None
        # ``area`` is MEASURED_MARKER_KEY: its absence is what tells the
        # background job the object has never been measured, so writing it here
        # would silently strand every deferred object as already-done.
        self.assertNotIn("area", deferred.features)
        for key in (
            "perimeter",
            "solidity",
            "eccentricity",
            "elongation",
            "major_axis_length",
            "minor_axis_length",
            "feret_diameter_max",
            "intensity_mean",
            "intensity_p10",
            "intensity_p50",
            "intensity_p90",
        ):
            self.assertNotIn(key, deferred.features, key)
            self.assertIn(key, measured.features, key)
        # The provenance marker still has to be there: SegmentObject.save infers
        # source_model from it, so an object without it is relabelled manual.
        self.assertIs(deferred.features[GENERATED_FLAG], True)

    def test_geometry_first_matches_the_geometry_the_measured_path_produces(self) -> None:
        measured = self._measured()
        deferred = self._deferred()
        assert measured is not None and deferred is not None
        self.assertEqual(deferred.area, measured.area)
        self.assertEqual(deferred.polygon_coords, measured.polygon_coords)
        self.assertEqual(deferred.centroid_xy, measured.centroid_xy)


class ExtractInstancesGeometryTests(SimpleTestCase):
    """The segmenter entry point the dial actually calls.

    ``extract_instances_geometry`` used to pass ``prob_maps={}``, so even with
    the fix above the per-model ``mean_prob_<model>`` key would be missing on
    the dial path while a fresh run wrote it -- breaking the invariant the dial
    is built on, that its objects are the objects a fresh run at that level
    would have produced. The segmenter is constructed but never loaded here: no
    weights are touched, because both entry points work on the array they are
    handed.
    """

    def setUp(self) -> None:
        self.stored = _stored_map()
        self.float_prob = dequantize_probability(self.stored)
        self.segmenter = DinoMitoSegmenter(min_area=1, fg_threshold=0.5)
        # Adopting the stored map is what the replay path does, and it is what
        # makes both extractions threshold the identical bytes.
        self.segmenter.adopt_native_probability_map(NativeProbabilityMap.from_stored(self.stored))

    def test_dial_path_writes_the_same_features_a_fresh_run_would(self) -> None:
        image = np.zeros(self.stored.shape, dtype=np.uint8)
        fresh = self.segmenter.extract_instances(
            self.float_prob,
            image,
            {DL_MODEL_NAME: self.float_prob},
            min_area=1,
        )
        dialled = self.segmenter.extract_instances_geometry(self.stored, min_area=1)

        self.assertEqual(len(dialled), len(fresh))
        self.assertTrue(fresh, "fixture produced no objects")
        for dial_segment, fresh_segment in zip(dialled, fresh, strict=True):
            for key in ("mean_prob", f"mean_prob_{DL_MODEL_NAME.lower()}"):
                self.assertIn(key, dial_segment.features, key)
                self.assertEqual(dial_segment.features[key], fresh_segment.features[key], key)
            self.assertEqual(dial_segment.confidence_score, fresh_segment.confidence_score)

    def test_quantised_float_input_agrees_with_the_stored_bytes(self) -> None:
        """A caller handing in its own float map gets the same answer.

        ``quantize_probability`` round-trips through the same levels the store
        holds, so this is the "no second decision procedure" rule applied to the
        measurement rather than to the threshold.
        """
        requantised = dequantize_probability(quantize_probability(self.float_prob))
        segments = self.segmenter.extract_instances_geometry(self.stored, min_area=1)
        via_float = self.segmenter.extract_instances(
            requantised,
            np.zeros(self.stored.shape, dtype=np.uint8),
            {DL_MODEL_NAME: requantised},
            min_area=1,
        )
        self.assertEqual(len(segments), len(via_float))
        for stored_segment, float_segment in zip(segments, via_float, strict=True):
            self.assertEqual(
                stored_segment.features["mean_prob"],
                float_segment.features["mean_prob"],
            )


class DeferredCandidateRowTests(TestCase):
    """The value has to reach the columns the product reads, not just the dict.

    ``confidence_score`` is a column, not a feature, and it is the one
    ``SegmentationUncertainSegmentsView`` filters on
    (``confidence_score__isnull=False`` over ``INFERRED``/``CANDIDATE``). A
    geometry-first pass that wrote NULL there emptied the Uncertain review mode
    for every freshly previewed image -- with no error anywhere, because an
    empty list is a valid answer to "which objects is the model unsure about".
    """

    def setUp(self) -> None:
        image = create_image_from_test_tiff("Geometry-first fixture", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.stored = _stored_map()
        self.segmenter = DinoMitoSegmenter(min_area=1, fg_threshold=0.5)
        self.segmenter.adopt_native_probability_map(NativeProbabilityMap.from_stored(self.stored))

    def test_a_deferred_candidate_is_written_with_its_probability_and_confidence(self) -> None:
        segments = self.segmenter.extract_instances_geometry(self.stored, min_area=1)
        self.assertTrue(segments, "fixture produced no objects")

        write_segments(
            self.segmentation,
            segments,
            run_identity=None,
            source_model=SOURCE_MODEL,
        )

        rows = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        self.assertEqual(len(rows), len(segments))
        for row in rows:
            self.assertIsNotNone(row.confidence_score)
            self.assertIn("mean_prob", row.features)
            self.assertAlmostEqual(row.confidence_score, row.features["mean_prob"], places=6)
            # Still unmeasured, so the coalesced feature job has work to do.
            self.assertNotIn("area", row.features)

"""The two feature writers must follow one rule, and they did not.

``SegmentObject.features`` is written by exactly two things:

* :func:`quantem.segmentation.features.measure.measure_segments` -- synchronous,
  on create and on geometry edit;
* :func:`quantem.segmentation.tasks.compute_segment_features_task` -- queued, via
  ``refresh_segment_features``.

They measure the same pixels with the same function and disagreed about what to
do with the result. The synchronous one merged; the queued one rebuilt the dict
from the extractor's output and carried forward only ``sam_score`` and ``run``.
So a single ``POST /segments/remove-area/`` with
``QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS=1`` destroyed ``mean_prob`` --
a column of ``objects.csv`` and the confidence fallback in
``serializers/segments.py`` -- and dropped ``mito_generated`` with it.

The second loss is what made the first unreadable. With the marker gone,
``analysis.morphometrics._coverage_note`` attributed the missing ``mean_prob``
to ``quantem:mito`` rather than to ``manual``, and only the ``{manual}`` case is
explained as expected. The bundle therefore reported **a destroyed measurement
as a model that produced no probability**.

These tests pin both writers to the same rule so they cannot drift again.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from django.test import TestCase
from PIL import Image
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.core.local_storage import storage_path
from quantem.segmentation.confidence import segment_confidence_score
from quantem.segmentation.features.measure import (
    MEASUREMENT_KEYS,
    measure_segments,
    merge_measured_features,
)
from quantem.segmentation.models import ImageSegmentation, ProbabilityMap, SegmentObject
from quantem.segmentation.tasks import compute_segment_features_task, prob_map_feature_keys
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256

RUN_IDENTITY = {
    "id": "11111111-2222-3333-4444-555555555555",
    "finished_at": "2026-08-07T09:15:02.481Z",
    "pack_id": "quantem:mito",
    "threshold": 0.45,
    "adapter_id": None,
    "ran_at_nm": 8.0,
    "native_pixel_size_nm": 5.0,
    "min_area": 60,
}

#: Everything a model-produced object carries that is *not* a measurement, and
#: that a re-measure therefore has no business touching.
PROVENANCE_FEATURES: dict[str, object] = {
    "mean_prob": 0.82,
    "mean_prob_dino": 0.79,
    "mito_generated": True,
    "source_model": "quantem:mito",
    "run": RUN_IDENTITY,
    "sam_score": 0.5,
}


def _square_coords(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _square_polygon(x0: int, y0: int, x1: int, y1: int) -> Polygon:
    return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))


class FeatureWriterAgreementTests(TestCase):
    """Both writers preserve, both writers replace, and they agree."""

    def setUp(self):
        self.image = create_small_test_image(
            "Feature writer agreement", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _segment(self, **extra_features) -> SegmentObject:
        polygon = _square_polygon(40, 40, 120, 120)
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            source_model="quantem:mito",
            confidence_score=None,
            features={
                **PROVENANCE_FEATURES,
                # A stale measurement from the outline before the edit.
                "area": 1.0,
                "intensity_mean": 1.0,
                **extra_features,
            },
        )

    def _run_synchronous_writer(self, segment: SegmentObject) -> dict:
        measure_segments(self.segmentation, [segment])
        segment.refresh_from_db()
        return dict(segment.features)

    def _run_queued_writer(self, segment: SegmentObject) -> dict:
        compute_segment_features_task(str(segment.id))
        segment.refresh_from_db()
        return dict(segment.features)

    def test_neither_writer_drops_provenance(self):
        for name, run_writer in (
            ("measure_segments", self._run_synchronous_writer),
            ("compute_segment_features_task", self._run_queued_writer),
        ):
            with self.subTest(writer=name):
                features = run_writer(self._segment())
                for key, expected in PROVENANCE_FEATURES.items():
                    self.assertIn(key, features, f"{name} dropped {key}")
                    self.assertEqual(features[key], expected, f"{name} changed {key}")

    def test_both_writers_replace_the_stale_measurements(self):
        for name, run_writer in (
            ("measure_segments", self._run_synchronous_writer),
            ("compute_segment_features_task", self._run_queued_writer),
        ):
            with self.subTest(writer=name):
                features = run_writer(self._segment())
                for key in MEASUREMENT_KEYS:
                    self.assertIn(key, features, f"{name} did not write {key}")
                # An 80x80 square, not the 1.0 that was sitting there.
                self.assertAlmostEqual(features["area"], 80 * 80, delta=400)

    def test_the_two_writers_produce_the_same_features(self):
        """The strongest form of the rule: same input, same output."""
        synchronous = self._run_synchronous_writer(self._segment())
        queued = self._run_queued_writer(self._segment())
        self.assertEqual(set(synchronous), set(queued))
        for key, value in synchronous.items():
            other = queued[key]
            if isinstance(value, float):
                self.assertAlmostEqual(
                    value, other, places=6, msg=f"writers disagree about {key}"
                )
            else:
                self.assertEqual(value, other, f"writers disagree about {key}")

    def test_an_unusable_sam_score_is_dropped_by_both(self):
        """Not preserved, not turned into a number: every reader of this key
        wants a score, and there is no reading of ``"not-a-number"`` that is
        one."""
        for name, run_writer in (
            ("measure_segments", self._run_synchronous_writer),
            ("compute_segment_features_task", self._run_queued_writer),
        ):
            with self.subTest(writer=name):
                features = run_writer(self._segment(sam_score="not-a-number"))
                self.assertNotIn("sam_score", features)

    def test_merge_never_removes_a_key_that_is_not_a_measurement(self):
        """The rule itself, stated once, without a database.

        This was named ``test_merge_never_removes_a_key_it_did_not_measure``,
        which claimed more than it checked and more than is true: it only ever
        asserted about ``mean_prob`` and ``mito_generated``, neither of which is
        a measurement. A key in ``MEASUREMENT_KEYS`` that the pass did *not*
        return is now cleared -- see the test below for why. Renamed to pin the
        claim the assertions actually make.
        """
        before = {"mean_prob": 0.82, "mito_generated": True, "area": 1.0}
        after = merge_measured_features(before, {"area": 6400.0})
        self.assertEqual(after["mean_prob"], 0.82)
        self.assertIs(after["mito_generated"], True)
        self.assertEqual(after["area"], 6400.0)
        # The input dict is not mutated: callers hold the pre-write value.
        self.assertEqual(before["area"], 1.0)

    def test_merge_clears_the_measurements_the_pass_did_not_produce(self):
        """A measurement pass owns ``MEASUREMENT_KEYS`` outright.

        Half-refreshed is the one state a reader cannot detect: every column is
        populated, and some describe the shape before the edit. The old merge
        only ever added, so ``{area, perimeter}`` left ``intensity_mean`` and
        ``eccentricity`` behind from the previous outline.
        """
        before = {
            "area": 1.0,
            "perimeter": 4.0,
            "intensity_mean": 128.0,
            "eccentricity": 0.1,
            "mean_prob": 0.82,
        }
        after = merge_measured_features(before, {"area": 6400.0, "perimeter": 320.0})
        self.assertEqual(after["area"], 6400.0)
        self.assertEqual(after["perimeter"], 320.0)
        self.assertNotIn("intensity_mean", after)
        self.assertNotIn("eccentricity", after)
        # Not a measurement, so untouched.
        self.assertEqual(after["mean_prob"], 0.82)

    def test_both_writers_clear_what_a_half_done_measurement_left_out(self):
        """Stated at the function both writers go through, so they cannot drift."""
        partial = {"area": 6400.0, "perimeter": 320.0}
        for name, run_writer, patch_target in (
            (
                "measure_segments",
                self._run_synchronous_writer,
                "quantem.segmentation.features.measure.measure_polygon",
            ),
            (
                "compute_segment_features_task",
                self._run_queued_writer,
                "quantem.segmentation.tasks.compute_segment_features",
            ),
        ):
            with self.subTest(writer=name):
                is_task = name == "compute_segment_features_task"
                return_value = (dict(partial), None) if is_task else dict(partial)
                with patch(patch_target, return_value=return_value):
                    features = run_writer(self._segment())
                self.assertEqual(features["area"], 6400.0)
                self.assertNotIn("intensity_mean", features, name)
                self.assertNotIn("eccentricity", features, name)
                # Provenance is not a measurement and is not collateral damage.
                self.assertIs(features["mito_generated"], True, name)


class RemoveAreaKeepsProvenanceAndDropsTheOldProbabilityTests(TestCase):
    """The reported reproduction, end to end -- and the line between the two halves.

    ``POST /segments/remove-area/`` returned 200 with both pieces still carrying
    ``mean_prob=0.82`` and ``mito_generated``, queued ``refresh_segment_features``,
    and after the handler ran the same objects read ``mean_prob: None,
    mito_generated: False``. Losing ``mito_generated`` is what made that
    unreadable: ``analysis.morphometrics._coverage_note`` then attributed the
    missing probability to ``quantem:mito`` rather than to ``manual``, and
    reported a destroyed measurement as a model that produced no probability.
    **Provenance must survive**, and this pins that it does.

    ``mean_prob`` is the other half, and it goes the other way. This test used
    to require that both pieces still read ``0.82`` after the cut. That number
    was measured over an outline twice their size -- one measurement, of a shape
    that no longer exists, reported for each of two smaller ones, in
    ``objects.csv`` and as the ``confidence_score`` fallback. It is not
    recomputable from a polygon, so the cut removes it, which is the ruling
    ``tasks._apply_prob_map_stats`` already makes about ``prob_<map>_*`` for
    exactly this event. Absent means "not measured"; 0.82 meant "the model was
    82% sure about *this* piece", and it never saw this piece.
    """

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Remove area probability", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = _square_polygon(40, 40, 200, 200)
        self.segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            source_model="quantem:mito",
            confidence_score=0.82,
            features={"mean_prob": 0.82, "mito_generated": True, "run": RUN_IDENTITY},
        )

    def _cut_in_two(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/remove-area/",
            # A cut through the middle, splitting the object into two pieces.
            {"areas": [{"geometry_coords": _square_coords(100, 20, 130, 240)}]},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["created"], 1)
        self.assertEqual(response.data["updated"], 1)
        # Everything measured, so nothing to report.
        self.assertIsNone(response.data["measurement"])
        return response

    @patch.dict(
        os.environ, {"QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS": "1"}
    )
    def test_the_queued_refresh_after_a_cut_keeps_every_provenance_key(self):
        from quantem.jobs.constants import JOB_TYPE_REFRESH_SEGMENT_FEATURES
        from quantem.jobs.models import Job

        self._cut_in_two()

        job = Job.objects.filter(type=JOB_TYPE_REFRESH_SEGMENT_FEATURES).first()
        self.assertIsNotNone(job, "the edit did not queue a feature refresh")

        for segment_id in job.payload_json["segment_ids"]:
            compute_segment_features_task(segment_id)

        pieces = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            self.assertIs(piece.features["mito_generated"], True)
            self.assertEqual(piece.features["run"], RUN_IDENTITY)
            self.assertEqual(piece.source_model, "quantem:mito")
            # And the measurements do follow the new outline.
            self.assertLess(piece.features["area"], 160 * 160)

    def test_neither_piece_claims_the_parents_probability(self):
        self._cut_in_two()

        pieces = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            self.assertNotIn("mean_prob", piece.features)
            # The column is written *from* mean_prob by seg_core.extraction, so
            # leaving it behind would only move the stale number: the confidence
            # every endpoint reports reads the column first.
            self.assertIsNone(piece.confidence_score)
            self.assertIsNone(segment_confidence_score(piece))

    def test_a_plain_refresh_of_an_unchanged_outline_keeps_the_probability(self):
        """The rule is scoped to a geometry edit, not to every re-measure.

        Flipping one label enqueues a refresh for *every* object in the
        segmentation. If the drop lived in ``merge_measured_features`` instead
        of behind ``geometry_changed``, that one click would erase a valid
        probability from every object in the image.
        """
        compute_segment_features_task(str(self.segment.id))
        measure_segments(self.segmentation, [self.segment])

        self.segment.refresh_from_db()
        self.assertEqual(self.segment.features["mean_prob"], 0.82)
        self.assertEqual(self.segment.confidence_score, 0.82)


class ProbabilityStatisticsAreNeverFabricatedTests(TestCase):
    """A probability that could not be computed is absent, not ``0.0``.

    ``analysis/tests/test_feature_vocabulary.py`` asserts this property about the
    *other* writer (``seg_core.extraction``) and does not reach this path.
    """

    def setUp(self):
        self.image = create_small_test_image(
            "Probability statistics", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = _square_polygon(40, 40, 120, 120)
        self.segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=None,
            features={},
        )

    def _prob_map(self, *, relative_path: str) -> ProbabilityMap:
        return ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            name="fg",
            file_path=relative_path,
        )

    def _write_map(self, relative_path: str, value: int) -> Path:
        path = storage_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.full((SIZE, SIZE), value, dtype=np.uint8), mode="L"
        ).save(path)
        return path

    def test_a_missing_map_file_stores_no_statistics_at_all(self):
        prob_map = self._prob_map(relative_path="prob_maps/does-not-exist.png")

        compute_segment_features_task(str(self.segment.id))

        self.segment.refresh_from_db()
        for key in prob_map_feature_keys(prob_map.id):
            self.assertNotIn(
                key,
                self.segment.features,
                f"{key} was fabricated for a probability map with no file",
            )

    def test_a_map_that_vanishes_does_not_leave_the_previous_value_behind(self):
        """Stale is a different kind of wrong from fabricated, not a lesser one."""
        relative_path = f"prob_maps/{self.segmentation.id}/fg.png"
        prob_map = self._prob_map(relative_path=relative_path)
        written = self._write_map(relative_path, 204)  # 204/255 ~= 0.8

        compute_segment_features_task(str(self.segment.id))
        self.segment.refresh_from_db()
        mean_key = prob_map_feature_keys(prob_map.id)[0]
        self.assertAlmostEqual(self.segment.features[mean_key], 0.8, places=2)

        written.unlink()
        compute_segment_features_task(str(self.segment.id))

        self.segment.refresh_from_db()
        self.assertNotIn(mean_key, self.segment.features)

"""``recompute_features`` was a documented invalidation that nothing performed.

``services/confirm_batch/feature_refresh.py`` documented the flag as marking
edits that "invalidate segmentation-level aggregates ... even when no individual
segment ids are supplied". ``jobs/handlers.py`` never read it, and both
label-change call sites enqueue ``segment_ids=[]``, so with the trigger on every
label flip queued a job whose payload looped zero times and then reported
*"segment feature refresh complete"* at 100%.

What a label flip actually invalidates is not any one outline -- the geometry
did not move -- but *which* objects the analysis aggregates over. An object that
was never measured contributes blank columns to ``objects.csv`` the moment it
joins that population, so that is what the flag now performs: a sweep for
unmeasured objects. Normally it finds none, and the job says so instead of
claiming a refresh it did not do.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.jobs.constants import JOB_TYPE_REFRESH_SEGMENT_FEATURES
from quantem.jobs.handlers import handle_refresh_segment_features
from quantem.jobs.models import Job
from quantem.segmentation.features.measure import MEASUREMENT_KEYS
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256


class _Reporter:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.progress: float = 0.0

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self.progress = progress
        if message:
            self.messages.append(message)

    def log(self, level: str, message: str) -> None:
        self.messages.append(message)


class _Cancel:
    def check_cancelled(self) -> None:
        return None


class FeatureRefreshSweepTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Feature refresh", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _segment(self, *, features: dict) -> SegmentObject:
        polygon = Polygon(((40, 40), (120, 40), (120, 120), (40, 120), (40, 40)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=None,
            features=features,
        )

    def _run(self, payload: dict) -> tuple[dict, _Reporter]:
        reporter = _Reporter()
        result = handle_refresh_segment_features(payload, reporter, _Cancel())
        return result, reporter

    def test_the_flag_measures_an_object_that_was_never_measured(self):
        unmeasured = self._segment(features={"mito_generated": True})

        result, _reporter = self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [],
                "recompute_features": True,
            }
        )

        self.assertEqual(result["segment_count"], 1)
        self.assertTrue(result["swept_segmentation"])
        unmeasured.refresh_from_db()
        for key in MEASUREMENT_KEYS:
            self.assertIn(key, unmeasured.features)
        # And provenance is still there afterwards.
        self.assertIs(unmeasured.features["mito_generated"], True)

    def test_a_sweep_that_finds_nothing_says_so_rather_than_claiming_a_refresh(self):
        self._segment(features={"area": 6400.0})

        result, reporter = self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [],
                "recompute_features": True,
            }
        )

        self.assertEqual(result["segment_count"], 0)
        self.assertTrue(result["swept_segmentation"])
        self.assertIn("already measured", reporter.messages[-1])
        self.assertNotIn("segment feature refresh complete", reporter.messages)

    def test_a_sweep_leaves_measured_objects_alone(self):
        measured = self._segment(features={"area": 1.0, "mean_prob": 0.82})

        self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [],
                "recompute_features": True,
            }
        )

        measured.refresh_from_db()
        self.assertEqual(measured.features["area"], 1.0)

    def test_explicit_ids_still_win_over_the_sweep(self):
        named = self._segment(features={"area": 1.0})

        result, _reporter = self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [str(named.id)],
                "recompute_features": True,
            }
        )

        self.assertEqual(result["segment_count"], 1)
        self.assertFalse(result["swept_segmentation"])
        named.refresh_from_db()
        self.assertAlmostEqual(named.features["area"], 80 * 80, delta=400)

    @patch.dict(os.environ, {"QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS": "1"})
    def test_a_label_flip_queues_a_job_that_has_something_to_do(self):
        unmeasured = self._segment(features={})
        response = self.client.post(
            f"/api/segments/{unmeasured.id}/label/",
            {"label_state": "EXCLUDED"},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)

        job = Job.objects.filter(type=JOB_TYPE_REFRESH_SEGMENT_FEATURES).get()
        self.assertTrue(job.payload_json["recompute_features"])

        result, _reporter = self._run(job.payload_json)
        self.assertEqual(result["segment_count"], 1)

    @patch.dict(os.environ, {"QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS": "1"})
    def test_a_proofreading_session_does_not_pile_up_identical_sweeps(self):
        """Hundreds of label flips are hundreds of requests for one sweep."""
        segments = []
        for index in range(4):
            offset = 10 * index
            polygon = Polygon(
                (
                    (140 + offset, 20),
                    (160 + offset, 20),
                    (160 + offset, 40),
                    (140 + offset, 40),
                    (140 + offset, 20),
                )
            )
            segments.append(
                SegmentObject.objects.create(
                    segmentation=self.segmentation,
                    geometry=polygon,
                    centroid=polygon.centroid,
                    bbox=polygon.envelope,
                    label_state="CANDIDATE",
                    features={"area": 400.0},
                )
            )

        for segment in segments:
            response = self.client.post(
                f"/api/segments/{segment.id}/label/",
                {"label_state": "CONFIRMED"},
                format="json",
            )
            self.assertEqual(response.status_code, 200, response.data)

        self.assertEqual(Job.objects.filter(type=JOB_TYPE_REFRESH_SEGMENT_FEATURES).count(), 1)

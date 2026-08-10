"""One object, one confidence, whichever endpoint you ask.

``GET /segments/at-point`` and ``POST /segments/query-region`` returned
different answers for the same row. The serializer behind the first fell back
from a NULL ``confidence_score`` to ``features["mean_prob"]``; the second fell
back to ``features["sam_score"]`` only, under a comment claiming it returned
null "when the object has no score of any kind" -- while ``features`` was in its
own ``.only(...)`` list. Observed on one object with ``confidence_score=NULL``
and ``features["mean_prob"]=0.82``: 0.82 from one, null from the other, and the
click-ranking key in the same module treated it as unscored.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.segmentation.confidence import (
    confidence_from_features,
    segment_confidence_score,
)
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256


def _polygon(x0: int, y0: int, x1: int, y1: int) -> Polygon:
    return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))


class ConfidenceRuleTests(TestCase):
    def test_the_column_wins(self):
        class _Segment:
            confidence_score = 0.31
            features = {"mean_prob": 0.82, "sam_score": 0.4}

        self.assertAlmostEqual(segment_confidence_score(_Segment()), 0.31)

    def test_mean_prob_is_the_first_fallback(self):
        """It is the measurement the column itself is filled from."""
        self.assertAlmostEqual(
            confidence_from_features({"mean_prob": 0.82, "sam_score": 0.4}), 0.82
        )

    def test_sam_score_is_used_when_that_is_all_there_is(self):
        self.assertAlmostEqual(confidence_from_features({"sam_score": 0.4}), 0.4)

    def test_no_score_is_none_and_never_zero(self):
        for features in ({}, {"area": 12.0}, {"mean_prob": None}, None, "nonsense"):
            with self.subTest(features=features):
                self.assertIsNone(confidence_from_features(features))

    def test_an_unparseable_value_is_no_score_rather_than_a_number(self):
        self.assertIsNone(confidence_from_features({"mean_prob": "very sure"}))
        # ``True`` is an int in Python and would otherwise read as 1.0.
        self.assertIsNone(confidence_from_features({"mean_prob": True}))


class ConfidenceAcrossEndpointsTests(TestCase):
    """The reported disagreement, from both endpoints, on the same row."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Confidence agreement", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = _polygon(40, 40, 120, 120)
        self.segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            source_model="quantem:mito",
            confidence_score=None,
            features={"mean_prob": 0.82},
        )
        self.base = f"/api/segmentations/{self.segmentation.id}"

    def _at_point(self) -> list[dict]:
        response = self.client.get(f"{self.base}/segments/at-point?x=60&y=60")
        self.assertEqual(response.status_code, 200, response.data)
        return response.data

    def _query_region(self) -> list[dict]:
        response = self.client.post(
            f"{self.base}/segments/query-region",
            {"bbox": {"x0": 0, "y0": 0, "x1": SIZE, "y1": SIZE}},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        return response.data["segments"]

    def test_both_endpoints_report_the_same_confidence(self):
        at_point = self._at_point()
        region = self._query_region()
        self.assertEqual(len(at_point), 1)
        self.assertEqual(len(region), 1)
        self.assertAlmostEqual(at_point[0]["confidence_score"], 0.82)
        self.assertAlmostEqual(region[0]["confidence_score"], 0.82)

    def test_an_object_with_no_score_reads_null_from_both(self):
        SegmentObject.objects.filter(id=self.segment.id).update(features={})
        self.assertIsNone(self._at_point()[0]["confidence_score"])
        self.assertIsNone(self._query_region()[0]["confidence_score"])

    def test_a_scored_object_outranks_an_unscored_one_under_the_cursor(self):
        """The ranking key had the same gap: mean_prob was invisible to it, so
        an object carrying 0.82 sorted as unscored."""
        drawn = _polygon(30, 30, 130, 130)  # larger, and covers the same point
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=drawn,
            centroid=drawn.centroid,
            bbox=drawn.envelope,
            label_state="CONFIRMED",
            source_model="manual",
            confidence_score=None,
            features={},
        )

        first = self._at_point()[0]
        self.assertEqual(first["id"], str(self.segment.id))
        self.assertAlmostEqual(first["confidence_score"], 0.82)

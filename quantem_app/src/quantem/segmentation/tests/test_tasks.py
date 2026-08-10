from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.tasks import DEFAULT_OUTSIDE_RING_PIXELS, compute_segment_features_task
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff


class ComputeSegmentFeaturesTaskTests(TestCase):
    def setUp(self) -> None:
        self.image = create_image_from_test_tiff("Segment Task Feature Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _create_segment(self, *, features: dict) -> SegmentObject:
        polygon = Polygon(
            (
                (100, 100),
                (110, 100),
                (110, 110),
                (100, 110),
                (100, 100),
            )
        )
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=None,
            features=features,
        )

    @patch("quantem.segmentation.tasks.compute_segment_features")
    @patch("quantem.segmentation.tasks.load_image_array")
    @patch("quantem.segmentation.tasks.load_image_roi_array")
    def test_feature_task_uses_roi_window_and_reuses_sam_score(
        self,
        load_image_roi_array_mock,
        load_image_array_mock,
        compute_segment_features_mock,
    ):
        segment = self._create_segment(features={"sam_score": "0.75"})
        load_image_roi_array_mock.return_value = np.zeros((34, 34), dtype=np.uint8)
        compute_segment_features_mock.return_value = ({"area": 123.0}, None)

        compute_segment_features_task(str(segment.id))

        load_image_array_mock.assert_not_called()
        load_image_roi_array_mock.assert_called_once_with(
            self.image,
            88,
            88,
            34,
            34,
        )
        args, kwargs = compute_segment_features_mock.call_args
        self.assertEqual(args[2], DEFAULT_OUTSIDE_RING_PIXELS)
        self.assertEqual(kwargs["image_offset"], (88, 88))

        segment.refresh_from_db()
        self.assertAlmostEqual(float(segment.features["sam_score"]), 0.75)
        self.assertAlmostEqual(float(segment.features["area"]), 123.0)

    @patch("quantem.segmentation.tasks.compute_segment_features")
    @patch("quantem.segmentation.tasks.load_image_roi_array")
    def test_feature_task_leaves_a_missing_sam_score_missing(
        self,
        load_image_roi_array_mock,
        compute_segment_features_mock,
    ):
        """An object with no score keeps having no score.

        This asserted ``sam_score == 0.0`` and was documenting the defect: SAM
        is not in this product, so a hand-drawn object has no score, and
        materialising a 0.0 here made it come back from
        ``/segments/query-region`` as ``confidence_score: 0.0`` -- the model's
        lowest possible certainty about a shape a human drew and confirmed.
        """
        segment = self._create_segment(features={"foo": "bar"})
        load_image_roi_array_mock.return_value = np.zeros((34, 34), dtype=np.uint8)
        compute_segment_features_mock.return_value = ({}, None)

        compute_segment_features_task(str(segment.id))

        segment.refresh_from_db()
        self.assertNotIn("sam_score", segment.features)

    @patch("quantem.segmentation.tasks.compute_segment_features")
    @patch("quantem.segmentation.tasks.load_image_roi_array")
    def test_feature_task_drops_a_stored_unparseable_sam_score(
        self,
        load_image_roi_array_mock,
        compute_segment_features_mock,
    ):
        segment = self._create_segment(features={"sam_score": "not-a-number"})
        load_image_roi_array_mock.return_value = np.zeros((34, 34), dtype=np.uint8)
        compute_segment_features_mock.return_value = ({}, None)

        compute_segment_features_task(str(segment.id))

        segment.refresh_from_db()
        self.assertNotIn("sam_score", segment.features)

    @patch("quantem.segmentation.tasks.compute_segment_features")
    @patch("quantem.segmentation.tasks.load_image_roi_array")
    def test_feature_task_preserves_the_run_that_made_the_object(
        self,
        load_image_roi_array_mock,
        compute_segment_features_mock,
    ):
        """Recomputing measurements must not erase provenance.

        ``compute_segment_features`` returns geometry and intensity only, and
        the task overwrites ``features`` wholesale with what it returns. Without
        this the first feature refresh would silently turn a model-produced
        object into one that reads as hand-drawn.
        """
        run = {
            "id": "11111111-2222-3333-4444-555555555555",
            "finished_at": "2026-08-07T09:15:02.481Z",
            "pack_id": "quantem:mito",
            "threshold": 0.45,
            "adapter_id": None,
            "ran_at_nm": 8.0,
            "native_pixel_size_nm": 5.0,
            "min_area": 60,
        }
        segment = self._create_segment(features={"run": run})
        load_image_roi_array_mock.return_value = np.zeros((34, 34), dtype=np.uint8)
        compute_segment_features_mock.return_value = ({"area": 42.0}, None)

        compute_segment_features_task(str(segment.id))

        segment.refresh_from_db()
        self.assertEqual(segment.features["run"], run)
        self.assertAlmostEqual(float(segment.features["area"]), 42.0)

    @patch.dict(os.environ, {"QUANTEM_ENABLE_SEGMENT_FEATURE_TASK": "0"})
    @patch("quantem.segmentation.tasks.compute_segment_features")
    @patch("quantem.segmentation.tasks.load_image_array")
    @patch("quantem.segmentation.tasks.load_image_roi_array")
    def test_feature_task_flag_disables_all_feature_work(
        self,
        load_image_roi_array_mock,
        load_image_array_mock,
        compute_segment_features_mock,
    ):
        segment = self._create_segment(features={"sam_score": 0.22, "existing": 5.0})

        compute_segment_features_task(str(segment.id))

        load_image_roi_array_mock.assert_not_called()
        load_image_array_mock.assert_not_called()
        compute_segment_features_mock.assert_not_called()

        segment.refresh_from_db()
        self.assertEqual(float(segment.features["sam_score"]), 0.22)
        self.assertEqual(float(segment.features["existing"]), 5.0)

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from django.test import TestCase

from quantem.assets.utils import create_roi_image_from_image
from quantem.seg_core.types import InferenceResult
from quantem.segmentation.models import ImageSegmentation, SegmentationConfig
from quantem.segmentation.organelle_tasks import (
    run_segmentation_full_task,
    run_segmentation_roi_task,
)
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_mitochondria_type,
)
from quantem.testing import create_image_from_test_tiff


class OrganelleTaskInstanceParamsTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Organelle Task Param Test Image")
        self.mito_segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.mito_config = SegmentationConfig.objects.create(
            segmentation=self.mito_segmentation,
            instance_params={
                "center_min_distance": 11,
                "center_confidence_threshold": 0.41,
                "segmentation_threshold": 0.63,
                "downsampling_factor": 2,
            },
        )
        self.er_segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
        )
        SegmentationConfig.objects.create(segmentation=self.er_segmentation)

    def _mock_inference_result(self):
        return (
            InferenceResult(
                prob_maps={"mock": np.zeros((8, 8), dtype=np.float32)},
                prob=np.zeros((8, 8), dtype=np.float32),
            ),
            np.zeros((8, 8), dtype=np.uint8),
        )

    def _mock_inference_with_status_message(self, *args, **kwargs):
        on_status = kwargs.get("on_status")
        if on_status is not None:
            on_status("RUNNING_INFERENCE", 100.0, "ResNet34: 100% (Tile 120/120)")
        return self._mock_inference_result()

    @patch("quantem.segmentation.organelle_tasks.persist_run_probability_maps")
    @patch("quantem.segmentation.organelle_tasks.run_inference_for_segmentation")
    @patch("quantem.segmentation.organelle_tasks.get_segmenter")
    def test_full_image_run_uses_persisted_instance_params(
        self,
        mock_get_segmenter,
        mock_run_inference,
        mock_persist,
    ):
        mock_get_segmenter.return_value = SimpleNamespace(name="mito")
        mock_run_inference.return_value = self._mock_inference_result()
        mock_persist.return_value = [object()]

        run_segmentation_full_task(
            segmentation_id=str(self.mito_segmentation.id),
            segmentation_type=self.mito_segmentation.segmentation_type.internal_name,
        )

        kwargs = mock_get_segmenter.call_args.kwargs
        self.assertEqual(
            kwargs["instance_params"],
            self.mito_config.get_instance_params(),
        )

    @patch("quantem.segmentation.organelle_tasks.persist_run_probability_maps")
    @patch("quantem.segmentation.organelle_tasks.run_inference_for_segmentation")
    @patch("quantem.segmentation.organelle_tasks.get_segmenter")
    def test_roi_run_uses_persisted_instance_params_and_roi(
        self,
        mock_get_segmenter,
        mock_run_inference,
        mock_persist,
    ):
        roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 96),
            height=min(self.image.height, 96),
            source="AUTO",
        )
        mock_get_segmenter.return_value = SimpleNamespace(name="mito")
        mock_run_inference.return_value = self._mock_inference_result()
        mock_persist.return_value = [object()]

        run_segmentation_roi_task(
            segmentation_id=str(self.mito_segmentation.id),
            segmentation_type=self.mito_segmentation.segmentation_type.internal_name,
            roi_id=str(roi.id),
        )

        kwargs = mock_get_segmenter.call_args.kwargs
        self.assertEqual(
            kwargs["instance_params"],
            self.mito_config.get_instance_params(),
        )
        self.assertEqual(mock_run_inference.call_args.args[3].id, roi.id)

    @patch("quantem.segmentation.organelle_tasks.persist_run_probability_maps")
    @patch("quantem.segmentation.organelle_tasks.run_inference_for_segmentation")
    @patch("quantem.segmentation.organelle_tasks.get_segmenter")
    def test_er_run_does_not_pass_instance_params(
        self,
        mock_get_segmenter,
        mock_run_inference,
        mock_persist,
    ):
        mock_get_segmenter.return_value = SimpleNamespace(name="er")
        mock_run_inference.return_value = self._mock_inference_result()
        mock_persist.return_value = [object()]

        run_segmentation_full_task(
            segmentation_id=str(self.er_segmentation.id),
            segmentation_type=self.er_segmentation.segmentation_type.internal_name,
        )

        kwargs = mock_get_segmenter.call_args.kwargs
        self.assertNotIn("instance_params", kwargs)

    @patch("quantem.segmentation.organelle_tasks.persist_run_probability_maps")
    @patch("quantem.segmentation.organelle_tasks.run_inference_for_segmentation")
    @patch("quantem.segmentation.organelle_tasks.get_segmenter")
    def test_full_image_status_forwards_tile_text_to_reporter(
        self,
        mock_get_segmenter,
        mock_run_inference,
        mock_persist,
    ):
        mock_get_segmenter.return_value = SimpleNamespace(name="mito")
        mock_run_inference.side_effect = self._mock_inference_with_status_message
        mock_persist.return_value = [object()]
        reporter = Mock()

        run_segmentation_full_task(
            segmentation_id=str(self.mito_segmentation.id),
            segmentation_type=self.mito_segmentation.segmentation_type.internal_name,
            reporter=reporter,
        )

        recorded_messages = [
            call.kwargs.get("message")
            for call in reporter.update.call_args_list
            if call.kwargs.get("message")
        ]
        self.assertTrue(
            any("Tile 120/120" in message for message in recorded_messages)
        )

    @patch("quantem.segmentation.organelle_tasks.persist_run_probability_maps")
    @patch("quantem.segmentation.organelle_tasks.run_inference_for_segmentation")
    @patch("quantem.segmentation.organelle_tasks.get_segmenter")
    def test_roi_status_does_not_forward_tile_text_to_reporter(
        self,
        mock_get_segmenter,
        mock_run_inference,
        mock_persist,
    ):
        roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 96),
            height=min(self.image.height, 96),
            source="AUTO",
        )
        mock_get_segmenter.return_value = SimpleNamespace(name="mito")
        mock_run_inference.side_effect = self._mock_inference_with_status_message
        mock_persist.return_value = [object()]
        reporter = Mock()

        run_segmentation_roi_task(
            segmentation_id=str(self.mito_segmentation.id),
            segmentation_type=self.mito_segmentation.segmentation_type.internal_name,
            roi_id=str(roi.id),
            reporter=reporter,
        )

        recorded_messages = [
            call.kwargs.get("message")
            for call in reporter.update.call_args_list
            if call.kwargs.get("message")
        ]
        self.assertFalse(any("Tile" in message for message in recorded_messages))

    @patch("quantem.segmentation.organelle_tasks.persist_run_probability_maps")
    @patch("quantem.segmentation.organelle_tasks.run_inference_for_segmentation")
    @patch("quantem.segmentation.organelle_tasks.get_segmenter")
    def test_force_recompute_flag_is_forwarded_to_inference(
        self,
        mock_get_segmenter,
        mock_run_inference,
        mock_persist,
    ):
        mock_get_segmenter.return_value = SimpleNamespace(name="mito")
        mock_run_inference.return_value = self._mock_inference_result()
        mock_persist.return_value = [object()]

        run_segmentation_full_task(
            segmentation_id=str(self.mito_segmentation.id),
            segmentation_type=self.mito_segmentation.segmentation_type.internal_name,
            force_recompute_prob_maps=True,
        )

        self.assertTrue(
            mock_run_inference.call_args.kwargs["force_recompute_prob_maps"]
        )

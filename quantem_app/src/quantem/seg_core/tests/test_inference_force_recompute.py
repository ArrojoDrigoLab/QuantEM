"""Cache/streaming behaviour of run_inference_for_segmentation.

These are DB-free: the ImageSegmentation, SegmentationConfig and asset openable
are mocks, and every module-level collaborator is patched. That keeps the test
meaningful without a database or image fixtures.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import SimpleTestCase

from quantem.seg_core.db.inference import run_inference_for_segmentation
from quantem.seg_core.types import InferenceResult


def _segmentation() -> SimpleNamespace:
    return SimpleNamespace(id="seg-1", asset_id="asset-1", asset=object())


def _openable(height: int = 16, width: int = 16) -> SimpleNamespace:
    return SimpleNamespace(height=height, width=width)


class _DummySegmenter:
    name = "mito"
    generated_flag = "mito_generated"
    prob_map_prefix = "mito"

    def load_models(self) -> None:
        return None

    def get_dl_model_names(self) -> list[str]:
        return ["DINO"]

    def predict(self, image, cached_prob_maps=None, **kwargs):
        _ = (image, cached_prob_maps, kwargs)
        prob = np.full((16, 16), 0.75, dtype=np.float32)
        return InferenceResult(prob_maps={"DINO": prob}, prob=prob)

    def get_probability_map_metadata(self, model_name: str) -> dict[str, object]:
        return {"family": "quantem", "model_name": model_name}


class _DummyStreamingSegmenter:
    name = "mito"
    generated_flag = "mito_generated"
    prob_map_prefix = "mito"
    supports_image_file_prediction = True
    persist_probability_maps = False

    def __init__(self):
        self.predict_from_image_file_called = False

    def load_models(self) -> None:
        return None

    def get_dl_model_names(self) -> list[str]:
        return []

    def estimate_image_file_prediction_units(
        self,
        image_shape: tuple[int, int],
    ) -> int | None:
        _ = image_shape
        return 4

    def predict(self, image, cached_prob_maps=None, **kwargs):
        raise AssertionError("predict() must not be used for streaming segmenters")

    def predict_from_image_file(self, image_file, cached_prob_maps=None, **kwargs):
        _ = (image_file, cached_prob_maps, kwargs)
        self.predict_from_image_file_called = True
        return InferenceResult(
            prob_maps={},
            prob=np.zeros((1, 1), dtype=np.float32),
            extracted_segments=[],
            artifacts={"mode": "tiled"},
        )


class ForceRecomputeInferenceTests(SimpleTestCase):
    @patch("quantem.seg_core.db.inference.save_probability_map")
    @patch("quantem.seg_core.db.inference.prob_map_file_exists")
    @patch("quantem.seg_core.db.inference.load_prob_map_from_path")
    @patch("quantem.seg_core.db.inference.load_image_array")
    @patch("quantem.seg_core.db.inference.get_asset_openable")
    def test_force_recompute_bypasses_cache_and_rewrites_maps(
        self,
        mock_get_asset_openable,
        mock_load_image_array,
        mock_load_prob_map,
        mock_prob_map_exists,
        mock_save_probability_map,
    ):
        mock_get_asset_openable.return_value = _openable()
        mock_load_image_array.return_value = (np.zeros((16, 16), dtype=np.uint8), 0.0)
        mock_prob_map_exists.return_value = True

        run_inference_for_segmentation(
            _DummySegmenter(),
            _segmentation(),
            MagicMock(),
            force_recompute_prob_maps=True,
        )

        mock_load_prob_map.assert_not_called()
        saved_models = [call.args[1] for call in mock_save_probability_map.call_args_list]
        self.assertEqual(saved_models, ["DINO"])

    @patch("quantem.seg_core.db.inference.load_image_array")
    @patch("quantem.seg_core.db.inference.get_asset_openable")
    def test_streaming_segmenter_skips_full_image_load(
        self,
        mock_get_asset_openable,
        mock_load_image_array,
    ):
        mock_get_asset_openable.return_value = _openable(65536, 65536)
        segmenter = _DummyStreamingSegmenter()

        result, img_array = run_inference_for_segmentation(
            segmenter,
            _segmentation(),
            MagicMock(),
        )

        mock_load_image_array.assert_not_called()
        self.assertTrue(segmenter.predict_from_image_file_called)
        self.assertEqual(result.artifacts, {"mode": "tiled"})
        self.assertEqual(img_array.shape, (1, 1))

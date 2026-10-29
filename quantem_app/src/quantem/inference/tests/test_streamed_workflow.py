from unittest.mock import patch

import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient

from quantem.assets.canonical_decode import decode_canonical_plane
from quantem.assets.ngff import bounded_ngff_build_resources, build_and_publish
from quantem.inference import engine
from quantem.inference.mapped_prediction import PyramidPlane
from quantem.inference.segmenter import DinoMitoSegmenter
from quantem.inference.specs import MODEL_SPECS
from quantem.seg_core.db.extraction import extract_and_save_segments
from quantem.seg_core.db.inference import run_inference_for_segmentation
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.overlay_ngff import rebuild_overlay_full
from quantem.segmentation.prob_maps.persistence import persist_run_probability_maps
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


class StreamedWorkflowTests(TestCase):
    def test_pyramid_through_prediction_extraction_storage_and_overlay(self):
        image = create_small_test_image(width=601, height=701)
        with bounded_ngff_build_resources():
            build_and_publish(image, decode_canonical_plane(image.path))
        seg = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_mitochondria_type()
        )

        def forward(tile):
            probability = np.full(tile.shape, 0.05, dtype=np.float32)
            probability[80:180, 80:180] = 0.9
            return probability

        engine.clear_model_cache()
        self.addCleanup(engine.clear_model_cache)
        engine.cache_model(
            engine.LoadedModel(
                spec=MODEL_SPECS["quantem:mito"], device="cpu", module=None, forward=forward
            )
        )
        segmenter = DinoMitoSegmenter(source_model="quantem:mito", device="cpu")
        with patch("quantem.inference.segmenter.STREAMING_MIN_PIXELS", 1):
            result, pixels = run_inference_for_segmentation(segmenter, seg, None)
        assert isinstance(pixels, PyramidPlane)
        assert not isinstance(result.prob, np.ndarray)
        stored = persist_run_probability_maps(
            segmentation=seg, segmenter=segmenter, prob_maps=result.prob_maps
        )
        assert len(stored) == 1
        count = extract_and_save_segments(segmenter, seg, result, pixels)
        assert count > 0
        state = rebuild_overlay_full(seg, desired_revision=1)
        token = f"{state.bundle_version}-{state.applied_revision}"
        client = APIClient()
        response = client.get(f"/api/segmentations/{seg.id}/overlay-lut/?rev={token}")
        assert response.status_code == 200
        tile = client.get(f"/segmentation-overlays/{seg.id}.zarr/labels/0/0.0?rev={token}")
        assert tile.status_code == 200
        if tile.streaming:
            assert b"".join(tile.streaming_content)

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from django.test import TestCase

from quantem.inference.postprocess import close_and_fill
from quantem.jobs.handlers.rethreshold import handle_reextract_at_include_level
from quantem.seg_core.types import InferenceResult
from quantem.segmentation.global_masks import load_global_mask
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_er_type
from quantem.testing import create_small_test_image


class _Reporter:
    job_id = "er-global-test"

    def update(self, **_kwargs):
        return None

    def log(self, *_args):
        return None


class _Cancel:
    def check_cancelled(self):
        return None


class ErGlobalApplyTests(TestCase):
    def setUp(self):
        image = create_small_test_image("ER ring", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_er_type(),
        )

    def test_er_apply_retains_ring_hole_while_object_postprocess_still_fills_it(self):
        probability = np.zeros((64, 64), dtype=np.float32)
        probability[8:56, 8:56] = 0.9
        probability[20:44, 20:44] = 0.0
        result = InferenceResult(prob_maps={"DINO": probability}, prob=probability)
        segmenter = SimpleNamespace(
            source_model="quantem:er",
            _organelle=SimpleNamespace(close_radius=1),
            _foreground_mask=lambda values: np.asarray(values) >= 0.5,
        )

        with (
            patch(
                "quantem.jobs.handlers.rethreshold.get_segmenter_or_none",
                return_value=segmenter,
            ),
            patch(
                "quantem.jobs.handlers.rethreshold.replay_stored_probability_map",
                return_value=(result, np.zeros((64, 64), dtype=np.uint8)),
            ),
        ):
            outcome = handle_reextract_at_include_level(
                {
                    "segmentation_id": str(self.segmentation.id),
                    "include_level": 0.5,
                    "source_model": "quantem:er",
                },
                _Reporter(),
                _Cancel(),
            )

        mask = load_global_mask(self.segmentation)
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[32, 32], "ER Apply must not fill a real ring hole")
        self.assertEqual(outcome["segment_count"], None)
        self.assertFalse(SegmentObject.objects.filter(segmentation=self.segmentation).exists())

        object_mask = close_and_fill(probability >= 0.5, close_radius=1)
        self.assertTrue(
            object_mask[32, 32],
            "the mitochondria/nucleus/lipid-droplet postprocess must keep hole fill",
        )

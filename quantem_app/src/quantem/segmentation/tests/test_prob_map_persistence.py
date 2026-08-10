import numpy as np
from django.test import TestCase
from PIL import Image

from quantem.core.config import STORAGE_DIR
from quantem.seg_core.db.prob_maps import (
    get_prob_map_file_path,
    load_prob_map_from_path,
    prob_map_file_exists,
    save_probability_map,
)
from quantem.segmentation.models import ProbabilityMap
from quantem.testing import (
    create_image_from_test_tiff,
    create_mitochondria_segmentation,
    create_roi,
)


class ProbabilityMapPersistenceTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Probability Map Persistence Test")
        self.segmentation = create_mitochondria_segmentation(self.image)
        self.roi = create_roi(self.segmentation, width=128, height=128)

    def test_roi_composite_upsert_works_without_json_contains_lookup(self):
        prob_data = np.full((self.roi.height, self.roi.width), 0.5, dtype=np.float32)

        save_probability_map(
            segmentation=self.segmentation,
            model_name="ResNet34",
            prob_data=prob_data,
            prefix="mito",
            generated_flag="mito_generated",
            roi_id=str(self.roi.id),
        )
        composite = ProbabilityMap.objects.get(
            segmentation=self.segmentation,
            name="MITO_ResNet34",
            metadata__composite=True,
            metadata__mito_generated=True,
        )

        save_probability_map(
            segmentation=self.segmentation,
            model_name="ResNet34",
            prob_data=prob_data,
            prefix="mito",
            generated_flag="mito_generated",
            roi_id=str(self.roi.id),
        )
        self.assertEqual(
            ProbabilityMap.objects.filter(
                segmentation=self.segmentation,
                name="MITO_ResNet34",
                metadata__composite=True,
                metadata__mito_generated=True,
            ).count(),
            1,
        )
        self.assertEqual(
            ProbabilityMap.objects.get(
                segmentation=self.segmentation,
                name="MITO_ResNet34",
                metadata__composite=True,
                metadata__mito_generated=True,
            ).id,
            composite.id,
        )
        composite.refresh_from_db()
        self.assertIn("/composite/", composite.file_path)
        self.assertFalse(
            get_prob_map_file_path(
                self.segmentation,
                "ResNet34",
                "mito",
                roi_id=None,
            ).exists()
        )

    def test_legacy_composite_full_path_is_ignored_for_full_image_cache(self):
        prob_data = np.full((self.roi.height, self.roi.width), 0.5, dtype=np.float32)
        full_path = get_prob_map_file_path(
            self.segmentation,
            "ResNet34",
            "mito",
            roi_id=None,
        )
        full_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((prob_data * 255).astype(np.uint8), mode="L").save(full_path)
        relative_path = str(full_path.relative_to(STORAGE_DIR)).replace("\\", "/")

        ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            name="MITO_ResNet34",
            file_path=relative_path,
            metadata={
                "model_type": "ResNet34",
                "mito_generated": True,
                "composite": True,
            },
        )

        self.assertFalse(
            prob_map_file_exists(
                self.segmentation,
                "ResNet34",
                "mito",
                roi_id=None,
            )
        )
        self.assertIsNone(
            load_prob_map_from_path(
                self.segmentation,
                "ResNet34",
                "mito",
                roi_id=None,
            )
        )

        ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            name="MITO_ResNet34",
            file_path=relative_path,
            metadata={
                "model_type": "ResNet34",
                "mito_generated": True,
            },
        )

        self.assertTrue(
            prob_map_file_exists(
                self.segmentation,
                "ResNet34",
                "mito",
                roi_id=None,
            )
        )
        loaded = load_prob_map_from_path(
            self.segmentation,
            "ResNet34",
            "mito",
            roi_id=None,
        )
        self.assertIsNotNone(loaded)

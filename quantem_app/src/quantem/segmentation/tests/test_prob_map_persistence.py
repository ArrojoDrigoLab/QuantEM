from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from django.test import TestCase
from PIL import Image

from quantem.core.config import STORAGE_DIR
from quantem.seg_core.db.prob_maps import (
    MAX_ROI_COMPOSITE_MEGAPIXELS,
    get_prob_map_file_path,
    load_prob_map_from_path,
    load_prob_map_uint8_from_path,
    prob_map_file_exists,
    save_probability_map,
)
from quantem.segmentation.models import ProbabilityMap
from quantem.segmentation.prob_maps.io import clone_probability_map
from quantem.segmentation.prob_maps.preview import (
    ensure_probability_preview,
    probability_preview_path,
)
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

    def test_chunked_writer_bounds_both_dimensions(self):
        levels = np.linspace(0, 1, 35 * 29, dtype=np.float32).reshape(35, 29)

        class SliceOnlySource:
            shape = levels.shape
            dtype = levels.dtype

            def __getitem__(self, key):
                block = levels[key]
                assert block.shape[0] <= 8 and block.shape[1] <= 8
                return block

        with (
            patch("quantem.seg_core.db.prob_maps.PROBABILITY_MAP_ZARR_MIN_PIXELS", 1),
            patch("quantem.seg_core.db.prob_maps.PROBABILITY_MAP_CHUNK_SIZE", 8),
        ):
            save_probability_map(
                self.segmentation, "DINO", SliceOnlySource(), "mito", "mito_generated"
            )
        from quantem.inference.resample import quantize_probability

        actual = load_prob_map_uint8_from_path(self.segmentation, "DINO", "mito")
        np.testing.assert_array_equal(np.asarray(actual), quantize_probability(levels))

    def test_format_switch_preserves_shared_map_and_selects_new_result(self):
        old_levels = np.full((35, 29), 204, dtype=np.uint8)
        new_levels = np.full((35, 29), 51, dtype=np.uint8)
        with patch("quantem.seg_core.db.prob_maps.PROBABILITY_MAP_ZARR_MIN_PIXELS", 1):
            old = save_probability_map(
                self.segmentation, "DINO", old_levels, "mito", "mito_generated"
            )
        other = create_mitochondria_segmentation(create_image_from_test_tiff("Shared map"))
        clone = clone_probability_map(old, other)
        old_path = STORAGE_DIR / clone.file_path
        new = save_probability_map(self.segmentation, "DINO", new_levels, "mito", "mito_generated")
        assert (
            get_prob_map_file_path(self.segmentation, "DINO", "mito") == STORAGE_DIR / new.file_path
        )
        np.testing.assert_array_equal(
            np.asarray(zarr.open_array(str(old_path), mode="r")), old_levels
        )
        assert probability_preview_path(old_path).is_file()
        np.testing.assert_array_equal(
            load_prob_map_uint8_from_path(self.segmentation, "DINO", "mito"),
            new_levels,
        )

    def test_format_switch_keeps_the_new_preview(self):
        levels = np.full((35, 29), 204, dtype=np.uint8)
        save_probability_map(self.segmentation, "DINO", levels, "mito", "mito_generated")
        with patch("quantem.seg_core.db.prob_maps.PROBABILITY_MAP_ZARR_MIN_PIXELS", 1):
            stored = save_probability_map(
                self.segmentation, "DINO", levels, "mito", "mito_generated"
            )
        assert probability_preview_path(STORAGE_DIR / stored.file_path).is_file()

    def test_failed_record_write_restores_previous_bytes_and_preview(self):
        old_levels = np.full((35, 29), 204, dtype=np.uint8)
        for minimum in (1, 10_000):
            with (
                self.subTest(minimum=minimum),
                patch(
                    "quantem.seg_core.db.prob_maps.PROBABILITY_MAP_ZARR_MIN_PIXELS",
                    minimum,
                ),
            ):
                stored = save_probability_map(
                    self.segmentation, "DINO", old_levels, "mito", "mito_generated"
                )
                path = STORAGE_DIR / stored.file_path
                preview = ensure_probability_preview(path, max_dimension=16)
                before = preview.read_bytes()
                with patch.object(
                    ProbabilityMap.objects,
                    "create",
                    side_effect=RuntimeError("database unavailable"),
                ):
                    with pytest.raises(RuntimeError, match="database unavailable"):
                        save_probability_map(
                            self.segmentation,
                            "DINO",
                            np.zeros_like(old_levels),
                            "mito",
                            "mito_generated",
                        )
                np.testing.assert_array_equal(
                    np.asarray(load_prob_map_uint8_from_path(self.segmentation, "DINO", "mito")),
                    old_levels,
                )
                assert preview.read_bytes() == before
                assert not list(path.parent.glob(".*.bak*"))
                assert not list(path.parent.glob(".*.tmp*"))

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

    def test_large_parent_keeps_the_roi_map_without_allocating_a_full_composite(self):
        prob_data = np.full((self.roi.height, self.roi.width), 0.5, dtype=np.float32)
        parent_edge = int((MAX_ROI_COMPOSITE_MEGAPIXELS * 1e6) ** 0.5) + 1

        with patch(
            "quantem.seg_core.db.prob_maps.get_asset_openable",
            return_value=SimpleNamespace(height=parent_edge, width=parent_edge),
        ):
            stored = save_probability_map(
                segmentation=self.segmentation,
                model_name="ResNet34",
                prob_data=prob_data,
                prefix="mito",
                generated_flag="mito_generated",
                roi_id=str(self.roi.id),
            )

        self.assertTrue(
            get_prob_map_file_path(
                self.segmentation, "ResNet34", "mito", roi_id=str(self.roi.id)
            ).exists()
        )
        self.assertEqual(stored.segmentation_id, self.segmentation.id)
        self.assertFalse(
            ProbabilityMap.objects.filter(
                segmentation=self.segmentation,
                metadata__composite=True,
            ).exists()
        )

    def test_large_maps_store_identical_uint8_levels_in_chunks(self):
        levels = np.arange(35 * 29, dtype=np.uint16).reshape(35, 29).astype(np.uint8)

        with patch(
            "quantem.seg_core.db.prob_maps.PROBABILITY_MAP_ZARR_MIN_PIXELS",
            1,
        ):
            stored = save_probability_map(
                segmentation=self.segmentation,
                model_name="DINO",
                prob_data=levels,
                prefix="mito",
                generated_flag="mito_generated",
            )

        path = get_prob_map_file_path(self.segmentation, "DINO", "mito")
        self.assertTrue(path.is_dir())
        self.assertEqual(path.suffix, ".zarr")
        expected_relative = str(path.relative_to(STORAGE_DIR)).replace("\\", "/")
        self.assertEqual(stored.file_path.replace("\\", "/"), expected_relative)

        chunked = load_prob_map_uint8_from_path(self.segmentation, "DINO", "mito")
        self.assertIsNotNone(chunked)
        np.testing.assert_array_equal(np.asarray(chunked), levels)

        preview = ensure_probability_preview(path, max_dimension=16)
        self.assertTrue(preview.is_file())
        with Image.open(preview) as image:
            self.assertEqual(image.mode, "L")

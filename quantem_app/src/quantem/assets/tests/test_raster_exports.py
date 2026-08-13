from __future__ import annotations

from io import BytesIO

import numpy as np
from django.test import TestCase
from PIL import Image
from rest_framework.test import APIClient
from shapely.geometry import Polygon, box

from quantem.assets.raster_exports import export_label, segmentation_export
from quantem.segmentation.global_masks import save_global_mask
from quantem.segmentation.models import AnalysisMaskObject, ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import (
    get_or_create_analysis_mask_type,
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image


def _png_response_array(response) -> np.ndarray:
    payload = b"".join(response.streaming_content)
    with Image.open(BytesIO(payload)) as image:
        assert image.mode == "L"
        return np.asarray(image).copy()


def _segment(segmentation: ImageSegmentation, geometry, *, state: str = "CONFIRMED"):
    return SegmentObject.objects.create(
        segmentation=segmentation,
        geometry=geometry,
        centroid=geometry.centroid,
        bbox=geometry.envelope,
        label_state=state,
    )


class RasterExportTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("Portal field", width=32, height=32, textured=True)

    def test_original_em_download_is_an_8_bit_grayscale_png(self):
        response = self.client.get(f"/api/assets/{self.image.asset.id}/export-png/?source=original")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "image/png")
        self.assertIn("Portal_field_EM_8bit.png", response["Content-Disposition"])
        exported = _png_response_array(response)
        self.assertEqual(exported.dtype, np.uint8)
        self.assertEqual(exported.shape, (32, 32))
        self.assertGreater(np.unique(exported).size, 1)

    def test_object_segmentation_uses_zero_background_and_sequential_ids(self):
        segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        _segment(segmentation, box(2, 2, 10, 10))
        _segment(segmentation, box(18, 18, 28, 28))
        _segment(segmentation, box(12, 12, 16, 16), state="EXCLUDED")

        response = self.client.get(
            f"/api/assets/{self.image.asset.id}/export-png/"
            f"?source=segmentation&segmentation_id={segmentation.id}"
        )

        self.assertEqual(response.status_code, 200)
        exported = _png_response_array(response)
        self.assertEqual(int(exported[0, 0]), 0)
        self.assertEqual(int(exported[5, 5]), 1)
        self.assertEqual(int(exported[22, 22]), 2)
        self.assertEqual(int(exported[14, 14]), 0)

    def test_object_ids_cycle_after_255_without_using_background(self):
        self.assertEqual(export_label(0), 1)
        self.assertEqual(export_label(254), 255)
        self.assertEqual(export_label(255), 1)
        self.assertEqual(export_label(256), 2)

    def test_analysis_mask_uses_last_object_for_overlaps_but_not_its_holes(self):
        segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_analysis_mask_type(),
            display_name="Compartments",
        )
        AnalysisMaskObject.objects.create(
            segmentation=segmentation,
            name="Object 1",
            color="#38bdf8",
            sort_order=1,
            geometry=box(2, 2, 18, 18),
        )
        later = Polygon(
            [(8, 8), (26, 8), (26, 26), (8, 26), (8, 8)],
            holes=[[(11, 11), (14, 11), (14, 14), (11, 14), (11, 11)]],
        )
        AnalysisMaskObject.objects.create(
            segmentation=segmentation,
            name="Object 2",
            color="#22c55e",
            sort_order=2,
            geometry=later,
        )

        exported = segmentation_export(segmentation)

        self.assertEqual(int(exported[5, 5]), 1)
        self.assertEqual(int(exported[9, 9]), 2, "the later object wins the overlap")
        self.assertEqual(int(exported[12, 12]), 1, "a hole does not erase an earlier object")
        self.assertEqual(int(exported[22, 22]), 2)

    def test_other_global_segmentations_export_binary_8_bit_masks(self):
        segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_tissue_type(),
        )
        mask = np.zeros((32, 32), dtype=bool)
        mask[4:12, 4:12] = True
        save_global_mask(segmentation, mask, source="manual")

        exported = segmentation_export(segmentation)

        self.assertEqual(set(np.unique(exported)), {0, 255})
        self.assertEqual(int(exported[8, 8]), 255)

    def test_legacy_analysis_mask_without_objects_exports_as_object_one(self):
        segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_analysis_mask_type(),
            display_name="Legacy region",
        )
        mask = np.zeros((32, 32), dtype=bool)
        mask[5:10, 5:10] = True
        save_global_mask(segmentation, mask, source="manual")

        exported = segmentation_export(segmentation)

        self.assertEqual(set(np.unique(exported)), {0, 1})
        self.assertEqual(int(exported[7, 7]), 1)

    def test_a_segmentation_from_another_image_cannot_be_exported(self):
        other = create_small_test_image("Other", width=32, height=32)
        segmentation = ImageSegmentation.objects.create(
            asset=other.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

        response = self.client.get(
            f"/api/assets/{self.image.asset.id}/export-png/"
            f"?source=segmentation&segmentation_id={segmentation.id}"
        )

        self.assertEqual(response.status_code, 404)

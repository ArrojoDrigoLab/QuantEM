from __future__ import annotations

import uuid

import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient

from quantem.segmentation.global_masks import load_global_mask
from quantem.segmentation.models import (
    AnalysisMaskObject,
    GlobalMask,
    ImageSegmentation,
)
from quantem.segmentation.type_service import (
    get_or_create_analysis_mask_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image


def _shape(x0: int, y0: int, x1: int, y1: int) -> dict:
    return {
        "rings": [
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
        ]
    }


class AnalysisMaskObjectApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        image = create_small_test_image("Analysis objects", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_analysis_mask_type(),
            display_name="Cells",
            status_stage="CANDIDATES_READY",
        )
        self.url = f"/api/segmentations/{self.segmentation.id}/analysis-mask-objects/"

    def test_first_shape_creates_named_colored_object_and_aggregate_mask(self):
        response = self.client.post(
            self.url,
            {"operation": "include", "shapes": [_shape(4, 4, 24, 24)]},
            format="json",
        )

        self.assertEqual(response.status_code, 201, response.data)
        payload = response.data["object"]
        self.assertEqual(payload["name"], "Object 1")
        self.assertRegex(payload["color"], r"^#[0-9a-f]{6}$")
        self.assertEqual(payload["geometry"]["type"], "Polygon")
        self.assertEqual(AnalysisMaskObject.objects.count(), 1)
        self.assertFalse(
            GlobalMask.objects.filter(segmentation=self.segmentation).exists(),
            "closing a shape must not block on a full-image mask rewrite",
        )
        saved = self.client.post(f"{self.url}save/", format="json")
        self.assertEqual(saved.status_code, 200, saved.data)
        mask = load_global_mask(self.segmentation)
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[30, 30])

    def test_one_object_retains_disconnected_regions_and_an_excluded_hole(self):
        created = self.client.post(
            self.url,
            {"operation": "include", "shapes": [_shape(4, 4, 40, 40)]},
            format="json",
        )
        object_id = created.data["object"]["id"]
        island = self.client.post(
            self.url,
            {
                "object_id": object_id,
                "operation": "include",
                "shapes": [_shape(48, 48, 60, 60)],
            },
            format="json",
        )
        carved = self.client.post(
            self.url,
            {
                "object_id": object_id,
                "operation": "exclude",
                "shapes": [_shape(16, 16, 28, 28)],
            },
            format="json",
        )

        self.assertEqual(island.status_code, 200, island.data)
        self.assertEqual(carved.status_code, 200, carved.data)
        self.assertEqual(carved.data["object"]["geometry"]["type"], "MultiPolygon")
        saved = self.client.post(f"{self.url}save/", format="json")
        self.assertEqual(saved.status_code, 200, saved.data)
        mask = load_global_mask(self.segmentation)
        self.assertTrue(mask[8, 8])
        self.assertFalse(mask[20, 20])
        self.assertTrue(mask[54, 54])

    def test_rename_and_delete_rebuild_the_union_without_other_objects(self):
        first = self.client.post(
            self.url,
            {"operation": "include", "shapes": [_shape(2, 2, 18, 18)]},
            format="json",
        ).data["object"]
        second = self.client.post(
            self.url,
            {"operation": "include", "shapes": [_shape(40, 40, 58, 58)]},
            format="json",
        ).data["object"]

        detail = f"{self.url}{first['id']}/"
        renamed = self.client.patch(detail, {"name": "Portal cell"}, format="json")
        deleted = self.client.delete(detail)

        self.assertEqual(renamed.status_code, 200, renamed.data)
        self.assertEqual(renamed.data["name"], "Portal cell")
        self.assertEqual(deleted.status_code, 200, deleted.data)
        self.assertEqual(deleted.data["deleted_id"], first["id"])
        self.assertEqual(
            list(AnalysisMaskObject.objects.values_list("id", flat=True)),
            [uuid.UUID(second["id"])],
        )
        saved = self.client.post(f"{self.url}save/", format="json")
        self.assertEqual(saved.status_code, 200, saved.data)
        mask = load_global_mask(self.segmentation)
        self.assertFalse(mask[8, 8])
        self.assertTrue(mask[48, 48])

    def test_new_object_cannot_start_with_exclude(self):
        response = self.client.post(
            self.url,
            {"operation": "exclude", "shapes": [_shape(4, 4, 24, 24)]},
            format="json",
        )
        self.assertEqual(response.status_code, 400, response.data)
        self.assertIn("starts in Include mode", response.data["error"])
        self.assertFalse(AnalysisMaskObject.objects.exists())

    def test_non_analysis_global_mask_cannot_use_named_object_api(self):
        tissue = ImageSegmentation.objects.create(
            asset=self.segmentation.asset,
            segmentation_type=get_or_create_tissue_type(),
        )
        response = self.client.get(f"/api/segmentations/{tissue.id}/analysis-mask-objects/")
        self.assertEqual(response.status_code, 400, response.data)


class AnalysisMaskObjectRasterTests(TestCase):
    def test_multiple_named_objects_are_one_analysis_denominator(self):
        image = create_small_test_image("Analysis denominator", width=32, height=32)
        segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_analysis_mask_type(),
            display_name="Regions",
        )
        client = APIClient()
        url = f"/api/segmentations/{segmentation.id}/analysis-mask-objects/"
        client.post(
            url,
            {"operation": "include", "shapes": [_shape(0, 0, 8, 8)]},
            format="json",
        )
        client.post(
            url,
            {"operation": "include", "shapes": [_shape(24, 24, 32, 32)]},
            format="json",
        )
        saved = client.post(f"{url}save/", format="json")
        self.assertEqual(saved.status_code, 200, saved.data)

        mask = load_global_mask(segmentation)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[4, 4])
        self.assertTrue(mask[28, 28])
        self.assertFalse(mask[16, 16])

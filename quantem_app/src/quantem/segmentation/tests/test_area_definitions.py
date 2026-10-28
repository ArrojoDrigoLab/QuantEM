import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient

from quantem.segmentation.global_masks import load_global_mask, save_global_mask
from quantem.segmentation.models import AnalysisMaskObject, ImageSegmentation
from quantem.segmentation.type_service import (
    get_or_create_cells_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image


def shape(x0, y0, x1, y1):
    return {
        "rings": [
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0],
            ]
        ]
    }


class AreaDefinitionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("Cell names", width=96, height=32)
        self.cells = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_cells_type(),
            status_stage="CANDIDATES_READY",
        )
        self.url = f"/api/segmentations/{self.cells.id}/analysis-mask-objects/"

    def test_cell_names_are_persistent_and_increment_above_current_highest(self):
        created = []
        for index in range(5):
            x0 = 2 + index * 14
            response = self.client.post(
                self.url,
                {"operation": "include", "shapes": [shape(x0, 2, x0 + 8, 10)]},
                format="json",
            )
            self.assertEqual(response.status_code, 201, response.data)
            created.append(response.data["object"])
        self.client.patch(
            f"{self.url}{created[1]['id']}/",
            {"name": "Custom Named Cell"},
            format="json",
        )
        self.client.delete(f"{self.url}{created[2]['id']}/")

        response = self.client.post(
            self.url,
            {"operation": "include", "shapes": [shape(76, 2, 84, 10)]},
            format="json",
        )

        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["object"]["name"], "Cell 6")
        self.assertEqual(
            list(
                AnalysisMaskObject.objects.filter(segmentation=self.cells)
                .order_by("sort_order")
                .values_list("name", flat=True)
            ),
            ["Cell 1", "Custom Named Cell", "Cell 4", "Cell 5", "Cell 6"],
        )

    def test_cells_reject_positive_area_overlap(self):
        first = self.client.post(
            self.url,
            {"operation": "include", "shapes": [shape(2, 2, 14, 14)]},
            format="json",
        )
        self.assertEqual(first.status_code, 201, first.data)

        overlap = self.client.post(
            self.url,
            {"operation": "include", "shapes": [shape(10, 10, 20, 20)]},
            format="json",
        )

        self.assertEqual(overlap.status_code, 400, overlap.data)
        self.assertIn("non-overlapping", overlap.data["error"])

    def test_custom_area_definition_is_application_wide_and_sparse(self):
        response = self.client.post(
            "/api/segmentation-types/",
            {
                "long_name": "Lesion areas",
                "description": "Regions with visible pathology.",
                "category": "area",
                "representation": "instances",
                "topology_policy": "overlapping",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["category"], "area")
        self.assertEqual(response.data["measurement_mode"], "global")
        self.assertEqual(response.data["topology_policy"], "overlapping")

    def test_opening_a_legacy_sample_area_preserves_its_holes_when_saved(self):
        sample = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_tissue_type(),
            status_stage="CANDIDATES_READY",
        )
        mask = np.zeros((32, 96), dtype=bool)
        mask[3:29, 3:40] = True
        mask[11:21, 12:25] = False
        save_global_mask(sample, mask, source="manual")
        url = f"/api/segmentations/{sample.id}/analysis-mask-objects/"

        opened = self.client.get(url)
        self.assertEqual(opened.status_code, 200, opened.data)
        self.assertEqual(len(opened.data["objects"]), 1)
        saved = self.client.post(f"{url}save/", {}, format="json")
        self.assertEqual(saved.status_code, 200, saved.data)

        rebuilt = load_global_mask(sample)
        self.assertTrue(rebuilt[5, 5])
        self.assertFalse(rebuilt[16, 16])

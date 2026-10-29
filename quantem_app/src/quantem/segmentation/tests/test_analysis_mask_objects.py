from __future__ import annotations

import uuid
from unittest.mock import patch

import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient

from quantem.analysis.compartments import CompartmentSet, area_fractions_by_region
from quantem.analysis.loaders import analysis_mask_regions, segmentation_mask
from quantem.segmentation.global_masks import load_global_mask
from quantem.segmentation.models import (
    AnalysisMaskObject,
    AnalysisMaskObjectAnnotation,
    GlobalMask,
    ImageSegmentation,
)
from quantem.segmentation.type_service import (
    get_or_create_analysis_mask_type,
    get_or_create_cells_type,
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

    def test_first_shape_creates_anonymous_colored_object_and_aggregate_mask(self):
        response = self.client.post(
            self.url,
            {"operation": "include", "shapes": [_shape(4, 4, 24, 24)]},
            format="json",
        )

        self.assertEqual(response.status_code, 201, response.data)
        payload = response.data["object"]
        self.assertEqual(payload["name"], "")
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

    def test_sample_area_can_use_the_shared_area_object_editor(self):
        tissue = ImageSegmentation.objects.create(
            asset=self.segmentation.asset,
            segmentation_type=get_or_create_tissue_type(),
        )
        response = self.client.get(f"/api/segmentations/{tissue.id}/analysis-mask-objects/")
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["objects"], [])

    def test_named_sample_area_object_names_the_segmentation_in_asset_lists(self):
        tissue = ImageSegmentation.objects.create(
            asset=self.segmentation.asset,
            segmentation_type=get_or_create_tissue_type(),
        )
        AnalysisMaskObject.objects.create(
            segmentation=tissue,
            name="Pericentral region",
            color="#38bdf8",
            sort_order=1,
        )

        response = self.client.get(f"/api/assets/{self.segmentation.asset_id}/segmentations/")

        self.assertEqual(response.status_code, 200, response.data)
        payload = next(row for row in response.data if row["id"] == str(tissue.id))
        self.assertEqual(payload["display_name"], "Pericentral region")


class CellObjectApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        image = create_small_test_image("Cells", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_cells_type(),
            display_name="Cells",
            status_stage="CANDIDATES_READY",
        )
        self.url = f"/api/segmentations/{self.segmentation.id}/analysis-mask-objects/"

    def test_new_cells_skip_legacy_raster_materialization(self):
        with patch(
            "quantem.segmentation.api_views.analysis_masks.materialize_legacy_area_instances"
        ) as materialize:
            listed = self.client.get(self.url)
            created = self.client.post(
                self.url,
                {"operation": "include", "shapes": [_shape(4, 4, 24, 24)]},
                format="json",
            )

        self.assertEqual(listed.status_code, 200, listed.data)
        self.assertEqual(listed.data["objects"], [])
        self.assertEqual(created.status_code, 201, created.data)
        self.assertEqual(created.data["object"]["name"], "Cell 1")
        self.assertFalse(GlobalMask.objects.filter(segmentation=self.segmentation).exists())
        materialize.assert_not_called()

    def test_geometry_metadata_and_deletion_supersede_existing_analysis(self):
        superseded = "quantem.analysis.superseded.mark_superseded"
        with patch(superseded) as mark:
            created = self.client.post(
                self.url,
                {"operation": "include", "shapes": [_shape(4, 4, 24, 24)]},
                format="json",
            )
            self.assertEqual(created.status_code, 201, created.data)
            mark.assert_called_once_with(self.segmentation)

        detail = f"{self.url}{created.data['object']['id']}/"
        with patch(superseded) as mark:
            renamed = self.client.patch(detail, {"name": "Portal cell"}, format="json")
            self.assertEqual(renamed.status_code, 200, renamed.data)
            mark.assert_called_once_with(self.segmentation)

        with patch(superseded) as mark:
            deleted = self.client.delete(detail)
            self.assertEqual(deleted.status_code, 200, deleted.data)
            mark.assert_called_once_with(self.segmentation)

    def test_cell_tags_are_suggested_app_wide_and_canonicalized_case_insensitively(self):
        other_image = create_small_test_image("Other cells", width=32, height=32)
        other_segmentation = ImageSegmentation.objects.create(
            asset=other_image.asset,
            segmentation_type=get_or_create_cells_type(),
            display_name="Cells",
        )
        other_cell = AnalysisMaskObject.objects.create(
            segmentation=other_segmentation,
            name="Cell 1",
            color="#38bdf8",
            sort_order=1,
        )
        AnalysisMaskObjectAnnotation.objects.create(
            object=other_cell,
            tags=[
                {"name": "Phenotype", "value": "Beta"},
                {"name": "Treatment", "value": "Control"},
            ],
        )
        created = self.client.post(
            self.url,
            {"operation": "include", "shapes": [_shape(4, 4, 24, 24)]},
            format="json",
        )
        self.assertEqual(created.status_code, 201, created.data)

        listed = self.client.get(self.url)

        self.assertEqual(listed.status_code, 200, listed.data)
        self.assertEqual(
            listed.data["tag_suggestions"],
            [
                {"name": "Phenotype", "values": ["Beta"]},
                {"name": "Treatment", "values": ["Control"]},
            ],
        )

        detail = f"{self.url}{created.data['object']['id']}/"
        updated = self.client.patch(
            detail,
            {
                "name": "Pericentral cell",
                "tags": [
                    {"name": "phenotype", "value": "beta"},
                    {"name": "State", "value": "Activated"},
                ],
            },
            format="json",
        )

        self.assertEqual(updated.status_code, 200, updated.data)
        self.assertEqual(updated.data["name"], "Pericentral cell")
        self.assertEqual(
            updated.data["tags"],
            [
                {"name": "Phenotype", "value": "Beta"},
                {"name": "State", "value": "Activated"},
            ],
        )
        annotation = AnalysisMaskObjectAnnotation.objects.get(
            object_id=created.data["object"]["id"]
        )
        self.assertEqual(annotation.tags, updated.data["tags"])

        duplicate = self.client.patch(
            detail,
            {
                "tags": [
                    {"name": "State", "value": "Activated"},
                    {"name": "state", "value": "Quiescent"},
                ]
            },
            format="json",
        )
        self.assertEqual(duplicate.status_code, 400, duplicate.data)
        self.assertIn("same tag name twice", duplicate.data["error"])


class AnalysisMaskObjectRasterTests(TestCase):
    def test_multiple_named_objects_are_a_union_and_independent_analysis_regions(self):
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

        loaded = segmentation_mask(segmentation, (32, 32))
        np.testing.assert_array_equal(loaded, mask)
        regions = analysis_mask_regions(segmentation, (32, 32))
        self.assertEqual([region.name for region in regions], ["Area 1", "Area 2"])

        mito = np.zeros((32, 32), dtype=bool)
        mito[0:8, 0:4] = True
        mito[24:32, 24:32] = True
        rows = area_fractions_by_region(
            CompartmentSet({"mito": mito}, tissue=loaded),
            regions,
            pixel_size_nm=5.0,
        )
        self.assertEqual([row["tissue_px"] for row in rows], [64, 64])
        self.assertEqual([row["tissue_um2"] for row in rows], [0.0016, 0.0016])
        self.assertEqual([row["areas_px"]["mito"] for row in rows], [32, 64])
        self.assertEqual([row["area_fractions"]["mito"] for row in rows], [0.5, 1.0])
        self.assertEqual([row["areas_um2"]["mito"] for row in rows], [0.0008, 0.0016])

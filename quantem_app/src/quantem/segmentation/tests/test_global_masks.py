from __future__ import annotations

import numpy as np
import zarr
from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon, box

from quantem.analysis.global_area import global_area_report
from quantem.core.local_storage import storage_path
from quantem.segmentation.global_masks import (
    load_global_mask,
    patch_global_mask,
    save_global_mask,
)
from quantem.segmentation.models import (
    GlobalMask,
    ImageSegmentation,
    SegmentationOverlayLabel,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff import (
    get_overlay_active_bundle_path,
    rebuild_overlay_full,
)
from quantem.segmentation.overlay_ngff.labels_lut import build_label_lut_json
from quantem.segmentation.overlay_ngff.manifest import build_overlay_manifest
from quantem.segmentation.type_service import (
    get_or_create_analysis_mask_type,
    get_or_create_er_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image


def _ring_mask(size: int = 64) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[8:56, 8:56] = True
    mask[24:40, 24:40] = False
    return mask


class GlobalMaskPersistenceTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("Global mask", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_tissue_type(),
        )

    def test_include_minus_exclude_persists_one_mask_with_holes_and_islands(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/",
            {
                "manual_creation": True,
                "segments": [
                    {
                        "operation": "include",
                        "geometry_rings": [
                            [[4, 4], [44, 4], [44, 44], [4, 44], [4, 4]],
                        ],
                    },
                    {
                        "operation": "include",
                        "geometry_coords": [[52, 52], [60, 52], [60, 60], [52, 60], [52, 52]],
                    },
                    {
                        "operation": "exclude",
                        "geometry_coords": [[16, 16], [32, 16], [32, 32], [16, 32], [16, 16]],
                    },
                ],
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(SegmentObject.objects.filter(segmentation=self.segmentation).count(), 0)
        self.assertTrue(GlobalMask.objects.filter(segmentation=self.segmentation).exists())

        reopened = ImageSegmentation.objects.select_related("asset", "segmentation_type").get(
            id=self.segmentation.id
        )
        mask = load_global_mask(reopened)
        self.assertTrue(mask[8, 8])
        self.assertFalse(mask[24, 24], "the draft exclusion must remain a hole")
        self.assertTrue(mask[55, 55], "a disconnected included island must survive")

    def test_exclude_only_draft_does_not_become_post_confirmation_remove(self):
        save_global_mask(self.segmentation, _ring_mask(), source="manual")
        before = load_global_mask(self.segmentation).copy()
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/",
            {
                "segments": [
                    {
                        "operation": "exclude",
                        "geometry_coords": [[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]],
                    }
                ]
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["updated"], 0)
        np.testing.assert_array_equal(load_global_mask(self.segmentation), before)

    def test_remove_area_remains_the_separate_post_confirmation_correction(self):
        save_global_mask(self.segmentation, np.ones((64, 64), dtype=bool), source="manual")
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/remove-area/",
            {"areas": [{"geometry_coords": [[20, 20], [40, 20], [40, 40], [20, 40], [20, 20]]}]},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["updated"], 1)
        self.assertFalse(load_global_mask(self.segmentation)[30, 30])

    def test_legacy_confirmed_objects_are_read_without_being_rewritten(self):
        legacy = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
        )
        polygon = Polygon(
            [(5, 5), (45, 5), (45, 45), (5, 45), (5, 5)],
            holes=[[(18, 18), (30, 18), (30, 30), (18, 30), (18, 18)]],
        )
        row = SegmentObject.objects.create(
            segmentation=legacy,
            label_state="CONFIRMED",
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )

        mask = load_global_mask(legacy)
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[24, 24])
        self.assertTrue(SegmentObject.objects.filter(id=row.id).exists())
        self.assertFalse(GlobalMask.objects.filter(segmentation=legacy).exists())

    def test_deleting_the_segmentation_removes_its_binary_mask_file(self):
        record = save_global_mask(self.segmentation, _ring_mask(), source="manual")
        path = storage_path(record.file_path)
        self.assertTrue(path.exists())

        response = self.client.delete(f"/api/segmentations/{self.segmentation.id}/")

        self.assertEqual(response.status_code, 200, response.data)
        self.assertFalse(path.exists())

    def test_manual_patch_preserves_model_and_adapter_provenance(self):
        save_global_mask(
            self.segmentation,
            _ring_mask(),
            source="model",
            metadata={"pack_id": "quantem:er", "adapter_id": "adapter-7"},
        )

        record = patch_global_mask(
            self.segmentation,
            include=[box(0, 0, 4, 4)],
            source="manual",
        )

        self.assertEqual(record.source, "model")
        self.assertEqual(record.metadata["pack_id"], "quantem:er")
        self.assertEqual(record.metadata["adapter_id"], "adapter-7")
        self.assertIs(record.metadata["manually_edited"], True)

        removed = patch_global_mask(
            self.segmentation,
            exclude=[box(1, 1, 2, 2)],
            source="manual-remove",
        )
        self.assertEqual(removed.source, "model")
        self.assertEqual(removed.metadata["adapter_id"], "adapter-7")


class GlobalMaskOverlayAndAnalysisTests(TestCase):
    def setUp(self):
        self.image = create_small_test_image("Global analysis", width=64, height=64)
        self.er = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
        )
        self.analysis_mask = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_analysis_mask_type(),
            display_name="Cell interior",
        )

    def test_binary_overlay_has_no_object_identity_and_retains_the_hole(self):
        mask = _ring_mask()
        save_global_mask(self.er, mask, source="model")
        state = rebuild_overlay_full(self.er, desired_revision=1)
        root = get_overlay_active_bundle_path(state)
        labels = np.asarray(zarr.open_array(str(root / "labels" / "0"), mode="r"))

        self.assertEqual(int(labels[10, 10]), 1)
        self.assertEqual(int(labels[30, 30]), 0)
        self.assertEqual(set(np.unique(labels)), {0, 1})
        self.assertFalse(SegmentationOverlayLabel.objects.filter(overlay_state=state).exists())
        manifest = build_overlay_manifest(self.er, state)
        self.assertEqual(manifest["overlay_kind"], "binary_mask")
        self.assertFalse(manifest["pickable"])
        lut = build_label_lut_json(state)
        self.assertEqual(lut["objects"], [])
        self.assertFalse(lut["pickable"])

    def test_percent_area_reports_only_explicit_pixel_numerators_and_denominators(self):
        foreground = _ring_mask()
        selected = np.zeros((64, 64), dtype=bool)
        selected[0:32, :] = True
        selected[12:20, 12:20] = False
        save_global_mask(self.er, foreground, source="model")
        save_global_mask(self.analysis_mask, selected, source="manual")

        report = global_area_report(self.er, analysis_mask_ids=[str(self.analysis_mask.id)])
        whole = report["whole_image"]
        self.assertEqual(whole["foreground_pixels"], int(foreground.sum()))
        self.assertEqual(whole["denominator_pixels"], 64 * 64)
        row = report["analysis_masks"][0]
        self.assertEqual(row["foreground_pixels"], int((foreground & selected).sum()))
        self.assertEqual(row["denominator_pixels"], int(selected.sum()))
        self.assertEqual(
            row["foreground_percent"],
            100.0 * float((foreground & selected).sum()) / float(selected.sum()),
        )
        self.assertEqual(
            set(report),
            {"measurement_mode", "metric", "segmentation_id", "whole_image", "analysis_masks"},
        )

    def test_area_endpoint_rejects_a_malformed_mask_identifier(self):
        save_global_mask(self.er, _ring_mask(), source="model")
        response = APIClient().post(
            f"/api/segmentations/{self.er.id}/analysis/global-area/",
            {"analysis_mask_ids": ["not-a-uuid"]},
            format="json",
        )

        self.assertEqual(response.status_code, 400, response.data)
        self.assertIn("valid identifier", response.data["error"])

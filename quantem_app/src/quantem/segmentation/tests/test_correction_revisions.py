from unittest import mock

import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.segmentation.global_masks import load_global_mask, save_global_mask
from quantem.segmentation.models import CorrectionRevision, ImageSegmentation, SegmentObject
from quantem.segmentation.revisions import restore_revision, save_revision
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image


class CorrectionRevisionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.asset = create_small_test_image("Corrections", width=32, height=32).asset
        self.seg = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        polygon = box(3, 3, 12, 12)
        self.obj = SegmentObject.objects.create(
            segmentation=self.seg,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon,
            label_state="CONFIRMED",
            source_model="manual",
        )

    @mock.patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles")
    def test_restore_preserves_ids_and_saves_current_state_first(self, overlay):
        revision = save_revision(self.seg, "Before edits")
        self.obj.delete()
        restore_revision(revision)
        restored = SegmentObject.objects.get(segmentation=self.seg, superseded_at__isnull=True)
        self.assertEqual(restored.label_state, "CONFIRMED")
        self.assertEqual(restored.geometry.area, 81)
        self.assertEqual(CorrectionRevision.objects.filter(segmentation=self.seg).count(), 2)
        self.assertTrue(overlay.called)

    @mock.patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles")
    def test_global_mask_revision_is_a_copy_not_a_pointer_to_overwritten_file(self, overlay):
        seg = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_tissue_type()
        )
        mask = np.zeros((32, 32), dtype=bool)
        mask[2:10, 2:10] = True
        save_global_mask(seg, mask, source="manual")
        revision = save_revision(seg, "Tissue before edit")
        save_global_mask(seg, ~mask, source="manual")
        restore_revision(revision)
        seg.refresh_from_db()
        np.testing.assert_array_equal(load_global_mask(seg), mask)

    @mock.patch(
        "quantem.segmentation.api_views.segments.labels.register_overlay_mutation_all_bundles"
    )
    def test_clear_is_recoverable(self, overlay):
        overlay.return_value = None
        response = self.client.post(
            f"/api/segmentations/{self.seg.id}/labels/clear", {}, format="json"
        )
        self.assertEqual(response.status_code, 200, response.content)
        self.assertTrue(response.data["revision_id"])
        self.assertFalse(SegmentObject.objects.filter(segmentation=self.seg).exists())

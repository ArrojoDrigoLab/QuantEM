from io import BytesIO
from unittest import mock
from zipfile import ZipFile

import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.assets.models import ImageROI
from quantem.segmentation.global_masks import load_global_mask, save_global_mask
from quantem.segmentation.models import (
    CorrectionRevision,
    ImageSegmentation,
    RoiSegmentationStatus,
    SegmentObject,
)
from quantem.segmentation.revision_rois import restore_roi_reviews
from quantem.segmentation.revisions import restore_revision, save_revision
from quantem.segmentation.services.adapt.extract_crops import annotated_regions
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
    def test_restore_clears_review_completion_added_after_snapshot(self, overlay):
        roi = ImageROI.objects.create(asset=self.asset, x=0, y=0, width=16, height=16)
        revision = save_revision(self.seg, "Partial annotations")
        RoiSegmentationStatus.objects.create(image_roi=roi, segmentation=self.seg, is_complete=True)
        restore_revision(revision)
        regions, _ = annotated_regions(self.seg, 32, 32)
        self.assertEqual(regions, [])

    @mock.patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles")
    def test_restore_keeps_saved_roi_geometry_without_changing_other_organelles(self, overlay):
        roi = ImageROI.objects.create(asset=self.asset, x=0, y=0, width=16, height=16)
        RoiSegmentationStatus.objects.create(image_roi=roi, segmentation=self.seg, is_complete=True)
        other = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_tissue_type()
        )
        RoiSegmentationStatus.objects.create(image_roi=roi, segmentation=other, is_complete=True)
        revision = save_revision(self.seg, "Reviewed window")
        roi.width = 24
        roi.save()
        restore_revision(revision)
        state = self.seg.roi_statuses.get(is_complete=True)
        self.assertEqual(state.image_roi.width, 16)
        self.assertNotEqual(state.image_roi_id, roi.id)
        roi.refresh_from_db()
        self.assertEqual(roi.width, 24)
        self.assertTrue(other.roi_statuses.get(image_roi=roi).is_complete)
        restore_revision(revision)
        self.assertEqual(self.seg.roi_statuses.get().image_roi_id, state.image_roi_id)

    @mock.patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles")
    def test_restore_recreates_deleted_reviewed_roi(self, overlay):
        roi = ImageROI.objects.create(asset=self.asset, x=0, y=0, width=16, height=16)
        RoiSegmentationStatus.objects.create(image_roi=roi, segmentation=self.seg, is_complete=True)
        revision = save_revision(self.seg, "Reviewed window")
        roi.delete()
        restore_revision(revision)
        regions, _ = annotated_regions(self.seg, 32, 32)
        self.assertEqual(len(regions), 1)

    def test_legacy_archive_without_review_state_never_retains_later_completion(self):
        roi = ImageROI.objects.create(asset=self.asset, x=0, y=0, width=16, height=16)
        RoiSegmentationStatus.objects.create(image_roi=roi, segmentation=self.seg, is_complete=True)
        with ZipFile(BytesIO(), "w") as archive:
            restore_roi_reviews(archive, self.seg)
        regions, _ = annotated_regions(self.seg, 32, 32)
        self.assertEqual(regions, [])

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

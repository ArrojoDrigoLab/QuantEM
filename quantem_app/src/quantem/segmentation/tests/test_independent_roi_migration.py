"""Upgrade real legacy rows without changing crop IDs, files or annotations."""

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase
from django.utils import timezone
from shapely.geometry import box

from quantem.assets.models import ImageROI
from quantem.assets.utils import create_roi_image_from_image
from quantem.core.config import ROIS_DIR
from quantem.jobs.models import Job
from quantem.segmentation.models import (
    ImageSegmentation,
    ProbabilityMap,
    RoiSegmentationStatus,
    SegmentObject,
)
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_nucleus_type,
)
from quantem.testing import create_small_test_image

BEFORE = ("segmentation", "0012_reconcile_definitions_and_model_results")
AFTER = ("segmentation", "0013_independent_organelle_rois")


class IndependentRoiMigrationTests(TransactionTestCase):
    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.migrate(executor.loader.graph.leaf_nodes())

    def test_upgrade_preserves_visible_rois_and_scoped_completion(self):
        image = create_small_test_image("ROI upgrade", width=128, height=128)
        mito = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        nucleus = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_nucleus_type()
        )
        roi = create_roi_image_from_image(image, x=12, y=20, width=48, height=48)
        spare = ImageROI.objects.create(
            asset=image.asset, x=70, y=80, width=32, height=32, is_complete=True
        )
        roi.segmentations.add(mito)
        timestamp = timezone.now()
        state = RoiSegmentationStatus.objects.create(
            image_roi=roi, segmentation=mito, is_complete=True, completed_at=timestamp
        )
        polygon = box(20, 30, 32, 42)
        obj = SegmentObject.objects.create(
            segmentation=mito,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon,
            label_state="CONFIRMED",
        )
        prob = ProbabilityMap.objects.create(
            segmentation=mito,
            name="ROI prediction",
            file_path="existing-prediction.png",
            metadata={"roi_id": str(roi.pk)},
        )
        job = Job.objects.create(
            type="run_segmentation_roi",
            payload_json={"segmentation_id": str(mito.pk), "roi_id": str(roi.pk)},
        )
        crop = ROIS_DIR / f"{roi.pk}.png"
        crop_bytes = crop.read_bytes()
        objects_before = bytes(obj.geometry_wkb)

        executor = MigrationExecutor(connection)
        executor.migrate([BEFORE])
        executor = MigrationExecutor(connection)
        executor.migrate([AFTER])
        for segmentation in (mito, nucleus):
            segmentation.refresh_from_db(fields=["rois_initialized"])
            self.assertTrue(segmentation.rois_initialized)
            self.assertEqual(
                set(segmentation.rois.values_list("pk", flat=True)), {roi.pk, spare.pk}
            )
            self.assertEqual(segmentation.roi_statuses.get(is_active=True).image_roi_id, roi.pk)
        state.refresh_from_db()
        self.assertTrue(state.is_complete)
        self.assertEqual(state.completed_at, timestamp)
        self.assertFalse(nucleus.roi_statuses.filter(is_complete=True).exists())
        self.assertFalse(mito.roi_statuses.get(image_roi=spare).is_complete)
        self.assertEqual(ImageROI.objects.count(), 2)
        self.assertEqual(crop.read_bytes(), crop_bytes)
        obj.refresh_from_db()
        self.assertEqual(bytes(obj.geometry_wkb), objects_before)
        prob.refresh_from_db()
        self.assertEqual(prob.metadata["roi_id"], str(roi.pk))
        self.assertEqual(prob.file_path, "existing-prediction.png")
        job.refresh_from_db()
        self.assertEqual(job.payload_json["roi_id"], str(roi.pk))

        # A downgrade and re-upgrade preserve rows without duplicating them.
        executor = MigrationExecutor(connection)
        executor.migrate([BEFORE])
        executor = MigrationExecutor(connection)
        executor.migrate([AFTER])
        self.assertEqual(ImageROI.objects.count(), 2)
        self.assertEqual(RoiSegmentationStatus.objects.count(), 4)
        self.assertEqual(crop.read_bytes(), crop_bytes)

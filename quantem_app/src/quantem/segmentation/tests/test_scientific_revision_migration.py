"""Existing vector ownership survives upgrading, including an empty edit."""

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase
from shapely.geometry import box

from quantem.segmentation.models import AnalysisMaskObject, GlobalMask, ImageSegmentation
from quantem.segmentation.type_service import get_or_create_tissue_type
from quantem.testing import create_small_test_image


class ScientificRevisionMigrationTests(TransactionTestCase):
    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.migrate(executor.loader.graph.leaf_nodes())

    def test_adopts_known_vectors_and_empty_saved_vector_layers_only(self):
        asset = create_small_test_image("Vector ownership upgrade", width=32, height=32).asset
        segmentations = [
            ImageSegmentation.objects.create(
                asset=asset,
                segmentation_type=get_or_create_tissue_type(),
                display_name=f"Area {index}",
            )
            for index in range(3)
        ]
        AnalysisMaskObject.objects.create(segmentation=segmentations[0], geometry=box(2, 2, 12, 12))
        for seg, metadata in zip(
            segmentations[1:], [{"editable_analysis_objects": True}, {}], strict=True
        ):
            GlobalMask.objects.create(
                segmentation=seg,
                width=32,
                height=32,
                foreground_pixels=100,
                file_path=f"data/global_masks/{seg.pk}/legacy.png",
                metadata=metadata,
            )
        executor = MigrationExecutor(connection)
        executor.migrate([("segmentation", "0013_independent_organelle_rois")])
        executor = MigrationExecutor(connection)
        executor.migrate(executor.loader.graph.leaf_nodes())
        for seg in segmentations:
            seg.refresh_from_db()
            self.assertEqual(seg.scientific_revision, 0)
            self.assertEqual(seg.geometry_revision, 0)
        self.assertTrue(segmentations[0].area_vectors_authoritative)
        self.assertTrue(segmentations[1].area_vectors_authoritative)
        self.assertFalse(segmentations[2].area_vectors_authoritative)

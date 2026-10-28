from __future__ import annotations

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase
from shapely.geometry import box

from quantem.segmentation.models import (
    AnalysisMaskObject,
    GlobalMask,
    ImageSegmentation,
    SegmentationType,
)
from quantem.segmentation.type_service import get_or_create_analysis_mask_type
from quantem.testing import create_small_test_image

BEFORE = ("segmentation", "0009_correctionrevision")
AFTER = ("segmentation", "0010_segmentation_definitions_and_area_instances")


def _executor() -> MigrationExecutor:
    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    return executor


class AreaDefinitionMigrationTests(TransactionTestCase):
    available_apps = None

    def tearDown(self):
        executor = _executor()
        executor.migrate(executor.loader.graph.leaf_nodes())

    def test_legacy_tissue_and_analysis_masks_become_sample_area_and_cells(self):
        # The test database starts at the leaf. Remove the definitions the data
        # migration seeded while it was created, then build the rows that an
        # installation on 0009 would actually contain before walking back.
        SegmentationType.objects.filter(
            internal_name__in=("quantem_internal_tissue", "quantem_internal_cells")
        ).delete()
        asset = create_small_test_image("Legacy areas", width=64, height=64).asset
        legacy_type = get_or_create_analysis_mask_type()
        tissue = ImageSegmentation.objects.create(
            asset=asset,
            segmentation_type=legacy_type,
            display_name="Tissue",
        )
        first_cells = ImageSegmentation.objects.create(
            asset=asset,
            segmentation_type=legacy_type,
            display_name="Cell regions A",
        )
        second_cells = ImageSegmentation.objects.create(
            asset=asset,
            segmentation_type=legacy_type,
            display_name="Cell regions B",
        )
        raster_cells = ImageSegmentation.objects.create(
            asset=asset,
            segmentation_type=legacy_type,
            display_name="Cell raster",
        )
        GlobalMask.objects.create(
            segmentation=raster_cells,
            file_path="segmentations/legacy-cell-raster.png",
            width=64,
            height=64,
            foreground_pixels=20,
        )
        for segmentation, name, order, geometry in (
            (first_cells, "Object 1", 1, box(1, 1, 8, 8)),
            (first_cells, "Kupffer cell", 2, box(10, 1, 18, 8)),
            (second_cells, "Object 4", 1, box(20, 1, 28, 8)),
        ):
            AnalysisMaskObject.objects.create(
                segmentation=segmentation,
                name=name,
                color="#38bdf8",
                sort_order=order,
                geometry_wkb=geometry.wkb,
            )

        executor = _executor()
        executor.migrate([BEFORE])
        executor = _executor()
        executor.migrate([AFTER])
        new_apps = executor.loader.project_state([AFTER]).apps
        NewType = new_apps.get_model("segmentation", "SegmentationType")
        NewSegmentation = new_apps.get_model("segmentation", "ImageSegmentation")
        NewObject = new_apps.get_model("segmentation", "AnalysisMaskObject")
        NewGlobalMask = new_apps.get_model("segmentation", "GlobalMask")
        AreaAnnotation = new_apps.get_model("segmentation", "AnalysisMaskObjectAnnotation")

        sample_type = NewType.objects.get(internal_name="quantem_internal_tissue")
        cells_type = NewType.objects.get(internal_name="quantem_internal_cells")
        self.assertEqual(
            (sample_type.long_name, sample_type.category, sample_type.representation),
            ("Sample Area", "area", "semantic"),
        )
        self.assertEqual(
            (cells_type.category, cells_type.representation, cells_type.topology_policy),
            ("area", "instances", "non_overlapping"),
        )

        migrated_tissue = NewSegmentation.objects.get(id=tissue.id)
        self.assertEqual(migrated_tissue.segmentation_type_id, sample_type.id)
        self.assertEqual(migrated_tissue.display_name, "")

        canonical_cells = NewSegmentation.objects.get(id=first_cells.id)
        merged_cells = NewSegmentation.objects.get(id=second_cells.id)
        merged_raster_cells = NewSegmentation.objects.get(id=raster_cells.id)
        self.assertEqual(canonical_cells.segmentation_type_id, cells_type.id)
        self.assertEqual(merged_cells.superseded_by_id, canonical_cells.id)
        self.assertEqual(merged_raster_cells.superseded_by_id, canonical_cells.id)
        self.assertEqual(
            list(
                NewObject.objects.filter(segmentation=canonical_cells)
                .order_by("sort_order")
                .values_list("name", flat=True)
            ),
            ["Cell 1", "Kupffer cell", "Cell 4"],
        )
        # Raster-only sources remain attached to their superseded row when
        # the canonical Cells layer already has vector instances. Runtime
        # materialisation can then add those connected components exactly
        # once instead of hiding the raster behind an existing vector layer.
        self.assertTrue(NewGlobalMask.objects.filter(segmentation=merged_raster_cells).exists())
        self.assertFalse(AreaAnnotation.objects.exists())

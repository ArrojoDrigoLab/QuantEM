import numpy as np
from django.test import TestCase
from shapely.geometry import Point, box

from quantem.analysis.loaders import segmentation_mask
from quantem.analysis.structured_outputs import build_structured_outputs
from quantem.segmentation.global_masks import save_global_mask, save_global_mask_from_geometries
from quantem.segmentation.models import AnalysisMaskObject, ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import (
    get_or_create_analysis_mask_type,
    get_or_create_cells_type,
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image


class StructuredOutputTests(TestCase):
    def test_migrated_sample_area_unions_all_superseded_legacy_masks(self):
        image = create_small_test_image("Migrated sample area", width=32, height=32)
        sample = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_tissue_type()
        )
        legacy = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_analysis_mask_type(),
            display_name="Tissue",
            superseded_by=sample,
        )
        left = np.zeros((32, 32), dtype=bool)
        left[:, :16] = True
        right = np.zeros((32, 32), dtype=bool)
        right[:, 16:] = True
        save_global_mask(sample, left, source="manual")
        save_global_mask(legacy, right, source="manual")

        combined = segmentation_mask(sample, (32, 32))

        self.assertTrue(combined.all())

    def test_clipped_cells_warn_and_keep_full_and_intersection_areas(self):
        image = create_small_test_image("Clipped cells", width=32, height=32)
        sample = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_tissue_type()
        )
        save_global_mask_from_geometries(sample, [box(0, 0, 20, 32)], source="manual")
        cells = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_cells_type()
        )
        for order, (name, geometry) in enumerate(
            [("Cell 1", box(2, 2, 10, 10)), ("Cell 2", box(16, 2, 28, 14))], start=1
        ):
            AnalysisMaskObject.objects.create(
                segmentation=cells,
                name=name,
                color="#38bdf8",
                sort_order=order,
                geometry_wkb=geometry.wkb,
            )
        mito = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        geometry = box(17, 4, 19, 6)
        SegmentObject.objects.create(
            segmentation=mito,
            geometry=geometry,
            centroid=Point(18, 5),
            bbox=box(*geometry.bounds),
            label_state="CONFIRMED",
            source_model="manual",
            features={
                "area": 4.0,
                "perimeter": 8.0,
                "eccentricity": 0.0,
                "solidity": 1.0,
                "elongation": 1.0,
                "major_axis_length": 2.0,
                "minor_axis_length": 2.0,
                "feret_diameter_max": 2.0,
                "intensity_mean": 1.0,
                "intensity_p10": 1.0,
                "intensity_p50": 1.0,
                "intensity_p90": 1.0,
                "mean_prob": None,
            },
        )

        result, tables = build_structured_outputs(
            mito,
            {
                "compartments": {"mito": str(mito.id)},
                "tissue_segmentation_id": str(sample.id),
                "cell_segmentation_id": str(cells.id),
                "outputs": {
                    "whole_image": True,
                    "sample_area": True,
                    "all_cells": True,
                    "per_cell": True,
                    "individual_objects": True,
                },
            },
        )

        self.assertEqual(result["warnings"][0]["code"], "cells_clipped_by_sample_area")
        self.assertEqual(result["warnings"][0]["cells"], [{"id": str(AnalysisMaskObject.objects.get(name="Cell 2").id), "name": "Cell 2"}])
        cell_2 = next(row for row in tables["cell_summary.csv"] if row["cell_name"] == "Cell 2")
        self.assertGreater(cell_2["cell_area_px"], cell_2["measurement_area_px"])
        self.assertTrue(cell_2["clipped_by_sample_area"])
        object_row = tables["organelle_objects.csv"][0]
        self.assertEqual(object_row["assigned_cell_name"], "Cell 2")
        self.assertEqual(object_row["centroid_x_px"], 18.0)
        self.assertTrue(np.isfinite(object_row["area_px"]))
        region_counts = {
            row["region_kind"]: row["object_count"]
            for row in tables["region_summary.csv"]
        }
        self.assertEqual(
            region_counts,
            {"whole_image": 1, "sample_area": 1, "all_cells": 1},
        )
        self.assertIn("area_px_mean", tables["region_summary.csv"][0])

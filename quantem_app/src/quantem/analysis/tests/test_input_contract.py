"""Cross-flow coverage, measurement regions, and saved-result dependencies."""

import json

from rest_framework.exceptions import ValidationError
from shapely.geometry import box

from quantem.analysis.batches import preview_batch, start_batch
from quantem.analysis.creation import prepare_analysis
from quantem.analysis.export_selection import select_export_files
from quantem.analysis.identity import input_fingerprint, run_is_current
from quantem.analysis.input_contract import dependency_ids
from quantem.analysis.models import AnalysisRun
from quantem.analysis.superseded import mark_superseded
from quantem.analysis.tests.test_analysis_runs import AnalysisRunTestCase
from quantem.segmentation.global_masks import save_global_mask_from_geometries
from quantem.segmentation.models import (
    AnalysisMaskObject,
    CompletedROI,
    ImageSegmentation,
    SavedSegmentationResult,
    SegmentationType,
    SegmentObjectAnnotation,
)
from quantem.segmentation.saved_results import retain_completed_annotations
from quantem.segmentation.type_service import get_or_create_cells_type


class InputContractTests(AnalysisRunTestCase):
    def setUp(self):
        super().setUp()
        self.asset.preprocess_stage = "DONE"
        self.asset.save(update_fields=["preprocess_stage"])
        self.segmentation.manual_full_image = False
        self.segmentation.save(update_fields=["manual_full_image"])
        self.batch_payload = {
            "asset_ids": [str(self.asset.id)],
            "segmentation_type_ids": [str(self.segmentation.segmentation_type_id)],
        }

    def accepted_model(self):
        return SavedSegmentationResult.objects.create(
            segmentation=self.segmentation,
            name="Default model",
            full_image=True,
            is_canonical=True,
            threshold=0.5,
            source_model="quantem:mito",
        )

    def test_partial_done_roi_cannot_run_even_with_small_measurement_area(self):
        CompletedROI.objects.create(segmentation=self.segmentation, geometry=box(0, 0, 80, 80))
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/analysis/",
            {"tissue_segmentation_id": str(self.tissue.id)},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("whole image", str(response.data))
        for require_complete in (True, False):
            row = preview_batch(
                self.asset.experiment, {**self.batch_payload, "require_complete": require_complete}
            )[0]
            self.assertIn("whole image", row["reason"])
        self.assertFalse(AnalysisRun.objects.exists())

    def test_full_manual_roi_and_explicit_full_manual_confirmation_are_ready(self):
        region = CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=box(0, 0, 200, 200)
        )
        self.assertTrue(prepare_analysis(self.segmentation, {}).identity["review_complete"])
        region.delete()
        self.segmentation.manual_full_image = True
        self.segmentation.save()
        self.assertEqual(preview_batch(self.asset.experiment, self.batch_payload)[0]["reason"], "")

    def test_zero_object_model_is_ready_for_both_entry_points(self):
        from quantem.segmentation.models import SegmentObject

        SegmentObject.objects.filter(segmentation=self.segmentation).delete()
        self.accepted_model()
        self.assertTrue(prepare_analysis(self.segmentation, {}).identity["review_complete"])
        self.assertEqual(preview_batch(self.asset.experiment, self.batch_payload)[0]["reason"], "")

    def test_partial_additional_compartment_blocks_a_complete_primary(self):
        self.accepted_model()
        definition = SegmentationType.objects.create(
            internal_name="custom_partial", short_name="Custom", long_name="Custom organelle"
        )
        partial = ImageSegmentation.objects.create(asset=self.asset, segmentation_type=definition)
        with self.assertRaisesRegex(ValidationError, "whole image"):
            prepare_analysis(self.segmentation, {"compartments": {"custom": str(partial.id)}})

    def test_single_and_batch_defaults_have_identical_measurement_inputs(self):
        self.accepted_model()
        save_global_mask_from_geometries(self.tissue, [box(0, 0, 150, 200)])
        cells = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_cells_type()
        )
        AnalysisMaskObject.objects.create(
            segmentation=cells, name="Cell 1", geometry_wkb=box(0, 0, 50, 50).wkb
        )
        row = preview_batch(self.asset.experiment, self.batch_payload)[0]
        single = prepare_analysis(
            self.segmentation,
            {
                "compartments": {"mito": str(self.segmentation.id)},
                "objects_in_measurement_area": True,
                "tissue_segmentation_id": str(self.tissue.id),
                "cell_segmentation_id": str(cells.id),
                "outputs": dict.fromkeys(
                    ("whole_image", "sample_area", "all_cells", "per_cell", "individual_objects"),
                    True,
                ),
            },
        )
        self.assertEqual(row["params"], single.params)
        self.assertEqual(row["fingerprint"], single.fingerprint)

    def test_custom_area_ids_work_and_missing_areas_never_fall_back(self):
        self.accepted_model()
        definition = SegmentationType.objects.create(
            internal_name="custom_region",
            short_name="Region",
            long_name="Custom region",
            category="area",
            representation="semantic",
            measurement_mode="global",
        )
        payload = {**self.batch_payload, "mask_definition_id": str(definition.id)}
        self.assertIn("missing", preview_batch(self.asset.experiment, payload)[0]["reason"])
        area = ImageSegmentation.objects.create(asset=self.asset, segmentation_type=definition)
        save_global_mask_from_geometries(area, [box(0, 0, 100, 100)])
        row = preview_batch(self.asset.experiment, payload)[0]
        self.assertEqual(row["reason"], "")
        self.assertEqual(row["params"]["tissue_segmentation_id"], str(area.id))

    def test_cell_rename_marks_saved_analysis_without_changing_results(self):
        self.accepted_model()
        cells = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_cells_type()
        )
        cell = AnalysisMaskObject.objects.create(
            segmentation=cells, name="Cell 1", geometry_wkb=box(0, 0, 50, 50).wkb
        )
        run = start_batch(self.asset.experiment, self.batch_payload).items.get().run
        run.status = "SUCCESS"
        run.results = {"saved": 123}
        run.save()
        response = self.client.patch(
            f"/api/segmentations/{cells.id}/analysis-mask-objects/{cell.id}/",
            {"name": "Cell Alpha"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertIsNotNone(run.superseded_at)
        self.assertFalse(run_is_current(run))
        self.assertEqual(run.results, {"saved": 123})

    def test_consolidated_area_source_is_shared_dependency(self):
        source = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_cells_type(),
            superseded_by=self.tissue,
        )
        params = {"tissue_segmentation_id": str(self.tissue.id)}
        run = AnalysisRun.objects.create(
            segmentation=self.segmentation,
            status="SUCCESS",
            params=params,
            input_fingerprint=input_fingerprint(self.segmentation, params),
        )
        self.assertIn(str(source.id), dependency_ids(self.segmentation.id, params))
        mark_superseded(source)
        run.refresh_from_db()
        self.assertIsNotNone(run.superseded_at)

    def test_clipped_done_objects_keep_names_tags_and_metadata(self):
        obj = self.objects[0]
        SegmentObjectAnnotation.objects.create(
            object=obj,
            display_name="Tracked mitochondrion",
            tags=["example"],
            metadata={"reviewer": "test"},
        )
        CompletedROI.objects.create(segmentation=self.segmentation, geometry=box(0, 0, 40, 200))
        retain_completed_annotations(self.segmentation)
        from quantem.segmentation.models import SegmentObject

        kept = SegmentObject.objects.get(segmentation=self.segmentation, superseded_at=None)
        self.assertEqual(kept.geometry.area, 200)
        self.assertEqual(kept.annotation.display_name, "Tracked mitochondrion")
        self.assertEqual(kept.annotation.tags, ["example"])
        self.assertEqual(kept.annotation.metadata, {"reviewer": "test"})
        obj.refresh_from_db()
        self.assertIsNotNone(obj.superseded_at)

    def test_file_choices_cover_both_object_formats_and_preserve_other_outputs(self):
        self.exports_root.mkdir(parents=True, exist_ok=True)
        for name in ("objects.csv", "organelle_objects.csv", "region_summary.csv"):
            (self.exports_root / name).write_text("saved", encoding="utf-8")
        (self.exports_root / "manifest.json").write_text(
            json.dumps(
                {
                    "outputs": {
                        name: {}
                        for name in ("objects.csv", "organelle_objects.csv", "region_summary.csv")
                    }
                }
            ),
            encoding="utf-8",
        )
        run = AnalysisRun(
            segmentation=self.segmentation,
            export_dir=str(self.exports_root),
            params={"output_files": {"objects.csv": False, "organelle_objects.csv": False}},
        )
        select_export_files(run)
        self.assertFalse((self.exports_root / "objects.csv").exists())
        self.assertFalse((self.exports_root / "organelle_objects.csv").exists())
        self.assertTrue((self.exports_root / "region_summary.csv").exists())
        manifest = json.loads((self.exports_root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(list(manifest["outputs"]), ["region_summary.csv"])

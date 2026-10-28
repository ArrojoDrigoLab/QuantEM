import csv
import hashlib
import io
import json
import zipfile

import numpy as np
from django.utils import timezone
from shapely.geometry import box

from quantem.analysis.batch_views import batch_body, measurement_rows, sample_summary
from quantem.analysis.batches import preview_batch, start_batch
from quantem.analysis.identity import run_is_current
from quantem.analysis.job import run_job
from quantem.analysis.models import AnalysisBatch, AnalysisRun
from quantem.analysis.tests.test_analysis_runs import AnalysisRunTestCase
from quantem.finetune.tests.fixtures import FakeCancel, FakeReporter
from quantem.jobs.models import Job
from quantem.library.models import Dataset, ImageWorkflowMetadata
from quantem.segmentation.global_masks import save_global_mask
from quantem.segmentation.models import AnalysisMaskObject, ImageSegmentation
from quantem.segmentation.type_service import (
    get_or_create_analysis_mask_type,
    get_or_create_cells_type,
    get_or_create_er_type,
)


class ExperimentBatchTests(AnalysisRunTestCase):
    def setUp(self):
        super().setUp()
        self.asset.preprocess_stage = "DONE"
        self.asset.save(update_fields=["preprocess_stage"])
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save(update_fields=["status_stage"])
        self.experiment = self.asset.experiment
        self.payload = {
            "asset_ids": [str(self.asset.id)],
            "segmentation_types": [self.segmentation.segmentation_type.internal_name],
            "name": "Broad metrics",
        }

    def _execute(self, item):
        run_job(item.job.payload_json, FakeReporter(), FakeCancel())
        item.job.status = "SUCCESS"
        item.job.save(update_fields=["status"])
        item.run.refresh_from_db()
        return item.run

    def test_overlapping_datasets_and_image_ids_measure_once(self):
        first = Dataset.objects.create(experiment=self.experiment, name="A")
        second = Dataset.objects.create(experiment=self.experiment, name="B")
        self.asset.datasets.add(first, second)
        batch = start_batch(
            self.experiment, {**self.payload, "dataset_ids": [str(first.id), str(second.id)]}
        )
        self.assertEqual(batch.items.count(), 1)
        self.assertEqual(Job.objects.filter(batch_id=f"analysis-batch:{batch.id}").count(), 1)

    def test_partial_predictions_are_not_whole_coverage_and_masks_never_fall_back(self):
        self.segmentation.status_stage = "CANDIDATES_READY"
        self.segmentation.manual_full_image = False
        self.segmentation.save()
        self.assertIn("whole image", preview_batch(self.experiment, self.payload)[0]["reason"])
        unchecked = {**self.payload, "require_complete": False}
        self.assertIn("whole image", preview_batch(self.experiment, unchecked)[0]["reason"])
        masked = {**unchecked, "mask_name": "Missing tissue"}
        self.assertIn("missing", preview_batch(self.experiment, masked)[0]["reason"])
        self.assertFalse(AnalysisBatch.objects.exists())

    def test_definition_id_batch_defaults_to_saved_sample_area_and_cells(self):
        sample = self.tissue
        sample.status_stage = "CANDIDATES_READY"
        sample.save(update_fields=["status_stage"])
        sample_pixels = np.zeros((200, 200), dtype=bool)
        sample_pixels[:150] = True
        save_global_mask(sample, sample_pixels, source="manual")
        cells = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_cells_type(),
            status_stage="CANDIDATES_READY",
        )
        cell_geometry = box(4, 4, 40, 40)
        AnalysisMaskObject.objects.create(
            segmentation=cells,
            name="Cell 1",
            color="#38bdf8",
            sort_order=1,
            geometry_wkb=cell_geometry.wkb,
        )

        row = preview_batch(
            self.experiment,
            {
                "asset_ids": [str(self.asset.id)],
                "segmentation_type_ids": [str(self.segmentation.segmentation_type_id)],
                "mask_definition_id": str(sample.segmentation_type_id),
            },
        )[0]

        self.assertEqual(row["reason"], "")
        self.assertEqual(row["params"]["tissue_segmentation_id"], str(sample.id))
        self.assertEqual(row["params"]["cell_segmentation_id"], str(cells.id))
        self.assertTrue(row["params"]["outputs"]["per_cell"])
        self.assertTrue(row["params"]["structured_outputs_enabled"])

    def test_reuse_matches_inputs_and_metadata_and_rejects_changed_calibration(self):
        ImageWorkflowMetadata.objects.create(asset=self.asset, sample_id="Donor 1", group="T1D")
        batch = start_batch(self.experiment, self.payload)
        run = self._execute(batch.items.get())
        self.assertTrue(run_is_current(run))
        reused = start_batch(self.experiment, self.payload).items.get()
        self.assertTrue(reused.reused)
        self.assertEqual(reused.run_id, run.id)
        self.asset.pixel_size_nm = 12
        self.asset.save(update_fields=["pixel_size_nm"])
        self.assertFalse(run_is_current(run))  # Also works on an already-loaded run/asset.
        self.assertFalse(start_batch(self.experiment, self.payload).items.get().reused)

    def test_queued_prediction_is_not_ready_to_measure(self):
        Job.objects.create(
            type="run_segmentation_full_task",
            status="PENDING",
            payload_json={"segmentation_id": str(self.segmentation.id)},
        )
        self.assertIn(
            "queued or running", preview_batch(self.experiment, self.payload)[0]["reason"]
        )

    def test_changed_queued_inputs_fail_without_publishing_measurements(self):
        item = start_batch(self.experiment, self.payload).items.get()
        self.objects[0].label_state = "EXCLUDED"
        self.objects[0].save()
        with self.assertRaisesRegex(ValueError, "Inputs changed"):
            self._execute(item)
        item.run.refresh_from_db()
        self.assertEqual(item.run.status, "FAILED")
        self.assertEqual(item.run.export_dir, "")

    def test_global_area_is_durable_exportable_and_fingerprint_tracks_masks(self):
        seg = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_er_type(), status_stage="COMPLETED"
        )
        pixels = np.zeros((200, 200), dtype=bool)
        pixels[:100] = True
        save_global_mask(seg, pixels, source="manual")
        batch = start_batch(
            self.experiment,
            {**self.payload, "segmentation_types": [seg.segmentation_type.internal_name]},
        )
        run = self._execute(batch.items.get())
        self.assertEqual(
            run.results["structured_analysis"]["region_summary"][0]["area_fraction"], 0.5
        )
        self.assertTrue((run.export_path / "region_summary.csv").is_file())
        self.assertTrue(run_is_current(run))
        save_global_mask(seg, ~pixels, source="manual")
        self.assertFalse(run_is_current(run))

    def test_named_mask_restricts_object_counts_by_centroid(self):
        mask = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_analysis_mask_type(),
            display_name="Cell",
        )
        pixels = np.zeros((200, 200), dtype=bool)
        pixels[:80, :80] = True
        save_global_mask(mask, pixels, source="manual")
        batch = start_batch(self.experiment, {**self.payload, "mask_name": "Cell"})
        run = self._execute(batch.items.get())
        self.assertEqual(run.results["objects"]["n"], 1)
        self.assertEqual(run.results["composition"]["tissue_px"], 6400)
        rows = measurement_rows(batch_body(batch))
        self.assertEqual(rows[0]["n_objects"], 1)
        self.assertTrue(any(key.startswith("mean_") for key in rows[0]))

    def test_export_choices_persist_without_changing_previous_outputs(self):
        first = self._execute(start_batch(self.experiment, self.payload).items.get())
        original = (first.export_path / "objects.csv").read_bytes()
        payload = {**self.payload, "output_files": {"objects.csv": False}}
        second = self._execute(start_batch(self.experiment, payload).items.get())
        self.assertNotEqual(first.id, second.id)
        self.assertFalse((second.export_path / "objects.csv").exists())
        self.assertTrue((second.export_path / "image_summary.csv").exists())
        self.assertEqual((first.export_path / "objects.csv").read_bytes(), original)
        self.assertTrue(run_is_current(second))
        self.assertTrue(start_batch(self.experiment, payload).items.get().reused)

    def test_superseded_history_retains_measurements_and_regeneration_creates_new_run(self):
        from quantem.analysis.history import history_page
        from quantem.analysis.superseded import mark_superseded

        batch = start_batch(self.experiment, self.payload)
        run = self._execute(batch.items.get())
        original = (run.export_path / "objects.csv").read_bytes()
        result = run.results
        mark_superseded(self.segmentation)
        run.refresh_from_db()
        self.assertFalse(run_is_current(run))
        self.assertEqual(run.results, result)
        self.assertEqual((run.export_path / "objects.csv").read_bytes(), original)
        entry = next(
            row for row in history_page(self.experiment)["results"] if row["id"] == str(batch.id)
        )
        self.assertEqual(entry["superseded_count"], 1)
        self.assertEqual(entry["superseded_scope"], "all")
        response = self.client.post(
            f"/api/analysis-batches/{batch.id}/", {"action": "regenerate"}, format="json"
        )
        self.assertEqual(response.status_code, 201, response.data)
        self.assertNotEqual(response.data["id"], str(batch.id))
        new_item = AnalysisBatch.objects.get(id=response.data["id"]).items.get()
        self.assertNotEqual(new_item.run_id, run.id)
        self.assertFalse(new_item.reused)

    def test_partial_zip_has_status_manifest_and_optional_object_tables(self):
        batch = start_batch(
            self.experiment,
            {
                **self.payload,
                "segmentation_types": [*self.payload["segmentation_types"], "missing"],
            },
        )
        item = batch.items.exclude(run=None).get()
        self._execute(item)
        response = self.client.get(f"/api/analysis-batches/{batch.id}/export/?objects=true")
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(b"".join(response.streaming_content))) as archive:
            rows = list(csv.DictReader(io.StringIO(archive.read("measurements.csv").decode())))
            self.assertEqual({row["status"] for row in rows}, {"SUCCESS", "SKIPPED"})
            self.assertTrue(any(name.endswith("/objects.csv") for name in archive.namelist()))
            samples = list(csv.DictReader(io.StringIO(archive.read("samples.csv").decode())))
            groups = list(csv.DictReader(io.StringIO(archive.read("groups.csv").decode())))
            self.assertTrue(samples)
            self.assertEqual({row["sample_id"] for row in samples}, {self.asset.display_name})
            self.assertEqual({row["n_units"] for row in groups}, {"1"})
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["settings"]["name"], "Broad metrics")
            self.assertTrue(any(entry["notes"] for entry in manifest["items"]))
            for name in archive.namelist():
                if name.endswith(".csv"):
                    columns = next(
                        csv.reader(io.StringIO(archive.read(name).decode("utf-8-sig"))), []
                    )
                    self.assertNotIn("caveats", columns)
                    self.assertNotIn("n_caveats", columns)
                elif name.startswith("runs/") and name.endswith("manifest.json"):
                    saved_manifest = json.loads(archive.read(name))
                    directory = name.rsplit("/", 1)[0]
                    for entry in saved_manifest["outputs"]["files"]:
                        content = archive.read(f"{directory}/{entry['filename']}")
                        self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
        # Streaming through APIClient already closes the response safely.
        self.assertTrue(response.closed)

    def test_cancelling_queued_batch_is_terminal_and_retry_keeps_old_run(self):
        batch = start_batch(self.experiment, self.payload)
        item = batch.items.get()
        first_run = item.run_id
        response = self.client.post(
            f"/api/analysis-batches/{batch.id}/", {"action": "cancel"}, format="json"
        )
        self.assertEqual(response.status_code, 200, response.data)
        item.refresh_from_db()
        self.assertEqual(item.job.status, "CANCELLED")
        self.assertEqual(item.run.status, "FAILED")
        response = self.client.post(
            f"/api/analysis-batches/{batch.id}/", {"action": "retry_failed"}, format="json"
        )
        self.assertEqual(response.status_code, 200, response.data)
        item.refresh_from_db()
        self.assertNotEqual(item.run_id, first_run)
        self.assertTrue(AnalysisRun.objects.filter(id=first_run).exists())

    def test_superseded_objects_do_not_change_live_metrics(self):
        self.objects[0].superseded_at = timezone.now()
        self.objects[0].save()
        run = self._execute(start_batch(self.experiment, self.payload).items.get())
        self.assertEqual(run.results["objects"]["n"], 2)

    def test_metadata_change_invalidates_existing_run(self):
        run = self._execute(start_batch(self.experiment, self.payload).items.get())
        ImageWorkflowMetadata.objects.create(asset=self.asset, sample_id="corrected donor")
        self.assertFalse(run_is_current(run))

    def test_invalid_ids_and_cross_experiment_scope_are_rejected(self):
        response = self.client.post(
            f"/api/experiments/{self.experiment.id}/analysis-batches/",
            {**self.payload, "asset_ids": ["bad"], "preview": True},
            format="json",
        )
        self.assertEqual(response.status_code, 400)

    def test_sample_means_do_not_weight_a_donor_by_image_count(self):
        common = {
            "status": "SUCCESS",
            "current": True,
            "group": "T1D",
            "organelle": "mito",
            "measurement_area": "Whole image",
        }
        rows = [{**common, "sample_id": "A", "area_fraction_mito": value} for value in (0.1, 0.3)]
        rows += [
            {**common, "sample_id": "B", "area_fraction_mito": 0.8},
        ]
        samples, groups = sample_summary(rows)
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(groups[0]["mean"], 0.5)
        self.assertEqual(groups[0]["n_units"], 2)

    def test_unnamed_samples_contribute_and_duplicate_filenames_stay_separate(self):
        common = {
            "status": "SUCCESS",
            "current": True,
            "group": "",
            "organelle": "mito",
            "measurement_area": "Whole image",
            "image": "scan.tif",
        }
        rows = [
            {**common, "asset_id": "one", "sample_id": "", "n_objects": 10},
            {**common, "asset_id": "two", "sample_id": "   ", "n_objects": 20},
            {**common, "asset_id": "three", "sample_id": "scan.tif", "n_objects": 60},
        ]
        samples, groups = sample_summary(rows)
        self.assertEqual(len(samples), 3)
        self.assertEqual({row["sample_id"] for row in samples}, {"scan.tif"})
        self.assertEqual(groups[0]["n_units"], 3)
        self.assertAlmostEqual(groups[0]["mean"], 30)

    def test_existing_batch_exports_use_image_names_without_remeasuring(self):
        batch = start_batch(self.experiment, self.payload)
        item = batch.items.get()
        run = self._execute(item)
        # Batches saved before the default changed have an empty sample ID.
        item.identity.pop("sample_is_explicit")
        item.identity["sample_id"] = ""
        item.save(update_fields=["identity"])
        rows = measurement_rows(batch_body(batch))
        self.assertEqual(rows[0]["sample_id"], self.asset.display_name)
        self.assertFalse(rows[0]["sample_is_explicit"])
        self.assertEqual(sample_summary(rows)[1][0]["n_units"], 1)
        self.assertTrue(run_is_current(run))

    def test_missing_scale_does_not_exclude_available_metrics(self):
        common = {
            "status": "SUCCESS",
            "current": True,
            "sample_id": "",
            "group": "",
            "organelle": "mito",
            "measurement_area": "Whole image",
        }
        rows = [
            {
                **common,
                "asset_id": "one",
                "image": "One",
                "pixel_size_nm": None,
                "n_objects": 10,
                "mean_area_um2": None,
            },
            {
                **common,
                "asset_id": "two",
                "image": "Two",
                "pixel_size_nm": 4,
                "n_objects": 20,
                "mean_area_um2": 0.5,
            },
            {**common, "asset_id": "old", "image": "Old", "current": False, "n_objects": 1000},
            {
                **common,
                "asset_id": "failed",
                "image": "Failed",
                "status": "FAILED",
                "n_objects": 1000,
            },
        ]
        _, groups = sample_summary(rows)
        counts = next(row for row in groups if row["metric"] == "n_objects")
        self.assertEqual(counts["n_units"], 2)
        self.assertEqual(counts["mean"], 15)
        area = next(row for row in groups if row["metric"] == "mean_area_um2")
        self.assertEqual(area["n_units"], 1)

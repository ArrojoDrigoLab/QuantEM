from django.test import TestCase
from rest_framework.test import APIClient

from quantem.analysis.models import AnalysisBatch, AnalysisBatchItem, AnalysisRun
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


class AnalysisHistoryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.asset = create_small_test_image("History image", width=32, height=32).asset
        self.experiment = self.asset.experiment
        self.seg = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        self.identity = {
            "asset_id": str(self.asset.id),
            "image": self.asset.display_name,
            "organelle": self.seg.segmentation_type.internal_name,
        }
        self.root = f"/api/experiments/{self.experiment.id}/"

    def run_record(self, **kwargs):
        return AnalysisRun.objects.create(
            segmentation=self.seg, status="SUCCESS", input_identity=self.identity, **kwargs
        )

    def batch_record(self, runs, reused=False):
        batch = AnalysisBatch.objects.create(
            experiment=self.experiment,
            experiment_name=self.experiment.name,
            name="Saved batch",
            settings={"mask_name": "Cells"},
        )
        for run in runs:
            AnalysisBatchItem.objects.create(
                batch=batch,
                run=run,
                segmentation=self.seg,
                identity=self.identity,
                params={"seed": 42},
                reused=reused,
            )
        return batch

    def test_batch_counts_once_per_image_and_individual_runs_remain_visible(self):
        single = self.run_record()
        batch = self.batch_record([self.run_record(), self.run_record()])
        rows = self.client.get(self.root + "analysis-history/").data
        self.assertEqual({row["id"] for row in rows}, {str(single.id), str(batch.id)})
        self.assertEqual(
            self.client.get(self.root + "workspace/").data["images"][0]["analysis_count"], 2
        )
        detail = self.client.get(f"/api/analysis-batches/{batch.id}/").data
        self.assertEqual(detail["settings"], {"mask_name": "Cells"})
        self.assertEqual(detail["items"][0]["params"], {"seed": 42})

    def test_reuse_does_not_erase_an_original_individual_analysis(self):
        single = self.run_record()
        self.batch_record([single], reused=True)
        self.assertEqual(len(self.client.get(self.root + "analysis-history/").data), 2)

    def test_failed_and_skipped_images_are_not_counted_as_done(self):
        failed = AnalysisRun.objects.create(
            segmentation=self.seg, status="FAILED", input_identity=self.identity
        )
        self.batch_record([failed])
        self.assertEqual(
            self.client.get(self.root + "workspace/").data["images"][0]["analysis_count"], 0
        )
        rows = self.client.get(self.root + f"analysis-history/?asset_id={self.asset.id}").data
        self.assertEqual(rows, [])
        self.assertEqual(len(self.client.get(self.root + "analysis-history/").data), 1)

    def test_deleted_segmentation_history_is_found_by_saved_image_identity(self):
        run = self.run_record()
        self.seg.delete()
        self.assertEqual(
            self.client.get(self.root + "analysis-history/").data[0]["id"], str(run.id)
        )

    def test_older_run_is_found_by_image_key_without_new_identity_fields(self):
        run = AnalysisRun.objects.create(
            status="SUCCESS", results={"image_key": str(self.asset.id)}
        )
        self.assertEqual(
            self.client.get(self.root + "analysis-history/").data[0]["id"], str(run.id)
        )

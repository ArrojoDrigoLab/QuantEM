"""``DELETE /api/segmentations/<id>/`` -- remove a segmentation and its record.

Paper-cut: there was no way to delete a segmentation at all, and because the
create endpoint is get_or_create per (asset, type), a segmentation created by
mistake occupied its organelle preset forever -- "Add segmentation" filtered the
preset out for as long as the row existed.

The rules mirror Mark Image Done's discard, because this is strictly more
destructive than that:

* refused while any job is queued/running/retrying on the segmentation, naming
  the job (pulling rows out from under a worker is a crash);
* refused while the completion lock is on ("done" stays final until unlocked);
* the optional acknowledged object count must match a fresh read, so a dialog
  that quoted a stale number cannot delete objects nobody was shown;
* analysis runs are *kept*: the run and its export bundle are the record of an
  analysis that happened, and they survive marked ``segmentation_deleted``.
"""

from __future__ import annotations

from django.test import TestCase
from shapely.geometry import Polygon

from quantem.analysis.models import AnalysisRun
from quantem.analysis.serializers import (
    AnalysisRunSerializer,
    AnalysisRunSummarySerializer,
)
from quantem.core.config import PROB_MAPS_DIR
from quantem.finetune.models import Adapter
from quantem.jobs.models import Job
from quantem.segmentation.models import (
    ImageSegmentation,
    ProbabilityMap,
    SegmentationOverlayState,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff.paths import get_overlay_root
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


class _DeleteTestBase(TestCase):
    def setUp(self):
        self.image = create_small_test_image("seg-delete")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )
        self.url = f"/api/segmentations/{self.segmentation.id}/"

    def _segment(self, index: int, *, label_state: str) -> SegmentObject:
        x = 10 + (index * 12)
        polygon = Polygon(((x, 10), (x + 8, 10), (x + 8, 18), (x, 18), (x, 10)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            source_model="quantem:mito",
            confidence_score=0.8,
            features={"mito_generated": True},
        )

    def _populate(self) -> None:
        self._segment(0, label_state="CONFIRMED")
        self._segment(1, label_state="EXCLUDED")
        self._segment(2, label_state="CANDIDATE")
        ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            name="MITO_test",
            file_path=f"prob_maps/{self.segmentation.id}/mito_test_prob.png",
        )
        SegmentationOverlayState.objects.create(
            segmentation=self.segmentation,
            candidate_source_model="quantem:mito",
        )
        Adapter.objects.create(
            segmentation=self.segmentation,
            base_model="quantem:mito",
            name="mito @ test",
        )


class DeletePreviewTests(_DeleteTestBase):
    """The confirm dialog quotes these counts, so they must exist to ask for."""

    def test_get_reports_live_counts_of_everything_deletion_destroys(self):
        self._populate()
        AnalysisRun.objects.create(segmentation=self.segmentation)

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        preview = response.data["delete_preview"]
        self.assertEqual(preview["object_count"], 3)
        self.assertEqual(preview["objects_by_label_state"]["CONFIRMED"], 1)
        self.assertEqual(preview["objects_by_label_state"]["EXCLUDED"], 1)
        self.assertEqual(preview["objects_by_label_state"]["CANDIDATE"], 1)
        self.assertEqual(preview["probability_map_count"], 1)
        self.assertEqual(preview["overlay_count"], 1)
        self.assertEqual(preview["adapter_count"], 1)
        self.assertEqual(preview["analysis_run_count"], 1)
        self.assertFalse(preview["locked"])
        # The ordinary serialized segmentation rides along, so one GET serves
        # both the dialog and any caller that just wants the row.
        self.assertEqual(response.data["id"], str(self.segmentation.id))

    def test_get_changes_nothing(self):
        self._populate()

        self.client.get(self.url)

        self.assertTrue(
            ImageSegmentation.objects.filter(id=self.segmentation.id).exists()
        )
        self.assertEqual(
            SegmentObject.objects.filter(segmentation=self.segmentation).count(), 3
        )


class DeleteRefusalTests(_DeleteTestBase):
    def test_refused_while_a_job_is_active_on_the_segmentation(self):
        job = Job.enqueue(
            job_type="run_segmentation_full_task",
            payload={"segmentation_id": str(self.segmentation.id)},
        )

        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 409)
        # The reason is named the way Tasks & Queues names it, and the screen
        # with the control on it is named. The job's id and type stay in the
        # payload as fields: this ``detail`` is rendered verbatim in the confirm
        # dialog, and it used to read "while a run_segmentation_full_task job is
        # queued on it (job 04a1...). Wait for it or remove it from the queue
        # (DELETE /api/jobs/04a1.../)" -- four of invariant I-12's classes in one
        # sentence, in front of someone who cannot issue a request.
        detail = response.data["detail"]
        self.assertIn("Run full-image segmentation", detail)
        self.assertIn("Tasks & Queues", detail)
        self.assertNotIn(str(job.id), detail)
        self.assertNotIn("run_segmentation_full_task", detail)
        self.assertNotIn("/api/", detail)
        self.assertEqual(response.data["job_id"], str(job.id))
        self.assertEqual(response.data["job_type"], "run_segmentation_full_task")
        self.assertTrue(
            ImageSegmentation.objects.filter(id=self.segmentation.id).exists()
        )

    def test_refused_while_locked_by_mark_image_done(self):
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save(update_fields=["status_stage"])

        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.data["locked"])
        self.assertIn("marked done", response.data["detail"])
        self.assertTrue(
            ImageSegmentation.objects.filter(id=self.segmentation.id).exists()
        )

    def test_refused_when_the_acknowledged_count_is_stale(self):
        self._segment(0, label_state="CANDIDATE")
        self._segment(1, label_state="CANDIDATE")

        response = self.client.delete(
            self.url,
            data={"acknowledged_object_count": 1},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Nothing was deleted", response.data["detail"])
        self.assertEqual(response.data["delete_preview"]["object_count"], 2)
        self.assertTrue(
            ImageSegmentation.objects.filter(id=self.segmentation.id).exists()
        )


class DeleteTests(_DeleteTestBase):
    def test_delete_removes_the_segmentation_and_everything_it_owns(self):
        self._populate()
        overlay_dir = get_overlay_root(str(self.segmentation.id))
        (overlay_dir / "versions").mkdir(parents=True, exist_ok=True)
        prob_dir = PROB_MAPS_DIR / str(self.segmentation.id)
        prob_dir.mkdir(parents=True, exist_ok=True)
        (prob_dir / "mito_test_prob.png").write_bytes(b"png")

        response = self.client.delete(
            self.url,
            data={"acknowledged_object_count": 3},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["deleted"]["object_count"], 3)
        self.assertFalse(
            ImageSegmentation.objects.filter(id=self.segmentation.id).exists()
        )
        self.assertEqual(SegmentObject.objects.count(), 0)
        self.assertEqual(ProbabilityMap.objects.count(), 0)
        self.assertEqual(SegmentationOverlayState.objects.count(), 0)
        self.assertEqual(Adapter.objects.count(), 0)
        self.assertFalse(overlay_dir.exists())
        self.assertFalse(prob_dir.exists())

    def test_delete_keeps_analysis_runs_marked_segmentation_deleted(self):
        run = AnalysisRun.objects.create(
            segmentation=self.segmentation,
            status=AnalysisRun.STATUS_SUCCESS,
            results={"calibrated": True},
            export_dir="D:/somewhere/bundle",
        )

        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["analysis_runs_kept"], 1)
        run.refresh_from_db()
        self.assertIsNone(run.segmentation_id)
        # The run's own record is intact; only the reference is gone.
        self.assertEqual(run.export_dir, "D:/somewhere/bundle")
        self.assertTrue(AnalysisRunSerializer(run).data["segmentation_deleted"])
        self.assertTrue(
            AnalysisRunSummarySerializer(run).data["segmentation_deleted"]
        )

    def test_a_surviving_run_does_not_claim_deletion_before_it_happens(self):
        run = AnalysisRun.objects.create(segmentation=self.segmentation)
        self.assertFalse(AnalysisRunSerializer(run).data["segmentation_deleted"])

    def test_the_organelle_preset_is_creatable_again_after_delete(self):
        """The whole point of the paper-cut: delete, then recreate."""
        response = self.client.delete(self.url)
        self.assertEqual(response.status_code, 200)

        recreate = self.client.post(
            f"/api/assets/{self.image.asset.id}/segmentations/",
            data={"segmentation_type_name": "Mitochondria"},
            content_type="application/json",
        )

        self.assertEqual(recreate.status_code, 201)
        self.assertNotEqual(recreate.data["id"], str(self.segmentation.id))
        self.assertEqual(
            ImageSegmentation.objects.filter(asset=self.image.asset).count(), 1
        )

    def test_a_finished_job_does_not_block_deletion(self):
        Job.enqueue(
            job_type="run_segmentation_full_task",
            payload={"segmentation_id": str(self.segmentation.id)},
        )
        Job.objects.update(status="SUCCESS")

        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(
            ImageSegmentation.objects.filter(id=self.segmentation.id).exists()
        )

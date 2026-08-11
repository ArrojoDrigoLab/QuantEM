"""Removing a queued job concludes the thing it was carrying.

``DELETE /api/jobs/<id>/`` is the only exit a queued job has -- ``JobCancelView``
refuses anything that is not RUNNING with a 409 -- and it *hard-deletes the
row*. That made it the one terminal path with no way back: every safety net in
``jobs.failure_reconcile`` is reached through the scheduler's orphan reaper,
which iterates ``status="RUNNING"`` and can never see a row that no longer
exists.

Measured through the endpoint before the fix, one click on "Remove" (or
"Cancel all", which deletes a whole queue) left:

==============================  =====================================
queued job removed              domain object left at
==============================  =====================================
``run_analysis``                ``AnalysisRun.status = PENDING``
``run_segmentation_full_task``  ``ImageSegmentation`` stage ``PENDING``
``train_organelle_adapter``     ``Adapter.status = PENDING``
==============================  =====================================

each with an empty ``error`` and no queue row left to explain it: the Analysis
screen promising results for a run nothing would ever pick up, and the Adapt
wizard reading a PENDING adapter as work in flight. Exactly the class
``failure_reconcile``'s docstring exists to prevent, and the adapter is the case
it names as worst.

These drive the HTTP endpoint rather than the reconciler, because the reconciler
was already correct -- it was simply never called from here.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.analysis.models import AnalysisRun
from quantem.assets.models import Asset
from quantem.finetune.models import Adapter
from quantem.jobs.constants import (
    ACTIVE_SEGMENTATION_JOB_TYPES,
    JOB_TYPE_RUN_ANALYSIS,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)
from quantem.jobs.failure_reconcile import REMOVED_FROM_QUEUE_DETAIL
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, SegmentationType

JOB_TYPE_RUN_SEGMENTATION_FULL = "run_segmentation_full_task"


def test_the_segmentation_job_type_under_test_is_a_real_one():
    """A job type the reconciler table does not know is a silent no-op."""
    assert JOB_TYPE_RUN_SEGMENTATION_FULL in ACTIVE_SEGMENTATION_JOB_TYPES


def _segmentation() -> ImageSegmentation:
    asset = Asset.objects.create(display_name="img", original_filename="img.tif")
    seg_type, _ = SegmentationType.objects.get_or_create(
        internal_name="dino_mito",
        defaults={"short_name": "Mito", "long_name": "Mitochondria"},
    )
    return ImageSegmentation.objects.create(asset=asset, segmentation_type=seg_type)


class RemovedQueuedJobTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _queued(self, job_type: str, payload: dict, *, status: str = "PENDING") -> Job:
        return Job.objects.create(type=job_type, status=status, payload_json=payload)

    def _remove(self, job: Job):
        return self.client.delete(f"/api/jobs/{job.id}/")

    def test_removing_a_queued_analysis_does_not_leave_it_pending_forever(self):
        run = AnalysisRun.objects.create(segmentation=_segmentation(), status="PENDING")
        job = self._queued(JOB_TYPE_RUN_ANALYSIS, {"analysis_run_id": str(run.id)})

        response = self._remove(job)

        self.assertEqual(response.status_code, 204)
        self.assertFalse(Job.objects.filter(id=job.id).exists())
        run.refresh_from_db()
        self.assertNotEqual(
            run.status,
            "PENDING",
            'the Analysis screen said "This run is pending. Results appear when '
            'it finishes" about a run with no queue row behind it',
        )
        self.assertEqual(run.status, "FAILED")
        self.assertEqual(run.error, REMOVED_FROM_QUEUE_DETAIL)
        self.assertIsNotNone(run.finished_at)

    def test_removing_a_queued_training_releases_the_adapt_wizard(self):
        adapter = Adapter.objects.create(
            segmentation=_segmentation(), base_model="quantem:mito", status="PENDING"
        )
        job = self._queued(JOB_TYPE_TRAIN_ORGANELLE_ADAPTER, {"adapter_id": str(adapter.id)})

        self._remove(job)

        adapter.refresh_from_db()
        self.assertEqual(adapter.status, "FAILED")
        self.assertEqual(adapter.error, REMOVED_FROM_QUEUE_DETAIL)

    def test_removing_a_queued_segmentation_run_unsticks_the_segmentation(self):
        segmentation = _segmentation()
        segmentation.status_stage = "PENDING"
        segmentation.save(update_fields=["status_stage"])
        job = self._queued(
            JOB_TYPE_RUN_SEGMENTATION_FULL,
            {"segmentation_id": str(segmentation.id)},
        )

        self._remove(job)

        segmentation.refresh_from_db()
        self.assertEqual(segmentation.status_stage, "FAILED")
        self.assertEqual(segmentation.status_error, REMOVED_FROM_QUEUE_DETAIL)

    def test_a_job_waiting_to_retry_is_the_same_case(self):
        """RETRY is a queued state, and DELETE accepts it."""
        run = AnalysisRun.objects.create(segmentation=_segmentation(), status="PENDING")
        job = self._queued(
            JOB_TYPE_RUN_ANALYSIS,
            {"analysis_run_id": str(run.id)},
            status="RETRY",
        )

        self.assertEqual(self._remove(job).status_code, 204)
        run.refresh_from_db()
        self.assertEqual(run.status, "FAILED")

    def test_the_message_says_it_never_started_not_that_it_crashed(self):
        run = AnalysisRun.objects.create(segmentation=_segmentation(), status="PENDING")
        job = self._queued(JOB_TYPE_RUN_ANALYSIS, {"analysis_run_id": str(run.id)})

        self._remove(job)

        run.refresh_from_db()
        self.assertIn("never ran", run.error)
        self.assertIn("start it again", run.error.lower())
        self.assertNotIn("worker", run.error.lower())

    def test_a_running_job_is_still_refused_and_nothing_is_concluded(self):
        """DELETE is for queued jobs; a running one goes through cancel."""
        run = AnalysisRun.objects.create(segmentation=_segmentation(), status="RUNNING")
        job = self._queued(
            JOB_TYPE_RUN_ANALYSIS,
            {"analysis_run_id": str(run.id)},
            status="RUNNING",
        )

        response = self._remove(job)

        self.assertEqual(response.status_code, 409)
        self.assertTrue(Job.objects.filter(id=job.id).exists())
        run.refresh_from_db()
        self.assertEqual(run.status, "RUNNING")

    def test_a_finished_record_is_left_alone(self):
        """A queued job can carry an id whose record already concluded."""
        run = AnalysisRun.objects.create(segmentation=_segmentation(), status="SUCCESS")
        job = self._queued(JOB_TYPE_RUN_ANALYSIS, {"analysis_run_id": str(run.id)})

        self._remove(job)

        run.refresh_from_db()
        self.assertEqual(run.status, "SUCCESS")
        self.assertEqual(run.error, "")

    def test_removing_a_queued_job_cannot_un_mark_an_image_done(self):
        """``COMPLETED`` is a segmentation's SUCCESS and carries the completion lock."""
        segmentation = _segmentation()
        segmentation.status_stage = "COMPLETED"
        segmentation.save(update_fields=["status_stage"])
        job = self._queued(
            JOB_TYPE_RUN_SEGMENTATION_FULL,
            {"segmentation_id": str(segmentation.id)},
        )

        self._remove(job)

        segmentation.refresh_from_db()
        self.assertEqual(segmentation.status_stage, "COMPLETED")
        self.assertEqual(segmentation.status_error, "")

    def test_a_job_with_no_domain_object_still_deletes_cleanly(self):
        job = self._queued("rebuild_segmentation_overlay", {})

        response = self._remove(job)

        self.assertEqual(response.status_code, 204)
        self.assertFalse(Job.objects.filter(id=job.id).exists())

    def test_cancel_all_concludes_every_run_it_removes(self):
        """ "Cancel all" is one DELETE per queued job; none may be left behind."""
        runs = [
            AnalysisRun.objects.create(segmentation=_segmentation(), status="PENDING")
            for _ in range(3)
        ]
        jobs = [
            self._queued(JOB_TYPE_RUN_ANALYSIS, {"analysis_run_id": str(run.id)}) for run in runs
        ]

        for job in jobs:
            self.assertEqual(self._remove(job).status_code, 204)

        for run in runs:
            run.refresh_from_db()
            self.assertEqual(run.status, "FAILED")

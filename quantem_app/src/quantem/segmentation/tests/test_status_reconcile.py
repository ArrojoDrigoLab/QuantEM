"""A run with no process behind it must stop claiming to be running.

A user killed the server tree mid-run. For 4m46s -- the job heartbeat's
staleness window -- the labeling screen showed *"Run full-image segmentation,
40%"* with a disabled ``Running…`` pill for a run that no longer existed.
"""

from __future__ import annotations

from datetime import timedelta

from django.test import TestCase
from django.utils import timezone
from shapely.geometry import Polygon

from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_FULL
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.status_reconcile import (
    ABANDONED_RUN_MESSAGE,
    reconcile_segmentation_status,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


class StatusReconcileTests(TestCase):
    def setUp(self):
        self.image = create_small_test_image("ghost run")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _mid_run(self, *, stage: str = "RUNNING_INFERENCE", age_seconds: int = 600):
        ImageSegmentation.objects.filter(id=self.segmentation.id).update(
            status_stage=stage,
            status_progress=40.0,
            updated_at=timezone.now() - timedelta(seconds=age_seconds),
        )
        self.segmentation.refresh_from_db()

    def _segment(self):
        polygon = Polygon(((10, 10), (20, 10), (20, 20), (10, 20), (10, 10)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CANDIDATE",
        )

    def _job(self, status: str):
        return Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            status=status,
            payload_json={"segmentation_id": str(self.segmentation.id)},
        )

    def test_a_ghost_run_with_objects_becomes_candidates_ready(self):
        self._segment()
        self._mid_run()

        self.assertTrue(reconcile_segmentation_status(self.segmentation))

        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "CANDIDATES_READY")
        self.assertEqual(self.segmentation.status_progress, 100.0)
        self.assertEqual(self.segmentation.status_error, ABANDONED_RUN_MESSAGE)

    def test_a_ghost_run_with_nothing_to_show_becomes_unstarted(self):
        self._mid_run()

        self.assertTrue(reconcile_segmentation_status(self.segmentation))

        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "UNSTARTED")
        self.assertEqual(self.segmentation.status_progress, 0.0)

    def test_the_message_does_not_blame_the_data(self):
        # Objects from an earlier successful run are intact and correct; the
        # only thing that failed is the run that was interrupted.
        self.assertIn("Nothing already saved was lost", ABANDONED_RUN_MESSAGE)

    def test_a_live_run_is_left_alone(self):
        self._job("RUNNING")
        self._mid_run()

        self.assertFalse(reconcile_segmentation_status(self.segmentation))

        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "RUNNING_INFERENCE")

    def test_a_queued_run_is_left_alone(self):
        self._job("PENDING")
        self._mid_run()

        self.assertFalse(reconcile_segmentation_status(self.segmentation))

    def test_a_run_that_only_just_reported_is_left_alone(self):
        # A queue-less run (the CLI, a test) writes a running stage with no Job
        # row; it reports at least every half second while it is alive.
        self._mid_run(age_seconds=0)

        self.assertFalse(reconcile_segmentation_status(self.segmentation))

    def test_a_resting_stage_is_never_touched(self):
        for stage in ("UNSTARTED", "CANDIDATES_READY", "COMPLETED", "FAILED"):
            with self.subTest(stage=stage):
                ImageSegmentation.objects.filter(id=self.segmentation.id).update(
                    status_stage=stage,
                    updated_at=timezone.now() - timedelta(seconds=600),
                )
                self.segmentation.refresh_from_db()
                self.assertFalse(reconcile_segmentation_status(self.segmentation))

    def test_the_list_endpoint_repairs_what_it_reports(self):
        self._segment()
        self._mid_run()

        response = self.client.get(f"/api/assets/{self.image.asset.id}/segmentations/")

        self.assertEqual(response.status_code, 200)
        stages = {row["status_stage"] for row in response.data}
        self.assertEqual(stages, {"CANDIDATES_READY"})

    def test_extracting_candidates_is_also_a_running_stage(self):
        self._segment()
        self._mid_run(stage="EXTRACTING_CANDIDATES")

        self.assertTrue(reconcile_segmentation_status(self.segmentation))

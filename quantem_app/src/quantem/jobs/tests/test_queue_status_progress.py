"""``GET /api/jobs/queue-status/`` is the endpoint the screens actually poll.

Tile progress landed on ``GET /api/jobs/<id>/`` and stopped there. Nothing in
the product polls that route: the Tasks drawer and the labeling screen both
poll ``queue-status``, which serialised its own compact dict and dropped every
one of the new columns. The measurable consequence was that during a real
56-tile run, 30 s of sampled page text contained the word "tile" zero times and
the only count that ever reached a user was parsed out of the free-text
message.

These tests pin the wire, not the pixels: what the endpoint has to carry for a
screen to be able to draw the owner's three indicators at all.
"""

import uuid

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.jobs.constants import (
    JOB_TYPE_INSTALL_MODEL_PACK,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    QUEUE_P2_UPLOAD,
    QUEUE_P4_FULL,
)
from quantem.jobs.models import UNIT_TILE, Job

ASSET_ID = str(uuid.uuid4())


def _enqueue_run() -> Job:
    """A queued run on the shared image, so the wave rule can group them.

    Enqueue order matters: ``Job.resolve_batch`` joins a new run to an existing
    wave only while something in that wave is still open, which is the rule that
    keeps yesterday's finished runs out of today's aggregate. A test that set a
    job SUCCESS before enqueuing the next one would silently get three waves of
    one.
    """
    return Job.enqueue(
        job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
        payload={"asset_id": ASSET_ID, "segmentation_id": str(uuid.uuid4())},
        queue_name=QUEUE_P4_FULL,
        resource_class="gpu",
    )


def _set_progress(job, *, units_done=None, units_total=None, stage="inference", status="RUNNING"):
    Job.objects.filter(id=job.id).update(
        status=status,
        progress=29.82456140350877,  # 17/57: the whole-job divisor, deliberately
        progress_units_done=units_done,
        progress_units_total=units_total,
        progress_unit_label=UNIT_TILE if units_total is not None else "",
        progress_stage=stage,
        progress_detail_json={"eta_seconds": 91.4, "organelle": "mito"},
        message="Segmenting: 30% (17 of 56 tiles)",
    )
    job.refresh_from_db()
    return job


def _run_job(**kwargs) -> Job:
    return _set_progress(_enqueue_run(), **kwargs)


class QueueStatusProgressTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _running(self) -> list[dict]:
        response = self.client.get("/api/jobs/queue-status/")
        self.assertEqual(response.status_code, 200)
        return response.data["running"]

    def test_the_polled_endpoint_carries_the_tile_count(self):
        _run_job(units_done=17, units_total=56)
        (payload,) = self._running()
        self.assertEqual(
            payload["unit_progress"],
            {
                "done": 17,
                "total": 56,
                "label": "tile",
                "percent": 30.4,
                "stage": "inference",
                "eta_seconds": 91.4,
            },
        )
        self.assertEqual(payload["progress_stage"], "inference")

    def test_the_tile_percentage_divides_by_the_tiling_plan(self):
        """One divisor, and it is the plan's.

        The row's ``progress`` is 29.8 -- 17 of 57, because the whole job is the
        tiles plus the work either side of them. ``unit_progress.percent`` is
        30.4, which is 17 of the plan's 56 tiles. Both are correct answers to
        different questions, and rendering them side by side is what produced a
        56 % bar labelled "57% (Tile 32/56)". A screen draws the tile line from
        `unit_progress` and nothing else.
        """
        job = _run_job(units_done=17, units_total=56)
        (payload,) = self._running()
        self.assertEqual(payload["unit_progress"]["percent"], round(100 * 17 / 56, 1))
        self.assertNotEqual(round(job.progress, 1), payload["unit_progress"]["percent"])

    def test_a_job_that_counts_nothing_says_so_rather_than_saying_zero(self):
        _run_job(units_done=None, units_total=None, stage="")
        (payload,) = self._running()
        self.assertIsNone(payload["unit_progress"])
        self.assertIsNone(payload["download"])

    def test_the_aggregate_across_every_organelle_run_for_one_image(self):
        _run_job(units_done=17, units_total=56)
        _run_job(units_done=2, units_total=6)
        running = self._running()
        self.assertEqual(len(running), 2)
        for payload in running:
            batch = payload["batch_progress"]
            self.assertIsNotNone(batch, "the wave rollup is missing from the poll")
            self.assertEqual(batch["units_done"], 19)
            self.assertEqual(batch["units_total"], 62)
            self.assertEqual(batch["units_reachable"], 62)
            self.assertEqual(batch["runs_total"], 2)
            self.assertEqual(batch["runs_running"], 2)
            self.assertFalse(batch["complete"])
        # Both runs are the same wave, so the rollup is the same object.
        self.assertEqual(
            running[0]["batch_progress"]["batch_id"],
            running[1]["batch_progress"]["batch_id"],
        )

    def test_a_cancelled_run_leaves_the_wave_denominator_and_is_named(self):
        finished, cancelled, remaining = (
            _enqueue_run(),
            _enqueue_run(),
            _enqueue_run(),
        )
        _set_progress(finished, units_done=56, units_total=56, status="SUCCESS")
        _set_progress(cancelled, units_done=3, units_total=6, status="CANCELLED")
        _set_progress(remaining, units_done=1, units_total=4)
        response = self.client.get("/api/jobs/queue-status/")
        (payload,) = [job for job in response.data["running"] if job["id"] == str(remaining.id)]
        batch = payload["batch_progress"]
        self.assertEqual(batch["units_abandoned"], 3)
        self.assertEqual(batch["units_reachable"], 63)
        self.assertEqual(batch["runs_cancelled"], 1)

    def test_a_model_download_is_bytes_and_is_named_for_a_person(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_INSTALL_MODEL_PACK,
            payload={"pack_id": "quantem:nucleus"},
            queue_name=QUEUE_P2_UPLOAD,
        )
        Job.objects.filter(id=job.id).update(
            status="RUNNING",
            progress_current_bytes=118_000_000,
            progress_total_bytes=365_000_000,
            progress_stage="downloading_model",
        )
        (payload,) = self._running()
        self.assertEqual(payload["download"]["current_bytes"], 118_000_000)
        self.assertEqual(payload["download"]["total_bytes"], 365_000_000)
        # Structurally apart from the tiles, so a caller cannot draw one as the
        # other even by accident.
        self.assertIsNone(payload["unit_progress"])
        self.assertEqual(payload["model_pack"]["id"], "quantem:nucleus")
        self.assertTrue(payload["model_pack"]["title"])

    def test_a_pack_this_build_does_not_know_degrades_instead_of_inventing(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_INSTALL_MODEL_PACK,
            payload={"pack_id": "someone-elses:organelle"},
            queue_name=QUEUE_P2_UPLOAD,
        )
        Job.objects.filter(id=job.id).update(status="RUNNING")
        (payload,) = self._running()
        self.assertEqual(payload["model_pack"], {"id": "someone-elses:organelle", "title": ""})

    def test_the_rollup_costs_one_query_however_many_runs_share_a_wave(self):
        """Polled every three seconds: a query per run would be a real cost."""
        jobs = [_enqueue_run() for _ in range(4)]
        for job in jobs:
            _set_progress(job, units_done=5, units_total=56)
        # Four run rows, one wave, one extra SELECT for the rollup: six queries
        # without it, seven with. `batch_progress_for` per job would be ten.
        with self.assertNumQueries(7):
            self.client.get("/api/jobs/queue-status/")

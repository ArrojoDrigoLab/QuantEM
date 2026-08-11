"""Across all organelle runs for this image, X of Y tiles done.

The question the owner asked for at the top of the run panel. It is answered
server-side from one grouping key written at enqueue time (the *run wave*), so
the three cases that break a naive sum -- runs started at different times, a run
that failed, a run that was cancelled -- are answered here rather than left to
whichever screen happens to be adding numbers up.
"""

from __future__ import annotations

import uuid

from django.test import TestCase

from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    QUEUE_P2_UPLOAD,
    QUEUE_P4_FULL,
)
from quantem.jobs.models import UNIT_TILE, Job
from quantem.jobs.serializers import (
    JobSerializer,
    aggregate_batch_progress,
    batch_progress_for,
)

ASSET = str(uuid.uuid4())
OTHER_ASSET = str(uuid.uuid4())


def _run(asset_id: str = ASSET, **payload) -> Job:
    return Job.enqueue(
        job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
        payload={"asset_id": asset_id, **payload},
        resource_class="gpu",
        queue_name=QUEUE_P4_FULL,
    )


def _tiles(job: Job, done: int, total: int, status: str = "RUNNING") -> Job:
    Job.objects.filter(id=job.id).update(
        progress_units_done=done,
        progress_units_total=total,
        progress_unit_label=UNIT_TILE,
        status=status,
    )
    job.refresh_from_db()
    return job


class RunWaveGroupingTests(TestCase):
    def test_runs_started_together_share_a_wave(self):
        mito = _run(segmentation_id="mito")
        nucleus = _run(segmentation_id="nucleus")
        assert mito.batch_id
        assert nucleus.batch_id == mito.batch_id
        assert (mito.batch_seq, nucleus.batch_seq) == (0, 1)

    def test_a_run_started_while_another_is_running_joins_it(self):
        mito = _run(segmentation_id="mito")
        Job.objects.filter(id=mito.id).update(status="RUNNING")
        nucleus = _run(segmentation_id="nucleus")
        assert nucleus.batch_id == mito.batch_id

    def test_a_run_started_after_everything_finished_begins_a_new_wave(self):
        first = _run(segmentation_id="mito")
        Job.objects.filter(id=first.id).update(status="SUCCESS")
        second = _run(segmentation_id="nucleus")
        assert second.batch_id != first.batch_id
        assert second.batch_seq == 0

    def test_a_failed_wave_does_not_swallow_the_next_one(self):
        first = _run(segmentation_id="mito")
        Job.objects.filter(id=first.id).update(status="FAILED")
        second = _run(segmentation_id="mito")
        assert second.batch_id != first.batch_id

    def test_waves_are_per_image(self):
        mine = _run(ASSET)
        theirs = _run(OTHER_ASSET)
        assert mine.batch_id != theirs.batch_id

    def test_a_job_that_cannot_count_tiles_is_not_given_a_wave(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            payload={"asset_id": ASSET},
            queue_name=QUEUE_P2_UPLOAD,
        )
        assert job.batch_id == ""
        assert batch_progress_for(job) is None

    def test_an_explicit_batch_id_wins(self):
        job = _run(segmentation_id="mito")
        joined = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload={"asset_id": OTHER_ASSET},
            resource_class="gpu",
            queue_name=QUEUE_P4_FULL,
            batch_id=job.batch_id,
        )
        assert joined.batch_id == job.batch_id
        assert joined.batch_seq == 1


class AggregateTests(TestCase):
    def test_a_wave_in_flight_sums_tiles_across_its_runs(self):
        mito = _tiles(_run(segmentation_id="mito"), 531, 858, "RUNNING")
        nucleus = _tiles(_run(segmentation_id="nucleus"), 0, 88, "PENDING")

        rollup = batch_progress_for(mito)
        assert rollup["units_done"] == 531
        assert rollup["units_total"] == 946
        assert rollup["units_reachable"] == 946
        assert rollup["percent"] == 56.1
        assert rollup["runs_total"] == 2
        assert rollup["runs_running"] == 1
        assert rollup["runs_pending"] == 1
        assert rollup["complete"] is False
        assert [run["job_id"] for run in rollup["runs"]] == [
            str(mito.id),
            str(nucleus.id),
        ]

    def test_a_run_that_never_started_still_counts_in_the_denominator(self):
        """Staggered starts must not make the total grow under the user."""
        mito = _tiles(_run(segmentation_id="mito"), 100, 200, "RUNNING")
        _tiles(_run(segmentation_id="nucleus"), 0, 100, "PENDING")
        assert batch_progress_for(mito)["units_total"] == 300

    def test_a_failed_runs_tiles_stay_in_the_denominator(self):
        """The bar must not fill on a wave that ran less than half of itself.

        Dropping abandoned tiles from the denominator -- which this used to do
        -- made a wave read ``100 % · 128 of 128 tiles`` after a run died 160
        tiles short, and made the percentage jump *upwards* at the moment of
        the failure. 128 of 288 is what happened.
        """
        mito, nucleus = _run(segmentation_id="mito"), _run(segmentation_id="nucleus")
        _tiles(mito, 40, 200, "FAILED")
        _tiles(nucleus, 88, 88, "SUCCESS")

        rollup = batch_progress_for(mito)
        assert rollup["units_done"] == 128
        assert rollup["units_total"] == 288
        assert rollup["units_abandoned"] == 160
        assert rollup["units_reachable"] == 128
        assert rollup["percent"] == 44.4
        assert rollup["runs_failed"] == 1
        assert rollup["complete"] is True

    def test_a_cancelled_run_is_treated_the_same_and_named(self):
        mito, nucleus = _run(segmentation_id="mito"), _run(segmentation_id="nucleus")
        _tiles(mito, 10, 200, "CANCELLED")
        _tiles(nucleus, 44, 88, "RUNNING")

        rollup = batch_progress_for(mito)
        assert rollup["units_abandoned"] == 190
        assert rollup["units_reachable"] == 98
        assert rollup["units_done"] == 54
        assert rollup["units_total"] == 288
        assert rollup["percent"] == 18.8
        assert rollup["runs_cancelled"] == 1
        assert rollup["complete"] is False

    def test_a_wave_whose_every_tile_was_abandoned_reads_zero(self):
        mito = _tiles(_run(segmentation_id="mito"), 0, 200, "CANCELLED")
        rollup = batch_progress_for(mito)
        assert rollup["units_reachable"] == 0
        assert rollup["units_total"] == 200
        assert rollup["percent"] == 0.0
        assert rollup["runs_cancelled"] == 1
        assert rollup["complete"] is True

    def test_a_wave_the_failure_cannot_shrink_is_monotone_through_a_cancel(self):
        """The percentage never moves backwards, and never moves up on a death.

        Both directions were live: dropping abandoned tiles made 16 % become
        23 % the instant a run was cancelled, and a queued run joining the
        count made 100 % become 76 %.
        """
        mito, nucleus = _run(segmentation_id="mito"), _run(segmentation_id="nucleus")
        _tiles(mito, 19, 56, "RUNNING")
        _tiles(nucleus, 0, 62, "PENDING")
        before = batch_progress_for(mito)["percent"]

        _tiles(mito, 19, 56, "CANCELLED")
        after = batch_progress_for(mito)["percent"]

        assert before == 16.1
        assert after == before

    def test_units_of_different_kinds_are_never_added_together(self):
        mito = _tiles(_run(segmentation_id="mito"), 5, 10, "RUNNING")
        other = _run(segmentation_id="nucleus")
        Job.objects.filter(id=other.id).update(
            progress_units_done=1,
            progress_units_total=4,
            progress_unit_label="crop",
            status="RUNNING",
        )
        assert batch_progress_for(mito) is None

    def test_a_wave_with_nothing_countable_yet_is_absent_not_zero(self):
        job = _run(segmentation_id="mito")
        assert batch_progress_for(job) is None

    def test_the_eta_is_withheld_while_anything_is_still_queued(self):
        mito = _tiles(_run(segmentation_id="mito"), 100, 200, "RUNNING")
        Job.objects.filter(id=mito.id).update(
            progress_detail_json={"eta_seconds": 120.0}
        )
        _tiles(_run(segmentation_id="nucleus"), 0, 100, "PENDING")
        assert batch_progress_for(mito)["eta_seconds"] is None

    def test_the_eta_is_the_slowest_live_run(self):
        mito = _tiles(_run(segmentation_id="mito"), 100, 200, "RUNNING")
        nucleus = _tiles(_run(segmentation_id="nucleus"), 10, 100, "RUNNING")
        Job.objects.filter(id=mito.id).update(
            progress_detail_json={"eta_seconds": 120.0}
        )
        Job.objects.filter(id=nucleus.id).update(
            progress_detail_json={"eta_seconds": 400.5}
        )
        assert batch_progress_for(mito)["eta_seconds"] == 400.5

    def test_aggregate_of_nothing_is_none(self):
        assert aggregate_batch_progress([]) is None


class SerializerTests(TestCase):
    def test_a_run_serializes_tiles_and_no_download(self):
        job = _tiles(_run(segmentation_id="mito"), 531, 858, "RUNNING")
        data = JobSerializer(job).data
        assert data["progress_units_done"] == 531
        assert data["progress_units_total"] == 858
        assert data["unit_progress"]["label"] == UNIT_TILE
        assert data["unit_progress"]["percent"] == 61.9
        assert data["download"] is None
        assert data["batch_progress"]["units_done"] == 531

    def test_a_download_serializes_bytes_and_no_tiles(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            payload={"asset_id": ASSET},
            queue_name=QUEUE_P2_UPLOAD,
        )
        Job.objects.filter(id=job.id).update(
            progress_current_bytes=118_000_000, progress_total_bytes=365_000_000
        )
        job.refresh_from_db()
        data = JobSerializer(job).data
        assert data["download"] == {
            "current_bytes": 118_000_000,
            "total_bytes": 365_000_000,
            "percent": 32.3,
        }
        assert data["unit_progress"] is None
        assert data["batch_progress"] is None

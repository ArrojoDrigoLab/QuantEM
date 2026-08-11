"""Honest progress on the job row: monotone, countable, and not conflated.

Covers invariant I-3 (progress never decreases and terminates at the total) and
the owner's request that a model download and a tile walk be different
indicators rather than one bar that means two things.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from unittest.mock import patch

from django.test import TestCase

from quantem.jobs import reporter as reporter_module
from quantem.jobs.models import (
    PROGRESS_STAGES,
    STAGE_EXTRACTING,
    STAGE_INFERENCE,
    STAGE_LOADING_MODEL,
    UNIT_TILE,
    Job,
)
from quantem.jobs.reporter import (
    JobReporter,
    NullUnitProgressScope,
    active_reporter,
    report_stage,
    unit_scope,
)


def _job(**kwargs) -> Job:
    defaults = {"type": "run_segmentation_full_task", "payload_json": {}}
    defaults.update(kwargs)
    return Job.objects.create(**defaults)


@contextmanager
def _every_write_allowed():
    """Drop the write floor for a test that runs a whole scope in microseconds.

    A scope writes at most once a second, so a loop with no work in it between
    units legitimately writes twice -- a real run spends 0.7 s per tile and
    reports every one. Tests that need to see an intermediate write say so here
    rather than by sleeping.
    """
    original = reporter_module.UNIT_WRITE_MIN_INTERVAL_SECONDS
    reporter_module.UNIT_WRITE_MIN_INTERVAL_SECONDS = 0.0
    try:
        yield
    finally:
        reporter_module.UNIT_WRITE_MIN_INTERVAL_SECONDS = original


class _ManualClock:
    """A monotonic clock the test drives by hand, in seconds."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ClearsTheActiveReporter:
    """Constructing a reporter claims the thread; a test must give it back."""

    def tearDown(self):  # noqa: N802 -- unittest naming
        reporter = active_reporter()
        if reporter is not None:
            reporter.deactivate()
        super().tearDown()


class ProgressMonotonicityTests(_ClearsTheActiveReporter, TestCase):
    def test_progress_never_runs_backwards_within_an_attempt(self):
        """I-3. The measured bug was 0 -> 5 -> 71 -> 55 -> 100."""
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)

        for value in (0.0, 5.0, 71.0, 55.0, 100.0):
            reporter.update(progress=value)
        job.refresh_from_db()
        assert job.progress == 100.0

        reporter.update(progress=12.0)
        job.refresh_from_db()
        assert job.progress == 100.0

    def test_a_throttled_report_still_raises_the_high_water_mark(self):
        """The clamp must not be defeated by the rate limiter."""
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=600.0)
        reporter.update(progress=10.0)  # first write goes through
        reporter.update(progress=80.0)  # throttled away, but remembered
        reporter.update(progress=20.0, stage=STAGE_INFERENCE)  # stage forces a write
        job.refresh_from_db()
        assert job.progress == 80.0

    def test_a_stage_change_is_never_withheld_by_the_throttle(self):
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=600.0)
        reporter.update(progress=1.0)
        reporter.stage(STAGE_LOADING_MODEL, detail={"model": "quantem:mito"})
        job.refresh_from_db()
        assert job.progress_stage == STAGE_LOADING_MODEL
        assert job.progress_detail_json == {"model": "quantem:mito"}


class UnitProgressTests(_ClearsTheActiveReporter, TestCase):
    def test_the_denominator_lands_before_the_first_unit(self):
        """A run says "0 of 858 tiles" while the weights are still loading."""
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)

        reporter.unit_scope(total=858, label=UNIT_TILE, stage=STAGE_INFERENCE)

        job.refresh_from_db()
        assert job.progress_units_done == 0
        assert job.progress_units_total == 858
        assert job.progress_unit_label == UNIT_TILE
        assert job.progress_stage == STAGE_INFERENCE

    def test_units_reach_the_total_exactly_and_never_regress(self):
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        with reporter.unit_scope(total=12, label=UNIT_TILE) as scope:
            for done in range(1, 13):
                scope.set(done, total=12)
            scope.set(4)  # a stale report from a slow caller
        job.refresh_from_db()
        assert job.progress_units_done == 12
        assert job.progress_units_total == 12

    def test_the_total_is_corrected_by_the_loop_that_knows_it(self):
        """The pre-run quote loses to the plan the loop actually built.

        And closing the scope does *not* round the count up to the total: the
        loop reports the tiles it ran, and a scope that ends early says so.
        """
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        with reporter.unit_scope(total=4, label=UNIT_TILE) as scope:
            scope.set(1, total=6)
        job.refresh_from_db()
        assert job.progress_units_total == 6
        assert job.progress_units_done == 1

    def test_a_failed_scope_does_not_claim_the_run_finished(self):
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        with _every_write_allowed():
            try:
                with reporter.unit_scope(total=10, label=UNIT_TILE) as scope:
                    scope.set(3)
                    raise RuntimeError("the worker died")
            except RuntimeError:
                pass
        job.refresh_from_db()
        assert job.progress_units_done == 3
        assert job.progress_units_total == 10

    def test_the_write_floor_throttles_without_losing_the_last_unit(self):
        """At most one write a second, and the end still lands.

        No monkeypatching: 200 units with no work between them take well under
        a second, so the floor is doing the throttling for real -- the opening
        write of the denominator, then the final count.
        """
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        with reporter.unit_scope(total=200, label=UNIT_TILE) as scope:
            for done in range(1, 201):
                scope.set(done)
            stats = dict(scope.write_stats)

        assert stats["skipped"] > 0, "the floor must skip intermediate writes"
        assert stats["writes"] <= 4, stats
        job.refresh_from_db()
        assert job.progress_units_done == 200

    def test_one_slow_write_does_not_switch_the_counter_off(self):
        """The regression the cumulative write budget used to cause.

        A single UPDATE on this database was measured at **1 129.8 ms** against
        a median of 0.24 ms. Under the old rule (``write_seconds <= 1 % of
        elapsed``) that one sample bought ~113 s of silence, so on a 50 s run
        the count froze at whatever it last said and the run read as stalled --
        and a frozen counter is indistinguishable from a stalled run.

        The outlier is injected the way a real one arrives: the clock moves
        *inside* the UPDATE, so it is charged to the writer either way. Under a
        wall-clock floor it costs the samples inside its own duration and
        nothing more.
        """
        clock = _ManualClock()
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        slow = {"pending": True}
        real_filter = Job.objects.filter

        def slow_filter(*args, **kwargs):
            queryset = real_filter(*args, **kwargs)
            original = queryset.update

            def update(**values):
                if slow["pending"]:
                    slow["pending"] = False
                    clock.advance(1.13)  # the measured outlier
                return original(**values)

            queryset.update = update
            return queryset

        with (
            patch.object(reporter_module.time, "perf_counter", clock),
            patch.object(Job.objects, "filter", side_effect=slow_filter),
        ):
            scope = reporter.unit_scope(total=20, label=UNIT_TILE)
            for done in range(1, 21):
                clock.advance(0.7)  # one CPU tile
                scope.set(done)
            stats = dict(scope.write_stats)

        assert stats["writes"] >= 10, stats
        job.refresh_from_db()
        assert job.progress_units_done == 20

    def test_tiles_are_never_written_onto_a_job_that_already_concluded(self):
        """A worker keeps its reporter after the job ends; the row must not move."""
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        with _every_write_allowed():
            with reporter.unit_scope(total=10, label=UNIT_TILE) as scope:
                scope.set(10)
            Job.objects.filter(id=job.id).update(status="SUCCESS", progress_units_done=10)
            with reporter.unit_scope(total=99, label=UNIT_TILE) as stray:
                stray.set(7)
        job.refresh_from_db()
        assert job.progress_units_done == 10
        assert job.progress_units_total == 10

    def test_an_eta_appears_only_once_there_is_evidence_for_one(self):
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        with _every_write_allowed():
            scope = reporter.unit_scope(total=100, label=UNIT_TILE)
            job.refresh_from_db()
            assert "eta_seconds" not in (job.progress_detail_json or {})
            scope.set(10)
        job.refresh_from_db()
        assert (job.progress_detail_json or {})["eta_seconds"] >= 0.0


class DownloadAndTilesStayApartTests(_ClearsTheActiveReporter, TestCase):
    def test_a_tile_walk_writes_no_bytes(self):
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        with reporter.unit_scope(total=5, label=UNIT_TILE) as scope:
            scope.set(5)
        job.refresh_from_db()
        assert job.progress_current_bytes is None
        assert job.progress_total_bytes is None
        assert job.progress_unit_label == UNIT_TILE

    def test_a_download_writes_no_tiles(self):
        job = _job(type="install_model_pack")
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        reporter.update(current_bytes=118_000_000, total_bytes=365_000_000)
        job.refresh_from_db()
        assert job.progress_current_bytes == 118_000_000
        assert job.progress_units_done is None
        assert job.progress_units_total is None
        assert job.progress_unit_label == ""


class ActiveReporterTests(_ClearsTheActiveReporter, TestCase):
    def test_a_reporter_claims_its_own_thread_and_only_its_own(self):
        job = _job()
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        assert active_reporter() is reporter

        seen: list[object] = []

        def other_thread():
            seen.append(active_reporter())

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert seen == [None]
        reporter.deactivate()
        assert active_reporter() is None

    def test_unit_scope_is_a_no_op_when_nothing_is_running(self):
        """Inference from a CLI call or a test must not need a job."""
        reporter = active_reporter()
        if reporter is not None:
            reporter.deactivate()
        scope = unit_scope(total=10, label=UNIT_TILE)
        assert isinstance(scope, NullUnitProgressScope)
        scope.set(5)
        scope.finish()
        report_stage(STAGE_EXTRACTING)  # must not raise

    def test_the_reporter_that_inference_finds_is_the_running_job(self):
        job = _job()
        JobReporter(str(job.id), min_interval_seconds=0.0)
        with unit_scope(total=3, label=UNIT_TILE, stage=STAGE_INFERENCE) as scope:
            scope.set(3)
        report_stage(STAGE_EXTRACTING)
        job.refresh_from_db()
        assert job.progress_units_done == 3
        assert job.progress_stage == STAGE_EXTRACTING


class StageVocabularyTests(TestCase):
    def test_inference_only_writes_stages_the_jobs_app_declares(self):
        from quantem.inference import segmenter as inference_segmenter

        used = {
            inference_segmenter.STAGE_LOADING_MODEL,
            inference_segmenter.STAGE_INFERENCE,
            inference_segmenter.STAGE_EXTRACTING,
        }
        assert used <= set(PROGRESS_STAGES)

    def test_no_stage_name_could_be_read_as_a_command(self):
        """I-12, at the source: these keys reach the UI."""
        for stage in PROGRESS_STAGES:
            assert stage.replace("_", "").isalpha()
            assert " " not in stage

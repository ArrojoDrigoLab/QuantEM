"""The tile counter has to survive a slow disk.

The failure this file is about was measured, not imagined: on this database a
single progress UPDATE took **1 129.8 ms** (median 0.24 ms over 400 samples).
:class:`quantem.jobs.reporter.UnitProgressScope` spends its write budget
cumulatively -- ``write_seconds <= 0.01 * elapsed`` -- so one sample like that
stops it writing for the next ~113 s, and on a 50 s run the tile count on
screen freezes until the forced final write. A frozen counter and a stalled run
look identical.

:class:`~quantem.seg_core.db.tile_progress.TileProgressWriter` is what the
counter a user reads comes from, and these tests pin the three properties that
make it robust: a slow write costs one sample, a failed write costs one sample,
and the denominator is on the row before the model has even loaded.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase

from quantem.jobs.models import Job
from quantem.jobs.reporter import JobReporter, UnitProgressScope
from quantem.seg_core.db import tile_progress
from quantem.seg_core.db.inference import run_inference_for_segmentation
from quantem.seg_core.db.tile_progress import TileProgressWriter
from quantem.seg_core.types import InferenceResult


def _running_job() -> Job:
    job = Job.objects.create(
        type="run_segmentation_full_task",
        status="RUNNING",
        payload_json={"segmentation_id": "seg-1"},
        queue_name="p4_full",
    )
    return job


class _SlowClock:
    """A monotonic clock the test drives by hand."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TileProgressWriterTests(TestCase):
    def setUp(self):
        self.job = _running_job()
        self.reporter = JobReporter(str(self.job.id))
        self.addCleanup(self.reporter.deactivate)

    def _units(self) -> tuple[int | None, int | None, str]:
        row = Job.objects.get(id=self.job.id)
        return row.progress_units_done, row.progress_units_total, row.progress_unit_label

    def test_announce_puts_the_denominator_on_the_row_before_any_tile(self):
        TileProgressWriter().announce(56)
        self.assertEqual(self._units(), (0, 56, "tile"))

    def test_a_slow_write_costs_one_sample_and_not_the_rest_of_the_run(self):
        """The regression that motivated this class.

        The same 1.1 s outlier is injected into both writers. The scope stops
        for the rest of the run; the writer misses the tiles inside its one
        second interval and then carries on.
        """
        clock = _SlowClock()
        slow_once = {"pending": True}
        real_update = Job.objects.filter

        def slow_filter(*args, **kwargs):
            queryset = real_update(*args, **kwargs)
            original = queryset.update

            def update(**values):
                if slow_once["pending"]:
                    slow_once["pending"] = False
                    clock.advance(1.13)
                return original(**values)

            queryset.update = update
            return queryset

        with (
            patch.object(tile_progress.time, "perf_counter", clock),
            patch.object(Job.objects, "filter", side_effect=slow_filter),
        ):
            writer = TileProgressWriter()
            for done in range(1, 21):
                clock.advance(0.7)  # one CPU tile
                writer.report(done, 20)

        self.assertEqual(self._units()[0], 20)
        self.assertGreaterEqual(
            writer.stats["writes"],
            10,
            "one slow write switched the counter off for the rest of the run",
        )

    def test_the_scope_this_replaces_does_freeze_after_the_same_outlier(self):
        """The contrast, so "robust" is a measurement and not an adjective."""
        clock = _SlowClock()
        with patch("quantem.jobs.reporter.time.perf_counter", clock):
            scope = UnitProgressScope(str(self.job.id), total=20, label="tile")
            clock.advance(1.13)  # the outlier, charged to the scope's budget
            scope._write_seconds = 1.13
            frozen_at = None
            for done in range(1, 20):
                clock.advance(0.7)
                scope.set(done)
                if frozen_at is None and Job.objects.get(
                    id=self.job.id
                ).progress_units_done != done:
                    frozen_at = done
        self.assertIsNotNone(
            frozen_at,
            "expected the cumulative budget to stop the scope writing",
        )

    def test_a_failed_write_is_one_missed_sample_not_a_dead_counter(self):
        writer = TileProgressWriter(min_interval_seconds=0.0)
        writer.report(1, 10)
        with patch.object(
            Job.objects, "filter", side_effect=RuntimeError("database is locked")
        ):
            writer.report(2, 10)
        writer.report(3, 10)
        self.assertEqual(self._units()[0], 3)
        self.assertEqual(writer.stats["failures"], 1)

    def test_a_write_never_lands_on_a_job_that_has_already_concluded(self):
        writer = TileProgressWriter(min_interval_seconds=0.0)
        writer.report(5, 10)
        Job.objects.filter(id=self.job.id).update(status="CANCELLED")
        writer.report(9, 10)
        self.assertEqual(self._units()[0], 5)

    def test_with_no_job_on_this_thread_everything_is_a_no_op(self):
        self.reporter.deactivate()
        writer = TileProgressWriter()
        writer.announce(56)
        writer.report(3, 56)
        self.assertFalse(writer.active)
        self.assertEqual(self._units(), (None, None, ""))


class _SlowLoadingSegmenter:
    """Reads the job row from inside ``load_models``, as a user reads a screen."""

    name = "mito"
    generated_flag = "mito_generated"
    prob_map_prefix = "mito"
    persist_probability_maps = False

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.units_during_load: tuple[int | None, int | None] = (None, None)

    def load_models(self) -> None:
        row = Job.objects.get(id=self.job_id)
        self.units_during_load = (row.progress_units_done, row.progress_units_total)

    def get_dl_model_names(self) -> list[str]:
        return ["DINO"]

    def estimate_dl_tile_count(self, image_shape) -> int:
        _ = image_shape
        return 56

    def predict(self, image, cached_prob_maps=None, on_progress=None, **kwargs):
        _ = (cached_prob_maps, kwargs)
        if on_progress is not None:
            on_progress("DINO", 1.0)
        prob = np.full((16, 16), 0.5, dtype=np.float32)
        return InferenceResult(prob_maps={"DINO": prob}, prob=prob)

    def get_probability_map_metadata(self, model_name: str) -> dict[str, object]:
        return {"model_name": model_name}


class DenominatorBeforeTheModelLoadsTests(TestCase):
    def test_the_row_says_0_of_56_while_the_model_is_still_loading(self):
        """The 4-20 s that used to read as a frozen 5 %.

        The tiling plan depends only on the region shape and the pack's
        canonical scale, so it is knowable before a single weight is read. With
        it on the row, that window can say "loading the model - 0 of 56 tiles".
        """
        job = _running_job()
        reporter = JobReporter(str(job.id))
        self.addCleanup(reporter.deactivate)
        segmenter = _SlowLoadingSegmenter(str(job.id))

        with (
            patch(
                "quantem.seg_core.db.inference.get_asset_openable",
                return_value=SimpleNamespace(height=2508, width=2892),
            ),
            patch(
                "quantem.seg_core.db.inference.load_image_array",
                return_value=(np.zeros((16, 16), dtype=np.uint8), 0.0),
            ),
        ):
            run_inference_for_segmentation(
                segmenter,
                SimpleNamespace(id="seg-1", asset_id="asset-1", asset=object()),
                MagicMock(),
            )

        self.assertEqual(segmenter.units_during_load, (0, 56))

"""The GPU fallback copy has a reader.

``DinoOrganelleSegmenter.device_notices`` was written with a docstring saying
*"Read by the run task after inference, exactly as ``encoder_tier`` is"*, and
nothing read it -- not in ``src/quantem``, not in ``frontend/src``. So a run
that could not use the graphics card, or ran short of memory and shrank its
batch, or moved to the processor half way through, finished twenty minutes
after an estimate of one and told the user nothing. That silence is exactly
what owner decision D3 was written to prevent.

The reader lives in :func:`quantem.seg_core.db.inference.report_device_notices`
rather than in the run task so that every caller of
:func:`~quantem.seg_core.db.inference.run_inference_for_segmentation` gets it --
a whole-image run, a patch run, a re-run -- and a new caller cannot forget.

Without that function these tests fail on the count of job-log rows: nothing
writes one, so ``JobLog.objects.count()`` is 0 where they expect the sentences.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase

from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_FULL
from quantem.jobs.models import Job, JobLog
from quantem.jobs.reporter import JobReporter
from quantem.seg_core.db.inference import (
    report_device_notices,
    run_inference_for_segmentation,
)
from quantem.seg_core.types import InferenceResult

_MOVED = (
    "This run moved to the processor part-way through: the graphics card ran "
    "out of memory. The result is complete; it took longer than it would have "
    "on the graphics card."
)
_SMALLER = (
    "There was not enough memory on the graphics card to run this model at "
    "full speed, so it ran in smaller pieces."
)


class _Segmenter:
    def __init__(self, notices):
        self.device_notices = list(notices)


class _NoSuchProperty:
    """A segmenter from before device notices existed. Must not blow up."""


class DeviceNoticesReachTheJobLogTests(TestCase):
    def setUp(self):
        self.job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload_json={},
        )
        self.reporter = JobReporter(str(self.job.id), min_interval_seconds=0.0)
        self.addCleanup(self.reporter.deactivate)

    def _logged(self) -> list[tuple[str, str]]:
        return list(
            JobLog.objects.filter(job=self.job)
            .order_by("timestamp", "id")
            .values_list("level", "message")
        )

    def test_a_fallback_is_written_where_the_user_can_read_it_afterwards(self):
        reported = report_device_notices(_Segmenter([_MOVED]))

        self.assertEqual(reported, [_MOVED])
        self.assertEqual(self._logged(), [("warning", _MOVED)])

    def test_every_sentence_is_kept_and_none_is_merged_into_another(self):
        report_device_notices(_Segmenter([_SMALLER, _MOVED]))

        self.assertEqual(self._logged(), [("warning", _SMALLER), ("warning", _MOVED)])

    def test_the_ordinary_run_says_nothing_at_all(self):
        """Empty is the normal case, and it must stay silent.

        A line per run saying "this ran where you expected" would train a
        reader to skip the one line that matters.
        """
        self.assertEqual(report_device_notices(_Segmenter([])), [])
        self.assertEqual(self._logged(), [])

    def test_blank_sentences_are_not_written(self):
        self.assertEqual(report_device_notices(_Segmenter(["", "   "])), [])
        self.assertEqual(self._logged(), [])

    def test_a_segmenter_that_reports_no_notices_at_all_is_fine(self):
        self.assertEqual(report_device_notices(_NoSuchProperty()), [])
        self.assertEqual(self._logged(), [])


class _FallingBackSegmenter:
    """A pass that ended up somewhere other than where it started."""

    name = "mito"
    generated_flag = "mito_generated"
    prob_map_prefix = "mito"
    persist_probability_maps = False
    device_notices: list[str] = []

    def load_models(self) -> None:
        self.device_notices = [_MOVED]

    def get_dl_model_names(self) -> list[str]:
        return []

    def predict(self, image, cached_prob_maps=None, **kwargs):
        _ = (image, cached_prob_maps, kwargs)
        prob = np.zeros((16, 16), dtype=np.float32)
        return InferenceResult(prob_maps={}, prob=prob)


class TheRunItselfReportsThemTests(TestCase):
    """Not only the helper: the pass that runs the model calls it.

    The reader could have been correct and never reached. This drives
    :func:`~quantem.seg_core.db.inference.run_inference_for_segmentation`, which
    is what every run goes through, and looks for the sentence on the job.
    """

    @patch("quantem.seg_core.db.inference.load_image_array")
    @patch("quantem.seg_core.db.inference.get_asset_openable")
    def test_a_run_that_fell_back_says_so_on_its_job(
        self,
        mock_get_asset_openable,
        mock_load_image_array,
    ):
        job = Job.objects.create(type=JOB_TYPE_RUN_SEGMENTATION_FULL, payload_json={})
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)
        self.addCleanup(reporter.deactivate)
        mock_get_asset_openable.return_value = SimpleNamespace(height=16, width=16)
        mock_load_image_array.return_value = (
            np.zeros((16, 16), dtype=np.uint8),
            0.0,
        )

        run_inference_for_segmentation(
            _FallingBackSegmenter(),
            SimpleNamespace(id="seg-1", asset_id="asset-1", asset=object()),
            MagicMock(),
        )

        self.assertEqual(
            list(JobLog.objects.filter(job=job).values_list("level", "message")),
            [("warning", _MOVED)],
        )


class NoJobBehindTheRunTests(TestCase):
    """``seg_core`` runs from the CLI and from tests with no job at all."""

    def test_a_run_outside_the_queue_still_returns_its_notices(self):
        reported = report_device_notices(_Segmenter([_MOVED]))

        self.assertEqual(reported, [_MOVED])
        self.assertEqual(JobLog.objects.count(), 0)

    def test_a_reporter_that_cannot_write_does_not_fail_the_run(self):
        """The numbers are already computed. A log line must never lose them."""

        class _Broken:
            def log(self, level, message):
                raise RuntimeError("the job row is gone")

        import quantem.seg_core.db.inference as module

        original = module._active_job_reporter
        module._active_job_reporter = lambda: _Broken()
        self.addCleanup(setattr, module, "_active_job_reporter", original)

        self.assertEqual(report_device_notices(_Segmenter([_MOVED])), [_MOVED])

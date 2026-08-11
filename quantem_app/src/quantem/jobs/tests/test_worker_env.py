"""Running a job in-process must not permanently claim the interpreter.

Round-14 validation traced a deterministically red CI leg to this. The worker
entry point marks its process with ``QUANTEM_JOB_WORKER=1`` before
``django.setup()`` so that a spawned worker does not start a second scheduler.
The marker was set and never cleared, so a test that exercises the real worker
path -- which is the only honest way to test it -- left the flag set for every
test that ran afterwards in the same interpreter. File logging is suppressed in
workers, so a later test watched ``quantem serve`` announce a log file that
could never be written, and blamed a slow machine.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from django.test import TestCase

from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_ROI
from quantem.jobs.models import Job
from quantem.jobs.runner import WORKER_PROCESS_ENV_VAR, run_job_in_subprocess


class WorkerMarkerDoesNotLeakTests(TestCase):
    def _run_one_job(self) -> None:
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="RUNNING",
            attempts=1,
            max_attempts=1,
            payload_json={},
            message="running",
        )
        with patch("quantem.jobs.registry.get_handler") as get_handler:
            get_handler.return_value = lambda payload, reporter, cancel: {"ok": True}
            run_job_in_subprocess(str(job.id))

    def test_an_unset_marker_is_unset_again_afterwards(self):
        os.environ.pop(WORKER_PROCESS_ENV_VAR, None)

        self._run_one_job()

        self.assertNotIn(WORKER_PROCESS_ENV_VAR, os.environ)

    def test_a_failing_job_also_puts_the_marker_back(self):
        os.environ.pop(WORKER_PROCESS_ENV_VAR, None)
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="RUNNING",
            attempts=1,
            max_attempts=1,
            payload_json={},
            message="running",
        )

        def explode(payload, reporter, cancel):
            raise RuntimeError("the pack is broken today")

        with patch("quantem.jobs.registry.get_handler") as get_handler:
            get_handler.return_value = explode
            run_job_in_subprocess(str(job.id))

        self.assertNotIn(WORKER_PROCESS_ENV_VAR, os.environ)

    def test_a_persistent_worker_keeps_its_own_claim(self):
        """The persistent worker marks itself before its loop and runs many
        jobs through this entry point; restoring the *previous* value must
        leave that claim standing rather than unmarking it after job one."""
        os.environ[WORKER_PROCESS_ENV_VAR] = "1"
        try:
            self._run_one_job()

            self.assertEqual(os.environ.get(WORKER_PROCESS_ENV_VAR), "1")
        finally:
            os.environ.pop(WORKER_PROCESS_ENV_VAR, None)


class FileLoggingPredicateTests(TestCase):
    """``quantem serve`` announces a log path; it must not lie about it."""

    def test_a_worker_is_not_a_file_logger(self):
        from quantem.core.config import file_logging_enabled

        with patch.dict(os.environ, {"QUANTEM_LOG_TO_FILE": "1", "QUANTEM_JOB_WORKER": "1"}):
            self.assertFalse(file_logging_enabled())

    def test_the_server_with_the_flag_on_is(self):
        from quantem.core.config import file_logging_enabled

        env = dict(os.environ)
        env["QUANTEM_LOG_TO_FILE"] = "1"
        env.pop("QUANTEM_JOB_WORKER", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(file_logging_enabled())

    def test_the_flag_off_means_no_promise(self):
        from quantem.core.config import file_logging_enabled

        with patch.dict(os.environ, {"QUANTEM_LOG_TO_FILE": "0"}):
            self.assertFalse(file_logging_enabled())

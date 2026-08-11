"""The scheduler is what actually runs the abandoned-upload sweep.

Finding F3: killing the server mid-import left a 2 074 034 677 B staged body in
``data/tmp/uploads/`` that survived a restart, and nothing in the tree ever
looked at that directory again. ``upload_staging.sweep_abandoned_uploads`` is
only a function; these tests pin the two places it is called from -- once when
the scheduler's database first comes up, which is when the previous session's
wreckage is still lying there, and then on a slow timer for the rest of the
session.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from django.test import TestCase

from quantem.assets.upload_staging import (
    DEFAULT_ABANDONED_UPLOAD_MAX_AGE_SECONDS,
    STAGING_PREFIX,
)
from quantem.core.config import UPLOADS_DIR
from quantem.jobs import scheduler as scheduler_module
from quantem.jobs.scheduler import UPLOAD_SWEEP_INTERVAL_SECONDS, JobScheduler


class _StopTheLoop(BaseException):
    """Not an ``Exception``: ``run_forever`` swallows those on purpose."""


class UploadSweepIsScheduledTests(TestCase):
    def setUp(self):
        self.scheduler = JobScheduler(poll_interval_seconds=0.0)
        self.swept: list[str] = []
        self.scheduler.sweep_abandoned_uploads = lambda: self.swept.append("sweep")

    def test_the_first_thing_a_ready_database_gets_is_a_sweep(self):
        self.scheduler._recover_orphaned_jobs = lambda **kwargs: None

        def stop() -> None:
            raise _StopTheLoop

        self.scheduler.tick = stop
        original = scheduler_module._wait_for_database
        scheduler_module._wait_for_database = lambda: True
        try:
            with self.assertRaises(_StopTheLoop):
                self.scheduler.run_forever()
        finally:
            scheduler_module._wait_for_database = original

        self.assertEqual(
            self.swept,
            ["sweep"],
            "a restart is the moment the last session's abandoned bytes are "
            "still on disk; nothing else looks for them",
        )

    def test_a_tick_does_not_sweep_until_the_interval_has_passed(self):
        self.scheduler.runner.poll = lambda: None
        self.scheduler.dispatch_ready = lambda: None

        self.scheduler.tick()

        self.assertEqual(self.swept, [], "the sweep must not run on every tick")

    def test_a_tick_sweeps_once_the_interval_has_passed(self):
        self.scheduler.runner.poll = lambda: None
        self.scheduler.dispatch_ready = lambda: None
        self.scheduler._last_upload_sweep_monotonic = (
            time.monotonic() - UPLOAD_SWEEP_INTERVAL_SECONDS - 1.0
        )

        self.scheduler.tick()
        self.scheduler.tick()

        self.assertEqual(self.swept, ["sweep"], "the timer must reset after it fires")


class UploadSweepIsWiredToTheRealSweeperTests(TestCase):
    """Not a mock: the scheduler's method has to delete real bytes."""

    def setUp(self):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.scheduler = JobScheduler(poll_interval_seconds=0.0)

    def test_it_removes_an_abandoned_staging_file(self):
        path = UPLOADS_DIR / f"{STAGING_PREFIX}{uuid.uuid4().hex}.tif"
        path.write_bytes(b"z" * 2048)
        self.addCleanup(path.unlink, True)
        old = time.time() - DEFAULT_ABANDONED_UPLOAD_MAX_AGE_SECONDS - 60
        os.utime(path, (old, old))

        self.scheduler.sweep_abandoned_uploads()

        self.assertFalse(path.exists(), "the scheduler did not reach the real sweeper")

    def test_a_failing_sweep_does_not_take_the_tick_down(self):
        # Housekeeping must never cost the queue its dispatch.
        def explode() -> None:
            raise OSError("the volume went away")

        from quantem.assets import upload_staging

        real = upload_staging.sweep_abandoned_uploads
        upload_staging.sweep_abandoned_uploads = explode
        try:
            self.scheduler.sweep_abandoned_uploads()  # must not raise
        finally:
            upload_staging.sweep_abandoned_uploads = real

    def test_the_sweep_leaves_a_file_that_is_not_old_enough(self):
        path = Path(UPLOADS_DIR / f"{STAGING_PREFIX}{uuid.uuid4().hex}.tif")
        path.write_bytes(b"arriving now")
        self.addCleanup(path.unlink, True)

        self.scheduler.sweep_abandoned_uploads()

        self.assertTrue(path.exists())

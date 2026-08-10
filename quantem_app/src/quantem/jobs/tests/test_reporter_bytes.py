"""Byte-level progress lands on the job row, for the Models screen.

``GET /api/models/`` reports an in-flight install's ``active_install`` with
``progress_current_bytes`` / ``progress_total_bytes`` read straight off the
job row; the download handler feeds them through ``JobReporter.update``. These
pin the write path so the contract cannot silently regress into parsing the
"435 of 1243 MB" message string back apart.
"""

from __future__ import annotations

from django.test import TestCase

from quantem.jobs.constants import JOB_TYPE_INSTALL_MODEL_PACK, JOB_TYPE_RUN_ANALYSIS
from quantem.jobs.models import Job
from quantem.jobs.reporter import JobReporter


class ReporterByteProgressTests(TestCase):
    def test_byte_counts_land_on_the_job_row(self):
        job = Job.enqueue(job_type=JOB_TYPE_INSTALL_MODEL_PACK, payload={})
        reporter = JobReporter(str(job.id), min_interval_seconds=0.0)

        reporter.update(
            progress=10.0,
            message="downloading omniem:mito: 5 of 10 MB",
            current_bytes=5_000_000,
            total_bytes=10_000_000,
        )

        job.refresh_from_db()
        self.assertEqual(job.progress_current_bytes, 5_000_000)
        self.assertEqual(job.progress_total_bytes, 10_000_000)

    def test_jobs_that_never_report_bytes_stay_null(self):
        """Null means "does not report bytes" -- not 0 of anything."""
        job = Job.enqueue(job_type=JOB_TYPE_RUN_ANALYSIS, payload={})

        JobReporter(str(job.id), min_interval_seconds=0.0).update(progress=50.0)

        job.refresh_from_db()
        self.assertIsNone(job.progress_current_bytes)
        self.assertIsNone(job.progress_total_bytes)

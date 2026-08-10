import logging
import time

from django.utils import timezone

from quantem.jobs.models import Job, JobLog

logger = logging.getLogger(__name__)


class JobCancelledError(Exception):
    pass


class CancelToken:
    def __init__(self, job_id: str):
        self.job_id = job_id

    def check_cancelled(self) -> None:
        if Job.objects.filter(id=self.job_id, cancel_requested=True).exists():
            raise JobCancelledError("Job cancellation requested.")


class JobReporter:
    def __init__(self, job_id: str, min_interval_seconds: float = 0.5):
        self.job_id = job_id
        self.min_interval_seconds = min_interval_seconds
        self._last_update = 0.0
        self._last_throttle_log = 0.0

    def update(
        self,
        progress: float | None = None,
        message: str | None = None,
        *,
        current_bytes: int | None = None,
        total_bytes: int | None = None,
    ) -> None:
        """Write progress onto the job row, rate-limited.

        ``current_bytes``/``total_bytes`` are for jobs whose work is moving
        bytes (model pack downloads): they land in the row's
        ``progress_current_bytes``/``progress_total_bytes`` so the Models
        screen can show a real byte count for an in-flight install instead of
        parsing it back out of ``message``. Jobs that never pass them leave the
        columns null, which reads as "does not report bytes".
        """
        now = time.time()
        if progress is not None:
            progress = max(0.0, min(float(progress), 100.0))

        should_update = (now - self._last_update) >= self.min_interval_seconds
        if progress is not None and progress >= 100.0:
            should_update = True

        if not should_update:
            if (
                progress is not None
                or message is not None
            ) and ((now - self._last_throttle_log) >= 10.0):
                self._last_throttle_log = now
            return

        updates = {}
        if progress is not None:
            updates["progress"] = progress
        if message is not None:
            updates["message"] = message
        if current_bytes is not None:
            updates["progress_current_bytes"] = max(0, int(current_bytes))
        if total_bytes is not None:
            updates["progress_total_bytes"] = max(0, int(total_bytes))
        if updates:
            updates["updated_at"] = timezone.now()
            Job.objects.filter(id=self.job_id).update(**updates)
            self._last_update = now

    def log(self, level: str, message: str) -> None:
        JobLog.objects.create(job_id=self.job_id, level=level, message=message)

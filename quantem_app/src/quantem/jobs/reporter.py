"""What a running job tells the rest of the app about itself.

Three kinds of progress, kept apart on purpose
----------------------------------------------
1. ``progress`` -- one percentage for the whole job. Generic, coarse, and now
   **monotone**: :meth:`JobReporter.update` keeps a high-water mark for the
   attempt, so the bar can no longer run 0 -> 5 -> 71 -> 55 -> 100 (invariant
   I-3).
2. ``progress_units_done`` / ``progress_units_total`` -- countable work. For a
   segmentation run that is **tiles**, taken from the sliding-window plan the
   engine lays out before the first forward pass, so "531 of 858 tiles" is a
   fact rather than a percentage read backwards.
3. ``progress_current_bytes`` / ``progress_total_bytes`` -- bytes moving over
   the network for a model download.

(2) and (3) are separate columns with separate stages because they are separate
facts about the machine. Presenting "downloading 1.2 GB" as though it were
"segmenting tiles" would be a lie about what is happening, and the owner asked
for the two as distinct indicators.

How tile counts reach here from the inference loop
--------------------------------------------------
The tiling loop lives four call frames below the job handler and is handed no
reporter: ``handler -> organelle_tasks -> seg_core.db.inference ->
BaseSegmenter.predict -> engine.predict_region``. Rather than thread a reporter
through every one of those signatures, a job's reporter registers itself as the
**active reporter for its thread** when it is constructed, and
:func:`unit_scope` looks it up. One job runs per worker thread at a time (see
``quantem.jobs.runner._run_job_in_subprocess``, which constructs exactly one
``JobReporter`` per attempt), so "the reporter for this thread" is unambiguous.
Inference driven outside the queue -- a CLI call, a test -- finds no active
reporter and reports to a no-op scope, which is why
:func:`quantem.inference.segmenter` needs no branch for it.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

from django.utils import timezone

from quantem.jobs.models import OPEN_JOB_STATUSES, Job, JobLog

logger = logging.getLogger(__name__)

#: Where the work currently being counted sits inside a bigger job's count, as
#: ``(base, grand_total)``, or None.
#:
#: **Why this is thread state and not an argument.** A run over four organelles
#: is one job row with one tile count, and *two* things write that count from
#: inside the tiling loop: :class:`UnitProgressScope` (opened four frames down
#: in ``quantem.inference.segmenter``) and
#: ``quantem.seg_core.db.tile_progress.TileProgressWriter``. Both are handed
#: this organelle's own plan and neither is in a position to know about the
#: three beside it. Threading a window through both call chains would mean
#: changing every signature between the job handler and the tiling loop -- the
#: exact plumbing the active-reporter lookup above exists to avoid.
#:
#: Measured consequence of not having it: a real two-organelle run ended with
#: ``progress_units_total = 1`` -- the last organelle's plan -- over a wave of
#: 5 tiles, because the second scope overwrote the first's denominator.
_unit_window = threading.local()


@contextlib.contextmanager
def unit_window(base: int, total: int) -> Iterator[None]:
    """While this is open, unit writes on this thread are offset into a wave.

    ``base`` is how many units the work before this leg actually completed, and
    ``total`` is the whole wave's plan. Restores whatever was in force before,
    so a nested or sequential use cannot leak into the next leg.
    """
    previous = getattr(_unit_window, "value", None)
    _unit_window.value = (max(int(base), 0), max(int(total), 0))
    try:
        yield
    finally:
        _unit_window.value = previous


def active_unit_window() -> tuple[int, int] | None:
    """The wave offset in force on this thread, or None."""
    return getattr(_unit_window, "value", None)


def _stored_detail(job_id: str) -> dict[str, Any]:
    """This job's ``progress_detail_json`` as it stands. ``{}`` on any trouble."""
    try:
        return dict(
            Job.objects.filter(id=job_id).values_list("progress_detail_json", flat=True).first()
            or {}
        )
    except Exception:  # pragma: no cover -- a progress read must never fail a run
        return {}


def apply_unit_window(done: int, total: int) -> tuple[int, int]:
    """Map one leg's ``(done, total)`` onto the wave's, or pass them through.

    The wave's total wins unless this leg alone would exceed it -- a plan
    corrected upwards mid-flight must widen the denominator rather than push the
    bar past 100 %.
    """
    window = active_unit_window()
    if window is None:
        return done, total
    base, grand = window
    return base + done, max(grand, base + total)


#: Shortest gap between two unit-progress writes for one scope. The first write
#: (the denominator, at construction) and the last one are never withheld.
#:
#: **This replaces a cumulative write-time budget** -- ``write_seconds <= 0.01 *
#: elapsed`` -- and the difference is the difference between a counter that
#: degrades and one that dies. A progress UPDATE on this database measures
#: ~0.24 ms, but one was measured at **1 129.8 ms** (a WAL checkpoint, an
#: antivirus scan, a disk waking up). Under the budget rule that single sample
#: bought the next ~113 seconds of silence, so on a 50 s run the tile count
#: froze at whatever it last said until the forced final write -- and on screen
#: a frozen counter is indistinguishable from a stalled run.
#:
#: A wall-clock floor cannot do that. A slow write delays the next sample by
#: the length of the slow write and nothing more. It also bounds the cost the
#: budget existed to bound, and bounds it better: at most one UPDATE per second
#: (~0.02 % of wall clock) whether a tile takes 0.7 s on CPU or 0.05 s on a
#: CUDA run at 20 tiles/s.
UNIT_WRITE_MIN_INTERVAL_SECONDS = 1.0


class JobCancelledError(Exception):
    pass


class CancelToken:
    def __init__(self, job_id: str):
        self.job_id = job_id

    def check_cancelled(self) -> None:
        if Job.objects.filter(id=self.job_id, cancel_requested=True).exists():
            raise JobCancelledError("Job cancellation requested.")


_ACTIVE = threading.local()


def active_reporter() -> JobReporter | None:
    """The reporter for the job running on this thread, if any."""
    return getattr(_ACTIVE, "reporter", None)


class JobReporter:
    def __init__(self, job_id: str, min_interval_seconds: float = 0.5):
        self.job_id = job_id
        self.min_interval_seconds = min_interval_seconds
        self._last_update = 0.0
        self._last_throttle_log = 0.0
        # High-water mark for this attempt. A retry constructs a new reporter
        # and legitimately starts again from zero; within one attempt the bar
        # only ever moves forward.
        self._max_progress = 0.0
        self.activate()

    # --- thread registration ---

    def activate(self) -> JobReporter:
        """Make this the reporter :func:`unit_scope` finds on this thread."""
        _ACTIVE.reporter = self
        return self

    def deactivate(self) -> None:
        """Forget this thread's reporter. Used by tests and by nested drivers."""
        if getattr(_ACTIVE, "reporter", None) is self:
            _ACTIVE.reporter = None

    # --- writes ---

    def update(
        self,
        progress: float | None = None,
        message: str | None = None,
        *,
        current_bytes: int | None = None,
        total_bytes: int | None = None,
        stage: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Write progress onto the job row, rate-limited.

        ``progress`` is clamped to the attempt's high-water mark: a caller that
        reports a smaller number than it did a moment ago is describing a new
        phase of the same job, not lost work, and a bar that jumps backwards is
        read as a bug in the run rather than in the bar.

        ``current_bytes``/``total_bytes`` are for jobs whose work is moving
        bytes (model pack downloads): they land in the row's
        ``progress_current_bytes``/``progress_total_bytes`` so the Models
        screen can show a real byte count for an in-flight install instead of
        parsing it back out of ``message``. Jobs that never pass them leave the
        columns null, which reads as "does not report bytes".

        ``stage`` is one of :data:`quantem.jobs.models.PROGRESS_STAGES` and
        ``detail`` is machine-readable extras for it; neither is ever rendered
        verbatim.
        """
        now = time.time()
        if progress is not None:
            progress = max(0.0, min(float(progress), 100.0))
            # Raise the mark even when this call is throttled away, or the
            # throttle would let the next, smaller, report through.
            progress = self._max_progress = max(self._max_progress, progress)

        should_update = (now - self._last_update) >= self.min_interval_seconds
        if progress is not None and progress >= 100.0:
            should_update = True
        if stage is not None:
            # A phase change is the one thing a user is waiting to see; it is
            # never worth withholding for a throttle window.
            should_update = True

        if not should_update:
            if (progress is not None or message is not None) and (
                (now - self._last_throttle_log) >= 10.0
            ):
                self._last_throttle_log = now
            return

        updates: dict[str, Any] = {}
        if progress is not None:
            updates["progress"] = progress
        if message is not None:
            updates["message"] = message
        if current_bytes is not None:
            updates["progress_current_bytes"] = max(0, int(current_bytes))
        if total_bytes is not None:
            updates["progress_total_bytes"] = max(0, int(total_bytes))
        if stage is not None:
            updates["progress_stage"] = str(stage)
        if detail is not None:
            updates["progress_detail_json"] = dict(detail)
        if updates:
            updates["updated_at"] = timezone.now()
            Job.objects.filter(id=self.job_id).update(**updates)
            self._last_update = now

    def stage(self, stage: str, *, detail: dict[str, Any] | None = None) -> None:
        """Record the phase this job is in, with no claim about how far along."""
        self.update(stage=stage, detail=detail)

    def unit_scope(
        self,
        *,
        total: int,
        label: str,
        stage: str | None = None,
        detail: dict[str, Any] | None = None,
        min_interval_seconds: float | None = None,
    ) -> UnitProgressScope:
        return UnitProgressScope(
            self.job_id,
            total=total,
            label=label,
            stage=stage,
            detail=detail,
            min_interval_seconds=min_interval_seconds,
        )

    def log(self, level: str, message: str) -> None:
        JobLog.objects.create(job_id=self.job_id, level=level, message=message)


class UnitProgressScope:
    """One countable phase of a job -- 858 tiles, say -- written as it runs.

    Opening the scope writes ``0 of total`` immediately, which is what removes
    the dead air before the first tile: the denominator is on screen before any
    work has happened. :meth:`set` is safe to call once per unit; it never
    writes a smaller ``done`` than it already wrote, and it always writes the
    last one.
    """

    def __init__(
        self,
        job_id: str,
        *,
        total: int,
        label: str,
        stage: str | None = None,
        detail: dict[str, Any] | None = None,
        min_interval_seconds: float | None = None,
    ) -> None:
        self.job_id = job_id
        self.label = str(label)
        self.stage = stage
        self.total = max(int(total), 0)
        self.done = 0
        self._started = time.perf_counter()
        # None means "use the module default, whatever it is when I ask", so a
        # test can lift the floor for a loop that runs in microseconds.
        self._min_interval = None if min_interval_seconds is None else float(min_interval_seconds)
        self._last_write = 0.0
        self._write_seconds = 0.0
        self._writes = 0
        self._skipped = 0
        self._detail = dict(detail or {})
        self._closed = False
        self._write(force=True)

    # --- the two calls the inference loop makes ---

    def set(self, done: int, *, total: int | None = None) -> None:
        """Report ``done`` units complete, optionally correcting the total.

        The total is corrected once in practice: the pre-run estimate is
        replaced by the tiling plan's exact count the moment the plan exists.
        """
        if self._closed:
            return
        if total is not None:
            self.total = max(int(total), 0)
        done = max(0, int(done))
        if self.total:
            done = min(done, self.total)
        if done < self.done:
            # Not a hard error: a caller reporting a stale count is a bug, but
            # refusing the run over it would be worse than refusing the number.
            logger.debug(
                "job %s: unit progress went backwards (%s -> %s %ss); holding",
                self.job_id,
                self.done,
                done,
                self.label,
            )
            return
        self.done = done
        self._write(force=self.total > 0 and done >= self.total)

    def advance(self, count: int = 1) -> None:
        self.set(self.done + max(int(count), 0))

    def finish(self) -> None:
        """Close the scope, writing the final count once."""
        if self._closed:
            return
        self._write(force=True)
        self._closed = True

    # --- cost accounting ---

    @property
    def overhead_fraction(self) -> float:
        """Share of this scope's wall clock spent writing progress rows."""
        elapsed = time.perf_counter() - self._started
        return self._write_seconds / elapsed if elapsed > 0 else 0.0

    @property
    def write_stats(self) -> dict[str, float]:
        return {
            "writes": self._writes,
            "skipped": self._skipped,
            "write_seconds": self._write_seconds,
            "overhead_fraction": self.overhead_fraction,
        }

    # --- internals ---

    def _may_write(self) -> bool:
        """Whether enough wall clock has passed since the last write."""
        interval = (
            UNIT_WRITE_MIN_INTERVAL_SECONDS if self._min_interval is None else self._min_interval
        )
        return (time.perf_counter() - self._last_write) >= interval

    def _write(self, *, force: bool = False) -> None:
        if not force and not self._may_write():
            self._skipped += 1
            return
        detail = dict(self._detail)
        eta = self._eta_seconds()
        if eta is not None:
            detail["eta_seconds"] = round(eta, 1)
        # Offset into the wave when this leg is one of several sharing a row.
        # Without a window in force these are the scope's own numbers, unchanged.
        windowed_done, windowed_total = apply_unit_window(self.done, self.total)
        if active_unit_window() is not None:
            # A shared row carries somebody else's keys -- the per-organelle
            # list the run panel draws its lines from -- and this column is
            # normally written whole. Read-modify-write only in the case where
            # there is something to preserve: at most one extra SELECT per
            # second, against per-organelle lines vanishing and reappearing for
            # the length of every run.
            detail = {**_stored_detail(self.job_id), **detail}
        updates: dict[str, Any] = {
            "progress_units_done": windowed_done,
            "progress_units_total": windowed_total,
            "progress_unit_label": self.label,
            "progress_detail_json": detail,
            "updated_at": timezone.now(),
        }
        if self.stage:
            updates["progress_stage"] = self.stage
        started = time.perf_counter()
        # Only onto a job that still has work ahead of it. A worker thread keeps
        # its reporter after the job concludes, so without this a stray tile
        # report could overwrite a finished run's final count -- and a number
        # written onto a concluded job is a lie, where silence is only silence.
        Job.objects.filter(id=self.job_id, status__in=OPEN_JOB_STATUSES).update(**updates)
        finished = time.perf_counter()
        self._write_seconds += finished - started
        # From the *end* of the write: a write that took a second has already
        # spent the next interval's worth of wall clock, and charging it from
        # the start would let a slow disk write continuously.
        self._last_write = finished
        self._writes += 1

    def _eta_seconds(self) -> float | None:
        if not self.total or self.done <= 0 or self.done >= self.total:
            return None
        elapsed = time.perf_counter() - self._started
        if elapsed <= 0:
            return None
        return elapsed * (self.total - self.done) / self.done

    # --- context manager ---

    def __enter__(self) -> UnitProgressScope:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # A failed run must not be left claiming its last successful tile was
        # the last one there was, so the final write only happens on success.
        if exc_type is None:
            self.finish()
        else:
            self._closed = True


class NullUnitProgressScope(UnitProgressScope):
    """The scope inference gets when nothing is watching (CLI, tests).

    Same surface, no database. Having this rather than ``None`` is why the
    inference loop carries no ``if reporting is enabled`` branch.
    """

    def __init__(self, *, total: int = 0, label: str = "", **_ignored: Any) -> None:
        self.job_id = ""
        self.label = label
        self.stage = None
        self.total = max(int(total), 0)
        self.done = 0
        self._started = time.perf_counter()
        self._min_interval = None
        self._last_write = 0.0
        self._write_seconds = 0.0
        self._writes = 0
        self._skipped = 0
        self._detail = {}
        self._closed = False

    def _write(self, *, force: bool = False) -> None:
        return


def unit_scope(
    *,
    total: int,
    label: str,
    stage: str | None = None,
    detail: dict[str, Any] | None = None,
) -> UnitProgressScope:
    """A unit-progress scope for whatever job owns this thread, or a no-op."""
    reporter = active_reporter()
    if reporter is None:
        return NullUnitProgressScope(total=total, label=label)
    return reporter.unit_scope(total=total, label=label, stage=stage, detail=detail)


def report_stage(stage: str, *, detail: dict[str, Any] | None = None) -> None:
    """Record a phase change on whatever job owns this thread, if any."""
    reporter = active_reporter()
    if reporter is not None:
        reporter.stage(stage, detail=detail)

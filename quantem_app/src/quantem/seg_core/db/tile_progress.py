"""Keeping the tile count on the job row current, whatever the disk is doing.

Why the tile count is written from here as well as from the scope
-----------------------------------------------------------------
:class:`quantem.jobs.reporter.UnitProgressScope` writes tile counts from inside
the tiling loop under a **cumulative** write-time budget: once the seconds it
has spent writing exceed 1 % of the seconds the scope has been open, it stops
writing until the ratio recovers. On a healthy database that never bites -- a
progress UPDATE measures ~0.24 ms and a CPU tile takes ~0.7 s.

It is not a healthy-machine guarantee. One UPDATE on this database was measured
at **1 129.8 ms** (a WAL checkpoint, an antivirus scan, a disk waking up). A
single sample like that buys the budget out for roughly the next 113 seconds,
and on a run shorter than that the tile counter freezes at whatever it last
said until the forced final write. On screen a frozen counter is
indistinguishable from a stalled run, and the user's only remedy is to guess.

So the number the user reads does not depend on that budget. This module writes
the same three columns, from the same per-tile callback, with three differences
that are the whole point:

* **wall-clock cadence, not a time budget.** A slow write can delay the next
  sample by the length of the slow write and nothing more; it can never switch
  reporting off for the rest of the run.
* **a failed write is one missed sample.** The exception is logged once and
  swallowed -- a progress write must never be the thing that fails a run -- and
  the next tile tries again.
* **the denominator goes on the row before the model loads.** The tiling plan is
  known from the image shape and the pack's canonical scale, so the row can say
  "0 of 56 tiles" through the 4-20 s model load instead of nothing.

Nothing here invents a number. ``done`` and ``total`` are the same integers the
tiling loop hands to :meth:`UnitProgressScope.set`; the writer only makes sure
they arrive.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

#: Shortest gap between two writes of the same run's tile count. A tile takes
#: ~0.7 s on CPU and ~0.05 s on CUDA, so this costs at most one UPDATE per
#: second (~0.02 % of wall clock) and bounds how stale the number on screen can
#: be at one second plus the length of one slow write.
MIN_WRITE_INTERVAL_SECONDS = 1.0


def _job_row_api():
    """``(Job, OPEN_JOB_STATUSES, UNIT_TILE, active_reporter)`` or None.

    Imported lazily and defensively: ``seg_core`` runs from the CLI and from
    tests with no job behind it, and on an install where ``quantem.jobs`` is not
    migrated. Finding nothing here is not an error -- it means nobody is
    watching, and reporting degrades to doing nothing.
    """
    try:
        from quantem.jobs.models import (  # noqa: PLC0415 -- optional at import time
            OPEN_JOB_STATUSES,
            UNIT_TILE,
            Job,
        )
        from quantem.jobs.reporter import active_reporter  # noqa: PLC0415
    except Exception:  # pragma: no cover -- no Django settings, or jobs absent
        return None
    return Job, OPEN_JOB_STATUSES, UNIT_TILE, active_reporter


def _apply_unit_window(done: int, total: int) -> tuple[int, int]:
    """Offset into a multi-organelle wave, when one is in force on this thread.

    The window belongs to :mod:`quantem.jobs.reporter` because the *other*
    writer of these columns -- ``UnitProgressScope``, opened inside the tiling
    loop -- has to agree with this one about it. Two writers with two opinions
    about the denominator is precisely the defect a shared row would otherwise
    reintroduce.
    """
    try:
        from quantem.jobs.reporter import apply_unit_window  # noqa: PLC0415
    except Exception:  # pragma: no cover -- jobs absent
        return done, total
    return apply_unit_window(done, total)


class TileProgressWriter:
    """The tile count for the job running on this thread, kept fresh.

    Constructing one is cheap and always succeeds. When there is no job -- a CLI
    run, a test -- every method is a no-op, which is why the inference path
    below carries no "is anyone watching" branch.
    """

    def __init__(self, *, min_interval_seconds: float = MIN_WRITE_INTERVAL_SECONDS):
        self._min_interval = float(min_interval_seconds)
        self._job_id: str | None = None
        self._api = _job_row_api()
        if self._api is not None:
            _job, _open, _label, active_reporter = self._api
            reporter = active_reporter()
            self._job_id = getattr(reporter, "job_id", None) if reporter else None
        self._done = -1
        self._total: int | None = None
        self._last_write = 0.0
        self._started = time.perf_counter()
        self._writes = 0
        self._failures = 0
        self._reported_failure = False

    @property
    def active(self) -> bool:
        """Whether there is a job row to write onto."""
        return bool(self._job_id)

    @property
    def stats(self) -> dict[str, int]:
        return {"writes": self._writes, "failures": self._failures}

    # --- the two calls the inference path makes ---

    def announce(self, total: int) -> None:
        """Put the denominator on the row before any work has happened."""
        total = max(int(total), 0)
        if not total:
            return
        self._total = total
        self._done = 0
        self._write(0, total, eta=None)

    def report(self, done: int, total: int) -> None:
        """Report ``done`` of ``total`` tiles walked.

        Silently held when the count has not moved, when it moved backwards, or
        when the previous write was less than :data:`MIN_WRITE_INTERVAL_SECONDS`
        ago -- unless this is the last tile, which is always written.
        """
        total = max(int(total), 0)
        done = max(int(done), 0)
        if total:
            done = min(done, total)
        if done <= self._done and total == self._total:
            return

        final = bool(total) and done >= total
        now = time.perf_counter()
        if not final and (now - self._last_write) < self._min_interval:
            # Hold the value, not the fact: the next tile past the interval
            # writes the *current* count, so a held sample is never replayed
            # stale.
            self._done = max(self._done, done)
            self._total = total
            return

        self._done = done
        self._total = total
        self._write(done, total, eta=self._eta_seconds(done, total))

    # --- internals ---

    def _eta_seconds(self, done: int, total: int) -> float | None:
        """Seconds left, measured on this leg's rate over the **wave's** remainder.

        The two halves come from different places on purpose. The rate can only
        be measured over the tiles this writer actually watched -- an organelle
        that started thirty seconds ago has no claim on the time the one before
        it took. The work still to come is the wave's, because that is what the
        person is waiting for. Quoting this leg's own remainder on a shared row
        would promise a finish time with three organelles still to run.
        """
        if not total or done <= 0 or done >= total:
            return None
        elapsed = time.perf_counter() - self._started
        if elapsed <= 0:
            return None
        windowed_done, windowed_total = _apply_unit_window(done, total)
        remaining = max(windowed_total - windowed_done, 0)
        if remaining <= 0:
            return None
        return elapsed * remaining / done

    def _write(self, done: int, total: int, *, eta: float | None) -> None:
        if self._api is None or not self._job_id:
            return
        Job, open_statuses, unit_tile, _active_reporter = self._api
        try:
            from django.utils import timezone  # noqa: PLC0415

            windowed_done, windowed_total = _apply_unit_window(done, total)
            updates = {
                "progress_units_done": windowed_done,
                "progress_units_total": windowed_total,
                "progress_unit_label": unit_tile,
                "updated_at": timezone.now(),
            }
            if eta is not None:
                # Read-modify-write, once per write, so the keys the scope put
                # there (model, organelle, device) survive. They are the only
                # machine-readable record of what produced these tiles.
                detail = dict(
                    Job.objects.filter(id=self._job_id)
                    .values_list("progress_detail_json", flat=True)
                    .first()
                    or {}
                )
                detail["eta_seconds"] = round(eta, 1)
                updates["progress_detail_json"] = detail
            # Only onto a job that still has work ahead of it: a count written
            # onto a concluded run is a lie, where silence is only silence.
            Job.objects.filter(id=self._job_id, status__in=open_statuses).update(**updates)
        except Exception:
            # One missed sample. Never a failed run, and never a reason to stop
            # trying -- that is the freeze this class exists to prevent.
            self._failures += 1
            if not self._reported_failure:
                self._reported_failure = True
                logger.warning(
                    "job %s: could not write the tile count; the run continues "
                    "and the next tile will try again",
                    self._job_id,
                    exc_info=True,
                )
            return
        self._writes += 1
        self._last_write = time.perf_counter()

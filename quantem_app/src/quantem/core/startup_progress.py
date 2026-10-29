"""Structured liveness and progress events for desktop startup.

The desktop cannot use HTTP until database preparation is complete, so the
sidecar streams one small JSON protocol over stdout.  Every startup path uses
the same state shape: ordinary initialization, schema upgrades, the one-time
SQLite cutover, and PostgreSQL crash recovery.  A heartbeat repeats the latest
state during opaque library or database calls without pretending that elapsed
time is percentage complete.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

STARTUP_PROGRESS_PREFIX = "QUANTEM_STARTUP_PROGRESS "
StartupProgressCallback = Callable[[dict[str, object]], None]


def emit_startup_progress(event: Mapping[str, object]) -> None:
    """Write one machine-readable event for the desktop shell."""

    print(
        STARTUP_PROGRESS_PREFIX + json.dumps(dict(event), separators=(",", ":"), sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


class StartupProgress:
    """Own the current startup state and keep it visibly alive."""

    def __init__(
        self,
        callback: StartupProgressCallback = emit_startup_progress,
        *,
        heartbeat_interval: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._callback = callback
        self._heartbeat_interval = heartbeat_interval
        self._clock = clock
        self._started_at = clock()
        self._stage_started_at = self._started_at
        self._last_progress_at = self._started_at
        self._payload: dict[str, object] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._heartbeat: threading.Thread | None = None
        if heartbeat_interval > 0:
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                name="quantem-startup-progress",
                daemon=True,
            )
            self._heartbeat.start()

    def stage(
        self,
        phase: str,
        message: str,
        *,
        detail: str = "",
        category: str = "startup",
        critical: bool = False,
        completed: int | None = None,
        total: int | None = None,
        unit: str = "",
        technical_detail: str = "",
    ) -> None:
        """Publish a new startup phase and reset its liveness clock."""

        now = self._clock()
        payload: dict[str, object] = {
            "phase": phase,
            "category": category,
            "message": message,
            "detail": detail,
            "critical": critical,
            "can_close": not critical,
            "completed": completed,
            "total": total,
            "unit": unit,
            "technical_detail": technical_detail,
            "heartbeat": False,
        }
        with self._lock:
            previous_phase = self._payload.get("phase")
            if previous_phase:
                payload["previous_phase"] = previous_phase
                payload["previous_phase_elapsed_seconds"] = round(
                    max(0.0, now - self._stage_started_at)
                )
            self._stage_started_at = now
            self._last_progress_at = now
            self._payload = payload
        self._emit(payload, now=now)

    def update(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        detail: str | None = None,
        unit: str | None = None,
        technical_detail: str | None = None,
        progress: bool = True,
    ) -> None:
        """Update measurable work within the current phase."""

        now = self._clock()
        with self._lock:
            payload = dict(self._payload)
            if completed is not None:
                payload["completed"] = completed
            if total is not None:
                payload["total"] = total
            if detail is not None:
                payload["detail"] = detail
            if unit is not None:
                payload["unit"] = unit
            if technical_detail is not None:
                payload["technical_detail"] = technical_detail
            payload["heartbeat"] = False
            if progress:
                self._last_progress_at = now
            self._payload = payload
        self._emit(payload, now=now)

    def forward(
        self,
        event: Mapping[str, object],
        *,
        category: str,
        critical: bool,
    ) -> None:
        """Forward a subsystem event through the general startup protocol."""

        now = self._clock()
        payload = dict(event)
        payload.update(
            {
                "category": category,
                "critical": critical,
                "can_close": not critical,
            }
        )
        heartbeat = bool(payload.get("heartbeat"))
        with self._lock:
            if not heartbeat:
                if payload.get("phase") != self._payload.get("phase"):
                    previous_phase = self._payload.get("phase")
                    if previous_phase:
                        payload["previous_phase"] = previous_phase
                        payload["previous_phase_elapsed_seconds"] = round(
                            max(0.0, now - self._stage_started_at)
                        )
                    self._stage_started_at = now
                self._last_progress_at = now
            self._payload = payload
        self._emit(payload, now=now, preserve_timing=True)

    def fail(self, error: BaseException) -> None:
        """Publish the phase and useful cause before the sidecar exits."""

        if self._closed:
            return
        now = self._clock()
        with self._lock:
            previous = dict(self._payload)
        payload = {
            "phase": "failed",
            "category": "failure",
            "message": "QuantEM could not finish starting",
            "detail": str(error) or error.__class__.__name__,
            "technical_detail": previous.get("technical_detail", ""),
            "failed_phase": previous.get("phase", "starting"),
            "critical": False,
            "can_close": True,
            "heartbeat": False,
        }
        with self._lock:
            self._payload = payload
        self._emit(payload, now=now)

    def ready(self) -> None:
        self.stage(
            "ready",
            "QuantEM is ready",
            detail="Opening your library",
            category="ready",
        )
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=min(self._heartbeat_interval, 1.0))
            self._heartbeat = None

    def _timed_payload(
        self,
        payload: Mapping[str, object],
        *,
        now: float,
        preserve_timing: bool,
    ) -> dict[str, object]:
        timed: dict[str, Any] = dict(payload)
        completed = timed.get("completed")
        total = timed.get("total")
        if not preserve_timing or "percent" not in timed:
            percent: float | None = None
            if isinstance(completed, int) and isinstance(total, int) and total > 0:
                percent = round(min(100.0, max(0.0, completed * 100.0 / total)), 1)
            timed["percent"] = percent
        phase_elapsed = round(max(0.0, now - self._stage_started_at))
        # These two fields belong to the general startup timeline, even when a
        # forwarded subsystem reports its own local elapsed time.
        timed["elapsed_seconds"] = round(max(0.0, now - self._started_at))
        timed["phase_elapsed_seconds"] = phase_elapsed
        if not preserve_timing or "last_progress_seconds" not in timed:
            timed["last_progress_seconds"] = round(max(0.0, now - self._last_progress_at))
        if (
            "eta_seconds" not in timed
            and isinstance(completed, int)
            and isinstance(total, int)
            and 0 < completed < total
            and phase_elapsed > 0
        ):
            timed["eta_seconds"] = max(
                1,
                round((total - completed) * phase_elapsed / completed),
            )
        return timed

    def _emit(
        self,
        payload: Mapping[str, object],
        *,
        now: float,
        preserve_timing: bool = False,
    ) -> None:
        try:
            self._callback(self._timed_payload(payload, now=now, preserve_timing=preserve_timing))
        except Exception:
            # Progress reporting must never be able to break a database upgrade.
            return

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_interval):
            now = self._clock()
            with self._lock:
                if not self._payload:
                    continue
                payload = dict(self._payload)
                payload["heartbeat"] = True
            # Recompute liveness from this reporter even when a forwarded event
            # carried its own earlier timing values.
            payload.pop("elapsed_seconds", None)
            payload.pop("last_progress_seconds", None)
            self._emit(payload, now=now, preserve_timing=True)


__all__ = [
    "STARTUP_PROGRESS_PREFIX",
    "StartupProgress",
    "StartupProgressCallback",
    "emit_startup_progress",
]

"""Cooperative, process-wide priority for interactive model work.

OS locks are released even when a worker is killed. Bulk inference acquires one
batch at a time; a waiting prompt takes priority at the next batch boundary.
This shares admission between the HTTP process and persistent job workers.
"""

from __future__ import annotations

import contextlib
import errno
import os
import threading
import time

_LOCAL = threading.local()
# Leave a short gap for persistence and the next pointer action in a burst.
# Otherwise a two-second CPU tile can start between every 100-ms prompt.
INTERACTIVE_GRACE_SECONDS = 0.35


def _path(name):
    from quantem.core.config import TMP_DIR

    root = TMP_DIR / "compute-admission"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name.replace(':', '-')}.lock"


def _try_lock(handle):
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
    return True


def _unlock(handle):
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def interactive_pending(*, include_grace=False) -> bool:
    with _path("interactive").open("a+b") as handle:
        if not _try_lock(handle):
            return True
        _unlock(handle)
        if include_grace:
            try:
                age = time.time() - _path("recent-interactive").stat().st_mtime
                return 0 <= age < INTERACTIVE_GRACE_SECONDS
            except OSError:
                pass
        return False


@contextlib.contextmanager
def interactive_request():
    """Hold prompt priority while selecting, loading and using its actual device."""
    if getattr(_LOCAL, "priority", False):
        yield 0.0
        return
    started = time.perf_counter()
    with _path("interactive").open("a+b") as handle:
        while not _try_lock(handle):
            time.sleep(0.02)
        _LOCAL.priority = True
        try:
            yield time.perf_counter() - started
        finally:
            _LOCAL.priority = False
            with contextlib.suppress(OSError):
                _path("recent-interactive").touch()
            _unlock(handle)


@contextlib.contextmanager
def compute_slot(device: str, *, interactive=False, on_yield=None, on_resume=None):
    """Serialize device execution, allowing bulk work to yield its allocation.

    Reentrancy is per device. A CPU fallback must acquire the CPU lane even
    when its caller already has an accelerator lane.
    """
    if interactive:
        with (
            interactive_request() as priority_wait,
            compute_slot(device, on_yield=on_yield, on_resume=on_resume) as device_wait,
        ):
            yield priority_wait + device_wait
        return
    if device == "cuda":
        assigned = os.environ.get("QUANTEM_WORKER_DEVICE", "")
        device = assigned if assigned.startswith("cuda:") else "cuda:0"
    started = time.perf_counter()
    held = getattr(_LOCAL, "devices", set())
    if device in held:
        yield 0.0
        return
    owns_priority = getattr(_LOCAL, "priority", False)
    if held and not owns_priority:
        raise RuntimeError("Release the current compute lane before switching devices.")
    yielded = False
    with _path(f"compute-{device}").open("a+b") as handle:
        while True:
            if not owns_priority and interactive_pending(include_grace=True):
                if not yielded and on_yield is not None:
                    on_yield()
                yielded = True
                time.sleep(0.02)
                continue
            if _try_lock(handle):
                # A prompt may have arrived while this worker acquired the
                # compute lock; release it before starting another batch.
                if not owns_priority and interactive_pending(include_grace=True):
                    _unlock(handle)
                    continue
                break
            time.sleep(0.02)
        _LOCAL.devices = held | {device}
        try:
            if yielded and on_resume is not None:
                on_resume()
            yield time.perf_counter() - started
        finally:
            _LOCAL.devices = held
            _unlock(handle)

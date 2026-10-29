import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from quantem.core.compute_admission import compute_slot, interactive_pending


@pytest.fixture(autouse=True)
def isolated_lanes(tmp_path, monkeypatch):
    from quantem.core import compute_admission

    monkeypatch.setattr(compute_admission, "_path", lambda name: tmp_path / (name + ".lock"))


def test_unexpected_lock_error_is_not_treated_as_contention(tmp_path, monkeypatch):
    import errno
    import os

    from quantem.core import compute_admission

    def broken(*args):
        raise OSError(errno.ENOSYS, "locking is unsupported")

    if os.name == "nt":
        monkeypatch.setattr("msvcrt.locking", broken)
    else:
        monkeypatch.setattr("fcntl.flock", broken)
    with (tmp_path / "lock").open("a+b") as handle:
        with pytest.raises(OSError, match="unsupported"):
            compute_admission._try_lock(handle)


def test_future_grace_timestamp_cannot_stall_background_work(monkeypatch):
    import os

    from quantem.core import compute_admission

    path = compute_admission._path("recent-interactive")
    path.touch()
    future = time.time() + 3600
    os.utime(path, (future, future))
    assert not interactive_pending(include_grace=True)


def test_interactive_device_switch_acquires_each_real_lane():
    from quantem.core import compute_admission

    with compute_admission.interactive_request():
        with compute_slot("cuda:0"):
            assert compute_admission._LOCAL.devices == {"cuda:0"}
        with compute_slot("cpu"):
            assert compute_admission._LOCAL.devices == {"cpu"}
    assert not compute_admission._LOCAL.devices


def test_prompt_precedes_next_batch_and_locks_release_after_failure():
    active = threading.Event()
    release = threading.Event()
    order = []

    def first_batch():
        with compute_slot("cpu"):
            active.set()
            assert release.wait(5)
            order.append("first")

    def prompt():
        with compute_slot("cpu", interactive=True):
            order.append("prompt")

    def second_batch():
        with compute_slot("cpu"):
            order.append("second")

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(first_batch)
        assert active.wait(5)
        interactive = pool.submit(prompt)
        deadline = time.monotonic() + 5
        while not interactive_pending() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert interactive_pending()
        second = pool.submit(second_batch)
        release.set()
        for task in (first, interactive, second):
            task.result(timeout=5)
    assert order == ["first", "prompt", "second"]
    try:
        with compute_slot("cpu", interactive=True):
            raise ValueError("request failed")
    except ValueError:
        pass
    assert not interactive_pending()
    with compute_slot("cpu"):
        pass


def test_killed_process_releases_interactive_and_compute_locks(tmp_path, monkeypatch):
    import os
    import signal
    import subprocess
    import sys

    from quantem.core import compute_admission

    monkeypatch.setattr(compute_admission, "_path", lambda name: tmp_path / (name + ".lock"))
    script = """
import os, sys, time
from pathlib import Path
from quantem.core import compute_admission as admission
admission._path = lambda name: Path(sys.argv[1]) / (name + '.lock')
with admission.compute_slot('cpu', interactive=True):
    print(os.getpid(), flush=True)
    time.sleep(30)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    actual_pid = None
    try:
        actual_pid = int(child.stdout.readline().strip())
        assert interactive_pending()
    finally:
        # A Windows venv can launch the interpreter through a redirector.
        # Popen.pid then names the launcher, not the process holding the locks.
        if actual_pid is not None:
            os.kill(actual_pid, signal.SIGTERM)
        else:
            child.terminate()
        child.wait(timeout=5)
        child.stdout.close()
    assert not interactive_pending()
    with compute_slot("cpu"):
        pass

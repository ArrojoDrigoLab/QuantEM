"""A worker must not outlive the process that spawned it.

The leak, MEASURED in wave-0 verification: force-killing the server left a
spawned worker holding **905 MB** running with a dead parent, and restarting the
server did not reap it. A second agent's server had produced an identical 911 MB
orphan the same session, so it is a shape, not an accident. In the shipped Tauri
app that is ~1 GB resident until reboot every time a user force-quits -- which
is how people leave an app that looks stuck.

``daemon=True`` does not help: on Windows a force-quit is ``TerminateProcess``,
so no ``atexit`` hook runs and multiprocessing never gets to terminate its
daemon children. The fix has to live in the child.

The integration test below is deliberately end to end -- a real spawned
grandchild, a real force-kill of its parent, and the operating system asked
afterwards -- because every cheaper version of it passes whether or not the
watchdog exists. The negative control runs the identical scenario with the
watchdog switched off and proves the orphan is still one environment variable
away.
"""

from __future__ import annotations

import ast
import ctypes
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import quantem
from quantem.jobs import pool, runner

SRC_ROOT = Path(quantem.__file__).resolve().parent

#: Long enough for a spawned child to import Django and run ``django.setup()``
#: on a cold cache; the watchdog itself reacts in milliseconds.
CHILD_START_TIMEOUT_SECONDS = 120.0

#: How long the child gets to notice its dead parent before the test calls it an
#: orphan. Waiting on a sentinel is immediate; this is slack, not a poll period.
ORPHAN_GRACE_SECONDS = 30.0

#: How long the *negative control* waits before concluding the child survived.
SURVIVAL_OBSERVATION_SECONDS = 10.0


def _process_is_alive(pid: int) -> bool:
    """Liveness without psutil, and without ``os.kill``.

    ``os.kill(pid, 0)`` is a probe on POSIX and a **kill** on Windows -- CPython
    maps any signal other than the console-control ones straight to
    ``TerminateProcess``, so the portable-looking version of this function would
    silently make the test pass by killing the thing it is asking about.
    """
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    SYNCHRONIZE = 0x00100000
    WAIT_TIMEOUT = 0x00000102
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def _wait_until_gone(pid: int, timeout: float) -> float | None:
    """Seconds the process took to exit, or ``None`` if it is still running."""
    started = time.monotonic()
    while (time.monotonic() - started) < timeout:
        if not _process_is_alive(pid):
            return time.monotonic() - started
        time.sleep(0.2)
    return None


#: A stand-in for the server: it spawns one worker the way ``JobRunner`` does and
#: then does nothing, so the only thing the test kills is a *parent*.
_PARENT_SCRIPT = textwrap.dedent(
    """
    import multiprocessing as mp
    import sys
    import time

    from quantem.jobs.pool import parent_death_probe

    if __name__ == "__main__":
        ctx = mp.get_context("spawn")
        child = ctx.Process(target=parent_death_probe, args=(sys.argv[1],))
        child.start()
        while True:
            time.sleep(1)
    """
)


def _start_parent_and_child(tmp_path: Path, *, watchdog: bool):
    pid_file = tmp_path / "worker.pid"
    env = os.environ.copy()
    env[pool.PARENT_DEATH_WATCHDOG_ENV_VAR] = "1" if watchdog else "0"
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT_SCRIPT, str(pid_file)],
        env=env,
        cwd=str(SRC_ROOT.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + CHILD_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if pid_file.exists():
            text = pid_file.read_text(encoding="utf-8").strip()
            if text:
                return parent, int(text)
        if parent.poll() is not None:
            _, err = parent.communicate()
            pytest.fail(f"the stand-in server died before spawning a worker: {err!r}")
        time.sleep(0.2)
    parent.kill()
    pytest.fail("the spawned worker never reported its pid")


def _force_kill(process: subprocess.Popen) -> None:
    """What a user's Task Manager / force-quit does: TerminateProcess."""
    process.kill()
    process.wait(timeout=30)


def test_a_worker_dies_when_its_parent_is_force_killed(tmp_path):
    parent, worker_pid = _start_parent_and_child(tmp_path, watchdog=True)
    try:
        assert _process_is_alive(worker_pid), "the worker was not running to begin with"
        _force_kill(parent)

        took = _wait_until_gone(worker_pid, ORPHAN_GRACE_SECONDS)
        assert took is not None, (
            f"worker {worker_pid} outlived its force-killed parent "
            f"{parent.pid} by more than {ORPHAN_GRACE_SECONDS:.0f}s. This is the "
            "905 MB orphan finding F4 measured."
        )
    finally:
        if parent.poll() is None:
            _force_kill(parent)
        if _process_is_alive(worker_pid):  # pragma: no cover - failure cleanup
            subprocess.run(
                ["taskkill", "/PID", str(worker_pid), "/F"]
                if sys.platform == "win32"
                else ["kill", "-9", str(worker_pid)],
                capture_output=True,
                check=False,
            )


def test_without_the_watchdog_the_orphan_is_reproduced(tmp_path):
    """The negative control: the bug, on demand.

    Without this the test above could be guarding nothing -- it would pass just
    as happily if spawned children had always died with their parents. They do
    not, on any platform.
    """
    parent, worker_pid = _start_parent_and_child(tmp_path, watchdog=False)
    orphan_survived = False
    try:
        _force_kill(parent)
        orphan_survived = _wait_until_gone(worker_pid, SURVIVAL_OBSERVATION_SECONDS) is None
    finally:
        if _process_is_alive(worker_pid):
            subprocess.run(
                ["taskkill", "/PID", str(worker_pid), "/F"]
                if sys.platform == "win32"
                else ["kill", "-9", str(worker_pid)],
                capture_output=True,
                check=False,
            )
        _wait_until_gone(worker_pid, 10.0)

    assert orphan_survived, (
        "a spawned child with the watchdog disabled died with its parent "
        "anyway. Either the platform changed -- in which case say so here and "
        "keep the watchdog -- or this control has stopped testing anything."
    )


class TestTheWatchdogIsInstalledWhereItMatters:
    def test_the_server_process_installs_nothing(self):
        """``parent_process()`` is ``None`` here, so there is nothing to watch.

        It matters that this is a no-op rather than an error: the same entry
        points run in-process in inline job mode and in the test suite.
        """
        assert pool.install_parent_death_watchdog() is False

    def test_the_pool_initializer_installs_it_first(self, monkeypatch):
        seen: list[str | None] = []

        def fake_watchdog() -> bool:
            # Recording the marker proves ordering: the initializer claims it
            # on the very next line.
            seen.append(os.environ.get(pool.WORKER_PROCESS_ENV_VAR))
            return True

        monkeypatch.setattr(pool, "install_parent_death_watchdog", fake_watchdog)
        # delenv records "was absent" and puts that back on teardown, including
        # removing the marker the initializer is about to set. Leaving it set
        # would suppress file logging for every later test in this interpreter.
        monkeypatch.delenv(pool.WORKER_PROCESS_ENV_VAR, raising=False)

        pool.django_pool_initializer()

        assert seen == [None], (
            "django.setup() can block for seconds; a child force-orphaned "
            "during it must already be watching its parent"
        )

    def test_the_job_worker_entry_point_installs_it(self, monkeypatch):
        installed: list[bool] = []

        def fake_watchdog() -> bool:
            installed.append(True)
            return False

        monkeypatch.setattr(runner, "install_parent_death_watchdog", fake_watchdog)
        monkeypatch.setattr(runner, "_run_job_in_subprocess", lambda *a, **k: None)

        runner.run_job_in_subprocess("00000000-0000-0000-0000-000000000000")

        assert installed == [True]

    def test_the_watchdog_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv(pool.PARENT_DEATH_WATCHDOG_ENV_VAR, "0")
        assert pool._watchdog_enabled() is False
        monkeypatch.setenv(pool.PARENT_DEATH_WATCHDOG_ENV_VAR, "1")
        assert pool._watchdog_enabled() is True
        monkeypatch.delenv(pool.PARENT_DEATH_WATCHDOG_ENV_VAR, raising=False)
        assert pool._watchdog_enabled() is True


class TestTheWorkerSetupIsNotDuplicated:
    def test_setup_django_is_the_pool_initializer(self, monkeypatch):
        """One implementation, so a worker and a pool child cannot diverge.

        They had already diverged: only the pool copy installs the watchdog.
        """
        calls: list[str] = []
        monkeypatch.setattr(runner, "django_pool_initializer", lambda: calls.append("initializer"))

        runner._setup_django()

        assert calls == ["initializer"]

    def test_the_worker_marker_is_imported_rather_than_redeclared(self):
        """Value equality is not enough here: CPython interns the literal, so
        two independent declarations of ``"QUANTEM_JOB_WORKER"`` are the same
        object and ``is`` would pass over the duplication this removes. Ask the
        syntax instead."""
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        redeclared = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "WORKER_PROCESS_ENV_VAR"
        ]
        assert not redeclared, (
            "jobs/runner.py declares WORKER_PROCESS_ENV_VAR again at line(s) "
            f"{redeclared}; import it from quantem.jobs.pool instead."
        )
        imported = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "quantem.jobs.pool"
            and any(alias.name == "WORKER_PROCESS_ENV_VAR" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert imported
        assert runner.WORKER_PROCESS_ENV_VAR == pool.WORKER_PROCESS_ENV_VAR


class _FakeWorker:
    def __init__(self, alive: bool = True):
        self.alive = alive
        self.terminated = False
        self.joined_with: float | None = None

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.joined_with = timeout


class TestShutdownStopsTheWorkersItStarted:
    """The clean-exit half: quitting must not wait minutes for a full run.

    ``multiprocessing``'s own exit hook *joins* non-daemon children, so a Quit
    during a segmentation blocked until the segmentation finished. An app that
    looks hung is force-quit, which is how the orphan gets made.
    """

    def _runner(self):
        job_runner = runner.JobRunner()
        job_runner.running.clear()
        job_runner.gpu_workers.clear()
        return job_runner

    def test_a_running_worker_is_terminated_and_joined(self):
        job_runner = self._runner()
        worker = _FakeWorker()
        job_runner.running["job-1"] = runner.RunningJob(worker, "cpu")

        job_runner.shutdown()

        assert worker.terminated
        assert worker.joined_with == runner.SHUTDOWN_JOIN_SECONDS

    def test_a_persistent_gpu_worker_is_terminated_too(self):
        job_runner = self._runner()
        worker = _FakeWorker()
        job_runner.gpu_workers[runner.GPU_POOL_KEY] = [worker]

        job_runner.shutdown()

        assert worker.terminated

    def test_a_worker_that_already_exited_is_left_alone(self):
        job_runner = self._runner()
        worker = _FakeWorker(alive=False)
        job_runner.running["job-1"] = runner.RunningJob(worker, "cpu")

        job_runner.shutdown()

        assert not worker.terminated
        assert worker.joined_with is None

    def test_an_inline_thread_is_skipped_rather_than_crashing_the_exit(self):
        import threading

        job_runner = self._runner()
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        job_runner.running["job-1"] = runner.RunningJob(thread, "cpu")

        job_runner.shutdown()  # must not raise: threads have no terminate()

    def test_a_new_runner_is_registered_with_the_process_exit_hook(self):
        # Not by calling _shutdown_live_runners here: that set holds every
        # runner any other test in this interpreter still has alive, and
        # terminating their workers from this test would be action at a
        # distance. Assert the wiring instead.
        job_runner = self._runner()

        assert job_runner in runner._LIVE_RUNNERS
        assert runner._ATEXIT_REGISTERED is True

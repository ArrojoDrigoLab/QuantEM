"""What ``quantem serve`` says when it starts.

The regression this pins down was reported as "``quantem serve`` prints
nothing -- not even the URL". Two causes, both real:

* the announcement was printed *after* ``migrate``, so a first launch sat on a
  silent terminal for the whole migration; and
* the prints were not flushed, so with stdout piped (a wrapper process, a log
  file) block buffering held them back until the process exited -- which a
  server never does. Nothing ever appeared.

So this test runs the real entry point with stdout *piped* -- the exact
configuration that used to show nothing -- and requires the four startup lines
(URL, data dir, log file, the models-on-demand pointer) to arrive while the
server is still running. Without ``flush=True`` in ``cmd_serve`` the lines sit
in the child's buffer and this test times out; that is the failure it exists to
catch.

The log-file line is paper-cut 6's other half: the packaged server used to
write no log file at all, so this test also waits for the announced file to
exist with content while the server is still up.
"""

from __future__ import annotations

import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_SRC.parent

#: Generous: a first launch does django.setup() on a cold interpreter. The
#: lines under test are printed *before* that, so in practice they arrive in a
#: couple of seconds; the margin is for a loaded CI machine.
STARTUP_LINE_TIMEOUT_S = 120.0

#: Far more generous: the *log file* only appears after the spawned server has
#: finished ``django.setup()``, which imports torch. Under a full-suite run the
#: box is saturated by sibling workers and that import alone was measured
#: blowing through 120 s (3/3 full-suite runs failed here while 4/4 isolated
#: runs passed in seconds). The deadline is not the thing under test -- the
#: flush-before-migrate behaviour above is -- so it errs way on the side of a
#: loaded machine rather than flaking the whole backend gate.
LOG_FILE_TIMEOUT_S = 420.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_serve_announces_url_data_dir_and_models_pointer_on_a_piped_stdout(tmp_path):
    data_dir = tmp_path / "serve-data"
    port = _free_port()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC)
    # The server proper is not under test and the scheduler thread only slows
    # the exit down. ``_prepare_env`` uses setdefault, so this wins.
    env["QUANTEM_AUTOSTART_JOBS"] = "0"
    env.pop("QUANTEM_DATA_DIR", None)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from quantem.cli import main; raise SystemExit(main())",
            # -c consumes only the command string; everything after the -c
            # argument lands in sys.argv[1:] for argparse.
            "serve",
            "--port",
            str(port),
            "--data-dir",
            str(data_dir),
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    lines: queue.Queue[str] = queue.Queue()

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line.rstrip("\r\n"))

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()

    try:
        received: list[str] = []
        for _ in range(4):
            try:
                received.append(lines.get(timeout=STARTUP_LINE_TIMEOUT_S))
            except queue.Empty:
                stderr = ""
                if proc.poll() is not None:
                    stderr = proc.communicate()[1] or ""
                raise AssertionError(
                    "quantem serve did not announce itself on a piped stdout "
                    f"within {STARTUP_LINE_TIMEOUT_S:.0f}s; got only "
                    f"{received!r}. This is the buffered-print regression: the "
                    "startup lines must be printed with flush=True, before the "
                    f"migrations. stderr: {stderr[:2000]}"
                ) from None

        # Order matters: the URL is the line a wrapper script scrapes first.
        assert f"http://127.0.0.1:{port}" in received[0], received
        assert str(data_dir) in received[1], received
        assert received[2].startswith("log file: "), received
        log_path = Path(received[2][len("log file: ") :])
        assert log_path == data_dir / "logs" / "quantem-server.log", received
        assert "downloaded on demand" in received[3], received
        assert "models" in received[3], received

        # And the process is still a server, not something that printed on its
        # way out: the lines above must have arrived from a *running* process.
        assert proc.poll() is None, (
            f"quantem serve exited (rc={proc.returncode}) before serving: "
            f"{proc.communicate()[1][:2000]}"
        )

        # Paper-cut 6: the announced log file must actually appear, with
        # content, while the server is still running -- the whole point is a
        # record that survives the session. It appears only after the child
        # finishes django.setup() (which imports torch), so this wait has its
        # own, much larger deadline: under a saturated full-suite run that
        # import alone can take minutes, and this test must not turn machine
        # load into a red backend gate.
        waited_from = time.monotonic()
        deadline = waited_from + LOG_FILE_TIMEOUT_S
        while time.monotonic() < deadline:
            if log_path.is_file() and log_path.stat().st_size > 0:
                break
            if proc.poll() is not None:
                raise AssertionError(
                    f"quantem serve exited (rc={proc.returncode}) before "
                    f"writing its log file: {proc.communicate()[1][:2000]}"
                )
            time.sleep(0.25)
        else:
            raise AssertionError(
                f"the announced log file {log_path} never appeared with "
                f"content although the server process was still alive after "
                f"{time.monotonic() - waited_from:.0f}s "
                f"(deadline {LOG_FILE_TIMEOUT_S:.0f}s). The server prints its "
                "banner before django.setup() and logs it after, so either "
                "file logging broke, or the machine is so loaded that even "
                f"{LOG_FILE_TIMEOUT_S:.0f}s was not enough for django.setup() "
                "(imports torch) in the child."
            )
        assert "serving on http://127.0.0.1" in log_path.read_text(
            encoding="utf-8", errors="replace"
        )
    finally:
        proc.kill()
        proc.wait(timeout=30)

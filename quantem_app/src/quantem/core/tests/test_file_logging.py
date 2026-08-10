"""The packaged server must leave a log file behind (paper-cut 6).

Before this, the only record of a session was the console -- which the desktop
shell swallows and a closed terminal erases -- so a crash report from the field
came with nothing attached. ``quantem serve`` and the frozen build (which runs
``cmd_serve`` through the same CLI) now set ``QUANTEM_LOG_TO_FILE=1``, and the
settings module answers by adding a size-capped rotating file handler writing
``logs/quantem-server.log`` under the data directory.

Three facts are pinned here, each in a subprocess because logging is configured
once per process at ``django.setup()``:

* the flag adds exactly one rotating file handler, INFO by default, under the
  data directory -- while the console handler keeps its own level, unchanged;
* no flag means no file handler: a dev ``runserver`` and the test suite write
  no log files anywhere;
* a spawned job worker (``QUANTEM_JOB_WORKER=1``) never gets the handler even
  with the flag inherited in its environment. One process, one writer: rotation
  renames the file, and on Windows a rename fails while any other process holds
  it open.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_SRC.parent

_PROBE = textwrap.dedent(
    """
    import json
    import logging
    from logging.handlers import RotatingFileHandler

    import django

    django.setup()

    root = logging.getLogger()
    out = []
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler):
            out.append(
                {
                    "file": handler.baseFilename,
                    "level": logging.getLevelName(handler.level),
                    "max_bytes": handler.maxBytes,
                    "backups": handler.backupCount,
                }
            )
    print("PROBE:" + json.dumps(out))
    """
)


def _rotating_handlers(env_overrides: dict[str, str | None]) -> list[dict]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC)
    env["DJANGO_SETTINGS_MODULE"] = "quantem.core.settings"
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    for line in out.stdout.splitlines():
        if line.startswith("PROBE:"):
            return json.loads(line[len("PROBE:"):])
    raise AssertionError(f"probe printed nothing: {out.stdout!r}")


def test_the_flag_adds_one_rotating_info_file_handler_under_the_data_dir(tmp_path):
    data_dir = tmp_path / "logging-data"
    handlers = _rotating_handlers(
        {
            "QUANTEM_DATA_DIR": str(data_dir),
            "QUANTEM_LOG_TO_FILE": "1",
            "QUANTEM_JOB_WORKER": None,
            "DJANGO_LOG_LEVEL": None,
        }
    )

    assert len(handlers) == 1, handlers
    handler = handlers[0]
    assert Path(handler["file"]) == data_dir.resolve() / "logs" / "quantem-server.log"
    assert handler["level"] == "INFO"
    assert 0 < handler["max_bytes"] <= 64 * 1024 * 1024  # size-capped
    assert 1 <= handler["backups"] <= 10  # a few rotations, not an archive


def test_without_the_flag_no_file_handler_exists(tmp_path):
    handlers = _rotating_handlers(
        {
            "QUANTEM_DATA_DIR": str(tmp_path / "plain-data"),
            "QUANTEM_LOG_TO_FILE": None,
            "QUANTEM_JOB_WORKER": None,
        }
    )
    assert handlers == []


def test_a_job_worker_never_writes_the_server_log(tmp_path):
    """The worker inherits the whole server environment; the marker must win."""
    handlers = _rotating_handlers(
        {
            "QUANTEM_DATA_DIR": str(tmp_path / "worker-data"),
            "QUANTEM_LOG_TO_FILE": "1",
            "QUANTEM_JOB_WORKER": "1",
        }
    )
    assert handlers == []


def test_the_console_handler_keeps_its_own_level_beside_the_file():
    """INFO in the file must not turn the console INFO too.

    The root logger has to pass INFO records for the file handler to see them,
    so the console handler needs its own explicit level. This is importable
    state, not subprocess state: the suite's own settings module suffices.
    """
    from django.conf import settings

    console = settings.LOGGING["handlers"]["console"]
    assert console.get("level") == settings.LOG_LEVEL
    assert logging.getLevelNamesMapping().get(console.get("level")) is not None

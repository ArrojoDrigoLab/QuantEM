"""``python -m quantem`` must be the CLI (adversarial round 13, minor 5).

The ``quantem`` console script only exists for an installed copy; a checkout
driven with ``PYTHONPATH=src`` has no scripts directory, and ``python -m
quantem`` is the standard spelling for both. It used to fail with "No module
named quantem.__main__".
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[3]


def test_python_dash_m_quantem_shows_the_cli_help():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC)

    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "quantem", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_SRC.parent),
    )

    assert proc.returncode == 0, proc.stderr[:2000]
    assert "serve" in proc.stdout
    assert "models" in proc.stdout

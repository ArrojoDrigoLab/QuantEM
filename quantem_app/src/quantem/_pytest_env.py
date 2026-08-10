"""Keep the test suite out of the user's real data directory.

Registered as a pytest plugin from ``pyproject.toml``::

    addopts = "... -p quantem._pytest_env"

**Why a plugin and not a ``conftest.py``.** ``quantem.core.config`` resolves
``STORAGE_DIR`` once, at import, and everything QuantEM writes hangs off it --
the database, renditions, overlays, model packs, caches, logs. With no
``QUANTEM_DATA_DIR`` set it falls back to the installation's own data
directory -- ``<sys.prefix>/quantem-data``, the venv this suite runs from --
which is right for the shipped application and wrong for ``pytest``: running
the suite would write into the directory holding a developer's actual work.

A root ``conftest.py`` cannot prevent that. ``pytest-django`` sets Django up in
``pytest_load_initial_conftests``, which wins the race against the rootdir
conftest body -- measured, not assumed. A ``-p`` plugin is imported before
that, which is early enough.

An explicit ``QUANTEM_DATA_DIR`` still wins, so the CI lane and the clean-start
runs point where they choose.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repo-local, beside the other scratch state and git-ignored.
TEST_DATA_DIR = Path(__file__).resolve().parents[2].parent / ".scratch" / "pytest_data"

ENV_VAR = "QUANTEM_DATA_DIR"


def _redirect_data_dir() -> None:
    if os.environ.get(ENV_VAR, "").strip():
        return
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_VAR] = str(TEST_DATA_DIR)


_redirect_data_dir()

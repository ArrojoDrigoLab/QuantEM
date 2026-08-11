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
_SCRATCH = Path(__file__).resolve().parents[2].parent / ".scratch"
TEST_DATA_DIR = _SCRATCH / "pytest_data"

#: pytest's own ``tmp_path`` tree. Redirecting only the data directory left this
#: behind: ``tmp_path_factory`` defaults to ``tempfile.gettempdir()``, so a run
#: on a stock Windows box wrote hundreds of files -- including whole model packs
#: copied by the registry tests -- into ``%LOCALAPPDATA%\\Temp``. This project may
#: not touch the C: drive at all, and a suite that quietly does is worse than one
#: that fails, because nobody looks.
TEST_TMP_DIR = _SCRATCH / "pytest_tmp"

ENV_VAR = "QUANTEM_DATA_DIR"


def _redirect_data_dir() -> None:
    if os.environ.get(ENV_VAR, "").strip():
        return
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_VAR] = str(TEST_DATA_DIR)


def _redirect_temp_dir() -> None:
    """Point both pytest's basetemp and ``tempfile`` at repo scratch.

    The environment variables cover code under test that calls ``tempfile``
    directly; ``pytest_configure`` below covers ``tmp_path``/``tmp_path_factory``,
    which read the option rather than the environment.
    """
    TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[name] = str(TEST_TMP_DIR)


def pytest_configure(config) -> None:
    if config.option.basetemp is None:
        config.option.basetemp = str(TEST_TMP_DIR / "basetemp")


def pytest_runtest_teardown(item, nextitem) -> None:
    """Forget any ``JobReporter`` a test left registered on this thread.

    ``JobReporter.__init__`` calls ``activate()``, so merely constructing one
    makes it the reporter ``unit_scope`` and the device-notice path will find.
    Tests that construct one and do not call ``deactivate()`` leak it into every
    test that follows on the same thread, while the ``Job`` row it points at is
    rolled back with its transaction.

    The damage is delayed and looks like someone else's bug: a later test does
    something that writes a progress update or a log line, that write lands
    against a job id that no longer exists, and the failure surfaces as a
    foreign-key violation in an unrelated module. It stayed hidden until a
    CUDA fallback started emitting device notices on this machine.

    Pairing every construction with a ``deactivate()`` is the tidy fix and is
    what the eight call sites should do; clearing it here is the one that
    cannot be forgotten by the ninth.
    """
    del item, nextitem
    try:
        from quantem.jobs.reporter import _ACTIVE
    except Exception:  # noqa: BLE001 - the app may not be importable at all
        return
    _ACTIVE.reporter = None


_redirect_data_dir()
_redirect_temp_dir()

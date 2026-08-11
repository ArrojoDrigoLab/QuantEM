"""Where QuantEM decides to write everything.

``STORAGE_DIR`` is resolved once, at the import of :mod:`quantem.core.config`,
and everything else -- the database, renditions, overlays, model packs, logs --
hangs off it. Two things about that resolution were wrong or untested:

* A ``.env`` file setting ``QUANTEM_DATA_DIR`` was documented in
  ``settings.py`` and had no effect, because importing the settings module
  imports the ``quantem.core`` package first and that is where the directory
  was already being resolved.
* ``pytest`` with no ``QUANTEM_DATA_DIR`` wrote into the *real* platform data
  directory, so running the suite polluted a developer's actual work.

The default itself is an owner ruling (2026-08-09): **all application storage
lives with the installation.**

* frozen desktop build: ``<install>\\data``, derived from the executable's own
  location (the layout is ``<install>\\QuantEM.exe`` beside
  ``<install>\\quantem-server\\quantem-server.exe``);
* pip install: ``<sys.prefix>/quantem-data`` -- the environment *is* the
  install location, which also covers venvs and this dev checkout (whose
  ``.env`` still overrides);
* ``QUANTEM_DATA_DIR`` always wins, and an unwritable computed location is a
  clear error naming that override -- **never** a silent fallback to
  ``%LOCALAPPDATA%``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_SRC = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_SRC.parent


def _resolve_storage_dir_in_subprocess(env_overrides: dict[str, str | None]) -> str:
    """Import Django the way a real launch does, and report STORAGE_DIR.

    A subprocess, because STORAGE_DIR is fixed at first import and this process
    has already done that.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC)
    env["DJANGO_SETTINGS_MODULE"] = "quantem.core.settings"
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    script = textwrap.dedent(
        """
        import django
        django.setup()
        from quantem.core.config import STORAGE_DIR
        print(STORAGE_DIR)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_an_explicit_env_var_wins(tmp_path):
    target = tmp_path / "explicit"
    resolved = _resolve_storage_dir_in_subprocess({"QUANTEM_DATA_DIR": str(target)})
    assert Path(resolved) == target.resolve()


@pytest.mark.skipif(
    not (REPO_SRC / "quantem" / ".env").is_file(),
    reason="no development .env in this checkout",
)
def test_a_dotenv_file_can_set_the_data_dir():
    """The regression: this was documented and inert.

    Only meaningful in a source checkout that has one; a packaged install has
    no ``.env`` at all.
    """
    from quantem.core.env_files import BACKEND_ENV_FILES

    assert ".env" in BACKEND_ENV_FILES
    resolved = _resolve_storage_dir_in_subprocess({"QUANTEM_DATA_DIR": None})
    declared = None
    for line in (REPO_SRC / "quantem" / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("QUANTEM_DATA_DIR="):
            declared = line.split("=", 1)[1].strip()
    assert declared, "the .env does not set QUANTEM_DATA_DIR"
    assert Path(resolved) == Path(declared).resolve()


def test_the_suite_does_not_write_to_the_real_data_directory():
    """``quantem._pytest_env`` redirects an unset ``QUANTEM_DATA_DIR``.

    Without it, ``pytest`` writes a database, renditions, overlays and caches
    into the directory holding the developer's actual images.

    Compared against the *platform* directory rather than
    :func:`quantem.cli.default_data_dir`, which honours ``QUANTEM_DATA_DIR``
    itself and would therefore agree with ``STORAGE_DIR`` no matter what.
    """
    from quantem.core.config import STORAGE_DIR

    if sys.platform == "win32":
        platform_root = Path(
            os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        )
    elif sys.platform == "darwin":
        platform_root = Path.home() / "Library" / "Application Support"
    else:
        platform_root = Path.home() / ".local" / "share"

    resolved = Path(STORAGE_DIR).resolve(strict=False)
    assert not resolved.is_relative_to(platform_root.resolve(strict=False)), (
        f"the suite is writing into the real user data directory ({resolved})"
    )
    # And not into the *new* default either: the environment's own
    # ``quantem-data``, which is where a developer's real data now lives.
    assert resolved != (Path(sys.prefix).resolve() / "quantem-data"), (
        f"the suite is writing into the environment's real data directory ({resolved})"
    )


def test_the_plugin_is_registered():
    """A guard that is not wired up protects nothing.

    It has to be a ``-p`` plugin: pytest-django sets Django up during
    ``pytest_load_initial_conftests``, which beats a rootdir ``conftest.py``.
    """
    config = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "-p quantem._pytest_env" in config


# --- The default location: with the installation, never a per-user directory


def test_the_pip_default_is_inside_the_environment(monkeypatch):
    """pip channel: the environment *is* the install location.

    ``pip install quantem-app`` into a venv, conda env or system Python must keep
    its storage with that environment -- delete the environment, and the data
    it owned goes with it -- not in a hidden per-user directory that outlives
    every install and is shared by all of them.
    """
    from quantem.cli import default_data_dir

    monkeypatch.delenv("QUANTEM_DATA_DIR", raising=False)
    assert default_data_dir() == Path(sys.prefix).resolve() / "quantem-data"


def test_the_frozen_default_sits_beside_the_install(monkeypatch, tmp_path):
    """Executable channel: ``<install>\\data``, derived from the exe path.

    The installer's layout is ``<install>\\QuantEM.exe`` plus
    ``<install>\\quantem-server\\quantem-server.exe``; the frozen server is the
    process that resolves the data directory, so the install root is its exe's
    grandparent.
    """
    from quantem.cli import default_data_dir

    exe = tmp_path / "SomeChosenDir" / "quantem-server" / "quantem-server.exe"
    monkeypatch.delenv("QUANTEM_DATA_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))

    assert default_data_dir() == tmp_path / "SomeChosenDir" / "data"


def test_the_env_var_still_wins_in_a_frozen_build(monkeypatch, tmp_path):
    """The desktop shell passes ``QUANTEM_DATA_DIR`` through; it must win."""
    from quantem.cli import default_data_dir

    override = tmp_path / "elsewhere"
    monkeypatch.setenv("QUANTEM_DATA_DIR", str(override))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        sys, "executable", str(tmp_path / "inst" / "quantem-server" / "qs.exe")
    )

    assert default_data_dir() == override


def test_an_unwritable_data_dir_is_a_clear_error_naming_the_override(
    monkeypatch, tmp_path
):
    """No silent fallback -- the owner ruling this whole module implements.

    A data directory that cannot be created (here: its parent is a *file*)
    must stop the launch with an error that names the directory and the
    ``QUANTEM_DATA_DIR`` override, not quietly relocate storage to
    ``%LOCALAPPDATA%`` where no one will ever look for it.
    """
    from quantem.cli import _prepare_env

    monkeypatch.delenv("QUANTEM_DATA_DIR", raising=False)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed", encoding="utf-8")
    target = blocker / "data"

    with pytest.raises(SystemExit) as excinfo:
        _prepare_env(target)

    message = str(excinfo.value)
    assert str(target) in message
    assert "QUANTEM_DATA_DIR" in message
    assert "LOCALAPPDATA" not in message

"""
Django project configuration package.

Ensures the user data directory and its subdirectories exist before anything
that writes to them is imported.

The ``.env`` load has to happen *here*, not in :mod:`quantem.core.settings`.
Django imports the settings module, which first imports this package, whose
body used to reach straight for :mod:`quantem.core.config` -- and
``STORAGE_DIR`` is computed at that module's import. So by the time settings.py
called ``load_backend_env_files`` the data directory had already been resolved,
and a ``.env`` setting ``QUANTEM_DATA_DIR`` was read into the environment far
too late to have any effect. The mechanism was documented, exercised by nobody,
and silently inert.
"""

from pathlib import Path

from quantem.core.env_files import load_backend_env_files

# BASE_DIR as quantem.core.settings defines it: the `quantem` package root.
load_backend_env_files(Path(__file__).resolve().parent.parent)

from quantem.core.config import ensure_directories  # noqa: E402

# Create all required directories when Django loads
ensure_directories()

__all__ = ()

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

The machine profile is applied on the *first* line, ahead of even that. This is
the Django half of BIG_IMAGE_DESIGN S0: ``DJANGO_SETTINGS_MODULE`` points at
:mod:`quantem.core.settings`, so every Django entry -- ``django.setup()``,
wsgi, pytest-django, a spawned job worker -- runs this package body before it
runs anything that could import numpy. ``configure_process`` pins
``OMP_NUM_THREADS`` and friends, which OpenBLAS and OpenMP read at numpy's
import to size their per-thread arenas and never read again. MEASURED on the
build box: 1 668 MB of commit unpinned against 252 MB pinned to two threads.
Ahead of the ``.env`` load because that load is itself an import, and because
the pin must precede all of them; ``QUANTEM_MACHINE_PROFILE`` is therefore a
real environment variable, not a ``.env`` key.
"""

from quantem.core.machine import configure_process

configure_process()

from pathlib import Path  # noqa: E402

from quantem.core.env_files import load_backend_env_files  # noqa: E402

# BASE_DIR as quantem.core.settings defines it: the `quantem` package root.
load_backend_env_files(Path(__file__).resolve().parent.parent)

from quantem.core.config import ensure_directories  # noqa: E402

# Create all required directories when Django loads
ensure_directories()

__all__ = ()

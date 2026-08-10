"""Frozen entry point for the ``quantem-server`` executable.

This is the module PyInstaller freezes. It is deliberately three lines of
behaviour on top of :func:`quantem.cli.main`:

* ``multiprocessing.freeze_support()`` **first**, before any other import runs
  application code. The job queue runs every segmentation in a spawned worker
  process (``quantem.jobs.runner`` uses the ``spawn`` context and
  ``mp.set_executable(sys.executable)``). In a frozen build the spawned child
  re-executes this very executable; ``freeze_support()`` is what turns that
  re-execution into a worker instead of a second server.

* ``DJANGO_SETTINGS_MODULE`` defaults to :mod:`quantem_server_settings`, a shim
  that re-exports the real settings and points ``QUANTEM_FRONTEND_DIST`` at the
  frontend bundle shipped inside this build. ``quantem.cli._prepare_env`` uses
  ``setdefault``, so the CLI honours the value set here. Nothing else about the
  server changes; a user who sets DJANGO_SETTINGS_MODULE themselves wins.

Everything else -- argument parsing, the data directory, port selection, the
job queue -- is exactly the pip-installed ``quantem`` package.
"""

import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantem_server_settings")
    from quantem.cli import main

    sys.exit(main())

"""Django settings for the frozen ``quantem-server`` build.

Identical to :mod:`quantem.core.settings` -- everything is re-exported -- plus
one value: ``QUANTEM_FRONTEND_DIST`` points at the frontend bundle PyInstaller
ships inside the application (``_internal/quantem_frontend/dist``).

Why a shim instead of data placed where the default path already looks:
``quantem.core.spa`` computes its default as ``parents[3]`` of its own file,
which in the frozen layout resolves *outside* the ``_internal`` directory --
and PyInstaller cannot install data files above ``_internal``. The settings
override seam (``settings.QUANTEM_FRONTEND_DIST``) exists for exactly this, so
the frozen build uses it rather than patching the package.

Selected by ``quantem_server_entry`` via ``DJANGO_SETTINGS_MODULE``; the CLI's
``setdefault`` keeps it, and an explicit user override still wins.
"""

import sys
from pathlib import Path

from quantem.core.settings import *  # noqa: F401,F403

if getattr(sys, "frozen", False):
    _dist = Path(getattr(sys, "_MEIPASS", ".")) / "quantem_frontend" / "dist"
    if (_dist / "index.html").is_file():
        QUANTEM_FRONTEND_DIST = str(_dist)

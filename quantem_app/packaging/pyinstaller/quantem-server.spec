# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ``quantem-server`` onedir build.

Build (from ``quantem_app/``, with the project venv's python)::

    python -m PyInstaller packaging/pyinstaller/quantem-server.spec \
        --workpath <scratch>/pyi_build --distpath packaging/pyinstaller/dist

Set ``PYINSTALLER_CONFIG_DIR`` (and TMP/TEMP) somewhere sensible first; on the
build machine everything must stay off C:.

What ships and why:

* ``collect_all`` for the first-party package and the Django stack. Django
  discovers management commands and migrations by scanning packages at
  runtime (``pkgutil.iter_modules``); PyInstaller's frozen importer supports
  that only for modules it actually collected, so the whole tree goes in.
* torch / torchvision / timm / skimage / cv2 / shapely / zarr / numcodecs /
  imagecodecs ride on their standard hooks (pyinstaller-hooks-contrib), with
  ``collect_submodules`` added where registries import plugins dynamically
  (numcodecs entry points, imagecodecs codec modules).
* The built frontend (``frontend/dist``) ships as data under
  ``quantem_frontend/dist``; ``quantem_server_settings`` points Django at it.
* ``quantem_server_settings`` itself is a hidden import: nothing imports it
  statically -- it is named via ``DJANGO_SETTINGS_MODULE``.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

SPEC_DIR = Path(SPECPATH).resolve()
APP_ROOT = SPEC_DIR.parents[1]  # quantem_app/
SRC_DIR = APP_ROOT / "src"
FRONTEND_DIST = APP_ROOT / "frontend" / "dist"

if not (FRONTEND_DIST / "index.html").is_file():
    raise SystemExit(
        f"frontend build missing at {FRONTEND_DIST}; run `npm run build` in frontend/ first"
    )

datas = [(str(FRONTEND_DIST), "quantem_frontend/dist")]
binaries = []
hiddenimports = ["quantem_server_settings", "waitress"]

# Packages whose submodules are imported dynamically (Django app/command/
# migration discovery, DRF renderers named in settings strings, codec
# registries). collect_all = submodules + data files + binaries.
for pkg in (
    "quantem",
    "django",
    "rest_framework",
    "django_filters",
    "corsheaders",
    "environ",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Codec/plugin registries that resolve implementations at runtime.
for pkg in ("zarr", "numcodecs", "imagecodecs", "tifffile"):
    hiddenimports += collect_submodules(pkg)

# Version lookups via importlib.metadata at runtime.
for dist_name in ("huggingface-hub", "timm", "zarr", "numcodecs", "imagecodecs"):
    try:
        datas += copy_metadata(dist_name)
    except Exception:
        pass

a = Analysis(
    [str(SPEC_DIR / "quantem_server_entry.py")],
    pathex=[str(SRC_DIR), str(SPEC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Dev/test tooling and GUI stacks that must never ride along.
        "tkinter",
        "IPython",
        "matplotlib",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "pytest",
        "mypy",
        "ruff",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="quantem-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # the sidecar's stdout/stderr are the server log
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="quantem-server",
)

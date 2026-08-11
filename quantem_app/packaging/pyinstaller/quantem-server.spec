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
from fnmatch import fnmatch
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

datas = [
    (str(FRONTEND_DIST), "quantem_frontend/dist"),
    # The frozen bundle is the one artifact that actually redistributes the
    # dependencies in binary form, which is what their BSD/MIT/Apache notice
    # terms attach to. The wheel and the conda package carry NOTICE through
    # their own metadata; this build has to be told.
    (str(APP_ROOT / "NOTICE"), "."),
]
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
    # ``excludes`` is the *module-graph* tool: it drops Python modules, and the
    # extension modules reached from them, before anything is collected. It is
    # the wrong instrument for a loose binary or data file -- nothing imports
    # one, so nothing can be excluded from importing it. Those are filtered off
    # ``a.binaries``/``a.datas`` below.
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
        # A ratchet, not a saving: nothing here reaches sympy today, so this
        # removes nothing from the current build. torch imports it from
        # ``torch.fx.experimental.symbolic_shapes`` and friends, and a torch
        # upgrade that puts one of those on an eagerly-imported path would
        # re-admit ~70 MB with no diff in this repository to explain it.
        "sympy",
        # torch's own test harness (~5 MB of Python). Imported by torch's test
        # suite only; ``torch.testing/__init__.py`` pulls in ``_comparison``,
        # never ``_internal``, and its modules want pytest/expecttest anyway.
        "torch.testing._internal",
        # imagecodecs is reached only through tifffile -- this application
        # never imports it (assets/tests/test_ngff_decode_chokepoint.py bans
        # the direct import), and tifffile's compression table names neither of
        # these: AVIF is an AV1-derived consumer still format and Brunsli is a
        # JPEG recompressor, and no TIFF compression tag maps to either.
        # ~20 MB of the 47 MB imagecodecs ships. imagecodecs loads each codec
        # lazily in ``__getattr__`` and substitutes a raise-on-use stub when the
        # extension is absent, so a missing one costs an error at call time
        # rather than an ImportError at startup.
        "imagecodecs._avif",
        "imagecodecs._brunsli",
    ],
    noarchive=False,
    optimize=0,
)

# --- Files that must not ride along ----------------------------------------
#
# ``collect_all`` and the third-party hooks copy these in wholesale, so
# ``excludes`` above cannot see them: no import statement leads to a loose
# binary or data file, so there is nothing to exclude from importing it. Match
# is against the *destination* name inside the bundle, and a hook may classify
# the same file as a binary or as a datum depending on how it found it, so both
# TOCs are filtered with the same list.
#
# The reverse mistake is just as wrong: a Python package cannot be removed here
# (its modules are already inside the PYZ), which is why the entries above and
# the entries below are two different lists.
NOT_FOR_THE_BUNDLE = (
    # The checkout-only data-directory redirect. pyproject.toml excludes
    # ``**/.env`` from the wheel *and* the sdist, in both targets, because it
    # holds an absolute path on whichever machine ran the build -- but nothing
    # excluded it here, and ``collect_all("quantem")`` swept it into the frozen
    # build as ``quantem/.env``, absolute QUANTEM_DATA_DIR and all.
    #
    # What that costs today is the leak itself: a shipped artifact naming a
    # directory on the build machine. It does *not* currently redirect anyone's
    # data, because ``read_env`` never overwrites an already-set variable (see
    # core/settings.py) and every entry point -- ``cli._prepare_env``, the
    # desktop shell's export -- sets QUANTEM_DATA_DIR before Django loads. So
    # the file is inert, and one missing ``_prepare_env`` call away from not
    # being. Do not read the masking as a reason to leave it in the bundle, and
    # do not read this filter as the thing that keeps the masking working.
    "quantem/.env",
    # OpenCV's ffmpeg video backend, ~29 MB -- the single largest removable
    # file in the bundle. cv2 loads it lazily, and only for VideoCapture /
    # VideoWriter; this application calls three OpenCV functions in total
    # (resize, fillPoly, dilate) and neither reads nor writes video.
    "cv2/opencv_videoio_ffmpeg*.dll",
    # protobuf's compiler, ~2.7 MB, shipped inside the torch wheel for people
    # building C++ extensions against it. No torch Python module executes it.
    "torch/bin/protoc.exe",
)


def _wanted(entry):
    dest = str(entry[0]).replace("\\", "/")
    return not any(fnmatch(dest, pattern) for pattern in NOT_FOR_THE_BUNDLE)


_before = len(a.binaries) + len(a.datas)
a.binaries = [entry for entry in a.binaries if _wanted(entry)]
a.datas = [entry for entry in a.datas if _wanted(entry)]
print(f"spec: dropped {_before - len(a.binaries) - len(a.datas)} file(s) that must not ship")

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

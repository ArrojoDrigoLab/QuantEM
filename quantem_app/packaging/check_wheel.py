"""Release gate for a built wheel (or sdist): no machine-local paths, no weights.

Usage::

    python packaging/check_wheel.py dist/quantem-0.1.0.dev0-py3-none-any.whl
    python packaging/check_wheel.py dist/quantem-0.1.0.dev0.tar.gz

Runs :func:`quantem.registry.release.find_local_paths` — the same scanner that
keeps build-machine paths out of model release bundles — over every text file in
the archive, and refuses archives that carry model weights, ``node_modules``, or
scratch litter.

The scanner is a path-shaped-string detector, so three kinds of matches are not
findings and are filtered here rather than by loosening the scanner:

* **URL routes** (``/api/...``, ``/roi/...``): paths this application *serves*.
* **Data-directory fragments** (``/cache/hf``): paths the application *writes*,
  always documented relative to the user's data directory.
* **Minified-bundle noise**: base64/wasm runs that start with ``/``, and
  emscripten's virtual-filesystem home ``/home/web_user`` in the vendored blosc
  codec. Neither is a place on anyone's disk.

What remains after filtering is compared against :data:`PINNED` — an exact,
per-file list of documentary path examples this project has decided may ship
(chiefly the docstrings of the path-sanitising modules themselves).
Pinned hits are printed, so a pass never hides them; **any hit not pinned fails
the gate**, so a new machine path cannot ride in on the precedent of an old one.

Exit status 0 iff the artifact carries nothing but pinned documentary hits.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantem.registry.release import find_local_paths  # noqa: E402

#: Suffixes read as text and scanned for local paths.
TEXT_SUFFIXES = {
    ".py", ".txt", ".md", ".html", ".js", ".mjs", ".css", ".map", ".json",
    ".yaml", ".yml", ".toml", ".cfg", ".svg", ".ts", ".tsx",
}

#: Extensions that mean model weights escaped into the application artifact.
WEIGHT_SUFFIXES = {".pt", ".pth", ".safetensors", ".ckpt", ".onnx", ".npz"}

#: A hit with one of these prefixes is a URL path this application serves, not
#: a filesystem path. Everything the frontend requests, and every route string
#: in core/urls.py, starts with one of these roots.
URL_PREFIXES = (
    "/api/", "/assets/", "/ngff/", "/segmentation-overlays/", "/static/",
    "/analysis/", "/adapt/", "/apply/", "/cancel/", "/retry/", "/install/",
    "/models/", "/label/", "/segments/", "/segmentations/", "/complete",
    "/exports/", "/labels/", "/probability-maps/", "/ngff-thumbnail/",
    "/refinement/", "/export/", "/data/images/",
    # route fragments in core/urls.py under the segmentation include
    "/overlay-manifest/", "/overlay-rebuild/", "/overlay-lut/", "/roi/",
    "/user-feedback/", "/config/", "/rerun-roi/", "/apply-full-image/",
)

#: Paths the application writes under the user's *data directory*; docs quote
#: them relative to it (``<QUANTEM_DATA_DIR>/cache/hf``), and the scanner sees
#: the tail as a POSIX absolute path.
DATA_DIR_PREFIXES = ("/cache/",)

#: A slash followed by 30+ base64 characters and nothing path-like: a chunk of
#: an embedded wasm/data blob, not a path. The blosc codec bundle is full of
#: these.
_BASE64_RUN = re.compile(r"/[A-Za-z0-9+/=]{30,}$")

#: Emscripten's hard-coded virtual-filesystem home, present in any wasm bundle
#: built with it (here: the vendored blosc codec). Not a machine.
_EMSCRIPTEN_HOME = "/home/web_user"

#: Documentary path examples that ship deliberately, pinned exactly.
#:
#: Keys are fnmatch patterns over *normalized* member names (sdist prefix
#: stripped; ``src/quantem/`` -> ``quantem/``; ``frontend/dist/`` ->
#: ``quantem/_frontend/``), values are the exact hit strings allowed there.
#: Every entry is prose *about* paths — the path-sanitising machinery
#: documenting what it removes. None is read, resolved, or executed. Remove a docstring example at the source and
#: the pin becomes dead; a NEW hit anywhere fails regardless of this table.
PINNED: dict[str, set[str]] = {
    # hf.py documents the hf-xet telemetry leak it closes: the docstring names
    # the C: log directory xet wrote to so the reader knows what the redirect
    # is for. A path in prose about a path bug, not a machine reference.
    "quantem/registry/hf.py": {"/.cache/huggingface/xet/logs``"},
    # The path scanner itself: its docstring and patterns are example paths by
    # construction (synthetic hosts, drive letters, fuzz remnants).
    "quantem/registry/release.py": {
        "\\\\SOMEHOST\\\\share\\\\...``",
        "\\\\r\\\\n``",
        "\\\\HOST\\share\\dir\\file",
        "D:\\Chris\\...",
        "V:/Chris/...",
        "V:/aw:!ez``",
        "L:\\z{=%0``",
        "/mnt/d/...``)",
        "/mnt/d/...",
        "/root/...",
        "/home/...",
        "/C/>`BO=``)",
        "/root/dino/foundation_weights/...``).",
    },
    # Docstrings explaining that provenance records *no* such path.
    "quantem/analysis/provenance.py": {
        "\\\\Chris\\\\uat4_data\\\\images\\\\x.png``",
        "\\\\Chris\\\\...``",
        "D:\\\\Chris\\\\uat4_data\\\\images\\\\x.png``",
        "D:\\\\Chris\\\\...``",
    },
    # Docstring explaining what the installer refuses to record.
    "quantem/registry/install.py": {
        "\\\\Chris\\\\...``",
        "D:\\\\Chris\\\\...``",
    },
    # Docstring examples of the drive-letter / WSL forms being normalised.
    "quantem/core/local_storage.py": {
        "C:/",
        "D:/...).",
        "/mnt/<drive>",
    },
    # The data-dir resolution docstring (owner ruling B, 2026-08-09) describes
    # the frozen install layout — ``<install>\QuantEM.exe`` beside
    # ``<install>\quantem-server\quantem-server.exe`` — to explain why the
    # install root is the exe's grandparent. Layout prose with a placeholder
    # root, not a machine reference.
    "quantem/cli.py": {
        "\\\\quantem-server\\\\quantem-server.exe``",
    },
    # UI placeholder text: `e.g. D:\quantem-models-0.1.0` in the install-from-
    # directory field on the Models screen. An example, versioned, no user or
    # host in it. The bundle name is content-hashed, hence the glob.
    "quantem/_frontend/assets/ModelsScreen-*.js": {
        "D:\\\\quantem-models-0.1.0",
    },
    # The project README's own examples: a deliberately generic data-dir and a
    # home-relative downloads folder (``~/Downloads/...``).
    "README.md": {
        "/where/you/want/it",
        "/Downloads/quantem-models-0.1.0",
    },
    # This project's own build configuration (ships in the sdist): hatchling
    # include/exclude globs and ruff per-file-ignore keys, which are
    # path-shaped by nature.
    "pyproject.toml": {
        "/tests/**",
        "/src/quantem",
        "/migrations/*",
        "/api_views/segments/shared.py",
    },
}


def normalize(name: str) -> str:
    """Map wheel and sdist member names into one namespace.

    ``quantem-<ver>/src/quantem/x.py`` (sdist) and ``quantem/x.py`` (wheel)
    are the same file; ``frontend/dist`` in the sdist becomes
    ``quantem/_frontend`` in the wheel. PINNED is written against the wheel's
    names.
    """
    parts = name.split("/")
    if parts and re.fullmatch(r"quantem-[0-9][^/]*", parts[0]) and not parts[0].endswith(".dist-info"):
        parts = parts[1:]
    name = "/".join(parts)
    if name.startswith("src/quantem/"):
        name = "quantem/" + name[len("src/quantem/"):]
    elif name.startswith("frontend/dist/"):
        name = "quantem/_frontend/" + name[len("frontend/dist/"):]
    return name


#: POSIX roots a real machine path starts from. In a minified bundle, a hit
#: starting with anything else (``/NVIDIA/i.exec``, ``/-/g``, ``/dist/${n}``)
#: is a regex literal or URL fragment, not a place on a build machine.
_UNIX_ROOTS = (
    "/home/", "/users/", "/mnt/", "/root/", "/tmp/", "/var/", "/opt/",
    "/etc/", "/usr/", "/srv/", "/media/", "/private/",
)

#: Regex/template syntax that never appears in a real leaked path but riddles
#: minified JS: character classes, alternation, `${...}` interpolation,
#: and `\uXXXX` escapes read by the scanner as UNC-shaped strings.
_JS_SYNTAX = re.compile(r"[(){}\[\]|]|\$\{|\\u[0-9A-Fa-f]{4}")


def _bundle_noise(hit: str) -> bool:
    """Noise test for minified vendor bundles under ``quantem/_frontend/assets``.

    A genuine build-machine leak in a vite bundle takes exactly two forms: a
    Windows drive/UNC path, or a POSIX path rooted at a real filesystem root.
    Keep those; everything else in minified JS is regex literals and template
    strings the path scanner was never meant to read.
    """
    if _JS_SYNTAX.search(hit):
        return True
    return hit.startswith("/") and not hit.lower().startswith(_UNIX_ROOTS)


def is_noise(name: str, hit: str) -> bool:
    if (
        hit.startswith(URL_PREFIXES)
        or hit.startswith(DATA_DIR_PREFIXES)
        or hit.startswith(_EMSCRIPTEN_HOME)
        or _BASE64_RUN.fullmatch(hit)
    ):
        return True
    return name.startswith("quantem/_frontend/assets/") and _bundle_noise(hit)


def pinned_for(name: str) -> set[str]:
    allowed: set[str] = set()
    for pattern, hits in PINNED.items():
        if fnmatch(name, pattern):
            allowed |= hits
    return allowed


#: Top-level files an artifact may carry besides the package itself.
_TOP_LEVEL = {"README.md", "LICENSE", "NOTICE", "pyproject.toml", "PKG-INFO", ".gitignore"}


def _expected_member(name: str) -> bool:
    """After :func:`normalize`: the package, its metadata, or a known top-level."""
    return (
        name.startswith("quantem/")
        or name.startswith("quantem-")  # *.dist-info in the wheel
        or name in _TOP_LEVEL
    )


def iter_members(archive: Path):
    """Yield ``(name, bytes)`` for every file in a .whl/.zip or .tar.gz."""
    if archive.suffix in (".whl", ".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    yield info.filename, zf.read(info)
    else:
        with tarfile.open(archive, "r:*") as tf:
            for member in tf.getmembers():
                if member.isfile():
                    fh = tf.extractfile(member)
                    assert fh is not None
                    yield member.name, fh.read()


def main(argv: list[str]) -> int:
    # Findings can contain arbitrary bytes from scanned files; never let a
    # cp1252 console turn a reportable finding into a UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    if len(argv) != 2:
        print(__doc__)
        return 2
    archive = Path(argv[1])
    problems: list[str] = []
    pinned_seen: list[str] = []
    n_files = n_scanned = 0
    for raw_name, payload in iter_members(archive):
        n_files += 1
        name = normalize(raw_name)
        posix = PurePosixPath(name)
        if posix.suffix.lower() in WEIGHT_SUFFIXES:
            problems.append(f"WEIGHTS  {name}")
            continue
        if {"node_modules", ".scratch", "__pycache__"} & set(posix.parts):
            problems.append(f"LITTER   {name}")
            continue
        if not _expected_member(name):
            # An unrooted hatchling include pattern once swept 800+ files out of
            # frontend/node_modules and desktop/ into the sdist. Everything a
            # release artifact may carry is enumerable; enumerate it.
            problems.append(f"UNEXPECTED {name}")
            continue
        if posix.suffix.lower() not in TEXT_SUFFIXES:
            continue
        n_scanned += 1
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"NOT-UTF8 {name}")
            continue
        if "tests" in posix.parts:
            # Tests ship only in the sdist, and test fixtures are path-shaped
            # by construction — the scrubber's own tests exist to hold
            # leak-shaped strings. Weights/litter checks above still apply.
            continue
        hits = [h for h in find_local_paths(text) if not is_noise(name, h)]
        if not hits:
            continue
        allowed = pinned_for(name)
        new = [h for h in hits if h not in allowed]
        old = [h for h in hits if h in allowed]
        if old:
            pinned_seen.append(f"{name}: {old}")
        if new:
            problems.append(f"PATHS    {name}: {new}")
    print(f"{archive.name}: {n_files} files, {n_scanned} scanned as text")
    if pinned_seen:
        print(f"\n{len(pinned_seen)} file(s) with pinned documentary hits (shipping deliberately):")
        for p in pinned_seen:
            print(" ", p)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" ", p)
        return 1
    print("clean: no weights, no litter, no path that is not pinned above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Release gate for a built wheel (or sdist): no machine-local paths, no weights.

Usage::

    python packaging/check_wheel.py dist/quantem_app-0.1.0-py3-none-any.whl
    python packaging/check_wheel.py dist/quantem_app-0.1.0.tar.gz

Runs :func:`quantem.registry.release.find_local_paths` — the same scanner that
keeps build-machine paths out of model release bundles — over every text file in
the archive, and refuses archives that carry model weights, ``node_modules``, or
scratch litter.

Every text member is read, including the ones with no suffix to read them by:
``METADATA`` embeds the whole README as the long description, which is the PyPI
project page, so a gate that skipped it could not see what the project actually
publishes. See :data:`TEXT_NAMES`.

Inside ``tests/`` the scan is narrowed rather than skipped — see the "Inside
tests" section below for why the old blanket skip let a live laboratory share
into the sdist, and what replaced it.

The scanner is a path-shaped-string detector, so four kinds of matches are not
findings and are filtered here rather than by loosening the scanner:

* **URL routes** (``/api/...``, ``/roi/...``): paths this application *serves*.
* **Data-directory fragments** (``/cache/hf``): paths the application *writes*,
  always documented relative to the user's data directory.
* **Documented platform storage** (``~/Library/Application Support/QuantEM``):
  where a platform says an application keeps its data, in documentation.
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

#: Text members that have no suffix, matched by name instead.
#:
#: A suffix test cannot see an extensionless file, so every one of these was
#: skipped before it was decoded. That was not a small gap: ``METADATA`` (and
#: its sdist twin ``PKG-INFO``) embeds the whole of ``README.md`` verbatim as
#: the long description, and that rendered README *is* the PyPI project page.
#: Anything in the README therefore reached the public through a file the gate
#: could not read. ``RECORD``, ``WHEEL``, ``LICENSE`` and ``NOTICE`` are the rest
#: of the same blind spot.
TEXT_NAMES = {"METADATA", "PKG-INFO", "RECORD", "WHEEL", "LICENSE", "NOTICE"}

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
    "/spot-check/", "/spot-check/answer", "/runs/",
    # The SAM app's box route. Only the fragment is listed, because the scanner
    # never sees the whole route: a path converter is not a path character, so
    # ``sam/segmentations/<uuid:seg_id>/box/`` reads as a POSIX absolute path
    # starting after the ``>``. There is deliberately no ``/sam/`` entry -- the
    # route strings in ``urls.py`` are relative and produce no leading slash,
    # and the client's ``/api/sam/...`` is already covered by ``/api/``.
    "/box/",
    # Same reason, in the fine-tune and library includes:
    # ``runs/<uuid:adapter_id>/progress/`` and ``assets/grouping/``.
    "/progress/", "/grouping/",
)

#: Paths the application writes under the user's *data directory*; docs quote
#: them relative to it (``<QUANTEM_DATA_DIR>/cache/hf``), and the scanner sees
#: the tail as a POSIX absolute path.
DATA_DIR_PREFIXES = ("/cache/",)

#: Per-user storage locations that a platform *documents* as the place an
#: application keeps its data. Written the way every platform's own
#: documentation writes them — home-relative, ``~/Library/Application
#: Support/QuantEM`` — so the scanner sees the tail from the first slash and
#: reports ``/Library/Application``.
#:
#: This is a category, not an exception, which is why it is a rule here rather
#: than a line in :data:`PINNED` (which only ever shrinks). Naming where an
#: application stores its data is what documentation is *for*, and the rule
#: stays tight because it only recognises the home-relative form: a real leak
#: from a build machine is ``/Users/<somebody>/Library/Application Support/…``,
#: whose hit starts at ``/Users/`` and is caught as it was before.
_PLATFORM_STORAGE = re.compile(
    r"/(?:Library/(?:Application|Caches|Logs|Preferences)"
    r"|\.local/(?:share|state)|\.config|\.cache)\b"
)

#: Members whose content is documentation: the READMEs, and the metadata files
#: that embed the README verbatim as the long description.
_DOC_NAMES = {"METADATA", "PKG-INFO"}

#: A slash followed by base64 characters only: a candidate chunk of an embedded
#: wasm/data blob rather than a path. The blosc codec bundle is full of these.
_BASE64_CHARS = re.compile(r"/[A-Za-z0-9+/=]+")


def _base64_run(hit: str) -> bool:
    """Is ``hit`` a slash plus a base64 blob, rather than a path?

    Length is measured on the longest *unbroken* run, not on the whole string.
    ``/`` is itself a base64 character, so "30+ base64 characters" alone also
    described ``/Users/somebody/Library/Application`` — a build machine's home
    directory, silently dismissed as blob noise. A path breaks every few
    characters at a separator; a blob does not.
    """
    if not _BASE64_CHARS.fullmatch(hit):
        return False
    return max(len(part) for part in hit.split("/")) >= 30

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
#:
#: **This table only shrinks** (owner ruling D8). It lost five whole keys on
#: 2026-08-10 when the docstrings behind them were rewritten platform-neutral:
#: ``analysis/provenance.py``, ``registry/install.py``, ``cli.py``, and the
#: Models screen bundle whose install-from-a-folder placeholder was this build
#: machine's drive letter shown to every user on every platform — plus two
#: entries under ``registry/release.py``. The source-side gate that keeps them
#: gone is ``registry/tests/test_d8_no_hardcoded_roots.py``, which reads the
#: source and the built frontend bundle rather than the wheel, so a leak is
#: caught before a release is built rather than at the end of one.
PINNED: dict[str, set[str]] = {
    # machine.py reads a container's memory limit so an 8 GB laptop and a 2 GB
    # container are not told they have the host's RAM. These are the two kernel
    # paths that carry it (cgroup v2, then v1). They are interfaces this code
    # must name, not references to anybody's machine, and they are read-only.
    "quantem/core/machine.py": {
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    },
    # hf.py documents the hf-xet telemetry leak it closes: the docstring names
    # the C: log directory xet wrote to so the reader knows what the redirect
    # is for. A path in prose about a path bug, not a machine reference.
    "quantem/registry/hf.py": {"/.cache/huggingface/xet/logs``"},
    # The path scanner itself: its docstring and patterns are example paths by
    # construction (synthetic hosts, drive letters, fuzz remnants).
    "quantem/registry/release.py": {
        "\\\\r\\\\n``",
        "\\\\HOST\\share\\dir\\file",
        "D:\\Chris\\...",
        "V:/Chris/...",
        "V:/aw:!ez``",
        "L:\\z{=%0``",
        "/mnt/d/...",
        "/root/...",
        "/home/...",
        "/C/>`BO=``)",
        "/root/dino/foundation_weights/...``).",
    },
    # Docstring examples of the drive-letter / WSL forms being normalised.
    "quantem/core/local_storage.py": {
        "C:/",
        "D:/...).",
        "/mnt/<drive>",
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
    # The distribution is ``quantem-app`` (pyproject ``name``), so wheels and
    # sdists are built as ``quantem_app-<ver>``; it was ``quantem-<ver>`` before
    # the rename. Accept both, or the sdist's prefix is never stripped, no
    # member is recognised, and the gate silently scans nothing -- which is
    # exactly what it was doing: "449 files, 0 scanned as text".
    if (
        parts
        and re.fullmatch(r"quantem(_app|-app)?-[0-9][^/]*", parts[0])
        and not parts[0].endswith(".dist-info")
    ):
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
        or _base64_run(hit)
    ):
        return True
    if (name.endswith(".md") or PurePosixPath(name).name in _DOC_NAMES) and (
        _PLATFORM_STORAGE.fullmatch(hit)
    ):
        return True
    return name.startswith("quantem/_frontend/assets/") and _bundle_noise(hit)


# --- Inside tests -----------------------------------------------------------
#
# Tests ship in the sdist and nowhere else, and they are genuinely full of
# path-shaped strings: the scrubber's own fixtures have to *be* leaks for the
# scrubber to be tested at all. Skipping the whole file for that reason left
# 167 files and 1.8 MB of the sdist unread, and four of them carried a live
# laboratory share — two as module-level default values, resolved on import.
#
# So the skip is narrowed from the file to the hit. Inside a test only one
# shape is reported: an absolute path that names a *specific machine's*
# storage. That is a Windows drive letter or a UNC host followed by at least
# two directory-shaped segments — the same discipline
# ``registry/release.py`` already applies when it scans a weight file ("a
# segment has to look like a directory name, there have to be at least two of
# them"). Everything else a fixture needs — POSIX paths, URL and zarr key
# fragments, regex literals, the deliberate garbage in the scrubber's fuzz
# corpus — is left alone, because none of it can name this machine.

#: A directory or file name, as opposed to regex punctuation or binary noise.
_NAME_SEGMENT = re.compile(r"[A-Za-z0-9 ._-]+")

#: Trailing characters a path picks up from the prose around it: reStructured
#: text literal markers, sentence punctuation, a closing bracket or quote. The
#: scanner runs a path to the first whitespace, so these ride along.
_TRAILING_PROSE = "`).,;:'\"" + "\u201d"

#: How a fixture says "this is an example, not a place". A test that needs a
#: machine-shaped path writes one with a word from this vocabulary in one of
#: its segments, and the gate then knows it is deliberate — a convention any
#: new fixture can adopt, rather than a table of blessed strings that has to
#: grow. Matched per *word* (``example-models`` and ``EXAMPLEHOST`` both
#: qualify) and never per substring, so a real ``annotations`` directory is not
#: excused by containing "not".
_EXAMPLE_WORDS = frozenset({
    "example", "examplehost", "somehost", "someone", "somewhere", "anywhere",
    "nowhere", "nonexistent", "not", "dead", "dummy", "fake", "placeholder",
})


def names_a_machine(hit: str) -> bool:
    """Does ``hit`` name storage on one particular computer?

    True for ``V:\\Chris\\SSD Dump`` and ``\\\\HOST\\share\\weights``; false for
    ``D:\\example\\legacy`` (says it is an example), for ``D:/anywhere`` (one
    segment names no place), and for ``L:\\z{=%0`` (not a path).
    """
    text = hit.rstrip(_TRAILING_PROSE)
    if "..." in text:
        return False  # an elided path is not a path: ``D:\...\uploads\x.png``
    if match := re.match(r"[A-Za-z]:", text):
        host: list[str] = []
    elif match := re.match(r"\\{2,4}(" + _NAME_SEGMENT.pattern + r")", text):
        host = [match.group(1)]  # UNC: the host names the machine, and is one
    else:
        return False
    # Split on separator runs: a docstring writes every backslash doubled, so
    # ``V:\\\\Chris\\\\fig4`` and ``V:\\Chris\\fig4`` are the same two segments.
    segments = [s for s in re.split(r"[\\/]+", text[match.end():]) if s]
    if len(segments) < 2:
        return False
    if not all(_NAME_SEGMENT.fullmatch(s) for s in segments):
        return False
    words = {w for s in host + segments for w in re.split(r"[^a-z0-9]+", s.lower())}
    return words.isdisjoint(_EXAMPLE_WORDS)


def pinned_for(name: str) -> set[str]:
    allowed: set[str] = set()
    for pattern, hits in PINNED.items():
        if fnmatch(name, pattern):
            allowed |= hits
    return allowed


#: Top-level files an artifact may carry besides the package itself.
_TOP_LEVEL = {
    "README.md", "LICENSE", "NOTICE", "pyproject.toml", "PKG-INFO", ".gitignore",
}


def _expected_member(name: str) -> bool:
    """After :func:`normalize`: the package, its metadata, or a known top-level."""
    return (
        name.startswith("quantem/")
        or name.startswith("quantem-")  # *.dist-info, pre-rename artifacts
        or name.startswith("quantem_app-")  # *.dist-info after the rename
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
    n_files = n_scanned = n_bytes = 0
    n_test_files = n_test_bytes = 0
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
        if posix.suffix.lower() not in TEXT_SUFFIXES and posix.name not in TEXT_NAMES:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"NOT-UTF8 {name}")
            continue
        n_scanned += 1
        n_bytes += len(payload)
        in_tests = "tests" in posix.parts
        if in_tests:
            n_test_files += 1
            n_test_bytes += len(payload)
        hits = [h for h in find_local_paths(text) if not is_noise(name, h)]
        if in_tests:
            # See "Inside tests" above: a fixture may be path-shaped, but it
            # may not name this machine.
            hits = [h for h in hits if names_a_machine(h)]
        if not hits:
            continue
        allowed = pinned_for(name)
        new = [h for h in hits if h not in allowed]
        old = [h for h in hits if h in allowed]
        if old:
            pinned_seen.append(f"{name}: {old}")
        if new:
            problems.append(f"PATHS    {name}: {new}")
    print(
        f"{archive.name}: {n_files} files, "
        f"{n_scanned} scanned as text ({n_bytes:,} bytes), "
        f"of which {n_test_files} test files ({n_test_bytes:,} bytes)"
    )
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

"""Owner ruling D8: no path that names the build machine ships, anywhere.

    "it's ok for the system to show relevant local paths, but those paths
    should not be hardcoded absolute paths; this application will be run on
    MANY different local machines. If absolute paths are shown, which may very
    well be appropriate in some cases, they should be constructed with dynamic
    roots based on the users machine / installation."

This is a **code** rule, not a copy rule, and it is stricter than D7. D7 says
where a path may appear; D8 says where it must come from: the resolved data
directory, the install root, or the value the user themselves supplied.

**Why a third gate module, and why it looks at files rather than at responses.**
``test_i12_no_cli_in_served_copy`` and ``test_i12_error_copy_sweep`` both read
values -- what the API serialises. A value cannot answer D8's question: a real
resolved data directory and a typed-in one are the same string. The only place
the difference is visible is the **source**, where a literal is a literal. So
this module reads text:

* every shipped Python module;
* every frontend source file that is not a test;
* and the **built bundle**, when one exists -- which is what D8 explicitly
  asks for, "since the two worst instances above are in TSX and the current
  gate only inspects the backend". The two worst instances were the Models
  screen's install-from-a-folder placeholder and the help text beside it, which
  showed a macOS user this build machine's drive letter as guidance.

The bundle half is what catches a path that is *composed* at build time from
something the source does not obviously contain -- an inlined config, a
sourcemap, a vendored asset. It is skipped, loudly, when no bundle has been
built in this checkout; the release gate (``packaging/check_wheel.py``) covers
the artifact itself, and the final build is where a bundle always exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantem.registry.tests.copy_gate import (
    HARDCODED_ROOT_KIND,
    Violation,
    hardcoded_root_violations,
)

_SRC = Path(__file__).resolve().parents[2]
_APP = _SRC.parents[1]
_FRONTEND = _APP / "frontend"
_BUNDLE = _FRONTEND / "dist"

#: The only modules allowed to contain a machine root, and the only reason that
#: is ever true: **their subject matter is detecting machine roots**, so an
#: example path in them is the specification rather than a leak.
#:
#: Asserted to be exactly this set by
#: :func:`test_the_exemption_list_is_the_three_modules_that_detect_paths`, so it
#: cannot grow quietly. Note what is *not* here any more: ``analysis/
#: provenance.py``, ``registry/install.py``, ``segmentation/overlay_ngff/
#: failure_text.py``, ``core/settings.py`` and ``assets/upload_staging.py`` all
#: used to carry one and were rewritten platform-neutral rather than pinned.
_PATH_MACHINERY: frozenset[str] = frozenset(
    {
        "registry/release.py",
        "core/local_storage.py",
        "registry/tests/copy_gate.py",
    }
)

#: Not scanned. ``_fig3`` is vendored research code that ships but is never
#: read by a user-facing path; ``tests`` construct fake paths as fixtures on
#: purpose, and a rule that forbade that would forbid testing the detector.
_SKIP_PARTS = frozenset({"tests", "migrations", "_fig3", "__pycache__"})

#: Text-ish members of a built bundle. Binary assets cannot carry a path a
#: reader will ever see, and the release gate hashes them anyway.
_BUNDLE_SUFFIXES = frozenset({".js", ".mjs", ".css", ".html", ".map", ".json", ".svg"})


def _python_modules() -> list[Path]:
    return [
        path
        for path in sorted(_SRC.rglob("*.py"))
        if not any(part in _SKIP_PARTS for part in path.relative_to(_SRC).parts)
    ]


def _frontend_sources() -> list[Path]:
    out = []
    for pattern in ("*.ts", "*.tsx", "*.css", "*.html"):
        for path in sorted((_FRONTEND / "src").rglob(pattern)):
            name = path.name
            if ".test." in name or name.endswith(".d.ts"):
                continue
            out.append(path)
    out.append(_FRONTEND / "index.html")
    return [path for path in out if path.is_file()]


def _scan(paths, root: Path) -> list[Violation]:
    found: list[Violation] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - unreadable file is another gate's
            continue
        rel = path.relative_to(root).as_posix()
        if rel in _PATH_MACHINERY:
            continue
        for violation in hardcoded_root_violations(text, rel):
            found.append(violation)
    return found


def _report(violations: list[Violation]) -> str:
    lines = []
    for violation in violations:
        lines.append(f"  {violation.where}: {violation.match!r}")
    return "\n".join(lines)


# --- The detector, proved on the strings that shipped ------------------------


def test_it_sees_the_two_instances_that_were_on_screen():
    """D8's own table as the detector's fixtures.

    Both were rendered to every user of the Models screen, on every platform:
    the install field's placeholder, and the example in the help text under it.
    Reproduced with the folder segment written as a visible example, so that a
    file shipping in the source distribution does not itself carry something
    shaped like one machine's directory; the shape the detector must see —
    drive letter, then a path — is unchanged.
    """
    placeholder = r"e.g. D:\quantem-models-0.1.0"
    help_text = r"a folder holding head.pt, e.g. D:\example\mito_quantem"

    for text in (placeholder, help_text):
        kinds = {v.kind for v in hardcoded_root_violations(text)}
        assert kinds == {HARDCODED_ROOT_KIND}, (text, kinds)


@pytest.mark.parametrize(
    "text",
    [
        # Every shape below is written with a visibly-example segment: these
        # ship in the source distribution, and a fixture that has to *be* a
        # machine path must still not name one.
        r"D:\example\uat4_data\images\x.png",
        "D:\\\\example\\\\...",  # the doubled form a docstring holds
        "/mnt/d/quantem/weights",
        "/home/someone/quantem",
        "/Users/someone/QuantEM/data",
        "\\\\SOMEHOST\\share\\weights",
        r"C:\Users\someone\AppData\Local\Temp",
    ],
)
def test_it_sees_every_shape_the_ruling_names(text: str):
    assert hardcoded_root_violations(text), text


@pytest.mark.parametrize(
    "text",
    [
        "https://huggingface.co/ArrojoeDrigoLab/quantem",
        "GET /api/models/",
        "the folder you unzipped a QuantEM model release into",
        "<data dir>/images/x.png",
        "packs/quantem__mito/head.pt",
        "/home/web_user",  # Emscripten's fixed virtual filesystem
        "Program Files is a folder with a space in it",
        "8 nm/px",
    ],
)
def test_it_leaves_everything_else_alone(text: str):
    """False positives are how a gate gets switched off."""
    assert hardcoded_root_violations(text) == [], text


def test_a_resolved_path_in_a_response_is_still_a_response_matter():
    """D8 and D7 ask different questions, and this keeps them apart.

    ``find_violations`` is what runs over serialised bodies, and it must go on
    letting a real resolved data directory through in a field of its own --
    that is D7. This rule is never applied there, so the two cannot fight.
    """
    from quantem.registry.tests.copy_gate import KINDS, find_violations

    assert HARDCODED_ROOT_KIND not in KINDS
    assert find_violations("D:\\example\\QuantEM\\data") == []


# --- The sweep: source, frontend, bundle -------------------------------------


def test_the_scan_actually_reaches_the_files_it_claims_to():
    """A sweep that quietly found no files would pass for ever."""
    modules = _python_modules()
    frontend = _frontend_sources()

    assert len(modules) >= 150, len(modules)
    assert len(frontend) >= 100, len(frontend)
    assert any(path.suffix == ".tsx" for path in frontend)


def test_no_shipped_python_module_hardcodes_a_machine_root():
    violations = _scan(_python_modules(), _SRC)

    assert not violations, (
        f"D8: {len(violations)} hardcoded machine root(s) in shipped Python:\n"
        f"{_report(violations)}\n"
        "Build the path from core.config's roots, or write the example "
        "platform-neutral (<data dir>/...)."
    )


def test_no_frontend_source_hardcodes_a_machine_root():
    violations = _scan(_frontend_sources(), _FRONTEND)

    assert not violations, (
        f"D8: {len(violations)} hardcoded machine root(s) in frontend source:\n"
        f"{_report(violations)}\n"
        "A path shown in the UI comes from the backend's resolved roots or "
        "from what the user typed, never from a literal."
    )


def test_the_exemption_list_is_the_three_modules_that_detect_paths():
    """The allowlist shrinks as things are fixed; it does not grow.

    Every name here is a module whose *job* is recognising a machine path, so
    an example is its specification. D8 says the release gate's equivalent
    allowlist "shrinks as they are fixed rather than growing", and the same
    applies here -- a fourth name means somebody pinned a leak.
    """
    assert _PATH_MACHINERY == {
        "registry/release.py",
        "core/local_storage.py",
        "registry/tests/copy_gate.py",
    }
    for name in _PATH_MACHINERY:
        assert (_SRC / name).is_file(), name


def test_the_built_bundle_carries_no_machine_root():
    """The half D8 asked for by name, over what actually reaches a browser.

    Skipped rather than silently passed when no bundle has been built in this
    checkout: a green tick from a scan of nothing is worse than a skip, and the
    frontend build is not this test's to run.
    """
    if not _BUNDLE.is_dir():
        pytest.skip(
            "no built frontend bundle in this checkout; build it and re-run, "
            "or rely on the release gate over the wheel"
        )
    members = [
        path
        for path in sorted(_BUNDLE.rglob("*"))
        if path.is_file() and path.suffix.lower() in _BUNDLE_SUFFIXES
    ]
    assert members, "the bundle directory holds no text members"

    violations = _scan(members, _BUNDLE)

    assert not violations, (
        f"D8: {len(violations)} hardcoded machine root(s) in the built bundle:\n"
        f"{_report(violations)}"
    )

"""The failure catalogue, held closed from both ends.

A code is only worth having if the two halves cannot drift apart, and they are
in different languages in different directories, so nothing but a test can hold
them together. Two directions, and both have a failure mode that has actually
happened in this codebase:

* **Backend to frontend.** ``duplicate_image`` and ``upload_too_large`` were
  already on the wire before :mod:`quantem.core.error_codes` existed -- invented
  at their call sites, in two apps, with no shared list and no client that knew
  what to do with either. A code a client has never heard of renders as nothing,
  so a payload carrying it is no better than one carrying no code at all.
  :func:`test_every_emitted_code_is_in_the_catalogue` is what stops a third one
  being invented the same way.

* **Frontend to backend.** A code added to the enum with no copy behind it
  renders as a blank space where the explanation should be -- worse than the
  plain sentence it replaced, because the surface has already decided to make
  room for it. :func:`test_every_code_has_copy_and_an_in_app_action` reads the
  real ``failures.ts`` and fails on the missing entry.

The frontend half is read as **text**, not imported: there is no TypeScript
runtime in the Python test session, and the alternative -- generating one file
from the other -- moves the failure out of the test suite and into a build step
that nobody runs before committing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from quantem.core.error_codes import (
    ERROR_CODE_FIELD,
    ErrorCode,
    classify_exception,
    with_error_code,
)

_SRC = Path(__file__).resolve().parents[2]
_FAILURES_TS = _SRC.parents[1] / "frontend" / "src" / "shared" / "copy" / "failures.ts"

#: Not scanned for emitted codes. ``tests`` is excluded on purpose: a test may
#: legitimately post an unknown code to prove the client tolerates one, and a
#: rule that forbade it would forbid testing the tolerant path.
_SKIP_PARTS = {"tests", "migrations", "_fig3", "__pycache__"}


# --- Backend: nothing emits a code that is not catalogued --------------------


def _module_paths() -> list[Path]:
    return [
        path
        for path in sorted(_SRC.rglob("*.py"))
        if not any(part in _SKIP_PARTS for part in path.relative_to(_SRC).parts)
    ]


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings, for one-hop resolution.

    ``DUPLICATE_IMPORT_ERROR_CODE = "duplicate_image"`` is assigned once and
    used by name, which is the right way to write it and invisible to a scanner
    that only looks at literals. One hop is enough for every emit site here and
    stops well short of reimplementing the interpreter.
    """
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = node.value.value
    return out


def _resolved(node: ast.AST, constants: dict[str, str]) -> str | None:
    """The string this expression puts on the wire, when that is knowable.

    Three shapes, and nothing else: a literal, a module constant by name, and
    ``ErrorCode.MEMBER`` (which is a :class:`~enum.StrEnum`, so it serialises as
    its value). Anything else -- a call, a variable from another module, a
    conditional -- returns ``None`` and is not judged, because guessing at it
    would produce a failure nobody can act on.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "ErrorCode":
            member = getattr(ErrorCode, node.attr, None)
            # An unknown member is a NameError at run time, not this test's
            # business; report the attribute so the message is still useful.
            return str(member) if member is not None else f"ErrorCode.{node.attr}"
    return None


def _emitted_codes() -> list[tuple[str, str]]:
    """``(where, code)`` for every error code written into ``src/quantem``."""
    found: list[tuple[str, str]] = []
    for path in _module_paths():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken tree is another gate's
            continue
        constants = _module_constants(tree)
        rel = path.relative_to(_SRC).as_posix()
        for node in ast.walk(tree):
            value: ast.AST | None = None
            if isinstance(node, ast.Dict):
                for key, item in zip(node.keys, node.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == ERROR_CODE_FIELD:
                        resolved = _resolved(item, constants)
                        if resolved is not None:
                            found.append((f"{rel}:{node.lineno}", resolved))
                continue
            if isinstance(node, ast.keyword) and node.arg == ERROR_CODE_FIELD:
                value = node.value
            elif isinstance(node, ast.Assign) and any(
                getattr(target, "attr", getattr(target, "id", None)) == ERROR_CODE_FIELD
                for target in node.targets
            ):
                value = node.value
            if value is None:
                continue
            resolved = _resolved(value, constants)
            if resolved is not None:
                found.append((f"{rel}:{getattr(node, 'lineno', 0)}", resolved))
    return found


def test_the_scanner_finds_the_codes_that_were_already_on_the_wire():
    """A scanner that quietly found nothing would pass this file for ever.

    Both of these predate the catalogue and are written in different shapes --
    one through a module constant, one as a literal in a dict -- so between
    them they exercise both resolution paths.
    """
    emitted = {code: where for where, code in _emitted_codes()}

    assert "duplicate_image" in emitted, sorted(emitted)
    assert "upload_too_large" in emitted, sorted(emitted)


def test_every_emitted_code_is_in_the_catalogue():
    known = {str(code) for code in ErrorCode}
    unknown = [(where, code) for where, code in _emitted_codes() if code not in known]

    if unknown:
        report = "\n".join(f"  {where}: {code!r}" for where, code in unknown)
        raise AssertionError(
            "error codes emitted that are not in quantem.core.error_codes."
            f"ErrorCode:\n{report}\n"
            "Add the code to the enum (and its copy to failures.ts), or emit "
            "one that is already there."
        )


# --- Frontend: every code has copy, and the copy has a way forward -----------


def _ts_source() -> str:
    assert _FAILURES_TS.is_file(), f"no copy file at {_FAILURES_TS}"
    return _FAILURES_TS.read_text(encoding="utf-8")


def _ts_listed_codes(source: str) -> list[str]:
    block = re.search(r"export const FAILURE_CODES = \[(.*?)\] as const;", source, re.S)
    assert block, "FAILURE_CODES is not where this test expects it"
    return re.findall(r'"([a-z0-9_]+)"', block.group(1))


def _ts_copy_entries(source: str) -> dict[str, str]:
    """``{code: the body of its entry}`` from ``FAILURE_COPY``.

    Split on top-level keys rather than parsed: the entries are flat object
    literals and the only question asked of each is whether it has the three
    required fields, which a substring answers honestly.
    """
    block = re.search(
        r"export const FAILURE_COPY: Record<FailureCode, FailureCopy> = \{(.*?)\n\};",
        source,
        re.S,
    )
    assert block, "FAILURE_COPY is not where this test expects it"
    body = block.group(1)
    keys = [
        (match.group(1), match.start()) for match in re.finditer(r"^  ([a-z0-9_]+): \{", body, re.M)
    ]
    entries: dict[str, str] = {}
    for index, (name, start) in enumerate(keys):
        end = keys[index + 1][1] if index + 1 < len(keys) else len(body)
        entries[name] = body[start:end]
    return entries


def test_the_two_lists_of_codes_are_the_same_list():
    listed = _ts_listed_codes(_ts_source())

    assert sorted(listed) == sorted(str(code) for code in ErrorCode), (
        sorted(listed),
        sorted(str(code) for code in ErrorCode),
    )


@pytest.mark.parametrize("code", [str(c) for c in ErrorCode])
def test_every_code_has_copy_and_an_in_app_action(code: str):
    entry = _ts_copy_entries(_ts_source()).get(code)

    assert entry, f"no FAILURE_COPY entry for {code!r}"
    for field in ("headline:", "body:", "action:"):
        assert field in entry, f"{code!r} has no {field.rstrip(':')}"
    # An action is a control in this application: a hash route, or a named
    # control the surface renders. A code whose "action" is advice to go and
    # read something is the failure this whole package exists to remove.
    assert 'href: "#/' in entry or "control:" in entry, f"{code!r} has no in-app action"


def test_no_copy_string_tells_the_user_to_type_a_command():
    """I-12, over the one file in the frontend that is nothing but copy.

    The registry's install hint reached three screens with
    ``quantem models install`` in it, and the reason it got that far is that
    nothing read the strings. This file is read.
    """
    from quantem.registry.tests.copy_gate import find_violations

    source = _ts_source()
    sentences = re.findall(r'^\s+"([^"]{20,})",?$', source, re.M)
    assert len(sentences) >= len(ErrorCode), len(sentences)

    violations = [
        violation
        for sentence in sentences
        for violation in find_violations(sentence, "failures.ts")
    ]
    assert violations == [], "\n".join(str(v) for v in violations)


# --- The classifier ----------------------------------------------------------


class ModelWeightsNotInstalled(Exception):
    """Named to match the model layer's own class, which is how it is matched."""


def test_it_classifies_the_failures_it_claims_to():
    assert classify_exception(ModelWeightsNotInstalled("no pack")) is ErrorCode.MODEL_NOT_INSTALLED
    assert classify_exception(MemoryError()) is ErrorCode.OUT_OF_MEMORY
    assert classify_exception(OSError(28, "No space left on device")) is ErrorCode.DISK_FULL


def test_an_ordinary_bug_gets_no_code():
    """``None`` is a real answer, and the common one.

    A bug has no remedy to offer, and inventing a code for it would put a
    button on screen that cannot fix anything.
    """
    assert classify_exception(KeyError("asset_id")) is None
    assert classify_exception(ValueError("polygon has 2 points")) is None
    # And specifically: no guessing from message text.
    assert classify_exception(RuntimeError("out of memory somewhere")) is None


def test_a_missing_code_does_not_put_a_null_on_the_wire():
    """A client checking for the key must not find one holding ``None``."""
    assert with_error_code({"error": "…"}, None) == {"error": "…"}
    assert with_error_code({"error": "…"}, ErrorCode.DISK_FULL) == {
        "error": "…",
        "error_code": "disk_full",
    }


def test_a_code_serialises_as_its_own_string():
    """``StrEnum``, so a serialiser needs no special case for it."""
    import json

    assert json.dumps({"error_code": ErrorCode.CANCELLED}) == ('{"error_code": "cancelled"}')

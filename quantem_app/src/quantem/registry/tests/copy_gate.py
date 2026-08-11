"""Invariant I-12's detector: is this string safe to put in front of a clicker?

I-12: **no user-facing string may contain a shell command, a module path, an
HTTP verb, an API endpoint, an internal task or model name, a Python exception
class, a raw UUID, or an absolute filesystem path.**

Its acceptance used to be "a grep over the built bundle", and that acceptance
was blind: on 2026-08-10 a verifier ran ``grep -r "quantem models install"
frontend/dist`` and got zero hits **while the string was on screen in three
places** -- the create-run dialog, the labeling header after a failed run, and
the viewer's overlay card. None of them had it baked into the JavaScript. All
three had fetched it from the backend at run time, out of
``quantem.registry.cache.INSTALL_HINT`` and ``INSTALL_INSTRUCTIONS``.

So the gate has to look where the copy actually comes from: **the strings the
API can serialise**. This module is the predicate half of it. Two test modules
apply it: ``test_i12_no_cli_in_served_copy.py`` (the registry's own surfaces,
where the first breach was found) and ``test_i12_error_copy_sweep.py`` (every
app's serialised error paths, static and live).

The original four defect classes were all present in the one string that
shipped::

    quantem:er cannot run on this machine. Not installed yet. Install it from
    the Models screen or with `quantem models install <pack id, e.g.
    quantem:mito>` -- QuantEM downloads and verifies it from Hugging Face. ...

``shell-command`` (a command a biologist will never type), ``module-path``
(``python -m quantem.registry.install``, which is worse -- it is a command that
looks like an internal), ``double-hyphen`` (a literal ``--`` where the rest of
the product uses an em dash, and which in this context reads as a CLI flag), and
``placeholder`` (``<pack id, e.g. quantem:mito>``: there is nowhere in the
application to type it). ``api-endpoint`` was found by the same sweep.

Wave 0d added the five that the same invariant names and that the gate could not
see, each one taken from a string a verifier read on screen::

    This segmentation cannot be deleted while a run_segmentation_full_task job
    is running on it (job 04a18666-11de-4c39-8fd2-c25a67b7d6c9). Cancel it
    (POST /api/jobs/04a18666-.../cancel/) and delete again once it has stopped.

``internal-name`` (``run_segmentation_full_task`` is a row in a table, not a
thing on any screen), ``raw-uuid``, ``http-verb``, and -- from two other
surfaces -- ``exception-class`` (``failed: ValueError: ...`` in the Tasks
drawer) and ``absolute-path`` (``cannot identify image file
'D:\\...\\data\\tmp\\uploads\\503fd362-....png'`` from an import).

**Prose versus datum.** Five of those classes are legitimate as *data*: a
``job_id`` field carrying a UUID, an ``unlock`` block carrying ``{"method":
"DELETE", "path": "/api/segmentations/<id>/complete"}``, a ``job_type`` field
carrying the row's own type, a ``data_dir`` field carrying a real folder. What
is forbidden is putting them **in a sentence**, where they are the app talking
to a person. So those rules fire only when the string is more than the datum
itself -- see :func:`_is_datum_only`. The test is a property of the value, not
an allow-list of field names: you cannot exempt a sentence by renaming its key,
and the only way to make a violation disappear is to take the English out, at
which point it is no longer copy.

The predicate is deliberately conservative about *English*. ``QuantEM`` the
product is spelled with capitals and ``quantem`` the console script is not, so
the command rules are case-sensitive; prose like "QuantEM downloads it" is not a
command and must not be flagged, or the gate gets switched off.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

__all__ = [
    "COMMAND_WORDS",
    "DATUM_KINDS",
    "HARDCODED_ROOT_KIND",
    "KINDS",
    "Violation",
    "find_violations",
    "hardcoded_root_violations",
    "interpolation_violations",
    "is_clean",
    "walk_strings",
]

#: Executables the application must never tell a desktop user to run. Lowercase
#: on purpose: ``quantem`` is the console script, ``QuantEM`` is the product.
COMMAND_WORDS: tuple[str, ...] = (
    "quantem",
    "python",
    "python3",
    "pip",
    "pip3",
    "conda",
    "npm",
    "npx",
    "node",
    "yarn",
    "pnpm",
    "django-admin",
    "manage.py",
    "pytest",
    "ruff",
    "powershell",
    "docker",
    "git",
    "curl",
    "wget",
    "bash",
    "sh",
    "cmd",
)

_CMD = "|".join(re.escape(word) for word in COMMAND_WORDS)

#: ``python -m package.module`` in any surrounding text. The single most common
#: shape of "an instruction written for the person who builds the release".
_MODULE_INVOCATION = re.compile(rf"\b(?:{_CMD})\s+-m\s+[\w.]+")

#: A command inside backticks or quotes -- how prose embeds one.
_QUOTED_COMMAND = re.compile(rf"[`'\"]\s*(?:{_CMD})\s+\S")

#: A whole line that *is* a command, which is how a terminal transcript or an
#: indented example reads.
_COMMAND_LINE = re.compile(rf"^[ \t>$]*(?:{_CMD})\s+\S", re.MULTILINE)

#: Fixed pairs that are unambiguous wherever they appear.
_COMMAND_PHRASE = re.compile(
    r"\b(?:pip3?\s+install|conda\s+install|npm\s+(?:run|install|ci)|npx\s+\S"
    r"|quantem\s+(?:models|serve|run)\b)"
)

#: An internal dotted path. ``quantem:mito`` (a pack id, which the user does see
#: and can copy from the Models screen) is not one -- the separator is a colon.
_MODULE_PATH = re.compile(r"\bquantem(?:\.[a-z_][A-Za-z0-9_]*)+")

#: A literal ``--`` used as punctuation, or as a CLI flag.
_DOUBLE_HYPHEN_PUNCT = re.compile(r"(?:\s--\s|\s--$|^--\s)")
_CLI_FLAG = re.compile(r"(?<![\w-])--[A-Za-z][\w-]*")

#: ``<something the user is meant to substitute>``. There is nowhere in a
#: point-and-click application to substitute anything.
_PLACEHOLDER = re.compile(r"<[^<>\n]{1,120}>")

#: An HTTP route. The client already knows its own API; a human does not.
_API_ENDPOINT = re.compile(r"(?:\b(?:GET|POST|PUT|PATCH|DELETE)\s+/|/api/)")

#: An HTTP method on its own. ``OPTIONS`` and ``HEAD`` are ordinary English
#: words in caps and are left out rather than risk a false positive that gets
#: the gate switched off; the five that matter here are not words.
_HTTP_VERB = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\b")

#: A canonical UUID. Job ids, segmentation ids and asset ids are all of them.
_RAW_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

#: An absolute filesystem path: a drive letter, a UNC share, or a POSIX root
#: that is actually a root and not the start of an API route. Stops at quotes
#: and brackets so a path quoted inside a sentence is matched without eating the
#: rest of the sentence. The lookbehind is what keeps ``https://`` out of it --
#: without it the ``s:/`` in every URL is a drive letter.
_DRIVE_OR_ROOT = (
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]"
    r"|\\\\[^\\/\s]+[\\/]"
    r"|/(?:home|Users|usr|var|opt|etc|mnt|media|tmp|private|Applications)/)"
)
_ABSOLUTE_PATH = re.compile(_DRIVE_OR_ROOT + r"[^\s'\"()\[\]<>|]*")

#: The whole string is one path and nothing else -- including the folder names
#: with spaces in them that Windows is full of.
_BARE_PATH = re.compile(_DRIVE_OR_ROOT + r"[^\n]*")


def _job_type_vocabulary() -> tuple[str, ...]:
    """Every job type this build knows, live rather than copied.

    ``quantem.jobs.constants`` is a flat module of literals with no Django in
    it, so this stays importable from a ``SimpleTestCase``. Reading the real
    vocabulary means a job type added tomorrow is covered the day it is added;
    a hand-kept list here would be one release behind for ever. The fallback
    keeps the detector usable if the constants ever grow an import.
    """
    try:
        from quantem.jobs.constants import (  # noqa: PLC0415
            ALLOWED_JOB_TYPES,
            LEGACY_JOB_TYPES,
        )
    except Exception:  # pragma: no cover - only if jobs.constants gains imports
        return ()
    return tuple(sorted(set(ALLOWED_JOB_TYPES) | set(LEGACY_JOB_TYPES), key=len, reverse=True))


#: An internal *task* name: a job type by name, or anything shaped like one.
#: The vocabulary is read from the queue's own constants; the suffix rule
#: catches an internal that has not been registered as a job type yet.
#:
#: Deliberately **not** "any snake_case token": ``polygon_coords must include
#: at least 3 points`` is a serializer naming its own field, which is a
#: different and much smaller sin than telling a biologist about
#: ``run_segmentation_full_task``, and a rule that flagged both would be turned
#: off within a week.
_INTERNAL_TASK = re.compile(
    r"(?<![\w.])(?:"
    + "|".join(
        [
            *(re.escape(t) for t in _job_type_vocabulary()),
            r"[a-z][a-z0-9_]*_(?:task|tasks|job|jobs|handler|worker|pipeline)",
        ]
    )
    + r")(?![\w])"
)

#: An internal *model* name: the encoder architectures and checkpoint families
#: this build can load. Pack ids (``quantem:mito``) are not here -- those are on
#: the Models screen and the user can read them back to you.
_INTERNAL_MODEL = re.compile(
    r"\b(?:dinov[23]|vit[_-]?[bslh]\d*|resnet\d+|convnext|efficientnet|unet|sam2?"
    r"|safetensors|torchscript)\b"
)

#: A Python exception class. Three shapes: anything ending ``Error``/
#: ``Exception``, the four builtins that end in neither, and the
#: ``ClassName: message`` prefix that ``f"{type(exc).__name__}: {exc}"``
#: produces -- which is how ``ModelWeightsNotInstalled:`` reached the Tasks
#: drawer with no ``Error`` in its name to give it away.
_EXCEPTION_CLASS = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception)\b"
    r"|\b(?:StopIteration|KeyboardInterrupt|SystemExit|GeneratorExit)\b"
    r"|\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b(?=:\s)"
)

#: Two adjacent lowercase words: the cheapest reliable sign that a string is a
#: sentence rather than a value. ``Program Files`` does not match (both words
#: are capitalised), ``run_segmentation_full_task`` does not match (no space).
_SENTENCE = re.compile(r"[a-z]{2,}\s+[a-z]{2,}")


class Violation(NamedTuple):
    """One defect in one string.

    ``where`` is a caller-supplied label (a JSON path, a function name) so a
    failure names the surface, not just the sentence.
    """

    where: str
    kind: str
    match: str
    text: str

    def __str__(self) -> str:  # pragma: no cover - failure formatting
        return f"{self.where}: {self.kind} {self.match!r}\n    in: {self.text!r}"


_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("shell-command", _MODULE_INVOCATION),
    ("shell-command", _QUOTED_COMMAND),
    ("shell-command", _COMMAND_LINE),
    ("shell-command", _COMMAND_PHRASE),
    ("module-path", _MODULE_PATH),
    ("double-hyphen", _DOUBLE_HYPHEN_PUNCT),
    ("cli-flag", _CLI_FLAG),
    ("placeholder", _PLACEHOLDER),
    ("api-endpoint", _API_ENDPOINT),
    ("http-verb", _HTTP_VERB),
    ("internal-name", _INTERNAL_TASK),
    ("internal-name", _INTERNAL_MODEL),
    ("exception-class", _EXCEPTION_CLASS),
    ("raw-uuid", _RAW_UUID),
    ("absolute-path", _ABSOLUTE_PATH),
)

#: Every kind the detector can report, so a caller can assert it is checking
#: for all of them rather than silently checking for four.
KINDS: frozenset[str] = frozenset(kind for kind, _ in _RULES)

#: The kinds that are a defect only *in a sentence*. A payload is allowed to
#: carry the datum itself in a field of its own: that is how the client learns
#: which job to cancel or which route unlocks a segmentation. See the module
#: docstring for why this is a property of the value and not a list of keys.
DATUM_KINDS: frozenset[str] = frozenset(
    {"api-endpoint", "http-verb", "internal-name", "raw-uuid", "absolute-path"}
)


def _is_datum_only(text: str, match: str) -> bool:
    """True when ``text`` is the datum ``match`` and not a sentence containing it.

    Three ways to be a datum, in decreasing order of how obvious it is:

    * the whole string *is* the match -- ``"DELETE"``, a lone job id;
    * the string has no English in it at all and is short -- a route, an
      identifier, a drive path with one space in a folder name;
    * the whole string is one absolute path, which is the one case where a
      value routinely contains several spaces (``C:\\Program Files\\...``).
    """
    stripped = text.strip()
    if stripped == match.strip():
        return True
    if _SENTENCE.search(text):
        return False
    return len(stripped.split()) <= 3 or bool(_BARE_PATH.fullmatch(stripped))


def _redact(text: str, user_supplied: Sequence[str]) -> str:
    """Blank out values that came in on the request, longest first."""
    for value in sorted((str(v) for v in user_supplied if v), key=len, reverse=True):
        for variant in {value, value.replace("\\", "/"), value.replace("/", "\\")}:
            text = text.replace(variant, "(what you chose)")
    return text


def find_violations(
    text: str,
    where: str = "",
    *,
    user_supplied: Sequence[str] = (),
) -> list[Violation]:
    """Every I-12 defect in one string, one entry per distinct match.

    ``user_supplied`` is for values **the caller handed the application in the
    same request** -- the folder chosen in "Install from a local folder", a
    filename typed into the import form. Quoting one of those back is the app
    answering the question that was asked; it is not the app leaking an
    internal. It is the only reason an absolute path may appear in a sentence,
    and the caller has to name the value, so it cannot be used to wave a path
    the application composed itself through the gate.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    scanned = _redact(text, user_supplied) if user_supplied else text
    found: list[Violation] = []
    seen: set[tuple[str, str]] = set()
    for kind, pattern in _RULES:
        for match in pattern.finditer(scanned):
            hit = match.group(0).strip()
            if kind in DATUM_KINDS and _is_datum_only(scanned, hit):
                continue
            key = (kind, hit)
            if key in seen:
                continue
            seen.add(key)
            found.append(Violation(where, kind, hit, text))
    return found


def is_clean(text: str) -> bool:
    """True when ``text`` may be shown to someone who will only ever click."""
    return not find_violations(text)


#: What an f-string *interpolates* into a sentence, judged from the source
#: expression rather than from a value. Used only by the static half of the
#: sweep: at run time the real value is available and the rules above see it
#: directly.
#:
#: Only two shapes are listed, and both are unconditional:
#:
#: * ``__name__`` / ``__class__`` / ``type(...)`` -- the only reason to reach
#:   for these in a message is to print a Python class name;
#: * ``.type`` on a job -- the job type is a table value, never a noun a person
#:   has seen.
#:
#: There is deliberately **no** ``*_id`` rule here. A name is not proof of a
#: UUID (``pack_id`` is ``quantem:mito``, ``run_id`` is an encoder run's folder
#: name), and guessing produces exactly the kind of false positive that gets a
#: gate disabled. Raw UUIDs are caught by ``raw-uuid`` above, against the real
#: serialised body, which is where the truth is.
_INTERPOLATION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("exception-class", re.compile(r"__name__|__class__|\btype\s*\(")),
    ("internal-name", re.compile(r"(?:^|\.)job\.type\b|\bjob\.type\b|\bjob_type\b")),
)


def interpolation_violations(expressions: list[str], where: str = "") -> list[Violation]:
    """I-12 defects visible in what an f-string substitutes in."""
    found: list[Violation] = []
    seen: set[tuple[str, str]] = set()
    for expression in expressions:
        for kind, pattern in _INTERPOLATION_RULES:
            if not pattern.search(expression):
                continue
            key = (kind, expression)
            if key in seen:
                continue
            seen.add(key)
            found.append(Violation(where, kind, expression, "{" + expression + "}"))
    return found


# --- D8: a path that was typed rather than resolved -------------------------

#: The kind :func:`hardcoded_root_violations` reports. Deliberately **not** in
#: :data:`KINDS`, and deliberately not one of the rules :func:`find_violations`
#: applies.
#:
#: D7 and D8 ask different questions of different material, and conflating them
#: makes one of the two useless:
#:
#: * ``absolute-path`` (D7) is about a **value in a payload**: may this string,
#:   right here, be put in front of a reader? A real resolved data directory in
#:   a ``data_dir`` field passes, because on a local desktop application the
#:   user benefits from knowing which folder;
#: * ``hardcoded-root`` (D8) is about a **literal in the source**: where did
#:   this path come from? A path *typed by whoever wrote the line* is wrong
#:   however it is presented, because it names the machine the release was
#:   built on and this application runs on many others. A resolved path cannot
#:   be told from a typed one by looking at the value, so this rule is applied
#:   to source text and to the built bundle -- where every string is, by
#:   definition, a literal -- and never to a serialised response.
HARDCODED_ROOT_KIND = "hardcoded-root"

#: ``/home/web_user``: Emscripten's fixed virtual filesystem, present in any
#: wasm bundle built with it (here, the vendored blosc codec). It is a constant
#: of the toolchain and names nobody's computer.
_TOOLCHAIN_ROOTS: frozenset[str] = frozenset({"/home/web_user"})

#: A drive letter followed by a separator and something path-shaped. The
#: ``{1,4}`` on the backslash is what makes this work on all three
#: representations of the same path this codebase contains: ``D:\x`` in a plain
#: literal, ``D:\\x`` in a Python docstring or a JSON string, and ``D:\\\\x``
#: in a docstring that is itself quoting an escaped path. The lookbehind keeps
#: the ``s:/`` of every ``https://`` out.
_HARDCODED_DRIVE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\{1,4}|/)[A-Za-z0-9_.<%-]")

#: ``\\HOST\share`` -- the worst of them, because it names a machine on a
#: network rather than only a disk on one.
#:
#: Three guards, each of them a false positive this rule produced on a real
#: vite bundle before it was added. Minified JavaScript is full of doubled
#: backslashes that are not paths:
#:
#: * ``(?!u[0-9A-Fa-f]{4})`` -- ``\\u00C0-\\u024F`` is a character-class range
#:   in a regex literal, and there were fourteen of them in one bundle;
#: * a host of **at least three characters** that neither starts nor ends with
#:   punctuation -- ``\\d\\d`` and ``\\.\\d`` are regex escapes, not shares;
#: * the lookbehind -- ``<install>\\quantem-server\\...`` is a layout written
#:   with a placeholder root, and reading it as a network host would flag the
#:   very docstrings that were rewritten to *avoid* naming a machine.
_HARDCODED_UNC = re.compile(
    r"(?<![A-Za-z0-9>}\]])"
    r"\\{2,8}"
    r"(?!u[0-9A-Fa-f]{4})"
    r"[A-Za-z0-9][A-Za-z0-9._-]{1,62}[A-Za-z0-9]"
    r"\\{1,4}"
    r"[A-Za-z0-9_.<]"
)

#: ``/home/<user>/``, ``/Users/<user>/``, ``/mnt/<drive>/``, ``/root/`` -- the
#: POSIX shapes D8 names. A segment after the root is required so that the bare
#: word in prose ("under Users") is not a hit.
_HARDCODED_POSIX = re.compile(r"/(?:home|Users|users|mnt|root)/[A-Za-z0-9_.<][A-Za-z0-9_.<>-]*")


def hardcoded_root_violations(text: str, where: str = "") -> list[Violation]:
    """Every typed-in machine root in one piece of **source or bundle** text.

    Not a copy rule and not applied to responses -- see
    :data:`HARDCODED_ROOT_KIND` for why the two questions are separate.

    Conservative about what counts as a root, because the material includes
    minified vendor JavaScript: a hit has to have a real path character after
    the separator, and the toolchain's own fixed virtual filesystem is excluded
    by name rather than by pattern.
    """
    if not isinstance(text, str) or not text:
        return []
    found: list[Violation] = []
    seen: set[str] = set()
    for pattern in (_HARDCODED_DRIVE, _HARDCODED_UNC, _HARDCODED_POSIX):
        for match in pattern.finditer(text):
            hit = match.group(0)
            if any(hit.startswith(root) for root in _TOOLCHAIN_ROOTS):
                continue
            if hit in seen:
                continue
            seen.add(hit)
            found.append(Violation(where, HARDCODED_ROOT_KIND, hit, text))
    return found


def walk_strings(value: Any, where: str = "$") -> Iterator[tuple[str, str]]:
    """``(json path, string)`` for every string anywhere inside ``value``.

    Dict *keys* are skipped: a key is a wire name the user never sees, and
    flagging ``download_bytes`` for its underscore would be noise.
    """
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk_strings(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from walk_strings(item, f"{where}[{index}]")

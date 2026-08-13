"""Invariant I-12, across **every** app's serialised error paths.

I-12: no user-facing string may contain a shell command, a module path, an HTTP
verb, an API endpoint, an internal task or model name, a Python exception class,
a raw UUID, or an absolute filesystem path.

**Why a second gate module.** ``test_i12_no_cli_in_served_copy`` enumerates the
registry's surfaces, which is where the first breach was found, and its own
docstring says other apps "should be added to it as their owners reach them".
Nobody did, and I-12 then broke twice more, in two different apps, on strings a
verifier read on screen::

    This segmentation cannot be deleted while a run_segmentation_full_task job
    is running on it (job 04a18666-11de-4c39-8fd2-c25a67b7d6c9). Cancel it
    (POST /api/jobs/04a18666-.../cancel/) and delete again once it has stopped.

    failed: ValueError: Error decoding PNG to 8-bit grayscale: ...

    Error reading PNG file: cannot identify image file
    'D:\\...\\data\\tmp\\uploads\\503fd362-....png'

Three waves, three apps, three new classes. A gate that has to be extended by
hand for each app is a gate that catches the app it was written for. So this one
finds its own surfaces.

**Two halves, and the seam between them is deliberate.**

*Static* (:func:`_is_in_layer`): every module that can put a string into an
HTTP response body -- detected by what it does, not by a list of filenames: it
constructs a ``Response``/``JsonResponse``/``HttpResponse``, raises a DRF
exception, or writes one of the fields that ends up on screen (``message``,
``status_error``, ``preprocess_error``, ``last_error``, ``failure_detail``).
Plus :data:`_ALSO_COPY`, modules that compose served sentences without touching
HTTP. That list can only *add* coverage: everything the structural test finds is
already in, so adding a name is safe and removing one is not.

The static half sees strings that no test can reach -- a branch that needs a
locked file or a dead network -- but it cannot see values. It judges what an
f-string interpolates from the source expression, and only where that is
unambiguous (``{exc.__class__.__name__}``, ``{job.type}``); it deliberately does
not guess that ``{run_id}`` is a UUID, because in this codebase it is sometimes
an encoder run's folder name.

*Live* (:class:`ServedErrorBodyTests`): real requests to real error paths in
each app, with the whole serialised body walked. This half sees the values --
the actual UUID, the actual path -- and is the one that would have caught W2.

Between them: a defect is caught if it is reachable *or* if it is written down.

**Three registers the first version of this module could not see, and the one
violation each of them was already hiding** (2026-08-10, found by walking the
router and the AST from outside the gate rather than by reading it):

* *served routes it was never told about* -- the live half named its own
  surfaces, so ``Asset with id <uuid> not found`` was being served on **nine**
  of them, P4's ``/api/assets/<id>/runs/`` among them, while every test here
  passed. :class:`EveryServedRouteTests` now asks Django's resolver instead;
* *the job log* -- ``reporter.log(level, sentence)`` and the progress-detail
  callback take their sentence **positionally**, and :func:`_copy_strings` reads
  keywords, dict values, assignments and returns. A literal ``--`` had been
  rendering in Tasks & Queues for a wave. :func:`_joblog_strings`, run over
  every module, because the writers are task modules and a task module composes
  no ``Response``;
* *the module a copy slot is filled from* -- ``last_error=describe_failure(exc)``
  is in layer, ``failure_text.py`` was not, and a planted
  ``f"{type(exc).__name__}: " + strerror`` went through both halves in silence
  while the detector, shown the string, named the defect at once.
  :func:`_copy_composers` follows the slot into the function that fills it.

The lesson each of the three repeats: the predicate was never what failed.
Nothing showed it the string.
"""

from __future__ import annotations

import ast
import io
import logging
import re
from functools import lru_cache
from pathlib import Path

from django.test import TestCase, override_settings
from django.urls import get_resolver
from django.urls.resolvers import URLPattern, URLResolver

from quantem.jobs.models import Job
from quantem.registry import cache
from quantem.registry.tests.copy_gate import (
    KINDS,
    Violation,
    find_violations,
    interpolation_violations,
    walk_strings,
)

# --- The string that started this wave ---------------------------------------

#: Verbatim, from ``w0c_verify_report.md`` W2: what the delete dialog said on
#: 2026-08-10, two clicks from the viewer.
SHIPPED_DELETE_REFUSAL = (
    "This segmentation cannot be deleted while a run_segmentation_full_task job "
    "is running on it (job 04a18666-11de-4c39-8fd2-c25a67b7d6c9). Cancel it "
    "(POST /api/jobs/04a18666-11de-4c39-8fd2-c25a67b7d6c9/cancel/) and delete "
    "again once it has stopped."
)

#: W5: the Tasks drawer, with a Python type in front of the app's own sentence.
SHIPPED_JOB_MESSAGE = "failed: ValueError: Error decoding PNG to 8-bit grayscale"

#: W8: an import refusal quoting the app's private staging directory.
SHIPPED_IMPORT_ERROR = (
    "Error reading PNG file: cannot identify image file "
    "'D:\\example\\QuantEM\\data\\tmp\\uploads\\503fd362-1111-4222-8333-444444444444.png'"
)


class DetectorTests(TestCase):
    """The five classes wave 0d added, each proved against a real string."""

    def test_it_sees_every_defect_in_the_delete_refusal_that_shipped(self):
        kinds = {v.kind for v in find_violations(SHIPPED_DELETE_REFUSAL)}

        assert "internal-name" in kinds, kinds  # run_segmentation_full_task
        assert "raw-uuid" in kinds, kinds
        assert "http-verb" in kinds, kinds
        assert "api-endpoint" in kinds, kinds

    def test_it_sees_the_exception_class_in_a_job_message(self):
        kinds = {v.kind for v in find_violations(SHIPPED_JOB_MESSAGE)}

        assert "exception-class" in kinds, kinds

    def test_it_sees_the_absolute_path_in_an_import_error(self):
        kinds = {v.kind for v in find_violations(SHIPPED_IMPORT_ERROR)}

        assert "absolute-path" in kinds, kinds

    def test_a_datum_on_its_own_is_data_and_a_datum_in_a_sentence_is_copy(self):
        """The distinction the whole gate turns on.

        A payload may carry the job id, the job type and the route that unlocks
        a segmentation -- in fields. The moment one of them is inside a
        sentence it is the application talking to a person, and a person can do
        nothing with any of them.
        """
        data = [
            "04a18666-11de-4c39-8fd2-c25a67b7d6c9",
            "run_segmentation_full_task",
            "/api/segmentations/04a18666-11de-4c39-8fd2-c25a67b7d6c9/complete",
            "DELETE",
            "D:\\example\\QuantEM\\data",
            "C:\\Program Files\\QuantEM\\data",
        ]
        for value in data:
            assert find_violations(value) == [], value

        for value in data:
            sentence = f"Something went wrong with {value} and you should look at it."
            assert find_violations(sentence), sentence

    def test_it_still_leaves_ordinary_product_prose_alone(self):
        innocent = [
            "Mitochondria 62% \u00b7 531 of 858 tiles \u00b7 about 4 min",
            "quantem:mito cannot run on this machine.",
            "Nothing changed: the 41 object(s) you have already labelled here "
            "are exactly as they were.",
            "Could not reach the QuantEM model repository "
            "(https://huggingface.co/ArrojoeDrigoLab/quantem).",
            "This run found 0 objects at include level 0.50.",
            "8 nm/px",
            "Your objects are safe. This only affects the picture the viewer draws from them.",
        ]
        for text in innocent:
            assert find_violations(text) == [], text

    def test_a_path_the_user_chose_may_be_quoted_back(self):
        chosen = "D:\\example\\Downloads\\quantem-models-0.1.0"
        message = f"No MANIFEST.json in {chosen}; that is not a QuantEM model bundle."

        assert find_violations(message)
        assert find_violations(message, user_supplied=[chosen]) == []

    def test_the_rule_set_covers_every_class_the_invariant_names(self):
        """A rule deleted from the table has to fail here, not silently pass."""
        assert KINDS == {
            "shell-command",
            "module-path",
            "double-hyphen",
            "cli-flag",
            "placeholder",
            "api-endpoint",
            "http-verb",
            "internal-name",
            "exception-class",
            "raw-uuid",
            "absolute-path",
        }


# --- The static half: find the serialisation layer, then read it -------------

_SRC = Path(__file__).resolve().parents[2]

#: Never scanned. ``migrations`` and ``tests`` are not copy; ``_fig3`` is
#: vendored research code that no view touches; ``management/commands`` and
#: ``registry/release.py`` are maintainer tools whose reader *is* at a terminal
#: -- ``--weights-root`` is the right thing to say to the person building a
#: release and the wrong thing to say to anyone else.
_NOT_COPY = {"tests", "migrations", "_fig3", "__pycache__", "commands"}
_MAINTAINER_TOOLS = {"registry/release.py", "_pytest_env.py", "testing.py", "cli.py"}

#: Modules that compose served sentences without constructing a response
#: themselves. Additive only: everything :func:`_layer_modules` finds
#: structurally is already covered, so a name here can only widen the sweep.
_ALSO_COPY = {
    "analysis/loaders.py",
    "analysis/provenance.py",
    "analysis/service.py",
    "assets/pyramid_authority.py",
    "jobs/failure_reconcile.py",
    "jobs/serializers.py",
    "registry/cache.py",
    "registry/catalogue.py",
    "registry/hf.py",
    "registry/install.py",
    "seg_core/base_segmenter.py",
    "seg_core/model_errors.py",
    "segmentation/completion.py",
    "segmentation/organelle_tasks.py",
    "segmentation/overlay_ngff/manifest.py",
}

_RESPONSE_CALLS = {
    "Response",
    "JsonResponse",
    "HttpResponse",
    "HttpResponseBadRequest",
    "HttpResponseForbidden",
    "HttpResponseNotFound",
    "HttpResponseServerError",
}
_DRF_ERRORS = {
    "ValidationError",
    "ParseError",
    "APIException",
    "NotFound",
    "PermissionDenied",
    "NotAcceptable",
    "Throttled",
    "MethodNotAllowed",
    "UnsupportedMediaType",
}
#: Model fields whose value is rendered on a screen: the Tasks drawer's message,
#: the labeling header's status_error, the library card's preprocess_error.
_SCREEN_FIELDS = {
    "message",
    "status_error",
    "preprocess_error",
    "last_error",
    "failure_detail",
    "status_message",
}
#: Dict keys and keyword arguments whose value is prose.
_COPY_KEYS = _SCREEN_FIELDS | {
    "detail",
    "error",
    "reason",
    "hint",
    "headline",
    "summary",
    "caveat",
    "notice",
    "warning",
    "next_steps",
    "advice",
    "explanation",
}

#: The terminal register, by design. ``INSTALL_INSTRUCTIONS`` names commands
#: because its reader is at a prompt; the runtime gate in
#: ``test_i12_no_cli_in_served_copy`` is what proves it is never *served*.
_TERMINAL_REGISTER = frozenset(cache.TERMINAL_ONLY_COPY)


def _module_paths() -> list[Path]:
    out = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC)
        if any(part in _NOT_COPY for part in rel.parts):
            continue
        if rel.as_posix() in _MAINTAINER_TOOLS or rel.name in _MAINTAINER_TOOLS:
            continue
        out.append(path)
    return out


def _is_in_layer(rel: Path, tree: ast.Module) -> bool:
    """Can this module put a string in front of a person?"""
    posix = rel.as_posix()
    if posix in _ALSO_COPY:
        return True
    if rel.name == "views.py" or "api_views" in rel.parts or "serializer" in posix:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _RESPONSE_CALLS or name in _DRF_ERRORS:
                return True
        elif isinstance(node, ast.keyword) and node.arg in _SCREEN_FIELDS:
            return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "attr", None) in _SCREEN_FIELDS for t in targets):
                return True
    return False


def _unparsed(node: ast.AST) -> tuple[str, list[str]]:
    """One blanked slot standing in for a value the source does not spell out."""
    try:
        return "\u2026", [ast.unparse(node)]
    except Exception:  # pragma: no cover - unparse is total in 3.13
        return "\u2026", ["?"]


def _rendered(node: ast.AST) -> tuple[str, list[str]] | None:
    """``(text with each {...} blanked, the expressions that filled them)``.

    Interpolations are blanked rather than left as ``{expr}`` so that a Python
    attribute chain inside an f-string cannot be mistaken for a dotted module
    path in the prose around it.

    A ``+`` whose other half is a variable is treated the same way as an
    interpolation rather than abandoned. It used to be abandoned, and that is
    how ``f"{type(exc).__name__}: " + strerror`` -- the whole of W5, written in
    two pieces instead of one -- was invisible: the left half is a perfectly
    readable f-string, and returning ``None`` for the pair threw it away along
    with the right.
    """
    if isinstance(node, ast.Constant):
        return (node.value, []) if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        expressions: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                blank, exprs = _unparsed(value.value)
                parts.append(blank)
                expressions.extend(exprs)
        return "".join(parts), expressions
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _rendered(node.left), _rendered(node.right)
        if left is None and right is None:
            return None
        left = left or _unparsed(node.left)
        right = right or _unparsed(node.right)
        return left[0] + right[0], left[1] + right[1]
    return None


def _copy_strings(tree: ast.Module) -> list[tuple[int, str, str, list[str]]]:
    """Every string in ``tree`` that a person can end up reading."""
    found: list[tuple[int, str, str, list[str]]] = []
    seen: set[int] = set()

    def take(node, origin: str, *, min_words: int = 1) -> None:
        if node is None or id(node) in seen:
            return
        got = _rendered(node)
        if got is None or len(got[0].split()) < min_words:
            return
        seen.add(id(node))
        found.append((getattr(node, "lineno", 0), origin, got[0], got[1]))

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in _COPY_KEYS:
                    take(value, f"{key.value}=")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            raises = name.endswith(("Error", "Exception", "NotInstalled", "Denied", "NotFound"))
            if (raises or name in _DRF_ERRORS) and node.args:
                take(node.args[0], f"{name}(")
            for keyword in node.keywords:
                if keyword.arg in _COPY_KEYS:
                    take(keyword.value, f"{keyword.arg}=")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if not name:
                    continue
                tail = name.lower().rsplit("_", 1)[-1]
                if name in _SCREEN_FIELDS or tail in _COPY_KEYS | {"copy", "text", "steps"}:
                    take(node.value, f"{name} =")
                elif name.isupper() and len(name) > 3:
                    # A module-level sentence constant: LOCKED_DETAIL,
                    # INSTALL_HINT, RERUN_NEEDS_THE_OBJECTS_GONE.
                    take(node.value, f"{name} =", min_words=5)
        elif isinstance(node, ast.Return):
            take(node.value, "return", min_words=5)
    return found


# --- Register two: the job log, which never touches a Response ---------------

#: Callables whose string argument is rendered in Tasks & Queues.
#: ``reporter.log(level, sentence)`` writes a ``JobLog`` row; the progress
#: callback -- ``on_detail``, and the ``report`` alias several modules bind it
#: to -- writes the job's ``message``.
#:
#: Every one of them takes the sentence **positionally**, and
#: :func:`_copy_strings` reads dict values, keyword arguments, assignments and
#: returns. So this whole register was invisible, and a ``--`` had been sitting
#: in the slow-model-load warning for a full wave: rendered on the Tasks screen,
#: in a rule the detector has had since wave 0d.
#:
#: The register is defined by the **sink**, not by the module: a sentence handed
#: to ``reporter.log`` is copy wherever it was written, so this half of the
#: sweep runs over every module rather than only over the ones
#: :func:`_is_in_layer` admits. That is the point -- the writers are task
#: modules, and a task module composes no ``Response``.
_JOB_LOG_SINKS = frozenset(
    {"log", "on_detail", "detail", "set_detail", "add_detail", "log_line", "report"}
)

#: ``logger.log`` is a developer log whose reader is a developer. Matched on the
#: receiver rather than excluded by name, so ``self._logger.log`` is out too and
#: ``reporter.log`` stays in.
_DEVELOPER_LOG_RECEIVER = re.compile(r"log(?:ger|ging)")

#: A level (``"warning"``) and a label are not sentences. Four words is the
#: shortest thing in this register that reads as one.
_JOB_LOG_MIN_WORDS = 4


def _joblog_strings(tree: ast.Module) -> list[tuple[int, str, str, list[str]]]:
    """Every sentence this module hands to a job-log or progress-detail sink."""
    found: list[tuple[int, str, str, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
            try:
                receiver = ast.unparse(func.value)
            except Exception:  # pragma: no cover - unparse is total in 3.13
                receiver = ""
            if _DEVELOPER_LOG_RECEIVER.search(receiver.lower()):
                continue
        else:
            name = getattr(func, "id", "")
        if name not in _JOB_LOG_SINKS:
            continue
        for arg in node.args:
            got = _rendered(arg)
            if got is None or len(got[0].split()) < _JOB_LOG_MIN_WORDS:
                continue
            found.append((getattr(node, "lineno", 0), f"{name}(", got[0], got[1]))
    return found


# --- Register three: the module a copy slot was filled from ------------------
#
# ``mutations.py`` is in layer -- it writes ``last_error=`` -- but the sentence
# it writes there is ``describe_failure(exc)``, and ``failure_text.py``
# constructs no ``Response``, raises no DRF error and writes no screen field, so
# nothing ever opened it. A planted ``f"{type(exc).__name__}: " + strerror``
# went through both halves of this gate in silence while the detector, handed
# the resulting string, named the defect immediately. The predicate was never
# the problem; nothing showed it the string.
#
# So the sweep follows the slot. Where an in-layer module fills a copy slot with
# a call to something it imported from elsewhere in the tree, that function --
# and the functions it calls in its own module -- are swept too. Function-scoped
# on purpose: pulling in the whole module instead flags
# ``analysis/morphometrics.py``'s internal ``ImportError`` consistency check,
# which no user will ever see, and a gate that cries wolf gets switched off.


def _quantem_imports(rel: Path, tree: ast.Module) -> dict[str, tuple[str, str]]:
    """``{local name: (module path, original name)}`` for in-tree from-imports."""
    table: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = rel.parent
            for _ in range(node.level - 1):
                base = base.parent
            module = base / Path((node.module or "").replace(".", "/"))
        elif (node.module or "").startswith("quantem."):
            module = Path(node.module.split(".", 1)[1].replace(".", "/"))
        else:
            continue
        for alias in node.names:
            table[alias.asname or alias.name] = (module.as_posix(), alias.name)
    return table


def _copy_slot_callees(tree: ast.Module) -> list[str]:
    """Names called to produce the value of a copy slot."""
    names: list[str] = []

    def note(value: ast.AST) -> None:
        if isinstance(value, ast.Call):
            func = value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name:
                names.append(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in _COPY_KEYS:
                    note(value)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in _COPY_KEYS:
                    note(keyword.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if not name:
                    continue
                if name in _SCREEN_FIELDS or name.lower().rsplit("_", 1)[-1] in _COPY_KEYS:
                    note(node.value)
    return names


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _copy_composers(trees: dict[Path, ast.Module], in_layer: set[Path]) -> dict[Path, set[str]]:
    """``{module: functions}`` an in-layer module fills a copy slot from."""
    composers: dict[Path, set[str]] = {}
    for rel in sorted(in_layer):
        imports = _quantem_imports(rel, trees[rel])
        for local in _copy_slot_callees(trees[rel]):
            if local not in imports:
                continue
            module, original = imports[local]
            target = Path(module + ".py")
            if target not in trees or target in in_layer:
                continue
            functions = _module_functions(trees[target])
            wanted: set[str] = set()
            stack = [original]
            while stack:
                name = stack.pop()
                if name in wanted or name not in functions:
                    continue
                wanted.add(name)
                for node in ast.walk(functions[name]):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    called = (
                        func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                    )
                    if called in functions:
                        stack.append(called)
            composers.setdefault(target, set()).update(wanted)
    return composers


def _composer_strings(tree: ast.Module, names: set[str]) -> list[tuple[int, str, str, list[str]]]:
    """What the named functions hand back.

    Two words is the floor rather than five, and a string made **entirely** of
    interpolations counts however short it is: ``f"{type(exc).__name__} on
    {where}"`` renders as three characters of prose and is exactly the defect
    this register exists for.
    """
    found: list[tuple[int, str, str, list[str]]] = []
    functions = _module_functions(tree)
    for name in sorted(names):
        node = functions.get(name)
        if node is None:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Return) or child.value is None:
                continue
            got = _rendered(child.value)
            if got is None:
                continue
            text, expressions = got
            if len(text.split()) < 2 and not expressions:
                continue
            found.append((getattr(child, "lineno", 0), f"{name}() return", text, expressions))
    return found


@lru_cache(maxsize=1)
def _parsed_modules() -> dict[Path, ast.Module]:
    trees: dict[Path, ast.Module] = {}
    for path in _module_paths():
        try:
            trees[path.relative_to(_SRC)] = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken tree is another gate's
            continue
    return trees


def _in_layer_modules() -> set[Path]:
    return {rel for rel, tree in _parsed_modules().items() if _is_in_layer(rel, tree)}


def _sweep_static() -> list[tuple[str, Violation]]:
    """Every I-12 defect the served-copy registers' source can be seen to hold."""
    trees = _parsed_modules()
    in_layer = _in_layer_modules()
    composers = _copy_composers(trees, in_layer)

    out: list[tuple[str, Violation]] = []
    seen: set[tuple[str, str, str]] = set()
    for rel, tree in trees.items():
        entries: list[tuple[int, str, str, list[str]]] = []
        if rel in in_layer:
            entries += _copy_strings(tree)
        entries += _joblog_strings(tree)
        if rel in composers:
            entries += _composer_strings(tree, composers[rel])
        for line, origin, text, expressions in entries:
            if text in _TERMINAL_REGISTER:
                continue
            where = f"{rel.as_posix()}:{line} [{origin}]"
            for violation in (
                *find_violations(text, where),
                *interpolation_violations(expressions, where),
            ):
                key = (where, violation.kind, violation.match)
                if key in seen:
                    continue
                seen.add(key)
                out.append((where, violation))
    return out


def test_the_static_sweep_actually_reaches_every_app():
    """A sweep that silently stopped finding modules would pass for ever.

    Two floors. The apps, because I-12 has now broken in three different ones
    and a gate that covers seven of eight is the gate that let W2 through. And
    a module count, because the structural test keys on names --
    ``Response(``, ``ValidationError(`` -- and a rename or an import-alias
    fashion could quietly empty it while every assertion still passed.
    """
    in_layer = sorted(_in_layer_modules())
    apps = {rel.parts[0] for rel in in_layer}

    assert len(in_layer) >= 45, (len(in_layer), sorted(str(r) for r in in_layer))
    assert {
        "analysis",
        "assets",
        "core",
        "finetune",
        "jobs",
        "registry",
        "seg_core",
        "segmentation",
    } <= apps, sorted(apps)


def test_no_serialised_error_copy_holds_a_shell_command_or_an_internal():
    violations = _sweep_static()

    if violations:
        report = "\n".join(f"  {where}\n    {v.kind} {v.match!r}" for where, v in violations)
        raise AssertionError(f"I-12: {len(violations)} defect(s) in serialised copy:\n{report}")


def test_the_sweep_would_have_caught_the_string_that_shipped():
    """The static half, run over a file that contains W2's original code.

    Proves the sweep is wired to the detector rather than merely returning an
    empty list: the same machinery, pointed at the code that shipped, reports
    the internal task name and the job id it interpolated.
    """
    source = (
        "from rest_framework.response import Response\n"
        "def delete(self, request, seg_id):\n"
        "    return Response({'detail': "
        "f'This segmentation cannot be deleted while a {job.type} job is "
        "running on it (job {job.id}). Cancel it "
        "(POST /api/jobs/{job.id}/cancel/) and delete again.'})\n"
    )
    tree = ast.parse(source)
    assert _is_in_layer(Path("segmentation/api_views/segmentation.py"), tree)

    kinds: set[str] = set()
    for _line, _origin, text, expressions in _copy_strings(tree):
        kinds.update(v.kind for v in find_violations(text))
        kinds.update(v.kind for v in interpolation_violations(expressions))

    assert {"http-verb", "api-endpoint", "internal-name"} <= kinds, kinds


def test_the_job_log_register_is_being_read_and_not_merely_declared():
    """A sink set that stopped matching anything would pass every assertion.

    Two floors, both of them things that were true when the register was added:
    a sentence count, and the modules it comes from -- which are *task* modules,
    none of which the structural test would ever admit.
    """
    found = {rel.as_posix(): _joblog_strings(tree) for rel, tree in _parsed_modules().items()}
    sentences = {rel: rows for rel, rows in found.items() if rows}
    total = sum(len(rows) for rows in sentences.values())

    assert total >= 12, (total, sorted(sentences))
    assert len(sentences) >= 3, sorted(sentences)
    assert "segmentation/organelle_tasks.py" in sentences, sorted(sentences)
    assert "seg_core/db/inference.py" in sentences, sorted(sentences)


def test_the_sweep_would_have_caught_the_job_log_line_that_shipped():
    """The ``--`` that was rendered in Tasks & Queues, as the code that wrote it.

    Positional, second argument, in a module that composes no response: every
    reason the old sweep could not see it, in four lines.
    """
    source = (
        "def run(reporter):\n"
        "    reporter.log(\n"
        "        'warning',\n"
        '        f"The model\'s exported encoder was missing, so it was rebuilt "\n'
        "        f\"from raw weights (tier '{tier}') -- this is why the load was slow.\",\n"
        "    )\n"
    )
    tree = ast.parse(source)

    assert not _copy_strings(tree), "the old register was never the one at fault"

    kinds = {
        v.kind
        for _line, _origin, text, _exprs in _joblog_strings(tree)
        for v in find_violations(text)
    }

    assert "double-hyphen" in kinds, kinds


def test_a_copy_slot_filled_by_another_module_pulls_that_module_in():
    """``last_error=describe_failure(exc)`` has to reach ``failure_text.py``."""
    trees = _parsed_modules()
    composers = _copy_composers(trees, _in_layer_modules())
    followed = {rel.as_posix(): names for rel, names in composers.items()}

    overlay = followed.get("segmentation/overlay_ngff/failure_text.py")

    assert overlay is not None, sorted(followed)
    # ``describe_failure`` is the one named at the slot; ``describe_os_error``
    # is where the sentence is actually built, and is reached only because the
    # follow is transitive inside the module.
    assert {"describe_failure", "describe_os_error"} <= overlay, overlay


def test_the_sweep_would_have_caught_the_overlay_failure_text_plant():
    """The regression that went through both halves in silence.

    ``f"{type(exc).__name__}: " + strerror`` -- a class name prefixed onto the
    OS's own description, which is W5 written in two pieces. The right operand
    is a bare name, and the old ``_rendered`` abandoned the whole expression the
    moment one half of a ``+`` was not a literal.
    """
    source = (
        "def describe_os_error(exc):\n"
        "    strerror = str(getattr(exc, 'strerror', ''))\n"
        "    return f'{type(exc).__name__}: ' + strerror\n"
    )
    tree = ast.parse(source)

    rows = _composer_strings(tree, {"describe_os_error"})
    assert rows, "the concatenation was thrown away before any rule ran"

    kinds = {
        v.kind
        for _line, _origin, text, expressions in rows
        for v in (*find_violations(text), *interpolation_violations(expressions))
    }

    assert "exception-class" in kinds, kinds


# --- The live half: real requests, real bodies -------------------------------


def _assert_body_is_clean(body, surface: str, *, user_supplied=()) -> None:
    pairs = list(walk_strings(body, "$"))
    assert pairs, f"{surface} serialised no strings at all"
    violations = [
        v
        for where, text in pairs
        for v in find_violations(text, where, user_supplied=user_supplied)
    ]
    if violations:
        report = "\n".join(f"  {v}" for v in violations)
        raise AssertionError(f"I-12: {len(violations)} defect(s) in {surface}:\n{report}")


@override_settings(ALLOWED_HOSTS=["*"])
class ServedErrorBodyTests(TestCase):
    """Every app's refusals, requested for real and read string by string."""

    def setUp(self):
        from quantem.segmentation.models import ImageSegmentation
        from quantem.segmentation.type_service import get_or_create_mitochondria_type
        from quantem.testing import create_small_test_image

        self.image = create_small_test_image("i12-sweep")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )

    # -- segmentation: W2, the one that shipped -------------------------------

    def test_deleting_a_segmentation_with_a_run_in_flight(self):
        """W2. The reproduction from the verification report, as a request."""
        job = Job.enqueue(
            job_type="run_segmentation_full_task",
            payload={"segmentation_id": str(self.segmentation.id)},
        )
        Job.objects.filter(id=job.id).update(status="RUNNING")

        response = self.client.delete(f"/api/segmentations/{self.segmentation.id}/")

        assert response.status_code == 409
        body = response.json()
        _assert_body_is_clean(body, "DELETE a segmentation with a run in flight")
        # The way out is a screen and a control, and the machine-readable half
        # is still there for the client.
        assert "Tasks & Queues" in body["detail"]
        assert body["job_id"] == str(job.id)
        assert body["job_type"] == "run_segmentation_full_task"

    def test_deleting_a_segmentation_with_a_run_queued(self):
        Job.enqueue(
            job_type="run_segmentation_full_task",
            payload={"segmentation_id": str(self.segmentation.id)},
        )

        response = self.client.delete(f"/api/segmentations/{self.segmentation.id}/")

        assert response.status_code == 409
        _assert_body_is_clean(response.json(), "DELETE a segmentation with a run queued")

    def test_deleting_a_segmentation_that_is_marked_done(self):
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save(update_fields=["status_stage"])

        response = self.client.delete(f"/api/segmentations/{self.segmentation.id}/")

        assert response.status_code == 409
        _assert_body_is_clean(response.json(), "DELETE a segmentation that is done")

    def test_deleting_with_a_stale_object_count(self):
        response = self.client.delete(
            f"/api/segmentations/{self.segmentation.id}/",
            data={"acknowledged_object_count": 99},
            content_type="application/json",
        )

        assert response.status_code == 409
        _assert_body_is_clean(response.json(), "DELETE with a stale count")

    def test_deleting_with_an_object_count_that_is_not_a_number(self):
        response = self.client.delete(
            f"/api/segmentations/{self.segmentation.id}/",
            data={"acknowledged_object_count": "lots"},
            content_type="application/json",
        )

        assert response.status_code == 400
        _assert_body_is_clean(response.json(), "DELETE with an unreadable count")

    def test_marking_done_cannot_unmark(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/complete",
            data={"is_complete": False},
            content_type="application/json",
        )

        assert response.status_code == 400
        _assert_body_is_clean(response.json(), "POST complete is_complete:false")

    def test_a_segmentation_that_does_not_exist(self):
        missing = "11111111-2222-4333-8444-555555555555"

        response = self.client.get(f"/api/segmentations/{missing}/")

        assert response.status_code == 404
        _assert_body_is_clean(response.json(), "GET a segmentation that is gone")

    # -- jobs -----------------------------------------------------------------

    def test_retrying_a_running_job(self):
        job = Job.enqueue(
            job_type="run_segmentation_full_task",
            payload={"segmentation_id": str(self.segmentation.id)},
        )
        Job.objects.filter(id=job.id).update(status="RUNNING")

        response = self.client.post(f"/api/jobs/{job.id}/retry/")

        assert response.status_code == 409
        _assert_body_is_clean(response.json(), "POST retry on a running job")

    def test_removing_a_running_job_from_the_queue(self):
        job = Job.enqueue(
            job_type="run_segmentation_full_task",
            payload={"segmentation_id": str(self.segmentation.id)},
        )
        Job.objects.filter(id=job.id).update(status="RUNNING")

        response = self.client.delete(f"/api/jobs/{job.id}/")

        assert response.status_code == 409
        _assert_body_is_clean(response.json(), "DELETE a running job")

    def test_cancelling_a_job_that_is_not_running(self):
        job = Job.enqueue(
            job_type="run_segmentation_full_task",
            payload={"segmentation_id": str(self.segmentation.id)},
        )

        response = self.client.post(f"/api/jobs/{job.id}/cancel/")

        assert response.status_code == 409
        _assert_body_is_clean(response.json(), "POST cancel on a queued job")

    # -- assets: the import door ---------------------------------------------

    def test_importing_with_no_file(self):
        response = self.client.post("/api/assets/upload/", data={})

        assert response.status_code == 400
        _assert_body_is_clean(response.json(), "POST an import with no file")

    def test_importing_something_that_is_not_an_image(self):
        upload = io.BytesIO(b"this is not a PNG")
        upload.name = "notes.txt"

        response = self.client.post("/api/assets/upload/", data={"file": upload})

        assert response.status_code == 400
        _assert_body_is_clean(
            response.json(),
            "POST an import of the wrong kind of file",
            user_supplied=["notes.txt"],
        )

    def test_importing_with_a_pixel_size_that_is_not_a_number(self):
        upload = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        upload.name = "image.png"

        response = self.client.post(
            "/api/assets/upload/", data={"file": upload, "pixel_size_nm": "big"}
        )

        assert response.status_code == 400
        _assert_body_is_clean(
            response.json(),
            "POST an import with a bad pixel size",
            user_supplied=["image.png"],
        )

    def test_importing_a_file_that_is_named_png_but_is_not_one(self):
        """W8's door, kept shut.

        This is where the import panel used to answer::

            Error reading PNG file: cannot identify image file
            'D:\\...\\data\\tmp\\uploads\\503fd362-....png'

        -- the app's private staging folder and the id of an asset that was
        never created, straight out of Pillow, through
        ``assets/utils.py::extract_png_metadata``'s
        ``f"Error reading PNG file: {e}"``. The magic-byte check in front of it
        now answers first, in the app's own words. That check is what keeps the
        path out of sight, so this asserts on the door rather than on the
        wording behind it.
        """
        upload = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"not really a png")
        upload.name = "micrograph.png"

        response = self.client.post("/api/assets/upload/", data={"file": upload})

        assert response.status_code == 400
        _assert_body_is_clean(
            response.json(),
            "POST an import of a file that is not the image it claims to be",
            user_supplied=["micrograph.png"],
        )

    def test_importing_a_tiff_that_falls_through_to_the_library(self):
        """The same door on the TIFF side, where one branch still gets through.

        A TIFF header with nothing behind it reaches
        ``extract_tiff_metadata``'s ``except Exception`` and is answered with
        whatever ``tifffile`` said -- today "list index out of range", which is
        clean by I-12's classes and useless by any other measure. It is one
        library release away from being a message with a path in it, and the
        fix belongs in ``assets/utils.py``, which is not this module's to
        change. This holds the line that exists: no path, no class name.
        """
        import numpy as np
        import tifffile

        buffer = io.BytesIO()
        tifffile.imwrite(buffer, np.zeros((8, 8), dtype=np.uint8))
        upload = io.BytesIO(buffer.getvalue()[:8])
        upload.name = "micrograph.tif"

        response = self.client.post("/api/assets/upload/", data={"file": upload})

        assert response.status_code == 400
        _assert_body_is_clean(
            response.json(),
            "POST an import of a TIFF with nothing behind its header",
            user_supplied=["micrograph.tif"],
        )

    def test_an_import_refused_before_the_view_ran(self):
        """W13. 101 files answered with Django's own empty 400 page.

        ``text/html``, an empty ``<p>``, no JSON: the import panel could not
        even parse it, so the user got a status code.
        """
        payload = {"pixel_size_nm": "8"}
        for index in range(101):
            handle = io.BytesIO(b"x")
            handle.name = f"{index}.png"
            payload[f"file{index}"] = handle

        response = self.client.post("/api/assets/upload/", data=payload)

        assert response.status_code == 400
        assert response.headers["Content-Type"].startswith("application/json"), response.headers[
            "Content-Type"
        ]
        body = response.json()
        _assert_body_is_clean(body, "an import refused during parsing")
        assert "too many files" in body["error"]

    # -- registry, analysis, finetune ----------------------------------------

    def test_installing_a_pack_that_does_not_exist(self):
        response = self.client.post(
            "/api/models/nope:nope/install/", data={}, content_type="application/json"
        )

        assert response.status_code == 404
        _assert_body_is_clean(response.json(), "POST install of an unknown pack")

    def test_the_analysis_rollup_filtered_by_something_that_is_not_an_id(self):
        response = self.client.get("/api/analysis/groups/?segmentation=not-a-uuid")

        assert response.status_code == 400
        _assert_body_is_clean(
            response.json(),
            "GET the analysis rollup with a bad filter",
            user_supplied=["not-a-uuid"],
        )

    def test_analysis_asked_for_with_parameters_that_do_not_make_sense(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/analysis/",
            data={"replicates": "as many as it takes", "band_edges_nm": ["wide"]},
            content_type="application/json",
        )

        assert response.status_code == 400
        _assert_body_is_clean(
            response.json(),
            "POST analysis with unusable parameters",
            user_supplied=["as many as it takes", "wide"],
        )

    def test_adapting_a_segmentation_with_nothing_confirmed(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/adapt/",
            data={},
            content_type="application/json",
        )

        assert response.status_code in {400, 409}
        _assert_body_is_clean(response.json(), "POST adapt with nothing confirmed")


# --- The live half, without the hand-kept list -------------------------------
#
# Every test above names its own surface, and that is the failure mode this
# class exists to remove. On 2026-08-10 a sweep that walked Django's resolver
# instead found ``Asset with id 00000000-... not found`` being served on **nine
# routes** -- ``/api/assets/<id>/``, its preview, its NGFF thumbnail, its
# segmentations and, newest of them, P4's ``/api/assets/<id>/runs/``. The
# ``raw-uuid`` rule had existed since wave 0d and the string was three days old.
# It was missed because none of those nine were on the list, and a list is one
# merge behind for ever.
#
# So this half asks the router what exists. A route added tomorrow is probed the
# day it is added, with no edit here.

#: A well-formed v4 UUID that is not in any table. Every ``<uuid:...>`` in the
#: URL conf gets it, which is what makes the answers *refusals* -- the register
#: where I-12 has broken every single time.
_GHOST_UUID = "00000000-0000-4000-8000-000000000000"

_URL_CONVERTER = re.compile(r"<(?:([a-z]+):)?[A-Za-z_][A-Za-z0-9_]*>")

#: Filled in for a non-uuid converter. Deliberately a word rather than a number
#: so a ``str``/``slug`` segment reads like something a user could have typed.
_GHOST_VALUES = {"uuid": _GHOST_UUID, "int": "1", "": "ghost"}


def _registered_routes() -> list[str]:
    """Every pattern in the URL conf, flattened to a single path string."""
    routes: list[str] = []

    def walk(resolver: URLResolver, prefix: str = "") -> None:
        for entry in resolver.url_patterns:
            if isinstance(entry, URLResolver):
                walk(entry, prefix + str(entry.pattern))
            elif isinstance(entry, URLPattern):
                routes.append(prefix + str(entry.pattern))

    walk(get_resolver())
    return routes


def _concretise(route: str) -> str | None:
    """``/api/assets/<uuid:asset_id>/`` -> a real requestable path, or None.

    Regex patterns are skipped rather than guessed at: filling one in wrongly
    produces a 404 from the router itself, which proves nothing about the copy
    the view would have served.
    """
    if "(?P" in route or "\\" in route:
        return None
    filled = _URL_CONVERTER.sub(lambda m: _GHOST_VALUES.get(m.group(1) or "", "ghost"), route)
    if "<" in filled or ">" in filled:
        return None
    return "/" + filled.lstrip("/")


@override_settings(ALLOWED_HOSTS=["*"])
class EveryServedRouteTests(TestCase):
    """Walk the router, ask each route about something that does not exist."""

    #: 60 paths and 240 probes when this was written. The floors are a little
    #: under that: they are here to catch the sweep silently finding *nothing*
    #: -- a renamed resolver helper, an app dropped from the URL conf -- not to
    #: be re-tuned every time a route lands.
    MINIMUM_PATHS = 55
    MINIMUM_PROBES = 200
    MINIMUM_BODIES = 40

    def test_no_route_answers_a_ghost_id_with_an_internal(self):
        paths: list[str] = []
        for route in _registered_routes():
            url = _concretise(route)
            if url and url.startswith("/api/") and url not in paths:
                paths.append(url)

        probed = 0
        bodies = 0
        violations: list[Violation] = []
        exploded: list[str] = []

        # Every probe is meant to be refused, and ``django.request`` logs a
        # WARNING for each one. Left on, a failure here arrives underneath 240
        # lines of "Not Found", which is the fastest way to make a report
        # unreadable.
        request_log = logging.getLogger("django.request")
        self.addCleanup(request_log.setLevel, request_log.level)
        request_log.setLevel(logging.CRITICAL)

        for url in paths:
            for method in ("get", "post", "patch", "delete"):
                probed += 1
                try:
                    if method in ("get", "delete"):
                        response = getattr(self.client, method)(url)
                    else:
                        response = getattr(self.client, method)(
                            url, data="{}", content_type="application/json"
                        )
                except Exception as exc:  # noqa: BLE001 - the surface is the point
                    exploded.append(f"{method.upper()} {url}: {type(exc).__name__}: {exc}")
                    continue
                # 405 is the framework answering before the app does, in DRF's
                # own words (``Method "POST" not allowed.``). It is not this
                # project's copy and no screen can reach it: the client only
                # ever sends methods its own API defines.
                if response.status_code == 405:
                    continue
                if "json" not in response.headers.get("Content-Type", ""):
                    continue
                try:
                    body = response.json()
                except ValueError:  # pragma: no cover - a lying content type
                    continue
                strings = list(walk_strings(body, "$"))
                if strings:
                    bodies += 1
                surface = f"{method.upper()} {url}"
                violations += [
                    v
                    for where, text in strings
                    for v in find_violations(text, f"{surface} {where}")
                ]

        assert len(paths) >= self.MINIMUM_PATHS, (len(paths), paths)
        assert probed >= self.MINIMUM_PROBES, probed
        assert bodies >= self.MINIMUM_BODIES, bodies
        # The route whose brand-new copy this sweep found. Named so that a
        # future URL-conf refactor that stops flattening nested includes fails
        # loudly instead of quietly probing a smaller API.
        assert f"/api/assets/{_GHOST_UUID}/runs/" in paths, paths

        if exploded:
            raise AssertionError(
                "a route raised on an id that does not exist, so the user gets a "
                "debug page rather than a sentence:\n  " + "\n  ".join(exploded)
            )
        if violations:
            report = "\n".join(f"  {v}" for v in violations)
            raise AssertionError(
                f"I-12: {len(violations)} defect(s) served on a ghost id:\n{report}"
            )

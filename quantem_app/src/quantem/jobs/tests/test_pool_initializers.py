"""Invariant I-9, the half that is checkable by reading: every process pool
in ``src/quantem`` initialises Django in its children.

Why a *source* test and not only a behavioural one. The failure this guards is
invisible until a pool is exercised at the scale that turns it on -- overlay
rasterisation only fans out above 2 000 objects, which no fixture image in this
suite had -- and when it does fire, the parent is handed
``BrokenProcessPool: A process in the process pool was terminated abruptly``,
which names neither the module that failed to import nor the reason. The
shipped product could not draw an overlay for any real EM image for exactly
this reason, and every green test run agreed it was fine.

So the rule is enforced where it is cheap and total: over the syntax. A new
``ProcessPoolExecutor`` anywhere in the package fails this test until it passes
an initializer, whether or not anybody writes a test that runs it.

The three tests below are: the source rule, a real pool proving the initializer
works, and a real pool *without* one proving the rule is guarding something.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import quantem
from quantem.jobs.pool import (
    WORKER_PROCESS_ENV_VAR,
    django_pool_initializer,
    pool_child_report,
)

#: Callables that satisfy the rule. Anything else named as an ``initializer``
#: is a failure: an initializer that does not call ``django.setup()`` is
#: indistinguishable, from the parent, from no initializer at all.
APPROVED_INITIALIZERS = frozenset({"django_pool_initializer"})

#: Constructors that spawn a fresh interpreter and therefore have to re-do
#: Django. ``ThreadPoolExecutor`` is deliberately absent -- its workers share
#: this interpreter's app registry and need nothing.
POOL_CONSTRUCTORS = frozenset({"ProcessPoolExecutor", "Pool"})

#: Bases whose ``.Pool(...)`` is a process pool.
MULTIPROCESSING_NAMESPACES = frozenset({"mp", "multiprocessing", "ctx"})

#: An explicit, greppable opt-out for a call that must not have an initializer
#: -- there is exactly one, the negative control in this file. Put it on the
#: construction line with a reason beside it.
EXEMPTION_MARKER = "pool-initializer-exempt"

SRC_ROOT = Path(quantem.__file__).resolve().parent


def _source_files() -> list[Path]:
    return [
        path
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def _callee_name(node: ast.Call) -> str | None:
    """The bare constructor name, whether called as ``X()`` or ``a.b.X()``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if func.attr == "Pool":
            base = func.value
            base_name = base.id if isinstance(base, ast.Name) else None
            if base_name not in MULTIPROCESSING_NAMESPACES:
                return None
        return func.attr
    return None


def _initializer_name(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg != "initializer":
            continue
        value = keyword.value
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            return value.attr
        return ast.dump(value)
    return None


def _is_exempt(lines: list[str], node: ast.Call) -> bool:
    start = max(0, node.lineno - 1)
    end = min(len(lines), (node.end_lineno or node.lineno))
    return any(EXEMPTION_MARKER in line for line in lines[start:end])


@lru_cache(maxsize=1)
def _process_pool_constructions() -> tuple[tuple[Path, int, str | None], ...]:
    """``(path, line, initializer name or None)`` for every process pool."""
    found: list[tuple[Path, int, str | None]] = []
    for path in _source_files():
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _callee_name(node)
            if name not in POOL_CONSTRUCTORS:
                continue
            if _is_exempt(lines, node):
                continue
            found.append((path, node.lineno, _initializer_name(node)))
    return tuple(found)


def test_the_scanner_finds_the_pools_it_is_supposed_to_guard():
    """A guard that matches nothing passes forever. Pin that it matches."""
    constructions = _process_pool_constructions()

    assert len(constructions) >= 2, (
        "expected at least the two overlay rasterisation pools; the AST scan "
        f"found {len(constructions)}, so it has stopped seeing them"
    )
    guarded_files = {path.name for path, _line, _init in constructions}
    assert "mutations.py" in guarded_files


def test_every_process_pool_passes_an_initializer():
    missing = [
        f"{path.relative_to(SRC_ROOT.parent)}:{line}"
        for path, line, initializer in _process_pool_constructions()
        if initializer is None
    ]

    assert not missing, (
        "these process pools spawn children with no django.setup(), so the "
        "child dies while importing the module holding its own task and the "
        "parent sees only BrokenProcessPool: "
        + ", ".join(missing)
        + f". Pass initializer=django_pool_initializer (quantem.jobs.pool), or "
        f"mark the line '{EXEMPTION_MARKER}' with a reason."
    )


def test_every_process_pool_initializer_actually_sets_django_up():
    wrong = [
        f"{path.relative_to(SRC_ROOT.parent)}:{line} uses {initializer}"
        for path, line, initializer in _process_pool_constructions()
        if initializer is not None and initializer not in APPROVED_INITIALIZERS
    ]

    assert not wrong, (
        "an initializer that does not call django.setup() leaves the child in "
        "exactly the broken state the rule exists to prevent: "
        + ", ".join(wrong)
        + f". Approved: {sorted(APPROVED_INITIALIZERS)}."
    )


def test_worker_marker_matches_the_runner():
    """``jobs.pool`` re-declares the runner's marker; keep the two identical.

    It cannot import it: ``jobs.runner`` reaches Django models at import time,
    and a pool child runs this module *before* ``django.setup()``.
    """
    from quantem.jobs.runner import WORKER_PROCESS_ENV_VAR as runner_marker

    assert WORKER_PROCESS_ENV_VAR == runner_marker


def _tiny_raster_payload() -> dict[str, object]:
    return {
        "region": (0, 0, 4, 4),
        "interior": (0, 0, 4, 4),
        "draw_ops": [],
        "border_width": 1,
    }


@pytest.mark.django_db
def test_a_pool_child_with_the_initializer_can_import_an_app_module():
    """The real callable, in a real spawned child, with the real initializer.

    ``rasterize_tile_worker`` is the exact function the overlay rasteriser
    submits; unpickling it in the child imports
    ``quantem.segmentation.overlay_ngff.render``, whose package ``__init__``
    reaches Django models. That import is the thing that used to kill every
    worker.
    """
    from quantem.segmentation.overlay_ngff import render

    with ProcessPoolExecutor(
        max_workers=1,
        initializer=django_pool_initializer,
    ) as executor:
        interior_x0, interior_y0, labels, border = executor.submit(
            render.rasterize_tile_worker, _tiny_raster_payload()
        ).result(timeout=120)
        report = executor.submit(pool_child_report).result(timeout=120)

    assert (interior_x0, interior_y0) == (0, 0)
    assert labels.shape == (4, 4)
    assert not np.any(labels)
    assert border.shape == (4, 4)
    assert report["apps_ready"] is True
    assert report["worker_marker"] == "1"
    assert report["settings_module"]


#: The negative control, run in an interpreter of its own.
#:
#: Two reasons it is a subprocess rather than another ``with
#: ProcessPoolExecutor(...)`` here. It has to be a *fresh* interpreter to be the
#: real thing -- the shipped job worker is one, and a pool child of this pytest
#: process inherits a ``sys.path`` and an environment that a production child
#: would not. And measurably: a pool that breaks in an interpreter where a
#: healthy pool has already run takes ~33 s to tear down on Windows (0.7 s in a
#: clean one), so in-process it would be both slower and dependent on test
#: order.
_NEGATIVE_CONTROL = textwrap.dedent(
    """
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantem.core.settings")

    import django
    django.setup()

    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from quantem.segmentation.overlay_ngff import render

    payload = {
        "region": (0, 0, 4, 4),
        "interior": (0, 0, 4, 4),
        "draw_ops": [],
        "border_width": 1,
    }

    if __name__ == "__main__":
        # No initializer: pool-initializer-exempt, this call IS the bug.
        executor = ProcessPoolExecutor(max_workers=1)
        try:
            executor.submit(render.rasterize_tile_worker, payload).result(timeout=120)
        except BrokenProcessPool as exc:
            print("BROKEN:" + str(exc))
        else:
            print("SURVIVED")
        finally:
            executor.shutdown()
    """
)


def test_a_pool_child_without_the_initializer_dies_on_that_import():
    """The negative control: the bug, reproduced on demand.

    Without this, the rule above could be guarding nothing -- it would pass
    just as happily if importing app modules in a spawned child had always been
    fine. It is not fine: this is the failure that made every real EM image
    undrawable, and it is one keyword argument away at all times.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _NEGATIVE_CONTROL],
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ.copy(),
        cwd=str(SRC_ROOT.parent),
    )

    assert completed.returncode == 0, completed.stderr
    assert "BROKEN:" in completed.stdout, (
        "a spawned pool child with no django.setup() imported an app module "
        "successfully. Either the packages stopped reaching Django models at "
        "import time -- in which case say so here and keep the rule -- or this "
        f"control has stopped testing anything.\nstdout: {completed.stdout}"
    )
    # The child's own traceback is the evidence of *why* it died, and it is the
    # sentence the parent's BrokenProcessPool never contains.
    assert "AppRegistryNotReady" in completed.stderr, completed.stderr

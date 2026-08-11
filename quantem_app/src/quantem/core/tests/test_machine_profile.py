"""The machine profile: the table, the degradations, and the import order.

Three kinds of test, because the thing being guarded fails in three different
ways.

**The table** is arithmetic and is tested as arithmetic: RAM picks a row, the
core count may only lower the derived counts, and a machine that will not say
how big it is lands on the floor rather than raising.

**The import order** is the whole point of the module and is invisible when it
breaks. ``OMP_NUM_THREADS`` set one line after ``import numpy`` behaves exactly
like ``OMP_NUM_THREADS`` set correctly -- the variable reads back the same, the
app runs, every test passes -- and the process has silently committed 1.4 GB it
did not need, which on the 8 GB laptop of owner ruling R3 is most of the
budget. So the order is enforced twice: once over the syntax, in the shape of
``jobs/tests/test_pool_initializers.py`` (a new module-level ``import numpy``
anywhere on the boot path fails the test whether or not anyone exercises it),
and once behaviourally, in a real subprocess that records ``os.environ`` at the
instant numpy is first imported.

**The number** is measured, not asserted from a comment: a child process
imports numpy, scipy and torch through the shipped path and reports its Win32
``PrivateUsage``. MEASURED on the build box (28 cores, 256 GB), committed bytes
after that import: 1 668 MB unpinned, 639 MB at 8 threads, 381 MB at 4,
252 MB at 2.

What the constrained-memory harness does and does not reproduce
---------------------------------------------------------------
The design's acceptance harness is a Windows Job Object with
``JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY``. It caps
*committed* bytes, so an allocation over the cap fails at the allocation --
``MemoryError``, or a C-level malloc returning NULL -- instead of the machine
paging. That bounds the same quantity a small machine runs out of and it names
the exact allocation that was too big. It does **not** reproduce the phase a
real 8 GB MacBook Air goes through first: memory compression, then swap, then
minutes of thrash, then a jetsam kill. A run that fits under the cap is
therefore evidence about *allocation*, not about wall time on the target.

S0 does not need the cap: the quantity under test here is the committed
baseline of a process that has done nothing yet, which is measured directly.
The cap earns its keep from S1 onward, where the failures are allocation
failures.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import queue
import socket
import subprocess
import sys
import textwrap
import threading
from functools import lru_cache
from pathlib import Path

import pytest

import quantem
from quantem.core import machine
from quantem.core.machine import (
    GIB,
    PROFILE_TABLE,
    THREAD_ENV_VARS,
    MachineProfile,
    profile_for,
)

SRC_ROOT = Path(quantem.__file__).resolve().parent
REPO_SRC = SRC_ROOT.parent
PROJECT_ROOT = REPO_SRC.parent

#: The three the design names. The module sets two more (numexpr, and Apple's
#: Accelerate) but these are the contract.
REQUIRED_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

#: BIG_IMAGE_DESIGN section 1.4(a), transcribed here independently of the
#: module so the test would notice the module's copy being edited:
#: ``(profile, heavy_slots, raster_workers, torch_threads, blas_threads, family)``.
DESIGN_TABLE = (
    ("small", 1, 2, 4, 2, "quantem"),
    ("standard", 2, 4, 8, 4, "omniem"),
    ("workstation", 4, 8, 16, 8, "omniem"),
)

#: Enough cores that no clamp fires, so the table rows come through as written.
GENEROUS_CORES = 64


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


def test_the_profile_table_is_the_design_table():
    """The module's copy of section 1.4(a), against a second copy up top."""
    transcribed = tuple(
        (
            row.name,
            row.heavy_slots,
            row.raster_workers,
            row.torch_threads,
            row.blas_threads,
            row.default_model_family,
        )
        for row in PROFILE_TABLE
    )
    assert transcribed == DESIGN_TABLE


@pytest.mark.parametrize(
    ("total_ram_bytes", "expected"),
    [
        # The two target machines of owner ruling R3.
        (8 * GIB, "small"),
        (7 * GIB + 512 * 1024 * 1024, "small"),
        # Boundaries. "< 12 GB" is small; "12-32" standard; "> 32" workstation.
        (12 * GIB - 1, "small"),
        (12 * GIB, "standard"),
        (16 * GIB, "standard"),
        (32 * GIB, "standard"),
        (32 * GIB + 1, "workstation"),
        (64 * GIB, "workstation"),
        (256 * GIB, "workstation"),
    ],
)
def test_ram_picks_the_row(total_ram_bytes, expected):
    assert profile_for(total_ram_bytes, GENEROUS_CORES).name == expected


@pytest.mark.parametrize(("name", "slots", "raster", "torch", "blas", "family"), DESIGN_TABLE)
def test_each_row_hands_out_the_designed_counts(name, slots, raster, torch, blas, family):
    ram = {"small": 8 * GIB, "standard": 16 * GIB, "workstation": 128 * GIB}[name]
    profile = profile_for(ram, GENEROUS_CORES)

    assert profile.name == name
    assert profile.heavy_slots == slots
    assert profile.raster_workers == raster
    assert profile.torch_threads == torch
    assert profile.blas_threads == blas
    assert profile.default_model_family == family


def test_the_four_core_laptop_of_ruling_r3_gets_the_designed_small_row():
    """The named target: 4 cores, 8 GB. No clamp should alter the small row."""
    profile = profile_for(8 * GIB, 4)

    assert (profile.name, profile.heavy_slots, profile.raster_workers) == ("small", 1, 2)
    assert (profile.blas_threads, profile.torch_threads) == (2, 4)
    assert profile.default_model_family == "quantem"


@pytest.mark.parametrize("cores", [1, 2, 3, 4, 8, 16])
def test_the_core_count_can_only_lower_the_counts(cores):
    """A 4-core box with 128 GB must not be told to run eight raster workers.

    Eight processes on four cores is eight copies of the per-process baseline
    contending for four cores, which is the opposite of what the workstation
    row is for. RAM chooses the row; cores clamp what the row hands out.
    """
    unclamped = profile_for(128 * GIB, GENEROUS_CORES)
    clamped = profile_for(128 * GIB, cores)

    assert clamped.name == unclamped.name == "workstation"
    for attr in ("heavy_slots", "raster_workers", "torch_threads", "blas_threads"):
        value = getattr(clamped, attr)
        assert 1 <= value <= getattr(unclamped, attr)
        assert value <= max(1, cores), f"{attr}={value} exceeds the {cores} cores available"


# --------------------------------------------------------------------------
# Degrading on a machine it cannot read
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ram", "cores"),
    [
        (None, None),          # a container that reports neither
        (None, 4),             # cgroup with no memory limit and no sysconf
        (0, 0),                # a probe that returned zeroes
        (-1, -4),              # a probe that returned nonsense
        ("8589934592", "4"),   # strings, e.g. an env override read raw
        (8.5 * GIB, 4.0),      # floats
        (None, 10**9),         # an absurd core count
    ],
)
def test_an_unreadable_machine_degrades_to_the_floor_instead_of_raising(ram, cores):
    """Refusing to start because we could not measure the box is the worse bug.

    Every one of these is a real shape: a container with no reported RAM, a
    ``cpu_count()`` of ``None``, a cgroup file holding ``max``, a sysconf that
    is absent on the platform. The answer is the conservative row and a
    recorded source, never a traceback at the top of a startup path.
    """
    profile = profile_for(ram, cores)

    assert profile.name in {"small", "standard", "workstation"}
    assert profile.cpu_count >= 1
    assert profile.heavy_slots >= 1
    assert profile.raster_workers >= 1
    assert profile.blas_threads >= 1
    assert profile.torch_threads >= 1
    if not isinstance(ram, int) or ram <= 0:
        # Unreadable RAM is the floor, and says so.
        assert profile.name == "small"
        assert profile.total_ram_bytes is None
        assert profile.ram_source == "unknown"


def test_an_unknown_core_count_is_treated_as_one_core():
    profile = profile_for(128 * GIB, None)

    assert profile.cpu_count == 1
    assert profile.cpu_source == "unknown"
    assert profile.raster_workers == 1
    assert profile.blas_threads == 1


def test_a_forced_profile_name_wins_and_says_so():
    profile = profile_for(256 * GIB, 28, forced_name="small")

    assert profile.name == "small"
    assert profile.basis == "forced:small"
    assert "forced:small" in profile.summary()


def test_a_nonsense_forced_name_falls_back_to_detection_rather_than_raising():
    profile = profile_for(256 * GIB, 28, forced_name="enormous")

    assert profile.name == "workstation"
    assert profile.basis == "detected"


def test_summary_and_as_dict_survive_an_unknown_machine():
    profile = profile_for(None, None)

    assert "RAM unknown" in profile.summary()
    assert json.loads(json.dumps(profile.as_dict()))["profile"] == "small"


# --------------------------------------------------------------------------
# Detection on whatever box is running this
# --------------------------------------------------------------------------


def test_detection_reads_this_machine():
    ram, ram_source = machine.detect_total_ram_bytes()
    cores, cpu_source = machine.detect_cpu_count()

    assert isinstance(cores, int) and cores >= 1, (cores, cpu_source)
    assert cpu_source in {"process", "os", "env", "unknown"}
    # A machine that cannot report its RAM is allowed (that is the degradation
    # path above); one that reports an implausible number is not.
    if ram is not None:
        assert ram_source in {"win32", "sysconf", "cgroup-v1", "cgroup-v2", "env"}
        assert GIB // 2 <= ram <= 8192 * GIB, f"{ram} bytes from {ram_source}"


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("4294967296\n", 4 * GIB),          # a 4 GB container
        ("max\n", None),                    # cgroup v2, no limit set
        ("9223372036854771712\n", None),    # cgroup v1's spelling of no limit
        ("", None),                         # an empty file
        ("not-a-number", None),             # a kernel we do not understand
    ],
)
def test_a_container_ceiling_is_read_from_the_cgroup(monkeypatch, tmp_path, contents, expected):
    """A 4 GB container on a 256 GB host must not detect as a workstation.

    Inside a container ``sysconf`` still reports the *host's* memory, so
    without this the app would size its pools for a machine it is not on and
    be OOM-killed by the runtime rather than degrading. The file locations are
    a module constant precisely so this can be exercised from a machine that is
    not in a container -- which is every machine this is developed on, and this
    build box in particular (Windows, where the cgroup probe always misses).
    """
    fake = tmp_path / "memory.max"
    fake.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(machine, "CGROUP_LIMIT_FILES", ((str(fake), "cgroup-v2"),))
    monkeypatch.delenv("QUANTEM_TOTAL_RAM_BYTES", raising=False)

    limit, source = machine._cgroup_limit_bytes()
    assert limit == expected
    assert source == ("cgroup-v2" if expected else "unknown")


def test_a_cgroup_ceiling_below_the_physical_ram_wins(monkeypatch, tmp_path):
    fake = tmp_path / "memory.max"
    fake.write_text(str(4 * GIB), encoding="utf-8")
    monkeypatch.setattr(machine, "CGROUP_LIMIT_FILES", ((str(fake), "cgroup-v2"),))
    monkeypatch.delenv("QUANTEM_TOTAL_RAM_BYTES", raising=False)

    total, source = machine.detect_total_ram_bytes()

    assert total == 4 * GIB
    assert source == "cgroup-v2"
    assert profile_for(total, 28, ram_source=source).name == "small"


def test_an_env_override_beats_every_probe(monkeypatch):
    monkeypatch.setenv("QUANTEM_TOTAL_RAM_BYTES", str(8 * GIB))
    monkeypatch.setenv("QUANTEM_CPU_COUNT", "4")

    assert machine.detect_total_ram_bytes() == (8 * GIB, "env")
    assert machine.detect_cpu_count() == (4, "env")
    assert machine.detect_profile().name == "small"


@pytest.mark.parametrize("bad", ["", "   ", "0", "-2", "eight", "8.5"])
def test_a_nonsense_env_override_is_ignored_rather_than_fatal(monkeypatch, bad):
    monkeypatch.setenv("QUANTEM_CPU_COUNT", bad)

    cores, source = machine.detect_cpu_count()
    assert cores >= 1
    assert source != "env"


def test_a_typo_in_an_override_is_visible_even_before_logging_is_configured(tmp_path):
    """Ignoring a misspelt override silently is how someone debugs for an hour.

    This runs before ``django.setup()``, so there are no handlers -- but
    ``logging`` falls back to its ``lastResort`` stderr handler at WARNING
    level, which is exactly the behaviour relied on here. A subprocess, because
    ``caplog`` would install a handler and test the wrong path.
    """
    env = _child_env(tmp_path)
    env["QUANTEM_CPU_COUNT"] = "banana"
    env["QUANTEM_MACHINE_PROFILE"] = "enormous"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from quantem.core.machine import configure_process; "
            "print(configure_process().summary())",
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "ignoring QUANTEM_CPU_COUNT='banana'" in completed.stderr, completed.stderr
    assert "ignoring QUANTEM_MACHINE_PROFILE='enormous'" in completed.stderr, completed.stderr
    assert completed.stdout.startswith("machine profile: ")


def test_the_singleton_is_configured_and_consistent():
    profile = machine.get_machine_profile()

    assert isinstance(profile, MachineProfile)
    assert profile is machine.get_machine_profile()
    assert profile is machine.configure_process()
    for name in REQUIRED_THREAD_VARS:
        assert os.environ.get(name), f"{name} is not set in this process"


def test_the_overlay_raster_pool_reads_its_worker_count_from_the_profile():
    """The one consumer that already exists, wired by name.

    ``segmentation/overlay_ngff/constants.py`` was written against this module
    before it existed: it looks up ``machine.get_machine_profile().raster_workers``
    and silently keeps its old constant if either name is missing. Silently is
    the problem -- so the join is asserted from this side. If this fails after
    somebody renames something, the fix is in whichever file moved, and both
    names are in the message.
    """
    from quantem.segmentation.overlay_ngff.constants import raster_process_pool_size

    assert raster_process_pool_size() == machine.get_machine_profile().raster_workers, (
        "overlay_ngff.constants.raster_process_pool_size() no longer agrees with "
        "quantem.core.machine.get_machine_profile().raster_workers; that accessor "
        "falls back to its own constant when the name is missing, so a rename here "
        "turns into a silent worker-count change there."
    )


# --------------------------------------------------------------------------
# Import order, over the syntax
# --------------------------------------------------------------------------

#: Modules that begin a process. Everything reachable from one of these at
#: *module* scope runs before the entry point has executed a single statement,
#: so anything on this closure that imports numpy has already lost.
BOOT_PATH_ENTRY_MODULES = (
    "quantem",            # every `import quantem.x` runs this first
    "quantem._pytest_env",  # -p plugin, the earliest import under pytest
    "quantem.cli",        # the console script and the frozen server
    "quantem.core",       # DJANGO_SETTINGS_MODULE's package
    "quantem.jobs.pool",  # unpickled in a spawned pool child before its work
)

#: Importing any of these commits per-thread BLAS/OpenMP arenas sized from the
#: core count, and cannot be undone.
FORBIDDEN_ON_THE_BOOT_PATH = frozenset(
    {"numpy", "scipy", "torch", "pandas", "skimage", "cv2", "tifffile", "zarr", "PIL"}
)


def _module_path(dotted: str) -> Path | None:
    rel = dotted.replace(".", os.sep)
    for candidate in (REPO_SRC / f"{rel}.py", REPO_SRC / rel / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_level_imports(tree: ast.AST) -> list[tuple[str, int]]:
    """``(dotted name, lineno)`` for every import outside a def or a class.

    ``if``/``try`` bodies are descended into deliberately -- a conditional
    import at module scope still runs at module scope -- and function and class
    bodies are not, because those run when they are called. The one exception
    is ``if TYPE_CHECKING:``, which provably does not run: excluding it is
    precision, not a hole.
    """
    found: list[tuple[str, int]] = []

    def _is_type_checking_guard(node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        test = node.test
        if isinstance(test, ast.Name):
            return test.id == "TYPE_CHECKING"
        return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"

    def walk(nodes) -> None:
        for node in nodes:
            if isinstance(node, ast.Import):
                found.extend((alias.name, node.lineno) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # a relative import; resolve nothing, ignore
                    continue
                base = node.module or ""
                found.append((base, node.lineno))
                found.extend((f"{base}.{alias.name}", node.lineno) for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            elif _is_type_checking_guard(node):
                walk(node.orelse)  # type: ignore[attr-defined]
            else:
                for field_name in getattr(node, "_fields", ()):
                    value = getattr(node, field_name, None)
                    if isinstance(value, list):
                        walk([n for n in value if isinstance(n, ast.AST)])
                    elif isinstance(value, ast.AST):
                        walk([value])

    walk(tree.body)  # type: ignore[attr-defined]
    return found


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _boot_path_closure() -> dict[str, list[tuple[str, int]]]:
    """Every quantem module reachable at import time from an entry point.

    Maps ``module -> [(imported name, lineno), ...]``, including the non-quantem
    imports, so the caller can look for the forbidden ones.
    """
    seen: dict[str, list[tuple[str, int]]] = {}
    pending = list(BOOT_PATH_ENTRY_MODULES)
    while pending:
        dotted = pending.pop()
        if dotted in seen:
            continue
        path = _module_path(dotted)
        if path is None:
            continue
        imports = _module_level_imports(ast.parse(_read(path), filename=str(path)))
        seen[dotted] = imports
        for name, _line in imports:
            if not name.startswith("quantem"):
                continue
            # `from quantem.core import machine` yields both `quantem.core` and
            # `quantem.core.machine`; whichever resolves to a file is followed,
            # and ancestor packages are followed too because importing a
            # submodule runs them.
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                ancestor = ".".join(parts[:depth])
                if ancestor not in seen and _module_path(ancestor) is not None:
                    pending.append(ancestor)
    return seen


def test_the_boot_path_scan_still_sees_the_modules_it_guards():
    """A scanner that matches nothing passes forever."""
    closure = _boot_path_closure()

    for entry in BOOT_PATH_ENTRY_MODULES:
        assert entry in closure, f"{entry} was not found under {REPO_SRC}"
    # Following imports, not just listing the seeds.
    assert "quantem.core.machine" in closure
    assert "quantem.core.config" in closure
    assert len(closure) >= len(BOOT_PATH_ENTRY_MODULES) + 2


def test_nothing_on_the_boot_path_imports_numpy_at_module_scope():
    """The rule, enforced where it is cheap and total: over the syntax.

    OpenBLAS and OpenMP size their arenas at numpy's import, from the core
    count, at roughly 27 MB a thread, and never look at the environment again.
    A module-level ``import numpy`` anywhere in this closure runs before the
    entry point's first statement, so the pin below it is decoration. Measured
    cost of losing this race on the build box: 1 668 MB of commit instead of
    252 MB.
    """
    offenders = [
        f"{module} imports {name} (line {line})"
        for module, imports in sorted(_boot_path_closure().items())
        for name, line in imports
        if name.split(".")[0] in FORBIDDEN_ON_THE_BOOT_PATH
    ]

    assert not offenders, (
        "these modules run before an entry point can pin the BLAS/OpenMP thread "
        "count, so the pin is already too late when they are reached: "
        + "; ".join(offenders)
        + ". Move the import inside the function that needs it."
    )


def _configure_call_index(body: list[ast.stmt]) -> int | None:
    for index, node in enumerate(body):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "configure_process"
        ):
            return index
    return None


def test_the_core_package_pins_threads_before_it_imports_anything_else():
    """``quantem.core`` is what ``DJANGO_SETTINGS_MODULE`` reaches first.

    The pin has to be the first thing in it, ahead of the ``.env`` load and
    ahead of ``pathlib``: every one of those is an import, and an import is
    exactly what could pull numpy in. Only the docstring and the import of
    ``configure_process`` itself may precede the call.
    """
    path = SRC_ROOT / "core" / "__init__.py"
    body = ast.parse(_read(path), filename=str(path)).body

    index = _configure_call_index(body)
    assert index is not None, (
        f"{path} no longer calls configure_process() at module scope; every "
        "Django entry point -- django.setup(), wsgi, pytest-django, a spawned "
        "job worker -- reaches this package before it reaches numpy, and this "
        "is where that is turned into a pinned thread count."
    )

    preceding = body[:index]
    allowed = []
    for node in preceding:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the module docstring
        if isinstance(node, ast.ImportFrom) and node.module == "quantem.core.machine":
            continue
        allowed.append(ast.dump(node)[:120])
    assert not allowed, (
        "statements run in quantem/core/__init__.py before the thread pin: "
        + "; ".join(allowed)
        + ". Anything here is an opportunity to import numpy first, which "
        "commits the arenas and makes the pin a no-op."
    )


def test_the_cli_pins_threads_when_it_prepares_the_process():
    """The CLI reaches numpy without going through ``quantem.core`` first.

    ``quantem serve`` publishes the data directory, then imports Django. The
    pin belongs in ``_prepare_env`` -- after the data directory is published,
    because importing ``quantem.core`` creates it, and before every heavy
    import, all of which live inside functions in that module.
    """
    path = SRC_ROOT / "cli.py"
    tree = ast.parse(_read(path), filename=str(path))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    prepare = functions.get("_prepare_env")
    assert prepare is not None, "quantem/cli.py no longer has _prepare_env"
    called = {
        node.func.id
        for node in ast.walk(prepare)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "configure_process" in called, (
        "quantem.cli._prepare_env must call configure_process(): it is the last "
        "thing that runs in the CLI before Django and before any import that "
        "could reach numpy."
    )

    # And every command must go through it, or the pin has a hole.
    for name in ("cmd_serve", "cmd_run", "_prepare_storage_only"):
        node = functions.get(name)
        assert node is not None, f"quantem/cli.py no longer has {name}"
        inner = {
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "_prepare_env" in inner, f"{name} does not call _prepare_env"


#: The single place allowed to ask how big the machine is (ruling R2), plus the
#: one caller that has not been moved onto the profile yet. That caller is a
#: real loose end and is named here rather than left to be discovered:
#: ``JobRunner.__init__`` sizes its CPU worker pool from ``os.cpu_count() - 1``,
#: which on the workstation profile is 27 concurrent heavy jobs against a
#: designed 4. Moving it is a scheduler change, not an S0 change.
CAPABILITY_PROBE_ALLOWLIST = {
    ("quantem/core/machine.py", "cpu_count"),
    ("quantem/core/machine.py", "virtual_memory"),
    ("quantem/jobs/runner.py", "cpu_count"),
}


def test_no_second_capability_probe_grows_outside_the_profile():
    """Owner ruling R2: detect capability once, in one module.

    Not a style rule. Two probes disagree the moment one of them learns about
    cgroups, affinity masks or a forced profile, and the disagreement shows up
    as a worker pool sized for a machine the rest of the app does not believe
    it is on.
    """
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix().replace("src/", "", 1)
        tree = ast.parse(_read(path), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            probe = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if probe not in {"cpu_count", "process_cpu_count", "virtual_memory"}:
                continue
            if probe == "process_cpu_count":
                probe = "cpu_count"
            if (rel, probe) in CAPABILITY_PROBE_ALLOWLIST:
                continue
            offenders.append(f"{rel}:{node.lineno} calls {probe}()")

    assert not offenders, (
        "a second capability probe: " + "; ".join(offenders) + ". Ask "
        "quantem.core.machine.get_machine_profile() instead -- it is the one "
        "place allowed to measure the machine (owner ruling R2), and it "
        "already accounts for cgroup limits, affinity masks and the "
        "QUANTEM_MACHINE_PROFILE override."
    )


# --------------------------------------------------------------------------
# Import order, behaviourally, in real processes
# --------------------------------------------------------------------------


def _child_env(tmp_path: Path) -> dict[str, str]:
    """A clean environment: no inherited thread pin, data dir inside tmp."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC)
    env["QUANTEM_DATA_DIR"] = str(tmp_path / "data")
    env["QUANTEM_AUTOSTART_JOBS"] = "0"
    env.pop("QUANTEM_MACHINE_PROFILE", None)
    for name in THREAD_ENV_VARS:
        env.pop(name, None)
    return env


def _run_child(source: str, tmp_path: Path, *args: str, timeout: float = 600.0) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source), *args],
        env=_child_env(tmp_path),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        f"child exited {completed.returncode}\n"
        f"stdout: {completed.stdout[-4000:]}\nstderr: {completed.stderr[-4000:]}"
    )
    last = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    assert last, f"child printed no result line\nstdout: {completed.stdout[-4000:]}"
    return json.loads(last[-1])


#: Records ``os.environ`` at the instant numpy is *first* looked up, from a
#: meta-path finder installed before anything else. A finder that returns None
#: does not change resolution; it only gets to watch.
_WATCHER = """
        import json, os, sys

        VARS = {vars!r}

        class _FirstImportWatcher:
            def __init__(self):
                self.seen = {{}}
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.partition(".")[0]
                if root in ("numpy", "scipy", "torch") and root not in self.seen:
                    self.seen[root] = {{v: os.environ.get(v) for v in VARS}}
                return None

        watcher = _FirstImportWatcher()
        sys.meta_path.insert(0, watcher)
"""


def test_the_django_boot_path_pins_the_threads_before_numpy_is_imported(tmp_path):
    """A real ``django.setup()``, watched.

    This is the behavioural half of the syntax rule above, and it is the test
    that would survive somebody restructuring the imports in a way the AST scan
    does not model. The watcher is a meta-path finder, so it sees the *first*
    lookup of ``numpy`` wherever it comes from -- Django's own machinery, an
    app module, or the explicit import at the end.
    """
    result = _run_child(
        _WATCHER.format(vars=list(THREAD_ENV_VARS))
        + """
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantem.core.settings")
        import django
        django.setup()
        import numpy  # noqa: F401

        from quantem.core.machine import get_machine_profile
        profile = get_machine_profile()
        print(json.dumps({
            "at_numpy": watcher.seen.get("numpy"),
            "blas_threads": profile.blas_threads,
            "profile": profile.name,
            "pinned_before_numpy": profile.pinned_before_numpy,
        }))
        """,
        tmp_path,
    )

    at_numpy = result["at_numpy"]
    assert at_numpy is not None, "numpy was never imported, so this proved nothing"
    expected = str(result["blas_threads"])
    for name in REQUIRED_THREAD_VARS:
        assert at_numpy[name] == expected, (
            f"{name} was {at_numpy[name]!r} when numpy was first imported, expected "
            f"{expected!r} for the {result['profile']} profile. OpenBLAS and OpenMP "
            "read these once, at that import, to size per-thread arenas; setting "
            "them afterwards changes nothing but the value that reads back."
        )
    assert result["pinned_before_numpy"] is True


def test_the_cli_pins_the_threads_before_it_imports_anything_heavy(tmp_path):
    """``_prepare_env`` is the CLI's seam, and it must leave numpy unimported.

    Two claims in one child: importing ``quantem.cli`` does not drag numpy in
    (so the pin still has somewhere to stand), and ``_prepare_env`` both
    publishes the data directory and pins the threads.
    """
    result = _run_child(
        f"        VARS = {list(THREAD_ENV_VARS)!r}\n"
        """
        import json, os, sys
        from pathlib import Path

        from quantem.cli import _prepare_env

        before = "numpy" in sys.modules
        _prepare_env(Path(sys.argv[1]))
        print(json.dumps({
            "numpy_imported_by_importing_cli": before,
            "numpy_imported_after_prepare_env": "numpy" in sys.modules,
            "env": {v: os.environ.get(v) for v in VARS},
            "data_dir": os.environ.get("QUANTEM_DATA_DIR"),
        }))
        """,
        tmp_path,
        str(tmp_path / "cli-data"),
    )

    assert result["numpy_imported_by_importing_cli"] is False, (
        "importing quantem.cli now imports numpy, so the CLI's thread pin can "
        "never run early enough. Move the new import inside a function."
    )
    assert result["numpy_imported_after_prepare_env"] is False, (
        "_prepare_env now imports numpy; it runs before the pin has any value."
    )
    for name in REQUIRED_THREAD_VARS:
        assert result["env"][name], f"{name} was not set by _prepare_env"
    assert result["data_dir"] == str(tmp_path / "cli-data")


def test_a_pre_set_thread_count_is_left_alone(tmp_path):
    """A user who exported ``OMP_NUM_THREADS`` keeps it.

    This is also the mechanism that makes spawned workers free: they inherit
    the parent's environment, so ``configure_process`` in the child finds the
    variables already set and does nothing. One decision, inherited, rather
    than one decision re-taken per process.
    """
    env = _child_env(tmp_path)
    env["OMP_NUM_THREADS"] = "3"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import json, os
                from quantem.core.machine import configure_process
                profile = configure_process()
                print(json.dumps({
                    "omp": os.environ["OMP_NUM_THREADS"],
                    "mkl": os.environ["MKL_NUM_THREADS"],
                    "applied": list(profile.thread_env_applied),
                }))
                """
            ),
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.splitlines()[-1])

    assert result["omp"] == "3", "an exported OMP_NUM_THREADS was overwritten"
    assert "OMP_NUM_THREADS" not in result["applied"]
    assert result["mkl"], "the variables the user did not set must still be pinned"


# --------------------------------------------------------------------------
# The number
# --------------------------------------------------------------------------

#: The design's per-stage ceiling for the process baseline plus the scientific
#: stack, BIG_IMAGE_DESIGN section 1.3.
IMPORT_COMMIT_BUDGET_MB = 700

#: Win32 ``PrivateUsage`` -- the private commit charge, which is the quantity a
#: Job Object's ``ProcessMemoryLimit`` caps and therefore the one that decides
#: whether a constrained run survives. ``WorkingSetSize`` is the Windows
#: spelling of RSS and is *not* the number: the OpenBLAS arenas are committed
#: but largely untouched, so they barely show in the working set (measured
#: 217 MB working set against 1 668 MB commit, unpinned).
_COMMIT_PROBE = """
    import ctypes, ctypes.wintypes as wt, json, os, sys

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # argtypes/restype are not optional here: GetCurrentProcess returns the
    # pseudo-handle (HANDLE)-1, and left to ctypes' default c_int that arrives
    # at GetProcessMemoryInfo as a 32-bit -1, which fails with ERROR_INVALID_HANDLE.
    _kernel32.GetCurrentProcess.restype = wt.HANDLE
    _psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    _psapi.GetProcessMemoryInfo.restype = wt.BOOL
    _hproc = _kernel32.GetCurrentProcess()

    def commit_mb():
        c = PMC(); c.cb = ctypes.sizeof(c)
        if not _psapi.GetProcessMemoryInfo(_hproc, ctypes.byref(c), c.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return c.PrivateUsage / (1024 * 1024), c.WorkingSetSize / (1024 * 1024)
"""


def _measure_import_commit(tmp_path: Path, *, pinned: bool, profile_name: str | None) -> dict:
    env = _child_env(tmp_path)
    if profile_name:
        env["QUANTEM_MACHINE_PROFILE"] = profile_name
    source = textwrap.dedent(_COMMIT_PROBE) + f"pin = {pinned!r}\n" + textwrap.dedent(
        """
        if pin:
            from quantem.core.machine import configure_process
            profile = configure_process()
            threads = profile.blas_threads
            name = profile.name
        else:
            threads = None
            name = "unpinned"

        base_priv, _ = commit_mb()
        import numpy, scipy, scipy.ndimage, torch  # noqa: F401
        priv, ws = commit_mb()
        print(json.dumps({
            "profile": name,
            "threads": threads,
            "commit_mb": round(priv, 1),
            "working_set_mb": round(ws, 1),
            "baseline_commit_mb": round(base_priv, 1),
            "torch_threads_default": torch.get_num_threads(),
            "cpu_count": os.cpu_count(),
        }))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, (
        f"child exited {completed.returncode}\nstderr: {completed.stderr[-4000:]}"
    )
    return json.loads(completed.stdout.splitlines()[-1])


requires_windows_commit_counters = pytest.mark.skipif(
    sys.platform != "win32",
    reason=(
        "committed bytes are measured through Win32 GetProcessMemoryInfo. The "
        "POSIX analogue (RLIMIT_AS, or RSS from getrusage) measures a different "
        "quantity; rather than assert a number that does not mean the same "
        "thing, this acceptance runs on Windows."
    ),
)
requires_scientific_stack = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy", "torch")),
    reason="numpy, scipy and torch must all be installed to measure their import",
)


@requires_windows_commit_counters
@requires_scientific_stack
def test_the_scientific_stack_imports_inside_the_budget_on_the_target_profile(tmp_path):
    """S0's acceptance, on the machine ruling R3 names: <= 700 MB committed.

    ``small`` is forced rather than detected because this build box is not the
    target -- 256 GB of RAM detects as ``workstation`` -- and the number that
    matters is the one an 8 GB laptop pays. The Job Object cap is not used
    here: the quantity under test is the committed baseline of a process that
    has allocated nothing, which is read directly from
    ``GetProcessMemoryInfo``. See this module's docstring for what the cap does
    and does not reproduce.
    """
    result = _measure_import_commit(tmp_path, pinned=True, profile_name="small")

    assert result["profile"] == "small"
    assert result["threads"] == 2
    assert result["commit_mb"] <= IMPORT_COMMIT_BUDGET_MB, (
        f"`import numpy, scipy, torch` committed {result['commit_mb']} MB on the "
        f"small profile ({result['threads']} threads), over the "
        f"{IMPORT_COMMIT_BUDGET_MB} MB ceiling in BIG_IMAGE_DESIGN 1.3. On an 8 GB "
        "machine the server process is budgeted 400 MB steady and a heavy worker "
        f"2 400 MB. Measured detail: {result}"
    )


@requires_windows_commit_counters
@requires_scientific_stack
def test_the_scientific_stack_imports_inside_the_budget_on_this_machine(tmp_path):
    """The same ceiling on whatever profile this box actually detects.

    The widest row is ``workstation`` at 8 BLAS threads, which measured 639 MB
    -- inside 700 MB, but not by much. If a future row raises the thread count
    this is the test that says so.
    """
    result = _measure_import_commit(tmp_path, pinned=True, profile_name=None)

    assert result["commit_mb"] <= IMPORT_COMMIT_BUDGET_MB, (
        f"`import numpy, scipy, torch` committed {result['commit_mb']} MB on the "
        f"detected {result['profile']} profile ({result['threads']} threads), over "
        f"the {IMPORT_COMMIT_BUDGET_MB} MB ceiling. Measured detail: {result}"
    )


@pytest.mark.slow
@requires_windows_commit_counters
@requires_scientific_stack
def test_without_the_pin_the_same_import_blows_the_budget(tmp_path):
    """The negative control: the cost of losing the race, on demand.

    Without this the budget test above could be guarding nothing -- it would
    pass just as happily if the arenas had always been small. They are not:
    unpinned, this machine's 28 cores turn the same three imports into roughly
    1.7 GB of commit, and a 4-core laptop still pays ~380 MB for arenas it
    never uses. Marked slow because it is a second cold torch import; the
    assertion it protects runs in the default lane.
    """
    unpinned = _measure_import_commit(tmp_path, pinned=False, profile_name=None)
    pinned = _measure_import_commit(tmp_path, pinned=True, profile_name="small")

    if unpinned["cpu_count"] and unpinned["cpu_count"] <= 4:
        pytest.skip(
            "this machine has <= 4 cores, so the unpinned arenas are already "
            f"small ({unpinned['commit_mb']} MB) and the lever cannot be shown "
            "here. Run it on a many-core box."
        )
    assert unpinned["commit_mb"] > IMPORT_COMMIT_BUDGET_MB, (
        "the unpinned import fitted the budget on this machine "
        f"({unpinned['commit_mb']} MB with {unpinned['cpu_count']} cores), so the "
        "pin is no longer the lever this module is built around. Re-measure "
        "before deleting anything."
    )
    assert pinned["commit_mb"] < unpinned["commit_mb"] / 2


# --------------------------------------------------------------------------
# The user-visible surface
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_quantem_serve_prints_the_machine_profile(tmp_path):
    """BIG_IMAGE_DESIGN S0: "logged at startup, printed by ``quantem serve``".

    A real child process, read on a piped stdout and killed as soon as the
    banner is out -- the line is printed before ``django.setup()``, so this
    never reaches waitress and never binds the port. "Why is this slow" and
    "why did it refuse" are both answered by this line, so it has to be in the
    terminal, not only in a log nobody has found yet.
    """
    port = _free_port()
    env = _child_env(tmp_path)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from quantem.cli import main; raise SystemExit(main())",
            "serve",
            "--port",
            str(port),
            "--data-dir",
            str(tmp_path / "serve-data"),
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue[str] = queue.Queue()

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line.rstrip("\r\n"))

    threading.Thread(target=_pump, daemon=True).start()
    try:
        banner: list[str] = []
        for _ in range(6):
            try:
                banner.append(lines.get(timeout=180.0))
            except queue.Empty:
                break
            if banner[-1].startswith("machine profile: "):
                break
        assert any(line.startswith("machine profile: ") for line in banner), (
            "quantem serve did not print the machine profile. Banner was: "
            f"{banner!r}"
        )
        line = next(x for x in banner if x.startswith("machine profile: "))
        assert " cores)" in line and "raster workers" in line, line
    finally:
        proc.kill()
        proc.wait(timeout=60)

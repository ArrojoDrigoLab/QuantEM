"""How big is this machine, and what may QuantEM therefore do on it.

One module, computed once, consulted everywhere. Owner ruling **R2**: detect
capability *once*, express it as a single profile the rest of the code reads,
and fork only where a lever is large and measured -- never a scatter of
``if gpu`` / ``os.cpu_count()`` checks that drift apart. Owner ruling **R3**
names the floor this is sized for: an 8 GB Apple Silicon MacBook Air and a
4-core / 8 GB Windows laptop, handling images of 2-3 GB.

Nothing else in ``src/quantem`` may ask how big the machine is.
``quantem/core/tests/test_machine_profile.py`` enforces that over the syntax.

The lever this module exists for
--------------------------------
Pinning the BLAS/OpenMP thread count before numpy is imported. It is not a
micro-optimisation; it is most of an 8 GB laptop's budget. MEASURED on this
build box (28 cores, 256 GB, Win32 ``PrivateUsage`` after
``import numpy, scipy, torch``):

===============  =====================
threads pinned   committed after import
===============  =====================
unpinned (28)    1 668 MB
8                  639 MB
4                  381 MB
2                  252 MB
1                  188 MB
===============  =====================

OpenBLAS and OpenMP size their per-thread arenas from the core count at
*import* time, at roughly 27 MB a thread. Setting the variables afterwards does
nothing at all: the arenas are already committed. That is the whole reason this
module has to run before the first ``import numpy`` anywhere in the process,
and why two entry points call it explicitly:

* :func:`quantem.cli._prepare_env` -- the CLI path, right after the data
  directory is published and before any heavy import; and
* ``quantem/core/__init__.py`` -- the Django path, on the first line of the
  package that ``DJANGO_SETTINGS_MODULE`` reaches.

Spawned children (job workers, the overlay raster pool) inherit the parent's
environment, so they are pinned for free; they call this again anyway and it is
a no-op.

The profile table
-----------------
BIG_IMAGE_DESIGN section 1.4(a). RAM picks the row; the core count can only
*lower* the derived counts, never raise them (see :func:`profile_for`).

============  ===============  ===========  ==============  =============  ============  ==============
profile       trigger          heavy slots  raster workers  torch threads  BLAS threads  default family
============  ===============  ===========  ==============  =============  ============  ==============
small         RAM < 12 GiB     1            2               4              2             quantem (ViT-B)
standard      12-32 GiB        2            4               8              4             omniem (ViT-L)
workstation   RAM > 32 GiB     4            8               16             8             omniem (ViT-L)
============  ===============  ===========  ==============  =============  ============  ==============

``blas_threads`` is deliberately lower than ``torch_threads``: the env pin is
the process *floor*, paid once by every process at import. A stage that wants
torch's full intra-op width asks for ``torch_threads`` explicitly and pays the
~27 MB a thread then, while it is holding the budget, rather than in every
worker that only ever touches numpy.

Overrides, and why they are environment variables
-------------------------------------------------
``QUANTEM_MACHINE_PROFILE=small|standard|workstation`` forces the row.
``QUANTEM_TOTAL_RAM_BYTES`` and ``QUANTEM_CPU_COUNT`` fake the detector.

All three are read from the real environment, **before** ``.env`` is loaded.
That is not an oversight: the pin has to happen before every import, and
``.env`` loading is itself an import. A developer who wants to hold this box
down to the laptop profile exports the variable.

A malformed override is ignored with a warning rather than being fatal, and the
warning is visible even though this runs before Django configures logging:
with no handlers installed, ``logging`` falls back to its ``lastResort``
stderr handler at WARNING level. Verified, not assumed --
``QUANTEM_CPU_COUNT=banana`` prints
``ignoring QUANTEM_CPU_COUNT='banana': not an integer`` on stderr and starts
normally.

Degrading rather than throwing
------------------------------
A container that reports no memory, a ``cpu_count()`` of ``None``, a cgroup
limit of ``max``, a sysconf that raises: every one of these lands on ``small``
with the source recorded, and never raises. Refusing to start because we could
not measure the machine would be a worse failure than running conservatively.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass, replace

logger = logging.getLogger("quantem.machine")

GIB = 1024**3

#: RAM thresholds, on the total the OS *reports* -- which is a little under the
#: number on the box (hardware-reserved pages, the cgroup limit, the hypervisor
#: balloon). A nominally-12 GB machine reporting 11.7 GiB therefore lands in
#: ``small``. That is the conservative direction and it is deliberate: the cost
#: of being wrong downward is a slower run, the cost of being wrong upward is an
#: allocation failure on a user's laptop.
SMALL_MAX_RAM_BYTES = 12 * GIB
STANDARD_MAX_RAM_BYTES = 32 * GIB

#: The three variables BIG_IMAGE_DESIGN S0 names, plus two the target machines
#: need in practice: ``NUMEXPR_NUM_THREADS`` (pandas/numexpr spins its own pool
#: on import) and ``VECLIB_MAXIMUM_THREADS`` (Apple's Accelerate framework --
#: the MacBook Air half of ruling R3 runs numpy against it).
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

PROFILE_NAMES: tuple[str, ...] = ("small", "standard", "workstation")

_PROFILE_ENV_VAR = "QUANTEM_MACHINE_PROFILE"
_RAM_ENV_VAR = "QUANTEM_TOTAL_RAM_BYTES"
_CPU_ENV_VAR = "QUANTEM_CPU_COUNT"

#: A cgroup limit at or above this is the kernel's way of saying "no limit"
#: (v1 writes 2**63-page values, v2 writes the literal ``max``). Anything this
#: large is not a limit anybody set.
_UNLIMITED_CGROUP_BYTES = 1 << 50  # 1 PiB


@dataclass(frozen=True)
class _ProfileRow:
    """One row of the design's table, before the core count clamps it."""

    name: str
    heavy_slots: int
    raster_workers: int
    torch_threads: int
    blas_threads: int
    default_model_family: str


#: BIG_IMAGE_DESIGN section 1.4(a), verbatim. The table test pins these numbers
#: against the design; change the design first.
PROFILE_TABLE: tuple[_ProfileRow, ...] = (
    _ProfileRow("small", 1, 2, 4, 2, "quantem"),
    _ProfileRow("standard", 2, 4, 8, 4, "omniem"),
    _ProfileRow("workstation", 4, 8, 16, 8, "omniem"),
)

_ROWS_BY_NAME = {row.name: row for row in PROFILE_TABLE}


@dataclass(frozen=True)
class MachineProfile:
    """What this machine is, and what the app may therefore do on it.

    Read it with :func:`get_machine_profile`. Never construct one to describe
    the running machine -- that is the second capability probe R2 forbids.
    :func:`profile_for` exists for tests and for reasoning about a machine that
    is not this one.
    """

    name: str
    total_ram_bytes: int | None
    cpu_count: int

    heavy_slots: int
    raster_workers: int
    torch_threads: int
    blas_threads: int
    default_model_family: str

    #: How the two facts above were obtained, for the diagnostics line: one of
    #: ``win32``, ``sysconf``, ``cgroup-v1``/``cgroup-v2``, ``env``, ``given``,
    #: ``unknown``.
    ram_source: str = "unknown"
    cpu_source: str = "unknown"
    #: ``detected`` or ``forced:<name>`` when ``QUANTEM_MACHINE_PROFILE`` won.
    basis: str = "detected"

    #: Which of :data:`THREAD_ENV_VARS` this process actually set. Empty when
    #: they were all inherited (a spawned worker) or set by the user.
    thread_env_applied: tuple[str, ...] = ()
    #: False when ``numpy`` was already in ``sys.modules`` at pin time, i.e. the
    #: arenas were committed before we got a say. Surfaced because a silent
    #: regression here costs a gigabyte and changes nothing visible.
    pinned_before_numpy: bool = True

    @property
    def total_ram_gib(self) -> float | None:
        if self.total_ram_bytes is None:
            return None
        return self.total_ram_bytes / GIB

    @property
    def is_small(self) -> bool:
        return self.name == "small"

    def summary(self) -> str:
        """One line, for the ``quantem serve`` banner and the startup log."""
        gib = self.total_ram_gib
        ram = f"{gib:.1f} GiB RAM" if gib is not None else "RAM unknown"
        line = (
            f"machine profile: {self.name} ({ram}, {self.cpu_count} cores) - "
            f"{self.heavy_slots} heavy job(s) at a time, "
            f"{self.raster_workers} raster workers, "
            f"{self.blas_threads} BLAS threads, {self.torch_threads} torch threads"
        )
        if self.basis != "detected":
            line += f" [{self.basis}]"
        if not self.pinned_before_numpy:
            line += " [WARNING: numpy was imported before the thread pin]"
        return line

    def as_dict(self) -> dict[str, object]:
        """JSON-safe, for the diagnostics endpoint and bug reports."""
        return {
            "profile": self.name,
            "total_ram_bytes": self.total_ram_bytes,
            "cpu_count": self.cpu_count,
            "heavy_slots": self.heavy_slots,
            "raster_workers": self.raster_workers,
            "torch_threads": self.torch_threads,
            "blas_threads": self.blas_threads,
            "default_model_family": self.default_model_family,
            "ram_source": self.ram_source,
            "cpu_source": self.cpu_source,
            "basis": self.basis,
            "pinned_before_numpy": self.pinned_before_numpy,
        }


# --- detection --------------------------------------------------------------


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring %s=%r: not an integer", name, raw)
        return None
    if value <= 0:
        logger.warning("ignoring %s=%r: must be positive", name, raw)
        return None
    return value


def _windows_total_ram_bytes() -> int | None:
    """``GlobalMemoryStatusEx().ullTotalPhys``.

    ctypes rather than psutil: psutil is not a dependency of this application
    and this module must import with nothing but the standard library, because
    it runs before everything.
    """
    import ctypes
    import ctypes.wintypes as wt

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD),
            ("dwMemoryLoad", wt.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys) or None


def _sysconf_total_ram_bytes() -> int | None:
    """POSIX physical memory. Present on Linux and on macOS."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    if pages is None or page_size is None or pages <= 0 or page_size <= 0:
        return None
    return int(pages) * int(page_size)


#: Where a container states its memory ceiling. A module constant rather than a
#: literal inside the function so the container case can be tested from a
#: machine that is not in a container -- which is every machine this is
#: developed on.
CGROUP_LIMIT_FILES: tuple[tuple[str, str], ...] = (
    ("/sys/fs/cgroup/memory.max", "cgroup-v2"),
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "cgroup-v1"),
)


def _cgroup_limit_bytes() -> tuple[int | None, str]:
    """The container's memory ceiling, which is the real total inside one.

    A 4 GB container on a 256 GB host reports the host's RAM through sysconf.
    Reading the cgroup is the difference between ``workstation`` and the
    ``small`` profile that container actually needs.
    """
    for path, source in CGROUP_LIMIT_FILES:
        try:
            with open(path, encoding="ascii") as handle:
                raw = handle.read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 < value < _UNLIMITED_CGROUP_BYTES:
            return value, source
    return None, "unknown"


def detect_total_ram_bytes() -> tuple[int | None, str]:
    """``(bytes, source)``; ``(None, "unknown")`` when the machine will not say."""
    override = _positive_int_env(_RAM_ENV_VAR)
    if override is not None:
        return override, "env"

    physical: int | None = None
    source = "unknown"
    try:
        if sys.platform == "win32":
            physical = _windows_total_ram_bytes()
            source = "win32" if physical else "unknown"
        else:
            physical = _sysconf_total_ram_bytes()
            source = "sysconf" if physical else "unknown"
    except Exception:  # noqa: BLE001 - a probe that raises must not stop startup
        logger.debug("total RAM probe failed", exc_info=True)
        physical = None
        source = "unknown"

    try:
        limit, limit_source = _cgroup_limit_bytes()
    except Exception:  # noqa: BLE001 - same
        logger.debug("cgroup probe failed", exc_info=True)
        limit, limit_source = None, "unknown"

    if limit is not None and (physical is None or limit < physical):
        return limit, limit_source
    return physical, source


def detect_cpu_count() -> tuple[int, str]:
    """Usable cores, honouring affinity, never below 1.

    ``os.process_cpu_count()`` (3.13+) respects a CPU affinity mask and the
    ``PYTHON_CPU_COUNT`` override, which ``os.cpu_count()`` does not: a job
    pinned to 4 of 28 cores should size its pools for 4.
    """
    override = _positive_int_env(_CPU_ENV_VAR)
    if override is not None:
        return override, "env"
    for attr, source in (("process_cpu_count", "process"), ("cpu_count", "os")):
        probe = getattr(os, attr, None)
        if probe is None:
            continue
        try:
            value = probe()
        except Exception:  # noqa: BLE001 - a probe that raises must not stop startup
            logger.debug("%s() failed", attr, exc_info=True)
            continue
        if isinstance(value, int) and value >= 1:
            return value, source
    return 1, "unknown"


# --- the table --------------------------------------------------------------


def _row_for_ram(total_ram_bytes: int | None) -> _ProfileRow:
    if total_ram_bytes is None:
        # Could not read the machine: assume the floor, ruling R3.
        return _ROWS_BY_NAME["small"]
    if total_ram_bytes < SMALL_MAX_RAM_BYTES:
        return _ROWS_BY_NAME["small"]
    if total_ram_bytes <= STANDARD_MAX_RAM_BYTES:
        return _ROWS_BY_NAME["standard"]
    return _ROWS_BY_NAME["workstation"]


def profile_for(
    total_ram_bytes: object,
    cpu_count: object,
    *,
    forced_name: str | None = None,
    ram_source: str = "given",
    cpu_source: str = "given",
) -> MachineProfile:
    """The profile a machine with these numbers gets. Pure, and never raises.

    RAM picks the row. The core count may only *lower* the derived counts:
    eight raster workers on a four-core laptop is eight processes fighting over
    four cores and eight copies of the per-process baseline, which is the
    opposite of what the workstation row is for. A machine whose core count is
    unreadable is treated as single-core, which is slow and correct.

    Every argument is deliberately typed ``object``: this is the function that
    has to survive a container reporting ``None``, a string, or a negative
    number, and returning a usable profile is more useful than a traceback at
    the top of a startup path.
    """
    ram = total_ram_bytes if isinstance(total_ram_bytes, int) and total_ram_bytes > 0 else None
    if ram is None:
        # Includes the case where a caller handed us a source for a number we
        # then rejected: an unusable reading is an unknown machine, and the
        # summary line must not claim otherwise.
        ram_source = "unknown"
    cores = cpu_count if isinstance(cpu_count, int) and cpu_count >= 1 else None
    if cores is None:
        cores = 1
        cpu_source = "unknown"

    basis = "detected"
    if forced_name is not None:
        row = _ROWS_BY_NAME.get(forced_name)
        if row is None:
            logger.warning(
                "ignoring %s=%r: not one of %s",
                _PROFILE_ENV_VAR,
                forced_name,
                ", ".join(PROFILE_NAMES),
            )
            row = _row_for_ram(ram)
        else:
            basis = f"forced:{forced_name}"
    else:
        row = _row_for_ram(ram)

    return MachineProfile(
        name=row.name,
        total_ram_bytes=ram,
        cpu_count=cores,
        heavy_slots=max(1, min(row.heavy_slots, max(1, cores // 2))),
        raster_workers=max(1, min(row.raster_workers, cores)),
        torch_threads=max(1, min(row.torch_threads, cores)),
        blas_threads=max(1, min(row.blas_threads, cores)),
        default_model_family=row.default_model_family,
        ram_source=ram_source,
        cpu_source=cpu_source,
        basis=basis,
    )


def detect_profile() -> MachineProfile:
    """Measure this machine and pick its row. No side effects."""
    ram, ram_source = detect_total_ram_bytes()
    cores, cpu_source = detect_cpu_count()
    forced = os.environ.get(_PROFILE_ENV_VAR, "").strip() or None
    return profile_for(
        ram,
        cores,
        forced_name=forced,
        ram_source=ram_source,
        cpu_source=cpu_source,
    )


# --- the process-wide singleton ---------------------------------------------

_profile: MachineProfile | None = None
_lock = threading.Lock()


def configure_process(*, force: bool = False) -> MachineProfile:
    """Detect the machine and pin the BLAS/OpenMP thread count. Idempotent.

    Call this before the first ``import numpy`` in the process and nowhere
    else. The variables are set with ``setdefault`` semantics, so:

    * a user who exported ``OMP_NUM_THREADS`` keeps their value -- this is not
      the module's business to overrule; and
    * a spawned worker, which inherited the parent's environment, does nothing
      here at all. That is the intended path: one decision, inherited by every
      child, rather than a decision re-taken per process.

    Never raises. A startup path is the worst possible place to discover that a
    capability probe has an edge case.
    """
    global _profile
    with _lock:
        if _profile is not None and not force:
            return _profile
        try:
            profile = detect_profile()
        except Exception:  # noqa: BLE001 - see the docstring
            logger.warning("machine detection failed; assuming the small profile", exc_info=True)
            profile = profile_for(None, None, ram_source="unknown", cpu_source="unknown")

        applied: list[str] = []
        for name in THREAD_ENV_VARS:
            if os.environ.get(name, "").strip():
                continue
            os.environ[name] = str(profile.blas_threads)
            applied.append(name)

        pinned_before_numpy = "numpy" not in sys.modules
        if applied and not pinned_before_numpy:
            # Worth a warning and not a debug line: the variables now say one
            # thing and the committed arenas say another, silently, for the
            # life of the process. MEASURED cost on this box: 1 668 MB instead
            # of 252 MB.
            logger.warning(
                "numpy was already imported when the machine profile was applied, so "
                "%s had no effect: the BLAS/OpenMP arenas are already committed at the "
                "core count. Something imported numpy before quantem.core.",
                ", ".join(applied),
            )

        _profile = replace(
            profile,
            thread_env_applied=tuple(applied),
            pinned_before_numpy=pinned_before_numpy,
        )
        return _profile


def get_machine_profile() -> MachineProfile:
    """The one profile, computed on first use.

    This is the accessor the rest of the tree consults --
    ``quantem.segmentation.overlay_ngff.constants.raster_process_pool_size``
    already reads ``.raster_workers`` off it by this name. If it is called
    before an entry point has run :func:`configure_process`, it does that work
    now; the thread pin will be too late to matter but the numbers will still
    be right.
    """
    if _profile is not None:
        return _profile
    return configure_process()


def log_profile(target: logging.Logger | None = None) -> MachineProfile:
    """Write the one-line summary to the log. Called once Django has handlers.

    ``configure_process`` deliberately does not log: it runs before
    ``django.setup()``, when logging has no handlers configured and the line
    would go nowhere.
    """
    profile = get_machine_profile()
    (target or logger).info("%s", profile.summary())
    return profile

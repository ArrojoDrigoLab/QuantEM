"""Process-pool initializers.

**Why this module exists.** A :class:`~concurrent.futures.ProcessPoolExecutor`
child on Windows is *spawned*: a fresh interpreter that inherits the parent's
environment and nothing else. It unpickles the submitted callable by importing
the module the callable lives in -- and in this codebase almost every such
module is inside a Django app, so the import reaches
``class Something(models.Model)`` and Django raises
``AppRegistryNotReady: Apps aren't loaded yet.`` before a single task runs.
Every worker dies on start-up, and the parent sees only
``BrokenProcessPool: A process in the process pool was terminated abruptly``,
which names neither the module nor the real cause.

That is not hypothetical: it is exactly what made overlays above
``RASTER_POOL_MIN_OBJECTS`` (2 000) objects -- i.e. every real EM image --
impossible to build in the shipped product, while the manifest endpoint quietly
re-queued the doomed rebuild on every poll so the user watched
"Overlay updating..." forever.

The cure is one line of contract: **every** ``ProcessPoolExecutor`` in
``src/quantem`` passes ``initializer=django_pool_initializer``, so the child has
a loaded app registry before it unpickles anything. It is invariant I-9 of the
v2 plan, and ``jobs/tests/test_pool_initializers.py`` asserts it over the source
tree so the class of bug cannot come back by inspection alone.

**The second thing this module owns** is the rule that a child never outlives
its parent -- see :func:`install_parent_death_watchdog`.

Nothing here may import Django models, ``quantem.core.settings``, or any app
module at import time: this module is itself imported by the child *before*
``django.setup()`` has run.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading

logger = logging.getLogger(__name__)

#: Marks the interpreter as a job worker. ``quantem.jobs.runner`` imports this
#: name rather than re-declaring it; the dependency only goes that way, because
#: importing ``runner`` pulls in Django models, which is the precise thing a
#: pool child cannot do yet. ``test_pool_initializers`` asserts the two are
#: equal, and now they are the same object.
WORKER_PROCESS_ENV_VAR = "QUANTEM_JOB_WORKER"

#: Django's own environment variable, and the settings module to fall back to.
DJANGO_SETTINGS_ENV_VAR = "DJANGO_SETTINGS_MODULE"
DEFAULT_DJANGO_SETTINGS_MODULE = "quantem.core.settings"

#: Set to ``0``/``false`` to leave a child running after its parent dies. The
#: only intended use is the negative control in
#: ``jobs/tests/test_parent_death.py``, which has to be able to reproduce the
#: orphan on demand; a debugger attaching to a stuck worker is the other.
PARENT_DEATH_WATCHDOG_ENV_VAR = "QUANTEM_PARENT_DEATH_WATCHDOG"

#: Exit code a worker uses when it stops because its parent went away. Distinct
#: from any handler failure, so ``worker_exit_message`` never reads it as one.
PARENT_GONE_EXIT_CODE = 3

_WATCHDOG_THREAD_NAME = "quantem-parent-death-watchdog"

_watchdog_installed = False
_watchdog_lock = threading.Lock()


def _watchdog_enabled() -> bool:
    return os.environ.get(PARENT_DEATH_WATCHDOG_ENV_VAR, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _exit_when_parent_exits(parent: object) -> None:
    """Block on the parent's sentinel, then leave immediately.

    ``os._exit`` rather than an exception or ``sys.exit``: this runs on a thread
    while the main thread is blocked in ``queue.get()`` or halfway through an
    inference pass, and neither can be persuaded to unwind. Nothing is lost by
    skipping the interpreter's teardown -- the parent that owned the job row is
    already gone, and the next server's startup reaper
    (``JobScheduler._recover_orphaned_jobs``) is what settles the row.
    """
    try:
        parent.join()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive; join has no failure mode
        logger.debug("Parent-death watchdog stopped waiting.", exc_info=True)
        return
    os._exit(PARENT_GONE_EXIT_CODE)


def install_parent_death_watchdog() -> bool:
    """Make this process exit when the process that spawned it does.

    **The leak this closes.** MEASURED in wave-0 verification: force-killing the
    server left a spawned worker holding **905 MB** running with a dead parent,
    and restarting the server did not reap it -- a second agent's server had
    produced an identical 911 MB orphan the same session. On Windows a
    force-quit is ``TerminateProcess``: no ``atexit``, so multiprocessing's own
    daemon-child cleanup never runs, and ``daemon=True`` therefore buys nothing.
    Force-quit is also how a desktop user actually leaves an app that looks
    stuck. Two of those and the machine is down ~2 GB until reboot.

    **Why the sentinel and not the parent pid.** ``multiprocessing`` hands every
    spawned child a *waitable handle* on its parent -- a duplicated process
    handle on Windows, a pipe end on POSIX -- reachable as
    ``multiprocessing.parent_process()``. Waiting on that is exact and immune to
    pid reuse, which polling ``os.getppid()`` is not: a recycled pid would make
    a live worker think its parent was alive, or a dead one think it was not.

    Returns whether a watchdog is now running. ``False`` in the server process
    itself (no parent process to watch: ``parent_process()`` is ``None``), when
    one is already installed, and when
    :data:`PARENT_DEATH_WATCHDOG_ENV_VAR` turns it off.
    """
    global _watchdog_installed

    parent = multiprocessing.parent_process()
    if parent is None:
        # The server, a test driving a worker in-process, or a fork whose parent
        # is this interpreter. Nothing to watch.
        return False
    if not _watchdog_enabled():
        return False

    with _watchdog_lock:
        if _watchdog_installed:
            return False
        _watchdog_installed = True

    threading.Thread(
        target=_exit_when_parent_exits,
        args=(parent,),
        name=_WATCHDOG_THREAD_NAME,
        daemon=True,
    ).start()
    return True


def django_pool_initializer() -> None:
    """Make a spawned pool child able to import app modules.

    Runs once per worker process, before its first task. Four steps, in this
    order:

    0. Arrange to die with the parent (:func:`install_parent_death_watchdog`).
       First, because everything after it can block: a child that hangs in
       ``django.setup()`` after its parent was force-killed is exactly the
       orphan the watchdog exists to prevent.
    1. Claim the process as a worker. A pool child inherits
       ``QUANTEM_AUTOSTART_JOBS=1`` from the server, and ``jobs.apps`` starts a
       :class:`JobScheduler` on the first DB connection unless this marker is
       set. N schedulers racing for the same rows is not a tidiness problem: it
       is duplicated job execution. ``jobs.runner._setup_django`` claims the
       same marker for the same reason.
    2. Point Django at the settings module if the environment has not already
       (``setdefault``, so an explicit choice -- the test suite's, a CLI's --
       always wins).
    3. ``django.setup()``, unless the registry is already populated (it can be
       when a "pool" is really a fork on Linux/macOS, or when a test drives the
       initializer in-process).

    Deliberately does *not* import the work module. Which module that is
    depends on the caller, the child imports it on the first unpickle anyway,
    and importing app code here would make every pool pay for the heaviest
    caller's imports.
    """
    install_parent_death_watchdog()
    os.environ[WORKER_PROCESS_ENV_VAR] = "1"
    os.environ.setdefault(DJANGO_SETTINGS_ENV_VAR, DEFAULT_DJANGO_SETTINGS_MODULE)

    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()


def pool_child_report() -> dict[str, object]:
    """What the initializer left behind, read from inside the child.

    A spawned child can only be asked to run a module-level callable, so
    checking the initializer's three post-conditions from the parent needs a
    named function to submit. This is that function: it exists for
    ``jobs/tests/test_pool_initializers.py``, which asserts the contract
    against a real pool rather than against a mock of one.
    """
    from django.apps import apps

    return {
        "apps_ready": bool(apps.ready),
        "worker_marker": os.environ.get(WORKER_PROCESS_ENV_VAR),
        "settings_module": os.environ.get(DJANGO_SETTINGS_ENV_VAR),
        "watchdog_running": any(
            thread.name == _WATCHDOG_THREAD_NAME for thread in threading.enumerate()
        ),
    }


def parent_death_probe(pid_path: str) -> None:
    """A worker that reports its pid and then blocks the way a real one does.

    For ``jobs/tests/test_parent_death.py``. A spawned child can only be given a
    module-level callable, and the test's whole point is to kill the *parent*
    and look at the operating system afterwards, so the child has to publish its
    pid somewhere the test can read it. It runs the real
    :func:`django_pool_initializer`, so what is under test is the shipped
    initializer and not a hand-rolled imitation of it.

    The block is ``Queue.get()`` with no timeout -- the same call
    ``run_job_in_persistent_worker`` sits in between jobs, and the reason a
    polling loop would not have caught this.
    """
    django_pool_initializer()
    with open(pid_path, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
        handle.flush()
        os.fsync(handle.fileno())
    multiprocessing.Queue().get()

"""``quantem-app`` console entry point.

This is the seam that makes one codebase serve both distribution channels:

* ``pip install quantem-app && quantem-app`` -- starts the loopback server and opens a
  native window via pywebview (falling back to the default browser).
* The desktop installer wraps this *same* package plus a frozen interpreter; the
  shell spawns ``quantem-app serve --port 0`` and hosts the UI itself.

Nothing here binds to anything but loopback. There is no authentication, no
multi-user mode and no remote access: this is one person's application running
on their own machine.

``--data-dir`` and where it may appear
--------------------------------------
Every subcommand accepts ``--data-dir``, before or after the subcommand name.
It used to be top-level only, so the obvious ``quantem-app --data-dir X serve``
worked and the equally obvious ``quantem-app serve --data-dir X`` died with an
unrecognised-argument error -- and ``quantem-app serve --help`` never mentioned the
flag at all, so there was nothing to read that would have said which one to
type. ``$QUANTEM_DATA_DIR`` is honoured too, and the flag wins over it.

Commands
--------
``run`` (the default), ``serve``, and ``models`` -- downloading model packs
from the QuantEM Hugging Face repository (or installing them offline from a
downloaded release bundle), listing what is installed, and re-verifying it.
Models are the one thing a fresh install cannot do without and the one thing not
shipped with the application, so obtaining them is a first-class command rather
than a module path a user has to be told.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

APP_NAME = "QuantEM"

#: Names the user data directory. ``_prepare_env`` publishes the resolved value
#: here before Django is configured; :mod:`quantem.core.config` reads it back.
DATA_DIR_ENV_VAR = "QUANTEM_DATA_DIR"


def default_data_dir() -> Path:
    """Where QuantEM keeps its data when the user has not said otherwise.

    ``$QUANTEM_DATA_DIR`` wins, so a shell that exports it does not have to
    repeat ``--data-dir`` on every command and cannot end up with the flagged
    and unflagged invocations pointing at two different databases.

    Otherwise the data lives **with the installation** -- an owner ruling
    (2026-08-09) that replaced the per-OS user data directory:

    * Frozen desktop build (PyInstaller sets ``sys.frozen``): a ``data``
      directory in the install root, derived from this executable's own
      location. The installer's layout puts ``QuantEM.exe`` in the install root
      and ``quantem-server.exe`` one level below it, in a ``quantem-server``
      directory, and the frozen server is the process resolving this, so the
      install root is the exe's grandparent. The user chose that directory at
      install time; their data sits next to it, findable and removed with it.
    * pip install: ``<sys.prefix>/quantem-data`` -- the environment *is* the
      install location. This covers venvs, conda envs and the dev checkout
      (whose ``.env`` still overrides by setting ``QUANTEM_DATA_DIR``).

    Never a hidden per-user directory: if the computed location is not
    writable, startup fails with an error naming ``QUANTEM_DATA_DIR`` (see
    :func:`_prepare_env`) rather than silently writing somewhere nobody chose.
    """
    raw = os.environ.get(DATA_DIR_ENV_VAR, "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate
        print(
            f"warning: ignoring ${DATA_DIR_ENV_VAR}={raw!r}; it must be an absolute path.",
            file=sys.stderr,
        )
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            # The one platform where storage-with-the-install cannot work.
            # Gatekeeper's app translocation runs a quarantined, unsigned app
            # from a randomised read-only mount, so the bundle's own directory
            # is not writable on exactly the first launch that matters -- and
            # the path changes between launches, so data written before the
            # user clears quarantine would be orphaned somewhere they will
            # never find. macOS gets the location macOS apps are expected to
            # use; QUANTEM_DATA_DIR still overrides, as everywhere else.
            return Path.home() / "Library" / "Application Support" / "QuantEM"
        return Path(sys.executable).resolve().parent.parent / "data"
    return Path(sys.prefix).resolve() / "quantem-data"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    """The data directory this invocation should use.

    ``--data-dir`` is declared on the top-level parser *and* on every subcommand
    with ``default=SUPPRESS``, so whichever position the user typed it in lands
    on the same namespace attribute and the subcommand's copy does not clobber a
    value given before the subcommand.
    """
    raw = getattr(args, "data_dir", None)
    return Path(raw).expanduser() if raw else default_data_dir()


def _ensure_writable(data_dir: Path) -> None:
    """Create ``data_dir`` and prove it accepts writes, or exit saying why.

    No silent fallback (owner ruling): if the computed location cannot be
    written -- an install under Program Files without rights, a read-only
    share -- QuantEM must not quietly relocate its storage to a per-user
    directory nobody chose and nobody will look in. It stops, names the
    directory it refused to fall away from, and names the override.
    """
    probe = None
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / f".quantem-write-probe-{os.getpid()}"
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("probe")
    except OSError as exc:
        raise SystemExit(
            f"error: the QuantEM data directory is not writable: {data_dir} "
            f"({exc}).\nNothing was stored anywhere else. Choose a writable "
            f"location with --data-dir or the {DATA_DIR_ENV_VAR} environment "
            "variable."
        ) from exc
    finally:
        if probe is not None:
            with contextlib.suppress(OSError):
                probe.unlink()


def _prepare_env(data_dir: Path) -> None:
    data_dir = Path(data_dir).expanduser()
    if not data_dir.is_absolute():
        raise SystemExit(
            f"error: --data-dir must be an absolute path (got {data_dir}). "
            "Storage must never be relative to the current working directory."
        )
    _ensure_writable(data_dir)
    # Overwritten, not setdefault: an explicit --data-dir has to beat an
    # inherited environment variable, or the flag would silently do nothing in a
    # shell that exports one. default_data_dir() already folded the env var in.
    os.environ[DATA_DIR_ENV_VAR] = str(data_dir)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantem.core.settings")
    os.environ.setdefault("DJANGO_DEBUG", "0")
    # A desktop app has no separate worker process to start: the queue runs in
    # this one. Without this, uploads enqueue a preprocessing job that nothing
    # ever picks up and every image sits at "NGFF pending" forever.
    # `QUANTEM_DISABLE_JOB_AUTOSTART=1` still opts out, which is what the test
    # suite and CI use.
    os.environ.setdefault("QUANTEM_AUTOSTART_JOBS", "1")
    # BIG_IMAGE_DESIGN S0 / owner ruling R2, the CLI half. Detect the machine
    # once and pin OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS
    # before anything in this process imports numpy -- OpenBLAS and OpenMP read
    # them at that import to size their per-thread arenas (~27 MB a thread) and
    # never read them again. MEASURED on the build box: `import numpy, scipy,
    # torch` commits 1 668 MB unpinned on 28 cores against 252 MB pinned to
    # two, which on the 8 GB laptop of ruling R3 is most of the budget.
    #
    # Here, and not at the top of this module: importing quantem.core runs its
    # package body, which creates the data directory. Before the line above
    # publishes QUANTEM_DATA_DIR that would be the *wrong* directory, so an
    # earlier pin would trade a gigabyte for a stray data folder next to the
    # installation. Every heavy import in this file is inside a function and
    # therefore lands after this call. quantem/core/__init__.py carries the
    # other half for Django entry points that never touch the CLI.
    from quantem.core.machine import configure_process

    configure_process()


def _prepare_storage_only(data_dir: Path) -> None:
    """Point the storage layer at ``data_dir`` without waking the job queue.

    ``quantem-app models ...`` touches the model cache and nothing else. Starting
    the background workers to copy some files would be a surprising thing for an
    install command to do, and on a machine with no database yet it would be a
    failing one.
    """
    _prepare_env(data_dir)
    os.environ["QUANTEM_AUTOSTART_JOBS"] = "0"


def _wait_until_up(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False


def _human_bytes(count: int) -> str:
    """``68719476736`` -> ``"64 GB"``. For a sentence, not for a log line."""
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
        if count >= size:
            scaled = count / size
            return f"{scaled:.0f} {unit}" if scaled >= 10 else f"{scaled:.1f} {unit}"
    return f"{count} bytes"


def _install_readable_oversize_error(limit_bytes: int) -> None:
    """Make waitress's 413 a sentence, in the shape the client already parses.

    waitress refuses an over-limit body the moment it has read the headers --
    which is the right moment, and the only fast one -- but it says so as
    ``text/plain`` reading ``exceeds max_body of 1073741824``: an internal
    phrase and a number with no unit. The client's error path expects
    ``{"error": ...}`` JSON, so today that text arrives as a failed JSON decode
    and is shown to the user verbatim.

    Only the message is replaced. The status, the timing and every other
    waitress error are untouched. ``parser.py`` holds its own reference to the
    class, so that is the name rebound.

    **What this does not fix.** When the client is still streaming its body,
    the server's response and immediate close reach it as an aborted
    connection, and the sentence is never read -- measured: a 1.93 GiB POST
    died with ``ConnectionAbortedError`` after 55 ms and no status code. The
    durable fix is for the client to check the file's size against
    ``max_upload_bytes`` before it starts. This makes the response honest for
    every client that does get to read it.
    """
    import waitress.parser as waitress_parser
    from waitress.utilities import RequestEntityTooLarge

    base = getattr(waitress_parser, "RequestEntityTooLarge", None)
    if base is None or not issubclass(base, RequestEntityTooLarge):  # pragma: no cover
        # A waitress that no longer has this seam: keep its own message rather
        # than break the server over the wording of an error.
        return

    sentence = (
        f"This upload is larger than the {_human_bytes(limit_bytes)} "
        "QuantEM accepts in one request, so nothing was saved."
    )
    payload = json.dumps(
        {
            "error": sentence,
            "error_code": "upload_too_large",
            "max_upload_bytes": limit_bytes,
        }
    ).encode("utf-8")

    class _ReadableRequestEntityTooLarge(RequestEntityTooLarge):
        def to_response(self, ident=None):  # noqa: ARG002 - waitress's signature
            del ident
            return (
                f"{self.code} {self.reason}",
                [("Content-Type", "application/json")],
                payload,
            )

    waitress_parser.RequestEntityTooLarge = _ReadableRequestEntityTooLarge


def _waitress_options() -> dict[str, object]:
    """The keyword arguments ``cmd_serve`` hands to waitress.

    Separate from the call so the two values that matter -- the body limit and
    the thread count -- can be asserted without starting a server.
    """
    from django.conf import settings

    return {
        "threads": 8,
        # Names the application in the ``Server:`` header and in waitress's own
        # error pages, which otherwise blame a library the user has never
        # heard of for the app's refusals.
        "ident": APP_NAME,
        "max_request_body_size": int(settings.QUANTEM_MAX_UPLOAD_BYTES),
    }


def _keep_temp_files_with_the_data(tmp_dir: Path) -> None:
    """Point this process's temp directory at the data directory's own.

    ``FILE_UPLOAD_TEMP_DIR`` moves Django's copy of an upload, but waitress
    spools any request body over 512 kB through ``tempfile.TemporaryFile``
    *before* Django is entered, and that reads ``TEMP``/``TMP``. Left alone,
    the first copy of every large import lands in the system temp directory --
    on Windows a path under ``C:``, which this project forbids and which is
    usually a different volume from the data directory.

    The desktop shell already exports these for the frozen build. Doing it here
    as well means ``quantem-app serve`` from a pip install behaves identically, and
    that the job workers spawned later inherit the same choice.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp_dir)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(tmp_dir)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the local server in the foreground."""
    import logging

    import django
    from waitress import serve as waitress_serve

    _prepare_env(_resolve_data_dir(args))
    # The server keeps a rotating log file under the data directory (the
    # packaged build used to write no log at all, so a crashed session left
    # nothing to attach to a bug report). setdefault: QUANTEM_LOG_TO_FILE=0
    # still turns it off. The settings module reads the flag and excludes
    # spawned job workers itself; see quantem.core.settings.
    os.environ.setdefault("QUANTEM_LOG_TO_FILE", "1")
    # The same path quantem.core.settings derives; imported (cheaply, before
    # Django) rather than re-spelled so the two can never disagree.
    from quantem.core.config import SERVER_LOG_PATH, TMP_DIR, file_logging_enabled

    # One half of "a big import never touches C:" -- the other is
    # FILE_UPLOAD_TEMP_DIR in quantem.core.settings. Before django.setup(),
    # because the job scheduler starts during it and every worker it spawns
    # inherits this process's environment.
    _keep_temp_files_with_the_data(TMP_DIR)

    # Announced before Django wakes up, and flushed. This used to print after
    # the migrations and without a flush, which had two failure modes: on a
    # first launch the user stared at a silent terminal for the whole migrate,
    # and with stdout piped (a wrapper process, a log file, `quantem-app serve >
    # out.txt`) block buffering held the lines back until the process *exited*
    # -- which a server never does, so `quantem-app serve` printed nothing at all,
    # not even the URL to open.
    port = args.port or free_port()
    print(f"{APP_NAME} serving on http://127.0.0.1:{port}", flush=True)
    print(f"data dir: {os.environ[DATA_DIR_ENV_VAR]}", flush=True)
    # Only when this process is really going to write it. Promising a path that
    # stays empty sends whoever is debugging to the wrong place.
    if file_logging_enabled():
        print(f"log file: {SERVER_LOG_PATH}", flush=True)
    print(
        "models are downloaded on demand: install them from the app's Models "
        "screen or with `quantem-app models install --all`; `quantem-app models list` "
        "shows what is installed.",
        flush=True,
    )
    # What this machine was measured to be and what the app will therefore do
    # on it -- worker counts, thread counts, how many heavy jobs run at once.
    # Printed last so it sits next to whatever the user is about to compare it
    # against, and printed at all because "why is this slow" and "why did this
    # refuse" are both answered by this line. _prepare_env computed it above.
    from quantem.core.machine import get_machine_profile

    print(get_machine_profile().summary(), flush=True)

    # Do not let the scheduler claim jobs while Django is inspecting or
    # changing the schema.  It is explicitly started again below once the
    # migration (and its recovery snapshot) have completed.
    os.environ["QUANTEM_DISABLE_JOB_AUTOSTART"] = "1"
    django.setup()
    from django.core.management import call_command
    from django.core.wsgi import get_wsgi_application

    from quantem.core.migration_safety import snapshot_before_pending_migrations

    # The first record in every session's log file: what is running, where,
    # over which data. Written after django.setup() because that is what
    # configures the handlers.
    logging.getLogger("quantem.serve").info(
        "%s serving on http://127.0.0.1:%s (data dir %s)",
        APP_NAME,
        port,
        os.environ[DATA_DIR_ENV_VAR],
    )
    # And into the log file, where a bug report can reach it. Not from
    # configure_process itself: that runs before django.setup(), when logging
    # has no handlers and the line would go nowhere.
    from quantem.core import machine

    machine.log_profile()

    snapshot_dir = None
    try:
        pending_migrations, snapshot_dir = snapshot_before_pending_migrations()
        if snapshot_dir is not None:
            logging.getLogger("quantem.serve").info(
                "Created pre-migration database snapshot at %s for %s.",
                snapshot_dir,
                ", ".join(pending_migrations),
            )
        call_command("migrate", interactive=False, verbosity=0)
    except Exception:
        if snapshot_dir is not None:
            logging.getLogger("quantem.serve").exception(
                "Database migration failed. The pre-migration snapshot remains at %s.",
                snapshot_dir,
            )
        raise
    finally:
        os.environ.pop("QUANTEM_DISABLE_JOB_AUTOSTART", None)

    # A process can die after the updater fences new jobs but before its
    # installer restarts it.  This is the new process, so that lock is stale.
    from quantem.jobs.apps import start_scheduler_if_needed
    from quantem.jobs.update_maintenance import clear_stale_update_apply_lock

    if clear_stale_update_apply_lock():
        logging.getLogger("quantem.serve").warning(
            "Cleared a stale desktop-update maintenance lock after startup."
        )
    start_scheduler_if_needed()

    # Ruling C, first-launch half: the desktop installer may have left a
    # one-shot request naming model packs to download. Turn it into ordinary
    # install jobs -- the app's verified download machinery, with progress and
    # cancel -- now that the job table exists. Never fatal; see
    # quantem.registry.pending_installs.
    from quantem.registry.pending_installs import process_pending_model_installs

    process_pending_model_installs()

    options = _waitress_options()
    _install_readable_oversize_error(int(options["max_request_body_size"]))
    waitress_serve(get_wsgi_application(), host="127.0.0.1", port=port, **options)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Start the server on a background thread and open the application window."""
    port = args.port or free_port()
    data_dir = _resolve_data_dir(args)
    _prepare_env(data_dir)

    serve_args = argparse.Namespace(port=port, data_dir=str(data_dir))
    t = threading.Thread(target=cmd_serve, args=(serve_args,), daemon=True)
    t.start()

    if not _wait_until_up(port):
        print("error: the QuantEM server did not start", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{port}/"
    try:
        import webview  # type: ignore[import-not-found]
    except ImportError:
        print("pywebview not installed; opening in your browser instead.")
        print("  for a native window:  pip install 'quantem-app[desktop]'")
        webbrowser.open(url)
        t.join()
        return 0

    webview.create_window(APP_NAME, url, width=1600, height=1000)
    webview.start()
    return 0


# --- Model packs ------------------------------------------------------------


def cmd_models_install(args: argparse.Namespace) -> int:
    """Install model packs: download them from Hugging Face, or copy a release bundle.

    Which route is decided by what was typed, so both obvious commands work:

    * ``quantem-app models install quantem:mito omniem:ld`` -- pack ids: download
      each from the QuantEM Hugging Face repository, verify, install.
    * ``quantem-app models install ./quantem-models-0.1.0`` -- a directory: the
      offline route, installing from a downloaded, unzipped release bundle.
    """
    _prepare_storage_only(_resolve_data_dir(args))

    from quantem.inference.specs import MODEL_SPECS
    from quantem.registry import cache, release
    from quantem.registry.install import InstallError, install_all_from_bundle

    sources: list[str] = list(args.sources or [])
    pack_ids = [s for s in sources if s in MODEL_SPECS]
    others = [s for s in sources if s not in MODEL_SPECS]

    if others and not args.hf:
        # A directory (the bundle route). Anything that is neither a known pack
        # id nor an existing directory is almost certainly a typo'd pack id.
        if len(others) > 1 or not Path(others[0]).expanduser().is_dir():
            bad = [s for s in others if not Path(s).expanduser().is_dir()]
            print(
                f"error: {', '.join(bad or others)} is neither a released pack id nor a "
                "directory.\nKnown pack ids: " + ", ".join(sorted(MODEL_SPECS)),
                file=sys.stderr,
            )
            return 2
        bundle_root = Path(others[0]).expanduser()
        try:
            results = install_all_from_bundle(
                bundle_root,
                pack_ids=pack_ids or None,
                force=args.force,
                on_progress=(lambda msg: print(msg, file=sys.stderr)) if args.verbose else None,
            )
        except (InstallError, release.BundleError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for r in results:
            print(f"{r.pack_id:18s} head={r.head_sha256[:16]}  {r.root}")
        total = sum(r.bytes_written for r in results)
        print(f"\n{len(results)} pack(s) installed under {cache.packs_root()}")
        print(
            f"{total / 1e9:.2f} GB written ({sum(r.reused_blobs for r in results)} blob(s) reused)"
        )
        return 0

    if others:  # --hf was given alongside names that are not pack ids
        print(
            f"error: unknown pack id(s) {', '.join(others)}. Known: "
            + ", ".join(sorted(MODEL_SPECS)),
            file=sys.stderr,
        )
        return 2
    if args.all:
        pack_ids = sorted(MODEL_SPECS)
    if not pack_ids:
        print(
            "nothing to do. Name pack ids to download them from Hugging Face "
            "(e.g. `quantem-app models install quantem:mito`, or --all), or give the "
            "directory of an unzipped release bundle.",
            file=sys.stderr,
        )
        return 2
    return _install_from_hf(pack_ids, force=args.force)


def _install_from_hf(pack_ids: list[str], *, force: bool) -> int:
    """Download and install packs from the HF repository, with progress lines."""
    from quantem.registry import cache, hf
    from quantem.registry.hf_install import install_pack_from_hf
    from quantem.registry.install import InstallError

    print(f"repository: {hf.HF_REPO_URL} @ {hf.hf_revision()[:12]}")
    failures = 0
    for pack_id in pack_ids:
        last = {"t": 0.0}

        def on_bytes(done: int, total: int, pack_id: str = pack_id, last: dict = last) -> None:
            now = time.monotonic()
            if now - last["t"] >= 1.0 or done >= total:
                last["t"] = now
                print(
                    f"  {pack_id}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB",
                    file=sys.stderr,
                )

        try:
            result = install_pack_from_hf(
                pack_id,
                force=force,
                on_status=lambda msg: print(f"  {msg}", file=sys.stderr),
                on_bytes=on_bytes,
            )
        except InstallError as exc:
            print(f"error: {pack_id}: {exc}", file=sys.stderr)
            failures += 1
            continue
        tier = "exported" if result.exported else "timm (eager)"
        print(
            f"{result.pack_id:18s} head={result.head_sha256[:16]}  encoder={tier}  "
            f"{result.downloaded_bytes / 1e6:.1f} MB downloaded  {result.root}"
        )
        if result.export_error:
            print(
                f"{'':18s} export failed ({result.export_error}); the pack runs "
                "through timm instead",
                file=sys.stderr,
            )
    print(f"\ninstalled under {cache.packs_root()}")
    return 1 if failures else 0


def cmd_models_list(args: argparse.Namespace) -> int:
    """Show every pack this data directory holds, and whether it can run."""
    _prepare_storage_only(_resolve_data_dir(args))

    from quantem.registry import cache, catalogue

    print(f"data dir: {os.environ[DATA_DIR_ENV_VAR]}")
    print(f"models:   {cache.models_root()}\n")
    entries = catalogue.packs()
    missing = 0
    for entry in entries:
        if not entry["installed"]:
            missing += 1
            print(f"{entry['id']:18s} {'-':10s} not installed")
            continue
        runnable = "runnable" if entry["runnable"] else "NOT RUNNABLE"
        print(
            f"{entry['id']:18s} {'installed':10s} {runnable:13s} "
            f"encoder={entry['encoder_tier'] or '-'}"
        )
        # Only a pack that is installed and still cannot run needs explaining;
        # repeating the install instructions once per absent pack turns an
        # eight-line listing into forty and buries the two lines that matter.
        if entry["reason"]:
            print(f"{'':18s} {entry['reason']}")
    if missing:
        print(f"\n{missing} pack(s) not installed. {cache.INSTALL_INSTRUCTIONS}")
    return 0


def cmd_models_verify(args: argparse.Namespace) -> int:
    """Re-hash installed packs against the digests recorded when they landed."""
    _prepare_storage_only(_resolve_data_dir(args))

    from quantem.registry import cache

    pack_ids = args.packs or cache.installed_packs()
    if not pack_ids:
        print(f"no packs installed under {cache.packs_root()}")
        return 0
    failed = 0
    for pack_id in pack_ids:
        try:
            results = cache.verify_pack(pack_id)
        except cache.PackNotInstalled as exc:
            print(f"{pack_id:18s} {exc}", file=sys.stderr)
            failed += 1
            continue
        bad = [name for name, ok in results.items() if not ok]
        failed += bool(bad)
        print(f"{pack_id:18s} {'OK' if not bad else f'MISMATCH {bad}'}")
    return 1 if failed else 0


# --- Argument parsing -------------------------------------------------------


def _data_dir_parent() -> argparse.ArgumentParser:
    """``--data-dir`` for a subcommand, in a way that does not clobber the root's.

    ``default=SUPPRESS`` is the whole trick: without it argparse writes the
    subparser's default over a value the user gave before the subcommand, and
    ``quantem-app --data-dir X serve`` would silently serve from the default
    directory instead.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--data-dir",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="where the database, models and exports live. May be given before "
        "or after the subcommand. Defaults to $QUANTEM_DATA_DIR, then the "
        f"installation's own data directory ({default_data_dir()}).",
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    """The whole command line, as a parser.

    Separate from :func:`main` so the argument *layout* -- in particular that
    ``--data-dir`` is honoured on either side of every subcommand -- can be
    asserted without starting a server.
    """
    dd = _data_dir_parent()
    verbose = argparse.ArgumentParser(add_help=False)
    verbose.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(
        prog="quantem-app",
        description=f"{APP_NAME} desktop application",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="where the database, models and exports live. Accepted here or "
        "after the subcommand. Defaults to $QUANTEM_DATA_DIR, then "
        f"{default_data_dir()}.",
    )
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", parents=[dd], help="open the application (default)")
    r.add_argument("--port", type=int, default=0)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", parents=[dd], help="run only the local server, no window")
    s.add_argument("--port", type=int, default=0)
    s.set_defaults(func=cmd_serve)

    from quantem.registry.cache import INSTALL_INSTRUCTIONS

    m = sub.add_parser(
        "models",
        parents=[dd],
        help="install and inspect the pretrained model packs",
        description=(
            "Model weights are not shipped with the application; they are "
            "downloaded on demand from the QuantEM Hugging Face repository, or "
            "installed offline from a downloaded release bundle."
        ),
        epilog=INSTALL_INSTRUCTIONS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    msub = m.add_subparsers(dest="models_command")

    mi = msub.add_parser(
        "install",
        parents=[dd, verbose],
        help="download packs from Hugging Face, or install an unzipped release bundle",
        description=(
            "Name pack ids (e.g. quantem:mito) to download them from the QuantEM "
            "Hugging Face repository at a pinned revision; every artifact's sha256 "
            "is verified before the pack is installed, and the encoder is exported "
            "to TorchScript so the pack runs with no research dependency. "
            "Give the directory of an unzipped QuantEM model release instead to "
            "install offline: every file is then re-hashed against the release's "
            "MANIFEST.json."
        ),
    )
    mi.add_argument(
        "sources",
        nargs="*",
        help="pack ids to download (e.g. quantem:mito omniem:ld), or one release "
        "bundle directory (optionally followed by pack ids to take from it)",
    )
    mi.add_argument(
        "--all", action="store_true", help="download and install all eight released packs"
    )
    mi.add_argument(
        "--hf",
        action="store_true",
        help="treat every argument as a pack id to download (never a directory)",
    )
    mi.add_argument("--force", action="store_true", help="reinstall packs already present")
    mi.set_defaults(func=cmd_models_install)

    ml = msub.add_parser("list", parents=[dd], help="show what is installed and what can run")
    ml.set_defaults(func=cmd_models_list)

    mv = msub.add_parser("verify", parents=[dd], help="re-hash installed packs")
    mv.add_argument("packs", nargs="*")
    mv.set_defaults(func=cmd_models_verify)

    # `quantem-app models` with no action is a help request, not an error with a
    # stack trace; the subparser is kept so main() can print its help.
    m.set_defaults(_group_parser=m)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.command == "models" and not getattr(args, "models_command", None):
        args._group_parser.print_help()
        return 2
    if not getattr(args, "func", None):
        args = p.parse_args((argv or []) + ["run"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

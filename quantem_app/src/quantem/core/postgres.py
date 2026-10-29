"""Lifecycle manager for QuantEM's private PostgreSQL server.

The cluster, credentials, logs, and socket endpoint all live below the active
QuantEM data directory. The server binds only to loopback, is never registered
as an operating-system service, and is started before Django imports its
settings. PostgreSQL's own WAL recovery handles an interrupted prior launch;
QuantEM never falls back to an older SQLite database.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import re
import secrets
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from quantem.core.config import (
    POSTGRES_CLUSTER_DIR,
    POSTGRES_CREDENTIALS_PATH,
    POSTGRES_DIR,
    POSTGRES_LOCK_PATH,
    POSTGRES_LOG_PATH,
)
from quantem.core.startup_progress import (
    StartupProgressCallback,
    emit_startup_progress,
)

POSTGRES_VERSION = "17.11"
POSTGRES_MAJOR = "17"
DEFAULT_DATABASE = "quantem"
DEFAULT_USER = "quantem"

# A normal private-cluster start is effectively immediate. Crash recovery is
# different: replaying the WAL produced by a multi-gigabyte SQLite cutover can
# take minutes or hours on slower disks. Keep a short deadline for an ordinary
# server that never answers at all, but never terminate a live recovery because
# a disk-bound replay or checkpoint exceeded a wall-clock estimate.
POSTGRES_STARTUP_TIMEOUT_SECONDS = 60
POSTGRES_RECOVERY_STALL_WARNING_SECONDS = 30 * 60
POSTGRES_RECOVERY_HEARTBEAT_SECONDS = 15

# pg_ctl performs a clean checkpoint during a fast shutdown. That checkpoint
# can be as large as the preceding database import, so the desktop shell and
# this process must both allow it to finish instead of forcing crash recovery
# on the next launch.
POSTGRES_SHUTDOWN_TIMEOUT_SECONDS = 15 * 60

POSTGRES_RUNTIME_OPTIONS = (
    "max_wal_size=4GB",
    "min_wal_size=256MB",
    "checkpoint_timeout=15min",
    "checkpoint_completion_target=0.9",
)

_REDO_PROGRESS_RE = re.compile(
    r"redo in progress, elapsed time: [\d.]+ s, current LSN: ([0-9A-F]+/[0-9A-F]+)",
    re.IGNORECASE,
)
_REDO_DONE_RE = re.compile(r"redo done at ([0-9A-F]+/[0-9A-F]+)", re.IGNORECASE)


class PostgresError(RuntimeError):
    """The private database server could not be started or managed safely."""


class _RecoveryLogMonitor:
    """Read only new PostgreSQL log lines and retain recovery liveness."""

    def __init__(self, offset: int) -> None:
        self.offset = offset
        self.pending = b""
        self.phase = "starting"
        self.current_lsn: str | None = None
        self.detail = "PostgreSQL is preparing crash recovery."

    def poll(self) -> bool:
        try:
            size = POSTGRES_LOG_PATH.stat().st_size
            if size < self.offset:
                self.offset = 0
                self.pending = b""
            with POSTGRES_LOG_PATH.open("rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except OSError:
            return False
        if not chunk:
            return False

        content = self.pending + chunk
        lines = content.splitlines(keepends=True)
        if lines and not lines[-1].endswith((b"\n", b"\r")):
            self.pending = lines.pop()
        else:
            self.pending = b""

        changed = False
        for raw_line in lines:
            line = raw_line.decode("utf-8", errors="replace")
            redo = _REDO_PROGRESS_RE.search(line)
            if redo is not None:
                lsn = redo.group(1).upper()
                if self.phase != "redo" or self.current_lsn != lsn:
                    self.phase = "redo"
                    self.current_lsn = lsn
                    self.detail = f"Recovery position {lsn}"
                    changed = True
                continue

            redo_done = _REDO_DONE_RE.search(line)
            if redo_done is not None:
                lsn = redo_done.group(1).upper()
                if self.phase != "checkpoint" or self.current_lsn != lsn:
                    self.phase = "checkpoint"
                    self.current_lsn = lsn
                    self.detail = "WAL replay is complete; finalizing the recovery checkpoint."
                    changed = True
                continue

            if "checkpoint starting: end-of-recovery" in line.lower():
                if self.phase != "checkpoint":
                    self.phase = "checkpoint"
                    self.detail = "WAL replay is complete; finalizing the recovery checkpoint."
                    changed = True
                continue

            if self.phase == "checkpoint" and "checkpoint complete:" in line.lower():
                self.phase = "finalizing"
                self.detail = "The recovery checkpoint is complete; opening the database."
                changed = True

        return changed

    def event(
        self,
        *,
        started_at: float,
        last_progress_at: float,
        now: float,
        heartbeat: bool,
    ) -> dict[str, object]:
        return {
            "phase": self.phase,
            "message": "Recovering the database after an interrupted shutdown",
            "detail": self.detail,
            "current_lsn": self.current_lsn,
            "elapsed_seconds": round(max(0.0, now - started_at)),
            "last_progress_seconds": round(max(0.0, now - last_progress_at)),
            "heartbeat": heartbeat,
        }


def _emit_recovery_progress(
    event: dict[str, object],
    callback: StartupProgressCallback | None = None,
) -> None:
    """Send recovery liveness through the general startup protocol."""

    if callback is not None:
        callback(event)
        return
    emit_startup_progress(
        {
            **event,
            "category": "database-recovery",
            "critical": True,
            "can_close": False,
        }
    )


@dataclass(frozen=True)
class PostgresEndpoint:
    host: str
    port: int
    database: str
    user: str
    password: str

    def publish(self) -> None:
        values = {
            "QUANTEM_DATABASE_ENGINE": "postgresql",
            "QUANTEM_POSTGRES_HOST": self.host,
            "QUANTEM_POSTGRES_PORT": str(self.port),
            "QUANTEM_POSTGRES_DATABASE": self.database,
            "QUANTEM_POSTGRES_USER": self.user,
            "QUANTEM_POSTGRES_PASSWORD": self.password,
        }
        os.environ.update(values)


def _runtime_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("QUANTEM_POSTGRES_BIN_DIR", "").strip()
    if configured:
        candidates.append(Path(configured))
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root) / "postgresql" / "bin")
    executable_root = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            executable_root / "postgresql" / "bin",
            executable_root / "_internal" / "postgresql" / "bin",
            Path(__file__).resolve().parents[1] / "_postgresql" / "bin",
        ]
    )
    return candidates


def postgres_bin_dir() -> Path:
    executable = "postgres.exe" if os.name == "nt" else "postgres"
    for candidate in _runtime_candidates():
        if (candidate / executable).is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in _runtime_candidates())
    raise PostgresError(
        "QuantEM's private PostgreSQL runtime is missing. Reinstall the application "
        f"or set QUANTEM_POSTGRES_BIN_DIR to its bin directory. Searched: {searched}"
    )


def _tool(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = postgres_bin_dir() / f"{name}{suffix}"
    if not path.is_file():
        raise PostgresError(f"The PostgreSQL runtime is incomplete: {path.name} is missing.")
    return path


def postgres_tool(name: str) -> Path:
    """Return a validated executable from the packaged PostgreSQL runtime."""

    return _tool(name)


def postgres_runtime_environment(
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that resolves the packaged client/server libraries."""

    configured = (environment if environment is not None else os.environ).copy()
    binary_dir = postgres_bin_dir()
    library_dir = binary_dir.parent / "lib"
    configured["PATH"] = os.pathsep.join(
        part for part in (str(binary_dir), configured.get("PATH", "")) if part
    )
    if sys.platform == "darwin":
        configured["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            part for part in (str(library_dir), configured.get("DYLD_LIBRARY_PATH", "")) if part
        )
    elif os.name != "nt":
        configured["LD_LIBRARY_PATH"] = os.pathsep.join(
            part for part in (str(library_dir), configured.get("LD_LIBRARY_PATH", "")) if part
        )
    return configured


def _run(
    command: list[str | os.PathLike[str]],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(part) for part in command],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=postgres_runtime_environment(env),
        timeout=timeout,
        check=False,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )
    if check and completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise PostgresError(f"PostgreSQL command failed ({completed.returncode}): {detail}")
    return completed


def _check_runtime_version() -> None:
    result = _run([_tool("postgres"), "--version"])
    reported = result.stdout.strip()
    if f" {POSTGRES_VERSION}" not in reported:
        raise PostgresError(
            f"QuantEM requires PostgreSQL {POSTGRES_VERSION}; the packaged runtime reports "
            f"{reported or 'an unknown version'}."
        )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    with contextlib.suppress(OSError):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _load_or_create_credentials() -> dict[str, str]:
    if POSTGRES_CREDENTIALS_PATH.exists():
        try:
            payload = json.loads(POSTGRES_CREDENTIALS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PostgresError(
                f"PostgreSQL credentials are unreadable: {POSTGRES_CREDENTIALS_PATH}"
            ) from exc
        required = {"database", "user", "password", "postgres_major"}
        if not required.issubset(payload) or payload["postgres_major"] != POSTGRES_MAJOR:
            raise PostgresError(
                "The private PostgreSQL credentials do not match this database runtime."
            )
        return {key: str(value) for key, value in payload.items()}

    payload = {
        "database": DEFAULT_DATABASE,
        "user": DEFAULT_USER,
        "password": secrets.token_urlsafe(48),
        "postgres_major": POSTGRES_MAJOR,
    }
    _atomic_json(POSTGRES_CREDENTIALS_PATH, payload)
    return payload


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _postgres_runtime_options() -> tuple[str, ...]:
    from quantem.core.app_settings import load_app_settings

    cache_mb = load_app_settings().database_cache_mb
    if cache_mb is None:
        # In particular, upgrades retain pre-existing postgresql.auto.conf values.
        return POSTGRES_RUNTIME_OPTIONS
    return (*POSTGRES_RUNTIME_OPTIONS, f"shared_buffers={cache_mb}MB")


def _postgres_server_arguments(port: int) -> list[str]:
    """Return postgres arguments shared by new and upgraded clusters.

    The same settings are written when QuantEM creates a cluster, but launch
    arguments are intentional as well: an existing v0.1.8 cluster otherwise
    keeps PostgreSQL's smaller WAL defaults during the v0.1.9 cutover.
    """

    arguments = [
        "-D",
        str(POSTGRES_CLUSTER_DIR),
        "-p",
        str(port),
        "-h",
        "127.0.0.1",
    ]
    for option in _postgres_runtime_options():
        arguments.extend(("-c", option))
    return arguments


def _postgres_server_options(port: int) -> str:
    """Return the pg_ctl ``-o`` form of the managed server arguments."""

    options = [f"-p {port}", "-h 127.0.0.1"]
    options.extend(f"-c {option}" for option in _postgres_runtime_options())
    return " ".join(options)


def _cluster_requires_recovery() -> bool:
    """Return whether pg_control records an interrupted prior shutdown."""

    result = _run(
        [_tool("pg_controldata"), "-D", POSTGRES_CLUSTER_DIR],
        timeout=15,
    )
    prefix = "database cluster state:"
    for line in result.stdout.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith(prefix):
            state = normalized.removeprefix(prefix).strip()
            return state != "shut down"
    raise PostgresError(
        "PostgreSQL did not report the private cluster state; startup was stopped "
        "before changing the database."
    )


def _cluster_major() -> str | None:
    version_file = POSTGRES_CLUSTER_DIR / "PG_VERSION"
    if not version_file.exists():
        return None
    try:
        return version_file.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise PostgresError(
            f"Cannot read the PostgreSQL cluster version at {version_file}."
        ) from exc


def _initialize_cluster(credentials: dict[str, str]) -> None:
    existing_major = _cluster_major()
    if existing_major is not None:
        if existing_major != POSTGRES_MAJOR:
            raise PostgresError(
                f"The data directory contains PostgreSQL {existing_major}, but this QuantEM "
                f"build carries PostgreSQL {POSTGRES_MAJOR}. Follow the documented database "
                "major-version upgrade procedure; the old cluster was not changed."
            )
        return

    if POSTGRES_CLUSTER_DIR.exists() and any(POSTGRES_CLUSTER_DIR.iterdir()):
        raise PostgresError(
            f"The PostgreSQL cluster directory is non-empty but uninitialized: {POSTGRES_CLUSTER_DIR}"
        )
    POSTGRES_CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    password_file = POSTGRES_DIR / f".init-password-{os.getpid()}"
    try:
        password_file.write_text(credentials["password"], encoding="utf-8")
        with contextlib.suppress(OSError):
            password_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        _run(
            [
                _tool("initdb"),
                "-D",
                POSTGRES_CLUSTER_DIR,
                "--username",
                credentials["user"],
                "--pwfile",
                password_file,
                "--encoding=UTF8",
                "--locale=C",
                "--auth-host=scram-sha-256",
                "--auth-local=scram-sha-256",
            ],
            timeout=180,
        )
    finally:
        with contextlib.suppress(OSError):
            password_file.unlink()

    with (POSTGRES_CLUSTER_DIR / "postgresql.auto.conf").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n# Managed by QuantEM.\n"
            "listen_addresses = '127.0.0.1'\n"
            "max_connections = 32\n"
            "shared_buffers = '32MB'\n"
            # The SQLite cutover can write several gigabytes in one
            # transaction. PostgreSQL's 1 GB default forced repeated,
            # multi-minute checkpoints during that copy. Keep durability on,
            # but give the private cluster enough WAL headroom to checkpoint
            # smoothly instead of thrashing the user's disk.
            "max_wal_size = '4GB'\n"
            "min_wal_size = '256MB'\n"
            "checkpoint_timeout = '15min'\n"
            "checkpoint_completion_target = 0.9\n"
            "password_encryption = 'scram-sha-256'\n"
            "fsync = on\n"
            "full_page_writes = on\n"
            "logging_collector = off\n"
        )


def _recover_stale_postmaster() -> None:
    """Stop a server left by a crashed owner of this exact private cluster.

    The caller holds QuantEM's data-directory lock. A live application would
    still hold that lock, so a running postmaster at this point can only be a
    child orphaned when its owning sidecar died. This never inspects or stops a
    system service or a cluster from another data directory.
    """

    if not (POSTGRES_CLUSTER_DIR / "postmaster.pid").exists():
        return
    status = _run(
        [_tool("pg_ctl"), "-D", POSTGRES_CLUSTER_DIR, "status"],
        timeout=15,
        check=False,
    )
    if status.returncode != 0:
        return
    stopped = _run(
        [
            _tool("pg_ctl"),
            "-D",
            POSTGRES_CLUSTER_DIR,
            "-w",
            "-t",
            str(POSTGRES_SHUTDOWN_TIMEOUT_SECONDS),
            "stop",
            "-m",
            "fast",
        ],
        timeout=POSTGRES_SHUTDOWN_TIMEOUT_SECONDS + 15,
        check=False,
    )
    if stopped.returncode:
        detail = (stopped.stderr or stopped.stdout).strip()
        raise PostgresError(
            "A PostgreSQL process survived an interrupted QuantEM launch and could not "
            f"be stopped safely: {detail}"
        )


def _lock_instance() -> BinaryIO:
    POSTGRES_DIR.mkdir(parents=True, exist_ok=True)
    handle = POSTGRES_LOCK_PATH.open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise PostgresError(
            "Another QuantEM process is already using this data directory. "
            "Use a separate QUANTEM_DATA_DIR for an isolated instance."
        ) from exc
    return handle


class EmbeddedPostgres:
    """Own one private cluster for the lifetime of the QuantEM server process."""

    def __init__(self) -> None:
        self.endpoint: PostgresEndpoint | None = None
        self._lock_handle: BinaryIO | None = None
        self._started = False
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None

    def start(
        self,
        progress_callback: StartupProgressCallback | None = None,
    ) -> PostgresEndpoint:
        if self.endpoint is not None:
            return self.endpoint
        _check_runtime_version()
        self._lock_handle = _lock_instance()
        try:
            credentials = _load_or_create_credentials()
            _initialize_cluster(credentials)
            _recover_stale_postmaster()
            recovery_expected = _cluster_requires_recovery()
            port = _free_loopback_port()
            endpoint = PostgresEndpoint(
                host="127.0.0.1",
                port=port,
                database=os.environ.get("QUANTEM_POSTGRES_DATABASE", credentials["database"]),
                user=credentials["user"],
                password=credentials["password"],
            )
            endpoint.publish()
            POSTGRES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            recovery_log_offset = (
                POSTGRES_LOG_PATH.stat().st_size if POSTGRES_LOG_PATH.exists() else 0
            )
            if os.name == "nt":
                # EDB's pg_ctl tries to manufacture a restricted Windows token
                # when its parent already runs under a job/sandbox token. That
                # path fails with ERROR_INVALID_PARAMETER on some desktop
                # launchers. postgres.exe itself supports the same foreground
                # contract and remains in QuantEM's process tree.
                self._log_handle = POSTGRES_LOG_PATH.open("ab")
                self._process = subprocess.Popen(
                    [str(_tool("postgres")), *_postgres_server_arguments(port)],
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                _run(
                    [
                        _tool("pg_ctl"),
                        "-D",
                        POSTGRES_CLUSTER_DIR,
                        "-l",
                        POSTGRES_LOG_PATH,
                        "start",
                        # Return after launching; _wait_ready below owns the
                        # normal-vs-recovery deadlines on every platform.
                        "-W",
                        "-o",
                        _postgres_server_options(port),
                    ],
                    timeout=15,
                )
            self._started = True
            self.endpoint = endpoint
            self._wait_ready(
                endpoint,
                database="postgres",
                recovery_expected=recovery_expected,
                recovery_log_offset=recovery_log_offset,
                progress_callback=progress_callback,
            )
            if endpoint.database == credentials["database"]:
                self._ensure_application_database(endpoint)
            self._wait_ready(endpoint)
            _atomic_json(
                POSTGRES_DIR / "server.json",
                {
                    "database": endpoint.database,
                    "host": endpoint.host,
                    "last_port": endpoint.port,
                    "postgres_version": POSTGRES_VERSION,
                    "started_at_unix": time.time(),
                },
            )
            return endpoint
        except Exception:
            self.stop()
            raise

    def _command_env(self, endpoint: PostgresEndpoint) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PGHOST": endpoint.host,
                "PGPORT": str(endpoint.port),
                "PGUSER": endpoint.user,
                "PGPASSWORD": endpoint.password,
                "PGDATABASE": endpoint.database,
            }
        )
        return environment

    def _ensure_application_database(self, endpoint: PostgresEndpoint) -> None:
        probe = _run(
            [
                _tool("psql"),
                "--dbname=postgres",
                "--no-psqlrc",
                "--tuples-only",
                "--no-align",
                "--command",
                "SELECT 1 FROM pg_database WHERE datname = current_setting('quantem.database', true);",
            ],
            env={
                **self._command_env(endpoint),
                "PGDATABASE": "postgres",
                "PGOPTIONS": f"-c quantem.database={endpoint.database}",
            },
        )
        if probe.stdout.strip() == "1":
            return
        _run(
            [
                _tool("createdb"),
                "--host",
                endpoint.host,
                "--port",
                str(endpoint.port),
                "--username",
                endpoint.user,
                "--owner",
                endpoint.user,
                "--encoding=UTF8",
                endpoint.database,
            ],
            env=self._command_env(endpoint),
        )

    def _wait_ready(
        self,
        endpoint: PostgresEndpoint,
        *,
        database: str | None = None,
        recovery_expected: bool = False,
        recovery_log_offset: int = 0,
        progress_callback: StartupProgressCallback | None = None,
    ) -> None:
        started_at = time.monotonic()
        deadline = started_at + POSTGRES_STARTUP_TIMEOUT_SECONDS
        detail = "no readiness response"
        monitor = _RecoveryLogMonitor(recovery_log_offset)
        last_progress_at = started_at
        last_emit_at = started_at
        stall_warning_emitted = False
        if recovery_expected:
            _emit_recovery_progress(
                monitor.event(
                    started_at=started_at,
                    last_progress_at=last_progress_at,
                    now=started_at,
                    heartbeat=False,
                ),
                progress_callback,
            )
        while True:
            now = time.monotonic()
            if not recovery_expected and now >= deadline:
                raise PostgresError(
                    "The private PostgreSQL server did not become ready within "
                    f"{POSTGRES_STARTUP_TIMEOUT_SECONDS} seconds: {detail}"
                )
            progress_changed = recovery_expected and monitor.poll()
            if progress_changed:
                last_progress_at = now
                stall_warning_emitted = False
                _emit_recovery_progress(
                    monitor.event(
                        started_at=started_at,
                        last_progress_at=last_progress_at,
                        now=now,
                        heartbeat=False,
                    ),
                    progress_callback,
                )
                last_emit_at = now

            result = _run(
                [
                    _tool("pg_isready"),
                    "--host",
                    endpoint.host,
                    "--port",
                    str(endpoint.port),
                    "--dbname",
                    database or endpoint.database,
                    "--username",
                    endpoint.user,
                    "--timeout=2",
                ],
                env=self._command_env(endpoint),
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                if recovery_expected:
                    monitor.phase = "complete"
                    monitor.detail = "Database recovery completed successfully."
                    _emit_recovery_progress(
                        monitor.event(
                            started_at=started_at,
                            last_progress_at=now,
                            now=now,
                            heartbeat=False,
                        ),
                        progress_callback,
                    )
                return
            detail = (result.stderr or result.stdout).strip() or detail
            if self._process is not None and self._process.poll() is not None:
                detail = f"postgres exited with code {self._process.returncode}; {detail}"
                break
            if recovery_expected:
                stalled_for = now - last_progress_at
                if (
                    stalled_for >= POSTGRES_RECOVERY_STALL_WARNING_SECONDS
                    and not stall_warning_emitted
                ):
                    monitor.phase = "stalled"
                    monitor.detail = (
                        "PostgreSQL is still running but its recovery position has not "
                        "advanced for 30 minutes. QuantEM will keep waiting safely."
                    )
                    _emit_recovery_progress(
                        monitor.event(
                            started_at=started_at,
                            last_progress_at=last_progress_at,
                            now=now,
                            heartbeat=False,
                        ),
                        progress_callback,
                    )
                    last_emit_at = now
                    stall_warning_emitted = True
                if now - last_emit_at >= POSTGRES_RECOVERY_HEARTBEAT_SECONDS:
                    _emit_recovery_progress(
                        monitor.event(
                            started_at=started_at,
                            last_progress_at=last_progress_at,
                            now=now,
                            heartbeat=True,
                        ),
                        progress_callback,
                    )
                    last_emit_at = now
                # pg_isready is a subprocess. Back off once recovery has been
                # running long enough that sub-second polling cannot improve
                # perceived readiness but can add thousands of processes and
                # repetitive authentication noise to the PostgreSQL log.
                time.sleep(1.0 if now - started_at < 30 else 3.0)
            else:
                time.sleep(0.25 if now - started_at < 5 else 1.0)
        raise PostgresError("The private PostgreSQL server exited during startup: " + detail)

    def stop(self) -> None:
        if self._started:
            stopped = False
            try:
                result = _run(
                    [
                        _tool("pg_ctl"),
                        "-D",
                        POSTGRES_CLUSTER_DIR,
                        "-w",
                        "-t",
                        str(POSTGRES_SHUTDOWN_TIMEOUT_SECONDS),
                        "stop",
                        "-m",
                        "fast",
                    ],
                    timeout=POSTGRES_SHUTDOWN_TIMEOUT_SECONDS + 15,
                    check=False,
                )
                stopped = result.returncode == 0
            except Exception:
                stopped = False
            if self._process is not None:
                if stopped:
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self._process.wait(timeout=10)
                if self._process.poll() is None:
                    self._process.terminate()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self._process.wait(timeout=10)
                if self._process.poll() is None:
                    self._process.kill()
                    self._process.wait(timeout=10)
                self._process = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
        self._started = False
        self.endpoint = None
        if self._lock_handle is not None:
            self._lock_handle.close()
            self._lock_handle = None


_MANAGED_SERVER: EmbeddedPostgres | None = None


def start_managed_postgres(
    progress_callback: StartupProgressCallback | None = None,
) -> EmbeddedPostgres | None:
    """Start and publish the private server unless SQLite was explicitly selected."""

    global _MANAGED_SERVER
    engine = os.environ.get("QUANTEM_DATABASE_ENGINE", "postgresql").strip().lower()
    if engine == "sqlite":
        return None
    if os.environ.get("QUANTEM_POSTGRES_MANAGED", "1").strip().lower() in {"0", "false", "no"}:
        return None
    if _MANAGED_SERVER is None:
        _MANAGED_SERVER = EmbeddedPostgres()
        _MANAGED_SERVER.start(progress_callback=progress_callback)
        atexit.register(_MANAGED_SERVER.stop)
    return _MANAGED_SERVER


def stop_managed_postgres() -> None:
    """Stop the server created by this process, if any."""

    global _MANAGED_SERVER
    if _MANAGED_SERVER is not None:
        _MANAGED_SERVER.stop()
        _MANAGED_SERVER = None

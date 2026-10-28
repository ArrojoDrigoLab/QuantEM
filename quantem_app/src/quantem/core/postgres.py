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

POSTGRES_VERSION = "17.11"
POSTGRES_MAJOR = "17"
DEFAULT_DATABASE = "quantem"
DEFAULT_USER = "quantem"


class PostgresError(RuntimeError):
    """The private database server could not be started or managed safely."""


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
            "60",
            "stop",
            "-m",
            "fast",
        ],
        timeout=75,
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

    def start(self) -> PostgresEndpoint:
        if self.endpoint is not None:
            return self.endpoint
        _check_runtime_version()
        self._lock_handle = _lock_instance()
        try:
            credentials = _load_or_create_credentials()
            _initialize_cluster(credentials)
            _recover_stale_postmaster()
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
            if os.name == "nt":
                # EDB's pg_ctl tries to manufacture a restricted Windows token
                # when its parent already runs under a job/sandbox token. That
                # path fails with ERROR_INVALID_PARAMETER on some desktop
                # launchers. postgres.exe itself supports the same foreground
                # contract and remains in QuantEM's process tree.
                self._log_handle = POSTGRES_LOG_PATH.open("ab")
                self._process = subprocess.Popen(
                    [
                        str(_tool("postgres")),
                        "-D",
                        str(POSTGRES_CLUSTER_DIR),
                        "-p",
                        str(port),
                        "-h",
                        "127.0.0.1",
                    ],
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
                        "-w",
                        "-t",
                        "60",
                        "start",
                        "-o",
                        f"-p {port} -h 127.0.0.1",
                    ],
                    timeout=90,
                )
            self._started = True
            self.endpoint = endpoint
            self._wait_ready(endpoint, database="postgres")
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

    def _wait_ready(self, endpoint: PostgresEndpoint, *, database: str | None = None) -> None:
        deadline = time.monotonic() + 60
        detail = "no readiness response"
        while time.monotonic() < deadline:
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
                return
            detail = (result.stderr or result.stdout).strip() or detail
            if self._process is not None and self._process.poll() is not None:
                detail = f"postgres exited with code {self._process.returncode}; {detail}"
                break
            time.sleep(0.25)
        raise PostgresError(
            "The private PostgreSQL server did not become ready within 60 seconds: " + detail
        )

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
                        "60",
                        "stop",
                        "-m",
                        "fast",
                    ],
                    timeout=75,
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


def start_managed_postgres() -> EmbeddedPostgres | None:
    """Start and publish the private server unless SQLite was explicitly selected."""

    global _MANAGED_SERVER
    engine = os.environ.get("QUANTEM_DATABASE_ENGINE", "postgresql").strip().lower()
    if engine == "sqlite":
        return None
    if os.environ.get("QUANTEM_POSTGRES_MANAGED", "1").strip().lower() in {"0", "false", "no"}:
        return None
    if _MANAGED_SERVER is None:
        _MANAGED_SERVER = EmbeddedPostgres()
        _MANAGED_SERVER.start()
        atexit.register(_MANAGED_SERVER.stop)
    return _MANAGED_SERVER


def stop_managed_postgres() -> None:
    """Stop the server created by this process, if any."""

    global _MANAGED_SERVER
    if _MANAGED_SERVER is not None:
        _MANAGED_SERVER.stop()
        _MANAGED_SERVER = None

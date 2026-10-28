"""Verified PostgreSQL backup and non-destructive restore operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg import sql

from quantem.core.config import STORAGE_DIR
from quantem.core.postgres import (
    POSTGRES_VERSION,
    postgres_runtime_environment,
    postgres_tool,
)

DATABASE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")


class BackupError(RuntimeError):
    """A backup or restore was refused or failed validation."""


def _connection_parameters(*, database: str | None = None) -> dict[str, object]:
    required = {
        "host": os.environ.get("QUANTEM_POSTGRES_HOST"),
        "port": os.environ.get("QUANTEM_POSTGRES_PORT"),
        "user": os.environ.get("QUANTEM_POSTGRES_USER"),
        "password": os.environ.get("QUANTEM_POSTGRES_PASSWORD"),
        "dbname": database or os.environ.get("QUANTEM_POSTGRES_DATABASE"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise BackupError(f"PostgreSQL connection settings are missing: {', '.join(missing)}")
    required["port"] = int(str(required["port"]))
    return required


def _tool_environment(*, database: str | None = None) -> dict[str, str]:
    parameters = _connection_parameters(database=database)
    environment = os.environ.copy()
    environment.update(
        {
            "PGHOST": str(parameters["host"]),
            "PGPORT": str(parameters["port"]),
            "PGUSER": str(parameters["user"]),
            "PGPASSWORD": str(parameters["password"]),
            "PGDATABASE": str(parameters["dbname"]),
        }
    )
    return environment


def _run(command: list[object], *, database: str | None = None, timeout: int = 600) -> None:
    completed = subprocess.run(
        [str(part) for part in command],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=postgres_runtime_environment(_tool_environment(database=database)),
        timeout=timeout,
        check=False,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise BackupError(f"PostgreSQL backup tool failed ({completed.returncode}): {detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_postgres_backup(
    destination: Path | None = None,
    *,
    reason: str = "manual",
    pending_migrations: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write a custom-format dump and a checksum manifest atomically."""

    if destination is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        destination = STORAGE_DIR / "backups" / "postgresql" / f"quantem-{stamp}.dump"
    destination = Path(destination).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise BackupError(f"Refusing to overwrite an existing backup: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        _run(
            [
                postgres_tool("pg_dump"),
                "--format=custom",
                "--compress=6",
                "--no-owner",
                "--no-privileges",
                "--file",
                temporary,
            ]
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise BackupError("pg_dump completed without producing a backup archive")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest = destination.with_suffix(destination.suffix + ".json")
    payload = {
        "archive": destination.name,
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": _sha256(destination),
        "created_at": datetime.now(UTC).isoformat(),
        "database": os.environ["QUANTEM_POSTGRES_DATABASE"],
        "format": "postgresql-custom",
        "pending_migrations": pending_migrations or [],
        "postgres_version": POSTGRES_VERSION,
        "reason": reason,
    }
    temporary_manifest = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest)
    return destination, manifest


def _validate_archive(archive: Path) -> dict[str, object] | None:
    if not archive.is_file():
        raise BackupError(f"Backup archive does not exist: {archive}")
    manifest = archive.with_suffix(archive.suffix + ".json")
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError(f"Backup manifest is unreadable: {manifest}") from exc
    expected = payload.get("archive_sha256")
    actual = _sha256(archive)
    if expected != actual:
        raise BackupError(
            f"Backup checksum mismatch for {archive.name}: expected {expected}, got {actual}"
        )
    return payload


def restore_postgres_backup(archive: Path, *, target_database: str) -> dict[str, object]:
    """Restore into a newly-created database without replacing active data."""

    archive = Path(archive).expanduser().resolve(strict=True)
    manifest = _validate_archive(archive)
    if not DATABASE_NAME_RE.fullmatch(target_database):
        raise BackupError(
            "The restore database name must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores."
        )
    active = str(os.environ.get("QUANTEM_POSTGRES_DATABASE", ""))
    if target_database == active:
        raise BackupError("Restore must target a new database; the active database is unchanged.")

    admin = _connection_parameters(database="postgres")
    with psycopg.connect(**admin, autocommit=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", [target_database]
        ).fetchone()
        if exists:
            raise BackupError(
                f"Restore target already exists: {target_database}. Choose a new database name."
            )
        connection.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(target_database),
                sql.Identifier(str(admin["user"])),
            )
        )

    try:
        _run(
            [
                postgres_tool("pg_restore"),
                "--exit-on-error",
                "--single-transaction",
                "--no-owner",
                "--no-privileges",
                "--dbname",
                target_database,
                archive,
            ],
            database=target_database,
        )
        with psycopg.connect(**_connection_parameters(database=target_database)) as connection:
            migration_count = connection.execute(
                "SELECT count(*) FROM django_migrations"
            ).fetchone()[0]
            table_count = connection.execute(
                "SELECT count(*) FROM pg_catalog.pg_tables WHERE schemaname = 'public'"
            ).fetchone()[0]
    except (BackupError, psycopg.Error) as exc:
        raise BackupError(
            f"Restore failed; the newly created database {target_database!r} was retained "
            f"for inspection and the active database was unchanged: {exc}"
        ) from exc
    return {
        "archive": str(archive),
        "manifest": manifest,
        "migration_count": int(migration_count),
        "table_count": int(table_count),
        "target_database": target_database,
    }

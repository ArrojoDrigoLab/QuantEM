"""Non-destructive safeguards for application database migrations.

An application update may carry Django migrations.  QuantEM keeps the database
next to the user's images and models, so a migration snapshot must be small,
consistent, and must never copy or delete those assets. SQLite uses its backup
API; PostgreSQL uses a verified custom-format ``pg_dump`` archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from quantem._version import FALLBACK_VERSION
from quantem.core.app_settings import AppSettings, load_app_settings
from quantem.core.config import DB_PATH, STORAGE_DIR

if TYPE_CHECKING:
    from collections.abc import Sequence


SNAPSHOT_ROOT = STORAGE_DIR / "backups" / "pre-migration"


def pending_migration_labels() -> list[str]:
    """Return forward migrations the installed database still needs."""

    from django.db import connection
    from django.db.migrations.executor import MigrationExecutor

    executor = MigrationExecutor(connection)
    plan = executor.migration_plan(executor.loader.graph.leaf_nodes())
    return [
        f"{migration.app_label}.{migration.name}" for migration, backwards in plan if not backwards
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remove_sqlite_sidecars(path: Path) -> None:
    """Remove sidecars belonging only to a closed temporary SQLite copy."""

    for suffix in ("-wal", "-shm"):
        path.with_name(f"{path.name}{suffix}").unlink(missing_ok=True)


def _remove_snapshot(path: Path) -> None:
    """Remove only a validated snapshot child while enforcing retention."""

    root = SNAPSHOT_ROOT.resolve(strict=False)
    candidate = path.resolve(strict=False)
    if candidate.parent != root:
        raise ValueError(f"refusing to remove a path outside the snapshot root: {path}")
    shutil.rmtree(candidate)


def _snapshot_created_at(path: Path) -> datetime:
    """Read a snapshot's recorded creation time, falling back to directory mtime."""

    manifests = [path / "manifest.json", *sorted(path.glob("*.dump.json"))]
    for manifest in manifests:
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(str(payload["created_at"]))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            return created_at.astimezone(UTC)
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def prune_pre_migration_snapshots(
    *,
    app_settings: AppSettings | None = None,
    now: datetime | None = None,
) -> list[Path]:
    """Apply the configured count and age limits to migration snapshots.

    The newest snapshot is protected only while it is within the configured
    age. Cleanup runs on every launch, after every new snapshot, and whenever
    the user changes the policy in Settings.
    """

    if not SNAPSHOT_ROOT.is_dir():
        return []
    policy = app_settings or load_app_settings()
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = current_time - timedelta(days=policy.pre_migration_backup_retention_days)
    snapshots = sorted(
        (
            (_snapshot_created_at(entry), entry)
            for entry in SNAPSHOT_ROOT.iterdir()
            if entry.is_dir()
        ),
        key=lambda item: (item[0], item[1].name),
        reverse=True,
    )
    removed: list[Path] = []
    survivors: list[tuple[datetime, Path]] = []
    for created_at, snapshot in snapshots:
        if created_at <= cutoff:
            _remove_snapshot(snapshot)
            removed.append(snapshot)
        else:
            survivors.append((created_at, snapshot))
    for _created_at, stale in survivors[policy.pre_migration_backup_retention_count :]:
        _remove_snapshot(stale)
        removed.append(stale)
    return removed


def _snapshot_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    version = FALLBACK_VERSION.replace("/", "-")
    return f"{timestamp}-to-{version}"


def create_pre_migration_snapshot(pending: Sequence[str]) -> Path | None:
    """Snapshot the existing application database before migrations run.

    A new installation has no database to preserve.  Existing databases are
    copied with :meth:`sqlite3.Connection.backup`, not ``copy2``, because WAL
    pages may otherwise still be separate from the main database file.
    """

    if not pending:
        return None

    from django.db import connection

    if connection.vendor == "postgresql":
        if not connection.introspection.table_names():
            return None
        SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
        snapshot_dir = SNAPSHOT_ROOT / _snapshot_name()
        suffix = 1
        while snapshot_dir.exists():
            snapshot_dir = SNAPSHOT_ROOT / f"{_snapshot_name()}-{suffix}"
            suffix += 1
        snapshot_dir.mkdir()
        try:
            from quantem.core.postgres_backup import create_postgres_backup

            create_postgres_backup(
                snapshot_dir / "quantem.dump",
                reason="pre-migration",
                pending_migrations=list(pending),
            )
        except Exception:
            _remove_snapshot(snapshot_dir)
            raise
        prune_pre_migration_snapshots()
        return snapshot_dir

    if connection.vendor != "sqlite":
        raise RuntimeError(
            f"Unsupported database backend for migration snapshot: {connection.vendor}"
        )
    if not DB_PATH.is_file():
        return None

    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_dir = SNAPSHOT_ROOT / _snapshot_name()
    # Two rapid test/start cycles can land in one second.  Do not overwrite an
    # existing snapshot; select a deterministic suffix instead.
    suffix = 1
    while snapshot_dir.exists():
        snapshot_dir = SNAPSHOT_ROOT / f"{_snapshot_name()}-{suffix}"
        suffix += 1
    snapshot_dir.mkdir()

    destination = snapshot_dir / "quantem.sqlite3"
    temporary = snapshot_dir / ".quantem.sqlite3.tmp"
    source: sqlite3.Connection | None = None
    target: sqlite3.Connection | None = None
    try:
        source = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
        target = sqlite3.connect(temporary)
        source.backup(target)
        target.close()
        target = None
        source.close()
        source = None

        # A successful backup is expected to be internally consistent.  Check
        # before naming it as a recovery point, rather than discovering a bad
        # copy only after a failed migration.
        verifier = sqlite3.connect(f"file:{temporary.as_posix()}?mode=ro", uri=True)
        try:
            integrity = verifier.execute("PRAGMA integrity_check").fetchone()
        finally:
            verifier.close()
        if integrity != ("ok",):
            raise RuntimeError("SQLite integrity check failed for migration snapshot")

        _remove_sqlite_sidecars(temporary)
        os.replace(temporary, destination)
        _atomic_json(
            snapshot_dir / "manifest.json",
            {
                "app_version": FALLBACK_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "database": destination.name,
                "database_sha256": _sha256(destination),
                "pending_migrations": list(pending),
            },
        )
    except Exception:
        if source is not None:
            source.close()
        if target is not None:
            target.close()
        _remove_sqlite_sidecars(temporary)
        # The directory is our own freshly-created child and contains only a
        # failed temporary backup, never user data.
        _remove_snapshot(snapshot_dir)
        raise

    prune_pre_migration_snapshots()
    return snapshot_dir


def snapshot_before_pending_migrations() -> tuple[list[str], Path | None]:
    """Inspect the migration plan and create a recovery point when necessary."""

    pending = pending_migration_labels()
    snapshot = create_pre_migration_snapshot(pending)
    # No pending migration means no new snapshot, but expired snapshots must
    # still disappear. Running this on every launch gives the age policy a
    # reliable cleanup point even during long stretches without schema changes.
    if snapshot is None:
        prune_pre_migration_snapshots()
    return pending, snapshot

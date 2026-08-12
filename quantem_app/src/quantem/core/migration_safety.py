"""Non-destructive safeguards for application database migrations.

An application update may carry Django migrations.  QuantEM keeps the database
next to the user's images and models, so a migration snapshot must be small,
consistent, and must never copy or delete those assets.  SQLite's backup API
captures a transactionally consistent database even when WAL mode is active.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from quantem._version import FALLBACK_VERSION
from quantem.core.config import DB_PATH, STORAGE_DIR

if TYPE_CHECKING:
    from collections.abc import Sequence


SNAPSHOT_RETENTION = 3
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


def _remove_snapshot(path: Path) -> None:
    """Remove only a validated snapshot child while enforcing retention."""

    root = SNAPSHOT_ROOT.resolve(strict=False)
    candidate = path.resolve(strict=False)
    if candidate.parent != root:
        raise ValueError(f"refusing to remove a path outside the snapshot root: {path}")
    shutil.rmtree(candidate)


def _trim_snapshots() -> None:
    if not SNAPSHOT_ROOT.is_dir():
        return
    snapshots = sorted(
        (entry for entry in SNAPSHOT_ROOT.iterdir() if entry.is_dir()),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    for stale in snapshots[SNAPSHOT_RETENTION:]:
        _remove_snapshot(stale)


def _snapshot_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    version = FALLBACK_VERSION.replace("/", "-")
    return f"{timestamp}-to-{version}"


def create_pre_migration_snapshot(pending: Sequence[str]) -> Path | None:
    """Snapshot the existing SQLite database before ``pending`` migrations run.

    A new installation has no database to preserve.  Existing databases are
    copied with :meth:`sqlite3.Connection.backup`, not ``copy2``, because WAL
    pages may otherwise still be separate from the main database file.
    """

    if not pending or not DB_PATH.is_file():
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
        # The directory is our own freshly-created child and contains only a
        # failed temporary backup, never user data.
        _remove_snapshot(snapshot_dir)
        raise

    _trim_snapshots()
    return snapshot_dir


def snapshot_before_pending_migrations() -> tuple[list[str], Path | None]:
    """Inspect the migration plan and create a recovery point when necessary."""

    pending = pending_migration_labels()
    return pending, create_pre_migration_snapshot(pending)

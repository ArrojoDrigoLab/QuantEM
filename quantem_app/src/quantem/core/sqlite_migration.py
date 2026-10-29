"""Explicit, validated SQLite-to-PostgreSQL migration."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import sqlite3
import threading
import time as time_module
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from django.apps import apps
from django.core.management.color import no_style
from django.db import DatabaseError, connections, transaction
from django.db.migrations.executor import MigrationExecutor
from django.utils import timezone

from quantem._version import FALLBACK_VERSION
from quantem.core.config import DB_PATH, STORAGE_DIR


class SQLiteMigrationError(RuntimeError):
    """Source data is incompatible or the destination is not safe to populate."""


MigrationProgressCallback = Callable[[dict[str, object]], None]


class _MigrationProgress:
    """Thread-safe, throttled progress and liveness reporting for a cutover.

    The desktop shell cannot open its HTTP UI until the database is validated,
    so stdout is the only live channel available during this operation.  A
    heartbeat repeats the current phase during opaque SQLite/PostgreSQL calls;
    ``last_progress_seconds`` distinguishes that liveness signal from a real
    increase in measured work.
    """

    STAGE_COUNT = 10

    def __init__(
        self,
        callback: MigrationProgressCallback | None,
        *,
        emit_interval: float = 1.0,
        heartbeat_interval: float = 15.0,
        clock: Callable[[], float] = time_module.monotonic,
    ) -> None:
        self._callback = callback
        self._emit_interval = emit_interval
        self._heartbeat_interval = heartbeat_interval
        self._clock = clock
        now = clock()
        self._started_at = now
        self._stage_started_at = now
        self._last_progress_at = now
        self._last_emit_at = 0.0
        self._revision = 0
        self._payload: dict[str, object] = {
            "phase": "starting",
            "stage": 0,
            "stage_count": self.STAGE_COUNT,
            "message": "Preparing the database upgrade",
            "detail": "",
            "completed": None,
            "total": None,
            "unit": "",
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        if callback is not None and heartbeat_interval > 0:
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                name="quantem-migration-progress",
                daemon=True,
            )
            self._heartbeat.start()

    def __enter__(self) -> _MigrationProgress:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc is not None:
            self.stage(
                "failed",
                self.STAGE_COUNT,
                "The database upgrade stopped before completion",
                detail=str(exc),
            )
        self.close()

    def close(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=min(self._heartbeat_interval, 1.0))
            self._heartbeat = None

    def stage(
        self,
        phase: str,
        stage: int,
        message: str,
        *,
        detail: str = "",
        completed: int | None = None,
        total: int | None = None,
        unit: str = "",
    ) -> None:
        now = self._clock()
        with self._lock:
            self._stage_started_at = now
            self._last_progress_at = now
            self._revision += 1
            self._payload.update(
                {
                    "phase": phase,
                    "stage": stage,
                    "message": message,
                    "detail": detail,
                    "completed": completed,
                    "total": total,
                    "unit": unit,
                }
            )
        self._emit(force=True)

    def update(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        detail: str | None = None,
        unit: str | None = None,
        force: bool = False,
    ) -> None:
        now = self._clock()
        with self._lock:
            prior_completed = self._payload.get("completed")
            if completed is not None:
                self._payload["completed"] = completed
            if total is not None:
                self._payload["total"] = total
            if detail is not None:
                self._payload["detail"] = detail
            if unit is not None:
                self._payload["unit"] = unit
            if completed is not None and completed != prior_completed:
                self._last_progress_at = now
                self._revision += 1
        self._emit(force=force)

    def _snapshot(self, *, heartbeat: bool) -> dict[str, object]:
        now = self._clock()
        with self._lock:
            payload = dict(self._payload)
            completed = payload.get("completed")
            total = payload.get("total")
            stage_elapsed = max(0.0, now - self._stage_started_at)
            percent: float | None = None
            eta_seconds: int | None = None
            if isinstance(completed, int) and isinstance(total, int) and total > 0:
                percent = round(min(100.0, max(0.0, completed * 100.0 / total)), 1)
                if completed >= total:
                    eta_seconds = 0
                elif completed > 0 and stage_elapsed >= 5.0:
                    eta_seconds = max(1, round((total - completed) * stage_elapsed / completed))
            payload.update(
                {
                    "percent": percent,
                    "eta_seconds": eta_seconds,
                    "elapsed_seconds": round(max(0.0, now - self._started_at)),
                    "stage_elapsed_seconds": round(stage_elapsed),
                    "last_progress_seconds": round(max(0.0, now - self._last_progress_at)),
                    "heartbeat": heartbeat,
                    "revision": self._revision,
                }
            )
            return payload

    def _emit(self, *, force: bool = False, heartbeat: bool = False) -> None:
        if self._callback is None:
            return
        now = self._clock()
        with self._lock:
            if not force and now - self._last_emit_at < self._emit_interval:
                return
            self._last_emit_at = now
        try:
            self._callback(self._snapshot(heartbeat=heartbeat))
        except Exception:
            # UI/log progress must never be able to fail the data migration.
            return

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_interval):
            self._emit(force=True, heartbeat=True)


def _sha256(
    path: Path,
    *,
    progress: _MigrationProgress | None = None,
    detail: str = "",
) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    completed = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            completed += len(block)
            if progress is not None:
                progress.update(
                    completed=completed,
                    total=total,
                    detail=detail,
                    unit="bytes",
                    force=completed >= total,
                )
    return digest.hexdigest()


def _copy_file(
    source: Path,
    destination: Path,
    *,
    progress: _MigrationProgress | None = None,
    detail: str = "",
) -> None:
    """Copy one regular file while exposing byte progress, then preserve metadata."""

    total = source.stat().st_size
    completed = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            for block in iter(lambda: reader.read(16 * 1024 * 1024), b""):
                writer.write(block)
                completed += len(block)
                if progress is not None:
                    progress.update(
                        completed=completed,
                        total=total,
                        detail=detail,
                        unit="bytes",
                        force=completed >= total,
                    )
        shutil.copystat(source, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remove_sqlite_sidecars(path: Path) -> None:
    """Remove sidecars belonging only to a closed temporary SQLite copy."""

    for suffix in ("-wal", "-shm"):
        path.with_name(f"{path.name}{suffix}").unlink(missing_ok=True)


def create_consistent_sqlite_backup(
    source: Path,
    destination_root: Path | None = None,
    *,
    progress: _MigrationProgress | None = None,
) -> Path:
    """Create and verify a read-consistent SQLite backup; never alter ``source``."""

    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise SQLiteMigrationError(f"SQLite source is not a file: {source}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = destination_root or STORAGE_DIR / "backups" / "sqlite-migration" / stamp
    root = Path(root).expanduser().resolve(strict=False)
    if root.exists():
        raise SQLiteMigrationError(f"SQLite backup destination already exists: {root}")
    root.mkdir(parents=True)
    destination = root / "quantem.sqlite3"
    temporary = root / ".quantem.sqlite3.tmp"
    source_connection: sqlite3.Connection | None = None
    target_connection: sqlite3.Connection | None = None
    if progress is not None:
        progress.stage(
            "snapshot",
            1,
            "Creating a recovery snapshot",
            detail="Copying the live SQLite database without changing it",
            unit="pages",
        )
    try:
        source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
        target_connection = sqlite3.connect(temporary)

        def backup_progress(status: int, remaining: int, total: int) -> None:
            del status
            if progress is not None:
                progress.update(
                    completed=total - remaining,
                    total=total,
                    detail="Copying the live SQLite database without changing it",
                    unit="pages",
                    force=remaining == 0,
                )

        source_connection.backup(
            target_connection,
            pages=2048,
            progress=backup_progress,
        )
    except (OSError, sqlite3.Error) as exc:
        if target_connection is not None:
            target_connection.close()
            target_connection = None
        if source_connection is not None:
            source_connection.close()
            source_connection = None
        temporary.unlink(missing_ok=True)
        _remove_sqlite_sidecars(temporary)
        with contextlib.suppress(OSError):
            root.rmdir()
        raise SQLiteMigrationError(f"Could not create a consistent SQLite backup: {exc}") from exc
    finally:
        if target_connection is not None:
            target_connection.close()
        if source_connection is not None:
            source_connection.close()
        _remove_sqlite_sidecars(temporary)
    verifier: sqlite3.Connection | None = None
    if progress is not None:
        progress.stage(
            "snapshot-verify",
            2,
            "Verifying the recovery snapshot",
            detail="Checking SQLite integrity",
        )
    try:
        verifier = sqlite3.connect(f"file:{temporary.as_posix()}?mode=ro", uri=True)
        integrity = verifier.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.Error as exc:
        if verifier is not None:
            verifier.close()
            verifier = None
        temporary.unlink(missing_ok=True)
        _remove_sqlite_sidecars(temporary)
        with contextlib.suppress(OSError):
            root.rmdir()
        raise SQLiteMigrationError(f"Could not verify the SQLite backup: {exc}") from exc
    finally:
        if verifier is not None:
            verifier.close()
        _remove_sqlite_sidecars(temporary)
    if integrity != ("ok",):
        temporary.unlink(missing_ok=True)
        _remove_sqlite_sidecars(temporary)
        with contextlib.suppress(OSError):
            root.rmdir()
        raise SQLiteMigrationError(f"SQLite integrity check failed: {integrity!r}")
    os.replace(temporary, destination)
    if progress is not None:
        progress.stage(
            "snapshot-verify",
            2,
            "Verifying the recovery snapshot",
            detail="Computing the recovery snapshot checksum",
            completed=0,
            total=destination.stat().st_size,
            unit="bytes",
        )
    backup_sha256 = _sha256(
        destination,
        progress=progress,
        detail="Computing the recovery snapshot checksum",
    )
    manifest = {
        "backup": destination.name,
        "backup_sha256": backup_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "source_retained": True,
    }
    _atomic_json(root / "manifest.json", manifest)
    return destination


def _models_in_dependency_order() -> list[type]:
    models = [
        model
        for model in apps.get_models(include_auto_created=True)
        if model._meta.managed and not model._meta.proxy
    ]
    by_label = {model._meta.label_lower: model for model in models}
    dependencies: dict[str, set[str]] = defaultdict(set)
    dependents: dict[str, set[str]] = defaultdict(set)
    for model in models:
        label = model._meta.label_lower
        for field in model._meta.concrete_fields:
            related = getattr(field.remote_field, "model", None)
            related_label = getattr(getattr(related, "_meta", None), "label_lower", None)
            if related_label in by_label and related_label != label:
                dependencies[label].add(related_label)
                dependents[related_label].add(label)
    queue = deque(sorted(label for label in by_label if not dependencies[label]))
    ordered: list[type] = []
    while queue:
        label = queue.popleft()
        ordered.append(by_label[label])
        for dependent in sorted(dependents[label]):
            dependencies[dependent].discard(label)
            if not dependencies[dependent]:
                queue.append(dependent)
    if len(ordered) != len(models):
        unresolved = sorted(set(by_label) - {model._meta.label_lower for model in ordered})
        raise SQLiteMigrationError(
            "The model graph contains a dependency cycle that cannot be migrated safely: "
            + ", ".join(unresolved)
        )
    return ordered


def _validate_value(value: Any, *, location: str) -> None:
    if isinstance(value, str):
        if "\x00" in value:
            raise SQLiteMigrationError(f"PostgreSQL cannot store NUL text at {location}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise SQLiteMigrationError(f"PostgreSQL JSON cannot store {value!r} at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SQLiteMigrationError(f"JSON object key is not text at {location}: {key!r}")
            _validate_value(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_value(child, location=f"{location}[{index}]")


def _canonical(value: Any) -> Any:
    if isinstance(value, memoryview):
        value = bytes(value)
    if isinstance(value, bytes):
        return {"binary_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical(child) for key, child in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(child) for child in value]
    return value


def _model_fingerprint(
    model: type,
    *,
    alias: str,
    progress: _MigrationProgress | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    fields = list(model._meta.concrete_fields)
    queryset = model._base_manager.using(alias).order_by(model._meta.pk.attname)
    count = 0
    for row in queryset.values_list(*(field.attname for field in fields)).iterator(chunk_size=500):
        payload = [_canonical(value) for value in row]
        digest.update(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        )
        count += 1
        if progress is not None and (count % 500 == 0):
            progress.update(
                completed=progress_offset + count,
                total=progress_total,
                detail=f"Reading {model._meta.label_lower}",
                unit="records",
            )
    if progress is not None:
        progress.update(
            completed=progress_offset + count,
            total=progress_total,
            detail=f"Read {model._meta.label_lower}",
            unit="records",
            force=True,
        )
    return count, digest.hexdigest()


def _configure_source(backup: Path, alias: str) -> None:
    connections.databases[alias] = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(backup),
        "OPTIONS": {"timeout": 30},
        "TIME_ZONE": "UTC",
        "CONN_HEALTH_CHECKS": False,
        "CONN_MAX_AGE": 0,
        "AUTOCOMMIT": True,
        "ATOMIC_REQUESTS": False,
    }


def _assert_source_not_ahead(executor: MigrationExecutor) -> None:
    known = set(executor.loader.graph.nodes)
    unknown = sorted(set(executor.loader.applied_migrations) - known)
    if unknown:
        labels = ", ".join(f"{app}.{name}" for app, name in unknown)
        raise SQLiteMigrationError(
            "The SQLite snapshot was written by a newer or incompatible QuantEM schema. "
            f"Unknown applied migrations: {labels}"
        )


def _assert_source_schema_current(alias: str) -> None:
    executor = MigrationExecutor(connections[alias])
    _assert_source_not_ahead(executor)
    pending = executor.migration_plan(executor.loader.graph.leaf_nodes())
    if pending:
        labels = ", ".join(f"{migration.app_label}.{migration.name}" for migration, _ in pending)
        raise SQLiteMigrationError(
            "The SQLite snapshot does not match this QuantEM version. Upgrade a copy with the "
            f"matching application before migrating. Pending: {labels}"
        )


def _upgrade_source_schema(
    alias: str,
    *,
    progress: _MigrationProgress | None = None,
) -> list[str]:
    """Advance a disposable SQLite copy through every skipped app migration."""

    connection = connections[alias]
    executor = MigrationExecutor(connection)
    _assert_source_not_ahead(executor)
    plan = executor.migration_plan(executor.loader.graph.leaf_nodes())
    backwards = [migration for migration, is_backwards in plan if is_backwards]
    if backwards:
        labels = ", ".join(f"{item.app_label}.{item.name}" for item in backwards)
        raise SQLiteMigrationError(
            "The SQLite snapshot would require reversing migrations, which automatic "
            f"upgrade never does: {labels}"
        )
    applied = [f"{migration.app_label}.{migration.name}" for migration, _ in plan]
    if progress is not None:
        progress.stage(
            "source-upgrade",
            3,
            "Preparing the SQLite snapshot",
            detail=(
                f"Applying {len(applied)} skipped schema migration(s) to the disposable copy"
                if applied
                else "The SQLite schema is already current"
            ),
        )
    if plan:
        executor.migrate(executor.loader.graph.leaf_nodes())
    verifier = MigrationExecutor(connection)
    _assert_source_not_ahead(verifier)
    remaining = verifier.migration_plan(verifier.loader.graph.leaf_nodes())
    if remaining:
        labels = ", ".join(f"{migration.app_label}.{migration.name}" for migration, _ in remaining)
        raise SQLiteMigrationError(
            f"The disposable SQLite upgrade did not reach the current schema: {labels}"
        )
    with connection.cursor() as cursor:
        if progress is not None:
            progress.update(
                detail="Checking the upgraded SQLite snapshot for corruption",
                force=True,
            )
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()
    if integrity != ("ok",):
        raise SQLiteMigrationError(
            f"SQLite integrity check failed after applying skipped migrations: {integrity!r}"
        )
    return applied


def _assert_target_pristine(alias: str) -> None:
    populated = []
    for label in (
        "assets.Asset",
        "library.Experiment",
        "library.Dataset",
        "segmentation.ImageSegmentation",
        "jobs.Job",
        "analysis.AnalysisRun",
        "finetune.Adapter",
        "assets.DatabaseImport",
    ):
        model = apps.get_model(label)
        count = model._base_manager.using(alias).count()
        if count:
            populated.append(f"{label}={count}")
    if populated:
        raise SQLiteMigrationError(
            "The PostgreSQL destination already contains user data and was not changed: "
            + ", ".join(populated)
        )


def _copy_model(
    model: type,
    *,
    source: str,
    target: str,
    progress: _MigrationProgress | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> int:
    fields = list(model._meta.concrete_fields)
    names = [field.attname for field in fields]
    batch: list[object] = []
    copied = 0
    # Django's SQL compiler calls Field.pre_save() even for bulk_create().
    # Disable timestamp automation temporarily or auto_now/auto_now_add would
    # replace the historical values this migration is required to preserve.
    automatic_timestamps = [
        (field, field.auto_now, field.auto_now_add)
        for field in fields
        if getattr(field, "auto_now", False) or getattr(field, "auto_now_add", False)
    ]
    for field, _, _ in automatic_timestamps:
        field.auto_now = False
        field.auto_now_add = False
    try:
        queryset = model._base_manager.using(source).order_by(model._meta.pk.attname)
        for values in queryset.values_list(*names).iterator(chunk_size=500):
            payload = dict(zip(names, values, strict=True))
            for name, value in payload.items():
                _validate_value(
                    value,
                    location=(f"{model._meta.label}.{payload.get(model._meta.pk.attname)}.{name}"),
                )
                if isinstance(value, memoryview):
                    payload[name] = bytes(value)
            batch.append(model(**payload))
            if len(batch) >= 500:
                model._base_manager.using(target).bulk_create(batch, batch_size=500)
                copied += len(batch)
                batch.clear()
                if progress is not None:
                    progress.update(
                        completed=progress_offset + copied,
                        total=progress_total,
                        detail=f"Copying {model._meta.label_lower}",
                        unit="records",
                    )
        if batch:
            model._base_manager.using(target).bulk_create(batch, batch_size=500)
            copied += len(batch)
            batch.clear()
        if progress is not None:
            progress.update(
                completed=progress_offset + copied,
                total=progress_total,
                detail=f"Copied {model._meta.label_lower}",
                unit="records",
                force=True,
            )
    finally:
        for field, auto_now, auto_now_add in automatic_timestamps:
            field.auto_now = auto_now
            field.auto_now_add = auto_now_add
    return copied


def _deactivate_copied_jobs(alias: str) -> list[str]:
    Job = apps.get_model("jobs", "Job")
    open_jobs = list(
        Job.objects.using(alias).filter(status__in=Job.OPEN_STATUSES).order_by("created_at")
    )
    if not open_jobs:
        return []
    now = timezone.now()
    message = (
        "This job was copied from a SQLite database and was not resumed. "
        "Start a new run if the work is still needed."
    )
    identifiers = [str(job.pk) for job in open_jobs]
    Job.objects.using(alias).filter(pk__in=[job.pk for job in open_jobs]).update(
        status="FAILED",
        finished_at=now,
        message=message,
        error_traceback=message,
        claimed_at=None,
        heartbeat_at=None,
    )
    return identifiers


def migrate_sqlite_to_postgres(
    source: Path,
    *,
    backup_root: Path | None = None,
    target_alias: str = "default",
    source_alias: str = "sqlite_migration_source",
    automatic: bool = False,
    progress_callback: MigrationProgressCallback | None = None,
) -> dict[str, object]:
    """Upgrade and copy a consistent SQLite snapshot into empty PostgreSQL."""

    with _MigrationProgress(progress_callback) as progress:
        report = _migrate_sqlite_to_postgres(
            source,
            backup_root=backup_root,
            target_alias=target_alias,
            source_alias=source_alias,
            automatic=automatic,
            progress=progress,
        )
        progress.stage(
            "complete",
            10,
            "Database migration complete",
            detail="The source SQLite database and verified snapshot were retained",
        )
        progress.update(completed=1, total=1, unit="step", force=True)
        return report


def _migrate_sqlite_to_postgres(
    source: Path,
    *,
    backup_root: Path | None,
    target_alias: str,
    source_alias: str,
    automatic: bool,
    progress: _MigrationProgress,
) -> dict[str, object]:
    target = connections[target_alias]
    if target.vendor != "postgresql":
        raise SQLiteMigrationError("The migration destination must be PostgreSQL.")
    _assert_target_pristine(target_alias)
    backup = create_consistent_sqlite_backup(source, backup_root, progress=progress)
    try:
        backup_manifest = json.loads((backup.parent / "manifest.json").read_text(encoding="utf-8"))
        backup_sha256 = str(backup_manifest["backup_sha256"])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SQLiteMigrationError("The verified SQLite backup manifest is unreadable.") from exc
    if len(backup_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in backup_sha256
    ):
        raise SQLiteMigrationError("The verified SQLite backup manifest has an invalid checksum.")
    working = backup.with_name(f".upgrade-working-{os.getpid()}.sqlite3")
    if working.exists():
        raise SQLiteMigrationError(f"SQLite upgrade workspace already exists: {working}")
    progress.stage(
        "source-upgrade",
        3,
        "Preparing the SQLite snapshot",
        detail="Creating a disposable working copy",
        completed=0,
        total=backup.stat().st_size,
        unit="bytes",
    )
    _copy_file(
        backup,
        working,
        progress=progress,
        detail="Creating a disposable working copy",
    )
    _configure_source(working, source_alias)
    source_connection = connections[source_alias]
    try:
        source_schema_upgrades = _upgrade_source_schema(source_alias, progress=progress)
        models = _models_in_dependency_order()
        progress.stage(
            "source-inventory",
            4,
            "Inventorying the SQLite data",
            completed=0,
            total=len(models),
            unit="tables",
        )
        source_counts: dict[str, int] = {}
        for index, model in enumerate(models, start=1):
            label = model._meta.label_lower
            source_counts[label] = model._base_manager.using(source_alias).count()
            progress.update(
                completed=index,
                total=len(models),
                detail=f"Counted {label}",
                unit="tables",
                force=index == len(models),
            )
        total_rows = sum(source_counts.values())

        progress.stage(
            "source-fingerprint",
            5,
            "Reading and fingerprinting the SQLite data",
            completed=0,
            total=total_rows,
            unit="records",
        )
        source_fingerprints: dict[str, tuple[int, str]] = {}
        offset = 0
        for model in models:
            label = model._meta.label_lower
            source_fingerprints[label] = _model_fingerprint(
                model,
                alias=source_alias,
                progress=progress,
                progress_offset=offset,
                progress_total=total_rows,
            )
            offset += source_counts[label]

        copied: dict[str, int] = {}
        with transaction.atomic(using=target_alias):
            progress.stage(
                "copy",
                6,
                "Copying records into PostgreSQL",
                completed=0,
                total=total_rows,
                unit="records",
            )
            with target.constraint_checks_disabled():
                for model in reversed(models):
                    model._base_manager.using(target_alias).all().delete()
                offset = 0
                for model in models:
                    label = model._meta.label_lower
                    copied[label] = _copy_model(
                        model,
                        source=source_alias,
                        target=target_alias,
                        progress=progress,
                        progress_offset=offset,
                        progress_total=total_rows,
                    )
                    offset += source_counts[label]
            progress.stage(
                "constraints",
                7,
                "Checking PostgreSQL relationships",
                detail=(
                    "This database operation does not expose intermediate counts; "
                    "the liveness timer will continue updating"
                ),
            )
            target.check_constraints()
            sequence_sql = target.ops.sequence_reset_sql(no_style(), models)
            with target.cursor() as cursor:
                for statement in sequence_sql:
                    cursor.execute(statement)
            progress.stage(
                "target-fingerprint",
                8,
                "Validating the PostgreSQL copy",
                detail="Comparing every stored field with the SQLite snapshot",
                completed=0,
                total=total_rows,
                unit="records",
            )
            target_fingerprints: dict[str, tuple[int, str]] = {}
            offset = 0
            for model in models:
                label = model._meta.label_lower
                target_fingerprints[label] = _model_fingerprint(
                    model,
                    alias=target_alias,
                    progress=progress,
                    progress_offset=offset,
                    progress_total=total_rows,
                )
                offset += source_counts[label]
            mismatches = {
                label: {"source": source_fingerprints[label], "target": target_fingerprints[label]}
                for label in source_fingerprints
                if source_fingerprints[label] != target_fingerprints[label]
            }
            if mismatches:
                raise SQLiteMigrationError(
                    "Content validation failed after copy: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            progress.stage(
                "commit",
                9,
                "Committing the validated PostgreSQL database",
                detail="All copied fields match the SQLite snapshot",
            )
            deactivated_jobs = _deactivate_copied_jobs(target_alias)
            DatabaseImport = apps.get_model("assets", "DatabaseImport")
            try:
                portable_snapshot = str(backup.parent.relative_to(STORAGE_DIR))
            except ValueError:
                portable_snapshot = str(backup.parent)
            import_record = DatabaseImport.objects.using(target_alias).create(
                source_backend="sqlite",
                source_sha256=backup_sha256,
                source_schema_migrations=source_schema_upgrades,
                snapshot_path=portable_snapshot,
                automatic=automatic,
                deactivated_job_ids=deactivated_jobs,
            )

        progress.update(detail="The PostgreSQL transaction committed successfully", force=True)
        report = {
            "backup": str(backup),
            "backup_sha256": backup_sha256,
            "completed_at": datetime.now(UTC).isoformat(),
            "copied_rows": copied,
            "deactivated_job_ids": deactivated_jobs,
            "database_import_id": str(import_record.pk),
            "automatic": automatic,
            "source_schema_upgrades": source_schema_upgrades,
            "source": str(Path(source).expanduser().resolve(strict=True)),
            "source_retained": True,
            "validated_models": len(models),
            "validation": "primary keys and canonical field content matched before job deactivation",
        }
        report_path = backup.parent / "migration-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["report"] = str(report_path)
        return report
    except SQLiteMigrationError:
        raise
    except (DatabaseError, TypeError, ValueError) as exc:
        raise SQLiteMigrationError(
            "The SQLite snapshot contains data PostgreSQL rejected; the destination "
            f"transaction was rolled back: {exc}"
        ) from exc
    finally:
        source_connection.close()
        for candidate in (
            working,
            Path(f"{working}-wal"),
            Path(f"{working}-shm"),
        ):
            candidate.unlink(missing_ok=True)


def _new_automatic_snapshot_root() -> Path:
    from quantem.core.migration_safety import SNAPSHOT_ROOT

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    version = FALLBACK_VERSION.replace("/", "-")
    candidate = SNAPSHOT_ROOT / f"{stamp}-to-{version}-sqlite-cutover"
    suffix = 1
    while candidate.exists():
        candidate = SNAPSHOT_ROOT / f"{stamp}-to-{version}-sqlite-cutover-{suffix}"
        suffix += 1
    return candidate


def _resolve_snapshot_directory(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = STORAGE_DIR / candidate
    return candidate.resolve(strict=False)


def _archive_legacy_sqlite_files(source: Path, snapshot_root: Path) -> list[str]:
    """Move the retired live files into their already-verified backup slot."""

    from quantem.core.migration_safety import SNAPSHOT_ROOT

    snapshot_root = snapshot_root.resolve(strict=True)
    if snapshot_root.parent != SNAPSHOT_ROOT.resolve(strict=False):
        raise SQLiteMigrationError(
            f"Refusing to archive SQLite outside the migration-backup root: {snapshot_root}"
        )
    original_root = snapshot_root / "original-live-files"
    original_root.mkdir(exist_ok=True)
    archived: list[str] = []
    for suffix in ("", "-wal", "-shm"):
        current = Path(f"{source}{suffix}")
        destination = original_root / current.name
        if current.exists() and destination.exists():
            raise SQLiteMigrationError(
                f"Both the live and archived SQLite files exist: {current} and {destination}"
            )
        if current.exists():
            os.replace(current, destination)
        if destination.exists():
            archived.append(str(destination.relative_to(snapshot_root)))
    return archived


def _finish_automatic_cutover(source: Path, snapshot_root: Path) -> list[str]:
    archived = _archive_legacy_sqlite_files(source, snapshot_root)
    manifest_path = snapshot_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SQLiteMigrationError(
            f"The completed migration snapshot manifest is unreadable: {manifest_path}"
        ) from exc
    manifest.update(
        {
            "backup_kind": "automatic-sqlite-to-postgresql-cutover",
            "original_live_files": archived,
            "source_archived": True,
        }
    )
    _atomic_json(manifest_path, manifest)
    return archived


def migrate_legacy_sqlite_on_startup(
    source: Path = DB_PATH,
    *,
    target_alias: str = "default",
    progress_callback: MigrationProgressCallback | None = None,
) -> dict[str, object] | None:
    """Perform or recover the one-time, unattended desktop database cutover."""

    target = connections[target_alias]
    if target.vendor != "postgresql":
        return None
    source = Path(source).expanduser().resolve(strict=False)
    DatabaseImport = apps.get_model("assets", "DatabaseImport")
    completed = (
        DatabaseImport.objects.using(target_alias)
        .filter(source_backend="sqlite", automatic=True)
        .order_by("-created_at")
        .first()
    )
    if completed is not None:
        if not source.exists():
            return None
        snapshot_root = _resolve_snapshot_directory(completed.snapshot_path)
        with _MigrationProgress(progress_callback) as progress:
            progress.stage(
                "archive",
                10,
                "Completing the verified database upgrade",
                detail="Archiving the retired SQLite files after the earlier PostgreSQL commit",
            )
            archived = _finish_automatic_cutover(source, snapshot_root)
            progress.update(completed=1, total=1, unit="step", force=True)
        return {
            "automatic": True,
            "database_import_id": str(completed.pk),
            "recovered_completed_import": True,
            "snapshot": str(snapshot_root),
            "archived_files": archived,
        }
    if not source.is_file():
        return None

    _assert_target_pristine(target_alias)
    snapshot_root = _new_automatic_snapshot_root()
    with _MigrationProgress(progress_callback) as progress:
        report = _migrate_sqlite_to_postgres(
            source,
            backup_root=snapshot_root,
            target_alias=target_alias,
            source_alias="sqlite_migration_source",
            automatic=True,
            progress=progress,
        )
        progress.stage(
            "archive",
            10,
            "Preserving the verified recovery backup",
            detail="Archiving the retired SQLite files only after the PostgreSQL commit",
        )
        report["archived_files"] = _finish_automatic_cutover(source, snapshot_root)
        report["snapshot"] = str(snapshot_root)
        progress.update(completed=1, total=1, unit="step", force=True)
        return report

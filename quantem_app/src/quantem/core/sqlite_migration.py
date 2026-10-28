"""Explicit, validated SQLite-to-PostgreSQL migration."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict, deque
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

from quantem.core.config import STORAGE_DIR


class SQLiteMigrationError(RuntimeError):
    """Source data is incompatible or the destination is not safe to populate."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_consistent_sqlite_backup(source: Path, destination_root: Path | None = None) -> Path:
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
    try:
        source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
        target_connection = sqlite3.connect(temporary)
        source_connection.backup(target_connection)
    except (OSError, sqlite3.Error) as exc:
        if target_connection is not None:
            target_connection.close()
            target_connection = None
        if source_connection is not None:
            source_connection.close()
            source_connection = None
        temporary.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            root.rmdir()
        raise SQLiteMigrationError(f"Could not create a consistent SQLite backup: {exc}") from exc
    finally:
        if target_connection is not None:
            target_connection.close()
        if source_connection is not None:
            source_connection.close()
    verifier: sqlite3.Connection | None = None
    try:
        verifier = sqlite3.connect(f"file:{temporary.as_posix()}?mode=ro", uri=True)
        integrity = verifier.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.Error as exc:
        if verifier is not None:
            verifier.close()
            verifier = None
        temporary.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            root.rmdir()
        raise SQLiteMigrationError(f"Could not verify the SQLite backup: {exc}") from exc
    finally:
        if verifier is not None:
            verifier.close()
    if integrity != ("ok",):
        temporary.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            root.rmdir()
        raise SQLiteMigrationError(f"SQLite integrity check failed: {integrity!r}")
    os.replace(temporary, destination)
    manifest = {
        "backup": destination.name,
        "backup_sha256": _sha256(destination),
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "source_retained": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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


def _model_fingerprint(model: type, *, alias: str) -> tuple[int, str]:
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


def _assert_source_schema_current(alias: str) -> None:
    executor = MigrationExecutor(connections[alias])
    pending = executor.migration_plan(executor.loader.graph.leaf_nodes())
    if pending:
        labels = ", ".join(f"{migration.app_label}.{migration.name}" for migration, _ in pending)
        raise SQLiteMigrationError(
            "The SQLite snapshot does not match this QuantEM version. Upgrade a copy with the "
            f"matching application before migrating. Pending: {labels}"
        )


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


def _copy_model(model: type, *, source: str, target: str) -> int:
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
                    location=(
                        f"{model._meta.label}.{payload.get(model._meta.pk.attname)}.{name}"
                    ),
                )
                if isinstance(value, memoryview):
                    payload[name] = bytes(value)
            batch.append(model(**payload))
            if len(batch) >= 500:
                model._base_manager.using(target).bulk_create(batch, batch_size=500)
                copied += len(batch)
                batch.clear()
        if batch:
            model._base_manager.using(target).bulk_create(batch, batch_size=500)
            copied += len(batch)
            batch.clear()
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
) -> dict[str, object]:
    """Copy a consistent SQLite snapshot into an empty PostgreSQL database."""

    target = connections[target_alias]
    if target.vendor != "postgresql":
        raise SQLiteMigrationError("The migration destination must be PostgreSQL.")
    backup = create_consistent_sqlite_backup(source, backup_root)
    _configure_source(backup, source_alias)
    source_connection = connections[source_alias]
    try:
        _assert_source_schema_current(source_alias)
        _assert_target_pristine(target_alias)
        models = _models_in_dependency_order()
        source_fingerprints = {
            model._meta.label_lower: _model_fingerprint(model, alias=source_alias)
            for model in models
        }
        copied: dict[str, int] = {}
        with transaction.atomic(using=target_alias):
            with target.constraint_checks_disabled():
                for model in reversed(models):
                    model._base_manager.using(target_alias).all().delete()
                for model in models:
                    copied[model._meta.label_lower] = _copy_model(
                        model, source=source_alias, target=target_alias
                    )
            target.check_constraints()
            sequence_sql = target.ops.sequence_reset_sql(no_style(), models)
            with target.cursor() as cursor:
                for statement in sequence_sql:
                    cursor.execute(statement)
            target_fingerprints = {
                model._meta.label_lower: _model_fingerprint(model, alias=target_alias)
                for model in models
            }
            mismatches = {
                label: {"source": source_fingerprints[label], "target": target_fingerprints[label]}
                for label in source_fingerprints
                if source_fingerprints[label] != target_fingerprints[label]
            }
            if mismatches:
                raise SQLiteMigrationError(
                    "Content validation failed after copy: " + json.dumps(mismatches, sort_keys=True)
                )
            deactivated_jobs = _deactivate_copied_jobs(target_alias)

        report = {
            "backup": str(backup),
            "backup_sha256": _sha256(backup),
            "completed_at": datetime.now(UTC).isoformat(),
            "copied_rows": copied,
            "deactivated_job_ids": deactivated_jobs,
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

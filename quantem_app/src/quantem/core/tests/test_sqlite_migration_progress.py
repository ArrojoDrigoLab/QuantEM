from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from quantem.cli import MIGRATION_PROGRESS_PREFIX, _emit_migration_progress
from quantem.core import postgres
from quantem.core.sqlite_migration import (
    _MigrationProgress,
    create_consistent_sqlite_backup,
)


def test_progress_reports_measured_percent_and_stage_eta() -> None:
    events: list[dict[str, object]] = []
    now = [100.0]
    progress = _MigrationProgress(
        events.append,
        emit_interval=0,
        heartbeat_interval=0,
        clock=lambda: now[0],
    )
    try:
        progress.stage(
            "copy",
            6,
            "Copying records into PostgreSQL",
            completed=0,
            total=100,
            unit="records",
        )
        now[0] += 10
        progress.update(completed=25, total=100, detail="Copying assets.asset", force=True)
    finally:
        progress.close()

    event = events[-1]
    assert event["phase"] == "copy"
    assert event["stage"] == 6
    assert event["stage_count"] == 10
    assert event["percent"] == 25.0
    assert event["eta_seconds"] == 30
    assert event["elapsed_seconds"] == 10
    assert event["last_progress_seconds"] == 0
    assert event["detail"] == "Copying assets.asset"


def test_consistent_backup_reports_copy_integrity_and_checksum_progress(tmp_path) -> None:
    source = tmp_path / "quantem.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE samples (id INTEGER PRIMARY KEY, payload BLOB)")
        connection.executemany(
            "INSERT INTO samples(payload) VALUES (?)",
            [(bytes([index % 256]) * 4096,) for index in range(32)],
        )

    events: list[dict[str, object]] = []
    progress = _MigrationProgress(events.append, emit_interval=0, heartbeat_interval=0)
    try:
        backup = create_consistent_sqlite_backup(
            source,
            tmp_path / "backup",
            progress=progress,
        )
    finally:
        progress.close()

    assert source.is_file()
    assert backup.is_file()
    assert [event["phase"] for event in events] == [
        "snapshot",
        "snapshot",
        "snapshot-verify",
        "snapshot-verify",
        "snapshot-verify",
    ]
    assert events[-1]["percent"] == 100.0
    assert events[-1]["completed"] == backup.stat().st_size
    assert events[-1]["unit"] == "bytes"
    manifest = json.loads((backup.parent / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["backup_sha256"]) == 64


def test_desktop_progress_event_is_one_prefixed_json_line(capsys) -> None:
    _emit_migration_progress(
        {
            "phase": "copy",
            "stage": 6,
            "stage_count": 10,
            "percent": 42.5,
        }
    )

    line = capsys.readouterr().out
    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert line.startswith(MIGRATION_PROGRESS_PREFIX)
    assert json.loads(line.removeprefix(MIGRATION_PROGRESS_PREFIX))["percent"] == 42.5


def test_new_private_cluster_has_cutover_wal_headroom(monkeypatch, tmp_path) -> None:
    postgres_root = tmp_path / "postgres"
    cluster = postgres_root / "cluster"
    monkeypatch.setattr(postgres, "POSTGRES_CLUSTER_DIR", cluster)
    monkeypatch.setattr(postgres, "POSTGRES_DIR", postgres_root)
    monkeypatch.setattr(postgres, "_tool", lambda name: Path(name))
    monkeypatch.setattr(
        postgres,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )

    postgres._initialize_cluster(
        {
            "database": "quantem",
            "user": "quantem",
            "password": "fixture-secret",
            "postgres_major": postgres.POSTGRES_MAJOR,
        }
    )

    configuration = (cluster / "postgresql.auto.conf").read_text(encoding="utf-8")
    assert "max_wal_size = '4GB'" in configuration
    assert "checkpoint_timeout = '15min'" in configuration
    assert "fsync = on" in configuration
    assert not list(postgres_root.glob(".init-password-*"))

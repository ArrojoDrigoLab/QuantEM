from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from quantem.core import db_config
from quantem.core.postgres import POSTGRES_VERSION, EmbeddedPostgres, PostgresEndpoint
from quantem.core.postgres_backup import DATABASE_NAME_RE
from quantem.core.sqlite_migration import SQLiteMigrationError, _canonical, _validate_value


def test_postgres_database_config_is_explicit(monkeypatch):
    values = {
        "QUANTEM_POSTGRES_DATABASE": "quantem_fixture",
        "QUANTEM_POSTGRES_USER": "quantem",
        "QUANTEM_POSTGRES_PASSWORD": "secret",
        "QUANTEM_POSTGRES_HOST": "127.0.0.1",
        "QUANTEM_POSTGRES_PORT": "54321",
    }
    monkeypatch.setenv("QUANTEM_DATABASE_ENGINE", "postgresql")
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    configured = db_config.get_database_config()

    assert configured["ENGINE"] == "django.db.backends.postgresql"
    assert configured["NAME"] == "quantem_fixture"
    assert configured["HOST"] == "127.0.0.1"
    assert configured["OPTIONS"]["sslmode"] == "disable"


def test_postgres_config_never_falls_back_to_sqlite(monkeypatch):
    monkeypatch.setenv("QUANTEM_DATABASE_ENGINE", "postgresql")
    for name in (
        "QUANTEM_POSTGRES_DATABASE",
        "QUANTEM_POSTGRES_USER",
        "QUANTEM_POSTGRES_PASSWORD",
        "QUANTEM_POSTGRES_HOST",
        "QUANTEM_POSTGRES_PORT",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="missing"):
        db_config.get_database_config()


def test_readiness_waits_through_recovery(monkeypatch):
    attempts = iter(
        [
            subprocess.CompletedProcess([], 1, "rejecting connections", ""),
            subprocess.CompletedProcess([], 0, "accepting connections", ""),
        ]
    )
    monkeypatch.setattr("quantem.core.postgres._run", lambda *args, **kwargs: next(attempts))
    monkeypatch.setattr("quantem.core.postgres._tool", lambda name: name)
    monkeypatch.setattr("quantem.core.postgres.time.sleep", lambda _: None)
    manager = EmbeddedPostgres()
    endpoint = PostgresEndpoint("127.0.0.1", 54321, "quantem", "quantem", "secret")

    manager._wait_ready(endpoint)


def test_migration_rejects_values_postgres_cannot_round_trip():
    with pytest.raises(SQLiteMigrationError, match="NUL"):
        _validate_value("not\x00text", location="fixture.text")
    with pytest.raises(SQLiteMigrationError, match="cannot store"):
        _validate_value({"bad": float("nan")}, location="fixture.json")


def test_binary_content_fingerprint_uses_length_and_digest():
    canonical = _canonical(memoryview(b"geometry-wkb"))
    assert canonical["bytes"] == 12
    assert len(canonical["binary_sha256"]) == 64


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("quantem_restore_20261028", True),
        ("quantem", True),
        ("QuantEM", False),
        ("1quantem", False),
        ("quantem-restore", False),
    ],
)
def test_restore_database_names_are_safe_identifiers(name, valid):
    assert bool(DATABASE_NAME_RE.fullmatch(name)) is valid


def test_desktop_runtime_manifest_matches_managed_server_version():
    app_root = Path(__file__).resolve().parents[4]
    specification = json.loads(
        (app_root / "packaging" / "postgresql" / "runtime.json").read_text(encoding="utf-8")
    )
    assert specification["postgresql_version"] == POSTGRES_VERSION
    assert set(specification["platforms"]) == {"windows-x64", "macos-universal"}
    assert all(
        len(platform_spec["sha256"]) == 64 for platform_spec in specification["platforms"].values()
    )

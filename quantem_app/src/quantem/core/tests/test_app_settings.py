from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from quantem.core import app_settings


def test_missing_settings_use_one_backup_for_thirty_days():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "settings.json"
        with patch.object(app_settings, "APP_SETTINGS_PATH", path):
            loaded = app_settings.load_app_settings()

    assert loaded.pre_migration_backup_retention_count == 1
    assert loaded.pre_migration_backup_retention_days == 30


def test_settings_round_trip_atomically():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "settings.json"
        with patch.object(app_settings, "APP_SETTINGS_PATH", path):
            saved = app_settings.update_app_settings(
                {
                    "pre_migration_backup_retention_count": 4,
                    "pre_migration_backup_retention_days": 90,
                }
            )
            loaded = app_settings.load_app_settings()
            payload = json.loads(path.read_text(encoding="utf-8"))

    assert saved == loaded
    assert payload == {
        "pre_migration_backup_retention_count": 4,
        "pre_migration_backup_retention_days": 90,
        "schema_version": 1,
        "database_cache_mb": None,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pre_migration_backup_retention_count", 0),
        ("pre_migration_backup_retention_days", 0),
        ("pre_migration_backup_retention_count", True),
        ("pre_migration_backup_retention_days", 30.5),
    ],
)
def test_invalid_settings_are_rejected(key, value):
    with pytest.raises(app_settings.AppSettingsValidationError):
        app_settings.update_app_settings({key: value})


def test_damaged_settings_keep_safe_defaults():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "settings.json"
        path.write_text("not-json", encoding="utf-8")
        with patch.object(app_settings, "APP_SETTINGS_PATH", path):
            loaded = app_settings.load_app_settings()

    assert loaded == app_settings.AppSettings()

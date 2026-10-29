from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from quantem.core import app_settings, database_cache, postgres


@pytest.mark.parametrize(
    ("ram_gib", "recommended", "maximum"),
    [
        (None, 256, 4096),
        (4, 128, 1024),
        (8, 256, 2048),
        (16, 512, 4096),
        (32, 1024, 8192),
        (64, 2048, 16384),
        (256, 4096, 65536),
    ],
)
def test_ram_policy_leaves_memory_for_image_processing(ram_gib, recommended, maximum):
    limits = database_cache.cache_limits(ram_gib * 1024**3 if ram_gib else None)
    assert limits["recommended_mb"] == recommended
    assert limits["max_mb"] == maximum


def test_upgrade_inherits_existing_cluster_setting_until_explicitly_changed(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"pre_migration_backup_retention_count": 3}')
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", settings_path)
    # No override on either Windows (argv) or macOS/Linux (pg_ctl -o).
    assert not any("shared_buffers" in x for x in postgres._postgres_server_arguments(54321))
    assert "shared_buffers" not in postgres._postgres_server_options(54321)
    app_settings.update_app_settings({"pre_migration_backup_retention_days": 60})
    assert "shared_buffers" not in postgres._postgres_server_options(54321)
    app_settings.update_app_settings({"database_cache_mb": 1024})
    assert "shared_buffers=1024MB" in postgres._postgres_server_arguments(54321)
    assert "-c shared_buffers=1024MB" in postgres._postgres_server_options(54321)
    assert app_settings.load_app_settings().pre_migration_backup_retention_count == 3


def test_installer_settings_are_consumed_without_a_first_launch_prompt(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text('{"schema_version": 1, "database_cache_mb": 512}')
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", path)
    assert "shared_buffers=512MB" in postgres._postgres_server_arguments(54321)
    assert app_settings.load_app_settings().pre_migration_backup_retention_days == 30


def test_existing_cluster_cache_file_is_not_rewritten_on_upgrade(tmp_path, monkeypatch):
    cluster = tmp_path / "cluster"
    cluster.mkdir()
    (cluster / "PG_VERSION").write_text(postgres.POSTGRES_MAJOR)
    config = cluster / "postgresql.auto.conf"
    original = "shared_buffers = '1GB'\nfsync = on\n"
    config.write_text(original)
    monkeypatch.setattr(postgres, "POSTGRES_CLUSTER_DIR", cluster)
    with patch.object(postgres, "_run") as run:
        postgres._initialize_cluster({})
    run.assert_not_called()
    assert config.read_text() == original


@pytest.mark.parametrize("value", [True, 0, 31, 256.5, "1024", 65537])
def test_invalid_cache_is_not_persisted(tmp_path, monkeypatch, value):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", path)
    with pytest.raises(app_settings.AppSettingsValidationError):
        app_settings.update_app_settings({"database_cache_mb": value})
    assert not path.exists()


def test_new_choice_is_bounded_by_ram_but_upgrades_keep_existing_choice(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"database_cache_mb": 4096}))
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", path)
    monkeypatch.setattr(app_settings, "machine_cache_limits", lambda: {"max_mb": 2048})
    assert app_settings.load_app_settings().database_cache_mb == 4096
    with pytest.raises(app_settings.AppSettingsValidationError, match="between 32 and 2048"):
        app_settings.update_app_settings({"database_cache_mb": 4096})
    app_settings.update_app_settings({"pre_migration_backup_retention_count": 2})
    assert app_settings.load_app_settings().database_cache_mb == 4096


def test_active_size_is_distinct_from_pending_size():
    connection = MagicMock(vendor="postgresql")
    connection.cursor.return_value.__enter__.return_value.fetchone.return_value = (1024,)
    with patch("django.db.connection", connection):
        inherited = database_cache.database_cache_status(None)
        pending = database_cache.database_cache_status(2048)
        unchanged = database_cache.database_cache_status(1024)
    assert inherited["active_mb"] == 1024
    assert not inherited["restart_required"]
    assert pending["restart_required"]
    assert not unchanged["restart_required"]
    for call in connection.cursor.return_value.__enter__.return_value.execute.call_args_list:
        assert call.args[0].startswith("SELECT ")

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from quantem.core import app_settings, migration_safety


class SystemSettingsTests(SimpleTestCase):
    def test_defaults_are_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            with patch.object(app_settings, "APP_SETTINGS_PATH", path):
                response = self.client.get("/api/system/settings/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "pre_migration_backup_retention_count": 1,
                "pre_migration_backup_retention_days": 30,
            },
        )

    def test_update_persists_and_applies_the_new_count_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings_path = root / "settings.json"
            snapshots = root / "backups" / "pre-migration"
            snapshots.mkdir(parents=True)
            for name in ("one", "two", "three"):
                snapshot = snapshots / name
                snapshot.mkdir()
                (snapshot / "manifest.json").write_text(
                    json.dumps({"created_at": datetime.now(UTC).isoformat()}),
                    encoding="utf-8",
                )

            with (
                patch.object(app_settings, "APP_SETTINGS_PATH", settings_path),
                patch.object(migration_safety, "SNAPSHOT_ROOT", snapshots),
            ):
                response = self.client.put(
                    "/api/system/settings/",
                    data=json.dumps(
                        {
                            "pre_migration_backup_retention_count": 2,
                            "pre_migration_backup_retention_days": 60,
                        }
                    ),
                    content_type="application/json",
                )
                persisted = app_settings.load_app_settings()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["removed_pre_migration_backups"], 1)
        self.assertEqual(persisted.pre_migration_backup_retention_count, 2)
        self.assertEqual(persisted.pre_migration_backup_retention_days, 60)

    def test_invalid_update_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            with patch.object(app_settings, "APP_SETTINGS_PATH", path):
                response = self.client.patch(
                    "/api/system/settings/",
                    data=json.dumps({"pre_migration_backup_retention_days": 0}),
                    content_type="application/json",
                )
                settings_file_created = path.exists()

        self.assertEqual(response.status_code, 400)
        self.assertIn("between 1 and", response.json()["error"])
        self.assertFalse(settings_file_created)

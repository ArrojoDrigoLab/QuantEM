import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from quantem.core import migration_safety
from quantem.core.app_settings import AppSettings


class MigrationSafetyTests(SimpleTestCase):
    def test_snapshot_is_consistent_and_never_copies_images_or_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "quantem.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE sample (value TEXT)")
            connection.execute("INSERT INTO sample VALUES ('before-migration')")
            connection.commit()
            connection.close()
            images = root / "data" / "images"
            images.mkdir(parents=True)
            (images / "image.tif").write_bytes(b"image-bytes")
            models = root / "models"
            models.mkdir()
            (models / "weights.safetensors").write_bytes(b"model-bytes")

            snapshot_root = root / "backups" / "pre-migration"
            with (
                patch.object(migration_safety, "DB_PATH", database),
                patch.object(migration_safety, "SNAPSHOT_ROOT", snapshot_root),
            ):
                snapshot = migration_safety.create_pre_migration_snapshot(["jobs.0004"])

            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(
                sorted(path.name for path in snapshot.iterdir()),
                ["manifest.json", "quantem.sqlite3"],
            )
            manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pending_migrations"], ["jobs.0004"])
            copied = sqlite3.connect(snapshot / "quantem.sqlite3")
            self.assertEqual(
                copied.execute("SELECT value FROM sample").fetchone(), ("before-migration",)
            )
            copied.close()
            self.assertFalse(any(path.name == "image.tif" for path in snapshot.rglob("*")))
            self.assertFalse(
                any(path.name == "weights.safetensors" for path in snapshot.rglob("*"))
            )

    def test_new_data_root_needs_no_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(migration_safety, "DB_PATH", root / "quantem.sqlite3"),
                patch.object(migration_safety, "SNAPSHOT_ROOT", root / "backups" / "pre-migration"),
            ):
                snapshot = migration_safety.create_pre_migration_snapshot(["jobs.0004"])
            self.assertIsNone(snapshot)

    def test_retention_keeps_the_newest_snapshot_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pre-migration"
            root.mkdir()
            for index in range(4):
                snapshot = root / f"snapshot-{index}"
                snapshot.mkdir()
                os.utime(snapshot, (index + 1, index + 1))

            with patch.object(migration_safety, "SNAPSHOT_ROOT", root):
                migration_safety.prune_pre_migration_snapshots(
                    app_settings=AppSettings(),
                    now=datetime(1970, 1, 2, tzinfo=UTC),
                )

            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["snapshot-3"],
            )

    def test_retention_removes_snapshots_at_the_configured_age(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pre-migration"
            root.mkdir()
            now = datetime(2026, 10, 28, tzinfo=UTC)
            old = root / "old"
            recent = root / "recent"
            old.mkdir()
            recent.mkdir()
            for snapshot, created_at in (
                (old, now - timedelta(days=30)),
                (recent, now - timedelta(days=29)),
            ):
                (snapshot / "manifest.json").write_text(
                    json.dumps({"created_at": created_at.isoformat()}),
                    encoding="utf-8",
                )

            with patch.object(migration_safety, "SNAPSHOT_ROOT", root):
                removed = migration_safety.prune_pre_migration_snapshots(
                    app_settings=AppSettings(
                        pre_migration_backup_retention_count=2,
                        pre_migration_backup_retention_days=30,
                    ),
                    now=now,
                )

            self.assertEqual([path.name for path in removed], ["old"])
            self.assertEqual([path.name for path in root.iterdir()], ["recent"])

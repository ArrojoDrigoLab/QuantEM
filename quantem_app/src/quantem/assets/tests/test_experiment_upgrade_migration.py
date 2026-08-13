"""The v0.1.2 upgrade files every existing active image into an experiment."""

from __future__ import annotations

from django.db import IntegrityError, connection, transaction
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

from quantem.assets.models import Asset

BEFORE = ("assets", "0005_asset_datasets_asset_experiment")
AFTER = ("assets", "0006_active_assets_require_experiment")


def _executor() -> MigrationExecutor:
    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    return executor


class ActiveAssetExperimentMigrationTests(TransactionTestCase):
    available_apps = None

    def tearDown(self):
        _executor().migrate([AFTER])

    def test_upgrade_creates_unique_display_name_experiments_and_enforces_them(self):
        executor = _executor()
        executor.migrate([BEFORE])
        old_apps = executor.loader.project_state([BEFORE]).apps
        OldAsset = old_apps.get_model("assets", "Asset")
        OldExperiment = old_apps.get_model("library", "Experiment")

        OldExperiment.objects.create(name="Shared name")
        first = OldAsset.objects.create(display_name="Shared name")
        second = OldAsset.objects.create(display_name="Shared name")
        deleted = OldAsset.objects.create(
            display_name="Deleted image",
            lifecycle_status="DELETED",
        )

        _executor().migrate([AFTER])

        names = [
            Asset.objects.get(id=first.id).experiment.name,
            Asset.objects.get(id=second.id).experiment.name,
        ]
        self.assertEqual(names, ["Shared name (2)", "Shared name (3)"])
        self.assertIsNone(Asset.objects.get(id=deleted.id).experiment_id)

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Asset.objects.filter(id=first.id).update(experiment=None)

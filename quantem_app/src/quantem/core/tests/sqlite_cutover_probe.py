"""Build and migrate a legacy SQLite fixture in a clean Django process."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch


def _legacy_sqlite_fixture(path: Path, alias: str) -> dict[str, uuid.UUID]:
    from django.db import connections
    from django.db.migrations.executor import MigrationExecutor
    from shapely.geometry import box

    from quantem.core import sqlite_migration

    sqlite_migration._configure_source(path, alias)
    source = connections[alias]
    executor = MigrationExecutor(source)
    targets = []
    for app_label, name in executor.loader.graph.leaf_nodes():
        if app_label == "assets":
            name = "0006_active_assets_require_experiment"
        elif app_label == "jobs":
            name = "0004_update_maintenance"
        elif app_label == "segmentation":
            name = "0009_correctionrevision"
        targets.append((app_label, name))
    executor.migrate(targets)
    historical = executor.loader.project_state(targets).apps

    Experiment = historical.get_model("library", "Experiment")
    Asset = historical.get_model("assets", "Asset")
    SegmentationType = historical.get_model("segmentation", "SegmentationType")
    LegacySegmentation = historical.get_model("segmentation", "ImageSegmentation")
    LegacyAreaObject = historical.get_model("segmentation", "AnalysisMaskObject")
    LegacyJob = historical.get_model("jobs", "Job")

    experiment = Experiment.objects.using(alias).create(name="Legacy experiment")
    asset = Asset.objects.using(alias).create(
        display_name="Legacy image",
        original_filename="legacy.tif",
        experiment_id=experiment.pk,
        logical_width=32,
        logical_height=32,
    )
    legacy_type = SegmentationType.objects.using(alias).create(
        internal_name="quantem_internal_analysis_mask",
        short_name="Analysis mask",
        long_name="Analysis Segmentation Mask",
        measurement_mode="global",
        default_color="#38bdf8",
    )
    segmentation = LegacySegmentation.objects.using(alias).create(
        asset_id=asset.pk,
        segmentation_type_id=legacy_type.pk,
        display_name="Cells",
        status_stage="CANDIDATES_READY",
    )
    area = LegacyAreaObject.objects.using(alias).create(
        segmentation_id=segmentation.pk,
        name="Object 1",
        color="#38bdf8",
        sort_order=1,
        geometry_wkb=box(2, 2, 12, 12).wkb,
    )
    job = LegacyJob.objects.using(alias).create(
        type="RUN_SEGMENTATION_FULL",
        status="PENDING",
        payload_json={"asset_id": str(asset.pk)},
    )
    source.close()
    return {
        "asset": asset.pk,
        "segmentation": segmentation.pk,
        "area": area.pk,
        "job": job.pk,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantem.core.settings")
    import django

    django.setup()

    from django.db import connections

    from quantem.core import app_settings, migration_safety, sqlite_migration

    identifiers = _legacy_sqlite_fixture(args.source, "legacy_cutover_fixture")
    with (
        patch.object(migration_safety, "SNAPSHOT_ROOT", args.snapshots),
        patch.object(app_settings, "APP_SETTINGS_PATH", args.settings),
    ):
        report = sqlite_migration.migrate_legacy_sqlite_on_startup(args.source)
    connections.close_all()
    args.report.write_text(
        json.dumps(
            {
                "report": report,
                "identifiers": {key: str(value) for key, value in identifiers.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Build and migrate a legacy SQLite fixture in a clean Django process."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch


def _legacy_sqlite_fixture(
    path: Path, alias: str, *, saved_results: bool = False
) -> dict[str, uuid.UUID]:
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
            name = "0011_model_application_batch" if saved_results else "0009_correctionrevision"
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
    identifiers = {
        "asset": asset.pk,
        "segmentation": segmentation.pk,
        "area": area.pk,
        "job": job.pk,
    }
    if saved_results:
        from django.utils import timezone

        mito_type, _ = SegmentationType.objects.using(alias).get_or_create(
            internal_name="quantem_internal_mito",
            defaults={"short_name": "Mito", "long_name": "Mitochondria"},
        )
        model_segmentation = LegacySegmentation.objects.using(alias).create(
            asset_id=asset.pk,
            segmentation_type_id=mito_type.pk,
            manual_full_image=True,
        )
        prediction = (
            historical.get_model("segmentation", "ProbabilityMap")
            .objects.using(alias)
            .create(
                segmentation_id=model_segmentation.pk,
                file_path="predictions/saved-model.npy",
                metadata={"source_model": "quantem:mito", "threshold": 0.73},
            )
        )
        SavedResult = historical.get_model("segmentation", "SavedSegmentationResult")
        for threshold, canonical in ((0.5, False), (0.73, True)):
            saved = SavedResult.objects.using(alias).create(
                segmentation_id=model_segmentation.pk,
                probability_map_id=prediction.pk,
                name=f"Saved threshold {threshold}",
                threshold=threshold,
                source_model="quantem:mito",
                full_image=True,
                is_canonical=canonical,
            )
            identifiers["canonical_result" if canonical else "old_result"] = saved.pk
        application = (
            historical.get_model("segmentation", "ModelApplicationBatch")
            .objects.using(alias)
            .create(
                experiment_id=experiment.pk,
                settings={"organelles": ["mito"], "threshold": 0.73},
                items=[{"asset_id": str(asset.pk), "job_id": str(job.pk), "ready": True}],
            )
        )
        analysis = (
            historical.get_model("analysis", "AnalysisRun")
            .objects.using(alias)
            .create(
                segmentation_id=model_segmentation.pk,
                status="SUCCESS",
                superseded_at=timezone.now(),
                input_identity={"asset_id": str(asset.pk)},
                results={"objects": {"n": 3}, "area_fraction_mito": 0.25},
            )
        )
        identifiers.update(
            model_segmentation=model_segmentation.pk,
            application=application.pk,
            analysis=analysis.pk,
        )
    source.close()
    return identifiers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--saved-results", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantem.core.settings")
    import django

    django.setup()

    from django.db import connections

    from quantem.core import app_settings, migration_safety, sqlite_migration

    identifiers = _legacy_sqlite_fixture(
        args.source, "legacy_cutover_fixture", saved_results=args.saved_results
    )
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

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from django.db import connection, connections
from shapely.geometry import box

from quantem.analysis.models import AnalysisRun
from quantem.assets.models import DatabaseImport
from quantem.jobs.models import Job
from quantem.segmentation.models import (
    AnalysisMaskObject,
    ImageSegmentation,
    ModelApplicationBatch,
    SavedSegmentationResult,
)

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(connection.vendor != "postgresql", reason="PostgreSQL integration test"),
]


@pytest.mark.parametrize("saved_results", [False, True], ids=["legacy-areas", "bulk-model-results"])
def test_first_start_upgrades_skipped_sqlite_schemas_and_archives_source(tmp_path, saved_results):
    source = tmp_path / "quantem.sqlite3"
    snapshots = tmp_path / "backups" / "pre-migration"
    settings_path = tmp_path / "settings.json"
    report_path = tmp_path / "cutover-report.json"
    env = os.environ.copy()
    env["QUANTEM_POSTGRES_DATABASE"] = str(connection.settings_dict["NAME"])
    connections.close_all()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "quantem.core.tests.sqlite_cutover_probe",
            "--source",
            str(source),
            "--snapshots",
            str(snapshots),
            "--settings",
            str(settings_path),
            "--report",
            str(report_path),
            *(["--saved-results"] if saved_results else []),
        ],
        check=True,
        env=env,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = payload["report"]
    identifiers = payload["identifiers"]

    assert report is not None
    assert report["automatic"] is True
    assert (
        "segmentation.0010_segmentation_definitions_and_area_instances"
        in report["source_schema_upgrades"]
    )
    assert not source.exists()
    snapshot = Path(report["snapshot"])
    assert (snapshot / "quantem.sqlite3").is_file()
    assert (snapshot / "original-live-files" / "quantem.sqlite3").is_file()
    assert (snapshot / "manifest.json").is_file()

    imported = DatabaseImport.objects.get(pk=report["database_import_id"])
    assert imported.automatic is True
    assert imported.source_sha256 == report["backup_sha256"]
    assert identifiers["job"] in imported.deactivated_job_ids

    segmentation = ImageSegmentation.objects.select_related("segmentation_type").get(
        pk=identifiers["segmentation"]
    )
    assert segmentation.segmentation_type.internal_name == "quantem_internal_cells"
    area = AnalysisMaskObject.objects.get(pk=identifiers["area"])
    assert area.name == "Cell 1"
    assert bytes(area.geometry_wkb) == box(2, 2, 12, 12).wkb

    copied_job = Job.objects.get(pk=identifiers["job"])
    assert copied_job.status == "FAILED"
    assert "was not resumed" in copied_job.message
    if saved_results:
        canonical = SavedSegmentationResult.objects.get(pk=identifiers["canonical_result"])
        assert canonical.is_canonical and canonical.full_image
        assert canonical.threshold == 0.73
        assert canonical.segmentation.manual_full_image
        assert canonical.probability_map.metadata["source_model"] == "quantem:mito"
        old = SavedSegmentationResult.objects.get(pk=identifiers["old_result"])
        assert not old.is_canonical and old.deleted_at is None
        application = ModelApplicationBatch.objects.get(pk=identifiers["application"])
        assert application.settings == {"organelles": ["mito"], "threshold": 0.73}
        assert application.items[0]["job_id"] == identifiers["job"]
        analysis = AnalysisRun.objects.get(pk=identifiers["analysis"])
        assert analysis.superseded_at is not None
        assert analysis.results == {"objects": {"n": 3}, "area_fraction_mito": 0.25}

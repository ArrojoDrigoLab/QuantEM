"""Recovery, sparse-image costs, and consistent revision semantics."""

import json
import zipfile
from unittest.mock import patch

import numpy as np
import pytest
from django.utils import timezone
from rest_framework.test import APIClient
from shapely.geometry import Polygon, box

from quantem.analysis.identity import input_fingerprint, run_is_current
from quantem.analysis.models import AnalysisRun
from quantem.analysis.serializers import AnalysisRunSerializer, AnalysisRunSummarySerializer
from quantem.core.local_storage import storage_path
from quantem.segmentation.area_instances import materialize_legacy_area_instances
from quantem.segmentation.area_pixels import count_geometry_pixels
from quantem.segmentation.artifact_cleanup import drain_retired_artifacts, retire_artifact
from quantem.segmentation.global_masks import (
    _rasterize_geometries,
    load_global_mask,
    save_global_mask,
)
from quantem.segmentation.models import (
    AnalysisMaskObject,
    CorrectionRevision,
    ImageSegmentation,
    RetiredResultArtifact,
    SegmentObjectAnnotation,
)
from quantem.segmentation.revisions import restore_revision, save_revision
from quantem.segmentation.tests.test_canonical_consistency import make_seg, object_at

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("tile_size", [3, 7, 32])
def test_tiled_area_count_matches_canonical_union_and_holes(tile_size):
    geometries = [
        Polygon(
            [(1.2, 1.1), (23.6, 2), (24, 26), (2, 27)], holes=[box(4, 4, 18, 18).exterior.coords]
        ),
        box(12, 10, 35, 40),
        box(-8, -8, 3, 3),
        None,
    ]
    expected = _rasterize_geometries(geometries, height=32, width=32)
    assert (
        count_geometry_pixels(geometries, height=32, width=32, tile_size=tile_size)
        == expected.sum()
    )


def test_sparse_montage_count_has_bounded_allocations():
    with patch("numpy.zeros", wraps=np.zeros) as allocate:
        assert (
            count_geometry_pixels(
                [box(800000, 800000, 800010, 800010)], height=1000000, width=1000000
            )
            == 100
        )
    assert allocate.call_count > 0
    assert all(np.prod(call.args[0]) <= 512 * 513 for call in allocate.call_args_list)


def test_importing_another_legacy_source_advances_adopted_area_revision():
    target = make_seg(area=True)
    target.area_vectors_authoritative = True
    target.save(update_fields=["area_vectors_authoritative"])
    source = ImageSegmentation.objects.create(
        asset=target.asset,
        segmentation_type=target.segmentation_type,
        display_name="Legacy source",
        superseded_by=target,
    )
    mask = np.zeros((64, 64), dtype=bool)
    mask[2:12, 2:12] = True
    save_global_mask(source, mask, source="manual")
    materialize_legacy_area_instances(target)
    target.refresh_from_db()
    assert target.scientific_revision == target.geometry_revision == 1
    assert AnalysisMaskObject.objects.filter(segmentation=target).count() == 1
    materialize_legacy_area_instances(target)
    target.refresh_from_db()
    assert target.scientific_revision == 1


def test_cleanup_checks_live_references_without_enumerating_all_artifacts():
    seg = make_seg(area=True)
    mask = save_global_mask(seg, np.ones((64, 64), dtype=bool), source="manual")
    path = storage_path(mask.file_path)
    retire_artifact(path)
    retire_artifact(path.parent, is_directory=True)
    with patch(
        "quantem.segmentation.artifact_cleanup.referenced_paths",
        side_effect=AssertionError("full scan"),
    ):
        assert drain_retired_artifacts() == 0
    assert path.is_file() and RetiredResultArtifact.objects.count() == 2
    # A similar prefix is not the same directory and must not pin unused files.
    unused = path.parent.with_name(path.parent.name + "-unused")
    unused.mkdir()
    retire_artifact(unused, is_directory=True)
    assert drain_retired_artifacts() == 1
    assert not unused.exists() and path.is_file()


@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "FAILED"])
def test_unfinished_runs_are_not_current_measurements(status):
    seg = make_seg()
    run = AnalysisRun.objects.create(
        segmentation=seg, status=status, input_fingerprint=input_fingerprint(seg, {})
    )
    assert not run_is_current(run)
    assert AnalysisRunSerializer().get_current(run) is None
    assert not AnalysisRunSummarySerializer().get_superseded(run)


def test_history_reuses_recipe_fingerprint_only_within_one_response():
    seg = make_seg()
    runs = [
        AnalysisRun.objects.create(
            segmentation=seg, status="SUCCESS", input_fingerprint=input_fingerprint(seg, {})
        )
        for _ in range(5)
    ]
    with patch(
        "quantem.analysis.identity.input_fingerprint", wraps=input_fingerprint
    ) as fingerprint:
        data = AnalysisRunSummarySerializer(runs, many=True).data
        assert fingerprint.call_count == 1
    assert not any(row["superseded"] for row in data)
    object_at(seg, box(2, 2, 12, 12))
    assert all(row["superseded"] for row in AnalysisRunSummarySerializer(runs, many=True).data)


def test_damaged_archive_is_a_client_error_and_preserves_current_state():
    seg = make_seg()
    obj = object_at(seg, box(2, 2, 12, 12))
    revision = save_revision(seg, "Damaged backup")
    storage_path(revision.archive_path).write_bytes(b"not a zip")
    response = APIClient().post(
        f"/api/segmentations/{seg.id}/revisions/", {"restore": str(revision.id)}, format="json"
    )
    assert response.status_code == 400
    assert "damaged" in str(response.data)
    obj.refresh_from_db()
    assert obj.superseded_at is None
    assert CorrectionRevision.objects.filter(segmentation=seg).count() == 1


def test_restore_repairs_missing_current_mask_and_labels_partial_backup():
    seg = make_seg(area=True)
    mask = np.zeros((64, 64), dtype=bool)
    mask[3:13, 3:13] = True
    first = save_global_mask(seg, mask, source="manual")
    revision = save_revision(seg, "Recoverable")
    storage_path(first.file_path).unlink()
    with pytest.raises(ValueError, match="mask is missing"):
        save_revision(seg, "Cannot promise missing pixels")
    with patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles"):
        restore_revision(revision)
    np.testing.assert_array_equal(load_global_mask(seg), mask)
    backup = CorrectionRevision.objects.exclude(pk=revision.pk).get(segmentation=seg)
    assert "mask unavailable" in backup.name
    with pytest.raises(ValueError, match="pixels cannot be restored"):
        restore_revision(backup)
    np.testing.assert_array_equal(load_global_mask(seg), mask)


def test_snapshot_and_fingerprint_exclude_inactive_object_history():
    seg = make_seg()
    current = object_at(seg, box(2, 2, 12, 12))
    inactive = object_at(seg, box(22, 22, 32, 32))
    inactive.superseded_at = timezone.now()
    inactive.save(update_fields=["superseded_at"])
    annotation = SegmentObjectAnnotation.objects.create(object=inactive, display_name="History")
    before = input_fingerprint(seg, {})
    annotation.display_name = "Old history"
    annotation.save()
    assert input_fingerprint(seg, {}) == before
    revision = save_revision(seg, "Current state")
    with zipfile.ZipFile(storage_path(revision.archive_path)) as archive:
        rows = archive.read("segmentobject.jsonl").splitlines()
        assert len(rows) == 1 and json.loads(rows[0])[0]["pk"] == str(current.pk)
        assert archive.read("segmentobjectannotation.jsonl") == b""


def test_scientific_revision_is_read_only_and_updates_api_identity():
    from quantem.segmentation.mutations import record_change
    from quantem.segmentation.serializers import ImageSegmentationSerializer

    seg = make_seg()
    previous_time = seg.updated_at
    record_change(seg, geometry=True)
    serializer = ImageSegmentationSerializer(
        seg, data={"scientific_revision": 99, "geometry_revision": 99}, partial=True
    )
    assert serializer.is_valid(), serializer.errors
    serializer.save()
    seg.refresh_from_db()
    assert seg.updated_at > previous_time
    assert serializer.data["scientific_revision"] == serializer.data["geometry_revision"] == 1


def test_revision_count_includes_area_objects():
    seg = make_seg(area=True)
    AnalysisMaskObject.objects.create(segmentation=seg, geometry=box(2, 2, 12, 12))
    assert save_revision(seg, "Cells").object_count == 1


def test_artifact_audit_reports_orphans_without_deleting_them():
    from io import StringIO

    from django.core.management import call_command

    seg = make_seg(area=True)
    mask = save_global_mask(seg, np.ones((64, 64), dtype=bool), source="manual")
    referenced = storage_path(mask.file_path)
    orphan = referenced.with_name("unreferenced.png")
    orphan.write_bytes(b"incomplete staged data")
    output = StringIO()
    call_command("audit_result_artifacts", min_age_hours=0, stdout=output)
    report = json.loads(output.getvalue())
    assert str(orphan) in report["unreferenced"]
    assert str(referenced) not in report["unreferenced"]
    assert orphan.is_file() and referenced.is_file()

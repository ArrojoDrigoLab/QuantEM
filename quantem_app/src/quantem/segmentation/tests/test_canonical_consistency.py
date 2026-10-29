from unittest.mock import patch

import numpy as np
import pytest
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.analysis.loaders import _settled_overlay_mask, segmentation_mask
from quantem.segmentation.coverage import coverage_state
from quantem.segmentation.models import (
    AnalysisMaskObject,
    AnalysisMaskObjectAnnotation,
    CompletedROI,
    ImageSegmentation,
    SavedSegmentationResult,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff import rebuild_overlay_full
from quantem.segmentation.revisions import restore_revision, save_revision
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image

pytestmark = pytest.mark.django_db


def test_drawing_api_records_scientific_geometry_revision():
    seg = make_seg()
    response = APIClient().post(
        f"/api/segmentations/{seg.pk}/segments/",
        {"geometry_coords": [[2, 2], [12, 2], [12, 12], [2, 12], [2, 2]]},
        format="json",
    )
    assert response.status_code == 201, response.content
    seg.refresh_from_db()
    assert seg.scientific_revision > 0 and seg.geometry_revision > 0


def test_scientific_revision_rolls_back_with_geometry():
    from django.db import transaction

    from quantem.segmentation.mutations import record_change

    seg = make_seg()
    with pytest.raises(RuntimeError), transaction.atomic():
        object_at(seg, box(2, 2, 12, 12))
        record_change(seg, geometry=True)
        raise RuntimeError("rollback")
    seg.refresh_from_db()
    assert seg.scientific_revision == seg.geometry_revision == 0
    assert not seg.segments.exists()


def test_overlay_certification_rejects_an_interleaved_geometry_edit():
    from quantem.segmentation.mutations import record_change

    seg = make_seg()
    obj = object_at(seg, box(2, 2, 12, 12))
    changed = False

    def edit_during_build(*_args):
        nonlocal changed
        if not changed:
            changed = True
            obj.geometry = box(32, 32, 48, 48)
            obj.centroid = obj.geometry.centroid
            obj.bbox = obj.geometry.envelope
            obj.save()
            record_change(seg, geometry=True)

    rebuild_overlay_full(seg, desired_revision=1, on_progress=edit_during_build)
    assert _settled_overlay_mask(seg, (64, 64)) is None
    rebuild_overlay_full(seg, desired_revision=1)
    cached = _settled_overlay_mask(seg, (64, 64))
    assert cached is not None and cached.sum() == 256


def test_revision_restores_canonical_identity_and_provenance():
    seg = make_seg()
    result = SavedSegmentationResult.objects.create(
        segmentation=seg,
        name="Accepted",
        full_image=True,
        is_canonical=True,
    )
    seg.final_result_provenance = {"result_id": str(result.pk), "threshold": 0.3}
    seg.save(update_fields=["final_result_provenance"])
    revision = save_revision(seg, "Accepted model")
    result.is_canonical = False
    result.save(update_fields=["is_canonical"])
    seg.final_result_provenance = {"later": True}
    seg.save(update_fields=["final_result_provenance"])
    with patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles"):
        restore_revision(revision)
    seg.refresh_from_db()
    result.refresh_from_db()
    assert result.is_canonical
    assert seg.final_result_provenance == {"result_id": str(result.pk), "threshold": 0.3}
    assert coverage_state(seg)["ready"]


def test_failed_file_cleanup_has_a_durable_retry(django_capture_on_commit_callbacks):
    from pathlib import Path

    from quantem.core.local_storage import storage_path
    from quantem.segmentation.artifact_cleanup import drain_retired_artifacts
    from quantem.segmentation.global_masks import save_global_mask
    from quantem.segmentation.models import RetiredResultArtifact

    seg = make_seg(area=True)
    first = save_global_mask(seg, np.ones((64, 64), dtype=bool), source="manual")
    path = storage_path(first.file_path)
    real_unlink = Path.unlink

    def locked_file(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("injected file lock")
        return real_unlink(candidate, *args, **kwargs)

    with (
        patch.object(Path, "unlink", locked_file),
        django_capture_on_commit_callbacks(execute=True),
    ):
        save_global_mask(seg, np.zeros((64, 64), dtype=bool), source="manual")
    assert path.exists()
    assert RetiredResultArtifact.objects.filter(
        path=first.file_path, last_error__contains="injected"
    ).exists()
    assert drain_retired_artifacts() == 1
    assert not path.exists()


def test_analysis_database_failure_removes_prepared_bundle():
    from django.db.models.query import QuerySet

    from quantem.analysis.global_runs import run_global_area
    from quantem.analysis.models import AnalysisRun
    from quantem.analysis.service import export_dir_for_run
    from quantem.segmentation.tests.test_er_global_apply import _Cancel

    seg = make_seg(area=True)
    AnalysisMaskObject.objects.create(segmentation=seg, geometry=box(2, 2, 12, 12))
    run = AnalysisRun.objects.create(segmentation=seg, params={"global_area": True})
    original = QuerySet.update

    def reject_success(queryset, **kwargs):
        if queryset.model is AnalysisRun and kwargs.get("status") == "SUCCESS":
            raise RuntimeError("injected database publication failure")
        return original(queryset, **kwargs)

    with (
        patch.object(QuerySet, "update", reject_success),
        pytest.raises(RuntimeError, match="publication failure"),
    ):
        run_global_area(run, progress=lambda *_: None, cancel_check=_Cancel().check_cancelled)
    run.refresh_from_db()
    assert run.status == "FAILED" and not run.export_dir
    assert not export_dir_for_run(run.pk).exists()


def test_unstamped_features_are_remeasured_and_centroids_follow_geometry():
    from quantem.analysis.loaders import (
        ensure_confirmed_object_measurements,
        object_centroids,
        object_features,
    )

    seg = make_seg()
    obj = object_at(seg, box(2, 2, 12, 12))
    obj.features = {"area": 999, "intensity_mean": 999}
    obj.save(update_fields=["features"])
    SegmentObject.objects.filter(pk=obj.pk).update(centroid_x=900, centroid_y=900)
    assert "area" not in object_features(seg)[str(obj.pk)]
    assert np.array_equal(object_centroids(seg), [[7, 7]])
    assert ensure_confirmed_object_measurements(seg) == 1
    assert object_features(seg)[str(obj.pk)]["area"] == 100
    assert ensure_confirmed_object_measurements(seg) == 0


def make_seg(*, area=False):
    asset = create_small_test_image("Canonical consistency", width=64, height=64).asset
    return ImageSegmentation.objects.create(
        asset=asset,
        segmentation_type=(
            get_or_create_tissue_type() if area else get_or_create_mitochondria_type()
        ),
        status_stage="CANDIDATES_READY",
    )


def object_at(seg, geom):
    return SegmentObject.objects.create(
        segmentation=seg,
        geometry=geom,
        centroid=geom.centroid,
        bbox=geom.envelope,
        label_state="CONFIRMED",
        source_model="manual",
    )


def test_restore_partial_snapshot_removes_later_manual_coverage():
    seg = make_seg()
    object_at(seg, box(2, 2, 12, 12))
    revision = save_revision(seg, "Partial")
    response = APIClient().post(
        f"/api/segmentations/{seg.id}/complete",
        {"acknowledge_full_manual": True, "acknowledge_no_model": True},
        format="json",
    )
    assert response.status_code == 200, response.content
    with patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles"):
        restore_revision(revision)
    seg.refresh_from_db()
    assert not CompletedROI.objects.filter(segmentation=seg).exists()
    assert not coverage_state(seg)["ready"], coverage_state(seg)


def test_restore_partial_snapshot_detaches_later_canonical_prediction():
    seg = make_seg()
    object_at(seg, box(2, 2, 12, 12))
    revision = save_revision(seg, "Partial")
    SavedSegmentationResult.objects.create(
        segmentation=seg,
        name="Later full image",
        full_image=True,
        is_canonical=True,
    )
    with patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles"):
        restore_revision(revision)
    seg.refresh_from_db()
    assert not coverage_state(seg)["ready"], coverage_state(seg)


def test_legacy_revision_never_inherits_later_whole_image_coverage():
    import zipfile

    from quantem.core.local_storage import storage_path

    seg = make_seg()
    object_at(seg, box(2, 2, 12, 12))
    revision = save_revision(seg, "Legacy partial")
    path = storage_path(revision.archive_path)
    with zipfile.ZipFile(path) as archive:
        members = {
            name: archive.read(name) for name in archive.namelist() if name != "revision.json"
        }
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    seg.manual_full_image = True
    seg.save(update_fields=["manual_full_image"])
    SavedSegmentationResult.objects.create(
        segmentation=seg, name="Later", full_image=True, is_canonical=True
    )
    with patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles"):
        restore_revision(revision)
    seg.refresh_from_db()
    assert not seg.manual_full_image
    assert not seg.saved_results.filter(is_canonical=True).exists()
    assert not coverage_state(seg)["ready"]


def test_restore_keeps_area_annotations():
    seg = make_seg(area=True)
    obj = AnalysisMaskObject.objects.create(segmentation=seg, geometry=box(2, 2, 12, 12), name="A")
    AnalysisMaskObjectAnnotation.objects.create(object=obj, tags=[{"name": "Zone", "value": "One"}])
    revision = save_revision(seg, "Tagged area")
    with patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles"):
        restore_revision(revision)
    assert AnalysisMaskObjectAnnotation.objects.filter(object_id=obj.pk).exists()


def test_deleting_last_saved_area_does_not_resurrect_raster():
    seg = make_seg(area=True)
    client = APIClient()
    url = f"/api/segmentations/{seg.id}/analysis-mask-objects/"
    created = client.post(
        url,
        {
            "operation": "include",
            "shapes": [{"rings": [[[4, 4], [24, 4], [24, 24], [4, 24], [4, 4]]]}],
        },
        format="json",
    )
    assert created.status_code == 201, created.content
    saved = client.post(url + "save/", format="json")
    assert saved.status_code == 200, saved.content
    deleted = client.delete(url + created.data["object"]["id"] + "/")
    assert deleted.status_code == 200, deleted.content
    assert not AnalysisMaskObject.objects.filter(segmentation=seg).exists()
    mask = segmentation_mask(seg, (64, 64))
    assert not mask.any(), f"Deleted area returned as {int(mask.sum())} foreground pixels"


def test_failed_confirmation_cannot_leave_settled_stale_analysis_raster():
    seg = make_seg()
    object_at(seg, box(2, 2, 12, 12))
    rebuild_overlay_full(seg, desired_revision=1)
    client = APIClient()
    with patch(
        "quantem.segmentation.api_views.segments.mutations.register_confirmation_overlay_mutation",
        side_effect=RuntimeError("injected invalidation failure"),
    ):
        with pytest.raises(RuntimeError, match="injected invalidation"):
            client.post(
                f"/api/segmentations/{seg.id}/segments/confirm-batch/",
                {
                    "segments": [
                        {"geometry_coords": [[32, 32], [48, 32], [48, 48], [32, 48], [32, 32]]}
                    ]
                },
                format="json",
            )
    cached = _settled_overlay_mask(seg, (64, 64))
    with patch("quantem.analysis.loaders._settled_overlay_mask", return_value=None):
        canonical = segmentation_mask(seg, (64, 64))
    assert cached is None or np.array_equal(cached, canonical), (
        f"Ready display has {int(cached.sum())} foreground pixels; canonical geometry has {int(canonical.sum())}"
    )


def test_zero_foreground_rethreshold_invalidates_clipped_protected_objects():
    from quantem.segmentation.models import SegmentationConfig
    from quantem.segmentation.tests.test_saved_results import replay, run

    image = create_small_test_image("Empty extraction", width=256, height=256, textured=True)
    image.asset.pixel_size_nm = 5
    image.asset.save(update_fields=["pixel_size_nm"])
    seg = ImageSegmentation.objects.create(
        asset=image.asset, segmentation_type=get_or_create_mitochondria_type()
    )
    SegmentationConfig.objects.create(segmentation=seg)
    run(seg)
    first = seg.segments.filter(superseded_at=None, label_state="CONFIRMED").first()
    x0, y0, x1, y1 = first.geometry.bounds
    CompletedROI.objects.create(segmentation=seg, geometry=box(x0, y0, (x0 + x1) / 2, y1))
    rebuild_overlay_full(seg, desired_revision=10)
    replay(seg, threshold=1)
    cached = _settled_overlay_mask(seg, (256, 256))
    with patch("quantem.analysis.loaders._settled_overlay_mask", return_value=None):
        canonical = segmentation_mask(seg, (256, 256))
    assert canonical.any()
    assert cached is None or np.array_equal(cached, canonical), (
        f"Ready display has {int(cached.sum())} foreground pixels; canonical geometry has {int(canonical.sum())}"
    )


def test_global_roi_rethreshold_preserves_completed_pixels():
    from types import SimpleNamespace

    from quantem.assets.models import ImageROI
    from quantem.jobs.handlers.rethreshold import handle_reextract_at_include_level
    from quantem.seg_core.db.prob_maps import save_probability_map
    from quantem.seg_core.types import InferenceResult
    from quantem.segmentation.global_masks import load_global_mask, save_global_mask
    from quantem.segmentation.tests.test_er_global_apply import _Cancel, _Reporter
    from quantem.segmentation.type_service import get_or_create_er_type

    seg = make_seg()
    seg.segmentation_type = get_or_create_er_type()
    seg.save(update_fields=["segmentation_type"])
    old = np.zeros((64, 64), dtype=bool)
    old[4:8, 4:8] = True
    save_global_mask(seg, old, source="manual")
    CompletedROI.objects.create(segmentation=seg, geometry=box(4, 4, 8, 8))
    roi = ImageROI.objects.create(asset=seg.asset, x=0, y=0, width=16, height=16)
    seg.rois.add(roi)
    probability = np.zeros((16, 16), dtype=np.float32)
    save_probability_map(seg, "DINO", probability, "er", "er_generated", roi_id=str(roi.pk))
    result = InferenceResult(prob_maps={"DINO": probability}, prob=probability)
    segmenter = SimpleNamespace(
        get_dl_model_names=lambda: ["DINO"],
        prob_map_prefix="er",
        source_model="quantem:er",
        _organelle=SimpleNamespace(close_radius=0),
        _foreground_mask=lambda values: values >= 0.5,
    )
    with (
        patch("quantem.seg_core.registry.get_segmenter_or_none", return_value=segmenter),
        patch(
            "quantem.seg_core.db.inference.replay_stored_probability_map",
            return_value=(result, None),
        ),
    ):
        handle_reextract_at_include_level(
            {
                "segmentation_id": str(seg.pk),
                "include_level": 0.5,
                "source_model": "quantem:er",
                "roi_id": str(roi.pk),
            },
            _Reporter(),
            _Cancel(),
        )
    assert np.array_equal(load_global_mask(seg)[4:8, 4:8], old[4:8, 4:8])


def test_analysis_mismatch_does_not_leave_published_unowned_bundle():
    from quantem.analysis.identity import input_fingerprint
    from quantem.analysis.job import run_job
    from quantem.analysis.models import AnalysisRun
    from quantem.analysis.service import export_dir_for_run
    from quantem.segmentation.tests.test_er_global_apply import _Cancel, _Reporter

    seg = make_seg(area=True)
    AnalysisMaskObject.objects.create(segmentation=seg, geometry=box(2, 2, 12, 12), name="A")
    params = {"global_area": True, "analysis_mask_ids": []}
    run = AnalysisRun.objects.create(
        segmentation=seg, params=params, input_fingerprint=input_fingerprint(seg, params)
    )
    observed = []

    def edit_before_final_check(running):
        running.refresh_from_db()
        observed.append(running.status)
        CompletedROI.objects.create(segmentation=seg, geometry=box(2, 2, 12, 12))

    with patch(
        "quantem.analysis.export_selection.select_export_files", side_effect=edit_before_final_check
    ):
        with pytest.raises(ValueError, match="Inputs changed during measurement"):
            run_job({"analysis_run_id": str(run.pk)}, _Reporter(), _Cancel())
    run.refresh_from_db()
    assert observed == ["RUNNING"]
    assert run.status == "FAILED" and not run.export_dir
    assert not export_dir_for_run(run.pk).exists(), (
        "Rejected run left a complete-looking export bundle"
    )


def test_interrupted_repeat_restore_does_not_corrupt_current_mask_file():
    from quantem.core.local_storage import storage_path
    from quantem.segmentation.global_masks import save_global_mask
    from quantem.segmentation.models import GlobalMask

    seg = make_seg(area=True)
    mask = np.zeros((64, 64), dtype=bool)
    mask[4:24, 4:24] = True
    save_global_mask(seg, mask, source="manual")
    revision = save_revision(seg, "Original mask")
    with patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles"):
        restore_revision(revision)
    record = GlobalMask.objects.get(segmentation=seg)
    current_path = storage_path(record.file_path)
    before = current_path.read_bytes()
    import shutil

    real_copy = shutil.copyfileobj

    def interrupted_copy(source, output, length=0):
        if "restore-" in str(getattr(output, "name", "")):
            output.write(source.read(12))
            raise OSError("injected interrupted restore")
        return real_copy(source, output, length)

    with patch("shutil.copyfileobj", side_effect=interrupted_copy):
        with pytest.raises(OSError, match="injected interrupted restore"):
            restore_revision(revision)
    record.refresh_from_db()
    assert storage_path(record.file_path) == current_path
    assert current_path.read_bytes() == before, (
        "Rollback kept the old DB pointer but truncated its mask"
    )


def test_revision_restores_a_tiff_mask():
    import tifffile

    from quantem.core.local_storage import storage_path, storage_relpath_for_path
    from quantem.segmentation.global_masks import load_global_mask, save_global_mask
    from quantem.segmentation.models import GlobalMask

    seg = make_seg(area=True)
    mask = np.zeros((64, 64), dtype=bool)
    mask[4:24, 4:24] = True
    mask[10:15, 10:15] = False
    original = save_global_mask(seg, mask, source="manual")
    # Archives may contain a TIFF regardless of the current writer's format.
    source = storage_path(original.file_path).with_suffix(".tif")
    tifffile.imwrite(source, mask.astype(np.uint8) * 255, compression="deflate")
    original.file_path = storage_relpath_for_path(source)
    original.save(update_fields=["file_path"])
    revision = save_revision(seg, "Streamed mask")
    changed = save_global_mask(seg, ~mask, source="manual")
    with patch("quantem.segmentation.overlay_ngff.mutations.register_overlay_mutation_all_bundles"):
        restore_revision(revision)
    seg.refresh_from_db()
    restored = GlobalMask.objects.get(segmentation=seg)
    assert restored.file_path not in {original.file_path, changed.file_path}
    np.testing.assert_array_equal(load_global_mask(seg), mask)

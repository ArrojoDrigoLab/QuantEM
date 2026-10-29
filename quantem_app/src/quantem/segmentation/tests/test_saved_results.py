from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.analysis.identity import input_fingerprint
from quantem.jobs.handlers.rethreshold import handle_reextract_at_include_level
from quantem.segmentation.completion import is_locked
from quantem.segmentation.coverage import coverage_state
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    SegmentationConfig,
)
from quantem.segmentation.organelle_tasks import run_segmentation_full_task
from quantem.segmentation.prob_maps.io import resolve_probability_map_path
from quantem.segmentation.result_reporting import InlineCancellation, InlineReporter
from quantem.segmentation.saved_results import publish_result
from quantem.segmentation.tests.test_threshold_replay import _fake_engine, _segmenter
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

pytestmark = pytest.mark.django_db


@pytest.fixture
def segmentation():
    image = create_small_test_image("Saved results", width=256, height=256, textured=True)
    image.asset.pixel_size_nm = 5
    image.asset.save(update_fields=["pixel_size_nm"])
    seg = ImageSegmentation.objects.create(
        asset=image.asset, segmentation_type=get_or_create_mitochondria_type()
    )
    SegmentationConfig.objects.create(segmentation=seg)
    return seg


def run(seg, threshold=0.5):
    with (
        patch("quantem.inference.segmenter.engine", _fake_engine()),
        patch(
            "quantem.segmentation.organelle_tasks.get_segmenter", return_value=_segmenter(threshold)
        ),
    ):
        run_segmentation_full_task(
            str(seg.id),
            "quantem_internal_mito",
            source_model="quantem:mito",
            force_recompute_prob_maps=True,
        )
    seg.refresh_from_db()
    return seg.saved_results.get(is_canonical=True)


def test_application_threshold_overrides_the_model_default(segmentation):
    with (
        patch("quantem.inference.segmenter.engine", _fake_engine()),
        patch("quantem.segmentation.organelle_tasks.get_segmenter", return_value=_segmenter(0.5)),
    ):
        run_segmentation_full_task(
            str(segmentation.id),
            "quantem_internal_mito",
            source_model="quantem:mito",
            force_recompute_prob_maps=True,
            threshold=0.73,
        )
    assert segmentation.saved_results.get(is_canonical=True).threshold == 0.73


def replay(seg, result=None, threshold=0.5, delete_ids=()):
    payload = {
        "segmentation_id": str(seg.id),
        "include_level": threshold,
        "source_model": "quantem:mito",
        "delete_result_ids": list(delete_ids),
    }
    if result:
        payload["saved_result_id"] = str(result.id)
    handle_reextract_at_include_level(payload, InlineReporter(), InlineCancellation())
    seg.refresh_from_db()


def outlines(seg):
    return sorted(
        bytes(value)
        for value in seg.segments.filter(superseded_at=None, label_state="CONFIRMED").values_list(
            "geometry_wkb", flat=True
        )
    )


def test_full_run_is_measurable_and_old_result_can_be_selected(segmentation):
    first = run(segmentation)
    expected = outlines(segmentation)
    assert expected
    assert coverage_state(segmentation)["ready"]
    assert not is_locked(segmentation)
    assert segmentation.final_result_provenance["quantem_version"]
    assert segmentation.final_result_provenance["finalized_at"]
    old_fingerprint = input_fingerprint(segmentation, {})
    replay(segmentation, threshold=0.85)
    assert segmentation.saved_results.count() == 2
    assert outlines(segmentation) != expected
    assert input_fingerprint(segmentation, {}) != old_fingerprint
    replay(segmentation, first)
    assert outlines(segmentation) == expected
    assert segmentation.saved_results.filter(deleted_at=None).count() == 2


def test_probability_bytes_are_not_overwritten_on_new_run(segmentation):
    first = run(segmentation)
    second = run(segmentation, 0.7)
    assert first.probability_map.file_path != second.probability_map.file_path
    assert first.probability_map.metadata == first.provenance


def test_zero_objects_still_establish_whole_image_coverage(segmentation):
    run(segmentation)
    replay(segmentation, threshold=1)
    assert coverage_state(segmentation)["ready"]
    assert outlines(segmentation) == []


def test_done_manual_roi_is_exactly_preserved(segmentation):
    run(segmentation)
    region = box(0, 0, 128, 256)
    CompletedROI.objects.create(segmentation=segmentation, geometry=region)
    existing = [
        row.geometry.intersection(region)
        for row in segmentation.segments.filter(superseded_at=None, label_state="CONFIRMED")
    ]
    from shapely.ops import unary_union

    before = unary_union(existing)
    replay(segmentation, threshold=0.9)
    after = unary_union(
        [
            row.geometry.intersection(region)
            for row in segmentation.segments.filter(superseded_at=None, label_state="CONFIRMED")
        ]
    )
    assert after.equals(before)
    assert (
        not segmentation.segments.filter(
            superseded_at=None,
            label_state="CONFIRMED",
        )
        .exclude(status=segmentation.segments.model.STATUS_CONFIRMED)
        .exists()
    )


def test_deletion_is_explicit_and_only_after_selection_succeeds(segmentation):
    first = run(segmentation)
    replay(segmentation, threshold=0.8)
    second = segmentation.saved_results.get(is_canonical=True)
    replay(segmentation, first, delete_ids=[str(second.id)])
    second.refresh_from_db()
    assert second.deleted_at is not None
    assert segmentation.saved_results.get(is_canonical=True).id == first.id


def test_confirmation_only_queues_work_until_replacement_succeeds(segmentation):
    first = run(segmentation)
    second = run(segmentation, 0.7)
    before = outlines(segmentation)
    paths = [resolve_probability_map_path(row.probability_map) for row in (first, second)]
    response = APIClient().post(
        f"/api/segmentations/{segmentation.id}/results/",
        {"result_id": str(first.id), "delete_result_ids": [str(second.id)]},
        format="json",
    )
    assert response.status_code == 202
    assert response.data["job_id"]
    assert segmentation.saved_results.get(is_canonical=True).id == second.id
    assert segmentation.saved_results.filter(deleted_at=None).count() == 2
    assert outlines(segmentation) == before
    assert all(path.is_file() for path in paths)


def test_old_files_are_deleted_only_after_replacement_commits(
    segmentation, django_capture_on_commit_callbacks
):
    first = run(segmentation)
    expected = outlines(segmentation)
    second = run(segmentation, 0.7)
    kept_path = resolve_probability_map_path(first.probability_map)
    deleted_path = resolve_probability_map_path(second.probability_map)
    from quantem.segmentation.prob_maps.preview import result_preview_cache_path

    response = APIClient().get(f"/api/segmentations/{segmentation.id}/results/{second.id}/preview/")
    assert response.status_code == 200
    b"".join(response.streaming_content)
    derived = result_preview_cache_path(deleted_path)
    assert derived.is_dir()
    with django_capture_on_commit_callbacks(execute=True):
        replay(segmentation, first, delete_ids=[str(second.id)])
        assert segmentation.saved_results.get(is_canonical=True).id == first.id
        assert outlines(segmentation) == expected
        assert coverage_state(segmentation)["ready"]
        # The transaction has published the replacement; disk cleanup still waits.
        assert kept_path.is_file()
        assert deleted_path.is_file()
    assert kept_path.is_file()
    assert not deleted_path.exists()
    assert not derived.exists()
    second.refresh_from_db()
    assert second.deleted_at is not None
    assert second.probability_map_id is None


def test_deleting_one_result_reclaims_only_its_cache_when_source_map_is_shared(
    segmentation, django_capture_on_commit_callbacks
):
    first = run(segmentation)
    replay(segmentation, threshold=0.7)
    second = segmentation.saved_results.get(is_canonical=True)
    assert second.probability_map_id == first.probability_map_id

    from quantem.segmentation.prob_maps.preview import result_preview_cache_path

    source = resolve_probability_map_path(second.probability_map)
    cache = result_preview_cache_path(source)
    detailed = cache / str(second.id) / ("a" * 32)
    detailed.mkdir(parents=True)
    (detailed / "ready.json").write_text("{}", encoding="utf-8")
    preview = cache / f"{second.id}.derived.png"
    preview.write_bytes(b"derived")

    with django_capture_on_commit_callbacks(execute=True):
        replay(segmentation, first, delete_ids=[str(second.id)])

    assert source.exists()
    assert first.probability_map_id is not None
    assert not (cache / str(second.id)).exists()
    assert not preview.exists()


def test_late_failure_rolls_back_confirmation_and_cancels_file_deletion(
    segmentation, django_capture_on_commit_callbacks
):
    first = run(segmentation)
    second = run(segmentation, 0.7)
    before = outlines(segmentation)
    paths = [resolve_probability_map_path(row.probability_map) for row in (first, second)]

    def fail_after_publishing(*args, **kwargs):
        publish_result(*args, **kwargs)
        raise RuntimeError("Interrupted before commit")

    with django_capture_on_commit_callbacks(execute=True) as callbacks:
        with patch(
            "quantem.segmentation.saved_results.publish_result", side_effect=fail_after_publishing
        ):
            with pytest.raises(RuntimeError, match="Interrupted before commit"):
                replay(segmentation, first, delete_ids=[str(second.id)])
    assert callbacks == []
    assert segmentation.saved_results.get(is_canonical=True).id == second.id
    assert segmentation.saved_results.filter(deleted_at=None).count() == 2
    assert outlines(segmentation) == before
    assert all(path.is_file() for path in paths)


def test_preview_leaves_foreground_opacity_to_each_comparison_pane(segmentation):
    result = run(segmentation)
    response = APIClient().get(f"/api/segmentations/{segmentation.id}/results/{result.id}/preview/")
    assert response.status_code == 200
    with Image.open(BytesIO(b"".join(response.streaming_content))) as image:
        alpha = np.asarray(image)[:, :, 3]
    assert set(np.unique(alpha)) == {0, 255}
    url = f"/api/segmentations/{segmentation.id}/results/{result.id}/preview/"
    with patch("quantem.segmentation.api_views.saved_results.decode_canonical_plane") as decode:
        cached = APIClient().get(url)
        assert cached.status_code == 200
        b"".join(cached.streaming_content)
        unchanged = APIClient().get(url, HTTP_IF_NONE_MATCH=response["ETag"])
        assert unchanged.status_code == 304
        decode.assert_not_called()


def test_manual_completion_requires_both_acknowledgements(segmentation):
    client = APIClient()
    url = f"/api/segmentations/{segmentation.id}/complete"
    assert client.post(url, {}, format="json").status_code == 400
    response = client.post(
        url, {"acknowledge_full_manual": True, "acknowledge_no_model": True}, format="json"
    )
    assert response.status_code == 200
    segmentation.refresh_from_db()
    assert coverage_state(segmentation) == {"ready": True, "kind": "manual", "result_id": None}
    assert not is_locked(segmentation)


def test_failed_replacement_keeps_canonical_geometry_and_history(segmentation):
    first = run(segmentation)
    expected = outlines(segmentation)
    with patch(
        "quantem.segmentation.saved_results.publish_result", side_effect=RuntimeError("Interrupted")
    ):
        with pytest.raises(RuntimeError, match="Interrupted"):
            replay(segmentation, threshold=0.9, delete_ids=[str(first.id)])
    assert outlines(segmentation) == expected
    assert segmentation.saved_results.get(is_canonical=True).id == first.id
    first.refresh_from_db()
    assert first.deleted_at is None


def test_global_mask_rollback_preserves_previous_bytes(segmentation):
    from django.db import transaction

    from quantem.segmentation.global_masks import load_global_mask, save_global_mask
    from quantem.segmentation.type_service import get_or_create_er_type

    seg = ImageSegmentation.objects.create(
        asset=segmentation.asset, segmentation_type=get_or_create_er_type()
    )
    original = np.zeros((256, 256), dtype=bool)
    original[10:30, 20:40] = True
    save_global_mask(seg, original, source="manual")
    with pytest.raises(RuntimeError):
        with transaction.atomic():
            save_global_mask(seg, ~original, source="model")
            raise RuntimeError("Interrupted")
    seg.refresh_from_db()
    assert np.array_equal(load_global_mask(seg), original)


def test_roi_can_start_empty_or_from_model_and_preserves_completed_areas(segmentation):
    from shapely.ops import unary_union

    from quantem.assets.models import ImageROI
    from quantem.segmentation.roi_initialization import initialize_roi

    run(segmentation)
    roi = ImageROI.objects.create(asset=segmentation.asset, x=0, y=0, width=256, height=128)
    before = outlines(segmentation)
    initialize_roi(segmentation, roi, prepopulate=True)
    assert outlines(segmentation) == before
    protected = box(0, 0, 128, 128)
    CompletedROI.objects.create(segmentation=segmentation, geometry=protected)
    existing = unary_union(
        [
            row.geometry
            for row in segmentation.segments.filter(superseded_at=None, label_state="CONFIRMED")
        ]
    )
    initialize_roi(segmentation, roi, prepopulate=False)
    remaining = unary_union(
        [
            row.geometry
            for row in segmentation.segments.filter(superseded_at=None, label_state="CONFIRMED")
        ]
    )
    assert remaining.intersection(protected).equals(existing.intersection(protected))
    assert remaining.intersection(box(128, 0, 256, 128)).area == 0

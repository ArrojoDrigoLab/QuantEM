from unittest.mock import patch

import numpy as np
import pytest
import zarr
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.jobs.constants import JOB_TYPE_BUILD_RESULT_COMPARISON
from quantem.jobs.models import Job
from quantem.segmentation.models import CompletedROI, SegmentObject
from quantem.segmentation.result_overlays import (
    build_comparison,
    comparison_directory,
    comparison_identity,
)
from quantem.segmentation.tests.test_saved_results import (
    outlines,
    replay,
    run,
)
from quantem.segmentation.tests.test_saved_results import (
    segmentation as segmentation_fixture,
)
from quantem.segmentation.tests.test_threshold_replay import _segmenter

pytestmark = pytest.mark.django_db
segmentation = segmentation_fixture


def build(result):
    result.refresh_from_db()
    key = comparison_identity(result)
    with patch("quantem.seg_core.registry.get_segmenter_or_none", return_value=_segmenter(0.5)):
        build_comparison(str(result.pk), key, cancel_check=lambda: None, progress=lambda *_: None)
    return key, comparison_directory(result, key) / "labels.zarr"


def read_mask(root):
    return np.asarray(zarr.open_array(str(root / "labels" / "0"), mode="r")) > 0


def test_detailed_alternative_matches_confirming_it_and_preserves_data(segmentation):
    first = run(segmentation)
    replay(segmentation, threshold=0.85)
    CompletedROI.objects.create(segmentation=segmentation, geometry=box(0, 0, 80, 80))
    geometry = box(10, 10, 18, 18)
    SegmentObject.objects.create(
        segmentation=segmentation,
        geometry=geometry,
        centroid=geometry.centroid,
        bbox=geometry.envelope,
        label_state="CONFIRMED",
        source_model="manual",
    )
    before = outlines(segmentation)
    before_count = SegmentObject.objects.filter(segmentation=segmentation).count()
    _, alternative = build(first)
    proposed = read_mask(alternative)
    assert outlines(segmentation) == before
    assert SegmentObject.objects.filter(segmentation=segmentation).count() == before_count
    assert proposed[12, 12]
    with patch("quantem.seg_core.registry.get_segmenter_or_none", return_value=_segmenter(0.5)):
        replay(segmentation, first)
    _, accepted = build(first)
    np.testing.assert_array_equal(proposed, read_mask(accepted))


def test_comparison_status_queues_once_and_serves_exact_generation(segmentation):
    result = run(segmentation)
    client = APIClient()
    url = f"/api/segmentations/{segmentation.id}/results/{result.id}/overlay/"
    assert client.get(url).data["status"] == "MISSING"
    assert not Job.objects.filter(type=JOB_TYPE_BUILD_RESULT_COMPARISON).exists()
    first = client.post(url).data
    assert client.post(url).data["job_id"] == first["job_id"]
    with patch(
        "quantem.segmentation.result_overlays.comparison_identity",
        side_effect=AssertionError("poll recomputed comparison identity"),
    ):
        pinned = client.get(url, {"revision": first["revision"]})
    assert pinned.data["status"] == "BUILDING"
    key, root = build(result)
    response = client.get(url)
    assert response.data["status"] == "READY"
    tile_url = response.data["ngff_url"] + "/labels/0/0.0"
    response = client.get(tile_url)
    assert response.status_code == 200
    assert b"".join(response.streaming_content) == (root / "labels" / "0" / "0.0").read_bytes()
    assert "immutable" in response["Cache-Control"]
    CompletedROI.objects.create(segmentation=segmentation, geometry=box(0, 0, 20, 20))
    assert client.get(url).data["status"] == "MISSING"
    # An old open comparison remains pinned to its own geometry.
    assert client.get(tile_url).status_code == 200
    assert client.get(tile_url.replace(key, "f" * 32)).status_code == 404
    assert client.get(tile_url.replace("labels/0/0.0", "../ready.json")).status_code == 404


def test_cancelled_or_changed_input_never_publishes_a_ready_comparison(segmentation):
    result = run(segmentation)
    key = comparison_identity(result)

    def cancel():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        build_comparison(str(result.pk), key, cancel_check=cancel, progress=lambda *_: None)
    assert not (comparison_directory(result, key) / "ready.json").exists()
    with patch(
        "quantem.segmentation.result_overlays.comparison_identity", side_effect=[key, "changed"]
    ):
        with pytest.raises(ValueError, match="Annotations changed"):
            build_comparison(
                str(result.pk), key, cancel_check=lambda: None, progress=lambda *_: None
            )
    assert not (comparison_directory(result, key) / "ready.json").exists()


def test_detail_retains_native_pixels_beyond_the_preview_limit(segmentation):
    from quantem.testing import create_small_test_image

    result = run(segmentation)
    image = create_small_test_image("Wide comparison", width=8193, height=32)
    segmentation.asset = image.asset
    segmentation.save(update_fields=["asset"])
    geometry = box(6001, 12, 6004, 15)
    SegmentObject.objects.create(
        segmentation=segmentation,
        geometry=geometry,
        centroid=geometry.centroid,
        bbox=geometry.envelope,
        label_state="CONFIRMED",
        source_model="manual",
    )
    key, root = build(result)
    mask = read_mask(root)
    assert mask.shape == (32, 8193)
    assert mask[13, 6002]
    assert not mask[13, 6000]
    assert not mask[13, 6005]
    # Sparse edge chunks retain the original dimensions even if the asset is
    # replaced while a comparison is open.
    sparse_url = (
        f"/api/segmentations/{segmentation.id}/results/{result.id}/overlay/{key}/labels/0/0.32"
    )
    assert not (root / "labels" / "0" / "0.32").exists()
    before = APIClient().get(sparse_url)
    assert before.status_code == 200
    segmentation.asset = create_small_test_image("Replacement", width=32, height=32).asset
    segmentation.save(update_fields=["asset"])
    after = APIClient().get(sparse_url)
    assert after.status_code == 200
    assert after.content == before.content


def test_detail_retry_replaces_an_incomplete_artifact(segmentation):
    result = run(segmentation)
    key, root = build(result)
    (root / ".zgroup").unlink()
    url = f"/api/segmentations/{segmentation.id}/results/{result.id}/overlay/"
    assert APIClient().get(url).data["status"] == "MISSING"
    rebuilt_key, root = build(result)
    assert rebuilt_key == key
    assert (root / ".zgroup").is_file()
    assert APIClient().get(url).data["status"] == "READY"


def test_comparison_identity_includes_bulk_label_edits(segmentation):
    from quantem.segmentation.mutations import register_state_mutation

    result = run(segmentation)
    key = comparison_identity(result)
    # This matches inexpensive answer paths that deliberately do not rewrite
    # every object's updated_at timestamp.
    SegmentObject.objects.filter(segmentation=segmentation).update(label_state="EXCLUDED")
    register_state_mutation(segmentation)
    assert comparison_identity(result) != key


def test_global_detail_applies_morphology_and_keeps_completed_mask(segmentation):
    from types import SimpleNamespace

    from skimage.morphology import closing, disk

    from quantem.segmentation.global_masks import save_global_mask
    from quantem.segmentation.result_overlays import comparison_geometries
    from quantem.segmentation.type_service import get_or_create_er_type

    result = run(segmentation)
    segmentation.segmentation_type = get_or_create_er_type()
    segmentation.save(update_fields=["segmentation_type"])
    old = np.zeros((256, 256), dtype=bool)
    old[5:10, 5:10] = True
    save_global_mask(segmentation, old, source="manual")
    CompletedROI.objects.create(segmentation=segmentation, geometry=box(0, 0, 30, 30))
    raw = np.zeros_like(old)
    raw[4:20, 4:20] = True
    raw[50:60, 50:60] = True
    raw[54, 54] = False
    result.segmentation = segmentation
    result.is_canonical = False
    result.source_model = "quantem:er"
    segmenter = SimpleNamespace(
        _foreground_mask=lambda _: raw.copy(), _organelle=SimpleNamespace(close_radius=1)
    )
    with (
        patch("quantem.seg_core.registry.get_segmenter_or_none", return_value=segmenter),
        patch(
            "quantem.seg_core.db.inference.replay_stored_probability_map",
            return_value=(SimpleNamespace(prob=raw), None),
        ),
    ):
        _, mask = comparison_geometries(result, cancel_check=lambda: None, progress=lambda *_: None)
    expected = closing(raw, disk(1))
    expected[:31, :31] = old[:31, :31]
    np.testing.assert_array_equal(mask, expected)

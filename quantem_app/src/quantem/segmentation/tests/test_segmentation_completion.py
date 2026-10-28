"""Manual coverage requires acknowledgement and never discards prediction history."""

from unittest.mock import patch

import numpy as np
import pytest
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.segmentation.completion import archive_and_discard, restore_last_archive
from quantem.segmentation.coverage import coverage_state
from quantem.segmentation.models import CompletedROI, ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image, write_prob_map_png

pytestmark = pytest.mark.django_db


@pytest.fixture
def segmentation():
    asset = create_small_test_image("Manual coverage", width=64, height=64).asset
    seg = ImageSegmentation.objects.create(
        asset=asset, segmentation_type=get_or_create_mitochondria_type()
    )
    shape = box(10, 10, 30, 30)
    SegmentObject.objects.create(
        segmentation=seg,
        geometry=shape,
        centroid=shape.centroid,
        bbox=shape.envelope,
        source_model="manual",
        label_state="CONFIRMED",
    )
    return seg


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"is_complete": True},
        {"acknowledge_full_manual": True},
        {"acknowledge_no_model": True},
        {"discard_unconfirmed": True, "acknowledged_discard_count": 1},
    ],
)
def test_incomplete_or_legacy_confirmation_never_discards(segmentation, body):
    ids = list(segmentation.segments.values_list("id", flat=True))
    response = APIClient().post(
        f"/api/segmentations/{segmentation.id}/complete", body, format="json"
    )
    assert response.status_code == 400
    assert list(segmentation.segments.values_list("id", flat=True)) == ids
    segmentation.refresh_from_db()
    assert not coverage_state(segmentation)["ready"]


def test_acknowledgement_establishes_manual_training_and_measurement_coverage(segmentation):
    stored = write_prob_map_png(segmentation, np.zeros((64, 64)), name="MITO_DINO")
    response = APIClient().post(
        f"/api/segmentations/{segmentation.id}/complete",
        {"acknowledge_full_manual": True, "acknowledge_no_model": True},
        format="json",
    )
    assert response.status_code == 200, response.data
    segmentation.refresh_from_db()
    assert coverage_state(segmentation)["ready"]
    assert segmentation.completed_rois.get().geometry.covers(box(0, 0, 64, 64))
    assert segmentation.probability_maps.filter(pk=stored.pk).exists()
    assert segmentation.segments.count() == 1


def test_model_objects_cannot_be_mislabelled_as_manual_only(segmentation):
    segmentation.segments.update(source_model="quantem:mito")
    response = APIClient().post(
        f"/api/segmentations/{segmentation.id}/complete",
        {"acknowledge_full_manual": True, "acknowledge_no_model": True},
        format="json",
    )
    assert response.status_code == 400
    assert segmentation.segments.count() == 1


def test_completed_roi_covering_the_image_is_ready_without_an_extra_step(segmentation):
    CompletedROI.objects.create(segmentation=segmentation, geometry=box(0, 0, 64, 64))
    assert coverage_state(segmentation)["ready"]


def test_objects_archived_by_an_older_installation_remain_recoverable(segmentation):
    row = segmentation.segments.get()
    row.label_state = "CANDIDATE"
    row.save()
    original = bytes(row.geometry_wkb)
    archive_and_discard(segmentation)
    assert not segmentation.segments.exists()
    response = APIClient().delete(f"/api/segmentations/{segmentation.id}/complete")
    assert response.status_code == 200
    restored = segmentation.segments.get(pk=row.pk)
    assert bytes(restored.geometry_wkb) == original
    assert restore_last_archive(segmentation)["restored_count"] == 0


def test_legacy_archive_failure_is_atomic(segmentation):
    segmentation.segments.update(label_state="CANDIDATE")
    with patch(
        "quantem.segmentation.completion.SegmentationCompletionArchive.objects.create",
        side_effect=RuntimeError("unavailable"),
    ):
        with pytest.raises(RuntimeError):
            archive_and_discard(segmentation)
    assert segmentation.segments.count() == 1

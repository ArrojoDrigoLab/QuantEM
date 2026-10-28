"""Coverage, including historical completion, does not lock editing."""

import pytest
from rest_framework.test import APIClient

from quantem.segmentation.completion import is_locked
from quantem.segmentation.models import ImageSegmentation, SegmentationConfig
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

pytestmark = pytest.mark.django_db


@pytest.fixture
def completed():
    asset = create_small_test_image("Editable completed field", width=256, height=256).asset
    seg = ImageSegmentation.objects.create(
        asset=asset, segmentation_type=get_or_create_mitochondria_type(), status_stage="COMPLETED"
    )
    SegmentationConfig.objects.create(segmentation=seg)
    return seg


@pytest.mark.parametrize(
    "suffix,body,expected",
    [
        ("segments/", {"geometry_coords": [[20, 20], [80, 20], [80, 80], [20, 80]]}, 201),
        ("completed-rois/", {"polygon_coords": [[20, 20], [80, 20], [80, 80], [20, 80]]}, 201),
        (
            "roi/",
            {"x": 20, "y": 20, "width": 64, "height": 64, "source": "MANUAL", "prepopulate": True},
            201,
        ),
        ("apply-full-image/", {"source_model": "quantem:mito"}, 202),
    ],
)
def test_completed_images_accept_edits_and_model_runs(completed, suffix, body, expected):
    response = APIClient().post(f"/api/segmentations/{completed.id}/{suffix}", body, format="json")
    assert response.status_code == expected, response.data
    assert not is_locked(completed)


def test_confirmed_image_objects_can_be_relabelled_without_unlocking(completed):
    client = APIClient()
    added = client.post(
        f"/api/segmentations/{completed.id}/segments/",
        {"geometry_coords": [[20, 20], [80, 20], [80, 80], [20, 80]]},
        format="json",
    )
    response = client.post(
        f"/api/segments/{added.data['id']}/label/", {"label_state": "EXCLUDED"}, format="json"
    )
    assert response.status_code == 200
    assert completed.segments.get().label_state == "EXCLUDED"

import pytest
from rest_framework.test import APIClient

from quantem.assets.asset_mutations import parse_pixel_size_nm
from quantem.testing import create_small_test_image


@pytest.mark.parametrize("value", [0, -1, "NaN", "Infinity", "-Infinity"])
@pytest.mark.django_db
def test_invalid_resolution_is_rejected_without_changing_image(value):
    image = create_small_test_image("Resolution validation", width=16, height=16)
    original = image.asset.pixel_size_nm
    response = APIClient().patch(
        f"/api/assets/{image.asset.id}/", {"pixel_size_nm": value}, format="json"
    )
    assert response.status_code == 400, response.data
    image.asset.refresh_from_db()
    assert image.asset.pixel_size_nm == original
    with pytest.raises(ValueError, match="finite number greater than zero"):
        parse_pixel_size_nm(value)


def test_unknown_and_positive_resolution_remain_supported():
    assert parse_pixel_size_nm(None) is None
    assert parse_pixel_size_nm("") is None
    assert parse_pixel_size_nm("2.75") == 2.75

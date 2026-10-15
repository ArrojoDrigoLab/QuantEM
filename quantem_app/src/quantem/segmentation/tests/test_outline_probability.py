import numpy as np
import pytest
from PIL import Image
from rest_framework.test import APIClient
from shapely.geometry import Polygon, box

from quantem.core.config import PROB_MAPS_DIR
from quantem.core.local_storage import storage_relpath_for_path
from quantem.segmentation.features.measure import measure_segments
from quantem.segmentation.features.probability import fill_missing_outline_probabilities
from quantem.segmentation.models import ImageSegmentation, ProbabilityMap, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


@pytest.fixture
def segmentation(db):
    image = create_small_test_image("Probability test", width=32, height=32)
    return ImageSegmentation.objects.create(
        asset=image.asset, segmentation_type=get_or_create_mitochondria_type()
    )


def save_map(tmp_path, segmentation, values, **metadata):
    directory = PROB_MAPS_DIR / str(segmentation.id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"map-{ProbabilityMap.objects.count()}.png"
    Image.fromarray(values).save(path)
    return ProbabilityMap.objects.create(
        segmentation=segmentation,
        name="quantem:mito",
        file_path=storage_relpath_for_path(path),
        metadata={"pack_id": "quantem:mito", "run_id": "run-a", **metadata},
    )


def manual(segmentation, polygon):
    return SegmentObject.objects.create(
        segmentation=segmentation,
        label_state="CONFIRMED",
        source_model="manual",
        geometry=polygon,
        bbox=polygon.envelope,
        centroid=polygon.centroid,
    )


def test_drawn_outline_averages_model_pixels_and_remains_manual(segmentation, tmp_path):
    data = np.zeros((32, 32), dtype=np.uint8)
    data[4:8, 4:6] = 255
    data[4:8, 6:8] = 51
    saved = save_map(tmp_path, segmentation, data)
    response = APIClient().post(
        f"/api/segmentations/{segmentation.id}/segments/",
        {"geometry_coords": [[4, 4], [8, 4], [8, 8], [4, 8]]},
        format="json",
    )
    assert response.status_code == 201, response.data
    segment = SegmentObject.objects.get(id=response.data["id"])
    assert segment.features["mean_prob"] == pytest.approx(0.6)
    assert segment.source_model == "manual"
    assert "run" not in segment.features
    assert segment.features["probability_source"]["map_id"] == str(saved.id)
    segment.geometry = box(6, 4, 8, 8)
    segment.bbox = segment.geometry.envelope
    measure_segments(segmentation, [segment], geometry_changed=True)
    assert segment.features["mean_prob"] == pytest.approx(0.2)


def test_analysis_backfill_respects_holes_and_roi_offsets(segmentation, tmp_path):
    data = np.full((8, 8), 102, dtype=np.uint8)
    data[2:6, 2:6] = 255
    save_map(tmp_path, segmentation, data, roi={"x": 8, "y": 8, "width": 8, "height": 8})
    segment = manual(
        segmentation, Polygon(box(8, 8, 16, 16).exterior, [box(10, 10, 14, 14).exterior])
    )
    partial = manual(segmentation, box(7, 8, 16, 16))
    fill_missing_outline_probabilities(segmentation)
    segment.refresh_from_db()
    partial.refresh_from_db()
    assert segment.features["mean_prob"] == pytest.approx(0.4)
    assert "mean_prob" not in partial.features


def test_missing_composite_and_different_run_maps_cannot_supply_a_score(segmentation, tmp_path):
    segment = manual(segmentation, box(4, 4, 8, 8))
    saved = save_map(tmp_path, segmentation, np.full((32, 32), 255, dtype=np.uint8), composite=True)
    fill_missing_outline_probabilities(segmentation)
    segment.refresh_from_db()
    assert "mean_prob" not in segment.features
    saved.metadata.pop("composite")
    saved.save()
    segment.source_model = "quantem:mito"
    segment.features = {"run": {"id": "another-run"}, "mean_prob": 0.9}
    measure_segments(segmentation, [segment], geometry_changed=True)
    assert "mean_prob" not in segment.features
    segment.source_model = "manual"
    saved.file_path = "missing.png"
    saved.save()
    measure_segments(segmentation, [segment], geometry_changed=True)
    assert "mean_prob" not in segment.features

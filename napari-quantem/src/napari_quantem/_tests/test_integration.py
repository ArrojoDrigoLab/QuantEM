"""End-to-end through the actual widgets, with real model weights.

Skipped unless ``QUANTEM_MODEL_DIR`` points at a directory holding the published artifacts. This is
the test that answers "can a user install this, open an image, and get a segmentation", and then
"can they correct it and adapt the model".
"""

from __future__ import annotations

import os

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("torch")

from napari_quantem import _models
from napari_quantem._widget_finetune import FineTuneWidget
from napari_quantem._widget_measure import MeasureWidget
from napari_quantem._widget_segment import SegmentWidget

pytestmark = pytest.mark.skipif(
    not os.environ.get("QUANTEM_MODEL_DIR"),
    reason="set QUANTEM_MODEL_DIR to a directory of published artifacts",
)


def _have_artifacts() -> bool:
    from quantem_em.registry import REGISTRY
    from quantem_em.weights import fetch

    return fetch.download_plan(list(REGISTRY.values()))["all_present"]


needs_weights = pytest.mark.skipif(not _have_artifacts(), reason="artifacts not all present")


@pytest.fixture(autouse=True)
def _auto_consent(monkeypatch):
    """The consent dialog is tested separately; here we accept so the run proceeds."""
    monkeypatch.setattr(_models, "confirm_download", lambda *a, **k: True)
    import napari_quantem._widget_finetune as ft
    import napari_quantem._widget_segment as seg

    monkeypatch.setattr(seg, "confirm_download", lambda *a, **k: True)
    monkeypatch.setattr(ft, "confirm_download", lambda *a, **k: True)


@pytest.fixture
def em_image():
    """A synthetic EM-like field: band-limited noise with brighter blobs."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(3)
    h, w = 640, 720
    base = ndi.gaussian_filter(rng.normal(0.5, 0.25, (h, w)), 3.0)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(14):
        cy, cx = rng.integers(60, h - 60), rng.integers(60, w - 60)
        ry, rx = rng.integers(20, 44), rng.integers(20, 44)
        base[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1] += 0.5
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    return (base * 255).astype(np.uint8)


@needs_weights
def test_segment_widget_produces_a_labels_layer(make_napari_viewer, qtbot, em_image):
    v = make_napari_viewer()
    v.add_image(em_image, name="em")
    w = SegmentWidget(v)
    w.organelle_buttons["mito"].setChecked(True)
    w.pixel.enabled.setCurrentIndex(1)
    w.pixel.value.setValue(8.0)

    n_before = len(v.layers)
    w._run()
    qtbot.waitUntil(
        lambda: len(v.layers) > n_before or "failed" in w.status.text(), timeout=600_000
    )
    assert "failed" not in w.status.text(), w.status.text()

    lab = v.layers[-1]
    assert lab.data.shape == em_image.shape
    assert lab.data.dtype == np.int32

    # Provenance must ride along with the result.
    c = lab.metadata["quantem"]
    assert c["model_id"] == "quantem/mito"
    assert c["arm_name"] == "F4v2_qem_cem"
    assert c["tile_size"] == 512 and c["stride"] == 384
    assert c["blend"] == "hann2d+1e-3" and c["pad_mode"] == "constant"
    assert c["threshold"] == 0.5
    assert c["min_area"] == 100  # owner ruling: 100 px for non-nucleus
    assert c["source_pixel_size_nm"] == (8.0, 8.0)
    assert "quantem_core_version" in c


@needs_weights
def test_probability_layer_is_optional_and_in_range(make_napari_viewer, qtbot, em_image):
    v = make_napari_viewer()
    v.add_image(em_image, name="em")
    w = SegmentWidget(v)
    w.prob_layer.setChecked(True)
    n_before = len(v.layers)
    w._run()
    qtbot.waitUntil(
        lambda: len(v.layers) >= n_before + 2 or "failed" in w.status.text(), timeout=600_000
    )
    prob = v.layers[-1]
    assert prob.data.dtype == np.float32
    assert 0.0 <= float(prob.data.min()) and float(prob.data.max()) <= 1.0


@needs_weights
def test_nucleus_min_area_is_500(make_napari_viewer, qtbot, em_image):
    v = make_napari_viewer()
    v.add_image(em_image, name="em")
    w = SegmentWidget(v)
    w.organelle_buttons["nucleus"].setChecked(True)
    n_before = len(v.layers)
    w._run()
    qtbot.waitUntil(
        lambda: len(v.layers) > n_before or "failed" in w.status.text(), timeout=600_000
    )
    assert v.layers[-1].metadata["quantem"]["min_area"] == 500


@needs_weights
def test_finetune_from_a_drawn_region(make_napari_viewer, qtbot, em_image):
    """The full corrective loop: segment, draw a reviewed region, adapt."""
    v = make_napari_viewer()
    v.add_image(em_image, name="em")

    from quantem_em.api import load_model

    model = load_model("quantem/mito")
    res = model.segment(em_image, pixel_size_nm=8.0)
    v.add_labels(res.labels, name="labels")

    sh = v.add_shapes(name="reviewed region")
    sh.add_rectangles(np.array([[[40, 40], [40, 400], [400, 400], [400, 40]]]))

    w = FineTuneWidget(v)
    w._refresh_layers()
    w.image_box.setCurrentText("em")
    w.labels_box.setCurrentText("labels")
    w.shapes_box.setCurrentText("reviewed region")
    w.organelle.setCurrentIndex(0)
    w.pixel.enabled.setCurrentIndex(1)
    w.pixel.value.setValue(8.0)
    w.steps.setValue(20)

    examples, problem = w._build_examples()
    assert problem is None and len(examples) == 1
    assert examples[0].valid.sum() < examples[0].valid.size, "region must be a subset"

    w._run()
    qtbot.waitUntil(lambda: w.save_btn.isEnabled() or "failed" in w.status.text(), timeout=900_000)
    assert "failed" not in w.status.text(), w.status.text()
    assert "trained" in w.status.text()
    assert "not held out" in w.status.text(), "the calibration Dice must be labelled honestly"
    assert w._model is not None


@needs_weights
def test_measure_after_segmentation(make_napari_viewer, em_image):
    from quantem_em.api import load_model

    v = make_napari_viewer()
    v.add_image(em_image, name="em")
    res = load_model("quantem/er").segment(em_image)
    v.add_labels(res.labels, name="labels")

    w = MeasureWidget(v)
    w._refresh()
    w.labels_box.setCurrentText("labels")
    w.image_box.setCurrentText("em")
    w.pixel.enabled.setCurrentIndex(1)
    w.pixel.value.setValue(4.0)
    w._measure()

    assert w._summary["n_objects"] == res.n_objects
    assert "area_nm2" in w._columns
    assert 0.0 <= w._summary["area_fraction"] <= 1.0

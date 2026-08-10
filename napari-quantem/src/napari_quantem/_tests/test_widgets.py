"""Widget construction and behaviour, against a real napari viewer.

These do not need model weights: they check that every widget builds, tracks layers, and refuses
to run when the inputs are wrong — which is where a plugin actually fails users.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

napari = pytest.importorskip("napari")

from napari_quantem._widget_batch import BatchWidget
from napari_quantem._widget_finetune import FineTuneWidget
from napari_quantem._widget_measure import MeasureWidget
from napari_quantem._widget_models import ModelManagerWidget
from napari_quantem._widget_proofread import ProofreadWidget
from napari_quantem._widget_segment import SegmentWidget

ALL_WIDGETS = [
    SegmentWidget,
    FineTuneWidget,
    ProofreadWidget,
    MeasureWidget,
    BatchWidget,
    ModelManagerWidget,
]


@pytest.fixture
def image(make_napari_viewer):
    v = make_napari_viewer()
    rng = np.random.default_rng(0)
    v.add_image(rng.integers(0, 255, (256, 300), dtype=np.uint8), name="em")
    return v


@pytest.mark.parametrize("cls", ALL_WIDGETS)
def test_widget_constructs(make_napari_viewer, cls):
    cls(make_napari_viewer())


def test_manifest_matches_widget_classes():
    """Every python_name in napari.yaml must actually import."""
    import importlib
    from pathlib import Path

    import yaml

    manifest = yaml.safe_load(
        (Path(__file__).parents[1] / "napari.yaml").read_text(encoding="utf-8")
    )
    for cmd in manifest["contributions"]["commands"]:
        mod, _, attr = cmd["python_name"].partition(":")
        assert hasattr(importlib.import_module(mod), attr), cmd["python_name"]


def test_segment_tracks_layers(image):
    w = SegmentWidget(image)
    assert w.layer_box.count() == 1
    image.add_image(np.zeros((64, 64), np.uint8), name="second")
    assert w.layer_box.count() == 2
    assert w.run_btn.isEnabled()


def test_segment_default_model_is_quantem_for_mito_and_omniem_elsewhere(image):
    from quantem_em.registry import DEFAULT_MODEL_FOR_ORGANELLE

    w = SegmentWidget(image)
    for organelle, expected in DEFAULT_MODEL_FOR_ORGANELLE.items():
        w.organelle_buttons[organelle].setChecked(True)
        assert w._specs[0].model_id == expected, organelle


def test_segment_shows_estimate_and_no_benchmark_numbers(image):
    w = SegmentWidget(image)
    assert "tile" in w.estimate.text()
    # Owner ruling: no benchmark Dice anywhere in the UI.
    for i in range(w.model_box.count()):
        assert "Dice" not in w.model_box.itemText(i)


def test_pixel_size_defaults_to_unspecified_and_says_so(image):
    """Empty by default -- never inferred -- and the consequence is stated, not hidden."""
    w = SegmentWidget(image)
    assert w.pixel.value_nm() is None
    text = w.pixel.note.text()
    assert "No pixel size given" in text
    assert "8 nm/px" in text, "the model's training resolution should be named"


def test_pixel_size_echoes_the_resample_in_words(image):
    w = SegmentWidget(image)
    w.pixel.enabled.setCurrentIndex(1)
    w.pixel.value.setValue(2.0)
    assert "2 → 8 nm/px" in w.pixel.note.text()
    assert "0.25× downsample" in w.pixel.note.text()


def test_large_upsample_is_flagged(image):
    w = SegmentWidget(image)
    w.pixel.enabled.setCurrentIndex(1)
    w.pixel.value.setValue(40.0)  # 40 -> 8 nm/px is a 5x upsample
    assert "degrade" in w.pixel.note.text()


def test_er_needs_no_pixel_size(image):
    w = SegmentWidget(image)
    w.organelle_buttons["er"].setChecked(True)
    assert "native resolution" in w.pixel.note.text()
    assert "no pixel size needed" in w.pixel.note.text().lower()


def test_finetune_refuses_without_a_reviewed_region(image):
    image.add_labels(np.zeros((256, 300), np.int32), name="labels")
    w = FineTuneWidget(image)
    w._refresh_layers()
    examples, problem = w._build_examples()
    assert not examples and problem
    assert "reviewed region" in problem or "whole image" in problem
    assert not w.run_btn.isEnabled()


def test_finetune_whole_image_scope_is_explicit(image):
    image.add_labels(np.zeros((256, 300), np.int32), name="labels")
    w = FineTuneWidget(image)
    w._refresh_layers()
    w.scope_image.setChecked(True)
    examples, problem = w._build_examples()
    assert problem is None and len(examples) == 1
    assert examples[0].valid.all()


def test_finetune_shapes_become_separate_regions(image):
    image.add_labels(np.zeros((256, 300), np.int32), name="labels")
    sh = image.add_shapes(name="reviewed region")
    sh.add_rectangles(np.array([[[10, 10], [10, 90], [90, 90], [90, 10]]]))
    sh.add_rectangles(np.array([[[120, 120], [120, 200], [200, 200], [200, 120]]]))
    w = FineTuneWidget(image)
    w._refresh_layers()
    examples, problem = w._build_examples()
    assert problem is None
    assert len(examples) == 2, "each shape should become its own annotated region"
    assert all(e.valid.any() and not e.valid.all() for e in examples)


def test_cv_button_needs_three_regions(image):
    image.add_labels(np.zeros((256, 300), np.int32), name="labels")
    w = FineTuneWidget(image)
    assert not w.cv_btn.isEnabled()


def test_proofread_operations(make_napari_viewer):
    from scipy import ndimage as ndi

    v = make_napari_viewer()
    lab = np.zeros((80, 80), np.int32)
    lab[10:20, 10:20] = 1  # 100 px
    lab[40:43, 40:43] = 2  # 9 px -- should be filtered
    v.add_labels(lab, name="labels")
    w = ProofreadWidget(v)
    w._refresh()

    w.min_area.setValue(50)
    w._remove_small()
    out = np.asarray(v.layers["labels"].data)
    assert out.max() == 1, "the 9 px object should be gone"
    assert (out > 0).sum() == 100

    # border removal
    lab2 = np.zeros((40, 40), np.int32)
    lab2[0:5, 0:5] = 1
    lab2[20:25, 20:25] = 2
    v.add_labels(lab2, name="edge")
    w._refresh()
    w.labels_box.setCurrentText("edge")
    w._remove_edge()
    assert np.asarray(v.layers["edge"].data).max() == 2
    assert not np.asarray(v.layers["edge"].data)[0:5, 0:5].any()
    assert ndi.label(np.asarray(v.layers["edge"].data) > 0)[1] == 1


def test_measure_writes_features_and_summary(make_napari_viewer):
    v = make_napari_viewer()
    lab = np.zeros((64, 64), np.int32)
    lab[8:24, 8:24] = 1
    lab[40:48, 40:48] = 2
    v.add_image(np.full((64, 64), 100, np.uint8), name="em")
    v.add_labels(lab, name="labels")
    w = MeasureWidget(v)
    w._refresh()
    w.labels_box.setCurrentText("labels")
    w.image_box.setCurrentText("em")
    w._measure()

    assert w._summary["n_objects"] == 2
    assert w._columns is not None and len(w._columns["label"]) == 2
    assert "area_px2" in w._columns, "no pixel size -> pixel units, named as such"
    assert "objects:" in w.summary.toPlainText()
    assert w.export_btn.isEnabled()


def test_measure_reports_physical_units_when_given_a_pixel_size(make_napari_viewer):
    v = make_napari_viewer()
    lab = np.zeros((32, 32), np.int32)
    lab[4:12, 4:12] = 1
    v.add_labels(lab, name="labels")
    w = MeasureWidget(v)
    w._refresh()
    w.pixel.enabled.setCurrentIndex(1)
    w.pixel.value.setValue(5.0)
    w._measure()
    assert "area_nm2" in w._columns
    assert w._summary["total_object_area_nm2"] == 64 * 25.0


def test_batch_finds_only_image_files(make_napari_viewer, tmp_path):
    import tifffile

    tifffile.imwrite(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
    tifffile.imwrite(tmp_path / "b.tif", np.zeros((8, 8), np.uint8))
    (tmp_path / "notes.txt").write_text("ignore me")
    w = BatchWidget(make_napari_viewer())
    w.in_dir.setText(str(tmp_path))
    assert len(w._files()) == 2
    assert "2 image file" in w.found.text()


def test_model_manager_lists_all_eight(make_napari_viewer):
    from quantem_em.registry import REGISTRY

    w = ModelManagerWidget(make_napari_viewer())
    assert w.table.rowCount() == len(REGISTRY) == 8
    ids = {w.table.item(r, 0).text() for r in range(w.table.rowCount())}
    assert ids == set(REGISTRY)


def test_download_consent_lists_a_licence_for_every_file():
    """The dialog tells the user the licence is 'listed above'. Make that literally true.

    Needs no display and no weights: the wording is built by a pure function precisely so it can
    be checked here.
    """
    from quantem_em.registry import REGISTRY
    from quantem_em.weights import fetch

    from napari_quantem import _models

    plan = {
        "missing": [
            {
                "filename": "x.safetensors",
                "bytes": 1000,
                "repo": "org/repo",
                "license": "CC BY 4.0",
            }
        ],
        "download_bytes": 1000,
    }
    headline, detail = _models.download_summary(plan)
    assert "1.0 KB" in headline
    assert "Licence: CC BY 4.0" in detail
    assert "huggingface.co/org/repo" in detail

    # And every real artifact carries one, so no user ever sees the fallback.
    for spec in REGISTRY.values():
        for name in fetch.artifacts_for(spec):
            assert fetch.artifact_info(name)["license"], f"{name} has no licence in the registry"


def test_drawing_a_region_updates_the_finetune_widget(image):
    """Drawing is this widget's central interaction, and it must not need a second click to land.

    ``viewer.layers.events.inserted`` fires when a layer appears, NOT when its data changes, so a
    widget that only listens to the former leaves the region count and the leave-one-region-out
    button frozen while the user draws.
    """
    import numpy as np

    from napari_quantem._widget_finetune import MIN_CV_REGIONS

    v = image
    v.add_labels(np.zeros((256, 300), np.int32), name="corrections")
    shapes = v.add_shapes(name="reviewed region")
    w = FineTuneWidget(v)
    w.scope_shapes.setChecked(True)
    w.shapes_box.setCurrentIndex(w.shapes_box.findText("reviewed region"))
    w.labels_box.setCurrentIndex(w.labels_box.findText("corrections"))

    # The observable must be something only an event can update -- calling _build_examples()
    # ourselves would re-read the shapes and pass even with the subscription removed.
    assert "Draw at least one reviewed region" in w.estimate.text()

    for i in range(MIN_CV_REGIONS):
        y = 10 + i * 60
        shapes.add_rectangles([np.array([[y, 10], [y, 120], [y + 40, 120], [y + 40, 10]])])

    assert "Draw at least one reviewed region" not in w.estimate.text(), (
        "the estimate never noticed the regions being drawn"
    )
    examples, problem = w._build_examples()
    assert len(examples) == MIN_CV_REGIONS and problem is None


def test_painting_updates_the_proofread_object_count(image):
    """The count is the only feedback while painting; frozen is worse than absent."""
    import numpy as np

    lab = np.zeros((256, 300), np.int32)
    lab[10:20, 10:20] = 1
    layer = image.add_labels(lab, name="labels")
    w = ProofreadWidget(image)
    w.labels_box.setCurrentIndex(w.labels_box.findText("labels"))
    assert "1 object" in w.info.text(), w.info.text()

    grown = layer.data.copy()
    grown[100:140, 100:140] = 2
    grown[200:220, 200:220] = 3
    layer.data = grown
    assert "3 object" in w.info.text(), f"count went stale: {w.info.text()}"


def test_regions_are_rasterised_once_not_once_per_shape(image, monkeypatch):
    """to_labels paints every shape on each call, so calling it per shape is quadratic -- and it
    now runs on every stroke."""
    import numpy as np

    v = image
    v.add_labels(np.zeros((256, 300), np.int32), name="corrections")
    shapes = v.add_shapes(name="reviewed region")
    for i in range(4):
        y = 10 + i * 50
        shapes.add_rectangles([np.array([[y, 10], [y, 120], [y + 30, 120], [y + 30, 10]])])

    w = FineTuneWidget(v)
    w.scope_shapes.setChecked(True)
    w.shapes_box.setCurrentIndex(w.shapes_box.findText("reviewed region"))
    w.labels_box.setCurrentIndex(w.labels_box.findText("corrections"))

    calls = {"n": 0}
    real = shapes.to_labels

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(shapes, "to_labels", counting)
    examples, _ = w._build_examples()
    assert len(examples) == 4
    assert calls["n"] == 1, f"rasterised {calls['n']}x for 4 regions; should be once"


def _painted(image):
    """Two objects that TOUCH, as they do after a watershed split."""
    import numpy as np

    lab = np.zeros((64, 64), np.int32)
    lab[10:30, 10:30] = 1
    lab[10:30, 30:50] = 2  # shares a border with label 1
    lab[55:58, 55:58] = 3  # tiny, and isolated
    return image.add_labels(lab, name="labels")


def test_renumber_does_not_re_segment(image):
    """Renumbering must compact ids, not re-derive components.

    label_objects(lab > 0) merges every pair of touching labels, which silently undoes each
    watershed split the user just made -- the exact work this widget exists to support.
    """
    layer = _painted(image)
    w = ProofreadWidget(image)
    w.labels_box.setCurrentIndex(w.labels_box.findText("labels"))
    w._compact()
    ids = set(int(i) for i in __import__("numpy").unique(layer.data)) - {0}
    assert ids == {1, 2, 3}, f"touching labels were fused by renumbering: {ids}"


def test_size_filter_measures_individual_labels(image):
    """Two touching 400 px objects must not survive a 500 px threshold as one 800 px blob."""
    import numpy as np

    layer = _painted(image)
    w = ProofreadWidget(image)
    w.labels_box.setCurrentIndex(w.labels_box.findText("labels"))
    w.min_area.setValue(50)
    w._remove_small()
    ids = set(int(i) for i in np.unique(layer.data)) - {0}
    assert 3 not in ids, "the 9 px object should have been removed"
    assert {1, 2} <= ids, f"the two 400 px objects should both survive, got {ids}"


def test_merge_reaches_across_a_one_pixel_seam(image):
    """Connected components are 4-connected, so no two labels are ever 4-adjacent: a single
    cross-shaped dilation can only ever find the object itself, and Merge was a silent no-op."""
    import numpy as np

    lab = np.zeros((64, 64), np.int32)
    lab[10:30, 10:29] = 1
    lab[10:30, 30:50] = 2  # one-pixel background seam at column 29
    layer = image.add_labels(lab, name="labels")
    w = ProofreadWidget(image)
    w.labels_box.setCurrentIndex(w.labels_box.findText("labels"))
    layer.selected_label = 1
    w._merge()
    ids = set(int(i) for i in np.unique(layer.data)) - {0}
    assert ids == {1}, f"merge did not bridge the seam: {ids}"
    assert "merged 1 label" in w.status.text(), w.status.text()


def test_merge_says_so_when_there_is_nothing_to_merge(image):
    """A no-op that reports success is worse than a no-op."""
    import numpy as np

    lab = np.zeros((64, 64), np.int32)
    lab[10:20, 10:20] = 1
    lab[50:60, 50:60] = 2  # far away
    layer = image.add_labels(lab, name="labels")
    w = ProofreadWidget(image)
    w.labels_box.setCurrentIndex(w.labels_box.findText("labels"))
    layer.selected_label = 1
    w._merge()
    assert set(int(i) for i in np.unique(layer.data)) - {0} == {1, 2}
    assert "nothing adjacent" in w.status.text(), w.status.text()


def test_batch_output_names_survive_subdirectories(make_napari_viewer, tmp_path):
    """A recursive pattern must not collapse s1/img.tif and s2/img.tif onto one output.

    With "skip images that already have an output" on -- the default -- the collision does not
    overwrite, it silently drops the second image and logs it as a clean skip.
    """
    import tifffile

    src = tmp_path / "in"
    for sub in ("s1", "s2"):
        (src / sub).mkdir(parents=True)
        tifffile.imwrite(src / sub / "img.tif", np.zeros((8, 8), np.uint8))

    w = BatchWidget(make_napari_viewer())
    w.in_dir.setText(str(src))
    w.pattern.setText("**/*.tif")
    files = w._files()
    assert len(files) == 2, f"expected both images, got {files}"

    root = Path(w.in_dir.text())
    stems = {"__".join(f.relative_to(root).with_suffix("").parts) for f in files}
    assert stems == {"s1__img", "s2__img"}, f"outputs would collide: {stems}"


def test_z_range_is_reachable_and_limits_the_run(make_napari_viewer):
    """The z_from/z_to spin boxes were built and never added to a layout: a planned control that
    no user could reach."""
    v = make_napari_viewer()
    v.add_image(np.zeros((10, 32, 32), np.uint8), name="stack")
    w = SegmentWidget(v)

    assert w.slices.count() == 3, "expected current slice / all slices / z range"
    w.slices.setCurrentIndex(2)
    assert w.z_row.isVisibleTo(w), "the z-range row must be reachable when z range is chosen"
    assert w.z_to.maximum() == 9, "range must be clamped to the stack depth"

    w.z_from.setValue(2)
    w.z_to.setValue(5)
    assert "4 slices" in w.estimate.text(), w.estimate.text()


def test_z_range_row_is_hidden_for_a_2d_image(make_napari_viewer):
    v = make_napari_viewer()
    v.add_image(np.zeros((32, 32), np.uint8), name="flat")
    w = SegmentWidget(v)
    assert not w.z_row.isVisibleTo(w)
    assert not w.slice_row.isVisibleTo(w)

import gc
from pathlib import Path

import numpy as np
import pytest

from quantem.inference import engine, resample
from quantem.inference.mapped_prediction import predict_native
from quantem.inference.specs import MODEL_SPECS


@pytest.mark.parametrize("pixel_size_nm", [8.0, 4.0, 3.3, 12.0, 13.1])
def test_streaming_native_bytes_match_resident_pipeline(pixel_size_nm):
    image = np.random.default_rng(3).integers(0, 256, (533, 617), dtype=np.uint8)
    model = engine.LoadedModel(
        spec=MODEL_SPECS["quantem:mito"],
        device="cpu",
        module=None,
        forward=lambda tile: tile.astype(np.float32) / 255.0,
    )
    expected = engine.predict_region(model, image, pixel_size_nm=pixel_size_nm)
    native = predict_native(model, image, pixel_size_nm=pixel_size_nm, overlap=0.25)
    np.testing.assert_array_equal(
        native.data[:], resample.probability_to_native_uint8(expected.prob, expected.context)
    )
    assert native.provenance(0.5) == resample.NativeProbabilityMap.from_model_grid(
        expected.prob, expected.context
    ).provenance(0.5)
    path = Path(native.data.store.root).parent
    assert not list(path.glob("*.f32"))
    del native
    gc.collect()
    assert not path.exists()


def test_streaming_never_coerces_the_source_to_a_whole_array():
    class WindowOnly:
        shape = (601, 701)

        def __array__(self, *args, **kwargs):
            raise AssertionError("whole source materialized")

        def __getitem__(self, slices):
            ys, xs = slices
            h = len(range(*ys.indices(self.shape[0])))
            w = len(range(*xs.indices(self.shape[1])))
            return np.full((h, w), 123, dtype=np.uint8)

    model = engine.LoadedModel(
        spec=MODEL_SPECS["quantem:mito"],
        device="cpu",
        module=None,
        forward=lambda tile: tile.astype(np.float32) / 255,
    )
    native = predict_native(model, WindowOnly(), pixel_size_nm=8, overlap=0.25)
    np.testing.assert_array_equal(native.data[:], np.full((601, 701), 123, dtype=np.uint8))


def test_insufficient_scratch_space_fails_before_prediction_and_cleans_up(monkeypatch):
    from types import SimpleNamespace

    from quantem.core.config import TMP_DIR
    from quantem.inference import mapped_prediction

    model = engine.LoadedModel(
        spec=MODEL_SPECS["quantem:mito"],
        device="cpu",
        module=None,
        forward=lambda tile: pytest.fail("inference started"),
    )
    root = TMP_DIR / "inference"
    before = set(root.iterdir()) if root.exists() else set()
    monkeypatch.setattr(
        mapped_prediction.shutil, "disk_usage", lambda path: SimpleNamespace(free=0)
    )
    with pytest.raises(ValueError, match="temporary disk space"):
        predict_native(model, np.zeros((600, 700), dtype=np.uint8), pixel_size_nm=8, overlap=0.25)
    assert set(root.iterdir()) == before


def test_rescaled_model_grid_selects_streaming_even_for_small_source(monkeypatch):
    from quantem.inference import segmenter as module

    segmenter = module.DinoMitoSegmenter(pixel_size_nm=32)
    segmenter._model = engine.LoadedModel(
        spec=MODEL_SPECS["quantem:mito"],
        device="cpu",
        module=None,
        forward=lambda tile: tile.astype(np.float32) / 255,
    )
    monkeypatch.setattr(module, "STREAMING_MIN_PIXELS", 200 * 200)
    monkeypatch.setattr(engine, "predict_region", lambda *a, **k: pytest.fail("resident grid"))
    result = segmenter.run_dl_inference(np.zeros((80, 80), np.uint8), {})
    assert not isinstance(result["DINO"], np.ndarray)
    assert result["DINO"].shape == (80, 80)


def test_dead_prediction_workspace_is_reclaimed_but_live_readers_survive(tmp_path, monkeypatch):
    import json
    import os
    import time

    from quantem.inference import out_of_core

    for name, owner in (("prediction-dead", 987654321), ("prediction-live", os.getpid())):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "owner.json").write_text(json.dumps({"pid": owner}))
        old = time.time() - 120
        os.utime(folder, (old, old))
    monkeypatch.setattr(out_of_core, "_process_is_alive", lambda pid: pid == os.getpid())
    out_of_core._cleanup_stale_workspaces(tmp_path, prefix="prediction-", min_age=60)
    assert not (tmp_path / "prediction-dead").exists()
    assert (tmp_path / "prediction-live").is_dir()


def test_corrupt_workspace_owner_does_not_block_future_predictions(tmp_path):
    import os
    import time

    from quantem.inference import out_of_core

    folder = tmp_path / "prediction-corrupt"
    folder.mkdir()
    (folder / "owner.json").write_text("[]")
    old = time.time() - 120
    os.utime(folder, (old, old))
    out_of_core._cleanup_stale_workspaces(tmp_path, prefix="prediction-", min_age=60)
    assert not folder.exists()

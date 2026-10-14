from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

from quantem.inference import out_of_core
from quantem.inference.out_of_core import postprocess_probability_source
from quantem.inference.postprocess import postprocess_mask
from quantem.inference.resample import NativeProbabilityMap, binarize_quantized
from quantem.inference.segmenter import DinoMitoSegmenter


def _compact_survivors(labels: np.ndarray) -> np.ndarray:
    """Compact nonzero reference ids without changing their order."""
    out = np.zeros(labels.shape, dtype=np.uint32)
    surviving = np.unique(labels)
    surviving = surviving[surviving != 0]
    for compact, original in enumerate(surviving, start=1):
        out[labels == original] = compact
    return out


@pytest.mark.parametrize(
    ("close_radius", "min_area"),
    [(0, 0), (1, 2), (2, 7), (3, 25)],
)
def test_out_of_core_matches_reference_across_many_stripe_seams(
    tmp_path,
    close_radius: int,
    min_area: int,
) -> None:
    rng = np.random.default_rng(8127 + close_radius * 100 + min_area)
    stored = rng.integers(0, 256, size=(53, 47), dtype=np.uint8)
    # Deliberate cavities, diagonal seam connections, and border components.
    stored[12:35, 8:32] = 240
    stored[18:29, 15:25] = 10
    stored[38, 27] = 240
    stored[39, 28] = 240
    stored[0:8, 40:47] = 240

    threshold = 0.55
    reference = postprocess_mask(
        binarize_quantized(stored, threshold),
        close_radius=close_radius,
        min_area=min_area,
    )
    # 47 bytes per row and a 144-byte target force three-row stripes. Most
    # components therefore cross several artificial boundaries.
    with postprocess_probability_source(
        stored,
        threshold=threshold,
        close_radius=close_radius,
        min_area=min_area,
        workspace_root=tmp_path,
        stripe_target_bytes=144,
    ) as actual:
        np.testing.assert_array_equal(np.asarray(actual), _compact_survivors(reference))


def test_hole_fill_is_global_not_per_stripe(tmp_path) -> None:
    stored = np.zeros((41, 37), dtype=np.uint8)
    stored[3:38, 4:33] = 255
    stored[8:33, 9:28] = 0
    # Keep the channel open after closing so hole filling must see the border.
    stored[0:20, 16:21] = 0

    reference = postprocess_mask(
        binarize_quantized(stored, 0.5),
        close_radius=1,
        min_area=0,
    )
    assert reference[15, 18] == 0
    with postprocess_probability_source(
        stored,
        threshold=0.5,
        close_radius=1,
        min_area=0,
        workspace_root=tmp_path,
        stripe_target_bytes=80,
    ) as actual:
        np.testing.assert_array_equal(np.asarray(actual), _compact_survivors(reference))


def test_workspace_is_removed_after_completion(tmp_path) -> None:
    stored = np.full((19, 23), 255, dtype=np.uint8)
    with postprocess_probability_source(
        stored,
        threshold=0.5,
        close_radius=1,
        min_area=0,
        workspace_root=tmp_path,
        stripe_target_bytes=46,
    ) as labels:
        assert np.any(labels)
        assert list(tmp_path.iterdir())
    assert list(tmp_path.iterdir()) == []


def test_workspace_is_removed_when_source_read_fails(tmp_path) -> None:
    class UnreadableSource:
        shape = (19, 23)
        dtype = np.dtype(np.uint8)

        def __getitem__(self, key):
            raise OSError("read failed")

    with pytest.raises(OSError, match="read failed"):
        with postprocess_probability_source(
            UnreadableSource(),
            threshold=0.5,
            close_radius=1,
            min_area=0,
            workspace_root=tmp_path,
        ):
            pytest.fail("unreadable source was accepted")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_stripe_offsets_do_not_overflow_scipys_signed_labels(tmp_path, monkeypatch, dtype):
    mask = np.ones((2, 2), dtype=bool)
    labels = np.memmap(tmp_path / "labels", mode="w+", dtype=dtype, shape=mask.shape)
    results = iter(
        [
            (np.array([[1, 0]], dtype=np.int32), 2**31),
            (np.array([[0, 1]], dtype=np.int32), 1),
        ]
    )
    monkeypatch.setattr(out_of_core.ndi, "label", lambda *_a, **_k: next(results))
    try:
        count = out_of_core._label_stripes(
            mask,
            labels,
            foreground=True,
            connectivity=2,
            stripe_rows=1,
        )
        assert count == 2**31 + 1
        assert labels[1, 1] == 2**31 + 1
    finally:
        out_of_core._close_memmap(labels)


def test_process_liveness_does_not_terminate_the_owner():
    assert out_of_core._process_is_alive(os.getpid())


def test_component_overflow_retries_with_a_wider_store(tmp_path, monkeypatch):
    stored = np.zeros((53, 53), dtype=np.uint8)
    stored[::3, ::3] = 255
    monkeypatch.setattr(out_of_core, "_LABEL_DTYPES", (np.dtype(np.uint8), np.dtype(np.uint16)))
    reference = postprocess_mask(stored > 0, close_radius=0, min_area=0)
    with postprocess_probability_source(
        stored,
        threshold=0.5,
        close_radius=0,
        min_area=0,
        workspace_root=tmp_path,
        stripe_target_bytes=159,
    ) as actual:
        assert actual.dtype == np.uint16
        np.testing.assert_array_equal(actual, reference)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("shape", [(1, 9), (9, 1), (7, 11)])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_uniform_maps_and_threshold_endpoints(tmp_path, shape, threshold):
    for value in (0, 255):
        stored = np.full(shape, value, dtype=np.uint8)
        reference = postprocess_mask(
            binarize_quantized(stored, threshold),
            close_radius=2,
            min_area=2,
        )
        with postprocess_probability_source(
            stored,
            threshold=threshold,
            close_radius=2,
            min_area=2,
            workspace_root=tmp_path,
            stripe_target_bytes=shape[1],
        ) as actual:
            np.testing.assert_array_equal(actual, _compact_survivors(reference))


def test_published_geometry_and_probability_are_identical(
    tmp_path,
    monkeypatch,
) -> None:
    import quantem.core.config as config
    import quantem.inference.out_of_core as out_of_core

    stored = np.zeros((67, 71), dtype=np.uint8)
    stored[4:17, 5:21] = 210
    stored[27:52, 11:39] = 190
    stored[33:43, 18:31] = 20  # a hole filled by the mito pipeline
    stored[49:65, 52:69] = 230
    stored[18, 42] = 220  # undersized component

    segmenter = DinoMitoSegmenter(min_area=5)
    segmenter.set_fg_threshold(0.55)
    segmenter.adopt_native_probability_map(NativeProbabilityMap.from_stored(stored))

    monkeypatch.setattr(out_of_core, "should_use_out_of_core", lambda _array: False)
    reference = segmenter.extract_instances_geometry(stored, min_area=5)
    image = np.arange(stored.size, dtype=np.uint16).reshape(stored.shape)
    reference_measured = segmenter.extract_instances(
        stored,
        image,
        {"DINO": stored},
        min_area=5,
    )

    chunked = zarr.open_array(
        str(tmp_path / "probability.zarr"),
        mode="w",
        shape=stored.shape,
        chunks=(16, 17),
        dtype=np.uint8,
        zarr_format=2,
    )
    chunked[:] = stored
    segmenter.adopt_native_probability_map(NativeProbabilityMap.from_stored(chunked))
    monkeypatch.setattr(config, "TMP_DIR", tmp_path)
    monkeypatch.setattr(out_of_core, "should_use_out_of_core", lambda _array: True)
    actual = segmenter.extract_instances_geometry(chunked, min_area=5)
    actual_measured = segmenter.extract_instances(
        chunked,
        image,
        {"DINO": chunked},
        min_area=5,
    )

    assert actual == reference
    assert actual_measured == reference_measured

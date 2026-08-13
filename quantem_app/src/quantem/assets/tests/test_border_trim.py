import numpy as np

from quantem.assets.border_trim import (
    should_trim_initial_import,
    trim_black_or_white_border,
)


def test_trims_black_and_white_edges_before_canonical_storage():
    plane = np.full((8, 9), 255, dtype=np.uint8)
    plane[2:6, 3:7] = 123

    cropped, trim = trim_black_or_white_border(plane)

    assert cropped.shape == (4, 4)
    assert trim is not None
    assert trim.as_metadata() == {
        "left": 3,
        "top": 2,
        "right": 2,
        "bottom": 2,
        "original_width": 9,
        "original_height": 8,
    }


def test_leaves_a_completely_blank_image_intact():
    plane = np.zeros((5, 6), dtype=np.uint8)

    cropped, trim = trim_black_or_white_border(plane)

    assert cropped.shape == plane.shape
    assert trim is None


def test_trims_alternating_black_and_white_border_layers():
    plane = np.full((9, 9), 255, dtype=np.uint8)
    plane[1:-1, 1:-1] = 0
    plane[2:-2, 2:-2] = 127

    cropped, trim = trim_black_or_white_border(plane)

    assert cropped.shape == (5, 5)
    assert np.all(cropped == 127)
    assert trim is not None
    assert (trim.left, trim.top, trim.right, trim.bottom) == (2, 2, 2, 2)


def test_leaves_nonuniform_edges_intact():
    plane = np.arange(25, dtype=np.uint8).reshape(5, 5)

    cropped, trim = trim_black_or_white_border(plane)

    assert np.array_equal(cropped, plane)
    assert trim is None


def test_only_a_staged_initial_import_is_allowed_to_trim():
    assert should_trim_initial_import(
        source_is_canonical_png=False,
        rendition_metadata={"upload_state": "staged"},
    )
    assert not should_trim_initial_import(
        source_is_canonical_png=False,
        rendition_metadata={"upload_state": "canonical"},
    )
    assert not should_trim_initial_import(
        source_is_canonical_png=True,
        rendition_metadata={"upload_state": "staged"},
    )
    assert not should_trim_initial_import(
        source_is_canonical_png=False,
        rendition_metadata={},
    )

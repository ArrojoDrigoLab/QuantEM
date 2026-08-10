"""The min-area rule is ours, not scikit-image's.

``skimage.morphology.remove_small_objects`` changed which side of ``min_size``
it keeps. Historically it removed ``size < min_size``; on 0.26 ``min_size`` is
deprecated and an object of *exactly* that many pixels is discarded instead.
QuantEM's floor is ``scikit-image>=0.22``, which spans the change, so the same
image and the same QuantEM would report different object counts on two machines
and nothing would say why -- the library version is the only thing that moved.

No published number is at stake: the manuscript pipeline
(``quantem.inference._fig3``) has no min-area step at all. This is an
application-layer filter, so the rule is QuantEM's to state, and these tests
state it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from skimage.morphology import remove_small_objects

from quantem.inference.postprocess import filter_min_area, postprocess_probability


def _labels_with_sizes(*sizes: int) -> np.ndarray:
    """One row per object, so each label has exactly the requested pixel count."""
    labels = np.zeros((len(sizes), max(sizes) + 2), dtype=np.int32)
    for row, size in enumerate(sizes):
        labels[row, :size] = row + 1
    return labels


def test_exactly_min_area_survives():
    """The pixel the FutureWarning is about."""
    labels = _labels_with_sizes(5)
    assert filter_min_area(labels, 5).max() == 1


def test_one_below_min_area_is_removed():
    labels = _labels_with_sizes(4)
    assert filter_min_area(labels, 5).max() == 0


def test_survivors_keep_their_label_ids():
    """Downstream reads regionprops on these ids; renumbering would silently
    re-associate every per-object measurement."""
    labels = _labels_with_sizes(2, 40, 3, 50)
    out = filter_min_area(labels, 10)
    assert sorted(np.unique(out)) == [0, 2, 4]


@pytest.mark.parametrize("min_area", [0, 1])
def test_trivial_thresholds_are_a_no_op(min_area):
    labels = _labels_with_sizes(1, 2, 3)
    assert np.array_equal(filter_min_area(labels, min_area), labels)


def test_empty_image():
    labels = np.zeros((8, 8), dtype=np.int32)
    assert filter_min_area(labels, 10).sum() == 0


def test_differs_from_skimage_026_only_at_the_boundary():
    """Pin the size of the change: exactly-``min_area`` objects, and nothing else.

    This is the evidence that adopting our own rule did not quietly alter
    anything beyond the one comparison it was meant to fix.
    """
    rng = np.random.default_rng(20260806)
    disagreements = 0
    for _ in range(50):
        labels = rng.integers(0, 6, size=(40, 40)).astype(np.int32)
        for min_area in (2, 5, 20, 100, 300):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                skimage_out = remove_small_objects(labels.copy(), min_size=min_area)
            ours = filter_min_area(labels.copy(), min_area)
            if np.array_equal(ours, skimage_out):
                continue
            differing = np.unique(labels[ours != skimage_out])
            for label_id in differing:
                # Every disagreement is an object of exactly min_area pixels,
                # which we keep and 0.26 drops.
                assert int((labels == label_id).sum()) == min_area
                disagreements += 1
    assert disagreements > 0, "the boundary case never came up; the test proves nothing"


def test_no_deprecation_warning_reaches_a_run():
    """A user's log should not carry a library warning about their object counts."""
    prob = np.zeros((16, 16), dtype=np.float32)
    prob[2:6, 2:6] = 0.9  # 16 px
    prob[10, 10] = 0.9  # 1 px
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        labels = postprocess_probability(prob, threshold=0.5, min_area=4)
    assert labels.max() == 1

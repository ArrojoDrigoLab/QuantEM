"""The extractor's spelling and the analysis's spelling must be one vocabulary.

They were two. ``quantem.seg_core.extraction`` wrote ``mean_intensity``;
``quantem.analysis.morphometrics`` read ``intensity_mean``. It never wrote the
ellipse axis lengths or the Feret diameter at all, though it computed the axes
to get ``elongation`` and threw them away. The result was ten of twenty-seven
columns of ``objects.csv`` blank for every model-produced object, filled only by
the handful a person had drawn by hand -- so a Feret diameter quoted "over 90
mitochondria" was four brush strokes.

Nothing in the code said so. These tests do.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import label, regionprops

from quantem.analysis.morphometrics import (
    CALIBRATED_METRIC_KEYS,
    PIXEL_METRIC_KEYS,
    STORED_FEATURE_FOR_METRIC,
    derive,
)
from quantem.seg_core.extraction import (
    INTENSITY_FEATURE_KEYS,
    PROBABILITY_FEATURE_KEYS,
    SEGMENT_FEATURE_KEYS,
    SHAPE_FEATURE_KEYS,
    build_segment_from_region,
)

GENERATED_FLAG = "mito_generated"


def _scene(*, with_intensity: bool = True, prob_value: float = 0.8):
    """One ellipse in a 96x96 frame, with a probability map and an image."""
    yy, xx = np.mgrid[0:96, 0:96]
    mask = ((yy - 48) / 20.0) ** 2 + ((xx - 48) / 12.0) ** 2 <= 1.0
    labels = label(mask)
    prob = np.full(mask.shape, 0.05, dtype=np.float32)
    prob[mask] = prob_value
    image = None
    if with_intensity:
        image = (40 + 60 * np.sin(xx / 5.0) + 100 * mask).astype(np.uint16)
    regions = regionprops(labels, intensity_image=image)
    return regions[0], labels, prob, image


def _build(**kwargs):
    region, labels, prob, image = _scene(**kwargs)
    segment = build_segment_from_region(
        region, labels, {"DINO": prob}, prob, GENERATED_FLAG, 0.0, 0.0, image
    )
    assert segment is not None
    return segment


class TestOneVocabulary:
    def test_the_extractor_writes_every_key_the_analysis_reads(self):
        features = _build().features
        missing = [key for key in SEGMENT_FEATURE_KEYS if key not in features]
        assert not missing, (
            f"{missing} are read by quantem.analysis.morphometrics and are not "
            "written by quantem.seg_core.extraction: those columns of "
            "objects.csv would be blank for every model-produced object."
        )

    def test_the_extractor_writes_nothing_the_analysis_cannot_read(self):
        """A key written under a name nobody reads is a lost measurement."""
        features = _build().features
        allowed = {*SEGMENT_FEATURE_KEYS, GENERATED_FLAG}
        unread = {
            key
            for key in features
            # Per-model probability means are named after the caller's models
            # and are deliberately outside the shared vocabulary.
            if key not in allowed and not key.startswith("mean_prob_")
        }
        assert not unread, (
            f"{sorted(unread)} are stored on SegmentObject.features and read by "
            "nothing. Either add them to the analysis vocabulary or stop "
            "computing them."
        )

    def test_the_hand_drawn_path_uses_the_same_names(self):
        """The third writer. It is measured by a different module and must not
        drift either -- that is how ``intensity_mean`` and ``mean_intensity``
        came to be two spellings of one number."""
        from quantem.segmentation.features.measure import MEASUREMENT_KEYS

        strays = set(MEASUREMENT_KEYS) - set(SEGMENT_FEATURE_KEYS)
        assert not strays, (
            f"{sorted(strays)} are written for hand-drawn objects under names "
            "the analysis does not read."
        )
        # mean_prob is the one key a drawn polygon legitimately lacks.
        assert set(SEGMENT_FEATURE_KEYS) - set(MEASUREMENT_KEYS) == set(
            PROBABILITY_FEATURE_KEYS
        )

    def test_every_stored_key_maps_to_exactly_one_column(self):
        assert set(STORED_FEATURE_FOR_METRIC.values()) == set(SEGMENT_FEATURE_KEYS)
        assert len(set(STORED_FEATURE_FOR_METRIC.values())) == len(
            STORED_FEATURE_FOR_METRIC
        )

    def test_a_real_extracted_object_fills_every_column(self):
        """End of the chain: extractor -> features -> objects.csv row."""
        row = derive(
            _build().features, object_id="o1", pixel_size_nm=8.0
        ).as_row()
        blank = [
            key
            for key in (*PIXEL_METRIC_KEYS, *CALIBRATED_METRIC_KEYS)
            if row.get(key) is None
        ]
        assert not blank, f"{blank} are blank for a model-produced object"

    @pytest.mark.parametrize("stored", SEGMENT_FEATURE_KEYS)
    def test_each_stored_key_is_what_fills_its_column(self, stored):
        """Rename one key and exactly one column empties -- which is the failure
        the ``mean_intensity``/``intensity_mean`` mismatch produced ten times
        over, silently."""
        features = dict(_build().features)
        column = next(
            metric for metric, key in STORED_FEATURE_FOR_METRIC.items() if key == stored
        )
        assert derive(features, object_id="o", pixel_size_nm=8.0).values[column] is not None

        features[f"{stored}_renamed"] = features.pop(stored)
        assert derive(features, object_id="o", pixel_size_nm=8.0).values[column] is None


class TestNoFabricatedNumbers:
    """A measurement that was not made is absent. 0.0 is a measurement."""

    def test_mean_probability_is_measured_not_placeheld(self):
        features = _build(prob_value=0.8).features
        assert features["mean_prob"] == pytest.approx(0.8, abs=1e-3)
        assert features["mean_prob_dino"] == pytest.approx(0.8, abs=1e-3)

    def test_a_zero_probability_is_reported_only_when_it_is_the_measurement(self):
        """The gate that used to write 0.0 made this indistinguishable from a
        genuine zero."""
        features = _build(prob_value=0.0).features
        assert features["mean_prob"] == 0.0

    def test_confidence_is_the_object_mean_not_one_pixel(self):
        segment = _build(prob_value=0.8)
        assert segment.confidence_score == pytest.approx(
            segment.features["mean_prob"]
        )

    def test_intensity_is_absent_rather_than_zero_when_there_is_no_image(self):
        features = _build(with_intensity=False).features
        for key in INTENSITY_FEATURE_KEYS:
            assert key not in features, (
                f"{key} was written without an image to measure. A zero here "
                "reads as a measured black object."
            )
        values = derive(features, object_id="o", pixel_size_nm=8.0).values
        for key in ("intensity_mean", "intensity_p10", "intensity_p50", "intensity_p90"):
            assert values[key] is None

    def test_shape_is_measured_with_or_without_an_image(self):
        features = _build(with_intensity=False).features
        for key in SHAPE_FEATURE_KEYS:
            assert key in features

    def test_percentiles_are_ordered_and_inside_the_object(self):
        features = _build().features
        assert (
            features["intensity_p10"]
            <= features["intensity_p50"]
            <= features["intensity_p90"]
        )
        # The ellipse is 100 grey levels brighter than its surround, so a mean
        # taken over the bounding box instead of the mask would fall short.
        assert features["intensity_mean"] > 100.0

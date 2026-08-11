"""``circularity`` is an estimator's output, and the export rules follow from
measurement, not preference.

History, because two estimators have shipped and their failure modes are
opposite. ``regionprops.perimeter`` (bundles whose
``environment.perimeter_estimator`` is absent) walked boundary-pixel centres:
its bias grew monotonically as objects shrank -- a 5 px square read 1.227 --
so a treatment that changed organelle *size* manufactured a roundness
difference. ``perimeter_crofton`` (owner ruling 2026-08-07) is close to
unbiased on round shapes but scatters ~+/-1.5% around 1.0 on a genuinely round
object, on *both* sides of the ceiling.

That scatter forces the reporting rule pinned here:

1. **Values in (1.0, 1.015] are reported as measured.** Blanking at exactly
   1.0 would censor roughly half of the roundest objects -- truncating the
   estimator's sampling distribution from above and biasing every round
   population's mean downward, by more the rounder the group. That is the
   same artefact shape the estimator switch was made to remove.

2. **Values above 1.015 are withheld.** The measured envelope of a true
   circle (discs r=10..100, both rasterised and polygon-drawn) tops out at
   1.011; beyond 1.015 the value measures the estimator failing on a small
   object. The reason travels with the blank.

3. **The estimator note rides along at full n** -- the surviving values are
   still estimates, and cornered shapes still read high by a roughly constant
   factor.

Every number below goes through the app's own
:func:`~quantem.segmentation.features.extraction.compute_segment_features`,
so prose cannot drift from the estimator again.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from quantem.analysis.morphometrics import (
    CIRCULARITY_ESTIMATOR_NOTE,
    CIRCULARITY_KEY,
    CIRCULARITY_MAX,
    CIRCULARITY_REPORT_CEILING,
    derive,
    summarize,
)
from quantem.segmentation.features.extraction import compute_segment_features

#: The default object-size floor a run applies. The point of quoting it here is
#: that the estimator's failure regime is *not* filtered out by it: a drawn
#: disc of r=5 covers 70 px, clears a 60 px floor, and reads above the report
#: ceiling.
DEFAULT_MIN_AREA_PX = 60

PI_OVER_4 = math.pi / 4


def _square(n: int) -> Polygon:
    """An axis-aligned square covering exactly n x n pixels, drawn on the edges.

    The app's pixel convention (``seg_core.rasterize``) puts pixel edges on the
    half-integers, so this outline rasterises to precisely the pixels
    ``[10, 10+n)`` in both axes -- area ``n**2``, and no half-covered border.
    """
    lo, hi = 9.5, 9.5 + n
    return Polygon([(lo, lo), (hi, lo), (hi, hi), (lo, hi)])


def _disc(r: float) -> Polygon:
    return Point(60.0, 60.0).buffer(float(r))


def _measure(polygon: Polygon, size: int = 200) -> dict[str, float]:
    """What the app stores for this outline: the real extractor, real raster."""
    image = np.zeros((size, size), dtype=np.uint8)
    features, _ = compute_segment_features(polygon, image, 2)
    assert features, "the shape rasterised to nothing"
    return features


def _circularity(polygon: Polygon, size: int = 200) -> float:
    features = _measure(polygon, size)
    return 4.0 * math.pi * features["area"] / features["perimeter"] ** 2


class TestCroftonsMeasuredBehaviour:
    """The direction claims, measured through the app's own extractor."""

    def test_a_square_stays_above_pi_over_4_and_flattens(self):
        """Cornered shapes read high by a roughly *constant* factor.

        Constant is the property that matters: a constant factor cancels
        between groups of the same shape class, where the old estimator's
        size-dependent factor did not. The plateau (~0.879, +12% over pi/4)
        is the cost of that trade, and the note states it.
        """
        sizes = [3, 5, 8, 10, 20, 50, 100]
        measured = [_circularity(_square(n), size=n + 40) for n in sizes]

        assert all(c > PI_OVER_4 for c in measured), dict(zip(sizes, measured, strict=True))
        assert measured == sorted(measured, reverse=True), dict(zip(sizes, measured, strict=True))
        # Flat from 50 px up: the residual size dependence is under 1%.
        assert abs(measured[-1] - measured[-2]) < 0.006
        # ...and it does NOT converge on pi/4; the offset is the constant factor.
        assert measured[-1] == pytest.approx(0.879, abs=0.002)

    def test_a_large_disc_reads_close_to_true(self):
        """The reason crofton was chosen: a big circle is measured as one."""
        assert _circularity(_disc(20), size=140) == pytest.approx(1.0, abs=0.015)
        assert _circularity(_disc(50), size=200) == pytest.approx(1.0, abs=0.015)

    def test_small_discs_are_the_failure_regime(self):
        """Below ~r=10 the estimator is unreliable in both directions.

        At 20 nm/px an r=7 disc is a 280 nm profile -- a peroxisome, a small
        lysosome, a vesicle -- so this regime is reachable by ordinary data,
        which is why the report ceiling exists.
        """
        assert _circularity(_disc(3), size=140) < CIRCULARITY_MAX  # low side
        assert _circularity(_disc(5), size=140) > CIRCULARITY_REPORT_CEILING
        assert _circularity(_disc(7), size=140) > CIRCULARITY_REPORT_CEILING

    def test_the_failure_regime_is_inside_the_default_min_area(self):
        """A 70 px drawn disc clears a 60 px floor and still reads > 1.015,
        so the min-area filter does not stand in for the report ceiling."""
        features = _measure(_disc(5), size=140)
        assert features["area"] == 70.0 > DEFAULT_MIN_AREA_PX
        assert _circularity(_disc(5), size=140) > CIRCULARITY_REPORT_CEILING

    def test_the_note_quotes_the_numbers_this_estimator_actually_produces(self):
        """Prose that can go stale is prose that can state the bias backwards."""
        quoted = {20: "0.900", 100: "0.879"}
        for n, text in quoted.items():
            assert text in CIRCULARITY_ESTIMATOR_NOTE
            assert f"{_circularity(_square(n), size=n + 40):.3f}" == text
        assert "perimeter_crofton" in CIRCULARITY_ESTIMATOR_NOTE
        assert "1.015" in CIRCULARITY_ESTIMATOR_NOTE


class TestTheReportingRule:
    def test_derive_blanks_a_failure_and_keeps_the_measurements_behind_it(self):
        features = _measure(_disc(5), size=140)
        assert features["area"] == 70.0

        m = derive(features, object_id="d5", pixel_size_nm=20.0)

        assert m.values[CIRCULARITY_KEY] is None
        # The inputs were measured and are still reported; only the impossible
        # combination of them is withheld.
        assert m.values["area_px"] == 70.0
        assert m.values["perimeter_px"] == pytest.approx(29.362, abs=0.001)
        assert m.values["area_um2"] == pytest.approx(70.0 * 0.02**2)
        assert CIRCULARITY_KEY in m.unreportable
        assert "1.015" in m.unreportable[CIRCULARITY_KEY]

    def test_a_value_inside_the_ceiling_is_untouched(self):
        m = derive(_measure(_square(20), size=60), object_id="s20", pixel_size_nm=20.0)
        assert m.values[CIRCULARITY_KEY] == pytest.approx(0.9000, abs=1e-4)
        assert m.unreportable == {}

    def test_a_round_object_slightly_above_one_is_reported_not_censored(self):
        """The anti-censoring rule, and the reason the report ceiling is not 1.0.

        A genuinely round 7,818 px disc measures 1.0034 under crofton. Blanking
        it would throw away the roundest objects' measurements -- truncating
        the estimator's distribution from above and biasing a round
        population's mean downward, by more the rounder the group. So a value
        in (1.0, 1.015] is a measurement, not a mistake.
        """
        features = _measure(_disc(50), size=200)
        circ = 4.0 * math.pi * features["area"] / features["perimeter"] ** 2
        assert CIRCULARITY_MAX < circ <= CIRCULARITY_REPORT_CEILING

        m = derive(features, object_id="d50", pixel_size_nm=20.0)
        assert m.values[CIRCULARITY_KEY] == pytest.approx(circ)
        assert m.unreportable == {}

    def test_the_summary_separates_a_refusal_from_a_missing_measurement(self):
        metrics = [
            derive(_measure(_disc(r), size=140), object_id=f"d{r}", pixel_size_nm=20.0)
            for r in (5, 7)
        ] + [
            derive(_measure(_square(n), size=n + 40), object_id=f"s{n}", pixel_size_nm=20.0)
            for n in (20, 50)
        ]

        stats = summarize(metrics)[CIRCULARITY_KEY]

        assert stats["n"] == 2
        assert stats["n_objects"] == 4
        assert stats["n_missing"] == 2
        assert stats["n_unreportable"] == 2
        assert stats["max"] <= CIRCULARITY_REPORT_CEILING
        note = stats["note"]
        assert "2 were measured and could not be reported" in note
        assert "1.015" in note
        # ...and the estimator that produced them, not only the two it refused.
        assert CIRCULARITY_ESTIMATOR_NOTE in note

    def test_the_estimator_is_reported_even_when_nothing_was_blanked(self):
        """n_missing == 0 means everything was measured, not that the values
        are geometry."""
        metrics = [
            derive(_measure(_square(n), size=n + 40), object_id=f"s{n}", pixel_size_nm=20.0)
            for n in (20, 50, 100)
        ]

        stats = summarize(metrics)[CIRCULARITY_KEY]

        assert stats["n"] == 3 and stats["n_missing"] == 0
        assert "n_unreportable" not in stats
        note = stats["note"]
        assert CIRCULARITY_ESTIMATOR_NOTE == note
        assert "perimeter_crofton" in note  # which estimator ran
        assert "estimator, not geometry" in note  # what the number is
        assert "censor the roundest" in note  # why >1.0 can be reported

    def test_another_metric_keeps_the_wording_it_had(self):
        """The refusal clause is circularity's; the plain-absence clause is not."""
        metrics = [
            derive(
                {"area": 500.0, "perimeter": 90.0, "mean_prob": 0.8},
                object_id="a",
                pixel_size_nm=20.0,
                source_model="quantem:mito",
            ),
            derive(
                {"area": 500.0, "perimeter": 90.0},
                object_id="b",
                pixel_size_nm=20.0,
                source_model="manual",
            ),
        ]

        note = summarize(metrics)["mean_prob"]["note"]

        assert "Measured on 1 of 2 confirmed objects" in note
        assert "1 carries no stored value for this metric (1 hand-drawn)" in note
        assert "no model probability" in note


class TestTheBundleNeverShipsAnEstimatorFailure:
    """The real entry point: run_analysis -> write_bundle -> objects.csv."""

    def _run(self, tmp_path):
        from quantem.analysis import CompartmentSet
        from quantem.analysis.service import AnalysisInputs, run_analysis, write_bundle

        tissue = np.zeros((200, 200), dtype=bool)
        tissue[10:190, 10:190] = True
        shapes = {
            "d5": _measure(_disc(5), size=140),  # failure regime, blanked
            "d7": _measure(_disc(7), size=140),  # failure regime, blanked
            "s20": _measure(_square(20), size=60),  # reported
            "s50": _measure(_square(50), size=90),  # reported
        }
        inputs = AnalysisInputs(
            image_key="img-circ",
            pixel_size_nm=20.0,
            compartments=CompartmentSet(masks={}, tissue=tissue),
            object_features=shapes,
            object_sources=dict.fromkeys(shapes, "quantem:mito"),
        )
        result = run_analysis(inputs)
        out = write_bundle([result], tmp_path / "bundle")
        rows = list(csv.DictReader((out / "objects.csv").open(encoding="utf-8-sig")))
        return result, rows

    def test_objects_csv_carries_no_circularity_above_the_report_ceiling(self, tmp_path):
        result, rows = self._run(tmp_path)

        assert len(rows) == 4
        values = [row[CIRCULARITY_KEY] for row in rows]
        reported = [float(v) for v in values if v != ""]
        assert reported and all(v <= CIRCULARITY_REPORT_CEILING for v in reported)
        assert sum(1 for v in values if v == "") == 2

        # The blanked row keeps everything that was measured.
        blanked = next(r for r in rows if r["object_id"] == "d5")
        assert float(blanked["area_px"]) == 70.0
        assert float(blanked["perimeter_px"]) == pytest.approx(29.362, abs=0.001)

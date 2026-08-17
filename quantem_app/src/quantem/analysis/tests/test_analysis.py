"""Analysis-suite tests.

These are pure numpy — no database, no Django, no model weights — so they run
everywhere including the CI lane.

The most important test in this file is :class:`VerifyNullTests`, which is a port
of ``gk_gold_seg/scripts/gold_pipeline/verify_null.py``. That script exists in
the reference implementation because the mistake it catches was actually made:
count-weighted pooling produced a random-data enrichment of 0.73 instead of 1.0.
It is a test here rather than a comment so it cannot be made again silently.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from quantem.analysis import (
    CompartmentSet,
    aggregate,
    area_fractions,
    assign_points,
    csr_null,
    distance_to_boundary,
    nearest_neighbour_nm,
    rollup,
    weighted_mean_for_comparison,
)
from quantem.analysis.montecarlo import sample_uniform_in_mask, self_check
from quantem.analysis.morphometrics import density, derive, summarize

H = W = 200


def _disc(cx, cy, r, shape=(H, W)):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def _comp():
    tissue = np.zeros((H, W), dtype=bool)
    tissue[20:180, 20:180] = True  # 160x160 = 25,600 px
    nucleus = _disc(70, 70, 30)  # inside tissue
    mito = _disc(140, 140, 20)  # inside tissue, in cytoplasm
    return CompartmentSet(
        masks={"nucleus": nucleus, "mito": mito},
        tissue=tissue,
        nested_in={"mito": "cytoplasm"},
    )


class TestAreaFractions:
    def test_fractions_are_relative_to_tissue_not_image(self):
        comp = _comp()
        af = area_fractions(comp, pixel_size_nm=10.0)
        assert af.tissue_px == 160 * 160
        # nucleus disc r=30 -> ~pi*900 = 2827 px, over 25600 tissue px
        assert 0.10 < af.fractions["nucleus"] < 0.12
        # cytoplasm is derived as tissue minus nucleus
        assert af.fractions["cytoplasm"] == pytest.approx(1.0 - af.fractions["nucleus"])

    def test_calibrated_areas(self):
        af = area_fractions(_comp(), pixel_size_nm=10.0)
        # 10 nm/px -> 1e-4 um2 per pixel
        assert af.tissue_um2 == pytest.approx(160 * 160 * 1e-4)

    def test_organelles_are_clipped_to_tissue(self):
        tissue = np.zeros((H, W), dtype=bool)
        tissue[0:50, 0:50] = True
        # a mito mask that pokes well outside the tissue
        comp = CompartmentSet(masks={"mito": _disc(50, 50, 40)}, tissue=tissue)
        af = area_fractions(comp)
        assert af.areas_px["mito"] <= af.tissue_px


class TestPointAssignment:
    def test_points_off_tissue_are_excluded_and_counted(self):
        comp = _comp()
        pts = np.array([[70.0, 70.0], [140.0, 140.0], [5.0, 5.0], [1.0, 199.0]])
        a = assign_points(pts, comp)
        assert a.n_total == 4
        assert a.n_off_tissue == 2
        assert a.n_on_tissue == 2
        assert a.counts["nucleus"] == 1
        assert a.counts["mito"] == 1

    def test_enrichment_is_one_for_uniform_points(self):
        comp = _comp()
        rng = np.random.default_rng(0)
        pts = sample_uniform_in_mask(comp.tissue_mask(), 40000, rng).astype(float)
        a = assign_points(pts, comp)
        for name, value in a.enrichment.items():
            assert value is not None
            assert value == pytest.approx(1.0, abs=0.06), name

    def test_enrichment_detects_concentration(self):
        comp = _comp()
        # every point inside the nucleus
        ys, xs = np.where(comp.masks["nucleus"])
        pts = np.column_stack([xs, ys]).astype(float)
        a = assign_points(pts, comp)
        assert a.enrichment["nucleus"] > 8.0
        assert a.enrichment["mito"] == 0.0

    def test_undefined_rather_than_infinite_for_empty_compartment(self):
        comp = CompartmentSet(
            masks={"empty": np.zeros((H, W), dtype=bool)},
            tissue=np.ones((H, W), dtype=bool),
        )
        a = assign_points(np.array([[10.0, 10.0]]), comp)
        assert a.enrichment["empty"] is None


class TestUnreadableCoordinates:
    """A coordinate that is not a number used to become a point at (0, 0).

    ``np.round(nan).astype(int)`` is ``INT_MIN`` and the clip to the image turns
    that into 0, so every unusable row was counted as a real observation at the
    origin. The only trace was a RuntimeWarning on stderr.
    """

    @staticmethod
    def _origin_compartment():
        """mito over [0:10, 0:10] -- 1% of a full-image tissue, and it covers (0, 0)."""
        tissue = np.ones((100, 100), dtype=bool)
        mito = np.zeros((100, 100), dtype=bool)
        mito[0:10, 0:10] = True
        return CompartmentSet(masks={"mito": mito}, tissue=tissue)

    def test_non_finite_rows_are_dropped_and_counted(self):
        comp = self._origin_compartment()
        pts = np.array(
            [
                [80.0, 80.0],
                [float("nan"), float("nan")],
                [float("inf"), 0.0],
                [float("-inf"), 0.0],
            ]
        )
        a = assign_points(pts, comp)

        # Before: n_on_tissue 4, counts {'mito': 3}, enrichment 75.0 -- a
        # 75-fold enrichment out of three coordinates that could not be read.
        assert a.n_total == 4
        assert a.n_unreadable == 3
        assert a.n_on_tissue == 1
        assert a.counts["mito"] == 0
        assert a.enrichment["mito"] == 0.0
        # An unreadable row is not an off-tissue one; the three counts partition.
        assert a.n_off_tissue == 0
        assert a.n_on_tissue + a.n_off_tissue + a.n_unreadable == a.n_total
        assert a.readable.tolist() == [True, False, False, False]
        assert a.on_tissue.tolist() == [True, False, False, False]

    def test_no_runtime_warning_is_emitted_for_them(self):
        comp = self._origin_compartment()
        pts = np.array([[10.0, 10.0], [float("nan"), float("inf")]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assign_points(pts, comp)

    def test_a_finite_coordinate_too_large_to_cast_is_not_a_point_at_the_origin(self):
        """1e300 overflows ``astype(int)`` to INT_MIN and used to clip to 0."""
        comp = self._origin_compartment()
        a = assign_points(np.array([[1e300, 1e300]]), comp)
        assert a.n_unreadable == 0  # it is finite, so it is readable
        assert a.n_out_of_bounds == 1
        assert a.counts["mito"] == 0  # clipped to (99, 99), not to (0, 0)

    def test_points_outside_the_image_are_counted_so_wrong_units_show_up(self):
        comp = self._origin_compartment()
        # A 100 px image measured in nanometres at 10 nm/px: every coordinate
        # is ten times too big and clips onto the far border.
        a = assign_points(np.array([[300.0, 400.0], [50.0, 50.0]]), comp)
        assert a.n_out_of_bounds == 1
        assert a.n_on_tissue == 2  # clipping is still what happens

    def test_an_object_on_the_edge_of_the_image_is_not_called_out_of_bounds(self):
        """Pixel indices stop at w - 1; the image does not. A centroid at 99.6
        on a 100 px image is inside it under either coordinate convention."""
        comp = self._origin_compartment()
        a = assign_points(np.array([[99.6, 99.6], [100.0, 0.0], [-0.4, 50.0]]), comp)
        assert a.n_out_of_bounds == 0

    def test_membership_masks_stay_aligned_with_the_input_rows(self):
        comp = self._origin_compartment()
        pts = np.array([[float("nan"), 0.0], [5.0, 5.0], [80.0, 80.0]])
        a = assign_points(pts, comp)
        assert a.membership["mito"].shape == (3,)
        assert a.membership["mito"].tolist() == [False, True, False]


class TestNoPointOnTissue:
    """0.0 is maximal depletion. 0/0 is not a measurement of anything."""

    @staticmethod
    def _centre_tissue():
        tissue = np.zeros((100, 100), dtype=bool)
        tissue[40:60, 40:60] = True
        mito = np.zeros((100, 100), dtype=bool)
        mito[40:50, 40:50] = True
        return CompartmentSet(masks={"mito": mito}, tissue=tissue)

    def test_enrichment_is_undefined_not_zero(self):
        comp = self._centre_tissue()
        af = area_fractions(comp)
        assert af.fractions["mito"] == 0.25  # the compartment has area
        a = assign_points(np.array([[1.0, 1.0], [2.0, 3.0], [90.0, 90.0]]), comp)
        assert a.n_on_tissue == 0
        assert a.n_off_tissue == 3
        # Before: 0.0, a defined value, which reads as total exclusion from a
        # compartment that occupies a quarter of the tissue.
        assert a.enrichment["mito"] is None

    def test_one_point_on_tissue_is_enough_for_a_real_zero(self):
        """The guard is on the denominator, not on the answer being small."""
        comp = self._centre_tissue()
        a = assign_points(np.array([[55.0, 55.0], [1.0, 1.0]]), comp)
        assert a.n_on_tissue == 1
        assert a.enrichment["mito"] == 0.0  # measured, not fabricated

    def test_the_null_has_no_statistic_to_report_either(self):
        comp = self._centre_tissue()
        null = csr_null(np.array([[1.0, 1.0], [2.0, 3.0]]), comp, image_key="k", replicates=5)
        # The default metric drops undefined enrichments, so there is nothing
        # to compare -- rather than an observed 0.0 against a null of zeros.
        assert null.observed == {}
        assert null.p_two_sided == {}


class TestDistances:
    def test_signed_distance_is_negative_inside(self):
        mask = _disc(100, 100, 30)
        pts = np.array([[100.0, 100.0], [100.0, 160.0]])
        r = distance_to_boundary(pts, mask, pixel_size_nm=10.0)
        assert r.distances_nm[0] < 0  # centre, inside
        assert r.distances_nm[1] > 0  # well outside
        assert r.inside.tolist() == [True, False]

    def test_distance_scales_with_pixel_size(self):
        mask = _disc(100, 100, 30)
        pts = np.array([[100.0, 160.0]])
        a = distance_to_boundary(pts, mask, pixel_size_nm=10.0)
        b = distance_to_boundary(pts, mask, pixel_size_nm=20.0)
        assert abs(b.distances_nm[0]) == pytest.approx(2 * abs(a.distances_nm[0]))

    def test_pixel_size_is_required(self):
        with pytest.raises(ValueError, match="pixel_size_nm"):
            distance_to_boundary(np.array([[1.0, 1.0]]), _disc(100, 100, 30), pixel_size_nm=0)

    def test_bands_partition_the_points(self):
        mask = _disc(100, 100, 30)
        rng = np.random.default_rng(1)
        pts = rng.uniform(0, W, size=(500, 2))
        r = distance_to_boundary(pts, mask, pixel_size_nm=10.0)
        assert sum(r.band_counts) == 500
        assert len(r.band_labels) == len(r.band_counts)

    def test_empty_inputs_do_not_raise(self):
        r = distance_to_boundary(np.empty((0, 2)), _disc(100, 100, 30), pixel_size_nm=10.0)
        assert r.n == 0 and r.median_nm is None


class TestDistancesReadTheSamePointsAsTheAssignment:
    """The two coordinate sites disagreed, and the distances lied harder.

    ``assign_points`` clipped the float and cast after; ``distance_to_boundary``
    cast the float and clipped after, which sends anything outside the int64
    range to ``INT_MIN`` and then to pixel 0. The service handed both functions
    the same array.
    """

    MASK = staticmethod(lambda: _disc(100, 100, 20))

    @staticmethod
    def _comp_with_mito():
        mito = _disc(100, 100, 20)
        return CompartmentSet(masks={"mito": mito}, tissue=np.ones((H, W), bool))

    def test_a_coordinate_too_large_to_cast_is_not_a_point_at_the_origin(self):
        """A median distance of 3.5e+30 nm -- 3.5e+21 metres -- was reported as
        the median distance from a gold particle to a mitochondrion."""
        r = distance_to_boundary(
            np.array([[100.0, 100.0], [1e30, 1e30]]), self.MASK(), pixel_size_nm=5.0
        )
        # Every distance is bounded by the image, because the point is measured
        # from the border pixel it is clipped onto -- the same pixel the
        # compartment counts put it in.
        diagonal_nm = float(np.hypot(H, W)) * 5.0
        assert np.all(np.abs(r.distances_nm) <= diagonal_nm)
        assert r.median_nm is not None and r.median_nm < diagonal_nm
        # (0, 0) is outside the disc; the old code put it inside and reported
        # n_inside = 1 from a coordinate nobody supplied.
        assert r.inside.tolist() == [True, False]
        assert r.n_out_of_image == 1

    def test_a_coordinate_that_is_not_a_position_is_dropped_not_measured(self):
        pts = np.array([[40.0, 40.0], [np.nan, np.nan], [np.inf, 0.0]])
        r = distance_to_boundary(pts, self.MASK(), pixel_size_nm=5.0)
        assert r.n_unreadable == 2
        assert r.n == 1  # one row measured, and it is the real one
        assert r.readable.tolist() == [True, False, False]
        assert sum(r.band_counts) == 1

    def test_an_overflowing_distance_never_becomes_a_silent_nan(self):
        """``1e300`` squared is inf, ``-inf * 0 + inf * 1`` is nan, and
        ``np.histogram`` drops a nan without saying so: band_counts summed to
        one of two points while band_fractions still read 1.0."""
        r = distance_to_boundary(
            np.array([[40.0, 40.0], [1e300, 1e300]]), self.MASK(), pixel_size_nm=5.0
        )
        assert np.all(np.isfinite(r.distances_nm))
        assert sum(r.band_counts) == r.n == 2
        assert sum(f for f in r.band_fractions if f) == pytest.approx(1.0)
        assert r.median_nm is not None and np.isfinite(r.median_nm)

    @pytest.mark.parametrize("point", [[1e30, 1e30], [1e300, 1e300], [np.nan, 1.0], [-np.inf, 5.0]])
    def test_both_functions_report_the_same_population_from_one_array(self, point):
        comp = self._comp_with_mito()
        pts = np.array([[100.0, 100.0], point], dtype=float)
        a = assign_points(pts, comp)
        r = distance_to_boundary(pts, comp.masks["mito"], pixel_size_nm=5.0)
        assert r.n_unreadable == a.n_unreadable
        assert r.n_out_of_image == a.n_out_of_bounds
        assert r.readable.tolist() == a.readable.tolist()
        # And they agree about which points are in the compartment: `inside`
        # covers the readable rows, `membership` covers all of them.
        assert r.inside.tolist() == a.membership["mito"][a.readable].tolist()
        if a.n_out_of_bounds:
            # ...including *how far away* it is. The distance is the one from
            # the border pixel the assignment counts it at, not from a
            # coordinate 1e30 px off the edge of a 200 px image.
            border = np.clip(np.asarray(point, dtype=float), [0, 0], [W - 1, H - 1])
            from_border = distance_to_boundary(
                border.reshape(1, 2), comp.masks["mito"], pixel_size_nm=5.0
            )
            assert r.distances_nm[-1] == pytest.approx(from_border.distances_nm[0])

    def test_nearest_neighbour_excludes_self(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = nearest_neighbour_nm(pts, pixel_size_nm=100.0)
        assert d[0] == pytest.approx(500.0)  # 5 px * 100 nm


class TestMonteCarlo:
    def test_each_replicate_is_cancellable_and_reports_progress(self):
        checks: list[int] = []
        updates: list[tuple[int, int]] = []
        comp = _comp()
        pts = np.array([[1.0, 1.0], [2.0, 3.0]])

        csr_null(
            pts,
            comp,
            image_key="progress",
            replicates=4,
            cancel_check=lambda: checks.append(1),
            on_progress=lambda done, total: updates.append((done, total)),
        )

        assert len(checks) == 4
        assert updates == [(1, 4), (2, 4), (3, 4), (4, 4)]

    def test_seed_is_independent_of_processing_order(self):
        """The defect this port fixes: the reference shared one global RNG, so a
        result depended on how many images were processed before it."""
        comp = _comp()
        pts = np.column_stack(np.where(comp.masks["nucleus"])[::-1]).astype(float)

        alone = csr_null(pts, comp, image_key="img-A", replicates=8, seed=7)
        # process a decoy first; a global RNG would shift the stream
        csr_null(pts, comp, image_key="img-DECOY", replicates=8, seed=7)
        after = csr_null(pts, comp, image_key="img-A", replicates=8, seed=7)

        assert alone.null_mean == after.null_mean
        assert alone.null_samples == after.null_samples

    def test_nuclear_concentration_is_significant(self):
        comp = _comp()
        ys, xs = np.where(comp.masks["nucleus"])
        pts = np.column_stack([xs, ys]).astype(float)
        r = csr_null(pts, comp, image_key="img", replicates=20, seed=3)
        assert r.z["enrichment_nucleus"] > 5.0
        assert r.p_two_sided["enrichment_nucleus"] < 0.05

    def test_uniform_points_are_not_significant(self):
        comp = _comp()
        rng = np.random.default_rng(11)
        pts = sample_uniform_in_mask(comp.tissue_mask(), 3000, rng).astype(float)
        r = csr_null(pts, comp, image_key="img", replicates=20, seed=3)
        assert abs(r.z["enrichment_nucleus"]) < 3.0

    def test_sampling_stays_inside_the_mask(self):
        comp = _comp()
        tis = comp.tissue_mask()
        pts = sample_uniform_in_mask(tis, 2000, np.random.default_rng(5))
        assert tis[pts[:, 1], pts[:, 0]].all()

    def test_self_check_recovers_unity(self):
        """The manuscript's stated internal control."""
        out = self_check(_comp())
        assert out["max_abs_deviation"] < 0.1
        assert out["skipped_reason"] is None

    def test_self_check_draw_is_proportionate_to_the_mask(self):
        """It runs on *every* analysis, so its cost has to be bounded by the
        geometry rather than by the smallest compartment alone.

        Past ``tissue_px`` draws the estimate has already converged on the exact
        area fraction it is compared against -- more points buy nothing and cost
        linear time on every run.
        """
        from quantem.analysis.montecarlo import (
            SELF_CHECK_MAX_POINTS,
            SELF_CHECK_MIN_POINTS,
        )

        comp = _comp()
        tissue_px = int(comp.tissue_mask().sum())
        out = self_check(comp)
        assert out["n_points"] <= tissue_px
        assert out["n_points"] <= SELF_CHECK_MAX_POINTS

        # A compartment small enough to ask for far more than the cap still
        # cannot make the check unbounded.
        sliver = np.zeros((H, W), dtype=bool)
        sliver[100, 100:102] = True  # 2 px out of 40,000
        big = CompartmentSet(masks={"sliver": sliver}, tissue=np.ones((H, W), dtype=bool))
        capped = self_check(big)
        assert capped["n_points"] <= SELF_CHECK_MAX_POINTS
        assert capped["n_points"] >= SELF_CHECK_MIN_POINTS

    def test_a_null_with_no_spread_reports_no_p_either(self):
        """``_z`` was guarded and ``_p_two_sided`` was not.

        One point in a compartment that covers 1.5 % of the tissue: twenty
        simulated singletons all miss it, the null is twenty identical zeros,
        and the empirical p comes out 1/21 = 0.0476 -- the *smallest* value
        twenty replicates can produce, and the first number anyone compares
        against 0.05 -- from a null with literally no variance and one point
        that cannot exhibit spatial structure at all.
        """
        sliver = np.zeros((H, W), dtype=bool)
        sliver[100:120, 100:120] = True  # 400 px of 40,000
        comp = CompartmentSet(masks={"mito": sliver}, tissue=np.ones((H, W), dtype=bool))
        null = csr_null(np.array([[110.0, 110.0]]), comp, image_key="one", replicates=20, seed=3)

        assert null.observed["enrichment_mito"] > 1.0  # the point is in it
        assert null.null_sd["enrichment_mito"] == 0.0  # and the null is flat
        assert null.z["enrichment_mito"] is None
        assert null.p_two_sided["enrichment_mito"] is None

    def test_a_single_replicate_has_no_sd_to_report(self):
        """One draw has no sample SD. Reporting 0.0 for it claims a null with
        no spread was measured, which is the same claim by another route."""
        comp = _comp()
        ys, xs = np.where(comp.masks["nucleus"])
        pts = np.column_stack([xs, ys]).astype(float)
        null = csr_null(pts, comp, image_key="img", replicates=1, seed=3)
        assert null.null_sd["enrichment_nucleus"] is None
        assert null.z["enrichment_nucleus"] is None
        assert null.p_two_sided["enrichment_nucleus"] is None
        assert null.null_mean["enrichment_nucleus"] is not None

    def test_a_null_mean_with_no_draws_behind_it_is_not_a_nan(self):
        """``nan`` is not JSON, and ``AnalysisRun.results`` is checked with
        ``JSON_VALID``: a nan here is a row the database refuses to store, at
        the end of a run that has already written its export bundle.

        ``metric`` is a documented extension point, and a statistic the observed
        points have and the null draws do not is how the mean ends up with no
        draws behind it.
        """
        comp = _comp()
        calls = {"n": 0}

        def metric(pts: np.ndarray) -> dict[str, float]:
            calls["n"] += 1
            if calls["n"] == 1:  # the observed call
                return {"shared": 1.0, "observed_only": 2.0}
            return {"shared": 1.0}

        null = csr_null(
            np.array([[100.0, 100.0]]),
            comp,
            image_key="k",
            metric=metric,
            replicates=3,
        )
        assert null.observed["observed_only"] == 2.0
        assert null.null_mean["observed_only"] is None
        assert null.null_sd["observed_only"] is None
        assert null.z["observed_only"] is None
        assert null.p_two_sided["observed_only"] is None

    def test_self_check_on_an_empty_mask_is_reported_not_raised(self):
        """A control that cannot run must not take the analysis down with it."""
        empty = CompartmentSet(
            masks={"mito": np.zeros((H, W), dtype=bool)},
            tissue=np.zeros((H, W), dtype=bool),
        )
        out = self_check(empty)
        assert out["n_points"] == 0
        assert "empty" in out["skipped_reason"]
        assert out["max_abs_deviation"] is None

    def test_a_custom_metric_sees_the_same_population_on_both_sides(self):
        """``metric`` is the documented extension point.

        The null can only scatter points inside the tissue, so the observed side
        has to be restricted to its on-tissue points too. It used to be handed
        *all* of them: a statistic that is not already a ratio -- a band
        fraction, a nearest-neighbour median, a raw count -- was then compared
        against a null of a different size.
        """
        comp = _comp()
        on = np.array([[70.0, 70.0], [140.0, 140.0]])
        off = np.array([[5.0, 5.0], [195.0, 195.0], [1.0, 1.0]])
        pts = np.vstack([on, off])
        assert assign_points(pts, comp).n_off_tissue == 3

        shapes: list[tuple[int, int]] = []

        def counting_metric(p: np.ndarray) -> dict[str, float]:
            shapes.append(p.shape)
            return {"n_points": float(len(p))}

        r = csr_null(pts, comp, image_key="img", metric=counting_metric, replicates=4, seed=1)

        # One observed call plus one per replicate, all the same size and width.
        assert len(shapes) == 5
        assert set(shapes) == {(2, 2)}
        assert r.observed["n_points"] == 2.0
        assert r.null_mean["n_points"] == 2.0

    def test_the_default_metric_is_unchanged_by_the_restriction(self):
        """Enrichment normalises the population away internally, so the fix
        above must not move any number the default path reports."""
        comp = _comp()
        ys, xs = np.where(comp.masks["nucleus"])
        inside = np.column_stack([xs, ys]).astype(float)
        pts = np.vstack([inside, np.array([[5.0, 5.0], [195.0, 195.0]])])

        r = csr_null(pts, comp, image_key="img", replicates=6, seed=3)
        direct = assign_points(pts, comp)

        assert r.observed["enrichment_nucleus"] == pytest.approx(direct.enrichment["nucleus"])


class VerifyNullTests:
    """Port of ``verify_null.py`` — the aggregation guard.

    Three synthetic "animals" with wildly different point counts, each
    individually at chance. The unweighted per-animal mean must recover ~1.0.
    A count-weighted mean must not be used, and this test records what it would
    have produced instead.
    """

    @staticmethod
    def _unit(n_points: int, seed: int) -> dict[str, float]:
        comp = _comp()
        rng = np.random.default_rng(seed)
        pts = sample_uniform_in_mask(comp.tissue_mask(), n_points, rng).astype(float)
        a = assign_points(pts, comp)
        return {
            "group": "g",
            "enrichment_nucleus": a.enrichment["nucleus"],
            "n_on_tissue": a.n_on_tissue,
        }


class TestRollup:
    def test_unweighted_mean_over_units_recovers_unity(self):
        rows = [
            VerifyNullTests._unit(4000, 1),
            VerifyNullTests._unit(400, 2),
            VerifyNullTests._unit(40, 3),
        ]
        grouped = rollup(rows, group_key="group", metrics=["enrichment_nucleus"])
        agg = grouped["g"]["enrichment_nucleus"]
        assert agg.n_units == 3
        assert agg.mean == pytest.approx(1.0, abs=0.25)

    def test_sem_uses_ddof_one(self):
        agg = aggregate([1.0, 2.0, 3.0])
        assert agg.sd == pytest.approx(1.0)
        assert agg.sem == pytest.approx(1.0 / np.sqrt(3))

    def test_single_unit_has_no_spread(self):
        agg = aggregate([2.0])
        assert agg.mean == 2.0 and agg.sd is None and agg.sem is None

    def test_nones_are_dropped_not_zeroed(self):
        agg = aggregate([1.0, None, 3.0])
        assert agg.n_units == 2 and agg.mean == pytest.approx(2.0)

    def test_weighted_helper_is_available_only_for_contrast(self):
        vals = [1.0, 1.0, 0.2]
        weights = [1.0, 1.0, 1000.0]
        assert weighted_mean_for_comparison(vals, weights) < 0.3
        assert aggregate(vals).mean == pytest.approx(0.733, abs=0.01)


class TestMorphometrics:
    def test_uncalibrated_is_flagged_not_guessed(self):
        m = derive({"area": 100.0, "perimeter": 40.0}, object_id="a", pixel_size_nm=None)
        assert m.calibrated is False
        assert "area_um2" not in m.values
        assert m.values["area_px"] == 100.0

    def test_calibrated_conversion(self):
        m = derive({"area": 10000.0, "perimeter": 400.0}, object_id="a", pixel_size_nm=10.0)
        assert m.calibrated is True
        # 10000 px * (0.01 um)^2 = 1.0 um2
        assert m.values["area_um2"] == pytest.approx(1.0)
        assert m.values["perimeter_um"] == pytest.approx(4.0)

    def test_circularity_of_a_disc_is_near_one(self):
        mask = _disc(100, 100, 40)
        area = float(mask.sum())
        perim = float(2 * np.pi * 40)
        m = derive({"area": area, "perimeter": perim}, object_id="d", pixel_size_nm=1.0)
        assert m.values["circularity"] == pytest.approx(1.0, abs=0.05)

    def test_summary_reports_n_and_spread(self):
        ms = [
            derive({"area": a}, object_id=str(i), pixel_size_nm=1.0)
            for i, a in enumerate([1.0, 2.0, 3.0, 4.0])
        ]
        s = summarize(ms, keys=["area_px"])
        assert s["area_px"]["n"] == 4
        assert s["area_px"]["median"] == pytest.approx(2.5)
        assert s["area_px"]["iqr"] == pytest.approx(1.5)

    def test_density_needs_calibration(self):
        assert density(10, tissue_area_px=1000, pixel_size_nm=None)["per_um2"] is None
        d = density(10, tissue_area_px=10000, pixel_size_nm=10.0)
        assert d["tissue_um2"] == pytest.approx(1.0)
        assert d["per_um2"] == pytest.approx(10.0)


class TestService:
    """The bundle layer: does an analysis run produce reportable files?"""

    def _inputs(self, tmp_path, *, pixel_size_nm=10.0, with_points=True):
        from quantem.analysis.service import AnalysisInputs

        comp = _comp()
        pts = None
        if with_points:
            ys, xs = np.where(comp.masks["nucleus"])
            keep = np.arange(0, len(xs), 7)
            pts = np.column_stack([xs[keep], ys[keep]]).astype(float)
        return AnalysisInputs(
            image_key="img-1",
            pixel_size_nm=pixel_size_nm,
            compartments=comp,
            object_features={
                "o1": {"area": 500.0, "perimeter": 80.0, "eccentricity": 0.3},
                "o2": {"area": 900.0, "perimeter": 110.0, "eccentricity": 0.6},
            },
            points_xy=pts,
            distance_target="mito",
            group="fasted",
        )

    def test_run_produces_every_section(self, tmp_path):
        from quantem.analysis.service import run_analysis

        r = run_analysis(self._inputs(tmp_path))
        assert r["calibrated"] is True
        assert r["objects"]["n"] == 2
        assert r["composition"]["tissue_um2"] > 0
        assert "enrichment" in r["points"]
        assert r["distances"]["target"] == "mito"
        assert "z" in r["monte_carlo"]

    def test_uncalibrated_run_is_flagged_with_a_caveat(self, tmp_path):
        from quantem.analysis.service import run_analysis

        r = run_analysis(self._inputs(tmp_path, pixel_size_nm=None))
        assert r["calibrated"] is False
        assert any("Pixel size" in c for c in r["caveats"])
        assert "distances" not in r  # cannot express nm without calibration

    def test_a_skipped_distance_analysis_is_named_not_merely_absent(self, tmp_path):
        """The user asked for distances, the job succeeded, and the section was
        simply gone -- with only the generic pixel-size caveat to explain it.

        Honesty rule 6: say which analysis did not run, by name.
        """
        from quantem.analysis.service import run_analysis

        r = run_analysis(self._inputs(tmp_path, pixel_size_nm=None))

        assert "distances" not in r
        skipped = [c for c in r["caveats"] if "Distance-to-mito" in c]
        assert skipped, r["caveats"]
        assert "skipped" in skipped[0]
        assert "pixel size" in skipped[0]

    def test_a_distance_target_with_no_points_is_also_named(self, tmp_path):
        from quantem.analysis.service import run_analysis

        r = run_analysis(self._inputs(tmp_path, with_points=False))

        assert "distances" not in r
        assert any("Distance-to-mito was requested but skipped" in c for c in r["caveats"])

    def test_an_empty_tissue_mask_is_a_caveat_not_a_crash(self, tmp_path):
        """``self_check`` always drew >= 5,000 points, and sampling inside an
        empty mask raises. The run is entitled to succeed and say why."""
        from quantem.analysis.service import AnalysisInputs, run_analysis

        comp = CompartmentSet(
            masks={"mito": _disc(140, 140, 20)},
            tissue=np.zeros((H, W), dtype=bool),
        )
        r = run_analysis(
            AnalysisInputs(
                image_key="img-empty",
                pixel_size_nm=10.0,
                compartments=comp,
                object_features={"o1": {"area": 500.0, "perimeter": 80.0}},
                points_xy=np.array([[140.0, 140.0], [70.0, 70.0]]),
            )
        )

        assert r["composition"]["tissue_px"] == 0
        assert "monte_carlo" not in r
        assert "monte_carlo_self_check" not in r
        empty = [c for c in r["caveats"] if "tissue mask is empty" in c]
        assert empty, r["caveats"]
        assert "no confirmed objects" in empty[0]

    def test_objects_csv_has_a_header_when_there_are_no_objects(self, tmp_path):
        """A zero-byte CSV is not an empty table.

        ``pandas.read_csv`` raises ``EmptyDataError`` on it, and the bundle is
        what a paper cites -- the reader hitting that error has no way to tell a
        run with no confirmed objects from a corrupt export.
        """
        import csv as _csv

        from quantem.analysis.service import (
            OBJECT_CSV_FIELDS,
            run_analysis,
            write_bundle,
        )

        inputs = self._inputs(tmp_path)
        inputs.object_features.clear()
        out = write_bundle([run_analysis(inputs)], tmp_path / "empty")

        for name, expected in (
            ("objects.csv", "object_id"),
            ("image_summary.csv", "image_key"),
        ):
            text = (out / name).read_text(encoding="utf-8")
            assert text.strip(), f"{name} is zero-byte"
            reader = _csv.DictReader((out / name).open(encoding="utf-8-sig"))
            assert expected in (reader.fieldnames or []), name

        rows = list(_csv.DictReader((out / "objects.csv").open(encoding="utf-8-sig")))
        assert rows == []
        header = _csv.reader((out / "objects.csv").open(encoding="utf-8-sig"))
        assert next(header) == list(OBJECT_CSV_FIELDS)

    def test_the_empty_bundle_is_readable_by_pandas(self, tmp_path):
        """The failure this guards is ``EmptyDataError``, so read it the way the
        person who hit it does."""
        pd = pytest.importorskip("pandas")

        from quantem.analysis.service import run_analysis, write_bundle

        inputs = self._inputs(tmp_path)
        inputs.object_features.clear()
        out = write_bundle([run_analysis(inputs)], tmp_path / "empty-pandas")

        objects = pd.read_csv(out / "objects.csv")
        summary = pd.read_csv(out / "image_summary.csv")
        assert len(objects) == 0
        assert "area_um2" in objects.columns
        assert len(summary) == 1

    def test_declared_object_columns_match_what_derive_produces(self):
        """The header is declared, not discovered, so it can drift. It must not."""
        from quantem.analysis.morphometrics import OBJECT_ROW_FIELDS

        full = {
            "area": 100.0,
            "perimeter": 40.0,
            "eccentricity": 0.3,
            "solidity": 0.9,
            "elongation": 1.1,
            "major_axis_length": 12.0,
            "minor_axis_length": 10.0,
            "feret_diameter_max": 13.0,
            "intensity_mean": 100.0,
            "intensity_p10": 90.0,
            "intensity_p50": 100.0,
            "intensity_p90": 110.0,
            "mean_prob": 0.8,
        }
        calibrated = derive(full, object_id="a", pixel_size_nm=5.0).as_row()
        uncalibrated = derive(full, object_id="a", pixel_size_nm=None).as_row()

        assert set(calibrated) == set(OBJECT_ROW_FIELDS)
        assert set(uncalibrated) <= set(OBJECT_ROW_FIELDS)

    def test_bundle_writes_tables_and_manifest(self, tmp_path):
        from quantem.analysis.service import run_analysis, write_bundle

        out = write_bundle([run_analysis(self._inputs(tmp_path))], tmp_path / "run1")
        assert (out / "objects.csv").exists()
        assert (out / "image_summary.csv").exists()
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_images"] == 1
        assert "unweighted means" in manifest["aggregation_rule"]
        assert manifest["monte_carlo"]["seeding"].startswith("per (image, replicate)")

    def test_object_rows_carry_their_image(self, tmp_path):
        import csv as _csv

        from quantem.analysis.service import run_analysis, write_bundle

        out = write_bundle([run_analysis(self._inputs(tmp_path))], tmp_path / "run2")
        rows = list(_csv.DictReader((out / "objects.csv").open(encoding="utf-8-sig")))
        assert len(rows) == 2
        assert {r["image_key"] for r in rows} == {"img-1"}
        assert all(r["group"] == "fasted" for r in rows)
        assert float(rows[0]["area_um2"]) > 0

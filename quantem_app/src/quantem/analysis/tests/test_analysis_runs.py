"""Database-backed analysis tests: loaders, service, job and API.

``test_analysis.py`` covers the numerics with hand-built arrays. This file
covers everything between those numerics and the app -- a real segmentation with
real confirmed objects, run through the real service, producing a real bundle on
disk.

The bundle is checked as a file, not as a return value. A result the user cannot
open in Excel is not a result.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from unittest import mock

from django.test import TestCase, override_settings
from rest_framework.test import APIClient, APIRequestFactory
from shapely.geometry import Polygon

from quantem.analysis import loaders, service
from quantem.analysis.job import run_job
from quantem.analysis.models import AnalysisRun
from quantem.core.config import STORAGE_DIR
from quantem.jobs.constants import JOB_TYPE_RUN_ANALYSIS
from quantem.jobs.models import Job
from quantem.jobs.reporter import CancelToken, JobCancelledError, JobReporter
from quantem.segmentation.api_views.analysis import AnalysisRunExportView
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import TEST_PIXEL_SIZE_NM, create_small_test_image

IMAGE_SIZE = 200

#: Confirmed mitochondria: three 20x20 squares well inside the tissue.
MITO_BOXES = ((30, 30), (90, 40), (60, 120))
#: A candidate the user has not accepted. Nothing may count it.
CANDIDATE_BOX = (150, 150)


def _square(x: float, y: float, side: float = 20.0) -> Polygon:
    return Polygon(((x, y), (x + side, y), (x + side, y + side), (x, y + side), (x, y)))


class AnalysisRunTestCase(TestCase):
    """Shared fixture: one calibrated image, mitochondria, and a tissue mask."""

    def setUp(self) -> None:
        self.client = APIClient()
        self.image = create_small_test_image("Analysis Image", width=IMAGE_SIZE, height=IMAGE_SIZE)
        self.asset = self.image.asset

        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        self.tissue = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_tissue_type()
        )

        self.objects = [
            self._segment(self.segmentation, _square(x, y), "CONFIRMED") for x, y in MITO_BOXES
        ]
        self.candidate = self._segment(self.segmentation, _square(*CANDIDATE_BOX), "CANDIDATE")
        # Tissue is the middle of the image: the corners are resin, and every
        # fraction below is relative to this, not to the whole frame.
        self._segment(self.tissue, _square(20, 20, side=160), "CONFIRMED")

        # Each run writes into its own directory under a per-test root, so a
        # failed assertion leaves the evidence and the next test starts clean.
        self.exports_root = STORAGE_DIR / "exports_test" / self.id().rsplit(".", 1)[-1]
        shutil.rmtree(self.exports_root, ignore_errors=True)
        patcher = mock.patch.object(service, "EXPORTS_DIR", self.exports_root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.exports_root, ignore_errors=True)

    @staticmethod
    def _segment(
        segmentation: ImageSegmentation, polygon: Polygon, label_state: str
    ) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            features={
                "area": polygon.area,
                "perimeter": polygon.length,
                "eccentricity": 0.25,
                "solidity": 0.98,
                "major_axis_length": 22.0,
                "minor_axis_length": 20.0,
                "intensity_mean": 100.0,
            },
        )

    def _make_run(self, **params) -> AnalysisRun:
        normalised = loaders.normalise_params(params, segmentation=self.segmentation)
        return AnalysisRun.objects.create(
            segmentation=self.segmentation,
            params=normalised,
            group=normalised["group"],
        )


class LoaderTests(AnalysisRunTestCase):
    def test_only_confirmed_objects_are_rasterised(self):
        mask = loaders.segmentation_mask(self.segmentation, (IMAGE_SIZE, IMAGE_SIZE))

        for x, y in MITO_BOXES:
            self.assertTrue(mask[y + 10, x + 10], f"confirmed box at {(x, y)} missing")
        cx, cy = CANDIDATE_BOX
        self.assertFalse(
            mask[cy + 10, cx + 10],
            "a CANDIDATE object was rasterised; only confirmed objects count",
        )
        # Three 20x20 squares are 3 * 400 px. This used to read 3 * 21 * 21 =
        # 1323, because cv2.fillPoly painted both boundaries of every span and a
        # square spanning 20 px covered 21. The mask is now what was drawn --
        # see quantem.seg_core.rasterize for the convention.
        self.assertEqual(int(mask.sum()), 3 * 20 * 20)

    def test_holes_survive_rasterisation(self):
        """``polygon_to_mask`` would fill this in; the tissue tool cuts holes."""
        outer = _square(0, 0, side=100)
        inner = _square(40, 40, side=20)
        with_hole = outer.difference(inner)
        # A second image: one tissue segmentation per asset is a DB constraint.
        other = create_small_test_image("Holed Tissue", width=IMAGE_SIZE, height=IMAGE_SIZE)
        segmentation = ImageSegmentation.objects.create(
            asset=other.asset, segmentation_type=get_or_create_tissue_type()
        )
        SegmentObject.objects.create(
            segmentation=segmentation,
            geometry=with_hole,
            centroid=outer.centroid,
            bbox=outer.envelope,
            label_state="CONFIRMED",
        )

        mask = loaders.segmentation_mask(segmentation, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(mask[10, 10], "the ring itself must be filled")
        self.assertFalse(mask[50, 50], "the hole must not be filled")

    def test_features_and_centroids_come_from_confirmed_objects(self):
        features = loaders.object_features(self.segmentation)
        centroids = loaders.object_centroids(self.segmentation)

        self.assertEqual(len(features), len(MITO_BOXES))
        self.assertNotIn(str(self.candidate.id), features)
        self.assertEqual(centroids.shape, (len(MITO_BOXES), 2))
        # Centroids are the stored float columns, not a recomputed polygon
        # centre. Compared as a set: objects created in the same clock tick tie
        # on created_at, and their order is then the UUID's, not the caller's.
        self.assertEqual(
            {tuple(row) for row in centroids},
            {(x + 10.0, y + 10.0) for x, y in MITO_BOXES},
        )

    def test_pixel_size_comes_from_the_asset(self):
        self.assertEqual(loaders.pixel_size_nm(self.segmentation), TEST_PIXEL_SIZE_NM)
        self.asset.pixel_size_nm = None
        self.asset.save(update_fields=["pixel_size_nm"])
        self.segmentation.refresh_from_db()
        self.assertIsNone(loaders.pixel_size_nm(self.segmentation))

    def test_compartments_default_to_the_segmentation_under_its_analysis_name(self):
        """``quantem_internal_mito`` is a database key, not a column header."""
        params = loaders.normalise_params({}, segmentation=self.segmentation)
        self.assertEqual(params["compartments"], {"mito": str(self.segmentation.id)})

    def test_a_compartment_on_another_image_is_refused(self):
        other = create_small_test_image("Other Image", width=64, height=64)
        elsewhere = ImageSegmentation.objects.create(
            asset=other.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        with self.assertRaises(loaders.AnalysisInputError) as caught:
            loaders.normalise_params(
                {"compartments": {"mito": str(elsewhere.id)}},
                segmentation=self.segmentation,
            )
        self.assertIn("different image", str(caught.exception))

    def test_a_malformed_segmentation_id_is_a_bad_request_not_a_crash(self):
        with self.assertRaises(loaders.AnalysisInputError):
            loaders.normalise_params(
                {"compartments": {"mito": "not-a-uuid"}},
                segmentation=self.segmentation,
            )

    def test_distance_target_must_name_a_compartment(self):
        with self.assertRaises(loaders.AnalysisInputError):
            loaders.normalise_params(
                {"points_source": "centroids", "distance_target": "golgi"},
                segmentation=self.segmentation,
            )

    def test_a_band_edge_that_is_not_a_length_is_refused(self):
        """NaN and infinity survive ``float()`` and defeat both shape checks.

        NaN compares False against everything, so ``[0, nan]`` passes a
        strictly-increasing test; ``inf`` genuinely increases. Both then reached
        ``AnalysisRun.params``, where ``json.dumps`` writes a bare ``NaN`` token
        -- not JSON -- and SQLite's ``JSON_VALID`` check rejected the insert.
        """
        for edges in ([0, "nan"], [0, "inf"], ["-inf", 0], [0, 1e999]):
            with self.subTest(edges=edges):
                with self.assertRaises(loaders.AnalysisInputError) as caught:
                    loaders.normalise_params(
                        {"band_edges_nm": edges}, segmentation=self.segmentation
                    )
                self.assertIn("finite", str(caught.exception))
                self.assertIn("band_edges_nm", str(caught.exception))

    def test_a_finite_band_edge_list_is_still_accepted(self):
        params = loaders.normalise_params(
            {"band_edges_nm": [0, 25, 1e300]}, segmentation=self.segmentation
        )
        self.assertEqual(params["band_edges_nm"], [0.0, 25.0, 1e300])

    def test_a_non_finite_replicate_count_or_seed_is_refused(self):
        """``int(nan)`` is a ValueError; ``int(inf)`` is an *OverflowError*.

        Only the first was caught, so the second left the validator as an
        OverflowError rather than an :class:`AnalysisInputError`.
        """
        for params in (
            {"replicates": float("inf")},
            {"replicates": float("nan")},
            {"seed": float("inf"), "replicates": 5},
            {"seed": float("nan"), "replicates": 5},
        ):
            with self.subTest(params=params):
                with self.assertRaises(loaders.AnalysisInputError) as caught:
                    loaders.normalise_params(params, segmentation=self.segmentation)
                self.assertIn("whole numbers", str(caught.exception))

    def test_points_csv_is_parsed_with_or_without_a_header(self):
        with_header = loaders.parse_points_csv("x,y\n10,20\n30,40\n")
        without = loaders.parse_points_csv("10,20\n30,40")
        self.assertEqual(with_header.xy.tolist(), [[10.0, 20.0], [30.0, 40.0]])
        self.assertEqual(without.xy.tolist(), with_header.xy.tolist())
        self.assertEqual(with_header.unreadable, ())
        self.assertIsNone(with_header.caveat())

    def test_points_csv_rejects_junk_rather_than_dropping_it(self):
        with self.assertRaises(loaders.AnalysisInputError) as caught:
            loaders.parse_points_csv("10,20\nnot,a number\n")
        self.assertIn("line 2", str(caught.exception))

    def test_a_coordinate_that_is_not_a_position_is_dropped_and_named(self):
        """``nan``/``inf`` parse as floats and used to become points at (0, 0)."""
        parsed = loaders.parse_points_csv("x,y\n10,20\nnan,nan\ninf,0\n-inf,0\n1e400,5\n30,40\n")
        self.assertEqual(parsed.xy.tolist(), [[10.0, 20.0], [30.0, 40.0]])
        self.assertEqual(parsed.n_unreadable, 4)
        self.assertEqual([line for line, _ in parsed.unreadable], [3, 4, 5, 6])
        caveat = parsed.caveat()
        self.assertIn("4 of 6 rows", caveat)
        self.assertIn("line 3 (nan,nan)", caveat)
        self.assertIn("not (0, 0)", caveat)

    def test_a_csv_of_nothing_but_unreadable_rows_is_an_error(self):
        with self.assertRaises(loaders.AnalysisInputError) as caught:
            loaders.parse_points_csv("x,y\nnan,nan\ninf,inf\n")
        self.assertIn("nothing", str(caught.exception).lower())


class NoticeTypographyTests(AnalysisRunTestCase):
    """The caveats are rendered verbatim; ``--`` is not a dash on a screen.

    The Analysis screen puts ``run.caveats`` straight into its "Read before
    quoting these numbers" panel, one ``<li>`` per sentence, and the rest of the
    UI writes an em dash. A reader who has just been told the count may be wrong
    should not also be reading source-code punctuation.

    The same strings are the ``caveats`` column of ``image_summary.csv``, which
    is why the writer gained a byte-order mark: an em dash in a UTF-8 CSV that
    Excel decodes as the system codepage becomes three mojibake characters, and
    that would be a worse trade than the ``--`` it replaced.
    """

    def _run_with_caveats(self) -> AnalysisRun:
        # A rejected candidate, so the proofreading caveat -- one of the
        # sentences with a dash in it -- is in the notice deterministically.
        self._segment(self.segmentation, _square(160, 20), loaders.REJECTED)
        run = self._make_run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="centroids",
            distance_target="mito",
            replicates=5,
        )
        service.run_for_segmentation(run)
        run.refresh_from_db()
        return run

    def test_no_caveat_the_screen_renders_contains_a_literal_double_hyphen(self):
        run = self._run_with_caveats()
        caveats = run.results["caveats"]
        self.assertTrue(caveats, "this fixture is meant to produce caveats")
        offenders = [c for c in caveats if "--" in c]
        self.assertEqual(offenders, [], "these render as -- in the notice panel")
        self.assertTrue(
            any("—" in c for c in caveats),
            "at least one of these sentences has a dash in it at all",
        )

    def test_the_summary_csv_carries_the_mark_that_makes_excel_read_it(self):
        run = self._run_with_caveats()
        path = Path(run.export_dir) / "image_summary.csv"

        self.assertTrue(
            path.read_bytes().startswith(b"\xef\xbb\xbf"),
            "without the BOM Excel decodes the caveats column as the system "
            "codepage and turns every em dash into three characters",
        )
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        self.assertEqual(list(rows[0])[0], "image_key")
        self.assertNotIn("--", rows[0]["caveats"])

    def test_the_manifest_writes_the_character_not_its_escape(self):
        """``json.dumps`` escapes non-ASCII by default, which would put a
        literal ``\\u2014`` in a file people open in a text editor."""
        run = self._run_with_caveats()
        text = (Path(run.export_dir) / "manifest.json").read_text(encoding="utf-8")

        self.assertNotIn("\\u2014", text)
        self.assertIn("—", text)
        json.loads(text)  # still JSON


class ServiceTests(AnalysisRunTestCase):
    def test_run_writes_a_bundle_and_stores_the_result(self):
        run = self._make_run(
            tissue_segmentation_id=str(self.tissue.id), group="fasted", replicates=5
        )

        service.run_for_segmentation(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_SUCCESS)
        self.assertEqual(run.error, "")
        self.assertIsNotNone(run.finished_at)
        self.assertEqual(run.group, "fasted")

        bundle = Path(run.export_dir)
        self.assertTrue(bundle.is_dir(), run.export_dir)
        for name in ("objects.csv", "image_summary.csv", "manifest.json"):
            self.assertTrue((bundle / name).is_file(), f"{name} was not written")

        results = run.results
        self.assertTrue(results["calibrated"])
        self.assertEqual(results["pixel_size_nm"], TEST_PIXEL_SIZE_NM)
        self.assertEqual(results["objects"]["n"], len(MITO_BOXES))
        self.assertEqual(results["n_object_rows"], len(MITO_BOXES))
        # Tissue is the 160x160 square, so the denominator is not the frame --
        # and it is 160 * 160, not the 161 * 161 the boundary-inclusive fill
        # used to make it (quantem.seg_core.rasterize).
        self.assertEqual(results["composition"]["tissue_px"], 160 * 160)
        self.assertGreater(results["composition"]["tissue_um2"], 0.0)
        self.assertIn("mito", results["composition"]["area_fractions"])
        # No dataclasses leaked into the JSON column.
        json.dumps(results)

    def test_object_rows_are_calibrated_and_traceable(self):
        run = self._make_run(tissue_segmentation_id=str(self.tissue.id), replicates=5)
        service.run_for_segmentation(run)
        run.refresh_from_db()

        rows = list(
            csv.DictReader((Path(run.export_dir) / "objects.csv").open(encoding="utf-8-sig"))
        )
        self.assertEqual(len(rows), len(MITO_BOXES))
        self.assertEqual({row["image_key"] for row in rows}, {str(self.asset.id)})
        self.assertTrue(all(row["calibrated"] == "True" for row in rows))
        # 400 px at 5 nm/px = 400 * (0.005 um)^2 = 0.01 um^2.
        self.assertAlmostEqual(float(rows[0]["area_um2"]), 0.01, places=6)

    def test_manifest_records_the_mask_provenance(self):
        run = self._make_run(tissue_segmentation_id=str(self.tissue.id), replicates=5)
        service.run_for_segmentation(run)
        run.refresh_from_db()

        manifest = json.loads((Path(run.export_dir) / "manifest.json").read_text(encoding="utf-8"))
        models = manifest["models"]
        self.assertIn("Confirmed segment objects", models["mask_source"])
        self.assertEqual(models["image_id"], str(self.asset.id))
        compartments = {entry["compartment"]: entry for entry in models["compartments"]}
        self.assertEqual(compartments["mito"]["n_confirmed_objects"], len(MITO_BOXES))
        self.assertIsNotNone(models["tissue"])
        self.assertIn("unweighted means", manifest["aggregation_rule"])

    def test_missing_tissue_mask_is_stated_not_assumed(self):
        run = self._make_run(replicates=5)
        service.run_for_segmentation(run)
        run.refresh_from_db()

        self.assertEqual(run.results["composition"]["tissue_px"], IMAGE_SIZE * IMAGE_SIZE)
        self.assertTrue(
            any("No tissue mask" in c for c in run.results["caveats"]),
            run.results["caveats"],
        )

    def test_uncalibrated_image_is_flagged_and_reports_no_microns(self):
        self.asset.pixel_size_nm = None
        self.asset.save(update_fields=["pixel_size_nm"])
        run = self._make_run(tissue_segmentation_id=str(self.tissue.id), replicates=5)

        service.run_for_segmentation(run)

        run.refresh_from_db()
        self.assertFalse(run.results["calibrated"])
        self.assertIsNone(run.results["composition"]["tissue_um2"])
        self.assertTrue(any("Pixel size" in c for c in run.results["caveats"]))

    def test_centroid_points_produce_enrichment_distances_and_a_circularity_caveat(self):
        run = self._make_run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="centroids",
            distance_target="mito",
            replicates=5,
        )

        service.run_for_segmentation(run)

        run.refresh_from_db()
        results = run.results
        self.assertEqual(results["points"]["n_total"], len(MITO_BOXES))
        self.assertEqual(results["points"]["n_off_tissue"], 0)
        self.assertEqual(results["distances"]["target"], "mito")
        self.assertEqual(results["monte_carlo"]["replicates"], 5)
        self.assertTrue(
            any("circular" in c for c in results["caveats"]),
            "measuring a compartment with its own centroids must be labelled",
        )

    def test_imported_csv_points_off_the_tissue_are_counted_not_dropped(self):
        run = self._make_run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="csv",
            points_csv="x,y\n100,100\n5,5\n195,195\n",
            replicates=5,
        )

        service.run_for_segmentation(run)

        run.refresh_from_db()
        points = run.results["points"]
        self.assertEqual(points["n_total"], 3)
        self.assertEqual(points["n_off_tissue"], 2)
        self.assertTrue(any("outside the tissue mask" in c for c in run.results["caveats"]))

    def test_an_empty_tissue_mask_succeeds_with_a_caveat(self):
        """A tissue segmentation with nothing confirmed in it.

        This is reachable from the API -- pass its id as
        ``tissue_segmentation_id`` -- and it used to reach
        ``sample_uniform_in_mask`` through ``self_check``, which always draws at
        least a few thousand points, and die with ``ValueError: cannot sample
        inside an empty mask``. A user is entitled to a run that comes back and
        tells them what is wrong with their input.
        """
        SegmentObject.objects.filter(segmentation=self.tissue).delete()
        run = self._make_run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="centroids",
            replicates=5,
        )

        service.run_for_segmentation(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_SUCCESS, run.error)
        self.assertEqual(run.error, "")
        results = run.results
        self.assertEqual(results["composition"]["tissue_px"], 0)

        empty = [c for c in results["caveats"] if "tissue mask is empty" in c]
        self.assertTrue(empty, results["caveats"])
        self.assertIn("no confirmed objects", empty[0])

        # The Monte-Carlo block is skipped whole -- not half-computed.
        self.assertNotIn("monte_carlo", results)
        self.assertNotIn("monte_carlo_self_check", results)

        # And the bundle is still written, because the composition and
        # morphometrics are still real measurements.
        bundle = Path(run.export_dir)
        for name in ("objects.csv", "image_summary.csv", "manifest.json"):
            self.assertTrue((bundle / name).is_file(), name)

    def test_a_requested_distance_analysis_says_when_it_is_skipped(self):
        """Uncalibrated image, distances requested: the job succeeds and the
        section is absent. Which analysis went missing has to be named."""
        self.asset.pixel_size_nm = None
        self.asset.save(update_fields=["pixel_size_nm"])
        run = self._make_run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="centroids",
            distance_target="mito",
            replicates=5,
        )

        service.run_for_segmentation(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_SUCCESS, run.error)
        self.assertNotIn("distances", run.results)
        named = [c for c in run.results["caveats"] if "Distance-to-mito" in c]
        self.assertTrue(named, run.results["caveats"])
        self.assertIn("skipped", named[0])
        self.assertIn("pixel size", named[0])

    def test_a_point_outside_the_image_does_not_fabricate_a_distance(self):
        """``1e30`` is finite, so it parses, and it used to cast to ``INT_MIN``
        and clip to pixel (0, 0) *only in the distance code*. The same array in
        ``assign_points`` was pinned to the far border. One run reported both,
        and the median distance to the nearest mitochondrion came out
        3.5e+30 nm -- 3.5e+21 metres -- in ``image_summary.csv``.
        """
        run = self._make_run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="csv",
            points_csv="x,y\n40,40\n1e30,1e30\n",
            distance_target="mito",
            replicates=5,
        )

        service.run_for_segmentation(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_SUCCESS, run.error)
        distances = run.results["distances"]
        # Nothing in a 200 px image at 5 nm/px is more than ~1,414 nm from
        # anything else in it.
        ceiling = IMAGE_SIZE * TEST_PIXEL_SIZE_NM * 1.5
        self.assertLess(distances["median_nm"], ceiling)
        self.assertEqual(distances["n_measured"], 2)
        self.assertEqual(sum(distances["band_counts"]), 2)

        # The two sections agree about where that point is, and about how many
        # points they are describing.
        points = run.results["points"]
        self.assertEqual(distances["n_out_of_image"], points["n_out_of_bounds"])
        self.assertEqual(distances["n_unreadable"], points["n_unreadable"])

        # And a caveat names the distance analysis, which none of them did.
        named = [c for c in run.results["caveats"] if "measured against mito" in c]
        self.assertTrue(named, run.results["caveats"])
        self.assertIn("border pixel", named[0])

        row = service.image_summary_row(run.results)
        self.assertLess(row["distance_median_nm"], ceiling)

    def test_a_result_that_cannot_be_stored_never_reaches_the_disk(self):
        """``nan`` is not JSON and ``AnalysisRun.results`` is checked with
        ``JSON_VALID``. The row used to fail on that constraint *after* the
        bundle was written, so the user got a FAILED run with an empty
        export_dir, three files on disk no row could name, and
        "CHECK constraint failed" as the explanation.
        """
        run = self._make_run(tissue_segmentation_id=str(self.tissue.id), replicates=5)
        real = service.run_analysis

        def poisoned(inputs):
            result = real(inputs)
            result["objects"]["density"]["per_um2"] = float("nan")
            return result

        with mock.patch.object(service, "run_analysis", poisoned):
            with self.assertRaises(ValueError) as caught:
                service.run_for_segmentation(run)

        self.assertIn("objects.density.per_um2 = nan", str(caught.exception))
        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_FAILED)
        self.assertIn("not a number", run.error)
        self.assertEqual(run.export_dir, "")
        self.assertFalse(
            service.export_dir_for_run(run.id).exists(),
            "a run that stored nothing must not leave a bundle behind",
        )

    def test_a_bundle_written_by_a_run_that_then_failed_is_not_left_behind(self):
        run = self._make_run(tissue_segmentation_id=str(self.tissue.id), replicates=5)
        real = service.write_bundle

        def write_then_fail(results, out_dir, **kwargs):
            real(results, out_dir, **kwargs)
            raise RuntimeError("the disk went away")

        with mock.patch.object(service, "write_bundle", write_then_fail):
            with self.assertRaises(RuntimeError):
                service.run_for_segmentation(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_FAILED)
        self.assertEqual(run.export_dir, "")
        self.assertFalse(service.export_dir_for_run(run.id).exists())
        self.assertIn("the disk went away", run.error)
        self.assertIn("removed", run.error)

    def test_a_flat_monte_carlo_null_reports_no_p_and_says_why(self):
        """A null whose twenty replicates all return the same number.

        Here the tissue mask *is* the mitochondrion, so every simulated draw
        lands in it and every replicate returns 1.0 -- the flat null this run
        can produce deterministically, rather than one that depends on whether
        twenty random singletons happened to miss a small compartment.
        ``compartments.assign_points`` names this case in its own docstring:
        "a Monte-Carlo null of twenty identical zeros that reported p = 1.0 as
        a statistic". ``z`` was already blank for it. The *p* was not, and the
        empirical p against a flat null is 1 / (R + 1) or 1.0 by construction,
        whatever the data are.
        """
        SegmentObject.objects.filter(
            segmentation=self.segmentation, label_state="CONFIRMED"
        ).exclude(id=self.objects[1].id).delete()
        SegmentObject.objects.filter(segmentation=self.tissue).delete()
        self._segment(self.tissue, _square(*MITO_BOXES[1]), "CONFIRMED")

        run = self._make_run(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="csv",
            points_csv="x,y\n100,50\n",
        )

        service.run_for_segmentation(run)

        run.refresh_from_db()
        mc = run.results["monte_carlo"]
        self.assertEqual(mc["replicates"], 20)
        self.assertEqual(mc["null_sd"]["enrichment_mito"], 0.0)
        self.assertIsNone(mc["z"]["enrichment_mito"])
        self.assertIsNone(mc["p_two_sided"]["enrichment_mito"])

        named = [c for c in run.results["caveats"] if "no spread" in c]
        self.assertTrue(named, run.results["caveats"])
        # The floor twenty replicates cannot go below, named in the caveat
        # because it is what the number would have been.
        self.assertIn("0.04762", named[0])
        self.assertIn("enrichment_mito", named[0])

        # And the blank reaches the exported table, not only the screen.
        row = service.image_summary_row(run.results)
        self.assertIsNone(row["z_enrichment_mito"])

    def test_the_bundle_is_readable_when_nothing_was_confirmed(self):
        """``objects.csv`` used to be a zero-byte file, which
        ``pandas.read_csv`` refuses with ``EmptyDataError``."""
        SegmentObject.objects.filter(segmentation=self.segmentation).update(label_state="CANDIDATE")
        run = self._make_run(tissue_segmentation_id=str(self.tissue.id), replicates=5)

        service.run_for_segmentation(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_SUCCESS, run.error)
        self.assertEqual(run.results["objects"]["n"], 0)

        bundle = Path(run.export_dir)
        for name in ("objects.csv", "image_summary.csv"):
            self.assertGreater((bundle / name).stat().st_size, 0, f"{name} is zero-byte")
        rows = list(csv.DictReader((bundle / "objects.csv").open(encoding="utf-8-sig")))
        self.assertEqual(rows, [])
        reader = csv.reader((bundle / "objects.csv").open(encoding="utf-8-sig"))
        self.assertIn("area_um2", next(reader))

    def test_failure_is_recorded_on_the_run_before_it_propagates(self):
        run = self._make_run(replicates=5)
        self.asset.logical_width = None
        self.asset.save(update_fields=["logical_width"])

        with self.assertRaises(loaders.AnalysisInputError):
            service.run_for_segmentation(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_FAILED)
        self.assertIn("no recorded size", run.error)
        self.assertIsNotNone(run.finished_at)

    def test_export_directory_is_per_run_under_the_data_directory(self):
        run = self._make_run(replicates=5)
        service.run_for_segmentation(run)
        run.refresh_from_db()
        self.assertEqual(Path(run.export_dir).name, str(run.id))
        self.assertEqual(Path(run.export_dir).parent, self.exports_root)


class JobTests(AnalysisRunTestCase):
    def _job(self) -> Job:
        return Job.enqueue(job_type=JOB_TYPE_RUN_ANALYSIS, payload={})

    def test_run_job_executes_the_run_and_reports_progress(self):
        run = self._make_run(tissue_segmentation_id=str(self.tissue.id), replicates=5)
        job = self._job()

        result = run_job(
            {"analysis_run_id": str(run.id)},
            JobReporter(str(job.id), min_interval_seconds=0.0),
            CancelToken(str(job.id)),
        )

        run.refresh_from_db()
        job.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_SUCCESS)
        self.assertEqual(result["analysis_run_id"], str(run.id))
        self.assertEqual(result["n_objects"], len(MITO_BOXES))
        self.assertEqual(result["export_dir"], run.export_dir)
        self.assertEqual(job.progress, 100.0)

    def test_run_job_requires_an_analysis_run_id(self):
        job = self._job()
        with self.assertRaises(ValueError):
            run_job({}, JobReporter(str(job.id)), CancelToken(str(job.id)))

    def test_a_cancelled_job_never_starts(self):
        run = self._make_run(replicates=5)
        job = self._job()
        job.cancel_requested = True
        job.save(update_fields=["cancel_requested"])

        with self.assertRaises(JobCancelledError):
            run_job(
                {"analysis_run_id": str(run.id)},
                JobReporter(str(job.id)),
                CancelToken(str(job.id)),
            )

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_PENDING)

    def test_cancelling_mid_run_is_recorded_in_the_user_s_words(self):
        run = self._make_run(replicates=5)
        job = self._job()
        real_run_analysis = service.run_analysis

        def cancel_after_measuring(inputs):
            out = real_run_analysis(inputs)
            Job.objects.filter(id=job.id).update(cancel_requested=True)
            return out

        with mock.patch.object(service, "run_analysis", cancel_after_measuring):
            with self.assertRaises(JobCancelledError):
                run_job(
                    {"analysis_run_id": str(run.id)},
                    JobReporter(str(job.id), min_interval_seconds=0.0),
                    CancelToken(str(job.id)),
                )

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_FAILED)
        self.assertEqual(run.error, "Cancelled before the analysis finished.")


@override_settings(ROOT_URLCONF="quantem.analysis.tests.urls")
class AnalysisApiTests(AnalysisRunTestCase):
    def _start(self, **body):
        return self.client.post(
            f"/api/segmentations/{self.segmentation.id}/analysis/", body, format="json"
        )

    def _completed_run(self) -> AnalysisRun:
        response = self._start(
            tissue_segmentation_id=str(self.tissue.id), replicates=5, group="fed"
        )
        self.assertEqual(response.status_code, 202, response.data)
        run = AnalysisRun.objects.get(id=response.data["analysis_run_id"])
        service.run_for_segmentation(run)
        run.refresh_from_db()
        return run

    def test_a_point_set_in_the_wrong_units_is_served_honestly_end_to_end(self):
        """POST, run the queued job, and read the run back the way the Analysis
        screen does.

        The API is where this reached a user: a CSV in nanometres parses, every
        coordinate is finite, and the distance section came back with a median
        of 3.5e+30 nm and no caveat mentioning distances at all.
        """
        response = self._start(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="csv",
            points_csv="x,y\n40,40\n1e30,1e30\n",
            distance_target="mito",
            replicates=5,
        )
        self.assertEqual(response.status_code, 202, response.data)
        run_id = response.data["analysis_run_id"]
        job = Job.objects.get(id=response.data["job_id"])

        run_job(
            job.payload_json,
            JobReporter(str(job.id), min_interval_seconds=0.0),
            CancelToken(str(job.id)),
        )

        served = self.client.get(f"/api/analysis/{run_id}/")
        self.assertEqual(served.status_code, 200, served.data)
        self.assertEqual(served.data["status"], "SUCCESS")
        distances = served.data["distances"]
        self.assertLess(distances["median_nm"], IMAGE_SIZE * TEST_PIXEL_SIZE_NM * 1.5)
        self.assertEqual(distances["n_out_of_image"], 1)
        self.assertEqual(sum(distances["band_counts"]), distances["n_measured"])
        self.assertTrue(
            any("measured against mito" in c for c in served.data["caveats"]),
            served.data["caveats"],
        )

    def test_post_queues_a_job_and_returns_both_ids(self):
        response = self._start(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="centroids",
            distance_target="mito",
            group="fasted",
            replicates=5,
        )

        self.assertEqual(response.status_code, 202, response.data)
        run = AnalysisRun.objects.get(id=response.data["analysis_run_id"])
        self.assertEqual(run.status, AnalysisRun.STATUS_PENDING)
        self.assertEqual(run.group, "fasted")
        self.assertEqual(run.params["replicates"], 5)

        job = Job.objects.get(id=response.data["job_id"])
        self.assertEqual(job.type, JOB_TYPE_RUN_ANALYSIS)
        self.assertEqual(job.payload_json["analysis_run_id"], str(run.id))
        # The queue screen resolves the image and organelle from these.
        self.assertEqual(job.payload_json["segmentation_id"], str(self.segmentation.id))
        self.assertEqual(job.payload_json["asset_id"], str(self.asset.id))

    def test_post_rejects_a_bad_request_with_one_sentence(self):
        response = self._start(distance_target="golgi", points_source="centroids")
        self.assertEqual(response.status_code, 400, response.data)
        self.assertIn("error", response.data)
        self.assertIn("golgi", response.data["error"])

    def test_a_non_finite_band_edge_is_a_bad_request_not_a_database_error(self):
        """The user's own route in, not just the validator.

        ``"nan"`` and ``"inf"`` pass DRF's ``FloatField`` (it calls ``float()``
        and catches only TypeError/ValueError), passed the monotonic check, and
        went into ``AnalysisRun.objects.create(params=...)``, which raised
        ``IntegrityError: CHECK constraint failed: JSON_VALID("params")`` -- an
        HTTP 500 quoting a database constraint at someone who typed a bad band
        edge. The frontend refuses these already; the documented API did not.
        """
        for edges in ([0, "nan"], [0, "inf"], ["-inf", 0]):
            with self.subTest(edges=edges):
                response = self._start(band_edges_nm=edges, replicates=5)
                self.assertEqual(response.status_code, 400, response.data)
                self.assertIn("finite", response.data["error"])
        self.assertFalse(AnalysisRun.objects.exists())

    def test_a_json_number_that_overflows_to_infinity_is_refused_too(self):
        """No string and no ``Infinity`` literal: ``1e999`` is a plain JSON
        number that ``json.loads`` turns into ``inf`` before any field sees it,
        so DRF's strict-JSON guard never fires on it."""
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/analysis/",
            '{"band_edges_nm": [0, 1e999], "replicates": 5}',
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400, response.data)
        self.assertIn("finite", response.data["error"])
        self.assertFalse(AnalysisRun.objects.exists())

    def test_detail_returns_the_flat_contract_shape(self):
        run = self._completed_run()

        response = self.client.get(f"/api/analysis/{run.id}/")

        self.assertEqual(response.status_code, 200)
        body = response.data
        self.assertEqual(body["id"], str(run.id))
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["group"], "fed")
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["pixel_size_nm"], TEST_PIXEL_SIZE_NM)
        self.assertEqual(body["objects"]["n"], len(MITO_BOXES))
        self.assertIn("area_fractions", body["composition"])
        self.assertEqual(body["export_dir"], run.export_dir)
        self.assertEqual(
            sorted(body["exports"]),
            ["image_summary.csv", "manifest.json", "objects.csv"],
        )
        # Sections that were not computed are null, not missing.
        self.assertIsNone(body["points"])
        self.assertIsNone(body["distances"])

    def test_list_is_newest_first(self):
        first = self._start(replicates=5).data["analysis_run_id"]
        second = self._start(replicates=5).data["analysis_run_id"]

        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/analysis/")

        self.assertEqual(response.status_code, 200)
        ids = [row["id"] for row in response.data]
        self.assertEqual(ids, [second, first])

    def test_export_downloads_as_an_attachment(self):
        run = self._completed_run()

        response = self.client.get(f"/api/analysis/{run.id}/export/objects.csv")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("attachment", response["Content-Disposition"])
        self.assertIn("objects.csv", response["Content-Disposition"])
        body = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("area_um2", body)

    def test_export_serves_the_manifest_as_json(self):
        run = self._completed_run()
        response = self.client.get(f"/api/analysis/{run.id}/export/manifest.json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        payload = json.loads(b"".join(response.streaming_content).decode("utf-8"))
        self.assertEqual(payload["n_images"], 1)

    def test_export_refuses_a_path_outside_the_run_directory(self):
        """The URL route forbids "/", but the view must not rely on that.

        Called directly with a traversing name, so the guard is tested rather
        than the router: a future route change must not silently open the
        filesystem.
        """
        run = self._completed_run()
        # A sibling run's bundle, reachable only by escaping this run's directory.
        secret = self.exports_root / "somewhere-else"
        secret.mkdir(parents=True, exist_ok=True)
        (secret / "objects.csv").write_text("not yours", encoding="utf-8")

        view = AnalysisRunExportView.as_view()
        for name in ("../somewhere-else/objects.csv", "..", "../"):
            request = APIRequestFactory().get("/")
            response = view(request, run_id=str(run.id), name=name)
            self.assertEqual(response.status_code, 400, name)
            self.assertIn("not part of this analysis", response.data["error"])

    def test_export_refuses_an_absolute_path(self):
        run = self._completed_run()
        outside = self.exports_root.parent / "not-an-export.csv"
        outside.write_text("not yours", encoding="utf-8")
        self.addCleanup(outside.unlink, True)

        request = APIRequestFactory().get("/")
        response = AnalysisRunExportView.as_view()(request, run_id=str(run.id), name=str(outside))

        self.assertEqual(response.status_code, 400, response.data)

    def test_export_of_an_unknown_file_is_a_404(self):
        run = self._completed_run()
        response = self.client.get(f"/api/analysis/{run.id}/export/nope.csv")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.data)

    def test_export_before_the_run_finished_is_a_404(self):
        response = self._start(replicates=5)
        run_id = response.data["analysis_run_id"]
        response = self.client.get(f"/api/analysis/{run_id}/export/objects.csv")
        self.assertEqual(response.status_code, 404)
        self.assertIn("has not written an export bundle", response.data["error"])

    def test_an_empty_tissue_mask_is_accepted_and_caveated_end_to_end(self):
        """The crash was reachable from the API with a valid request: nothing
        rejects a tissue segmentation that happens to have nothing confirmed."""
        SegmentObject.objects.filter(segmentation=self.tissue).delete()

        response = self._start(
            tissue_segmentation_id=str(self.tissue.id),
            points_source="centroids",
            replicates=5,
        )
        self.assertEqual(response.status_code, 202, response.data)
        run = AnalysisRun.objects.get(id=response.data["analysis_run_id"])
        service.run_for_segmentation(run)

        detail = self.client.get(f"/api/analysis/{run.id}/")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.data["status"], "SUCCESS")
        self.assertIsNone(detail.data["monte_carlo"])
        self.assertIsNone(detail.data["monte_carlo_self_check"])
        self.assertTrue(
            any("tissue mask is empty" in c for c in detail.data["caveats"]),
            detail.data["caveats"],
        )


@override_settings(ROOT_URLCONF="quantem.analysis.tests.urls")
class GroupRollupApiTests(AnalysisRunTestCase):
    """``GET /api/analysis/groups/``.

    ``rollup.py`` held the one rule in the suite that is easiest to get wrong
    and had no production caller: ``AnalysisRun.group`` was stored, written to
    the manifest, and never aggregated.
    """

    def _run(self, segmentation, *, group: str, **params) -> AnalysisRun:
        normalised = loaders.normalise_params(
            {"group": group, "replicates": 5, **params}, segmentation=segmentation
        )
        run = AnalysisRun.objects.create(segmentation=segmentation, params=normalised, group=group)
        service.run_for_segmentation(run)
        run.refresh_from_db()
        return run

    def _second_unit(self) -> ImageSegmentation:
        """A second image with two confirmed mitochondria instead of three."""
        other = create_small_test_image("Second Unit", width=IMAGE_SIZE, height=IMAGE_SIZE)
        segmentation = ImageSegmentation.objects.create(
            asset=other.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        for x, y in MITO_BOXES[:2]:
            self._segment(segmentation, _square(x, y), "CONFIRMED")
        return segmentation

    def _groups(self, **query) -> dict:
        response = self.client.get("/api/analysis/groups/", query)
        self.assertEqual(response.status_code, 200, response.data)
        return response.data

    def test_group_mean_is_unweighted_over_units_and_says_so(self):
        self._run(self.segmentation, group="fasted")  # 3 objects
        self._run(self._second_unit(), group="fasted")  # 2 objects

        body = self._groups()

        self.assertIn("Unweighted mean", body["aggregation_rule"])
        self.assertIn("never weighted by point count", body["aggregation_rule"].lower())
        self.assertEqual(body["scope"]["n_runs"], 2)

        (group,) = [g for g in body["groups"] if g["group"] == "fasted"]
        self.assertEqual(group["n_units"], 2)
        self.assertIn("analysis run", group["unit"])
        n_objects = group["metrics"]["n_objects"]
        self.assertEqual(n_objects["n_units"], 2)
        self.assertAlmostEqual(n_objects["mean"], 2.5)
        self.assertAlmostEqual(n_objects["sd"], 0.7071067, places=5)
        self.assertAlmostEqual(n_objects["sem"], 0.5)
        self.assertEqual(sorted(n_objects["values"]), [2.0, 3.0])
        # Two genuinely different images: nothing to warn about.
        self.assertEqual(group["warnings"], [])
        self.assertEqual(len(group["image_keys"]), 2)

        # An acquisition setting is not a measurement: the distinct values are
        # reported, never a "mean pixel size".
        self.assertEqual(group["pixel_sizes_nm"], [TEST_PIXEL_SIZE_NM])
        self.assertNotIn("pixel_size_nm", body["metrics"])

    def test_repeated_runs_of_one_image_are_not_two_experimental_units(self):
        """Pseudo-replication is the exact failure the unweighted rule exists to
        prevent; averaging one image twice must not read as n = 2."""
        self._run(self.segmentation, group="fasted")
        self._run(self.segmentation, group="fasted")

        (group,) = self._groups()["groups"]

        self.assertEqual(group["n_units"], 2)
        self.assertEqual(len(group["image_keys"]), 1)
        self.assertTrue(group["warnings"], "repeated images must be flagged")
        self.assertIn("not independent experimental units", group["warnings"][0])

    def test_the_segmentation_filter_scopes_the_rollup(self):
        self._run(self.segmentation, group="fasted")
        other = self._second_unit()
        self._run(other, group="fasted")

        scoped = self._groups(segmentation=str(other.id))

        self.assertEqual(scoped["scope"]["segmentation"], str(other.id))
        self.assertEqual(scoped["scope"]["n_runs"], 1)
        (group,) = scoped["groups"]
        self.assertEqual(group["n_units"], 1)
        # One unit has no spread, and the payload must not invent one.
        self.assertIsNone(group["metrics"]["n_objects"]["sem"])

    def test_only_successful_runs_are_counted(self):
        """A pending or failed run has no numbers; counting it as a unit would
        deflate every mean it is silently included in."""
        self._run(self.segmentation, group="fasted")
        AnalysisRun.objects.create(
            segmentation=self.segmentation,
            params=loaders.normalise_params({}, segmentation=self.segmentation),
            group="fasted",
            status=AnalysisRun.STATUS_FAILED,
        )

        body = self._groups()

        self.assertEqual(body["scope"]["n_runs"], 1)
        self.assertEqual(body["groups"][0]["n_units"], 1)

    def test_ungrouped_runs_are_not_presented_as_a_group(self):
        self._run(self.segmentation, group="")

        (group,) = self._groups()["groups"]

        self.assertEqual(group["group"], "")
        self.assertTrue(any("no group label" in w for w in group["warnings"]), group["warnings"])

    def test_a_circular_metric_is_named_as_such_in_the_rollup(self):
        """The rollup is the endpoint whose output goes in a figure, and it was
        publishing ``enrichment_mito`` with a mean, an SD and an SEM while every
        run behind it said that column is 1 / area fraction by construction and
        must not be reported as a result."""
        self._run(self.segmentation, group="fasted", points_source="centroids")

        body = self._groups()
        (group,) = body["groups"]

        self.assertIn("enrichment_mito", group["metrics"])
        self.assertIn("enrichment_mito", body["circular_metrics"])
        self.assertIn("z_enrichment_mito", group["circular_metrics"])
        self.assertTrue(group["metrics"]["enrichment_mito"]["circular"])
        self.assertIn("must not be reported", group["metrics"]["enrichment_mito"]["note"])
        # A real measurement beside it is not flagged, or the flag means nothing.
        self.assertFalse(group["metrics"]["area_fraction_mito"]["circular"])
        self.assertNotIn("area_fraction_mito", body["circular_metrics"])
        self.assertTrue(any("circular" in w for w in group["warnings"]), group["warnings"])

    def test_the_runs_caveats_travel_with_the_group_mean(self):
        run = self._run(self.segmentation, group="fasted", points_source="centroids")

        (group,) = self._groups()["groups"]

        texts = [entry["text"] for entry in group["caveats"]]
        self.assertTrue(texts, "a group mean carries the caveats of its runs")
        for caveat in run.results["caveats"]:
            self.assertIn(caveat, texts)
        circular = next(entry for entry in group["caveats"] if "circular" in entry["text"])
        self.assertEqual(circular["run_ids"], [str(run.id)])
        self.assertEqual(circular["n_runs"], 1)

    def test_a_caveat_shared_by_two_runs_is_listed_once_and_names_both(self):
        first = self._run(self.segmentation, group="fasted")
        second = self._run(self._second_unit(), group="fasted")

        (group,) = self._groups()["groups"]

        shared = [entry for entry in group["caveats"] if entry["n_runs"] == 2]
        self.assertTrue(shared, group["caveats"])
        self.assertEqual(sorted(shared[0]["run_ids"]), sorted([str(first.id), str(second.id)]))
        texts = [entry["text"] for entry in group["caveats"]]
        self.assertEqual(len(texts), len(set(texts)), "each sentence appears once")

    def test_a_group_that_mixes_pixel_sizes_says_so_in_words(self):
        """Two distinct entries in ``pixel_sizes_nm`` is data, not a statement,
        and every micron metric below is averaged across both."""
        self._run(self.segmentation, group="fasted")
        other = self._second_unit()
        other.asset.pixel_size_nm = TEST_PIXEL_SIZE_NM * 4
        other.asset.save(update_fields=["pixel_size_nm"])
        self._run(other, group="fasted")

        (group,) = self._groups()["groups"]

        self.assertEqual(len(group["pixel_sizes_nm"]), 2)
        named = [w for w in group["warnings"] if "pixel sizes" in w]
        self.assertTrue(named, group["warnings"])
        self.assertIn("microns", named[0])

    def test_a_malformed_segmentation_filter_is_a_bad_request(self):
        response = self.client.get("/api/analysis/groups/", {"segmentation": "nope"})
        self.assertEqual(response.status_code, 400, response.data)
        self.assertIn("error", response.data)

    def test_no_runs_is_an_empty_rollup_not_an_error(self):
        body = self._groups()
        self.assertEqual(body["groups"], [])
        self.assertEqual(body["scope"]["n_runs"], 0)

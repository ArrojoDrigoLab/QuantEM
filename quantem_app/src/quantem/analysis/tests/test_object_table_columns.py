"""What ``objects.csv`` is allowed to call a column, and to offer twice.

Three faults found by reading one real bundle, all of them invisible to anyone
who trusted the manifest's column list:

* **``aspect_ratio`` and ``elongation`` were the same number.** The extractor
  measures ``elongation`` as ``major / max(minor, 1)``;
  :func:`~quantem.analysis.morphometrics.derive` recomputed ``major / minor``
  from the same two axes and exported it as ``aspect_ratio``. String-identical
  in all 38 rows. The two obvious things to do with this file are a correlation
  matrix and a PCA, and both silently double-weight an axis that appears twice.

* **Two ``pixel_size_nm`` columns disagreed inside one bundle.**
  ``image_summary.csv`` said ``5.0``; ``objects.csv`` left its column empty on
  every row of the same run. Both readings are right -- "what the image
  records" against "what these values are in" -- and the manifest, which tells
  the reader to join the two files on ``image_key``, explained neither. The join
  hands back ``NaN`` where 5.0 was expected.

* **``pixel_size_nm`` was summarised as a distribution.** One constant repeated
  per row got a mean, an sd, a median and an IQR in a table of morphometrics.

The rule these pin: the measurement layer stores what it measured, and the
export layer decides what the table may call it and what it must not offer
twice -- in the manifest, where a reader looks, not only in the code.
"""

from __future__ import annotations

import csv

from quantem.analysis import service
from quantem.analysis.morphometrics import derive
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)

from .test_calibrated_after_the_fact import _uncalibrated_stamp
from .test_run_identity import RunIdentityTestCase, _square, _stamp

#: A model-extracted object, in the extractor's own vocabulary.
MODEL_FEATURES = {
    "area": 400.0,
    "perimeter": 80.0,
    "eccentricity": 0.4,
    "solidity": 0.97,
    "elongation": 24.0 / 21.0,
    "major_axis_length": 24.0,
    "minor_axis_length": 21.0,
    "feret_diameter_max": 26.0,
    "intensity_mean": 118.0,
    "intensity_p10": 90.0,
    "intensity_p50": 119.0,
    "intensity_p90": 145.0,
    "mean_prob": 0.82,
}


class ObjectTableTestCase(RunIdentityTestCase):
    """Mitochondria on a 5 nm/px image, measured against a tissue mask."""

    def _mito(self, *, uncalibrated: bool = False, n: int = 3):
        mito = self._segmentation(get_or_create_mitochondria_type)
        stamp = (
            _uncalibrated_stamp(pack_id="quantem:mito")
            if uncalibrated
            else _stamp(pack_id="quantem:mito", ran_at_nm=8.0)
        )
        for i in range(n):
            obj = self._object(
                mito,
                _square(20 + 30 * i, 20),
                source_model="quantem:mito",
                stamp=dict(stamp),
            )
            obj.features = {**MODEL_FEATURES, "run": dict(stamp)}
            obj.save(update_fields=["features"])
        return mito

    def _tissue(self):
        tissue = self._segmentation(get_or_create_tissue_type)
        self._object(tissue, _square(5, 5, 200), source_model="manual")
        return tissue

    def _bundle(self, *, uncalibrated: bool = False):
        mito = self._mito(uncalibrated=uncalibrated)
        run, got = self._run(
            mito,
            compartments={"mito": str(mito.id)},
            tissue_segmentation_id=str(self._tissue().id),
        )
        out = service.export_dir_for_run(run.id)
        return {
            "result": got["result"],
            "manifest": got["manifest"],
            "objects": list(
                csv.DictReader((out / "objects.csv").open(encoding="utf-8-sig"))
            ),
            "images": list(
                csv.DictReader((out / "image_summary.csv").open(encoding="utf-8-sig"))
            ),
        }

    @staticmethod
    def _file_entry(manifest: dict, name: str) -> dict:
        for entry in manifest["outputs"]["files"]:
            if entry["filename"] == name:
                return entry
        raise AssertionError(
            f"{name} is not in the outputs manifest: "
            f"{[e['filename'] for e in manifest['outputs']['files']]}"
        )


class OneRatioNotTwoTests(ObjectTableTestCase):
    def test_the_derived_ratio_really_was_the_stored_one(self):
        """The premise of the removal, checked against the measurement layer.

        If ``derive`` ever makes these two genuinely different quantities, the
        export has to carry both again -- and this is the test that says so.
        """
        values = derive(MODEL_FEATURES, object_id="o1", pixel_size_nm=8.0).values

        self.assertAlmostEqual(values["aspect_ratio"], values["elongation"], places=12)

    def test_objects_csv_offers_the_ratio_once(self):
        bundle = self._bundle()
        columns = bundle["objects"][0].keys()

        self.assertIn("elongation", columns)
        self.assertNotIn(
            "aspect_ratio",
            columns,
            "two columns of one number double-weight that axis in a PCA",
        )

    def test_the_summary_offers_it_once_too(self):
        summary = self._bundle()["result"]["objects"]["summary"]

        self.assertIn("elongation", summary)
        self.assertNotIn("aspect_ratio", summary)

    def test_the_manifest_says_where_the_column_went_and_why(self):
        entry = self._file_entry(self._bundle()["manifest"], "objects.csv")

        self.assertNotIn("aspect_ratio", entry["columns"])
        reason = entry["columns_not_written"]["aspect_ratio"]
        self.assertIn("elongation", reason)
        self.assertIn("PCA", reason)

    def test_the_manifest_defines_the_surviving_ratio(self):
        """"Which is which" cannot be answered by the name on its own."""
        entry = self._file_entry(self._bundle()["manifest"], "objects.csv")

        self.assertTrue(
            entry["column_notes"]["elongation"].startswith(
                "major_axis_px / max(minor_axis_px, 1)"
            ),
            entry["column_notes"]["elongation"],
        )

    def test_the_manifest_names_the_columns_that_are_not_independent(self):
        entry = self._file_entry(self._bundle()["manifest"], "objects.csv")
        derived = entry["columns_derived_from"]

        self.assertEqual(derived["equivalent_diameter_px"], "area_px alone (a monotone transform of it)")
        self.assertIn("area_px", derived["area_um2"])
        self.assertIn("correlation matrix", entry["columns_are_not_independent"])


class TwoPixelSizeColumnsTests(ObjectTableTestCase):
    """One name meaning two things across two files people are told to join."""

    def test_the_only_columns_in_both_files_mean_the_same_thing_in_both(self):
        """A shared name is a promise that a join can rely on it."""
        bundle = self._bundle()
        shared = set(bundle["objects"][0]) & set(bundle["images"][0])

        self.assertEqual(
            shared,
            {
                # identifiers, and `calibrated`, which asks the same question of
                # a row of objects.csv as it does of a row of image_summary.csv.
                "image_key",
                "group",
                "segmentation_id",
                "analysis_run_id",
                "n_caveats",
                "calibrated",
            },
            "pixel_size_nm was in both, meaning something different in each",
        )
        self.assertNotIn("pixel_size_nm", bundle["objects"][0])

    def test_a_calibrated_run_agrees_across_the_join(self):
        bundle = self._bundle()
        image = bundle["images"][0]

        self.assertEqual(image["pixel_size_nm"], "5.0")
        self.assertEqual(image["calibrated"], "True")
        for row in bundle["objects"]:
            self.assertEqual(row["values_in_pixel_size_nm"], "5.0")

    def test_the_disagreement_is_the_expected_reading_and_is_explained(self):
        """The reported case: image_summary 5.0, objects.csv blank on every row.

        Neither file is wrong. What was missing is anything saying they answer
        different questions, in the manifest that tells the reader to join them.
        """
        bundle = self._bundle(uncalibrated=True)
        image = bundle["images"][0]

        self.assertEqual(image["pixel_size_nm"], "5.0")
        self.assertEqual(image["calibrated"], "False")
        for row in bundle["objects"]:
            self.assertEqual(row["values_in_pixel_size_nm"], "")

        objects_note = self._file_entry(bundle["manifest"], "objects.csv")[
            "column_notes"
        ]["values_in_pixel_size_nm"]
        self.assertIn("image_summary.csv", objects_note)
        self.assertIn("calibrated column is true", objects_note)

        images_note = self._file_entry(bundle["manifest"], "image_summary.csv")[
            "column_notes"
        ]["pixel_size_nm"]
        self.assertIn("values_in_pixel_size_nm", images_note)

        self.assertIn("image_key", bundle["manifest"]["outputs"]["joining"])
        self.assertIn(
            "values_in_pixel_size_nm", bundle["manifest"]["outputs"]["joining"]
        )


class TheScaleIsNotADistributionTests(ObjectTableTestCase):
    def test_the_summary_does_not_give_the_pixel_size_a_mean_and_a_spread(self):
        objects = self._bundle()["result"]["objects"]

        self.assertNotIn("pixel_size_nm", objects["summary"])
        self.assertIn("area_um2", objects["summary"], "the real metrics are still here")

    def test_the_constant_is_still_reported_as_a_constant(self):
        objects = self._bundle()["result"]["objects"]

        self.assertEqual(objects["values_in_pixel_size_nm"], 5.0)

    def test_an_uncalibrated_run_says_the_values_are_in_no_physical_scale(self):
        objects = self._bundle(uncalibrated=True)["result"]["objects"]

        self.assertIsNone(objects["values_in_pixel_size_nm"])
        self.assertNotIn("pixel_size_nm", objects["summary"])

    def test_the_metric_count_in_the_caveats_counts_metrics(self):
        """``N of M metrics are measured on fewer than ...`` had a constant in M."""
        result = self._bundle()["result"]
        summary = result["objects"]["summary"]

        self.assertTrue(
            all(key not in summary for key in ("pixel_size_nm", "aspect_ratio"))
        )
        for caveat in result["caveats"]:
            if "metrics" in caveat and "measured on fewer than" in caveat:
                self.assertIn(f"of {len(summary)} metrics", caveat)

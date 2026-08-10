""""Set the image's pixel size and re-run inference" was advice that could not work.

A user read that caveat, set the pixel size, and re-ran full segmentation over a
proofread image. The run completed SUCCESS and returned::

    {"segment_count": 0, "found_objects": false,
     "next_steps": ["Nothing changed: the 41 object(s) you have already labelled
                     here are exactly as they were.", ...]}

Every part of that is correct.
:func:`quantem.seg_core.db.extraction.extract_and_save_segments` drops a
candidate overlapping a CONFIRMED object by >=30% or an EXCLUDED one by >=80%,
which is what stops a re-run destroying a day of proofreading. The consequence
is that **once an image has been proofread, "re-run inference" can never lift
the uncalibrated stamp**: those objects keep ``native_pixel_size_nm: null`` for
good and every future bundle repeats the same caveat. The user followed the
instruction, got a green success, and was no further forward.

The route that does work -- discard the objects produced without a pixel size,
then re-run -- was never named anywhere. It is named here, and the fact that no
screen offers it is stated rather than implied.

Also pinned: the third "calibrated after the fact" leak, which is the same
mistake in a different sentence. ``pixel_size_provenance`` told a reader that
"Every micron column in this bundle, and the scale any model resampled to,
rests on it" beside a bundle whose micron columns are all blank and whose models
resampled to nothing.
"""

from __future__ import annotations

from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_mitochondria_type,
)

from .test_calibrated_after_the_fact import _uncalibrated_stamp
from .test_run_identity import RunIdentityTestCase, _square, _stamp


class RerunAdviceTestCase(RunIdentityTestCase):
    """The asset is 5 nm/px throughout; what varies is when the objects were made."""

    def _mito(self, *, uncalibrated: bool):
        mito = self._segmentation(get_or_create_mitochondria_type)
        stamp = (
            _uncalibrated_stamp(pack_id="quantem:mito")
            if uncalibrated
            else _stamp(pack_id="quantem:mito", ran_at_nm=8.0)
        )
        for i in range(3):
            self._object(
                mito,
                _square(20 + 30 * i, 20),
                source_model="quantem:mito",
                stamp=stamp,
            )
        return mito

    def _caveats(self, *, uncalibrated: bool = True) -> str:
        mito = self._mito(uncalibrated=uncalibrated)
        _run, got = self._run(mito, compartments={"mito": str(mito.id)})
        self._last = got
        return " ".join(got["result"]["caveats"])


class TheAdviceIsTrueTests(RerunAdviceTestCase):
    def test_it_says_a_re_run_over_confirmed_objects_changes_nothing(self):
        caveats = self._caveats()

        self.assertIn("Re-running inference is not by itself enough", caveats)
        self.assertIn("already confirmed or excluded is dropped", caveats)
        self.assertIn("reports no new objects", caveats)

    def test_it_names_the_route_that_does_work(self):
        caveats = self._caveats()

        self.assertIn("/labels/clear", caveats)
        self.assertIn("No screen offers that yet", caveats)
        self.assertIn("re-importing the image", caveats)

    def test_the_recalibration_caveat_stops_short_of_the_same_promise(self):
        """It also ended on a bare "Re-run inference before reporting"."""
        self._caveats()
        recalibration = next(
            c
            for c in self._last["result"]["caveats"]
            if "would have resampled the image to its own scale" in c
        )

        self.assertIn("Re-run inference", recalibration)
        self.assertIn("after discarding the objects it produced", recalibration)
        self.assertIn("because they are confirmed", recalibration)

    def test_an_uncalibrated_image_gets_the_same_route(self):
        """The image has no pixel size at all, so the advice is "set it" *and* this."""
        self.asset.pixel_size_nm = None
        self.asset.save(update_fields=["pixel_size_nm"])

        self._caveats(uncalibrated=True)

        scale = next(
            c for c in self._last["result"]["caveats"] if "not trained for" in c
        )
        self.assertIn("Set the image's pixel size and re-run inference", scale)
        self.assertIn("Re-running inference is not by itself enough", scale)
        self.assertIn("/labels/clear", scale)

    def test_a_correct_run_is_told_none_of_this(self):
        caveats = self._caveats(uncalibrated=False)

        self.assertNotIn("Re-running inference is not by itself enough", caveats)
        self.assertNotIn("/labels/clear", caveats)

    def test_a_pack_that_skipped_no_resample_is_told_none_of_it_either(self):
        """quantem:er declares no canonical_nm, so its uncalibrated run is the
        run a calibrated one would have been. Re-running would return the same
        objects, and telling this user to discard 41 of them would be wrong."""
        er = self._segmentation(get_or_create_er_type)
        for i in range(3):
            self._object(
                er,
                _square(20 + 30 * i, 20),
                source_model="quantem:er",
                stamp=_uncalibrated_stamp(pack_id="quantem:er"),
            )
        _run, got = self._run(er, compartments={"er": str(er.id)})

        caveats = " ".join(got["result"]["caveats"])
        self.assertNotIn("/labels/clear", caveats)
        self.assertNotIn("Re-running inference is not by itself enough", caveats)


class WhatTheCurrentPixelSizeHoldsUpTests(RerunAdviceTestCase):
    """``pixel_size_provenance``, the third sentence keyed on the wrong value."""

    def _block(self, *, uncalibrated: bool) -> dict:
        self._caveats(uncalibrated=uncalibrated)
        return self._last["manifest"]["models"]["image"]["pixel_size_provenance"]

    def test_it_does_not_claim_the_blank_micron_columns_rest_on_it(self):
        block = self._block(uncalibrated=True)

        self.assertEqual(block["effective_nm"], 5.0)
        self.assertEqual(block["source"], "entered_by_hand")
        self.assertFalse(block["applies_to_these_measurements"])
        self.assertIn("typed by a person", block["note"])
        self.assertIn("nothing here rests on it", block["note"])
        self.assertIn("every micron column is blank", block["note"])

    def test_an_ordinary_run_still_says_what_rests_on_it(self):
        block = self._block(uncalibrated=False)

        self.assertTrue(block["applies_to_these_measurements"])
        self.assertIn("Every micron column in this bundle", block["note"])

    def test_the_scale_block_says_which_pixel_size_it_is_reporting(self):
        """``native_pixel_size_nm`` names the stamp field and is not the stamp."""
        self._caveats(uncalibrated=True)
        scale = self._compartment(self._last["manifest"], "mito")["run"]["scale"]

        self.assertEqual(scale["native_pixel_size_nm"], 5.0)
        self.assertIn("as it is now", scale["native_pixel_size_nm_is"])
        self.assertIn("run stamp", scale["native_pixel_size_nm_is"])

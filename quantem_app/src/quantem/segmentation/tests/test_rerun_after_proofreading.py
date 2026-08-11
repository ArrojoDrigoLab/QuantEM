"""What a proofread image is told when it is re-run to pick up a pixel size.

The reported sequence, end to end:

1. An analysis bundle said "Set the image's pixel size and re-run inference".
2. The user set it and re-ran full segmentation.
3. The run completed SUCCESS with ``segment_count: 0`` and
   ``next_steps: ["Nothing changed: the 41 object(s) you have already labelled
   here are exactly as they were.", "A candidate that lands on an object you
   have already confirmed or excluded is not added again, ...", ...]``.

Nothing there is false. What is missing is the one thing the user needed: those
41 objects still record ``native_pixel_size_nm: null``, no re-run can change
that while they exist, and every export of them will carry the same caveat that
sent them here. None of the steps mentioned calibration at all.

These pin the step that closes the loop, and that it fires only when it is true
-- an object with no run stamp says nothing about the scale it was made at, and
a segmentation nobody has labelled is the other branch entirely.
"""

from __future__ import annotations

from django.test import TestCase
from shapely.geometry import Polygon

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.organelle_tasks import zero_object_outcome
from quantem.segmentation.run_identity import build_run_identity
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 128
RUN_ID = "33333333-3333-4333-8333-333333333333"


def _stamp(*, native_pixel_size_nm: float | None) -> dict:
    return build_run_identity(
        run_id=RUN_ID,
        pack_id="quantem:mito",
        threshold=0.5,
        adapter_id=None,
        # A pack with a canonical_nm cannot resample without a pixel size.
        ran_at_nm=None if native_pixel_size_nm is None else 8.0,
        native_pixel_size_nm=native_pixel_size_nm,
        min_area=60,
    )


class RerunAfterProofreadingTests(TestCase):
    def setUp(self):
        self.image = create_small_test_image("Re-run", width=SIZE, height=SIZE)
        self.asset = self.image.asset
        self.asset.pixel_size_nm = 5.0
        self.asset.save(update_fields=["pixel_size_nm"])
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_mitochondria_type()
        )

    def _object(self, index=0, *, label_state="CONFIRMED", stamp=None):
        x = 10 + 30 * index
        polygon = Polygon(((x, 10), (x + 20, 10), (x + 20, 30), (x, 30), (x, 10)))
        features = {"area": polygon.area, "perimeter": polygon.length}
        if stamp is not None:
            features["run"] = stamp
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            source_model="quantem:mito",
            features=features,
        )

    def _steps(self) -> list[str]:
        return zero_object_outcome(self.segmentation)[1]

    def test_it_says_the_re_run_could_not_have_worked(self):
        self._object(0, stamp=_stamp(native_pixel_size_nm=None))
        self._object(1, stamp=_stamp(native_pixel_size_nm=None))

        step = next(s for s in self._steps() if "pixel size" in s)

        self.assertIn("2 object(s) here were produced while this image had no "
                      "pixel size", step)
        self.assertIn("5 nm/px now", step)
        self.assertIn("A re-run cannot replace them", step)

    def test_it_names_the_only_route_there_is(self):
        self._object(0, stamp=_stamp(native_pixel_size_nm=None))

        step = next(s for s in self._steps() if "Discard objects and re-run" in s)

        # Named the endpoint and claimed no screen offered it. I-12 forbids the
        # first; the labeling header's button falsified the second.
        self.assertIn("Discard objects and re-run, on the labeling screen", step)
        self.assertNotIn("/api/", step)
        self.assertNotIn("No screen offers that yet", step)

    def test_the_advice_that_was_already_right_is_still_there(self):
        self._object(0, stamp=_stamp(native_pixel_size_nm=None))
        steps = self._steps()

        self.assertIn("Nothing changed", steps[0])
        self.assertIn("not added again", steps[1])
        self.assertIn("run over an area you have not labelled yet", steps[-1])
        for step in steps:
            self.assertNotIn("Lower the detection threshold", step)

    def test_an_excluded_object_counts_too(self):
        """EXCLUDED objects suppress candidates and carry the same stamp."""
        self._object(0, label_state="EXCLUDED", stamp=_stamp(native_pixel_size_nm=None))

        self.assertTrue(any("Discard objects and re-run" in s for s in self._steps()))

    def test_a_calibrated_run_is_told_nothing_about_calibration(self):
        self._object(0, stamp=_stamp(native_pixel_size_nm=5.0))

        self.assertFalse([s for s in self._steps() if "Discard objects and re-run" in s])

    def test_an_unstamped_object_says_nothing_about_when_it_was_made(self):
        """No stamp is a hand-drawn outline or one made before stamping existed.

        Either way it is not evidence that a run happened without a pixel size,
        and telling this user to discard their work on that basis would be the
        guess this record exists to refuse.
        """
        self._object(0, stamp=None)

        self.assertFalse([s for s in self._steps() if "Discard objects and re-run" in s])

    def test_an_image_with_nothing_labelled_is_still_the_other_branch(self):
        """Nothing suppresses anything here, so the model really did find nothing."""
        self._object(0, label_state="CANDIDATE", stamp=_stamp(native_pixel_size_nm=None))

        steps = self._steps()
        self.assertFalse([s for s in steps if "Discard objects and re-run" in s])
        self.assertTrue(any("threshold" in s for s in steps))

    def test_an_image_that_is_still_uncalibrated_is_told_setting_it_is_not_enough(self):
        self.asset.pixel_size_nm = None
        self.asset.save(update_fields=["pixel_size_nm"])
        self._object(0, stamp=_stamp(native_pixel_size_nm=None))

        step = next(s for s in self._steps() if "Discard objects and re-run" in s)

        self.assertIn("it still has none", step)
        self.assertIn("applied when inference runs, not afterwards", step)

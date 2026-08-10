"""A run that finds nothing needs to know *why* before it gives advice.

Reported: re-running full segmentation over a fully proofread image returned
``segment_count: 0`` with ``next_steps: ["Lower the detection threshold and run
again."]``. It found nothing new because everything was already labelled --
extraction drops any candidate that lands on a confirmed or excluded object --
so that message pushes someone to change a setting they should not touch, and to
re-run a model over work that is already done.

Two things were still wrong with the other branch, and both are pinned here.

*The advice never mentioned the pixel size*, which on this application is the
likeliest reason a run finds nothing: the run resamples the image to the pack's
``canonical_nm``, so the scale decides what apparent size the model sees an
organelle at. One reported image produced 0, 25 and 134 objects over
byte-identical pixels at 5 nm, unset, and 10 nm.

*None of it reached a screen.* The message and the next steps went to the job
log and the job result, which nothing renders; the labeling header and the
viewer chip read ``status_stage``, and a finished run leaves
``CANDIDATES_READY`` behind whether it found two hundred objects or none. So the
user with zero objects read "Mitochondria — Candidates ready" and nothing else.
``ImageSegmentationSerializer.run_notice`` puts the finding on the same payload
as the stage it qualifies.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.jobs.handlers import _segmentation_run_outcome
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.organelle_tasks import (
    NO_OBJECTS_MESSAGE,
    zero_object_notice,
    zero_object_outcome,
)
from quantem.segmentation.serializers import ImageSegmentationSerializer
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image

SIZE = 128


class ZeroResultAdviceTests(TestCase):
    def setUp(self):
        self.image = create_small_test_image("Zero result", width=SIZE, height=SIZE)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _object(self, label_state: str) -> SegmentObject:
        polygon = Polygon(((10, 10), (40, 10), (40, 40), (10, 40), (10, 10)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            features={},
        )

    def test_an_image_with_nothing_labelled_still_gets_the_threshold_advice(self):
        message, next_steps = zero_object_outcome(self.segmentation)
        self.assertEqual(message, NO_OBJECTS_MESSAGE)
        self.assertTrue(any("threshold" in step for step in next_steps))

    def test_candidates_alone_are_not_proofreading(self):
        """An unlabelled candidate suppresses nothing, so the advice still holds."""
        self._object("CANDIDATE")
        message, next_steps = zero_object_outcome(self.segmentation)
        self.assertEqual(message, NO_OBJECTS_MESSAGE)
        self.assertTrue(any("threshold" in step for step in next_steps))

    def test_a_proofread_image_is_never_told_to_lower_the_threshold(self):
        for label_state in ("CONFIRMED", "EXCLUDED"):
            with self.subTest(label_state=label_state):
                SegmentObject.objects.filter(segmentation=self.segmentation).delete()
                self._object(label_state)

                message, next_steps = zero_object_outcome(self.segmentation)

                self.assertNotIn("Lower the detection threshold", message)
                for step in next_steps:
                    self.assertNotIn("Lower the detection threshold", step)

    def test_it_explains_that_nothing_was_lost_and_why_nothing_was_added(self):
        self._object("CONFIRMED")
        message, next_steps = zero_object_outcome(self.segmentation)

        self.assertIn("no new objects", message)
        self.assertIn("1 object(s) already labelled", message)
        joined = " ".join(next_steps)
        self.assertIn("Nothing changed", joined)
        self.assertIn("not added again", joined)

    def test_the_job_result_carries_the_same_answer(self):
        self._object("CONFIRMED")

        message, outcome = _segmentation_run_outcome(0, segmentation=self.segmentation)

        self.assertFalse(outcome["found_objects"])
        self.assertEqual(outcome["segment_count"], 0)
        self.assertNotIn(
            "Lower the detection threshold", " ".join(outcome["next_steps"])
        )
        self.assertIn("no new objects", message)

    def test_a_run_that_found_objects_is_unchanged(self):
        message, outcome = _segmentation_run_outcome(7, segmentation=self.segmentation)
        self.assertEqual(outcome, {"segment_count": 7, "found_objects": True})
        self.assertIn("7 objects found", message)


class PixelSizeIsTheFirstThingToCheckTests(TestCase):
    """The scale leads the advice, and it was not on the list at all.

    Lowering the threshold on a wrongly-scaled run does not recover the missing
    objects -- the model is looking at organelles at the wrong apparent size --
    so it produces different rubbish. The scale is therefore named before the
    threshold rather than after it.
    """

    def _segmentation(self, *, pixel_size_nm):
        image = create_small_test_image("Scale advice", width=SIZE, height=SIZE)
        image.asset.pixel_size_nm = pixel_size_nm
        image.asset.save(update_fields=["pixel_size_nm"])
        return ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def test_an_uncalibrated_image_is_told_to_set_the_pixel_size(self):
        _message, next_steps = zero_object_outcome(self._segmentation(pixel_size_nm=None))

        self.assertIn("no pixel size", next_steps[0])
        self.assertIn("Set the pixel size", next_steps[0])
        # Before the threshold, not after it.
        self.assertTrue(any("threshold" in step for step in next_steps[1:]))

    def test_a_calibrated_image_is_told_to_check_the_value_it_has(self):
        _message, next_steps = zero_object_outcome(self._segmentation(pixel_size_nm=5.0))

        self.assertIn("5 nm/px", next_steps[0])
        self.assertIn("before the threshold", next_steps[0])

    def test_the_headline_names_the_pixel_size_before_anything_else(self):
        message, _ = zero_object_outcome(self._segmentation(pixel_size_nm=5.0))
        self.assertEqual(message, NO_OBJECTS_MESSAGE)
        self.assertLess(message.index("pixel size"), message.index("threshold"))


class ZeroResultReachesAScreenTests(TestCase):
    """``run_notice`` on the segmentation payload, or ``null``."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Zero result on screen", width=SIZE, height=SIZE
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )

    def _notice(self, segmentation=None):
        return ImageSegmentationSerializer(segmentation or self.segmentation).data[
            "run_notice"
        ]

    def _object(self, label_state: str = "CANDIDATE") -> SegmentObject:
        polygon = Polygon(((10, 10), (40, 10), (40, 40), (10, 40), (10, 10)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            features={},
        )

    def test_candidates_ready_with_nothing_in_it_carries_the_explanation(self):
        notice = self._notice()

        self.assertIsNotNone(notice, '"Candidates ready" was the whole story again')
        self.assertEqual(notice["kind"], "no_objects")
        self.assertIn("without finding any objects", notice["message"])
        # The advice that used to reach only the job log.
        self.assertTrue(any("pixel size" in step for step in notice["next_steps"]))

    def test_it_is_the_same_advice_the_job_log_gets(self):
        self.assertEqual(
            self._notice()["next_steps"],
            zero_object_notice(self.segmentation)["next_steps"],
        )
        self.assertEqual(
            self._notice()["next_steps"],
            zero_object_outcome(self.segmentation)[1],
        )

    def test_one_object_of_any_label_state_clears_it(self):
        for label_state in ("CANDIDATE", "INFERRED", "CONFIRMED", "EXCLUDED"):
            with self.subTest(label_state=label_state):
                SegmentObject.objects.filter(segmentation=self.segmentation).delete()
                self._object(label_state)
                self.assertIsNone(self._notice())

    def test_a_run_still_going_says_nothing(self):
        for stage in ("UNSTARTED", "RUNNING_INFERENCE", "EXTRACTING_CANDIDATES", "FAILED"):
            with self.subTest(stage=stage):
                self.segmentation.status_stage = stage
                self.segmentation.save(update_fields=["status_stage"])
                self.assertIsNone(self._notice())

    def test_a_manual_only_type_is_never_told_to_lower_a_threshold(self):
        """Tissue is created at ``CANDIDATES_READY`` with no run behind it.

        There is no model, no threshold and no scale resampling in that
        workflow, so the empty case is not a finding about it.
        """
        tissue = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_tissue_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )
        self.assertIsNone(self._notice(tissue))

    def test_it_arrives_on_the_list_endpoint_the_screens_read(self):
        response = self.client.get(
            f"/api/assets/{self.image.asset.id}/segmentations/"
        )
        self.assertEqual(response.status_code, 200, response.data)
        payload = next(
            row for row in response.data if row["id"] == str(self.segmentation.id)
        )
        self.assertEqual(payload["status_stage"], "CANDIDATES_READY")
        self.assertIsNotNone(payload["run_notice"])

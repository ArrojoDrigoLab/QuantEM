"""Invariant I-5, end to end: a re-run invalidates the quality estimate.

*"A quality estimate is invalidated and visibly greyed the moment
``run_version`` changes."*

Everything the invariant needs on the read side was already here -- the
column, the unique constraints, ``current_version_for``, and a
``previous_version`` block in every quality payload -- and **nothing ever
advanced the number**. No code path created a
:class:`~quantem.segmentation.models.SegmentationResultVersion` row, no writer
set :attr:`~quantem.segmentation.models.SegmentObject.run_version` to anything
but the model default, so the version could only ever be 1, the invalidation
could never fire, and a spot check taken against one candidate set kept feeding
the headline after the model had been re-run at a different threshold.

All seven of these fail against that tree, every one of them on the version
being stuck at 1: ``AssertionError: 1 != 2``, ``1 != 10``, and a
``SegmentationResultVersion.DoesNotExist`` where a row should have been
recorded. :meth:`StaleQualityEstimateTests.
test_a_re_run_invalidates_the_answers_a_user_already_gave` is the one that
spells out the user-visible consequence: it draws twelve questions, answers
all twelve, re-runs the model over a completely different set of objects, and
finds the same twelve answers still live in ``checks`` with the headline still
resting on them.
"""

from __future__ import annotations

import numpy as np
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.seg_core.db.extraction import run_extraction
from quantem.seg_core.types import ExtractedSegment, InferenceResult
from quantem.segmentation.models import (
    ImageSegmentation,
    QualityCheck,
    SegmentationResultVersion,
    SegmentObject,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

_MITO = "quantem:mito"


def _square(x: float, y: float, size: float = 24.0) -> list[tuple[float, float]]:
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]


def _extracted(x: float, y: float, size: float = 24.0) -> ExtractedSegment:
    coords = _square(x, y, size)
    polygon = Polygon(coords)
    return ExtractedSegment(
        polygon_coords=coords,
        bbox_xyxy=polygon.bounds,
        centroid_xy=(polygon.centroid.x, polygon.centroid.y),
        area=int(polygon.area),
        confidence_score=0.8,
        features={"mito_generated": True},
    )


class _StubSegmenter:
    """Hands back a fixed candidate set. No model, no probability map.

    The version is advanced by the *replacement*, not by the arithmetic that
    produced it, so a stub is the honest fixture here: it isolates the thing
    under test from a four-second model load.
    """

    name = "mito"
    generated_flag = "mito_generated"
    source_model = _MITO
    min_area = 1

    def __init__(self, segments: list[ExtractedSegment]):
        self._segments = segments

    def extract_instances(
        self,
        prob,
        image,
        prob_maps,
        *,
        min_area,
        coordinate_offset,
        on_progress=None,
    ):
        return list(self._segments)


class ResultVersionTestCase(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Result version test image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.pixels = np.zeros((128, 128), dtype=np.uint8)

    def _pass(self, *xs: float, run_id: str = "run") -> None:
        """One model pass that writes a candidate at each x."""
        run_extraction(
            _StubSegmenter([_extracted(x, 10.0) for x in xs]),
            self.segmentation,
            InferenceResult(prob_maps={}, prob=None),
            self.pixels,
            run_identity={"id": run_id, "pack_id": _MITO},
        )

    def _live(self):
        return SegmentObject.objects.filter(
            segmentation=self.segmentation, superseded_at__isnull=True
        )

    def _version(self) -> int:
        return SegmentationResultVersion.current_version_for(self.segmentation)


class ResultVersionAdvancesTests(ResultVersionTestCase):
    def test_the_first_pass_on_a_new_segmentation_is_version_one(self):
        """Not version 2. There was no earlier result for it to follow.

        The number is read before the pass deletes anything, precisely so that
        "how many results have there been" is not answered by looking at the
        objects this pass has just written.
        """
        self._pass(10, 50, 90)

        self.assertEqual(self._version(), 1)
        self.assertEqual(
            sorted(self._live().values_list("run_version", flat=True)), [1, 1, 1]
        )
        row = SegmentationResultVersion.objects.get(segmentation=self.segmentation)
        self.assertEqual(row.version, 1)
        self.assertEqual(row.object_count, 3)
        self.assertEqual(row.run_identity["pack_id"], _MITO)

    def test_a_second_pass_numbers_a_new_result_and_moves_the_objects_onto_it(self):
        self._pass(10, 50, 90, run_id="first")
        self._pass(12, 52, 92, run_id="second")

        self.assertEqual(self._version(), 2)
        self.assertEqual(
            sorted(self._live().values_list("run_version", flat=True)), [2, 2, 2]
        )
        self.assertEqual(
            sorted(
                SegmentationResultVersion.objects.filter(
                    segmentation=self.segmentation
                ).values_list("version", flat=True)
            ),
            [1, 2],
        )

    def test_ten_passes_number_ten_results(self):
        """The dial dragged back and forth. Monotone, no reuse, no gaps."""
        for index in range(10):
            self._pass(10 + index, run_id=f"run-{index}")

        self.assertEqual(self._version(), 10)
        self.assertEqual(
            list(
                SegmentationResultVersion.objects.filter(
                    segmentation=self.segmentation
                )
                .order_by("version")
                .values_list("version", flat=True)
            ),
            list(range(1, 11)),
        )

    def test_a_pass_that_changes_nothing_does_not_number_a_new_result(self):
        """Nothing deleted, nothing written: there is no new result to number.

        The version is what invalidates a user's answers, so advancing it for a
        pass that left every object exactly where it was would throw away real
        work to describe a change that did not happen. (A pass that finds
        nothing where there *were* candidates is a different case: it deleted
        them, so the objects on screen did change.)
        """
        run_extraction(
            _StubSegmenter([]),
            self.segmentation,
            InferenceResult(prob_maps={}, prob=None),
            self.pixels,
            run_identity={"id": "empty", "pack_id": _MITO},
        )

        self.assertEqual(self._version(), 1)
        self.assertFalse(
            SegmentationResultVersion.objects.filter(
                segmentation=self.segmentation
            ).exists()
        )

    def test_a_pass_that_finds_nothing_still_numbers_the_emptiness(self):
        """It deleted every candidate, so the objects on screen did change.

        A user who spot-checked the old set is now looking at an image with no
        objects in it; keeping their estimate live would attach a precision to
        a result that no longer exists.
        """
        self._pass(10, 50)

        run_extraction(
            _StubSegmenter([]),
            self.segmentation,
            InferenceResult(prob_maps={}, prob=None),
            self.pixels,
            run_identity={"id": "empty", "pack_id": _MITO},
        )

        self.assertEqual(self._version(), 2)
        self.assertEqual(self._live().count(), 0)
        self.assertEqual(
            SegmentationResultVersion.objects.get(
                segmentation=self.segmentation, version=2
            ).object_count,
            0,
        )

    def test_an_object_the_user_confirmed_comes_with_the_new_version(self):
        """It is still on screen, so it is still part of the result.

        A CONFIRMED object survives a pass untouched -- that is the promise the
        whole replacement mechanism exists to keep -- but it is not thereby a
        member of the *previous* result only. Leaving it on the old number
        would drop it out of ``live_model_objects`` and the count the spot
        check quotes would quietly stop counting everything already confirmed.
        """
        self._pass(10, 50)
        kept = self._live().first()
        kept.label_state = "CONFIRMED"
        kept.save()

        self._pass(200, run_id="second")

        kept.refresh_from_db()
        self.assertEqual(kept.label_state, "CONFIRMED")
        self.assertEqual(kept.run_version, 2)
        self.assertEqual(
            SegmentationResultVersion.objects.get(
                segmentation=self.segmentation, version=2
            ).object_count,
            self._live().count(),
        )

    def test_a_hand_drawn_object_is_not_a_result_to_replace(self):
        """A segmentation whose only objects are the user's own has no result.

        Its first model pass is still version 1: the person's outlines are not
        a numbered model result, and counting them as one would open every
        hand-annotated image on "version 2".
        """
        by_hand = Polygon(_square(300, 300))
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            label_state="CONFIRMED",
            source_model="manual",
            geometry=by_hand,
            centroid=by_hand.centroid,
            bbox=by_hand.envelope,
        )

        self._pass(10, 50)

        self.assertEqual(self._version(), 1)


class StaleQualityEstimateTests(ResultVersionTestCase):
    """The consequence the invariant is actually about."""

    def _spot_check(self, n: int = 12) -> dict:
        response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/spot-check/?n={n}"
        )
        self.assertEqual(response.status_code, 200, response.content)
        return response.json()

    def test_a_re_run_invalidates_the_answers_a_user_already_gave(self):
        self._pass(*[10 + 30 * i for i in range(14)], run_id="first")

        drawn = self._spot_check()
        self.assertEqual(drawn["run_version"], 1)
        self.assertGreaterEqual(len(drawn["checks"]), 12)
        self.assertIsNone(drawn["previous_version"]["spot_check"])

        for check in drawn["checks"]:
            answered = self.client.post(
                f"/api/segmentations/{self.segmentation.id}/spot-check/answer",
                {"check_id": check["id"], "answer": QualityCheck.ANSWER_YES},
                content_type="application/json",
            )
            self.assertEqual(answered.status_code, 200, answered.content)
        self.assertEqual(answered.json()["counts"]["scored"], len(drawn["checks"]))

        # The model runs again. Every answer above is about objects that are no
        # longer on screen.
        self._pass(*[400 + 30 * i for i in range(14)], run_id="second")

        after = self._spot_check()
        self.assertEqual(after["run_version"], 2)

        previous = after["previous_version"]["spot_check"]
        self.assertIsNotNone(
            previous,
            "the earlier sample has to be reported so the client can grey it",
        )
        self.assertEqual(previous["run_version"], 1)
        self.assertEqual(previous["counts"]["scored"], len(drawn["checks"]))

        # And the new version starts from nothing rather than inheriting a
        # headline it did not earn.
        answered_now = [c for c in after["checks"] if c["answer"]]
        self.assertEqual(answered_now, [])
        self.assertFalse(after["headline_ready"])
        self.assertIn("not_enough_checks", after["headline_blockers"])

    def test_the_new_version_counts_the_objects_that_are_actually_there(self):
        self._pass(*[10 + 30 * i for i in range(4)], run_id="first")
        self.assertEqual(self._spot_check(n=1)["object_count"], 4)

        self._pass(*[400 + 30 * i for i in range(7)], run_id="second")

        payload = self._spot_check(n=1)
        self.assertEqual(payload["run_version"], 2)
        self.assertEqual(payload["object_count"], 7)

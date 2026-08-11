"""The two-number quality answer, from the server's side.

Every test here is one of the properties the estimate rests on. They are worth
naming, because each of them is a way the feature could look like it works
while reporting a number that means nothing:

* a sample that reshuffles between requests -- "1 of 12" would be a different
  object each time, and the twelve answers would be about twenty objects;
* a sample drawn from objects the user already confirmed -- the model would be
  scored on the user's own work;
* "not sure" counted as either a hit or a miss -- a judgement the user
  explicitly declined, invented on their behalf;
* a headline shown from the spot check alone -- precision sold as accuracy,
  which is the exact failure the count box exists to prevent;
* answers carried across a new result version -- a score for objects that are
  no longer on screen.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.segmentation.models import (
    CountBox,
    ImageSegmentation,
    QualityCheck,
    SegmentationResultVersion,
    SegmentObject,
)
from quantem.segmentation.quality_sampling import (
    MIN_SPOT_CHECK_SAMPLE,
    count_answers,
    derive_seed,
    order_by_sample,
    self_confirmation,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 512


def _polygon(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))


class QualityFixture(TestCase):
    """A segmentation with a run's worth of untouched model objects."""

    n_objects = 40

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Quality answer", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.segments = [self._make_segment(index) for index in range(self.n_objects)]
        self.base = f"/api/segmentations/{self.segmentation.id}"

    def _make_segment(
        self,
        index: int,
        *,
        label_state: str = "INFERRED",
        source_model: str = "quantem:mito",
        refined: str = "UNREFINED",
    ) -> SegmentObject:
        column, row = index % 8, index // 8
        polygon = _polygon(column * 60, row * 60, column * 60 + 40, row * 60 + 40)
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            refined=refined,
            source_model=source_model,
            confidence_score=0.5 + (index % 10) / 100.0,
        )

    def _spot_check(self, n: int | None = None) -> dict:
        query = "" if n is None else f"?n={n}"
        response = self.client.get(f"{self.base}/spot-check/{query}")
        self.assertEqual(response.status_code, 200, response.data)
        return response.data

    def _answer(self, check: dict, answer: str, *, expect: int = 200) -> dict:
        response = self.client.post(
            f"{self.base}/spot-check/answer",
            {"check_id": check["id"], "answer": answer},
            format="json",
        )
        self.assertEqual(response.status_code, expect, response.data)
        return response.data

    def _answer_all(self, answers: list[str]) -> dict:
        payload = self._spot_check(len(answers))
        latest = payload
        for check, answer in zip(payload["checks"], answers, strict=True):
            latest = self._answer(check, answer)
        return latest


class StableSampleTests(QualityFixture):
    def test_the_same_twelve_come_back_in_the_same_order(self):
        first = self._spot_check(12)
        second = self._spot_check(12)
        self.assertEqual(len(first["checks"]), 12)
        self.assertEqual(
            [c["segment"]["id"] for c in first["checks"]],
            [c["segment"]["id"] for c in second["checks"]],
        )
        self.assertEqual([c["ordinal"] for c in first["checks"]], list(range(12)))

    def test_the_draw_is_written_down_rather_than_recomputed(self):
        payload = self._spot_check(12)
        self.assertEqual(QualityCheck.objects.filter(segmentation=self.segmentation).count(), 12)
        seeds = set(
            QualityCheck.objects.filter(segmentation=self.segmentation).values_list(
                "sample_seed", flat=True
            )
        )
        self.assertEqual(seeds, {payload["sample_seed"]})

    def test_asking_for_more_keeps_the_first_twelve_where_they_were(self):
        """Extending is the next twenty-four of the same order, not a reshuffle.

        A recomputed sample would reshuffle as answers labelled objects out of
        the untouched pool, so "1 of 12" and "1 of 36" would be different
        objects and the first twelve answers would be about a sample that no
        longer exists.
        """
        first = self._spot_check(12)
        self._answer(first["checks"][0], QualityCheck.ANSWER_YES)
        self._answer(first["checks"][1], QualityCheck.ANSWER_NOT_THE_THING)

        extended = self._spot_check(24)
        self.assertEqual(len(extended["checks"]), 24)
        self.assertEqual(
            [c["segment"]["id"] for c in extended["checks"][:12]],
            [c["segment"]["id"] for c in first["checks"]],
        )
        self.assertEqual(extended["checks"][0]["answer"], QualityCheck.ANSWER_YES)

    def test_no_object_is_asked_about_twice(self):
        payload = self._spot_check(24)
        drawn = [c["segment"]["id"] for c in payload["checks"]]
        self.assertEqual(len(drawn), len(set(drawn)))

    def test_a_wrong_shape_answer_does_not_put_the_object_back_in_the_pool(self):
        """It writes no label, so only the drawn-id exclusion keeps it out."""
        first = self._spot_check(12)
        answered = first["checks"][0]["segment"]["id"]
        self._answer(first["checks"][0], QualityCheck.ANSWER_WRONG_SHAPE)
        extended = self._spot_check(24)
        drawn = [c["segment"]["id"] for c in extended["checks"]]
        self.assertEqual(drawn.count(answered), 1)

    def test_the_order_survives_a_different_process(self):
        """The property a server restart needs, and the one ``hash()`` breaks.

        ``PYTHONHASHSEED`` randomises :func:`hash` for strings per process, so
        an order built on it silently differs between the request that drew the
        sample and the request that reads it back. This runs the real ordering
        function in a fresh interpreter with a different hash seed and requires
        the identical order.
        """
        ids = [str(segment.id) for segment in self.segments]
        seed = derive_seed("spot-check", self.segmentation.pk, 1)
        here = order_by_sample(ids, seed)

        source_root = str(Path(sys.modules["quantem"].__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "12345"
        env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("DJANGO_SETTINGS_MODULE", "quantem.core.settings")
        program = (
            "import json, sys, django; django.setup();"
            "from quantem.segmentation.quality_sampling import order_by_sample;"
            "payload = json.loads(sys.stdin.read());"
            "print(json.dumps(order_by_sample(payload['ids'], payload['seed'])))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            input=json.dumps({"ids": ids, "seed": seed}),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout.strip()), here)


class SamplePoolTests(QualityFixture):
    n_objects = 12

    def test_only_untouched_model_objects_are_drawn(self):
        kept = self._make_segment(100, label_state="CONFIRMED")
        removed = self._make_segment(101, label_state="EXCLUDED")
        hand_drawn = self._make_segment(102, label_state="INFERRED", source_model="manual")
        edited = self._make_segment(103, refined="MANUAL")

        payload = self._spot_check(30)
        drawn = {c["segment"]["id"] for c in payload["checks"]}
        for excluded in (kept, removed, hand_drawn, edited):
            self.assertNotIn(str(excluded.id), drawn)
        self.assertEqual(len(drawn), self.n_objects)

    def test_a_superseded_object_is_not_part_of_this_result(self):
        stale = self._make_segment(200)
        SegmentObject.objects.filter(id=stale.id).update(superseded_at=timezone.now())
        payload = self._spot_check(30)
        drawn = {c["segment"]["id"] for c in payload["checks"]}
        self.assertNotIn(str(stale.id), drawn)

    def test_the_object_count_is_what_the_model_found(self):
        self._make_segment(300, label_state="CONFIRMED")
        self._make_segment(301, source_model="manual")
        payload = self._spot_check(1)
        # Thirteen model objects (twelve untouched plus the confirmed one);
        # the hand-drawn outline is not something the model found.
        self.assertEqual(payload["object_count"], self.n_objects + 1)

    def test_a_run_with_nothing_untouched_left_draws_nothing_rather_than_failing(self):
        SegmentObject.objects.filter(segmentation=self.segmentation).update(label_state="CONFIRMED")
        payload = self._spot_check(12)
        self.assertEqual(payload["checks"], [])
        self.assertEqual(payload["pool_remaining"], 0)
        self.assertFalse(payload["headline_ready"])


class UnsureIsExcludedTests(QualityFixture):
    def test_not_sure_never_reaches_the_denominator(self):
        answers = [QualityCheck.ANSWER_YES] * 8
        answers += [QualityCheck.ANSWER_NOT_THE_THING] * 2
        answers += [QualityCheck.ANSWER_UNSURE] * 2
        payload = self._answer_all(answers)

        counts = payload["counts"]
        self.assertEqual(counts["answered"], 12)
        self.assertEqual(counts["unsure"], 2)
        self.assertEqual(counts["scored"], 10)
        self.assertEqual(counts["positive"], 8)

    def test_the_count_of_unsure_is_reported_so_the_sentence_can_say_it(self):
        payload = self._answer_all([QualityCheck.ANSWER_UNSURE] * 3)
        self.assertEqual(payload["counts"]["unsure"], 3)
        self.assertEqual(payload["counts"]["scored"], 0)

    def test_a_wrong_shape_counts_against_and_is_reported_apart(self):
        payload = self._answer_all([QualityCheck.ANSWER_YES, QualityCheck.ANSWER_WRONG_SHAPE])
        counts = payload["counts"]
        self.assertEqual(counts["scored"], 2)
        self.assertEqual(counts["positive"], 1)
        self.assertEqual(counts["wrong_shape"], 1)


class AnswersWriteTheNormalLabelTests(QualityFixture):
    """Every answer is also review, so the minute is never wasted work.

    But only where the answer says one thing unambiguously. "Wrong shape" is a
    real object with a bad outline and "not sure" is not an instruction; both
    leave the label alone, because writing one would put a decision in the data
    that the user did not make.
    """

    def test_yes_keeps_the_object(self):
        payload = self._spot_check(4)
        check = payload["checks"][0]
        self._answer(check, QualityCheck.ANSWER_YES)
        segment = SegmentObject.objects.get(id=check["segment"]["id"])
        self.assertEqual(segment.label_state, "CONFIRMED")

    def test_not_the_thing_removes_the_object(self):
        payload = self._spot_check(4)
        check = payload["checks"][1]
        self._answer(check, QualityCheck.ANSWER_NOT_THE_THING)
        segment = SegmentObject.objects.get(id=check["segment"]["id"])
        self.assertEqual(segment.label_state, "EXCLUDED")

    def test_wrong_shape_and_not_sure_leave_the_label_alone(self):
        payload = self._spot_check(4)
        for check, answer in (
            (payload["checks"][2], QualityCheck.ANSWER_WRONG_SHAPE),
            (payload["checks"][3], QualityCheck.ANSWER_UNSURE),
        ):
            with self.subTest(answer=answer):
                self._answer(check, answer)
                segment = SegmentObject.objects.get(id=check["segment"]["id"])
                self.assertEqual(segment.label_state, "INFERRED")

    def test_an_answer_that_is_not_one_of_the_four_is_refused(self):
        payload = self._spot_check(1)
        response = self.client.post(
            f"{self.base}/spot-check/answer",
            {"check_id": payload["checks"][0]["id"], "answer": "probably"},
            format="json",
        )
        self.assertEqual(response.status_code, 400, response.data)

    def test_re_answering_replaces_the_answer_and_moves_its_timestamp(self):
        payload = self._spot_check(1)
        check = payload["checks"][0]
        first = self._answer(check, QualityCheck.ANSWER_UNSURE)
        second = self._answer(check, QualityCheck.ANSWER_YES)
        self.assertEqual(second["check"]["answer"], QualityCheck.ANSWER_YES)
        self.assertNotEqual(first["check"]["answered_at"], second["check"]["answered_at"])
        self.assertEqual(second["counts"]["answered"], 1)


class HeadlineNeedsBothHalvesTests(QualityFixture):
    """The single most important rule in the plan, enforced server-side.

    A precision-only headline is how a model that finds 511 of 1 300 real
    objects gets reported as "9 in 10" while the user's counts run 60 % low.
    """

    def _complete_count_box(self, *, n_marked: int, n_matched: int) -> dict:
        response = self.client.post(
            f"{self.base}/count-box",
            {"n_marked": n_marked, "n_matched": n_matched, "complete": True},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        return response.data

    def test_twelve_answers_alone_do_not_earn_a_headline(self):
        payload = self._answer_all([QualityCheck.ANSWER_YES] * MIN_SPOT_CHECK_SAMPLE)
        self.assertFalse(payload["headline_ready"])
        self.assertIn("no_count_box", payload["headline_blockers"])

    def test_a_count_box_alone_does_not_earn_a_headline(self):
        payload = self._complete_count_box(n_marked=31, n_matched=23)
        self.assertFalse(payload["headline_ready"])
        self.assertIn("not_enough_checks", payload["headline_blockers"])

    def test_an_unfinished_count_box_is_not_a_count_box(self):
        self._answer_all([QualityCheck.ANSWER_YES] * MIN_SPOT_CHECK_SAMPLE)
        response = self.client.post(f"{self.base}/count-box", {"n_marked": 4}, format="json")
        self.assertEqual(response.status_code, 200, response.data)
        self.assertFalse(response.data["headline_ready"])
        self.assertIn("count_box_unfinished", response.data["headline_blockers"])

    def test_eleven_scored_answers_are_not_twelve(self):
        """Eleven yes and one "not sure" is a sample of eleven, not of twelve."""
        answers = [QualityCheck.ANSWER_YES] * 11 + [QualityCheck.ANSWER_UNSURE]
        self._answer_all(answers)
        payload = self._complete_count_box(n_marked=31, n_matched=23)
        self.assertEqual(payload["counts"]["scored"], 11)
        self.assertFalse(payload["headline_ready"])
        self.assertIn("not_enough_checks", payload["headline_blockers"])

    def test_both_halves_earn_the_headline(self):
        self._answer_all([QualityCheck.ANSWER_YES] * MIN_SPOT_CHECK_SAMPLE)
        payload = self._complete_count_box(n_marked=31, n_matched=23)
        self.assertTrue(payload["headline_ready"])
        self.assertEqual(payload["headline_blockers"], [])

    def test_the_recall_half_reports_marked_matched_and_missed(self):
        """The 511-of-1 300 case: the payload has to make the miss visible.

        The wording ("missing about 3 in 5") is the client's; what the server
        owes it is the three numbers, and a ``n_missed`` it does not have to
        subtract for itself.
        """
        payload = self._complete_count_box(n_marked=1300, n_matched=511)
        box = payload["count_box"]
        self.assertEqual(box["n_marked"], 1300)
        self.assertEqual(box["n_matched"], 511)
        self.assertEqual(box["n_missed"], 789)

    def test_more_matched_than_marked_is_refused(self):
        response = self.client.post(
            f"{self.base}/count-box",
            {"n_marked": 5, "n_matched": 9},
            format="json",
        )
        self.assertEqual(response.status_code, 400, response.data)


class CountBoxPlacementTests(QualityFixture):
    def test_the_app_proposes_the_box_and_the_get_does_not_create_it(self):
        response = self.client.get(f"{self.base}/count-box")
        self.assertEqual(response.status_code, 200, response.data)
        self.assertIsNone(response.data["count_box"])
        proposed = response.data["proposed_count_box"]
        self.assertIsNotNone(proposed)
        self.assertGreater(proposed["width"], 0)
        self.assertGreater(proposed["height"], 0)
        self.assertEqual(CountBox.objects.count(), 0)

    def test_the_proposal_is_the_same_rectangle_every_time(self):
        first = self.client.get(f"{self.base}/count-box").data["proposed_count_box"]
        second = self.client.get(f"{self.base}/count-box").data["proposed_count_box"]
        self.assertEqual(first, second)

    def test_the_box_that_is_saved_is_the_box_that_was_proposed(self):
        proposed = self.client.get(f"{self.base}/count-box").data["proposed_count_box"]
        saved = self.client.post(f"{self.base}/count-box", {}, format="json")
        self.assertEqual(saved.status_code, 200, saved.data)
        box = saved.data["count_box"]
        for key in ("x", "y", "width", "height", "seed"):
            self.assertEqual(box[key], proposed[key], key)

    def test_the_box_stays_inside_the_image(self):
        proposed = self.client.get(f"{self.base}/count-box").data["proposed_count_box"]
        self.assertGreaterEqual(proposed["x"], 0)
        self.assertGreaterEqual(proposed["y"], 0)
        self.assertLessEqual(proposed["x"] + proposed["width"], SIZE)
        self.assertLessEqual(proposed["y"] + proposed["height"], SIZE)

    def test_there_is_one_box_per_result_version(self):
        self.client.post(f"{self.base}/count-box", {"n_marked": 3}, format="json")
        self.client.post(f"{self.base}/count-box", {"n_marked": 7}, format="json")
        self.assertEqual(CountBox.objects.count(), 1)
        self.assertEqual(CountBox.objects.get().n_marked, 7)

    def test_how_the_box_was_placed_is_recorded_beside_it(self):
        """A centred fallback measures the middle of the image, not the tissue.

        Re-deriving it later needs the image, and the reason to know it is
        precisely that the image could not be read -- so it is stored.
        """
        saved = self.client.post(f"{self.base}/count-box", {}, format="json")
        self.assertEqual(saved.status_code, 200, saved.data)
        self.assertEqual(saved.data["count_box"]["placement"], "tissue_scored")
        self.assertEqual(CountBox.objects.get().placement, "tissue_scored")

    def test_the_caller_cannot_choose_where_the_box_goes(self):
        """A user-chosen box is a biased box, so there is no way to send one."""
        proposed = self.client.get(f"{self.base}/count-box").data["proposed_count_box"]
        response = self.client.post(
            f"{self.base}/count-box",
            {"x": 0, "y": 0, "width": 32, "height": 32},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        box = response.data["count_box"]
        self.assertEqual(box["x"], proposed["x"])
        self.assertEqual(box["y"], proposed["y"])
        self.assertEqual(box["width"], proposed["width"])


class NewResultVersionInvalidatesBothHalvesTests(QualityFixture):
    """A new version is new objects, so both judgements are about the past."""

    def _publish_version_two(self):
        SegmentationResultVersion.objects.create(
            segmentation=self.segmentation, version=2, object_count=self.n_objects
        )
        SegmentObject.objects.filter(segmentation=self.segmentation).update(run_version=2)

    def test_the_answers_and_the_box_do_not_carry_over(self):
        self._answer_all([QualityCheck.ANSWER_YES] * MIN_SPOT_CHECK_SAMPLE)
        self.client.post(
            f"{self.base}/count-box",
            {"n_marked": 31, "n_matched": 23, "complete": True},
            format="json",
        )
        before = self._spot_check(12)
        self.assertTrue(before["headline_ready"])

        self._publish_version_two()
        after = self.client.get(f"{self.base}/count-box").data

        self.assertEqual(after["run_version"], 2)
        self.assertEqual(after["counts"]["answered"], 0)
        self.assertIsNone(after["count_box"])
        self.assertFalse(after["headline_ready"])

    def test_the_previous_version_is_reported_so_it_can_be_greyed_not_hidden(self):
        self._answer_all([QualityCheck.ANSWER_YES] * MIN_SPOT_CHECK_SAMPLE)
        self.client.post(
            f"{self.base}/count-box",
            {"n_marked": 31, "n_matched": 23, "complete": True},
            format="json",
        )
        self._publish_version_two()
        payload = self.client.get(f"{self.base}/count-box").data

        previous = payload["previous_version"]
        self.assertEqual(previous["spot_check"]["run_version"], 1)
        self.assertEqual(previous["spot_check"]["counts"]["answered"], MIN_SPOT_CHECK_SAMPLE)
        self.assertEqual(previous["count_box"]["n_marked"], 31)

    def test_a_new_version_draws_its_own_sample(self):
        first = self._spot_check(12)
        self._publish_version_two()
        second = self._spot_check(12)
        self.assertEqual(second["run_version"], 2)
        self.assertEqual(len(second["checks"]), 12)
        self.assertNotEqual(second["sample_seed"], first["sample_seed"])
        self.assertEqual(
            QualityCheck.objects.filter(segmentation=self.segmentation, run_version=1).count(),
            12,
        )


class SelfConfirmationCaveatTests(QualityFixture):
    def test_agreeing_with_the_models_own_guesses_raises_the_caveat(self):
        payload = self._answer_all([QualityCheck.ANSWER_YES] * MIN_SPOT_CHECK_SAMPLE)
        caveat = payload["self_confirmation"]
        self.assertEqual(caveat["n_positive"], MIN_SPOT_CHECK_SAMPLE)
        self.assertEqual(caveat["fraction"], 1.0)
        self.assertTrue(caveat["applies"])

    def test_with_nothing_agreed_with_there_is_nothing_to_caveat(self):
        payload = self._answer_all([QualityCheck.ANSWER_NOT_THE_THING] * 3)
        caveat = payload["self_confirmation"]
        self.assertEqual(caveat["n_positive"], 0)
        self.assertIsNone(caveat["fraction"])
        self.assertFalse(caveat["applies"])

    def test_the_threshold_is_reported_beside_the_fraction(self):
        payload = self._spot_check(1)
        self.assertEqual(payload["self_confirmation"]["threshold"], 0.8)


class CompletionLockTests(QualityFixture):
    def _mark_finished(self):
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save(update_fields=["status_stage"])

    def test_a_finished_image_refuses_an_answer(self):
        payload = self._spot_check(1)
        check = payload["checks"][0]
        self._mark_finished()
        self._answer(check, QualityCheck.ANSWER_YES, expect=409)

    def test_a_finished_image_refuses_a_count_box(self):
        self._mark_finished()
        response = self.client.post(f"{self.base}/count-box", {}, format="json")
        self.assertEqual(response.status_code, 409, response.data)

    def test_a_finished_image_still_reports_what_was_already_checked(self):
        self._answer_all([QualityCheck.ANSWER_YES] * 3)
        self._mark_finished()
        payload = self._spot_check(3)
        self.assertEqual(payload["counts"]["positive"], 3)
        self.assertTrue(payload["locked"])

    def test_a_finished_image_does_not_draw_questions_nobody_can_answer(self):
        self._spot_check(3)
        self._mark_finished()
        payload = self._spot_check(12)
        self.assertEqual(len(payload["checks"]), 3)
        self.assertEqual(payload["drawing_refused"], "locked")


class RequestValidationTests(QualityFixture):
    n_objects = 4

    def test_a_sample_size_that_is_not_a_number_is_refused(self):
        response = self.client.get(f"{self.base}/spot-check/?n=twelve")
        self.assertEqual(response.status_code, 400, response.data)

    def test_a_sample_of_zero_is_refused(self):
        response = self.client.get(f"{self.base}/spot-check/?n=0")
        self.assertEqual(response.status_code, 400, response.data)

    def test_an_absurd_sample_size_is_refused_rather_than_written(self):
        response = self.client.get(f"{self.base}/spot-check/?n=100000")
        self.assertEqual(response.status_code, 400, response.data)
        self.assertEqual(QualityCheck.objects.count(), 0)

    def test_an_unknown_segmentation_is_a_404(self):
        missing = "11111111-1111-1111-1111-111111111111"
        for suffix in ("/spot-check/", "/count-box"):
            with self.subTest(suffix=suffix):
                response = self.client.get(f"/api/segmentations/{missing}{suffix}")
                self.assertEqual(response.status_code, 404)

    def test_an_answer_to_a_question_that_is_not_in_this_sample_is_a_404(self):
        self._spot_check(2)
        response = self.client.post(
            f"{self.base}/spot-check/answer",
            {
                "check_id": "22222222-2222-2222-2222-222222222222",
                "answer": QualityCheck.ANSWER_YES,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 404, response.data)


class CountingHelperTests(TestCase):
    """The tally, without a database in the way."""

    class _Check:
        def __init__(self, answer, segment=None):
            self.answer = answer
            self.segment = segment

    class _Segment:
        def __init__(self, source_model="quantem:mito"):
            self.source_model = source_model

    def test_an_unanswered_row_is_drawn_but_not_answered(self):
        counts = count_answers([self._Check(""), self._Check("")])
        self.assertEqual((counts.drawn, counts.answered, counts.scored), (2, 0, 0))

    def test_the_denominator_is_answered_minus_unsure(self):
        counts = count_answers(
            [
                self._Check(QualityCheck.ANSWER_YES),
                self._Check(QualityCheck.ANSWER_UNSURE),
                self._Check(QualityCheck.ANSWER_NOT_THE_THING),
                self._Check(""),
            ]
        )
        self.assertEqual(counts.drawn, 4)
        self.assertEqual(counts.answered, 3)
        self.assertEqual(counts.scored, 2)
        self.assertEqual(counts.positive, 1)

    def test_a_positive_whose_object_is_gone_cannot_be_attributed(self):
        caveat = self_confirmation(
            [
                self._Check(QualityCheck.ANSWER_YES, self._Segment()),
                self._Check(QualityCheck.ANSWER_YES, None),
            ]
        )
        self.assertEqual(caveat["n_positive"], 2)
        self.assertEqual(caveat["n_positive_unattributable"], 1)
        self.assertEqual(caveat["fraction"], 1.0)

    def test_agreeing_with_your_own_outlines_is_not_self_confirmation(self):
        caveat = self_confirmation(
            [
                self._Check(QualityCheck.ANSWER_YES, self._Segment("manual")),
                self._Check(QualityCheck.ANSWER_YES, self._Segment("manual")),
                self._Check(QualityCheck.ANSWER_YES, self._Segment()),
            ]
        )
        self.assertAlmostEqual(caveat["fraction"], 1 / 3)
        self.assertFalse(caveat["applies"])

    def test_the_seed_is_derived_and_not_random(self):
        self.assertEqual(derive_seed("a", 1), derive_seed("a", 1))
        self.assertNotEqual(derive_seed("a", 1), derive_seed("a", 2))
        self.assertGreaterEqual(derive_seed("a", 1), 0)

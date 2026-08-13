"""Two things the labeling screen could not know, and one it was told wrongly.

**The scale the objects were made at.** The header reads ``5 nm/px · entered by
hand`` beside an ordinary objects chip, over a set produced before that number
existed. A pack that declares a ``canonical_nm`` resamples the image to it
before inference, so an uncalibrated run produced a *different object set* --
not the same objects in the wrong units. Nothing on the screen where a user
decides the work is finished said so, and neither did the Analysis screen before
a run was spent: it surfaced only in the finished bundle, as blank micron
columns and ``calibrated: false``. ``objects_pixel_size`` is that fact, read off
the objects' own run stamps and served on the payload both screens already poll.

**A run that completed and did nothing.** ``get_run_notice`` returned ``None``
the moment any ``SegmentObject`` existed, and the proofread branch of
``_zero_object_advice`` needs ``labelled > 0`` -- so that branch was unreachable
by construction. A user with 12 confirmed objects ran the model,
got SUCCESS in four seconds with nothing new, read "Candidates ready", and
polled for two and a half minutes to be sure. The sentences that explained it,
including the only route that fixes it, were in ``job.result_json.next_steps``,
which nothing renders.

The two are the same defect twice: the app knew, and the screen did not.
"""

from __future__ import annotations

from datetime import timedelta

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.utils import timezone
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.analysis import loaders, service
from quantem.analysis.models import AnalysisRun
from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
)
from quantem.jobs.handlers import _segmentation_run_outcome
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.organelle_tasks import zero_object_outcome
from quantem.segmentation.run_identity import build_run_identity
from quantem.segmentation.serializers import ImageSegmentationSerializer
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_small_test_image

SIZE = 240
RUN_A = "11111111-1111-4111-8111-111111111111"
RUN_B = "22222222-2222-4222-8222-222222222222"


class _CountQueries(CaptureQueriesContext):
    """``assertNumQueries`` without having to know the number in advance."""

    def __init__(self):
        super().__init__(connection)

    @property
    def count(self) -> int:
        return len(self.captured_queries)


def _stamp(*, run_id: str = RUN_A, native_pixel_size_nm: float | None) -> dict:
    """One object's ``features["run"]``, built by the writer inference calls.

    Hand-assembling the dict here would let this file keep passing while the two
    halves of the contract drifted apart.
    """
    return build_run_identity(
        run_id=run_id,
        pack_id="quantem:mito",
        threshold=0.5,
        adapter_id=None,
        # A pack with a canonical_nm cannot resample without a pixel size, which
        # is the whole point: null here is a different object set, not a units
        # problem.
        ran_at_nm=None if native_pixel_size_nm is None else 8.0,
        native_pixel_size_nm=native_pixel_size_nm,
        min_area=60,
    )


class _SegmentationTestCase(TestCase):
    """One 5 nm/px image with one mitochondria segmentation."""

    pixel_size_nm: float | None = 5.0

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("Objects pixel size", width=SIZE, height=SIZE)
        self.asset = self.image.asset
        self.asset.pixel_size_nm = self.pixel_size_nm
        self.asset.save(update_fields=["pixel_size_nm"])
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )
        self._next_index = 0

    def _object(self, *, stamp=None, label_state="CONFIRMED", features=None):
        index = self._next_index
        self._next_index += 1
        x = 10 + 25 * (index % 8)
        y = 10 + 25 * (index // 8)
        polygon = Polygon(((x, y), (x + 20, y), (x + 20, y + 20), (x, y + 20), (x, y)))
        payload = {
            "area": polygon.area,
            "perimeter": polygon.length,
            "eccentricity": 0.25,
            "solidity": 0.98,
            "intensity_mean": 100.0,
        }
        if features is not None:
            payload.update(features)
        if stamp is not None:
            payload["run"] = stamp
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            source_model="quantem:mito",
            features=payload,
        )

    def _payload(self, segmentation=None):
        return ImageSegmentationSerializer(segmentation or self.segmentation).data

    def _objects_pixel_size(self, segmentation=None):
        return self._payload(segmentation)["objects_pixel_size"]

    def _finished_run(
        self,
        *,
        segment_count: int,
        job_type: str = JOB_TYPE_RUN_SEGMENTATION_FULL,
        finished_at=None,
    ) -> Job:
        """A SUCCESS job carrying the result the real handler would have written.

        ``_segmentation_run_outcome`` is the producer, so the serializer is read
        against the shape that actually reaches the database rather than against
        a literal this file made up.
        """
        _message, outcome = _segmentation_run_outcome(segment_count, segmentation=self.segmentation)
        return Job.objects.create(
            type=job_type,
            status="SUCCESS",
            payload_json={
                "segmentation_id": str(self.segmentation.id),
                "segmentation_type": "mitochondria",
                "source_model": "quantem:mito",
            },
            result_json={
                "segmentation_id": str(self.segmentation.id),
                **outcome,
            },
            finished_at=finished_at or timezone.now(),
        )


class ObjectsPixelSizeTests(_SegmentationTestCase):
    """What the stamps say, and nothing the stamps do not say."""

    def test_a_segmentation_with_no_objects_reports_nothing(self):
        """Every field would otherwise be a statement about no objects."""
        self.assertIsNone(self._objects_pixel_size())

    def test_it_reports_the_distinct_scales_the_runs_recorded(self):
        self._object(stamp=_stamp(native_pixel_size_nm=5.0))
        self._object(stamp=_stamp(native_pixel_size_nm=5.0))
        self._object(stamp=_stamp(run_id=RUN_B, native_pixel_size_nm=20.0))

        self.assertEqual(
            self._objects_pixel_size()["produced_nm"],
            [5.0, 20.0],
            "two runs at different scales are not one population",
        )

    def test_null_is_a_member_and_it_sorts_last(self):
        """A run with no pixel size is a recorded value, not a gap."""
        self._object(stamp=_stamp(native_pixel_size_nm=5.0))
        self._object(stamp=_stamp(run_id=RUN_B, native_pixel_size_nm=None))

        self.assertEqual(self._objects_pixel_size()["produced_nm"], [5.0, None])

    def test_it_says_the_objects_predate_the_pixel_size_the_image_shows(self):
        """The header's `5 nm/px` and these objects are about different runs."""
        self._object(stamp=_stamp(native_pixel_size_nm=None))

        block = self._objects_pixel_size()
        self.assertTrue(block["predates_calibration"])
        self.assertEqual(block["produced_nm"], [None])

    def test_a_calibrated_run_is_not_accused_of_anything(self):
        """The crying-wolf failure: this must not fire on the ordinary path."""
        self._object(stamp=_stamp(native_pixel_size_nm=5.0))

        self.assertFalse(self._objects_pixel_size()["predates_calibration"])

    def test_an_image_that_is_still_uncalibrated_has_nothing_to_predate(self):
        """No pixel size now means no false reassurance to correct."""
        self.asset.pixel_size_nm = None
        self.asset.save(update_fields=["pixel_size_nm"])
        self._object(stamp=_stamp(native_pixel_size_nm=None))

        block = self._objects_pixel_size()
        self.assertEqual(block["produced_nm"], [None])
        self.assertFalse(block["predates_calibration"])

    def test_a_pixel_size_that_is_not_a_length_does_not_count_as_one(self):
        """`0` and a negative are the same as unset to anything converting."""
        for bad in (0.0, -5.0):
            with self.subTest(pixel_size_nm=bad):
                self.asset.pixel_size_nm = bad
                self.asset.save(update_fields=["pixel_size_nm"])
                SegmentObject.objects.filter(segmentation=self.segmentation).delete()
                self._object(stamp=_stamp(native_pixel_size_nm=None))

                self.assertFalse(self._objects_pixel_size()["predates_calibration"])

    def test_a_hand_drawn_object_is_counted_not_guessed_about(self):
        """No stamp means "no model made this", not "made at an unknown scale".

        Folding it into ``produced_nm`` as a null would tell someone their own
        polygons predate the calibration and should be discarded.
        """
        self._object(stamp=None)
        self._object(stamp=None)
        self._object(stamp=_stamp(native_pixel_size_nm=5.0))

        block = self._objects_pixel_size()
        self.assertEqual(block["unstamped_count"], 2)
        self.assertEqual(block["produced_nm"], [5.0])
        self.assertFalse(block["predates_calibration"])

    def test_a_run_entry_with_no_id_is_unstamped_too(self):
        """The same test the analysis manifest applies: a stamp needs an id."""
        self._object(features={"run": {"native_pixel_size_nm": None}})

        block = self._objects_pixel_size()
        self.assertEqual(block["unstamped_count"], 1)
        self.assertEqual(block["produced_nm"], [])
        self.assertFalse(
            block["predates_calibration"],
            "a broken stamp is not evidence a run happened uncalibrated",
        )

    def test_a_damaged_stamp_does_not_take_the_endpoint_down_with_it(self):
        """``features`` is JSON out of the database and can hold anything.

        ``sorted`` over a mixture raises ``TypeError``, and this is a read a
        screen polls. The damage is reported as its own value rather than
        swallowed, and it is not counted as "ran with no pixel size" -- that is
        a specific finding, not a place to put whatever could not be parsed.
        """
        self._object(stamp=_stamp(native_pixel_size_nm=5.0))
        self._object(features={"run": {"id": RUN_B, "native_pixel_size_nm": "five"}})

        block = self._objects_pixel_size()
        self.assertEqual(block["produced_nm"], [5.0, "five"])
        self.assertFalse(block["predates_calibration"])

    def test_candidates_count_before_anyone_confirms_them(self):
        """This is the screen where the work is decided to be finished.

        Waiting for a confirmation would surface the warning only after the
        labelling it should have prevented.
        """
        self._object(stamp=_stamp(native_pixel_size_nm=None), label_state="CANDIDATE")

        self.assertTrue(self._objects_pixel_size()["predates_calibration"])

    def test_it_arrives_on_the_list_endpoint_both_screens_read(self):
        self._object(stamp=_stamp(native_pixel_size_nm=None))

        response = self.client.get(f"/api/assets/{self.asset.id}/segmentations/")

        self.assertEqual(response.status_code, 200, response.data)
        row = next(item for item in response.data if item["id"] == str(self.segmentation.id))
        self.assertTrue(row["objects_pixel_size"]["predates_calibration"])

    def _field_queries(self, *, join_asset: bool = True) -> int:
        related = ["segmentation_type", "config"]
        if join_asset:
            related.append("asset")
        instance = ImageSegmentation.objects.select_related(*related).get(id=self.segmentation.id)
        with _CountQueries() as counted:
            ImageSegmentationSerializer().get_objects_pixel_size(instance)
        return counted.count

    def _fill(self, count: int) -> None:
        for index in range(count):
            self._object(
                stamp=_stamp(
                    run_id=RUN_A if index % 2 else RUN_B,
                    native_pixel_size_nm=5.0 if index % 2 else None,
                )
            )

    def test_it_is_one_query_however_many_objects_there_are(self):
        """It runs per segmentation in a list the labeling screen polls."""
        self._fill(3)
        self.assertEqual(self._field_queries(), 1)

        self._fill(120)
        self.assertEqual(
            self._field_queries(),
            1,
            "the object rows are grouped in the database, not in Python",
        )

    def test_the_asset_is_joined_rather_than_fetched_per_row(self):
        """Why ``AssetSegmentationListCreateView.get`` select_relateds ``asset``.

        ``predates_calibration`` compares the stamps with the image's pixel size
        now, and reaching for it lazily is one extra query per row of a polled
        list.
        """
        self._fill(3)
        self.assertEqual(self._field_queries(join_asset=False), 2)
        self.assertEqual(self._field_queries(join_asset=True), 1)

    def test_the_endpoint_does_not_slow_down_with_the_object_count(self):
        url = f"/api/assets/{self.asset.id}/segmentations/"
        self._fill(3)
        with _CountQueries() as few:
            self.client.get(url)

        self._fill(120)
        with _CountQueries() as many:
            self.client.get(url)

        self.assertEqual(many.count, few.count)


class ScreenAndBundleCannotDisagreeTests(_SegmentationTestCase):
    """The predicate is shared, and this is what sharing it has to buy.

    ``predates_calibration`` is
    :func:`quantem.segmentation.run_identity.calibrated_after_the_fact`, which is
    also what ``run_analysis`` blanks every physical unit on. Restating it in the
    serializer would have let the labeling screen call a set of objects fine
    while the bundle refused to convert them, and nobody sees that until they
    put a screen and an export side by side.
    """

    def _analysis_result(self) -> dict:
        run = AnalysisRun.objects.create(
            segmentation=self.segmentation,
            params=loaders.normalise_params(
                {"compartments": {"mito": str(self.segmentation.id)}},
                segmentation=self.segmentation,
            ),
        )
        loaded = loaders.load_inputs(run)
        inputs = loaded.inputs if hasattr(loaded, "inputs") else loaded
        return service.run_analysis(inputs)

    def test_a_screen_that_says_predates_gets_a_bundle_that_refuses_to_convert(self):
        for _ in range(3):
            self._object(stamp=_stamp(native_pixel_size_nm=None))

        self.assertTrue(self._objects_pixel_size()["predates_calibration"])
        self.assertFalse(
            self._analysis_result()["calibrated"],
            "the screen warned and the bundle converted anyway",
        )

    def test_a_screen_that_says_nothing_gets_a_bundle_that_converts(self):
        for _ in range(3):
            self._object(stamp=_stamp(native_pixel_size_nm=5.0))

        self.assertFalse(self._objects_pixel_size()["predates_calibration"])
        self.assertTrue(
            self._analysis_result()["calibrated"],
            "the screen was silent and the bundle blanked every micron column",
        )


class RunNoticeReachesAProofreadImageTests(_SegmentationTestCase):
    """The branch that existed, was correct, and could not be reached."""

    def _notice(self, segmentation=None):
        return self._payload(segmentation)["run_notice"]

    def _labelled(self, count: int, *, native_pixel_size_nm=None):
        for _ in range(count):
            self._object(stamp=_stamp(native_pixel_size_nm=native_pixel_size_nm))

    def test_twelve_confirmed_objects_and_a_run_that_added_none_says_so(self):
        self._labelled(12)
        self._finished_run(segment_count=0)

        notice = self._notice()

        self.assertIsNotNone(notice, '"Candidates ready" was the whole story again')
        self.assertEqual(notice["kind"], "no_new_objects")
        self.assertIn("added no new objects", notice["message"])
        self.assertIn("12 object(s)", notice["message"])

    def test_it_carries_the_route_that_works_and_no_advice_to_lower_a_threshold(self):
        """The remedy the bundle recommended cannot work; this names the one that
        can."""
        self._labelled(12)
        self._finished_run(segment_count=0)

        steps = " ".join(self._notice()["next_steps"])

        # It named ``POST /api/segmentations/<id>/labels/clear``; the button
        # that does it is "Discard objects and re-run..." on the labeling
        # header, and I-12 forbids the route (api-endpoint) in copy.
        self.assertIn("Discard objects and re-run", steps)
        self.assertNotIn("/api/", steps)
        self.assertIn("A re-run cannot replace them", steps)
        self.assertNotIn("Lower the detection threshold", steps)

    def test_it_is_the_same_advice_that_reached_only_the_job_log(self):
        self._labelled(12)
        self._finished_run(segment_count=0)

        self.assertEqual(
            self._notice()["next_steps"],
            zero_object_outcome(self.segmentation)[1],
        )

    def test_the_chip_line_does_not_claim_there_are_no_objects(self):
        """ "Ran and found no objects" over twelve confirmed ones is a new lie."""
        self._labelled(12)
        self._finished_run(segment_count=0)

        self.assertEqual(self._notice()["summary"], "Ran and added no new objects")

    def test_a_run_that_produced_something_says_nothing(self):
        self._labelled(12)
        self._finished_run(segment_count=7)

        self.assertIsNone(self._notice())

    def test_the_most_recent_run_is_the_one_reported(self):
        """A later productive run clears it with nothing having to remember to."""
        self._labelled(12)
        earlier = timezone.now() - timedelta(hours=1)
        self._finished_run(segment_count=0, finished_at=earlier)
        self._finished_run(segment_count=7)

        self.assertIsNone(self._notice())

    def test_an_roi_run_counts_the_same_as_a_full_one(self):
        self._labelled(12)
        self._finished_run(segment_count=0, job_type=JOB_TYPE_RUN_SEGMENTATION_ROI)

        self.assertIsNotNone(self._notice())

    def test_labelled_objects_with_no_run_behind_them_say_nothing(self):
        """Objects can arrive without a job row; there is then no run to report."""
        self._labelled(12)

        self.assertIsNone(self._notice())

    def test_unlabelled_candidates_are_not_the_proofread_case(self):
        """Nothing here suppresses anything, so there is no benign explanation.

        The empty-run wording -- "this run finished without finding any objects"
        -- would be false over the candidates on screen, so the notice is
        withheld rather than guessed at.
        """
        for _ in range(12):
            self._object(stamp=_stamp(native_pixel_size_nm=5.0), label_state="CANDIDATE")
        self._finished_run(segment_count=0)

        self.assertIsNone(self._notice())

    def test_an_excluded_object_is_labelling_too(self):
        """EXCLUDED suppresses a candidate exactly as CONFIRMED does."""
        self._object(stamp=_stamp(native_pixel_size_nm=None), label_state="EXCLUDED")
        self._finished_run(segment_count=0)

        self.assertEqual(self._notice()["kind"], "no_new_objects")

    def test_a_locked_segmentation_is_not_second_guessed(self):
        self._labelled(12)
        self._finished_run(segment_count=0)
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save(update_fields=["status_stage"])

        self.assertIsNone(self._notice())

    def test_a_manual_only_type_is_still_never_told_to_lower_a_threshold(self):
        tissue = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_tissue_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )
        self.assertIsNone(self._notice(tissue))

    def test_the_empty_run_notice_is_unchanged(self):
        """The branch that already worked keeps its wording and its chip line."""
        self._finished_run(segment_count=0)
        notice = self._notice()

        self.assertEqual(notice["kind"], "no_objects")
        self.assertEqual(notice["source_model"], "quantem:mito")
        self.assertEqual(notice["summary"], "Ran and found no objects")
        self.assertIn("without finding any objects", notice["message"])

    def test_it_arrives_on_the_list_endpoint_the_screens_read(self):
        self._labelled(12)
        self._finished_run(segment_count=0)

        response = self.client.get(f"/api/assets/{self.asset.id}/segmentations/")

        self.assertEqual(response.status_code, 200, response.data)
        row = next(item for item in response.data if item["id"] == str(self.segmentation.id))
        self.assertEqual(row["run_notice"]["kind"], "no_new_objects")

"""The include-level dial, from the request to the objects it produces.

``test_threshold_replay`` already proves the hard part -- that re-thresholding
the stored map gives *exactly* the objects a fresh run at that level would give.
These are about the route and the worker on top of it: that asking for a level
re-derives the objects without loading a model, that the new set is numbered as
its own result carrying the level, that the level is written where every screen
reads it, and that the ways it can refuse say the right thing.

The refusals get as much room as the success, because they are the whole
difference between a control that fails under the user's hand and one that is
greyed out with the reason beside it. In particular the two unavailable-map
cases must not collapse into one sentence: a map that was never stored is fixed
for good by running once, and a map from an older build will go on being refused
until it is replaced, and a user told the same thing either way cannot tell
which they are in.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from django.test import TestCase
from django.urls import reverse
from shapely.geometry import box

from quantem.assets.utils import create_roi_image_from_image
from quantem.inference import resample
from quantem.inference.segmenter import DL_MODEL_NAME, DinoMitoSegmenter
from quantem.inference.specs import get_model_spec
from quantem.jobs.constants import (
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
    QUEUE_P1_INTERACTIVE,
)
from quantem.jobs.handlers.rethreshold import handle_reextract_at_include_level
from quantem.jobs.models import Job
from quantem.jobs.registry import _HANDLERS
from quantem.jobs.reporter import CancelToken, JobReporter
from quantem.seg_core.db.inference import StoredMapUnavailable
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    ProbabilityMap,
    RoiSegmentationStatus,
    SegmentationConfig,
    SegmentationResultVersion,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import (
    run_segmentation_full_task,
    run_segmentation_roi_task,
)
from quantem.segmentation.prob_maps.persistence import (
    LEGACY_MAP_MESSAGE,
    NO_STORED_MAP_MESSAGE,
    REPLAY_FROM_OLDER_BUILD,
    REPLAY_NOT_STORED,
    REPLAY_READY,
    stored_map_readiness,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256
PIXEL_SIZE_NM = 5.0
MITO_INTERNAL_NAME = "quantem_internal_mito"
SOURCE_MODEL = "quantem:mito"


def _model_field(shape: tuple[int, int]) -> np.ndarray:
    """Blobs of different confidence, so the object count moves with the dial."""
    height, width = shape
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    field = np.zeros(shape, dtype=np.float32)
    for row_frac, col_frac, sigma, peak in (
        (0.30, 0.30, 16.0, 0.97),
        (0.30, 0.70, 13.0, 0.83),
        (0.70, 0.30, 12.0, 0.72),
        (0.70, 0.70, 11.0, 0.58),
        (0.50, 0.50, 9.0, 0.45),
    ):
        squared = ((rows - row_frac * height) ** 2 + (cols - col_frac * width) ** 2) / (
            2.0 * sigma * sigma
        )
        field = np.maximum(field, peak * np.exp(-squared))
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def _fake_engine():
    spec = get_model_spec("quantem", "mito")

    def predict_region(_model, image, *, pixel_size_nm=None, **_kwargs):
        context = resample.plan_resample(image.shape[:2], pixel_size_nm, spec.canonical_nm)
        return SimpleNamespace(prob=_model_field(context.model_shape), context=context, plan=None)

    return SimpleNamespace(
        load_model=lambda *_a, **_k: SimpleNamespace(device="cpu"),
        load_adapted_model=lambda *_a, **_k: SimpleNamespace(device="cpu"),
        predict_region=predict_region,
        estimate_tiles=lambda *_a, **_k: 1,
    )


def _segmenter(threshold: float = 0.5) -> DinoMitoSegmenter:
    return DinoMitoSegmenter(
        source_model=SOURCE_MODEL,
        fg_threshold=threshold,
        pixel_size_nm=PIXEL_SIZE_NM,
    )


class IncludeLevelDialTestCase(TestCase):
    def setUp(self):
        self.image = create_small_test_image("Dial", width=SIZE, height=SIZE, textured=True)
        asset = self.image.asset
        asset.pixel_size_nm = PIXEL_SIZE_NM
        asset.save(update_fields=["pixel_size_nm"])
        self.segmentation = ImageSegmentation.objects.create(
            asset=asset, segmentation_type=get_or_create_mitochondria_type()
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)
        self.url = reverse("segmentation-include-level", args=[str(self.segmentation.id)])
        self.confirm_url = reverse(
            "segmentation-confirm-model-output",
            args=[str(self.segmentation.id)],
        )

    def run_the_model(self, threshold: float = 0.5) -> int:
        segmenter = _segmenter(threshold)
        with (
            patch("quantem.inference.segmenter.engine", _fake_engine()),
            patch(
                "quantem.segmentation.organelle_tasks.get_segmenter",
                return_value=segmenter,
            ),
        ):
            return run_segmentation_full_task(
                segmentation_id=str(self.segmentation.id),
                segmentation_type=MITO_INTERNAL_NAME,
                source_model=SOURCE_MODEL,
            )

    def run_the_model_on_roi(self):
        roi = create_roi_image_from_image(
            self.image,
            x=24,
            y=32,
            width=128,
            height=112,
            source="MANUAL",
            is_active=True,
        )
        segmenter = _segmenter()
        with (
            patch("quantem.inference.segmenter.engine", _fake_engine()),
            patch(
                "quantem.segmentation.organelle_tasks.get_segmenter",
                return_value=segmenter,
            ),
        ):
            run_segmentation_roi_task(
                segmentation_id=str(self.segmentation.id),
                segmentation_type=MITO_INTERNAL_NAME,
                roi_id=str(roi.id),
                source_model=SOURCE_MODEL,
            )
        return roi

    def work_the_job(self, include_level: float) -> dict:
        """Run the handler the way the worker runs it."""
        job = Job.enqueue(
            job_type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
            payload={
                "segmentation_id": str(self.segmentation.id),
                "segmentation_type": MITO_INTERNAL_NAME,
                "include_level": include_level,
                "source_model": SOURCE_MODEL,
            },
            priority="high",
            resource_class="cpu",
            queue_name=QUEUE_P1_INTERACTIVE,
            max_attempts=1,
        )
        return self.dispatch(job)

    def dispatch(self, job: Job) -> dict:
        """Call the handler, then give the thread its reporter back.

        ``JobReporter.__init__`` registers itself as *this thread's* active
        reporter and nothing un-registers it. In a worker that is correct -- the
        thread belongs to the job for its whole life. In a test it outlives the
        transaction that created the job row, so the next test on this thread to
        drive real inference finds a live reporter pointing at a job that has
        been rolled back, writes its tile counts and log lines against that id,
        and dies at teardown with a foreign key violation in a test that has
        nothing to do with this one. ``deactivate`` exists for exactly this and
        says so.
        """
        reporter = JobReporter(str(job.id))
        try:
            return handle_reextract_at_include_level(
                job.payload_json, reporter, CancelToken(str(job.id))
            )
        finally:
            reporter.deactivate()

    def live_candidates(self) -> int:
        return SegmentObject.objects.filter(
            segmentation=self.segmentation,
            label_state="CANDIDATE",
            superseded_at__isnull=True,
        ).count()

    def candidate(
        self,
        bounds: tuple[float, float, float, float],
        *,
        source_model: str = SOURCE_MODEL,
    ) -> SegmentObject:
        geometry = box(*bounds)
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=geometry,
            centroid=geometry.centroid,
            bbox=geometry.envelope,
            label_state="CANDIDATE",
            source_model=source_model,
            confidence_score=0.8,
            features={"source_model": source_model},
        )


class TheWorkerTests(IncludeLevelDialTestCase):
    def test_a_registered_handler_is_what_makes_the_job_enqueueable(self):
        """Until one existed the serializer refused the type at the door."""
        assert JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL in _HANDLERS

    def test_moving_the_dial_re_derives_the_objects_without_running_the_model(self):
        """The point of the dial: no forward pass, and the count moves.

        ``engine`` is left unpatched, so any attempt to resolve a pack raises
        rather than quietly taking thirty seconds -- the same guard
        ``test_threshold_replay`` uses.
        """
        self.run_the_model(0.5)
        at_half = self.live_candidates()
        assert at_half == 0, "running the model must not create candidates"

        with patch.object(DinoMitoSegmenter, "load_models", side_effect=AssertionError("loaded")):
            outcome = self.work_the_job(0.25)

        assert outcome["segment_count"] == self.live_candidates()
        assert outcome["include_level"] == 0.25
        assert self.live_candidates() > at_half

    def test_the_new_object_set_is_its_own_numbered_result(self):
        self.run_the_model(0.5)
        self.work_the_job(0.5)
        first = SegmentationResultVersion.current_version_for(self.segmentation)

        self.work_the_job(0.3)

        latest = (
            SegmentationResultVersion.objects.filter(segmentation=self.segmentation)
            .order_by("-version")
            .first()
        )
        assert latest is not None
        assert latest.version > first
        assert latest.include_level == 0.3, (
            "the numbered result did not record the level it was extracted at"
        )

    def test_a_model_run_records_no_level_at_all(self):
        """``None`` is not 0.5. Nobody moved a dial, so none is shown.

        The guard on the test above: if a plain run wrote its own threshold into
        ``include_level``, every image would show a dial position its user never
        chose, and the field would stop meaning anything.
        """
        self.run_the_model(0.5)

        assert not SegmentationResultVersion.objects.filter(
            segmentation=self.segmentation
        ).exists(), "a preview is not yet a candidate result"
        self.segmentation.refresh_from_db()
        assert self.segmentation.include_level is None
        assert self.segmentation.status_stage == "THRESHOLD_READY"

    def test_the_level_is_written_where_every_screen_reads_it(self):
        self.run_the_model(0.5)
        self.work_the_job(0.42)

        self.segmentation.refresh_from_db()
        assert self.segmentation.include_level == 0.42

    def test_the_objects_carry_the_level_in_their_own_provenance(self):
        self.run_the_model(0.5)
        self.work_the_job(0.35)

        stamps = [
            (segment.features or {}).get("run") or {}
            for segment in SegmentObject.objects.filter(
                segmentation=self.segmentation,
                label_state="CANDIDATE",
                superseded_at__isnull=True,
            )
        ]
        assert stamps
        for stamp in stamps:
            assert stamp["include_level"] == 0.35
            assert stamp["scope"] == "full"

    def test_a_level_outside_the_range_is_refused_by_the_worker_too(self):
        """The serializer guards the door; the payload can also be written here."""
        self.run_the_model(0.5)
        job = Job.enqueue(
            job_type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
            payload={
                "segmentation_id": str(self.segmentation.id),
                "include_level": 4.0,
            },
            priority="high",
            resource_class="cpu",
            queue_name=QUEUE_P1_INTERACTIVE,
            max_attempts=1,
        )
        with self.assertRaises(ValueError) as caught:
            self.dispatch(job)
        assert "between 0 and 1" in str(caught.exception)


class WhenTheDialCannotMoveTests(IncludeLevelDialTestCase):
    """Two ways the map is unusable, and they must not read the same."""

    def test_a_map_that_was_never_stored_says_running_once_fixes_it(self):
        assert not ProbabilityMap.objects.filter(segmentation=self.segmentation).exists()

        readiness = stored_map_readiness(
            segmentation=self.segmentation,
            segmenter=_segmenter(),
            model_name=DL_MODEL_NAME,
        )

        assert readiness.status == REPLAY_NOT_STORED
        assert readiness.detail == NO_STORED_MAP_MESSAGE
        assert "from then on" in readiness.detail, (
            "the sentence does not tell the user this is a one-off"
        )

    def test_a_map_from_an_older_build_says_the_stored_result_is_the_problem(self):
        self.run_the_model(0.5)
        record = ProbabilityMap.objects.get(segmentation=self.segmentation)
        # What an upgraded install looks like: real bytes, and a row whose
        # provenance markers are not the ones this build writes.
        record.metadata = {"native_coordinates": True}
        record.save(update_fields=["metadata"])

        readiness = stored_map_readiness(
            segmentation=self.segmentation,
            segmenter=_segmenter(),
            model_name=DL_MODEL_NAME,
        )

        assert readiness.status == REPLAY_FROM_OLDER_BUILD
        assert readiness.detail == LEGACY_MAP_MESSAGE
        assert "earlier version" in readiness.detail

    def test_the_two_sentences_are_different_sentences(self):
        """The one assertion that would catch them being merged."""
        assert NO_STORED_MAP_MESSAGE != LEGACY_MAP_MESSAGE

    def test_the_readiness_check_agrees_with_the_replay_it_guards(self):
        """A "yes" here and a failure there could only ever be a race.

        Two implementations of one rule is how a control comes to be offered for
        work that cannot be done, so the cheap check and the real load are
        pinned to each other rather than merely written to match.
        """
        segmenter = _segmenter()
        assert (
            stored_map_readiness(
                segmentation=self.segmentation,
                segmenter=segmenter,
                model_name=DL_MODEL_NAME,
            ).status
            == REPLAY_NOT_STORED
        )
        with self.assertRaises(StoredMapUnavailable):
            from quantem.seg_core.db.inference import replay_stored_probability_map

            replay_stored_probability_map(segmenter, self.segmentation, threshold=0.5)

        self.run_the_model(0.5)
        assert (
            stored_map_readiness(
                segmentation=self.segmentation,
                segmenter=_segmenter(),
                model_name=DL_MODEL_NAME,
            ).status
            == REPLAY_READY
        )

    def test_an_unusable_map_carries_the_code_the_client_puts_a_button_behind(self):
        """The catalogue named this class and nothing inherited it.

        ``classify_exception`` walks exception class names, and
        ``ProbabilityMapMissing`` was in the table matching nothing -- so the one
        failure with a "Run inference again" control reached the client uncoded
        and rendered as red text with no way forward.
        """
        from quantem.core.error_codes import ErrorCode, classify_exception

        assert (
            classify_exception(StoredMapUnavailable("no map")) is ErrorCode.PROBABILITY_MAP_MISSING
        )


class TheRouteTests(IncludeLevelDialTestCase):
    def test_the_route_exists_and_reports_where_the_dial_is(self):
        self.run_the_model(0.5)

        response = self.client.get(self.url)

        assert response.status_code == 200
        body = response.json()
        assert body["include_level"] is None  # nobody has moved it
        assert body["can_move"] is True
        assert body["detail"] == ""
        assert body["minimum"] == 0.0
        assert body["maximum"] == 1.0
        assert body["object_count"] == 0
        assert body["measurement_mode"] == "objects"
        assert body["candidate_count"] == 0
        assert body["confirmable_candidate_count"] == 0
        assert body["manual_roi_candidate_count"] == 0
        assert body["confirmed_model_count"] == 0
        assert body["preview_url"].endswith("/include-level/map")
        assert body["preview_bounds"] == [0, 0, SIZE, SIZE]

    def test_the_preview_route_serves_the_saved_map_without_caching_it(self):
        self.run_the_model(0.5)
        response = self.client.get(
            reverse(
                "segmentation-include-level-map",
                args=[str(self.segmentation.id)],
            ),
            {"source_model": SOURCE_MODEL},
        )

        assert response.status_code == 200
        assert response["Content-Type"] == "image/png"
        assert response["Cache-Control"] == "no-store"
        assert b"".join(response.streaming_content).startswith(b"\x89PNG")

    def test_an_roi_test_exposes_only_that_rois_preview_and_bounds(self):
        roi = self.run_the_model_on_roi()

        response = self.client.get(
            self.url,
            {"source_model": SOURCE_MODEL, "roi_id": str(roi.id)},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["can_move"] is True
        assert body["preview_bounds"] == [24, 32, 128, 112]
        assert f"roi_id={roi.id}" in body["preview_url"]

        preview = self.client.get(body["preview_url"])
        assert preview.status_code == 200
        assert preview["Content-Type"] == "image/png"
        assert b"".join(preview.streaming_content).startswith(b"\x89PNG")

    def test_an_roi_include_level_job_keeps_the_roi_scope(self):
        roi = self.run_the_model_on_roi()

        response = self.client.post(
            self.url,
            {
                "include_level": 0.3,
                "source_model": SOURCE_MODEL,
                "roi_id": str(roi.id),
            },
            content_type="application/json",
        )

        assert response.status_code == 202
        job = Job.objects.get(id=response.json()["job_id"])
        assert job.payload_json["roi_id"] == str(roi.id)

    def test_asking_for_a_level_queues_one_job_and_says_so(self):
        self.run_the_model(0.5)

        response = self.client.post(
            self.url, {"include_level": 0.3}, content_type="application/json"
        )

        assert response.status_code == 202
        body = response.json()
        assert body["include_level"] == 0.3
        job = Job.objects.get(id=body["job_id"])
        assert job.type == JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL
        assert job.queue_name == QUEUE_P1_INTERACTIVE
        assert job.max_attempts == 1
        # The reconcilers read this exact key to release an image whose worker
        # died; without it a failed dial move leaves the image showing as busy.
        assert job.payload_json["segmentation_id"] == str(self.segmentation.id)

    def test_a_level_outside_the_range_is_refused_before_anything_is_queued(self):
        self.run_the_model(0.5)

        for level in (-0.1, 1.4):
            with self.subTest(level=level):
                response = self.client.post(
                    self.url, {"include_level": level}, content_type="application/json"
                )
                assert response.status_code == 400
                assert not Job.objects.filter(type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL).exists()

    def test_no_stored_map_is_refused_at_the_door_rather_than_queued_to_fail(self):
        """A task that is certain to go red is worse than a refusal.

        The user would wait, watch it fail, and read the reason a minute after
        the moment they could have acted on it.
        """
        response = self.client.post(
            self.url, {"include_level": 0.3}, content_type="application/json"
        )

        assert response.status_code == 409
        body = response.json()
        assert body["detail"] == NO_STORED_MAP_MESSAGE
        assert body["error_code"] == "probability_map_missing"
        assert not Job.objects.filter(type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL).exists()

    def test_a_map_from_an_older_build_is_refused_with_its_own_sentence(self):
        self.run_the_model(0.5)
        record = ProbabilityMap.objects.get(segmentation=self.segmentation)
        record.metadata = {"native_coordinates": True}
        record.save(update_fields=["metadata"])

        response = self.client.post(
            self.url, {"include_level": 0.3}, content_type="application/json"
        )

        assert response.status_code == 409
        assert response.json()["detail"] == LEGACY_MAP_MESSAGE

    def test_the_greyed_out_control_shows_the_same_sentence_the_refusal_would(self):
        """One check behind both, or the control looks usable and then is not."""
        get_body = self.client.get(self.url).json()
        post_body = self.client.post(
            self.url, {"include_level": 0.3}, content_type="application/json"
        ).json()

        assert get_body["can_move"] is False
        assert get_body["detail"] == post_body["detail"]

    def test_a_run_already_working_on_the_image_blocks_the_dial(self):
        self.run_the_model(0.5)
        Job.objects.create(
            type="run_segmentation_full_task",
            status="RUNNING",
            payload_json={"segmentation_id": str(self.segmentation.id)},
            queue_name="p4_full",
            resource_class="gpu",
        )

        response = self.client.post(
            self.url, {"include_level": 0.3}, content_type="application/json"
        )

        assert response.status_code == 409
        assert "already running" in response.json()["detail"]

    def test_the_level_reaches_the_segmentation_payload_every_screen_reads(self):
        self.run_the_model(0.5)
        self.work_the_job(0.42)

        detail = self.client.get(
            reverse("segmentation-detail", args=[str(self.segmentation.id)])
        ).json()

        assert detail["include_level"] == 0.42


class ConfirmWholeModelOutputTests(IncludeLevelDialTestCase):
    def post_confirm(self, source_model: str = SOURCE_MODEL):
        return self.client.post(
            self.confirm_url,
            {"source_model": source_model},
            content_type="application/json",
        )

    def test_confirm_accepts_every_current_candidate_from_the_selected_model(self):
        first = self.candidate((10, 10, 30, 30))
        second = self.candidate((50, 50, 75, 75))
        other_model = self.candidate((90, 90, 110, 110), source_model="omniem:mito")

        response = self.post_confirm()

        assert response.status_code == 200
        assert response.json()["confirmed_count"] == 2
        assert response.json()["skipped_manual_roi_count"] == 0
        first.refresh_from_db()
        second.refresh_from_db()
        other_model.refresh_from_db()
        assert first.label_state == "CONFIRMED"
        assert second.label_state == "CONFIRMED"
        assert other_model.label_state == "CANDIDATE", (
            "confirming QuantEM must not also accept OmniEM's comparison output"
        )

    def test_confirm_leaves_candidates_in_both_manual_roi_representations_unchanged(self):
        freeform_candidate = self.candidate((20, 20, 35, 35))
        rectangular_candidate = self.candidate((120, 120, 140, 140))
        outside_candidate = self.candidate((200, 200, 220, 220))
        CompletedROI.objects.create(
            segmentation=self.segmentation,
            geometry=box(10, 10, 50, 50),
        )
        reviewed_roi = create_roi_image_from_image(
            self.image,
            x=100,
            y=100,
            width=64,
            height=64,
            source="MANUAL",
            is_active=False,
        )
        RoiSegmentationStatus.objects.create(
            image_roi=reviewed_roi,
            segmentation=self.segmentation,
            is_complete=True,
        )

        dial = self.client.get(self.url, {"source_model": SOURCE_MODEL}).json()
        assert dial["candidate_count"] == 3
        assert dial["confirmable_candidate_count"] == 1
        assert dial["manual_roi_candidate_count"] == 2
        assert dial["manual_roi_count"] == 2

        response = self.post_confirm()

        assert response.status_code == 200
        assert response.json()["confirmed_count"] == 1
        assert response.json()["skipped_manual_roi_count"] == 2
        freeform_candidate.refresh_from_db()
        rectangular_candidate.refresh_from_db()
        outside_candidate.refresh_from_db()
        assert freeform_candidate.label_state == "CANDIDATE"
        assert rectangular_candidate.label_state == "CANDIDATE"
        assert outside_candidate.label_state == "CONFIRMED"

    def test_a_completed_segmentation_refuses_bulk_confirmation(self):
        self.candidate((10, 10, 30, 30))
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save(update_fields=["status_stage"])

        response = self.post_confirm()

        assert response.status_code == 409
        assert "Unlock it first" in response.json()["detail"]

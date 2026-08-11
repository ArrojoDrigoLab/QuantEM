"""Invariant I-12 over the queue's own copy, from the endpoint the drawer polls.

The Tasks drawer renders ``Job.message`` verbatim. A verifier reading 84 848
characters of that drawer during real runs found two strings that no user
should ever be handed::

    failed: ModelWeightsNotInstalled: Model pack 'quantem:er' is not installed.
    Install it on the Models screen. ...
    failed: ValueError: Error decoding PNG to 8-bit grayscale: image file is
    truncated

The sentences after the second colon are the app's own copy and are fine. The
``failed: <ClassName>:`` in front of them is a Python class name, which the
invariant forbids and which tells a biologist nothing -- the row is already
badged FAILED.

The detector is :mod:`quantem.registry.tests.copy_gate`, reused rather than
re-written: it is the module that knows what a violation looks like, and a
second opinion about that is how the first gate came to be blind. This file is
the half that enumerates **the jobs app's** surfaces and serialises them for
real.
"""

from __future__ import annotations

from uuid import uuid4

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.assets.models import Asset, Rendition
from quantem.jobs.constants import (
    JOB_TYPE_INSTALL_MODEL_PACK,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    QUEUE_P1_INTERACTIVE,
    QUEUE_P4_FULL,
)
from quantem.jobs.failure_reconcile import (
    UNEXPLAINED_FAILURE_MESSAGE,
    failure_message,
)
from quantem.jobs.models import UNIT_TILE, Job
from quantem.jobs.serializers import JobSerializer
from quantem.registry.tests.copy_gate import find_violations, walk_strings
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.type_service import get_or_create_mitochondria_type

#: The one field on ``GET /api/jobs/<id>/`` this gate does not read, and why.
#:
#: ``error_traceback`` is a Python traceback: module paths, absolute paths and
#: the exception class, by definition. It is on the wire because a traceback is
#: the only thing that identifies a crash in a bug report. It is **not** copy,
#: and the fix if a screen renders it is to render ``message`` instead -- which
#: is now a sentence for exactly that reason. Named here rather than skipped
#: quietly so that a second exemption cannot be added without editing this line.
DIAGNOSTIC_FIELDS: frozenset[str] = frozenset({"error_traceback"})


class _ModelWeightsNotInstalled(Exception):
    """Stands in for the model layer's own exception, by class name."""


def _asset() -> Asset:
    asset = Asset.objects.create(
        display_name="montage16real",
        original_filename="montage16real.png",
        logical_width=2892,
        logical_height=2508,
        channels=1,
        bit_depth=8,
        pixel_size_nm=8.0,
        preprocess_stage="DONE",
    )
    # The rendition is what inference opens, so it is also what the tiling plan
    # is measured from. An imported image always has one by the time a run can
    # be started on it.
    Rendition.objects.create(
        asset=asset,
        type=Rendition.TYPE_FULL,
        storage_root="DATA_DIR",
        stored_path=f"images/montage_{asset.id}.png",
        path_exists=False,
        is_directory=False,
        stored_width=2892,
        stored_height=2508,
        stored_channels=1,
        stored_bit_depth=8,
    )
    return asset


def _assert_clean(pairs, surface: str) -> None:
    violations = [v for where, text in pairs for v in find_violations(text, where)]
    if violations:
        report = "\n".join(f"  {v}" for v in violations)
        raise AssertionError(
            f"I-12: {len(violations)} defect(s) in {surface}:\n{report}"
        )


class TheGateWouldHaveCaughtWhatShipped(TestCase):
    """If this ever stops failing the detector has been weakened."""

    def test_the_two_strings_the_verifier_read_are_violations(self):
        for shipped in (
            "failed: ModelWeightsNotInstalled: Model pack 'quantem:er' is not "
            "installed. Install it on the Models screen.",
            "failed: ValueError: Error decoding PNG to 8-bit grayscale: image "
            "file is truncated",
        ):
            kinds = {v.kind for v in find_violations(shipped)}
            assert "exception-class" in kinds, (shipped, kinds)

    def test_the_message_the_queue_writes_now_is_not_one(self):
        exc = _ModelWeightsNotInstalled(
            "Model pack 'quantem:er' is not installed. Install it on the "
            "Models screen."
        )
        assert find_violations(failure_message(exc)) == []
        assert "ModelWeightsNotInstalled" not in failure_message(exc)

    def test_an_exception_with_nothing_sayable_in_it_is_replaced(self):
        assert failure_message(KeyError("asset_id")) == UNEXPLAINED_FAILURE_MESSAGE
        assert failure_message(RuntimeError("")) == UNEXPLAINED_FAILURE_MESSAGE
        assert find_violations(UNEXPLAINED_FAILURE_MESSAGE) == []

    def test_a_real_sentence_is_kept_word_for_word(self):
        spoken = "The image could not be read: it stops part way through the file."
        assert failure_message(ValueError(spoken)) == spoken


class TheQueueEndpointsAreClean(TestCase):
    """Real requests, real bodies, every string in them checked."""

    def setUp(self):
        self.client = APIClient()
        self.asset = _asset()
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        self.run = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload={
                "segmentation_id": str(self.segmentation.id),
                "segmentation_type": "mitochondria",
                "asset_id": str(self.asset.id),
                "source_model": "quantem:mito",
            },
            resource_class="gpu",
            queue_name=QUEUE_P4_FULL,
        )

    def _queue_status(self) -> dict:
        response = self.client.get("/api/jobs/queue-status/")
        assert response.status_code == 200
        return response.json()

    def test_a_wave_with_a_failure_a_cancellation_and_a_run_in_flight(self):
        Job.objects.filter(id=self.run.id).update(
            status="RUNNING",
            progress_units_done=19,
            message="Segmenting: 34% (19 of 56 tiles)",
        )
        failed = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload={
                "segmentation_id": str(self.segmentation.id),
                "asset_id": str(self.asset.id),
                "source_model": "quantem:mito",
            },
            resource_class="gpu",
            queue_name=QUEUE_P4_FULL,
        )
        Job.objects.filter(id=failed.id).update(
            status="FAILED",
            message=failure_message(
                _ModelWeightsNotInstalled(
                    "Model pack 'quantem:er' is not installed. Install it on "
                    "the Models screen."
                )
            ),
        )

        body = self._queue_status()

        _assert_clean(list(walk_strings(body, "$")), "GET queue-status (wave)")

    def test_a_download_row(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_INSTALL_MODEL_PACK,
            payload={"pack_id": "quantem:mito"},
            queue_name=QUEUE_P1_INTERACTIVE,
        )
        Job.objects.filter(id=job.id).update(
            status="RUNNING",
            progress_current_bytes=118_000_000,
            progress_total_bytes=365_000_000,
            message="Downloading QuantEM — Mitochondria — 118 of 365 MB",
        )

        _assert_clean(
            list(walk_strings(self._queue_status(), "$")),
            "GET queue-status (download)",
        )

    def test_the_job_detail_body_apart_from_its_traceback(self):
        Job.objects.filter(id=self.run.id).update(
            status="FAILED",
            message=failure_message(ValueError("the image stops part way through")),
            error_traceback="Traceback…\nValueError: the image stops part way through",
        )
        self.run.refresh_from_db()

        body = JobSerializer(self.run).data
        assert set(DIAGNOSTIC_FIELDS) <= set(body), DIAGNOSTIC_FIELDS
        pairs = [
            (where, text)
            for where, text in walk_strings(body, "$")
            if where.split(".")[1].split("[")[0] not in DIAGNOSTIC_FIELDS
        ]

        _assert_clean(pairs, "GET job detail")

    def test_the_download_message_the_handler_writes_names_the_model_not_the_key(self):
        from quantem.jobs.handlers import _model_display_name

        assert _model_display_name("quantem:mito") == "QuantEM — Mitochondria"
        # An id this build does not know is still the most honest name for it.
        assert _model_display_name("future:organelle") == "future:organelle"


class TheQueuedRunSaysHowBigItIs(TestCase):
    """The number the plan asked for and could not have: "waiting · 88 tiles"."""

    def test_a_pending_run_serialises_its_tile_count(self):
        asset = _asset()
        segmentation = ImageSegmentation.objects.create(
            asset=asset, segmentation_type=get_or_create_mitochondria_type()
        )
        Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload={
                "segmentation_id": str(segmentation.id),
                "asset_id": str(asset.id),
                "source_model": "quantem:mito",
            },
            resource_class="gpu",
            queue_name=QUEUE_P4_FULL,
        )

        body = APIClient().get("/api/jobs/queue-status/").json()
        pending = body["queues"][0]["pending"][0]

        assert pending["unit_progress"] == {
            "done": 0,
            "total": 56,
            "label": UNIT_TILE,
            "percent": 0.0,
            "stage": "queued",
            "eta_seconds": None,
        }
        assert pending["progress_stage"] == "queued"


class OneNumberPerJob(TestCase):
    """W7: ``progress`` and ``unit_progress.percent`` used to disagree.

    Measured on the wire throughout a real run: ``progress`` divided by 57 and
    ``unit_progress`` by 56, so ``7/57 = 12.3`` sat beside ``7/56 = 12.5`` for
    the same instant of the same job. Nothing rendered both, which is the only
    reason it was not on screen.
    """

    def setUp(self):
        self.job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            status="RUNNING",
            payload_json={"segmentation_id": str(uuid4())},
            queue_name=QUEUE_P4_FULL,
            progress=12.28,  # 7/57, as the stage-weighted column computes it
            progress_units_done=7,
            progress_units_total=56,
            progress_unit_label=UNIT_TILE,
        )

    def test_both_endpoints_quote_the_tiling_plans_fraction(self):
        detail = JobSerializer(self.job).data
        running = APIClient().get("/api/jobs/queue-status/").json()["running"][0]

        assert detail["progress"] == detail["unit_progress"]["percent"] == 12.5
        assert running["progress"] == running["unit_progress"]["percent"] == 12.5

    def test_a_finished_run_is_a_hundred_whatever_its_tiles_say(self):
        """A run that reused an earlier result walks no tiles and is complete."""
        Job.objects.filter(id=self.job.id).update(
            status="SUCCESS", progress_units_done=0, progress=100.0
        )
        self.job.refresh_from_db()

        assert JobSerializer(self.job).data["progress"] == 100.0

    def test_a_job_that_counts_nothing_keeps_its_own_percentage(self):
        job = Job.objects.create(
            type=JOB_TYPE_INSTALL_MODEL_PACK,
            status="RUNNING",
            payload_json={"pack_id": "quantem:mito"},
            queue_name=QUEUE_P1_INTERACTIVE,
            progress=32.3,
        )

        data = JobSerializer(job).data
        assert data["unit_progress"] is None
        assert data["progress"] == 32.3

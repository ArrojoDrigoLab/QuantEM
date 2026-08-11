"""Nothing a segmentation refuses may answer with an HTTP verb or a route.

Finding V12. Starting a run while one is already in flight answered with::

    A ROI/full segmentation task is already running for this segmentation
    (job 5f2...). Cancel it (POST /api/jobs/<id>/cancel/) and run again. ...

That sentence is rendered verbatim in the application. Its reader is a
biologist with a mouse, and it hands them a request they have no way to make,
in place of the control that would make it -- while the button that actually
cancels the run is two clicks away in Tasks & Queues. Invariant I-12 is exactly
about this class of string.

The detector is ``quantem.registry.tests.copy_gate``, which already encodes
I-12's rules and was written for the same defect on the Models screen. This
module points it at the two refusal bodies the segmentation API produces.

Only ``detail`` is gated. ``locked_payload`` also carries an ``unlock`` block
with a method and a path in it -- deliberately, as machine-readable fields for
a client. A wire name a person never sees is not user-facing copy.
"""

from __future__ import annotations

import re

from django.test import TestCase

from quantem.jobs.constants import (
    JOB_TYPE_LABELS,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
)
from quantem.jobs.models import Job
from quantem.registry.tests.copy_gate import find_violations
from quantem.segmentation.api_views.shared import blocking_job_response_payload
from quantem.segmentation.completion import LOCKED_DETAIL, locked_payload
from quantem.segmentation.models import ImageSegmentation, SegmentationConfig
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

#: An HTTP method named as one. Uppercase on purpose: "Get the numbers" is
#: English, "GET" is a request the reader cannot make.
_HTTP_VERB = re.compile(r"(?<![A-Za-z])(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)(?![A-Za-z])")

#: A route fragment. ``copy_gate``'s own rule only fires on ``/api/`` or on a
#: verb immediately followed by a slash, and the locked refusal slipped past
#: both: it said "DELETE this segmentation's /complete endpoint" -- a verb, a
#: route and the word "endpoint", none of them adjacent.
_ROUTE = re.compile(r"(?<![\w.])/[a-z][a-z0-9_./<>-]*")


def _assert_clean(testcase: TestCase, where: str, text: str) -> None:
    """No shell command, no module path, no HTTP verb, no route. Invariant I-12."""
    violations = find_violations(text, where)
    testcase.assertEqual(
        violations,
        [],
        "\n".join(str(violation) for violation in violations),
    )
    testcase.assertIsNone(_HTTP_VERB.search(text), f"{where}: HTTP verb in {text!r}")
    testcase.assertIsNone(_ROUTE.search(text), f"{where}: route in {text!r}")
    testcase.assertNotIn("endpoint", text.lower(), f"{where}: {text!r}")


class BlockingJobCopyTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("I-12 Conflict Copy Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        SegmentationConfig.objects.create(segmentation=self.segmentation)

    def _job(self, *, status: str, job_type: str = JOB_TYPE_RUN_SEGMENTATION_FULL):
        return Job.objects.create(
            type=job_type,
            status=status,
            payload_json={"segmentation_id": str(self.segmentation.id)},
        )

    def test_a_run_refused_by_a_running_job_says_nothing_a_user_cannot_do(self):
        payload = blocking_job_response_payload(self._job(status="RUNNING"))

        _assert_clean(self, "blocking_job_response_payload(RUNNING).detail", payload["detail"])
        # Positively: it names the task the way the queue panel names it, and
        # the place the cancel control lives.
        self.assertIn(JOB_TYPE_LABELS[JOB_TYPE_RUN_SEGMENTATION_FULL], payload["detail"])
        self.assertIn("Tasks & Queues", payload["detail"])
        # The identifier stays available to clients, out of the prose.
        self.assertEqual(payload["job_status"], "RUNNING")
        self.assertNotIn(payload["job_id"], payload["detail"])

    def test_a_run_refused_by_a_queued_job_says_nothing_a_user_cannot_do(self):
        payload = blocking_job_response_payload(
            self._job(status="PENDING", job_type=JOB_TYPE_RUN_SEGMENTATION_ROI)
        )

        _assert_clean(self, "blocking_job_response_payload(PENDING).detail", payload["detail"])
        self.assertIn(JOB_TYPE_LABELS[JOB_TYPE_RUN_SEGMENTATION_ROI], payload["detail"])
        self.assertIn("Tasks & Queues", payload["detail"])

    def test_an_unknown_job_type_does_not_leak_its_internal_name(self):
        """A type with no label must not fall back to the wire string."""
        payload = blocking_job_response_payload(
            self._job(status="RUNNING", job_type="run_segmentation_full_task")
        )

        _assert_clean(self, "blocking_job_response_payload(unknown).detail", payload["detail"])
        self.assertNotIn("run_segmentation_full_task", payload["detail"])

    def test_the_live_refusal_is_clean_end_to_end(self):
        """Through the real view, not just the helper."""
        self._job(status="RUNNING")

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/apply-full-image/",
            {},
            format="json",
        )

        self.assertEqual(response.status_code, 409)
        _assert_clean(self, "apply-full-image 409 detail", response.data["detail"])


class CompletionLockCopyTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("I-12 Locked Copy Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="COMPLETED",
        )

    def test_the_locked_refusal_names_the_button_and_no_route(self):
        _assert_clean(self, "LOCKED_DETAIL", LOCKED_DETAIL)
        self.assertIn("Unlock segmentation", LOCKED_DETAIL)

    def test_the_machine_readable_way_out_is_still_on_the_payload(self):
        """The route did not disappear; it moved out of the sentence."""
        payload = locked_payload(self.segmentation)

        _assert_clean(self, "locked_payload.detail", payload["detail"])
        self.assertEqual(payload["unlock"]["method"], "DELETE")
        self.assertIn("/complete", payload["unlock"]["path"])

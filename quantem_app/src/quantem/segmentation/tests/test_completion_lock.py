"""A segmentation marked done is locked, and the backend is what enforces it.

The dialog has always said *"Marking it done locks the segmentation."* Nothing
enforced it. A user took a segmentation marked ``COMPLETED``, relabelled all
twenty objects, added a completed ROI and ran full segmentation again -- every
one accepted -- and their point is the whole reason to fix it rather than
reword it: *"'Done' is precisely the state a lab relies on to mean the numbers
are final."*

So mutations are refused with 409 and a message that names the way out, and
``DELETE /complete`` still unlocks. Reads are untouched: a locked segmentation
is still fully browsable, which is most of what anyone does with a finished one.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
)
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, SegmentationConfig, SegmentObject
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_mitochondria_type,
)
from quantem.testing import create_small_test_image

SIZE = 256


def _square(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class CompletionLockTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Completion lock", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)
        self.segment = self._segment()
        self.base = f"/api/segmentations/{self.segmentation.id}"

    def _segment(self) -> SegmentObject:
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/",
            {"geometry_coords": _square(40, 40, 100, 100)},
            format="json",
        )
        assert response.status_code == 201, response.data
        return SegmentObject.objects.get(id=response.data["id"])

    def _mark_done(self) -> None:
        response = self.client.post(f"{self.base}/complete", {}, format="json")
        self.assertEqual(response.status_code, 200, response.data)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "COMPLETED")

    def _mutations(self) -> list[tuple[str, str, str, dict]]:
        """Every request the labeling screen can make that changes something."""
        return [
            (
                "confirm/reject object",
                "post",
                f"/api/segments/{self.segment.id}/label/",
                {"label_state": "EXCLUDED"},
            ),
            (
                "confirm/reject group",
                "post",
                "/api/segments/labels/batch/",
                {"labels": [{"id": str(self.segment.id), "label_state": "EXCLUDED"}]},
            ),
            (
                "draw object",
                "post",
                f"{self.base}/segments/",
                {"geometry_coords": _square(150, 150, 200, 200)},
            ),
            (
                "confirm batch",
                "post",
                f"{self.base}/segments/confirm-batch/",
                {"segments": [{"geometry_coords": _square(150, 150, 200, 200)}]},
            ),
            (
                "remove area",
                "post",
                f"{self.base}/segments/remove-area/",
                {"areas": [{"geometry_coords": _square(50, 50, 90, 90)}]},
            ),
            (
                "remove objects",
                "post",
                f"{self.base}/segments/delete-batch/",
                {"ids": [str(self.segment.id)]},
            ),
            (
                "clear manual labels",
                "post",
                f"{self.base}/labels/clear",
                {},
            ),
            (
                "add completed ROI",
                "post",
                f"{self.base}/completed-rois/",
                {"polygon_coords": _square(10, 10, 60, 60)},
            ),
            (
                "subtract completed ROI",
                "post",
                f"{self.base}/completed-rois/subtract/",
                {"polygon_coords": _square(10, 10, 60, 60)},
            ),
            (
                "run full segmentation",
                "post",
                f"{self.base}/apply-full-image/",
                {},
            ),
            (
                "run ROI segmentation",
                "post",
                f"{self.base}/rerun-roi/",
                {},
            ),
            (
                # The settings the next run would use. Every endpoint that could
                # start that run already refuses; this one accepted a new
                # detection threshold with a 200, so the settings on a finished
                # segmentation no longer described the objects in it.
                "change detection threshold",
                "patch",
                f"{self.base}/config/",
                {"segmentation_threshold": 0.9},
            ),
        ]

    def test_every_mutation_is_refused_while_the_segmentation_is_done(self):
        self._mark_done()
        for name, method, url, body in self._mutations():
            with self.subTest(action=name):
                response = getattr(self.client, method)(url, body, format="json")
                self.assertEqual(
                    response.status_code,
                    409,
                    f"{name} was accepted on a segmentation marked done",
                )
                self.assertTrue(response.data.get("locked"))
                self.assertIn("marked done", response.data["detail"])
                # A refusal that does not say how to proceed is just a wall.
                self.assertIn("Unlock", response.data["detail"])
                self.assertEqual(response.data["unlock"]["method"], "DELETE")

    def test_a_refused_run_does_not_queue_a_job(self):
        """The 409 has to happen before the enqueue, or the run happens anyway."""
        self._mark_done()
        self.client.post(f"{self.base}/apply-full-image/", {}, format="json")
        self.client.post(f"{self.base}/rerun-roi/", {}, format="json")
        self.assertFalse(
            Job.objects.filter(
                type__in=[
                    JOB_TYPE_RUN_SEGMENTATION_FULL,
                    JOB_TYPE_RUN_SEGMENTATION_ROI,
                ],
                payload_json__segmentation_id=str(self.segmentation.id),
            ).exists()
        )

    def test_nothing_was_changed_by_a_refused_mutation(self):
        self._mark_done()
        before = SegmentObject.objects.get(id=self.segment.id)
        config_before = SegmentationConfig.objects.get(
            segmentation=self.segmentation
        ).get_instance_params()
        for _name, method, url, body in self._mutations():
            getattr(self.client, method)(url, body, format="json")

        self.assertEqual(SegmentObject.objects.filter(segmentation=self.segmentation).count(), 1)
        after = SegmentObject.objects.get(id=self.segment.id)
        self.assertEqual(after.label_state, before.label_state)
        self.assertEqual(after.features, before.features)
        self.assertEqual(
            SegmentationConfig.objects.get(
                segmentation=self.segmentation
            ).get_instance_params(),
            config_before,
        )

    def test_posting_is_complete_false_does_not_lock_it_instead(self):
        """The one request shape that asked for the opposite of what it got.

        ``POST /complete`` never read ``is_complete``: a body saying ``false``
        marked the segmentation done and answered 200. Unlocking is ``DELETE``,
        and the refusal has to say so or the caller has no way to find it.
        """
        response = self.client.post(
            f"{self.base}/complete", {"is_complete": False}, format="json"
        )
        self.assertEqual(response.status_code, 400, response.data)
        self.assertEqual(response.data["unlock"]["method"], "DELETE")
        self.assertIn("DELETE", response.data["detail"])

        self.segmentation.refresh_from_db()
        self.assertNotEqual(self.segmentation.status_stage, "COMPLETED")

    def test_posting_is_complete_true_still_marks_it_done(self):
        """A client that spells out what POST already means is not wrong."""
        response = self.client.post(
            f"{self.base}/complete", {"is_complete": True}, format="json"
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "COMPLETED")

    def test_reads_still_work_on_a_locked_segmentation(self):
        """Locking freezes the data, not the screen."""
        self._mark_done()
        for url in (
            f"{self.base}/segments/at-point?x=60&y=60",
            f"{self.base}/inferred",
            f"{self.base}/completed-rois/",
            f"{self.base}/complete",
        ):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)

        region = self.client.post(
            f"{self.base}/segments/query-region",
            {"bbox": {"x0": 0, "y0": 0, "x1": SIZE, "y1": SIZE}},
            format="json",
        )
        self.assertEqual(region.status_code, 200, region.data)

    def test_unlocking_makes_every_mutation_work_again(self):
        self._mark_done()
        unlock = self.client.delete(f"{self.base}/complete")
        self.assertEqual(unlock.status_code, 200, unlock.data)

        response = self.client.post(
            f"/api/segments/{self.segment.id}/label/",
            {"label_state": "EXCLUDED"},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.segment.refresh_from_db()
        self.assertEqual(self.segment.label_state, "EXCLUDED")

    def test_a_batch_spanning_a_locked_segmentation_is_refused_whole(self):
        """Half an applied batch is worse than a refused one."""
        other = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
        )
        other_response = self.client.post(
            f"/api/segmentations/{other.id}/segments/",
            {"geometry_coords": _square(150, 150, 210, 210)},
            format="json",
        )
        self.assertEqual(other_response.status_code, 201, other_response.data)
        other_segment_id = other_response.data["id"]

        self._mark_done()
        response = self.client.post(
            "/api/segments/labels/batch/",
            {
                "labels": [
                    {"id": str(self.segment.id), "label_state": "EXCLUDED"},
                    {"id": other_segment_id, "label_state": "EXCLUDED"},
                ]
            },
            format="json",
        )
        self.assertEqual(response.status_code, 409, response.data)
        self.assertEqual(
            SegmentObject.objects.get(id=other_segment_id).label_state, "CONFIRMED"
        )

    def test_creating_the_same_segmentation_again_does_not_start_a_run(self):
        """``POST /assets/<id>/segmentations/`` is get_or_create and queues a run."""
        self._mark_done()
        response = self.client.post(
            f"/api/assets/{self.image.asset.id}/segmentations/",
            {"segmentation_type_name": "mitochondria"},
            format="json",
        )
        self.assertEqual(response.status_code, 409, response.data)
        self.assertFalse(
            Job.objects.filter(
                type=JOB_TYPE_RUN_SEGMENTATION_ROI,
                payload_json__segmentation_id=str(self.segmentation.id),
            ).exists()
        )

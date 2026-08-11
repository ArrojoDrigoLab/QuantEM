from datetime import timedelta

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from quantem.jobs.constants import QUEUE_P1_INTERACTIVE
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, UserFeedback
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff


class UserFeedbackViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("User Feedback View Test Image")
        self.seg_type = get_or_create_mitochondria_type()
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=self.seg_type,
        )

    def test_create_point_feedback_enqueues_high_priority_job(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/user-feedback/",
            {
                "input_type": "point",
                "point": {"x": 45, "y": 67},
                "feedback_type": "CONFIRMED",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        feedback = UserFeedback.objects.get(id=response.data["id"])
        self.assertEqual(feedback.segmentation_id, self.segmentation.id)
        self.assertEqual(feedback.input_type, UserFeedback.INPUT_TYPE_POINT)
        self.assertEqual(feedback.feedback_type, UserFeedback.FEEDBACK_TYPE_CONFIRMED)
        self.assertEqual(feedback.utilized_status, UserFeedback.STATUS_QUEUED)
        self.assertAlmostEqual(float(feedback.pt_x), 45.0)
        self.assertAlmostEqual(float(feedback.pt_y), 67.0)

        queued_job = Job.objects.get(id=response.data["job_id"])
        self.assertEqual(queued_job.type, "process_user_feedback")
        self.assertEqual(queued_job.priority, "high")
        self.assertEqual(queued_job.queue_name, QUEUE_P1_INTERACTIVE)
        self.assertEqual(
            queued_job.payload_json.get("user_feedback_id"),
            str(feedback.id),
        )
        self.assertEqual(
            queued_job.payload_json.get("segmentation_id"),
            str(self.segmentation.id),
        )

    def test_list_feedback_filters_by_statuses(self):
        queued_feedback = UserFeedback.objects.create(
            segmentation=self.segmentation,
            input_type=UserFeedback.INPUT_TYPE_POINT,
            pt_x=10.0,
            pt_y=10.0,
            feedback_type=UserFeedback.FEEDBACK_TYPE_CONFIRMED,
            utilized_status=UserFeedback.STATUS_QUEUED,
        )
        processing_feedback = UserFeedback.objects.create(
            segmentation=self.segmentation,
            input_type=UserFeedback.INPUT_TYPE_POINT,
            pt_x=20.0,
            pt_y=20.0,
            feedback_type=UserFeedback.FEEDBACK_TYPE_REJECTED,
            utilized_status=UserFeedback.STATUS_PROCESSING,
        )
        UserFeedback.objects.create(
            segmentation=self.segmentation,
            input_type=UserFeedback.INPUT_TYPE_POINT,
            pt_x=30.0,
            pt_y=30.0,
            feedback_type=UserFeedback.FEEDBACK_TYPE_CONFIRMED,
            utilized_status=UserFeedback.STATUS_SUCCESS,
        )

        response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/user-feedback/"
            "?utilized_statuses=QUEUED,PROCESSING"
        )
        self.assertEqual(response.status_code, 200)

        returned_ids = {item["id"] for item in response.data}
        self.assertSetEqual(
            returned_ids,
            {str(queued_feedback.id), str(processing_feedback.id)},
        )

    def test_list_feedback_expires_stale_processing_rows(self):
        stale_feedback = UserFeedback.objects.create(
            segmentation=self.segmentation,
            input_type=UserFeedback.INPUT_TYPE_POINT,
            pt_x=40.0,
            pt_y=40.0,
            feedback_type=UserFeedback.FEEDBACK_TYPE_CONFIRMED,
            utilized_status=UserFeedback.STATUS_PROCESSING,
        )
        UserFeedback.objects.filter(id=stale_feedback.id).update(
            updated_at=timezone.now() - timedelta(minutes=10)
        )

        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/user-feedback/")

        self.assertEqual(response.status_code, 200)
        stale_feedback.refresh_from_db()
        self.assertEqual(stale_feedback.utilized_status, UserFeedback.STATUS_FAILED)
        returned_row = next(item for item in response.data if item["id"] == str(stale_feedback.id))
        self.assertEqual(returned_row["utilized_status"], UserFeedback.STATUS_FAILED)

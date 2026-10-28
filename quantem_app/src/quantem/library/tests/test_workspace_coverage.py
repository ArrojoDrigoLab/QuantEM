from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.segmentation.models import CompletedROI, ImageSegmentation, ProbabilityMap
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


class WorkspaceCoverageTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.asset = create_small_test_image("Coverage image", width=32, height=32).asset
        self.url = f"/api/experiments/{self.asset.experiment_id}/"
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def summary(self):
        return self.client.get(self.url + "workspace/").data["images"][0]["segmentations"][0]

    def test_overlapping_annotated_regions_count_once_and_use_image_area(self):
        CompletedROI.objects.create(segmentation=self.segmentation, geometry=box(0, 0, 16, 16))
        CompletedROI.objects.create(segmentation=self.segmentation, geometry=box(8, 0, 24, 16))
        summary = self.summary()
        self.assertEqual(summary["coverage"], "partial")
        self.assertEqual(summary["annotated_area_count"], 1)
        self.assertEqual(summary["annotated_percent"], 37.5)
        self.assertEqual(summary["short_name"], "Mitos")

    def test_roi_composite_is_not_a_whole_image_result(self):
        record = ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            file_path="unused",
            metadata={"run_scope": "roi", "composite": True},
        )
        self.assertEqual(self.summary()["coverage"], "none")
        record.metadata = {"run_scope": "full"}
        record.save()
        self.assertEqual(self.summary()["coverage"], "none")
        from quantem.segmentation.saved_results import publish_result

        publish_result(
            self.segmentation, source_model="quantem:mito", threshold=0.5, probability_map=record
        )
        self.assertEqual(self.summary()["coverage"], "whole")

    def test_manual_whole_image_completion_needs_no_model_run(self):
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save()
        self.assertEqual(self.summary()["coverage"], "whole")

    def test_empty_segmentation_is_not_listed_as_available(self):
        self.assertEqual(self.summary()["coverage"], "none")

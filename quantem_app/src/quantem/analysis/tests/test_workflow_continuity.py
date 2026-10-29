"""Saved workflows expose their running job and the actual protection boundary."""

import pytest
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.analysis.models import AnalysisRun
from quantem.assets.models import ImageROI
from quantem.jobs.constants import JOB_TYPE_RUN_ANALYSIS
from quantem.jobs.models import Job
from quantem.segmentation.models import CompletedROI, ImageSegmentation, RoiSegmentationStatus
from quantem.segmentation.serializers import ImageSegmentationSerializer
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

pytestmark = pytest.mark.django_db


@pytest.fixture
def segmentation():
    asset = create_small_test_image("Workflow", width=64, height=64).asset
    return ImageSegmentation.objects.create(
        asset=asset, segmentation_type=get_or_create_mitochondria_type()
    )


def test_saved_analysis_exposes_its_latest_job_and_handles_removed_jobs(segmentation):
    run = AnalysisRun.objects.create(segmentation=segmentation, status="RUNNING")
    job = Job.objects.create(
        type=JOB_TYPE_RUN_ANALYSIS,
        status="RUNNING",
        payload_json={"analysis_run_id": str(run.id)},
    )
    client = APIClient()
    response = client.get(f"/api/analysis/{run.id}/")
    assert response.status_code == 200
    assert response.data["job_id"] == str(job.id)
    job.delete()
    assert client.get(f"/api/analysis/{run.id}/").data["job_id"] is None


def test_protection_summary_is_specific_to_the_selected_structure(segmentation):
    roi = ImageROI.objects.create(asset=segmentation.asset, x=0, y=0, width=16, height=16)
    status = RoiSegmentationStatus.objects.create(
        image_roi=roi, segmentation=segmentation, is_complete=False
    )
    serializer = ImageSegmentationSerializer()
    assert serializer.get_protection(segmentation) == {
        "whole_image": False,
        "completed_areas": 0,
        "done_rois": 0,
    }
    CompletedROI.objects.create(segmentation=segmentation, geometry=box(1, 1, 10, 10))
    status.is_complete = True
    status.save(update_fields=["is_complete"])
    assert serializer.get_protection(segmentation) == {
        "whole_image": False,
        "completed_areas": 1,
        "done_rois": 1,
    }
    segmentation.manual_full_image = True
    assert serializer.get_protection(segmentation)["whole_image"] is True


def test_protection_counts_use_two_queries_for_the_entire_list(
    segmentation, django_assert_num_queries
):
    other = ImageSegmentation.objects.create(
        asset=segmentation.asset,
        segmentation_type=segmentation.segmentation_type,
        display_name="Other",
    )
    CompletedROI.objects.create(segmentation=segmentation, geometry=box(1, 1, 10, 10))
    serializer = ImageSegmentationSerializer([segmentation, other], many=True)
    with django_assert_num_queries(2):
        assert serializer.child.get_protection(segmentation)["completed_areas"] == 1
        assert serializer.child.get_protection(other)["completed_areas"] == 0

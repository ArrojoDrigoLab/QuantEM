from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.segment_status import (
    SEGMENT_STATUS_CANDIDATE,
    SEGMENT_STATUS_CONFIRMED,
    SEGMENT_STATUS_REFINED,
)
from quantem.segmentation.serializers import ImageSegmentationSerializer, SegmentObjectSerializer
from quantem.segmentation.source_models import SOURCE_MODEL_MANUAL
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

QUANTEM_MITO_SOURCE_MODEL = "quantem:mito"
OMNIEM_MITO_SOURCE_MODEL = "omniem:mito"


class SegmentObjectSourceModelTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Segment Source Model Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    @staticmethod
    def _square(x0: float, y0: float, x1: float, y1: float) -> Polygon:
        return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))

    def _create_segment(
        self,
        *,
        polygon: Polygon,
        label_state: str,
        source_model: str,
        features: dict | None = None,
        refined: str = "UNREFINED",
    ) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            refined=refined,
            source_model=source_model,
            confidence_score=0.75 if label_state in {"CANDIDATE", "INFERRED"} else None,
            features=features or {},
        )

    def test_segment_serializer_exposes_status_and_source_model(self):
        segment = self._create_segment(
            polygon=self._square(10, 10, 30, 30),
            label_state="CONFIRMED",
            source_model=QUANTEM_MITO_SOURCE_MODEL,
        )

        data = SegmentObjectSerializer(segment).data

        self.assertEqual(data["status"], SEGMENT_STATUS_CONFIRMED)
        self.assertEqual(data["status_label"], "CONFIRMED")
        self.assertEqual(data["source_model"], QUANTEM_MITO_SOURCE_MODEL)

    def test_refined_segment_gets_refined_status(self):
        segment = self._create_segment(
            polygon=self._square(10, 10, 30, 30),
            label_state="CONFIRMED",
            source_model=QUANTEM_MITO_SOURCE_MODEL,
            refined="MANUAL",
        )

        self.assertEqual(segment.status, SEGMENT_STATUS_REFINED)

    def test_confirm_and_reject_batch_sync_status(self):
        candidate = self._create_segment(
            polygon=self._square(10, 10, 30, 30),
            label_state="CANDIDATE",
            source_model=QUANTEM_MITO_SOURCE_MODEL,
            features={"mito_generated": True},
        )
        confirmed_response = self.client.post(
            "/api/segments/labels/batch/",
            {
                "source_model": QUANTEM_MITO_SOURCE_MODEL,
                "labels": [{"id": str(candidate.id), "label_state": "CONFIRMED"}],
            },
            format="json",
        )
        self.assertEqual(confirmed_response.status_code, 200)
        candidate.refresh_from_db()
        self.assertEqual(candidate.status, SEGMENT_STATUS_CONFIRMED)
        self.assertEqual(candidate.refined, "UNREFINED")

        rejected_response = self.client.post(
            "/api/segments/labels/batch/",
            {
                "source_model": QUANTEM_MITO_SOURCE_MODEL,
                "labels": [{"id": str(candidate.id), "label_state": "EXCLUDED"}],
            },
            format="json",
        )
        self.assertEqual(rejected_response.status_code, 200)
        candidate.refresh_from_db()
        self.assertEqual(candidate.status, SEGMENT_STATUS_CANDIDATE)
        self.assertEqual(candidate.label_state, "EXCLUDED")

    def test_source_filter_includes_active_candidates_manual_and_all_confirmed(self):
        mito_candidate = self._create_segment(
            polygon=self._square(10, 10, 30, 30),
            label_state="CANDIDATE",
            source_model=QUANTEM_MITO_SOURCE_MODEL,
        )
        er_candidate = self._create_segment(
            polygon=self._square(40, 10, 60, 30),
            label_state="CANDIDATE",
            source_model=OMNIEM_MITO_SOURCE_MODEL,
        )
        manual_candidate = self._create_segment(
            polygon=self._square(70, 10, 90, 30),
            label_state="CANDIDATE",
            source_model=SOURCE_MODEL_MANUAL,
        )
        confirmed_other_source = self._create_segment(
            polygon=self._square(100, 10, 120, 30),
            label_state="CONFIRMED",
            source_model=OMNIEM_MITO_SOURCE_MODEL,
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/query-region",
            {
                "bbox": {"x0": 0, "y0": 0, "x1": 140, "y1": 40},
                "states": ["CANDIDATE", "CONFIRMED"],
                "source_model": QUANTEM_MITO_SOURCE_MODEL,
                "include_geometry": False,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        returned_ids = {item["id"] for item in response.data["segments"]}
        self.assertIn(str(mito_candidate.id), returned_ids)
        self.assertIn(str(manual_candidate.id), returned_ids)
        self.assertIn(str(confirmed_other_source.id), returned_ids)
        self.assertNotIn(str(er_candidate.id), returned_ids)

    def test_segmentation_serializer_returns_source_model_counts(self):
        self._create_segment(
            polygon=self._square(10, 10, 30, 30),
            label_state="CANDIDATE",
            source_model=QUANTEM_MITO_SOURCE_MODEL,
        )

        data = ImageSegmentationSerializer(self.segmentation).data

        values = {item["value"] for item in data["source_models"]}
        self.assertIn(QUANTEM_MITO_SOURCE_MODEL, values)
        self.assertIn(SOURCE_MODEL_MANUAL, values)
        self.assertEqual(
            data["segment_counts_by_source_model"][QUANTEM_MITO_SOURCE_MODEL]["CANDIDATE"],
            1,
        )

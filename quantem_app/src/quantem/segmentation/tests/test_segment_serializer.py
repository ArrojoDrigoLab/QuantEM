from __future__ import annotations

from django.test import TestCase
from shapely.geometry import Polygon

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.serializers import (
    ImageSegmentationSerializer,
    SegmentObjectSerializer,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff


class SegmentObjectSerializerTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Segment Serializer Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    @staticmethod
    def _square(x0: float, y0: float, x1: float, y1: float) -> Polygon:
        return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))

    @staticmethod
    def _dense_square(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        samples_per_edge: int = 256,
    ) -> Polygon:
        coords: list[tuple[float, float]] = []
        for step in range(samples_per_edge):
            ratio = step / samples_per_edge
            coords.append((x0 + ((x1 - x0) * ratio), y0))
        for step in range(samples_per_edge):
            ratio = step / samples_per_edge
            coords.append((x1, y0 + ((y1 - y0) * ratio)))
        for step in range(samples_per_edge):
            ratio = step / samples_per_edge
            coords.append((x1 - ((x1 - x0) * ratio), y1))
        for step in range(samples_per_edge):
            ratio = step / samples_per_edge
            coords.append((x0, y1 - ((y1 - y0) * ratio)))
        coords.append(coords[0])
        return Polygon(coords)

    def test_prefers_model_confidence_score(self):
        polygon = self._square(10, 10, 20, 20)
        segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CANDIDATE",
            confidence_score=0.88,
            features={"mean_prob": 0.12},
        )

        data = SegmentObjectSerializer(segment).data
        self.assertAlmostEqual(data["confidence_score"], 0.88, places=6)
        self.assertEqual(data["refined"], "UNREFINED")

    def test_falls_back_to_mean_prob_when_confidence_missing(self):
        polygon = self._square(30, 30, 40, 40)
        segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CANDIDATE",
            confidence_score=None,
            features={"mean_prob": 0.73},
        )

        data = SegmentObjectSerializer(segment).data
        self.assertAlmostEqual(data["confidence_score"], 0.73, places=6)
        self.assertEqual(data["refined"], "UNREFINED")

    def test_hover_geometry_detail_simplifies_segment_geometry(self):
        polygon = self._dense_square(30, 30, 50, 50)
        segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CANDIDATE",
            confidence_score=0.73,
            features={"mean_prob": 0.73},
        )

        full_data = SegmentObjectSerializer(segment).data
        hover_data = SegmentObjectSerializer(
            segment,
            context={"geometry_detail": "hover"},
        ).data

        self.assertGreater(len(full_data["geometry_coords"]), len(hover_data["geometry_coords"]))
        self.assertGreaterEqual(len(hover_data["geometry_coords"]), 4)

    def test_image_segmentation_serializer_counts_segment_label_states(self):
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=self._square(0, 0, 10, 10),
            centroid=(self._square(0, 0, 10, 10)).centroid,
            bbox=(self._square(0, 0, 10, 10)).envelope,
            label_state="CONFIRMED",
            confidence_score=0.8,
        )
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=self._square(20, 20, 30, 30),
            centroid=(self._square(20, 20, 30, 30)).centroid,
            bbox=(self._square(20, 20, 30, 30)).envelope,
            label_state="CANDIDATE",
            confidence_score=0.4,
        )

        data = ImageSegmentationSerializer(self.segmentation).data

        self.assertEqual(data["segment_counts"]["CONFIRMED"], 1)
        self.assertEqual(data["segment_counts"]["CANDIDATE"], 1)
        self.assertEqual(data["is_complete"], False)

    def test_image_segmentation_serializer_reports_completed_state(self):
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save(update_fields=["status_stage"])

        data = ImageSegmentationSerializer(self.segmentation).data

        self.assertEqual(data["is_complete"], True)

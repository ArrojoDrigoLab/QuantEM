from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Point, box
from shapely.ops import unary_union

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.serializers.segments import SegmentObjectSerializer
from quantem.segmentation.services.confirm_batch.overlap import (
    MAX_SKELETON_RASTER_PIXELS,
    OverlapResolutionError,
    _component_mask,
    overlap_qualifies_for_union,
    resolve_overlap_between_families,
)
from quantem.segmentation.services.confirm_batch.types import _ConfirmedFamily
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


def _family(geometry, *, manual=False):
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    return _ConfirmedFamily(
        segment=None,
        polygons=polygons,
        features={},
        is_manual_new=manual,
    )


def _assert_exact_partition(testcase: TestCase, first, second):
    original_union = first.union(second)
    overlap = first.intersection(second)
    first_family = _family(first, manual=True)
    second_family = _family(second)
    testcase.assertTrue(resolve_overlap_between_families(first_family, second_family))
    allocated_first = first_family.union_geometry()
    allocated_second = second_family.union_geometry()
    testcase.assertIsNotNone(allocated_first)
    testcase.assertIsNotNone(allocated_second)
    testcase.assertLess(allocated_first.intersection(allocated_second).area, 1e-6)
    testcase.assertLess(
        allocated_first.union(allocated_second).symmetric_difference(original_union).area,
        1e-6,
    )
    first_share = allocated_first.intersection(overlap).area
    second_share = allocated_second.intersection(overlap).area
    testcase.assertAlmostEqual(first_share + second_share, overlap.area, delta=1e-6)
    testcase.assertLess(abs(first_share - second_share), max(overlap.area * 0.10, 1.0))
    return allocated_first, allocated_second


class ManualOverlapSeamTests(TestCase):
    def test_new_object_overlap_direction_over_seventy_percent_unions(self):
        existing = box(0, 0, 100, 100)
        new = box(10, 10, 30, 30)
        self.assertTrue(overlap_qualifies_for_union(new, existing))

    def test_existing_object_overlap_direction_over_seventy_percent_unions(self):
        existing = box(10, 10, 30, 30)
        new = box(0, 0, 100, 100)
        self.assertTrue(overlap_qualifies_for_union(new, existing))

    def test_nonqualifying_lens_is_divided_once_and_evenly(self):
        first = Point(0, 0).buffer(10, resolution=32)
        second = Point(10, 0).buffer(10, resolution=32)
        self.assertFalse(overlap_qualifies_for_union(first, second))
        _assert_exact_partition(self, first, second)

    def test_elongated_branched_overlap_uses_a_balanced_junction_seam(self):
        first = unary_union([box(0, 35, 100, 65), box(35, 0, 65, 100)])
        second = unary_union(
            [
                box(20, 20, 80, 45),
                box(20, 55, 80, 80),
                box(20, 20, 45, 80),
                box(55, 20, 80, 80),
            ]
        )
        _assert_exact_partition(self, first, second)

    def test_overlap_with_a_hole_keeps_the_union_hole_and_allocates_once(self):
        first = (
            Point(0, 0).buffer(20, resolution=64).difference(Point(0, 0).buffer(7, resolution=64))
        )
        second = box(-25, -4, 25, 12)
        allocated_first, allocated_second = _assert_exact_partition(self, first, second)
        combined = allocated_first.union(allocated_second)
        self.assertFalse(combined.contains(Point(0, -6)))

    def test_large_overlap_seam_raster_has_a_fixed_memory_budget(self):
        mask, _x0, _y0, scale_factor = _component_mask(box(0, 0, 10_000, 10_000))

        self.assertLessEqual(mask.size, MAX_SKELETON_RASTER_PIXELS)
        self.assertLess(scale_factor, 1.0)


class ManualOverlapPersistenceTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        image = create_small_test_image("Manual overlap", width=128, height=96, textured=True)
        self.segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _confirmed(self, polygon):
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            label_state="CONFIRMED",
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            features={"area": float(polygon.area)},
        )

    def test_one_manual_shape_unions_every_qualifying_confirmed_row(self):
        first = self._confirmed(box(10, 20, 45, 55))
        second = self._confirmed(box(65, 20, 100, 55))
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/",
            {
                "manual_creation": True,
                "segments": [
                    {"geometry_coords": [[5, 15], [105, 15], [105, 60], [5, 60], [5, 15]]}
                ],
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        rows = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, first.id, "the first persisted confirmed row survives")
        self.assertFalse(SegmentObject.objects.filter(id=second.id).exists())
        self.assertAlmostEqual(rows[0].geometry.area, 4500.0)
        self.assertAlmostEqual(rows[0].features["area"], 4500.0)

    def test_object_draft_include_minus_exclude_serializes_and_stores_the_hole(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/",
            {
                "manual_creation": True,
                "segments": [
                    {
                        "operation": "include",
                        "geometry_coords": [[10, 10], [80, 10], [80, 80], [10, 80], [10, 10]],
                    },
                    {
                        "operation": "exclude",
                        "geometry_coords": [[30, 30], [55, 30], [55, 55], [30, 55], [30, 30]],
                    },
                ],
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        row = SegmentObject.objects.get(segmentation=self.segmentation)
        self.assertEqual(len(row.geometry.interiors), 1)
        serialized = SegmentObjectSerializer(row).data
        self.assertEqual(serialized["geometry"]["type"], "Polygon")
        self.assertEqual(len(serialized["geometry"]["coordinates"]), 2)

    def test_unresolvable_overlap_rejects_and_rolls_back_the_whole_edit(self):
        existing = self._confirmed(box(10, 20, 45, 55))
        original_geometry = existing.geometry

        with patch(
            "quantem.segmentation.services.confirm_batch.service.resolve_overlap_between_families",
            side_effect=OverlapResolutionError("ambiguous overlap"),
        ):
            response = self.client.post(
                f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/",
                {
                    "manual_creation": True,
                    "segments": [
                        {"geometry_coords": [[30, 20], [65, 20], [65, 55], [30, 55], [30, 20]]}
                    ],
                },
                format="json",
            )

        self.assertEqual(response.status_code, 409, response.data)
        self.assertEqual(response.data["detail"], "ambiguous overlap")
        rows = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, existing.id)
        self.assertTrue(rows[0].geometry.equals_exact(original_geometry, 1e-9))

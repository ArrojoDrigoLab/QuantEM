"""Box, crop and mask geometry.

The global-to-crop-and-back round trip is where this class of port breaks: the
crop offset gets applied once, twice or with the wrong sign, and every object
lands somewhere plausible but wrong. These tests pin the trip in both
directions, including the case the arithmetic is easiest to get wrong -- a mask
whose resolution differs from the crop it came from.
"""

from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase

from quantem.sam.config import BBOX_CONTEXT_RADIUS, CROP_GRID
from quantem.sam.geometry import (
    Box,
    Crop,
    binarize,
    box_to_crop,
    mask_to_global_polygon,
    plan_crop,
)

IMAGE_W = 4096
IMAGE_H = 3072


class BoxTests(SimpleTestCase):
    def test_corners_may_arrive_in_any_order(self):
        dragged_up_and_left = Box.normalized(300, 240, 100, 120)
        self.assertEqual(dragged_up_and_left.as_tuple(), (100, 120, 300, 240))

    def test_centre_is_the_middle(self):
        self.assertEqual(Box(100, 200, 300, 400).center, (200.0, 300.0))


class PlanCropTests(SimpleTestCase):
    def test_the_window_is_the_grid_cell_plus_context(self):
        crop = plan_crop(Box(500, 500, 600, 600), IMAGE_W, IMAGE_H)
        self.assertEqual(crop.x, 0)  # cell 0 starts at 0, minus radius, clamped
        self.assertEqual(crop.y, 0)
        self.assertEqual(crop.x1, CROP_GRID + BBOX_CONTEXT_RADIUS)
        self.assertEqual(crop.y1, CROP_GRID + BBOX_CONTEXT_RADIUS)

    def test_the_window_contains_the_box(self):
        for box in (
            Box(500, 500, 600, 600),
            Box(1100, 1100, 1200, 1200),
            Box(4000, 3000, 4090, 3070),
            Box(0, 0, 30, 30),
        ):
            crop = plan_crop(box, IMAGE_W, IMAGE_H)
            visible = Box(
                max(box.x0, 0),
                max(box.y0, 0),
                min(box.x1, IMAGE_W),
                min(box.y1, IMAGE_H),
            )
            self.assertTrue(crop.contains(visible), f"{crop} does not hold {box}")

    def test_two_boxes_in_one_cell_share_a_window(self):
        """This is the whole optimisation -- if it stops holding, the cache dies."""
        first = plan_crop(Box(1200, 1200, 1260, 1260), IMAGE_W, IMAGE_H)
        second = plan_crop(Box(1700, 1500, 1760, 1560), IMAGE_W, IMAGE_H)
        self.assertEqual(first.key(), second.key())

    def test_boxes_in_different_cells_do_not_share_a_window(self):
        first = plan_crop(Box(500, 500, 560, 560), IMAGE_W, IMAGE_H)
        second = plan_crop(Box(2500, 2500, 2560, 2560), IMAGE_W, IMAGE_H)
        self.assertNotEqual(first.key(), second.key())

    def test_a_box_too_big_for_the_shared_window_gets_its_own(self):
        big = Box(1100, 1100, 2900, 2900)
        crop = plan_crop(big, IMAGE_W, IMAGE_H)
        self.assertTrue(crop.contains(big))
        self.assertEqual(crop.x, int(big.x0) - BBOX_CONTEXT_RADIUS)
        self.assertEqual(crop.y, int(big.y0) - BBOX_CONTEXT_RADIUS)

    def test_the_window_never_leaves_the_image(self):
        for box in (Box(0, 0, 10, 10), Box(4080, 3060, 4096, 3072)):
            crop = plan_crop(box, IMAGE_W, IMAGE_H)
            self.assertGreaterEqual(crop.x, 0)
            self.assertGreaterEqual(crop.y, 0)
            self.assertLessEqual(crop.x1, IMAGE_W)
            self.assertLessEqual(crop.y1, IMAGE_H)

    def test_a_tiny_image_still_yields_a_usable_window(self):
        crop = plan_crop(Box(2, 2, 8, 8), 16, 16)
        self.assertEqual((crop.x, crop.y, crop.width, crop.height), (0, 0, 16, 16))


class BoxToCropTests(SimpleTestCase):
    def test_the_offset_is_subtracted_once(self):
        crop = Crop(x=1000, y=800, width=500, height=400)
        local = box_to_crop(Box(1100, 900, 1200, 1000), crop)
        self.assertEqual(local.as_tuple(), (100.0, 100.0, 200.0, 200.0))

    def test_a_box_hanging_off_the_crop_is_clipped_not_extended(self):
        crop = Crop(x=1000, y=800, width=500, height=400)
        local = box_to_crop(Box(900, 700, 1100, 900), crop)
        self.assertEqual(local.as_tuple(), (0.0, 0.0, 100.0, 100.0))

    def test_a_box_entirely_outside_the_crop_is_refused(self):
        crop = Crop(x=1000, y=800, width=500, height=400)
        self.assertIsNone(box_to_crop(Box(10, 10, 50, 50), crop))


class BinarizeTests(SimpleTestCase):
    def test_bool_passes_through(self):
        mask = np.array([[True, False]], dtype=bool)
        self.assertIs(binarize(mask), mask)

    def test_probabilities_split_at_a_half(self):
        mask = np.array([[0.2, 0.5, 0.51, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(binarize(mask), np.array([[False, False, True, True]]))

    def test_floats_above_one_split_at_zero(self):
        mask = np.array([[-1.0, 0.0, 0.3, 7.0]], dtype=np.float32)
        np.testing.assert_array_equal(binarize(mask), np.array([[False, False, True, True]]))

    def test_zero_one_integers_are_taken_as_labels(self):
        mask = np.array([[0, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(binarize(mask), np.array([[False, True, True]]))

    def test_zero_two_five_five_images_split_at_the_midpoint(self):
        mask = np.array([[0, 127, 128, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(binarize(mask), np.array([[False, False, True, True]]))


class MaskToGlobalPolygonTests(SimpleTestCase):
    def _square_mask(self, size=200, x0=50, y0=60, x1=150, y1=140):
        mask = np.zeros((size, size), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    def test_an_empty_mask_yields_nothing(self):
        self.assertIsNone(
            mask_to_global_polygon(np.zeros((32, 32), dtype=bool), Crop(0, 0, 32, 32))
        )

    def test_a_square_at_the_origin_keeps_its_place_and_size(self):
        crop = Crop(x=0, y=0, width=200, height=200)
        polygon, area = mask_to_global_polygon(self._square_mask(), crop)
        minx, miny, maxx, maxy = polygon.bounds
        self.assertAlmostEqual(minx, 50.0, delta=1.0)
        self.assertAlmostEqual(miny, 60.0, delta=1.0)
        self.assertAlmostEqual(maxx, 149.0, delta=1.0)
        self.assertAlmostEqual(maxy, 139.0, delta=1.0)
        self.assertAlmostEqual(area, 100.0 * 80.0, delta=400.0)

    def test_the_crop_offset_is_added_exactly_once(self):
        """The round trip: place a box globally, cut a crop, segment, come back."""
        crop = Crop(x=1000, y=800, width=200, height=200)
        polygon, _area = mask_to_global_polygon(self._square_mask(), crop)
        minx, miny, maxx, maxy = polygon.bounds
        self.assertAlmostEqual(minx, 1000.0 + 50.0, delta=1.0)
        self.assertAlmostEqual(miny, 800.0 + 60.0, delta=1.0)
        self.assertAlmostEqual(maxx, 1000.0 + 149.0, delta=1.0)
        self.assertAlmostEqual(maxy, 800.0 + 139.0, delta=1.0)

    def test_a_mask_at_half_the_crop_resolution_is_scaled_back_up(self):
        """A backend returning masks on its own grid must not shrink the object."""
        small = np.zeros((100, 100), dtype=bool)
        small[30:70, 25:75] = True
        crop = Crop(x=500, y=400, width=200, height=200)
        polygon, _area = mask_to_global_polygon(small, crop)
        minx, miny, maxx, maxy = polygon.bounds
        self.assertAlmostEqual(minx, 500.0 + 50.0, delta=2.0)
        self.assertAlmostEqual(miny, 400.0 + 60.0, delta=2.0)
        self.assertAlmostEqual(maxx, 500.0 + 148.0, delta=2.0)
        self.assertAlmostEqual(maxy, 400.0 + 138.0, delta=2.0)

    def test_a_blob_touching_the_crop_edge_still_closes(self):
        """Without the pad-by-one the contour is an open fragment, not a ring."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[0:20, 0:20] = True
        polygon, area = mask_to_global_polygon(mask, Crop(0, 0, 64, 64))
        self.assertTrue(polygon.exterior.is_ring)
        self.assertGreater(area, 200.0)

    def test_the_full_trip_lands_a_global_box_back_on_itself(self):
        """Draw at (1234, 987)-(1334, 1067); the object must come back there.

        The strongest statement of the invariant: the only geometry input is a
        box in global pixels, the mask is built in crop pixels from the crop the
        planner chose, and the polygon has to return to the global box. Any
        single missing or doubled offset moves it by hundreds of pixels.
        """
        box = Box(1234, 987, 1334, 1067)
        crop = plan_crop(box, IMAGE_W, IMAGE_H)
        local = box_to_crop(box, crop)

        mask = np.zeros((crop.height, crop.width), dtype=bool)
        mask[
            int(local.y0) : int(local.y1),
            int(local.x0) : int(local.x1),
        ] = True

        polygon, _area = mask_to_global_polygon(mask, crop)
        minx, miny, maxx, maxy = polygon.bounds
        self.assertAlmostEqual(minx, box.x0, delta=1.5)
        self.assertAlmostEqual(miny, box.y0, delta=1.5)
        self.assertAlmostEqual(maxx, box.x1, delta=1.5)
        self.assertAlmostEqual(maxy, box.y1, delta=1.5)

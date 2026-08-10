import numpy as np
from django.test import SimpleTestCase

from quantem.assets.ngff import _normalize_ngff_preview_plane, _select_ngff_preview_plane


class SelectNgffPreviewPlaneTests(SimpleTestCase):
    def test_skips_blank_leading_plane_of_three_plane_subset(self):
        # [c, z, y, x] with an empty plane 0 and real data in planes 1-2 -- the
        # exact shape of the webknossos 3-plane decimation that rendered black.
        data = np.zeros((1, 3, 8, 8), dtype=np.uint8)
        data[0, 1] = 120
        data[0, 2] = 118

        plane = _select_ngff_preview_plane(data)

        self.assertEqual(plane.shape, (8, 8))
        self.assertGreater(float(plane.mean()), 0.0)

    def test_selects_only_populated_plane_when_data_is_in_last_plane(self):
        data = np.zeros((1, 3, 8, 8), dtype=np.uint8)
        data[0, 2] = 200  # planes 0 and 1 blank; data only in the last plane

        plane = _select_ngff_preview_plane(data)

        self.assertEqual(plane.shape, (8, 8))
        self.assertAlmostEqual(float(plane.mean()), 200.0)

    def test_all_zero_store_falls_back_to_a_2d_plane_without_error(self):
        data = np.zeros((1, 3, 8, 8), dtype=np.uint8)

        plane = _select_ngff_preview_plane(data)

        self.assertEqual(plane.shape, (8, 8))
        self.assertEqual(float(plane.max()), 0.0)

    def test_two_dimensional_single_channel_passthrough(self):
        data = np.full((1, 8, 8), 50, dtype=np.uint8)

        plane = _select_ngff_preview_plane(data)

        self.assertEqual(plane.shape, (8, 8))
        self.assertAlmostEqual(float(plane.mean()), 50.0)

    def test_bare_two_dimensional_plane_passthrough(self):
        data = np.full((8, 8), 77, dtype=np.uint8)

        plane = _select_ngff_preview_plane(data)

        self.assertEqual(plane.shape, (8, 8))
        self.assertAlmostEqual(float(plane.mean()), 77.0)

    def test_populated_plane_normalizes_to_non_black_preview(self):
        # End-to-end with normalization: the chosen plane must not normalize to all
        # black, which is what the dashboard ultimately renders.
        data = np.zeros((1, 3, 16, 16), dtype=np.uint8)
        data[0, 1, 4:12, 4:12] = 200

        normalized = _normalize_ngff_preview_plane(_select_ngff_preview_plane(data))

        self.assertEqual(normalized.shape, (16, 16))
        self.assertGreater(int(normalized.max()), 0)

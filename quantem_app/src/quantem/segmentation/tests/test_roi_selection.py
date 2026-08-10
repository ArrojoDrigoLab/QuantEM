from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from quantem.segmentation.roi_selection import _choose_scored_candidate, select_roi_for_image


class RoiSelectionTests(TestCase):
    def test_choose_scored_candidate_ignores_non_finite_scores(self):
        candidate = _choose_scored_candidate(
            [
                (float("inf"), 10, 20),
                (float("nan"), 30, 40),
                (5.0, 50, 60),
            ],
            seed=1,
        )

        self.assertEqual(candidate, (5.0, 50, 60))

    def test_select_roi_handles_all_infinite_candidate_scores(self):
        image = SimpleNamespace(
            id="image-1",
            width=10000,
            height=10000,
        )
        preview = np.full((256, 256), 128, dtype=np.uint8)

        with (
            patch(
                "quantem.segmentation.roi_selection.load_image_preview_array",
                return_value=preview,
            ),
            patch("quantem.segmentation.roi_selection._score_window", return_value=float("inf")),
        ):
            result = select_roi_for_image(image, roi_size=3000, seed=1)

        self.assertGreaterEqual(result.x, 0)
        self.assertGreaterEqual(result.y, 0)
        self.assertLessEqual(result.x, image.width - result.width)
        self.assertLessEqual(result.y, image.height - result.height)

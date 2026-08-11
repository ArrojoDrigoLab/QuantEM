"""Crop-extraction tests against a real database fixture.

These run the live schema, not a mock: an asset with a PNG on disk, an
``ImageSegmentation``, a ``CompletedROI``, ``SegmentObject`` rows in several
label states, and a stored probability map. The thing being tested is the
ground-truth contract, and the only way to be sure of it is to build the same
rows the viewer builds.
"""

from __future__ import annotations

import numpy as np
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.segmentation.models import CompletedROI, ImageSegmentation, SegmentObject
from quantem.segmentation.services.adapt import (
    IGNORE,
    CompletedRoiRequired,
    collect_crops,
    plan_split,
    require_crops,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image, write_prob_map_png

SIZE = 256


def _square(x0, y0, x1, y1) -> Polygon:
    return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))


def _l_shape() -> Polygon:
    """A completed area whose bounding box contains unannotated pixels."""
    return Polygon(((20, 20), (140, 20), (140, 80), (80, 80), (80, 140), (20, 140), (20, 20)))


class CropExtractionTests(TestCase):
    def setUp(self):
        self.image = create_small_test_image("Crop Test", width=SIZE, height=SIZE, textured=True)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="CANDIDATES_READY",
        )

    def _segment(self, polygon: Polygon, *, label_state: str = "CONFIRMED"):
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            label_state=label_state,
            confidence_score=0.9,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )

    def _prob_map(self, segmentation=None, *, value: float = 0.9):
        seg = segmentation or self.segmentation
        prob = np.zeros((SIZE, SIZE), dtype=np.float32)
        prob[30:70, 30:70] = value
        return write_prob_map_png(seg, prob, name="MITO_Test")

    # -- the hard blocker ---------------------------------------------------

    def test_refuses_when_there_is_no_completed_roi(self):
        """The reference refused here too: with no exhaustively annotated
        region there is no valid background, so Dice would be measuring the
        user's unfinished work rather than the model."""
        self._segment(_square(30, 30, 70, 70))

        crop_set = collect_crops(self.segmentation)
        assert crop_set.crops == []
        assert not crop_set.ready
        assert "marked as finished" in crop_set.blockers[0]

        with self.assertRaises(CompletedRoiRequired) as caught:
            require_crops(self.segmentation)
        assert "marked as finished" in str(caught.exception)

    def test_completed_roi_without_confirmed_objects_is_blocked(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 140, 140)
        )
        self._segment(_square(30, 30, 70, 70), label_state="INFERRED")

        crop_set = collect_crops(self.segmentation)
        assert not crop_set.ready
        assert "confirmed objects" in crop_set.blockers[0]

    def test_a_missing_probability_map_blocks_only_threshold_calibration(self):
        """Head training predicts its own maps, so it must stay reachable.

        Blocking every mode over a missing map is what made guided fine-tuning
        unreachable: the crops endpoint asked for a probability map
        unconditionally, so a user who had annotated an image had nothing they
        could press.
        """
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 140, 140)
        )
        self._segment(_square(30, 30, 70, 70))

        crop_set = collect_crops(self.segmentation)
        assert crop_set.ready
        assert crop_set.blockers == []
        assert not crop_set.has_probability
        assert any("probability map" in w for w in crop_set.warnings)

        modes = crop_set.mode_blockers()
        assert any("probability map" in b for b in modes["threshold_only"])
        assert modes["head"] == []

        body = crop_set.as_api_dict()
        assert body["has_probability"] is False
        assert body["mode_blockers"]["head"] == []

    def test_threshold_calibration_still_refuses_without_a_map(self):
        """``require_probability=True`` is what the calibration job asks for."""
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 140, 140)
        )
        self._segment(_square(30, 30, 70, 70))

        crop_set = collect_crops(self.segmentation, require_probability=True)
        assert not crop_set.ready
        assert "probability map" in crop_set.blockers[0]

        with self.assertRaises(CompletedRoiRequired) as caught:
            require_crops(self.segmentation, require_probability=True)
        assert "probability map" in str(caught.exception)

    # -- the contract -------------------------------------------------------

    def test_inside_the_roi_is_supervision_outside_it_is_ignore(self):
        CompletedROI.objects.create(segmentation=self.segmentation, geometry=_l_shape())
        self._segment(_square(30, 30, 70, 70))  # confirmed, inside
        self._segment(_square(100, 100, 130, 130))  # confirmed, outside the L
        self._segment(_square(90, 30, 130, 60), label_state="INFERRED")  # not GT
        self._prob_map()

        crop_set = require_crops(self.segmentation, load_prob=True)
        assert crop_set.ready
        (crop,) = crop_set.crops

        # The crop is the ROI bounding box, clipped to the image.
        assert (crop.x, crop.y, crop.width, crop.height) == (20, 20, 120, 120)

        # Inside the L: annotated. In the notch: not annotated.
        assert crop.valid[10, 10] == 1  # (30, 30) image space
        assert crop.valid[100, 100] == 0  # (120, 120), the notch

        # Only the confirmed object inside the ROI is foreground.
        assert crop.gt[30, 30] == 1  # (50, 50), confirmed
        assert crop.gt[20, 80] == 0  # (100, 40), inferred only
        assert crop.n_objects == 1
        # The ground truth is the shape the person drew: 30..70 is 40 px wide,
        # so 1600 px are supervised as foreground. It used to be 41 * 41,
        # because cv2.fillPoly painted both boundaries of every span and made
        # every annotated object a half-pixel fatter than it was drawn -- a
        # boundary a fine-tune would have learnt. See quantem.seg_core.rasterize.
        assert crop.foreground_px == 40 * 40

        # Everything outside the completed area is ignore, never background.
        target = crop.target()
        assert target[100, 100] == IGNORE
        assert set(np.unique(target)) <= {0, 1, IGNORE}

    def test_confirmed_object_outside_the_roi_is_not_foreground(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 100, 100)
        )
        self._segment(_square(30, 30, 60, 60))
        self._segment(_square(150, 150, 200, 200))  # confirmed, but elsewhere
        self._prob_map()

        (crop,) = require_crops(self.segmentation, load_prob=True).crops
        assert crop.n_objects == 1
        # 30..60 is a 30 px square (was 31 * 31 under the inclusive fill).
        assert crop.foreground_px == 30 * 30

    def test_a_hole_punched_in_the_completed_area_is_not_annotated(self):
        """``CompletedRoiSubtractView`` produces interior rings. A hole is the
        user taking an area back, so it must not read as background."""
        ring = Polygon(
            _square(20, 20, 140, 140).exterior.coords,
            [tuple(_square(60, 60, 100, 100).exterior.coords)],
        )
        CompletedROI.objects.create(segmentation=self.segmentation, geometry=ring)
        self._segment(_square(30, 30, 50, 50))
        self._prob_map()

        (crop,) = require_crops(self.segmentation, load_prob=True).crops
        assert crop.valid[10, 10] == 1  # (30, 30), still annotated
        assert crop.valid[60, 60] == 0  # (80, 80), inside the hole
        assert crop.target()[60, 60] == IGNORE

    def test_em_and_probability_windows_match_the_crop(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 140, 140)
        )
        self._segment(_square(30, 30, 70, 70))
        self._prob_map(value=0.9)

        (crop,) = require_crops(self.segmentation, load_em=True, load_prob=True).crops
        assert crop.em is not None and crop.em.shape == (120, 120)
        assert crop.prob is not None and crop.prob.shape == (120, 120)
        # The map's high-probability block is (30..70, 30..70) in image space.
        assert crop.prob[30, 30] > 0.85  # (50, 50)
        assert crop.prob[100, 100] < 0.05  # (120, 120)

    def test_a_probability_map_of_unknown_extent_is_not_used(self):
        """An ROI-scoped map is stored at the ROI's size with no offset recorded.
        Reading it as if it started at (0, 0) would score the model against the
        wrong pixels, which is worse than reporting that nothing has been run."""
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 140, 140)
        )
        self._segment(_square(30, 30, 70, 70))
        write_prob_map_png(
            self.segmentation, np.full((64, 64), 0.9, dtype=np.float32), name="MITO_Roi"
        )

        crop_set = collect_crops(self.segmentation, require_probability=True)
        assert not crop_set.ready
        assert "probability map" in crop_set.blockers[0]

    def test_a_map_that_records_its_window_is_read_at_that_offset(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 140, 140)
        )
        self._segment(_square(30, 30, 70, 70))
        window = np.zeros((120, 120), dtype=np.float32)
        window[10:50, 10:50] = 0.9  # (30..70, 30..70) in image space
        prob_map = write_prob_map_png(self.segmentation, window, name="MITO_Roi")
        prob_map.metadata = {"roi": {"x": 20, "y": 20, "width": 120, "height": 120}}
        prob_map.save(update_fields=["metadata"])

        (crop,) = require_crops(self.segmentation, load_prob=True).crops
        assert crop.prob is not None and crop.prob.shape == (120, 120)
        assert crop.prob[30, 30] > 0.85  # (50, 50)
        assert crop.prob[100, 100] < 0.05  # (120, 120)

    def test_two_roi_runs_each_cover_their_own_completed_area(self):
        """Running the model over one ROI, annotating it, then doing the same on
        a second leaves two real maps and neither covers the other's crop.
        Choosing a single winner silently halved the data the threshold was fit
        on; each crop reads the map that actually covers it."""
        for x0 in (20, 140):
            CompletedROI.objects.create(
                segmentation=self.segmentation,
                geometry=_square(x0, 20, x0 + 100, 120),
            )
            window = np.full((100, 100), 0.9, dtype=np.float32)
            prob_map = write_prob_map_png(self.segmentation, window, name=f"MITO_Roi_{x0}")
            prob_map.metadata = {"roi": {"x": x0, "y": 20, "width": 100, "height": 100}}
            prob_map.save(update_fields=["metadata"])
            self._segment(_square(x0 + 10, 30, x0 + 40, 60))

        crop_set = require_crops(self.segmentation, require_probability=True, load_prob=True)
        assert len(crop_set.crops) == 2
        assert all(c.has_probability for c in crop_set.crops)
        assert all(c.prob is not None and float(c.prob.min()) > 0.85 for c in crop_set.crops)

    def test_a_composite_is_only_warned_about_when_a_crop_actually_used_it(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 140, 140)
        )
        self._segment(_square(30, 30, 70, 70))
        composite = write_prob_map_png(
            self.segmentation,
            np.full((SIZE, SIZE), 0.4, dtype=np.float32),
            name="MITO_DINO",
        )
        composite.metadata = {"composite": True}
        composite.save(update_fields=["metadata"])

        crop_set = collect_crops(self.segmentation)
        (crop,) = crop_set.crops
        assert crop.has_probability
        assert crop.prob_is_composite
        assert any("composited from" in w for w in crop_set.warnings)

    def test_tiny_completed_area_is_skipped_with_a_reason(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(10, 10, 20, 20)
        )
        self._segment(_square(11, 11, 19, 19))
        self._prob_map()

        crop_set = collect_crops(self.segmentation)
        assert crop_set.crops == []
        assert any("too small" in w for w in crop_set.warnings)


class SplitModeTests(TestCase):
    """What kind of held-out number the annotations can support."""

    def _annotated_image(self, name: str, *, rois: int = 1) -> ImageSegmentation:
        image = create_small_test_image(name, width=SIZE, height=SIZE, textured=True)
        segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        for index in range(rois):
            x0 = 20 + index * 100
            CompletedROI.objects.create(
                segmentation=segmentation,
                geometry=_square(x0, 20, x0 + 80, 100),
            )
            polygon = _square(x0 + 10, 30, x0 + 40, 60)
            SegmentObject.objects.create(
                segmentation=segmentation,
                label_state="CONFIRMED",
                geometry=polygon,
                centroid=polygon.centroid,
                bbox=polygon.envelope,
            )
        prob = np.zeros((SIZE, SIZE), dtype=np.float32)
        prob[30:60, 30:60] = 0.9
        write_prob_map_png(segmentation, prob, name="MITO_Test")
        return segmentation

    def test_one_region_has_no_heldout_and_says_so(self):
        segmentation = self._annotated_image("Single region")
        crop_set = collect_crops(segmentation)
        assert crop_set.split_mode == "no-heldout"
        assert any("no held-out score" in w for w in crop_set.warnings)

    def test_two_regions_on_one_image_are_within_image(self):
        segmentation = self._annotated_image("Two regions", rois=2)
        crop_set = collect_crops(segmentation)
        assert crop_set.n_images == 1
        assert crop_set.split_mode == "within-image"
        assert any("within-image" in w for w in crop_set.warnings)

    def test_two_images_are_image_disjoint(self):
        first = self._annotated_image("Image one")
        self._annotated_image("Image two")

        crop_set = collect_crops(first)
        assert crop_set.n_images == 2
        assert crop_set.split_mode == "image-disjoint"

        train, heldout, mode = plan_split(crop_set.crops)
        assert mode == "image-disjoint"
        assert train and heldout
        # The whole point: no image appears on both sides.
        assert not {c.image_key for c in train} & {c.image_key for c in heldout}

    def test_sibling_images_can_be_excluded(self):
        first = self._annotated_image("Image one")
        self._annotated_image("Image two")
        crop_set = collect_crops(first, include_siblings=False)
        assert crop_set.n_images == 1

    def test_api_dict_carries_the_split_and_the_fitted_crops(self):
        first = self._annotated_image("Image one")
        self._annotated_image("Image two")
        body = collect_crops(first).as_api_dict()

        assert body["ready"] is True
        assert body["blockers"] == []
        assert body["split_mode"] == "image-disjoint"
        assert body["n_images"] == 2
        # Honesty rule 2: the UI is told which crops the threshold is fit on.
        assert body["train_crop_names"] and body["heldout_crop_names"]
        assert set(body["train_crop_names"]).isdisjoint(body["heldout_crop_names"])
        assert {c["name"] for c in body["crops"]} == set(
            body["train_crop_names"] + body["heldout_crop_names"]
        )

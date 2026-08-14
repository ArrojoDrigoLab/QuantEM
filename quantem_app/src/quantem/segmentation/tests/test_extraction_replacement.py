import numpy as np
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.assets.utils import create_roi_image_from_image
from quantem.seg_core.db.extraction import extract_and_save_segments
from quantem.seg_core.types import ExtractedSegment, InferenceResult
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff


def _square_coords(x: float, y: float, size: float = 10.0) -> list[tuple[float, float]]:
    return [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
        (x, y),
    ]


def _bbox_from_coords(coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def _center_from_coords(coords: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [pt[0] for pt in coords[:-1]]
    ys = [pt[1] for pt in coords[:-1]]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


class _StubSegmenter:
    name = "mito"
    generated_flag = "mito_generated"

    def __init__(self, extracted_segments: list[ExtractedSegment]):
        self._extracted_segments = extracted_segments

    def extract_instances(
        self,
        prob,
        image,
        prob_maps,
        *,
        min_area,
        coordinate_offset,
        on_progress=None,
    ):
        if on_progress is not None:
            on_progress(1.0)
        return list(self._extracted_segments)


class ExtractionReplacementGuaranteeTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Extraction Replacement Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
        )

    def _create_segment(
        self,
        *,
        label_state: str,
        coords: list[tuple[float, float]],
        generated: bool,
    ) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            label_state=label_state,
            confidence_score=0.9,
            features={"mito_generated": generated},
            geometry=Polygon(coords),
            centroid=(Polygon(coords)).centroid,
            bbox=(Polygon(coords)).envelope,
        )

    def test_rerun_replaces_generated_outputs_and_preserves_manual_labels(self):
        old_generated_candidate = self._create_segment(
            label_state="CANDIDATE",
            coords=_square_coords(10, 10),
            generated=True,
        )
        old_generated_inferred = self._create_segment(
            label_state="INFERRED",
            coords=_square_coords(25, 10),
            generated=True,
        )
        old_generated_outside_roi = self._create_segment(
            label_state="CANDIDATE",
            coords=_square_coords(220, 220),
            generated=True,
        )
        preserved_manual_candidate = self._create_segment(
            label_state="CANDIDATE",
            coords=_square_coords(40, 10),
            generated=False,
        )
        preserved_confirmed = self._create_segment(
            label_state="CONFIRMED",
            coords=_square_coords(55, 10),
            generated=True,
        )
        preserved_excluded = self._create_segment(
            label_state="EXCLUDED",
            coords=_square_coords(70, 10),
            generated=True,
        )

        overlapping_confirmed_coords = _square_coords(55, 10)
        new_candidate_coords = _square_coords(85, 10)
        extracted = [
            ExtractedSegment(
                polygon_coords=overlapping_confirmed_coords,
                centroid_xy=_center_from_coords(overlapping_confirmed_coords),
                bbox_xyxy=_bbox_from_coords(overlapping_confirmed_coords),
                area=100,
                features={"mito_generated": True},
                confidence_score=0.8,
            ),
            ExtractedSegment(
                polygon_coords=new_candidate_coords,
                centroid_xy=_center_from_coords(new_candidate_coords),
                bbox_xyxy=_bbox_from_coords(new_candidate_coords),
                area=100,
                features={"mito_generated": True},
                confidence_score=0.8,
            ),
        ]
        segmenter = _StubSegmenter(extracted)
        inference_result = InferenceResult(
            prob_maps={"mock": np.zeros((32, 32), dtype=np.float32)},
            prob=np.zeros((32, 32), dtype=np.float32),
        )

        created = extract_and_save_segments(
            segmenter=segmenter,
            segmentation=self.segmentation,
            result=inference_result,
            image=np.zeros((self.roi.height, self.roi.width), dtype=np.uint8),
            roi=self.roi,
        )

        self.assertEqual(created, 2)
        self.assertFalse(SegmentObject.objects.filter(id=old_generated_candidate.id).exists())
        self.assertFalse(SegmentObject.objects.filter(id=old_generated_inferred.id).exists())
        self.assertTrue(SegmentObject.objects.filter(id=old_generated_outside_roi.id).exists())
        self.assertTrue(SegmentObject.objects.filter(id=preserved_manual_candidate.id).exists())
        self.assertTrue(SegmentObject.objects.filter(id=preserved_confirmed.id).exists())
        self.assertTrue(SegmentObject.objects.filter(id=preserved_excluded.id).exists())

        generated_candidates = SegmentObject.objects.filter(
            segmentation=self.segmentation,
            label_state="CANDIDATE",
            features__mito_generated=True,
        )
        # The old out-of-ROI candidate survives and Preview writes both new
        # full-map candidates, including the one beneath CONFIRMED geometry.
        self.assertEqual(generated_candidates.count(), 3)

    def test_preview_keeps_confirmed_overlap_but_excluded_still_blocks_at_80_percent(self):
        self._create_segment(
            label_state="CONFIRMED",
            coords=_square_coords(0, 0),
            generated=True,
        )
        self._create_segment(
            label_state="EXCLUDED",
            coords=_square_coords(40, 0),
            generated=True,
        )

        extracted = [
            ExtractedSegment(
                polygon_coords=_square_coords(7, 0),
                centroid_xy=_center_from_coords(_square_coords(7, 0)),
                bbox_xyxy=_bbox_from_coords(_square_coords(7, 0)),
                area=100,
                features={"mito_generated": True},
                confidence_score=0.8,
            ),
            ExtractedSegment(
                polygon_coords=_square_coords(8, 0),
                centroid_xy=_center_from_coords(_square_coords(8, 0)),
                bbox_xyxy=_bbox_from_coords(_square_coords(8, 0)),
                area=100,
                features={"mito_generated": True},
                confidence_score=0.8,
            ),
            ExtractedSegment(
                polygon_coords=_square_coords(47, 0),
                centroid_xy=_center_from_coords(_square_coords(47, 0)),
                bbox_xyxy=_bbox_from_coords(_square_coords(47, 0)),
                area=100,
                features={"mito_generated": True},
                confidence_score=0.8,
            ),
            ExtractedSegment(
                polygon_coords=_square_coords(40, 0),
                centroid_xy=_center_from_coords(_square_coords(40, 0)),
                bbox_xyxy=_bbox_from_coords(_square_coords(40, 0)),
                area=100,
                features={"mito_generated": True},
                confidence_score=0.8,
            ),
        ]

        created = extract_and_save_segments(
            segmenter=_StubSegmenter(extracted),
            segmentation=self.segmentation,
            result=InferenceResult(
                prob_maps={"mock": np.zeros((32, 32), dtype=np.float32)},
                prob=np.zeros((32, 32), dtype=np.float32),
            ),
            image=np.zeros((self.roi.height, self.roi.width), dtype=np.uint8),
            roi=self.roi,
        )

        self.assertEqual(created, 3)
        created_centroids = {
            (round(seg.centroid.x), round(seg.centroid.y))
            for seg in SegmentObject.objects.filter(
                segmentation=self.segmentation,
                label_state="CANDIDATE",
                features__mito_generated=True,
            )
        }
        self.assertEqual(created_centroids, {(12, 5), (13, 5), (52, 5)})

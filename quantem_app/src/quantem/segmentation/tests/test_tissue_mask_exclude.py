from django.test import TestCase
from rest_framework.test import APIClient

from quantem.segmentation.global_masks import load_global_mask
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_tissue_type
from quantem.testing import create_image_from_test_tiff


class TissueMaskExcludeTests(TestCase):
    """The tissue exclude tool must cut a hole out of the drawn mask.

    Existing remove-area tests only cover splitting a segment in two and fully
    deleting it; this covers the interior-hole case the tissue exclude tool
    relies on (draw a filled region, then carve a polygon out of its middle).
    """

    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Tissue Mask Exclude Image")
        self.asset = self.image.asset
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_tissue_type(),
        )

    def test_exclude_cuts_a_hole_in_the_confirmed_mask(self):
        # 1) Add a big filled square like the tissue brush/polygon add does.
        add = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/",
            {
                "merge_overlaps": True,
                "manual_creation": True,
                "segments": [
                    {
                        "geometry_coords": [
                            [10, 10],
                            [90, 10],
                            [90, 90],
                            [10, 90],
                            [10, 10],
                        ]
                    }
                ],
            },
            format="json",
        )
        self.assertEqual(add.status_code, 200, add.data)
        self.assertFalse(SegmentObject.objects.filter(segmentation=self.segmentation).exists())
        self.assertTrue(load_global_mask(self.segmentation)[20, 20])

        # 2) Exclude an inner polygon like the tissue exclude tool does.
        exclude = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/remove-area/",
            {
                "areas": [
                    {
                        "geometry_coords": [
                            [40, 40],
                            [60, 40],
                            [60, 60],
                            [40, 60],
                            [40, 40],
                        ]
                    }
                ]
            },
            format="json",
        )
        self.assertEqual(exclude.status_code, 200, exclude.data)
        # The one binary mask should be updated (a hole cut), not represented
        # as connected components.
        self.assertEqual(int(exclude.data["updated"]), 1, exclude.data)
        self.assertIsNotNone(exclude.data.get("overlay"), "an overlay refresh must be returned")

        # 3) The resulting geometry must actually contain a hole.
        mask = load_global_mask(self.segmentation)
        self.assertTrue(mask[15, 15])
        self.assertFalse(mask[50, 50])
        self.assertFalse(SegmentObject.objects.filter(segmentation=self.segmentation).exists())

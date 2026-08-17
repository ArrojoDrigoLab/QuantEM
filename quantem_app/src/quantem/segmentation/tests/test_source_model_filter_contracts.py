"""The two source-model filters are different rules and must stay different.

``source_model_queryset_filter`` answers "what can the user see and act on
under this model selection" and ``overlay_bundle_source_filter`` answers "what
does this model's raster bundle contain". They were briefly collapsed into one
function so that the model-specific raster would stop absorbing hand-drawn and
other-model confirmed objects. That silently broke the segment
click/hover/region/ROI endpoints: a hand-drawn confirmed object stayed painted
by the source-less confirmed display but ``/segments/at-point`` returned ``[]``
for it, so un-mark, reject and delete became no-ops with no toast and no error.

What separates them is **another model's confirmed objects**, and only those.
Hand-drawn objects are in both, because the two rules happen to agree about
them for unrelated reasons: the user must be able to click one under any model
selection, *and* the named raster is the only place the viewer can paint an
unconfirmed one (the confirmed display is served through a confirmed-only LUT).
Excluding manual objects from bundle membership -- which this file first
asserted -- made a hand-drawn outline the user had not yet confirmed invisible
everywhere, against owner ruling R13; see
``tests.test_annotation_preservation_invariant`` and
``source_models.overlay_bundle_source_filter``.

These tests pin both contracts against the same fixture, so re-merging them
fails here rather than in the field.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff import labels_lut
from quantem.segmentation.source_models import (
    SOURCE_MODEL_MANUAL,
    overlay_bundle_source_filter,
    source_model_queryset_filter,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

QUANTEM_MITO_SOURCE_MODEL = "quantem:mito"
OMNIEM_MITO_SOURCE_MODEL = "omniem:mito"


class SourceModelFilterContractTests(TestCase):
    """One fixture, two opposite expectations."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Source Model Filter Contract Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        # Five objects, each in its own square so a point query resolves to
        # exactly one of them.
        self.active_candidate = self._create_segment(
            polygon=self._square(10, 10, 30, 30),
            label_state="CANDIDATE",
            source_model=QUANTEM_MITO_SOURCE_MODEL,
        )
        self.other_model_candidate = self._create_segment(
            polygon=self._square(40, 10, 60, 30),
            label_state="CANDIDATE",
            source_model=OMNIEM_MITO_SOURCE_MODEL,
        )
        self.manual_confirmed = self._create_segment(
            polygon=self._square(70, 10, 90, 30),
            label_state="CONFIRMED",
            source_model=SOURCE_MODEL_MANUAL,
        )
        self.other_model_confirmed = self._create_segment(
            polygon=self._square(100, 10, 120, 30),
            label_state="CONFIRMED",
            source_model=OMNIEM_MITO_SOURCE_MODEL,
        )
        # An outline the user drew and has not judged yet. It is the object both
        # rules are most easily wrong about: no label to make it "confirmed
        # work", no model to make it a candidate under review.
        self.manual_candidate = self._create_segment(
            polygon=self._square(130, 10, 150, 30),
            label_state="CANDIDATE",
            source_model=SOURCE_MODEL_MANUAL,
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
    ) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            source_model=source_model,
            confidence_score=0.75 if label_state in {"CANDIDATE", "INFERRED"} else None,
            features={},
        )

    def _ids_matching(self, source_filter) -> set[str]:
        queryset = SegmentObject.objects.filter(segmentation=self.segmentation)
        if source_filter is not None:
            queryset = queryset.filter(source_filter)
        return {str(value) for value in queryset.values_list("id", flat=True)}

    # -- user selection rule -------------------------------------------------
    def test_selection_filter_keeps_manual_and_all_confirmed(self):
        """The picker chooses whose candidates are under review, nothing more."""
        matched = self._ids_matching(source_model_queryset_filter(QUANTEM_MITO_SOURCE_MODEL))

        self.assertIn(str(self.active_candidate.id), matched)
        self.assertIn(str(self.manual_confirmed.id), matched)
        self.assertIn(str(self.manual_candidate.id), matched)
        self.assertIn(str(self.other_model_confirmed.id), matched)
        self.assertNotIn(str(self.other_model_candidate.id), matched)

    def test_at_point_returns_hand_drawn_confirmed_under_a_model_picker(self):
        """The un-mark/reject/delete path: hover a hand-drawn object on a model.

        ``useReviewPointActions`` sends the picker's ``source_model`` alongside
        ``states=CONFIRMED``; if this comes back empty the client's
        ``if (!segment) return;`` exits silently and the object looks broken.
        """
        response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/segments/at-point",
            {
                "x": 80,
                "y": 20,
                "states": "CONFIRMED",
                "source_model": QUANTEM_MITO_SOURCE_MODEL,
            },
        )

        self.assertEqual(response.status_code, 200)
        returned_ids = {item["id"] for item in response.data}
        self.assertIn(str(self.manual_confirmed.id), returned_ids)

    def test_at_point_returns_other_model_confirmed_under_a_model_picker(self):
        response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/segments/at-point",
            {
                "x": 110,
                "y": 20,
                "states": "CONFIRMED",
                "source_model": QUANTEM_MITO_SOURCE_MODEL,
            },
        )

        self.assertEqual(response.status_code, 200)
        returned_ids = {item["id"] for item in response.data}
        self.assertIn(str(self.other_model_confirmed.id), returned_ids)

    def test_at_point_still_hides_another_models_candidate(self):
        """The narrowing half of the selection rule is real and still applies."""
        response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/segments/at-point",
            {
                "x": 50,
                "y": 20,
                "source_model": QUANTEM_MITO_SOURCE_MODEL,
            },
        )

        self.assertEqual(response.status_code, 200)
        returned_ids = {item["id"] for item in response.data}
        self.assertNotIn(str(self.other_model_candidate.id), returned_ids)

    # -- overlay bundle membership rule --------------------------------------
    def test_bundle_filter_is_this_models_objects_plus_the_hand_drawn_ones(self):
        matched = self._ids_matching(overlay_bundle_source_filter(QUANTEM_MITO_SOURCE_MODEL))

        self.assertEqual(
            matched,
            {
                str(self.active_candidate.id),
                str(self.manual_confirmed.id),
                str(self.manual_candidate.id),
            },
        )

    def test_named_bundle_queryset_excludes_the_other_model_entirely(self):
        """The model comparison layer paints one model, not both of them.

        Neither the other model's unreviewed candidates nor its confirmed
        objects are in here: the confirmed display already paints the latter,
        and admitting them is what made one confirmation dirty every model's
        raster.
        """
        bundle_ids = {
            str(obj.id)
            for obj in labels_lut.bundle_queryset(self.segmentation, QUANTEM_MITO_SOURCE_MODEL)
        }

        self.assertNotIn(str(self.other_model_candidate.id), bundle_ids)
        self.assertNotIn(str(self.other_model_confirmed.id), bundle_ids)

    def test_a_hand_drawn_candidate_is_in_the_named_bundle(self):
        """Owner ruling R13, at the one place the raster can still break it.

        The viewer composites two layers: the named bundle through a LUT that
        hides ``confirmed``, and the source-less bundle through one that hides
        ``candidate``. So an unconfirmed hand-drawn outline that is only in the
        source-less bundle is painted by neither, and the user's own work
        disappears the moment a model bundle exists. Membership here is what
        keeps it on screen, and it must not depend on ``label_state`` -- that is
        also what keeps Confirm a LUT-only update.
        """
        bundle_ids = {
            str(obj.id)
            for obj in labels_lut.bundle_queryset(self.segmentation, QUANTEM_MITO_SOURCE_MODEL)
        }

        self.assertIn(str(self.manual_candidate.id), bundle_ids)
        self.assertIn(str(self.manual_confirmed.id), bundle_ids)

    def test_source_less_bundle_queryset_keeps_every_object(self):
        """The confirmed display is the home of hand-drawn and other-model work."""
        bundle_ids = {str(obj.id) for obj in labels_lut.bundle_queryset(self.segmentation, None)}

        self.assertEqual(
            bundle_ids,
            {
                str(self.active_candidate.id),
                str(self.other_model_candidate.id),
                str(self.manual_confirmed.id),
                str(self.other_model_confirmed.id),
                str(self.manual_candidate.id),
            },
        )

    # -- the two rules must not be the same rule -----------------------------
    def test_the_two_filters_disagree_about_the_other_model_s_confirmed_objects(self):
        """A single assertion that fails the moment the two are re-merged.

        The difference is one object rather than two now that hand-drawn work is
        in both sets, and it is still the whole of the distinction: the picker
        keeps every confirmed object addressable, the model raster paints only
        one model. Re-merging the functions empties this difference either way
        round, and dropping hand-drawn objects from bundle membership adds
        ``manual_confirmed``/``manual_candidate`` back into it -- both fail.
        """
        selection = self._ids_matching(source_model_queryset_filter(QUANTEM_MITO_SOURCE_MODEL))
        bundle = self._ids_matching(overlay_bundle_source_filter(QUANTEM_MITO_SOURCE_MODEL))

        self.assertEqual(selection - bundle, {str(self.other_model_confirmed.id)})
        self.assertEqual(bundle - selection, set())

    def test_both_filters_pass_everything_through_when_no_model_is_selected(self):
        self.assertIsNone(source_model_queryset_filter(None))
        self.assertIsNone(source_model_queryset_filter("   "))
        self.assertIsNone(overlay_bundle_source_filter(None))
        self.assertIsNone(overlay_bundle_source_filter("   "))

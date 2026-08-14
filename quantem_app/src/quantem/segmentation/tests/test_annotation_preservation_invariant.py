"""Owner ruling R13: a run must not overwrite or hide what the user annotated.

    Running a fine-tuned model, **or any model**, on an image that carries
    annotations must not overwrite or hide the annotated areas. After the run
    the user sees those areas exactly as they drew them.

"Annotated areas" is three things, and they are not the same shape of claim:

* a **user-confirmed or user-drawn object** -- a statement about one outline;
* a **:class:`~quantem.segmentation.models.CompletedROI` polygon** -- a
  statement about a *region*: "everything of this organelle inside here is
  already outlined";
* a **ROI marked done** (:class:`RoiSegmentationStatus` with
  ``is_complete``) -- the same statement about a rectangle, which is what
  ``RoiSegmentationStatus``'s own docstring already calls the ROI-scoped
  ground-truth contract.

The first was already enforced, in two places that between them are the whole
promise ``ImprovePanel`` prints to the user:
:func:`~quantem.seg_core.db.extraction.delete_replaced_candidates` will not
delete a labeled or hand-made row, and
:mod:`~quantem.seg_core.db.candidate_protection` drops a fresh candidate that
lands on one. The tests below pin that, byte for byte, on both the model path
and the include-level dial -- because "exactly as they drew them" is a claim
about the stored outline, and a pass that rewrote a confirmed polygon to a
near-identical one would satisfy every count in the app and change every
perimeter in the paper.

The second and third were **not** enforced, and that is what these tests
started out proving. A region the user marked finished is a region they have
told the app is exhaustively outlined; a run that sprinkles fresh candidates
into the gaps between their outlines has not overwritten anything, but the area
is no longer as they left it -- it is covered in new blue shapes they now have
to go and reject one by one, which is the work marking it done was supposed to
end. R13's "may add objects outside those areas freely" is the same rule read
from the other side.

Both are enforced now, by
:func:`~quantem.seg_core.db.candidate_protection.build_protection_index`, which
loads the finished regions alongside the labeled objects. A candidate whose
centre lands in a finished region is dropped; one that merely clips the edge of
it is not, because a region boundary is where the user stopped drawing, not a
claim about what straddles it.

Nothing here mocks the code under test. The forward pass is replaced -- a
released pack is a gigabyte and this is not a model test -- and everything
after it is the real thresholding, extraction, protection and write path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from django.test import TestCase
from shapely.geometry import Polygon, box

from quantem.assets.utils import create_roi_image_from_image
from quantem.inference import resample
from quantem.inference.segmenter import DinoMitoSegmenter
from quantem.inference.specs import get_model_spec
from quantem.seg_core.db.extraction import extract_and_save_segments, resolve_min_area
from quantem.seg_core.db.inference import replay_stored_probability_map
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    RoiSegmentationStatus,
    SegmentationConfig,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import run_segmentation_full_task
from quantem.segmentation.overlay_ngff.labels_lut import (
    bundle_queryset,
    resolve_object_style,
)
from quantem.segmentation.run_identity import run_identity_from_segmenter
from quantem.segmentation.source_models import SOURCE_MODEL_MANUAL
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256
PIXEL_SIZE_NM = 5.0
MITO_INTERNAL_NAME = "quantem_internal_mito"
SOURCE_MODEL = "quantem:mito"

#: Where the fixture's five blobs sit, as (row, col) fractions of the image.
#: Repeated from the field builder so a test can say "the area around this blob"
#: without recomputing the model's own arithmetic.
BLOBS = ((0.30, 0.30), (0.30, 0.70), (0.70, 0.30), (0.70, 0.70), (0.50, 0.50))


def _model_field(shape: tuple[int, int]) -> np.ndarray:
    """Five confident blobs on the model's own grid.

    The same fixture as ``test_threshold_replay``: five peaks spread over the
    frame so that a run at the default threshold finds several objects, in
    several places, and a test about one region still has objects elsewhere to
    prove the run was not simply suppressed everywhere.
    """
    height, width = shape
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    field = np.zeros(shape, dtype=np.float32)
    for (row_frac, col_frac), sigma, peak in zip(
        BLOBS,
        (16.0, 13.0, 12.0, 11.0, 9.0),
        (0.97, 0.83, 0.72, 0.58, 0.45),
        strict=True,
    ):
        squared = ((rows - row_frac * height) ** 2 + (cols - col_frac * width) ** 2) / (
            2.0 * sigma * sigma
        )
        field = np.maximum(field, peak * np.exp(-squared))
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def _fake_engine():
    """The ``engine`` surface the segmenter uses, with no weights loaded."""
    spec = get_model_spec("quantem", "mito")

    def predict_region(_model, image, *, pixel_size_nm=None, **_kwargs):
        context = resample.plan_resample(image.shape[:2], pixel_size_nm, spec.canonical_nm)
        return SimpleNamespace(prob=_model_field(context.model_shape), context=context, plan=None)

    return SimpleNamespace(
        load_model=lambda *_a, **_k: SimpleNamespace(device="cpu"),
        load_adapted_model=lambda *_a, **_k: SimpleNamespace(device="cpu"),
        predict_region=predict_region,
        estimate_tiles=lambda *_a, **_k: 1,
    )


def _segmenter(threshold: float = 0.5) -> DinoMitoSegmenter:
    return DinoMitoSegmenter(
        source_model=SOURCE_MODEL,
        fg_threshold=threshold,
        pixel_size_nm=PIXEL_SIZE_NM,
    )


def _annotation_fingerprint(segment: SegmentObject) -> tuple:
    """Everything about an annotated object a later reader can see.

    The outline as stored, not a redrawing of it: ``geometry_wkb`` is the bytes,
    so a pass that reconstructed a visually identical polygon with different
    vertices fails here rather than passing quietly and changing every
    measurement downstream. ``label_state``, ``refined`` and ``source_model``
    are in it because an object silently demoted from CONFIRMED to CANDIDATE is
    still on screen but is no longer the user's decision.
    """
    fresh = SegmentObject.objects.get(pk=segment.pk)
    return (
        bytes(fresh.geometry_wkb),
        float(fresh.centroid_x),
        float(fresh.centroid_y),
        float(fresh.bbox_minx),
        float(fresh.bbox_miny),
        float(fresh.bbox_maxx),
        float(fresh.bbox_maxy),
        fresh.label_state,
        fresh.refined,
        fresh.source_model,
        fresh.features,
        fresh.superseded_at,
    )


class AnnotationPreservationTestCase(TestCase):
    """Shared fixture: one 256 px mito image the fake model finds five blobs in."""

    def setUp(self):
        self.image = create_small_test_image("Preservation", width=SIZE, height=SIZE, textured=True)
        self.asset = self.image.asset
        self.asset.pixel_size_nm = PIXEL_SIZE_NM
        self.asset.save(update_fields=["pixel_size_nm"])
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)

    # --- acts -------------------------------------------------------------

    def run_model(self) -> int:
        """Run the model, then accept its default threshold.

        The preservation invariant is exercised by candidate replacement, and
        candidate replacement now belongs exclusively to Apply. Keeping both
        user actions here makes every test below cover the accepted result
        rather than passing vacuously on the preview-only model run.
        """
        segmenter = _segmenter()
        with (
            patch("quantem.inference.segmenter.engine", _fake_engine()),
            patch(
                "quantem.segmentation.organelle_tasks.get_segmenter",
                return_value=segmenter,
            ),
        ):
            run_segmentation_full_task(
                segmentation_id=str(self.segmentation.id),
                segmentation_type=MITO_INTERNAL_NAME,
                source_model=SOURCE_MODEL,
            )
        return self.move_the_dial(0.5)

    def move_the_dial(self, include_level: float) -> int:
        """A re-extract from the stored map, as the include-level job performs it."""
        segmenter = _segmenter(include_level)
        result, image_array = replay_stored_probability_map(
            segmenter, self.segmentation, threshold=include_level
        )
        area_floor = resolve_min_area(segmenter, None)
        return extract_and_save_segments(
            segmenter,
            self.segmentation,
            result,
            image_array,
            None,
            min_area=area_floor,
            run_identity=run_identity_from_segmenter(
                segmenter,
                run_id="dial",
                pack_id_fallback=SOURCE_MODEL,
                native_pixel_size_nm=PIXEL_SIZE_NM,
                min_area=area_floor,
                include_level=include_level,
            ),
            include_level=include_level,
        )

    # --- annotations ------------------------------------------------------

    def blob_box(self, index: int, half: float = 22.0) -> tuple[float, float, float, float]:
        """A square in image pixels around one of the fixture's blobs."""
        row_frac, col_frac = BLOBS[index]
        x = col_frac * SIZE
        y = row_frac * SIZE
        return (x - half, y - half, x + half, y + half)

    def annotate_object(
        self,
        index: int,
        *,
        label_state: str = "CONFIRMED",
        source_model: str = SOURCE_MODEL,
        refined: str = "UNREFINED",
        half: float = 9.0,
    ) -> SegmentObject:
        """One object the user has decided about, over one blob."""
        min_x, min_y, max_x, max_y = self.blob_box(index, half=half)
        polygon = Polygon(
            (
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
                (min_x, min_y),
            )
        )
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            refined=refined,
            source_model=source_model,
            features={"area": polygon.area, "perimeter": polygon.length},
        )

    def complete_area(self, index: int, half: float = 22.0) -> CompletedROI:
        """Mark a polygon around one blob as exhaustively outlined."""
        min_x, min_y, max_x, max_y = self.blob_box(index, half=half)
        return CompletedROI.objects.create(
            segmentation=self.segmentation,
            geometry=box(min_x, min_y, max_x, max_y),
        )

    def complete_roi(self, index: int, half: float = 22.0) -> RoiSegmentationStatus:
        """Mark a rectangular ROI around one blob as done for this organelle."""
        min_x, min_y, max_x, max_y = self.blob_box(index, half=half)
        roi = create_roi_image_from_image(
            self.image,
            x=int(min_x),
            y=int(min_y),
            width=int(max_x - min_x),
            height=int(max_y - min_y),
            source="MANUAL",
            is_active=False,
        )
        status = RoiSegmentationStatus.objects.create(image_roi=roi, segmentation=self.segmentation)
        status.set_complete(True)
        status.save()
        return status

    # --- reads ------------------------------------------------------------

    def fresh_candidates(self) -> list[SegmentObject]:
        """Objects this pass wrote: model candidates nobody has judged."""
        return list(
            SegmentObject.objects.filter(
                segmentation=self.segmentation,
                label_state="CANDIDATE",
                source_model=SOURCE_MODEL,
                superseded_at__isnull=True,
            )
        )

    def candidates_inside(self, region) -> list[SegmentObject]:
        """Fresh candidates whose centre landed inside ``region``."""
        return [
            candidate
            for candidate in self.fresh_candidates()
            if region.contains(candidate.geometry.centroid)
        ]

    def assert_still_visible(self, segment: SegmentObject) -> None:
        """The object is still live, still in the drawing, and still on top.

        Three separate ways an annotation can stop being visible without being
        deleted, and all three have to be false: superseded out of the live set,
        dropped from the bundle the viewer renders, or out-priorited in the
        raster by a fresh candidate that overlaps it -- the last being the one
        that leaves the row in the database and the outline off the screen.
        """
        fresh = SegmentObject.objects.get(pk=segment.pk)
        assert fresh.superseded_at is None, "the annotated object was superseded"
        assert bundle_queryset(self.segmentation, SOURCE_MODEL).filter(pk=fresh.pk).exists(), (
            "the annotated object dropped out of the overlay bundle"
        )

        mine, _state, _colour = resolve_object_style(fresh)
        geometry = fresh.geometry
        for candidate in self.fresh_candidates():
            if candidate.pk == fresh.pk:
                continue
            if not candidate.geometry.intersects(geometry):
                continue
            theirs, _s, _c = resolve_object_style(candidate)
            assert theirs < mine, (
                "a fresh candidate overlaps the annotated object and would win "
                "the contested pixels, hiding it in the overlay"
            )


class ObjectsTheUserDecidedAboutTests(AnnotationPreservationTestCase):
    """One outline, one decision: the pass must leave it byte for byte."""

    def test_a_confirmed_object_survives_a_model_run_unchanged(self):
        confirmed = self.annotate_object(0)
        before = _annotation_fingerprint(confirmed)

        assert self.run_model() > 0, "the fixture found nothing, so this proves nothing"

        assert _annotation_fingerprint(confirmed) == before
        self.assert_still_visible(confirmed)
        assert any(
            candidate.geometry.intersects(confirmed.geometry)
            for candidate in self.fresh_candidates()
        ), (
            "Preview did not materialize the full model result around a confirmed "
            "object; overlap resolution belongs to Confirm"
        )

    def test_a_hand_drawn_object_survives_a_model_run_unchanged(self):
        """What the drawing tool creates: ``source_model="manual"``, CONFIRMED."""
        drawn = self.annotate_object(1, source_model=SOURCE_MODEL_MANUAL, refined="MANUAL")
        before = _annotation_fingerprint(drawn)

        self.run_model()

        assert _annotation_fingerprint(drawn) == before
        self.assert_still_visible(drawn)

    def test_a_hand_drawn_object_the_user_has_not_confirmed_also_survives(self):
        """A drawn outline left as a candidate is still the user's own work.

        It is the weakest of the annotations -- no label, no confirmation -- and
        therefore the one a pass is most likely to treat as its own to replace.
        It is not: ``delete_replaced_candidates`` is scoped to the running
        model's ``source_model``, and a hand-made row is not that model's.
        """
        drawn = self.annotate_object(
            2,
            label_state="CANDIDATE",
            source_model=SOURCE_MODEL_MANUAL,
            refined="MANUAL",
        )
        before = _annotation_fingerprint(drawn)

        self.run_model()

        assert _annotation_fingerprint(drawn) == before
        self.assert_still_visible(drawn)

    def test_a_rejected_object_is_not_offered_again(self):
        """EXCLUDED is a decision too, and a re-run must not undo it."""
        rejected = self.annotate_object(3, label_state="EXCLUDED")
        before = _annotation_fingerprint(rejected)

        self.run_model()

        assert _annotation_fingerprint(rejected) == before
        assert not self.candidates_inside(rejected.geometry), (
            "the model re-proposed an object the user had already rejected"
        )

    def test_a_confirmed_object_survives_the_include_level_dial_unchanged(self):
        """The dial is a run, and R13 says "or any model" for a reason.

        A re-extract replaces the candidate set from the stored map, which is
        the same replacement a model pass performs -- so it is the same risk,
        and ``ImprovePanel`` makes the same promise about it.
        """
        self.run_model()
        confirmed = self.annotate_object(0)
        before = _annotation_fingerprint(confirmed)

        self.move_the_dial(0.3)

        assert _annotation_fingerprint(confirmed) == before
        self.assert_still_visible(confirmed)

    def test_a_hand_drawn_object_survives_the_include_level_dial_unchanged(self):
        self.run_model()
        drawn = self.annotate_object(1, source_model=SOURCE_MODEL_MANUAL, refined="MANUAL")
        before = _annotation_fingerprint(drawn)

        self.move_the_dial(0.3)

        assert _annotation_fingerprint(drawn) == before
        self.assert_still_visible(drawn)


class AreasTheUserMarkedFinishedTests(AnnotationPreservationTestCase):
    """A finished region is a statement about the region, not about one outline.

    "Everything of this organelle inside here is already outlined" is only worth
    making if the app believes it. A pass that adds candidates inside a finished
    area contradicts it, and hands the user the rejection work that marking it
    done was meant to end.
    """

    # No confirmed object is placed inside the finished areas below, and that is
    # deliberate. Confirming one over the blob would suppress the candidate on
    # its own -- the CONFIRMED overlap rule would do all the work -- and the test
    # would pass on an implementation that had never heard of a finished area.
    # An empty finished region is also a real state and the sharpest form of the
    # claim: "I have looked in here, and there is nothing of this organelle."

    def test_a_run_adds_nothing_inside_a_completed_area(self):
        area = self.complete_area(0)

        self.run_model()

        assert not self.candidates_inside(area.geometry), (
            "a model run put fresh candidates inside an area the user had already marked finished"
        )

    def test_a_run_adds_nothing_inside_a_roi_marked_done(self):
        status = self.complete_roi(1)
        roi = status.image_roi
        rectangle = box(roi.x, roi.y, roi.x + roi.width, roi.y + roi.height)

        self.run_model()

        assert not self.candidates_inside(rectangle), (
            "a model run put fresh candidates inside an ROI the user had marked "
            "done for this organelle"
        )

    def test_a_roi_that_is_not_marked_done_protects_nothing(self):
        """The flag is the claim. An ROI is otherwise just a viewport.

        Without this the previous two tests would pass on an implementation that
        refused to write candidates anywhere a rectangle had ever been drawn.
        """
        status = self.complete_roi(0)
        status.set_complete(False)
        status.save()
        roi = status.image_roi
        rectangle = box(roi.x, roi.y, roi.x + roi.width, roi.y + roi.height)

        self.run_model()

        assert self.candidates_inside(rectangle), (
            "an ROI nobody marked done suppressed candidates inside it"
        )

    def test_the_rest_of_the_image_is_still_segmented(self):
        """R13 allows -- requires -- new objects outside the finished areas.

        A protection rule that quietly suppressed the whole run would pass every
        assertion above and make the feature useless.
        """
        area = self.complete_area(0)

        self.run_model()

        outside = [
            candidate
            for candidate in self.fresh_candidates()
            if not area.geometry.contains(candidate.geometry.centroid)
        ]
        assert outside, "the run found nothing anywhere outside the finished area"

    def test_the_models_own_unjudged_guess_inside_a_finished_area_is_still_its_own(self):
        """Where the region rule stops, stated rather than left to be discovered.

        A finished region protects the *user's* work. It does not turn a
        leftover candidate -- one this model proposed and nobody ever judged --
        into an annotation. A pass still clears its own unjudged guesses there,
        which is exactly the promise the app prints: "only my own guesses are
        replaced". The area is then left holding what the user actually decided,
        which is what marking it finished said it held.

        The alternative -- freezing the region against deletion too -- would
        preserve stale guesses from an older run inside the one area the user
        has declared settled, and there is no reading of R13 that asks for that.
        """
        self.run_model()
        area = self.complete_area(0)
        # A real one this pass produced, not a hand-built row: only a genuine
        # model candidate carries the generated flag that marks it as this
        # model's to replace, and a fixture without it would prove nothing.
        leftover = next(
            candidate
            for candidate in self.fresh_candidates()
            if area.geometry.contains(candidate.geometry.centroid)
        )

        self.run_model()

        assert not SegmentObject.objects.filter(pk=leftover.pk).exists(), (
            "an unjudged guess from this model's own previous pass survived, so "
            "a finished area now freezes stale candidates in place"
        )

    def test_the_dial_adds_nothing_inside_a_completed_area_either(self):
        """Lowering the include level finds more objects. Not in there."""
        self.run_model()
        area = self.complete_area(4)

        self.move_the_dial(0.2)

        assert not self.candidates_inside(area.geometry), (
            "moving the include level put fresh candidates inside an area the "
            "user had already marked finished"
        )

    def test_a_candidate_may_cross_a_finished_area_when_its_center_is_outside(self):
        """Finished areas suppress objects centered inside, not boundary crossings."""
        self.run_model()
        target = self.fresh_candidates()[0]
        min_x, min_y, _max_x, max_y = target.geometry.bounds
        center_x = target.geometry.centroid.x
        overlap_strip = box(min_x - 1, min_y - 1, center_x - 1, max_y + 1)
        CompletedROI.objects.create(
            segmentation=self.segmentation,
            geometry=overlap_strip,
        )

        self.move_the_dial(0.5)

        crossing = [
            candidate
            for candidate in self.fresh_candidates()
            if candidate.geometry.intersects(overlap_strip)
            and not overlap_strip.contains(candidate.geometry.centroid)
        ]
        assert crossing, "a boundary-crossing candidate was incorrectly suppressed"

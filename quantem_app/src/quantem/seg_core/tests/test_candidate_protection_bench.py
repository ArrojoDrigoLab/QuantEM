"""Candidate protection has to be both unchanged and fast.

The rule this file guards is a user-facing promise: a model pass never takes
back a decision a person made. The implementation of that rule got an index
(:mod:`quantem.seg_core.db.candidate_protection` builds a
:class:`shapely.STRtree`), because the nested loop it replaced ran
``len(new) x len(labeled)`` shapely tests -- on a well-proofread 60 MP image
that is roughly 15 million of them, and the more carefully the user had worked
the slower their next run got.

An index is only allowed to change *which pairs are worth testing*, never *which
candidates survive*. So the first test here carries the pre-index implementation
verbatim -- copied out of ``git show 6589f2c:quantem_app/src/quantem/seg_core/db/
extraction.py`` -- and asserts the two agree on the identity of every retained
and every dropped object over a seeded fixture plus the boundary cases that are
easy to get wrong. Counts would not catch a swap; ids do, so ids are what is
compared.

The second test is the budget: 3 000 confirmed objects against 5 000 fresh
candidates, in under 200 ms. Measured against the same fixture, the loop this
replaced takes tens of seconds.
"""

from __future__ import annotations

import random
import time

import shapely
from django.test import TestCase
from shapely.geometry import Polygon, box

from quantem.seg_core.db.candidate_protection import (
    CONFIRMED_OVERLAP_THRESHOLD,
    EXCLUDED_OVERLAP_THRESHOLD,
    build_protection_index,
    load_protected_geometries,
)
from quantem.seg_core.db.segment_writer import write_segments
from quantem.seg_core.types import ExtractedSegment
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.source_models import SOURCE_MODEL_MANUAL
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

#: The 3 000 x 5 000 protection pass this package exists to make bearable.
PROTECTION_BUDGET_S = 0.200

#: The thresholds as they stood at 6589f2c, written out rather than imported.
#: The oracle below has to be independent of the module under test: importing
#: the constants would move both sides of the comparison together and a changed
#: threshold would slip through green. ``test_the_thresholds_are_unchanged``
#: is what notices a deliberate change.
REFERENCE_CONFIRMED_THRESHOLD = 0.3
REFERENCE_EXCLUDED_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# The pre-index implementation, kept verbatim as the oracle.
# ---------------------------------------------------------------------------
def reference_overlaps_with_labeled(
    candidate_geom, labeled_geoms: list, threshold: float = 0.8
) -> bool:
    """``extraction._overlaps_with_labeled`` as it stood at commit 6589f2c."""
    candidate_bounds = candidate_geom.bounds
    candidate_area = candidate_geom.area
    for labeled in labeled_geoms:
        try:
            # Cheap bbox reject before the exact intersection.
            lb = labeled.bounds
            if (
                lb[2] < candidate_bounds[0]
                or lb[0] > candidate_bounds[2]
                or lb[3] < candidate_bounds[1]
                or lb[1] > candidate_bounds[3]
            ):
                continue
            intersection = candidate_geom.intersection(labeled)
            if intersection.is_empty:
                continue
            inter_area = intersection.area
            if candidate_area > 0 and inter_area / candidate_area >= threshold:
                return True
            if labeled.area > 0 and inter_area / labeled.area >= threshold:
                return True
        except Exception:
            continue
    return False


def reference_suppresses(polygon, confirmed_geoms, excluded_geoms) -> bool:
    """The two-threshold decision ``extract_and_save_segments`` used to inline."""
    return reference_overlaps_with_labeled(
        polygon, confirmed_geoms, threshold=REFERENCE_CONFIRMED_THRESHOLD
    ) or reference_overlaps_with_labeled(
        polygon, excluded_geoms, threshold=REFERENCE_EXCLUDED_THRESHOLD
    )


def square(x: float, y: float, size: float = 20.0) -> Polygon:
    return box(x, y, x + size, y + size)


class _FixtureMixin:
    source_model = "quantem:mito"

    def make_segmentation(self) -> ImageSegmentation:
        image = create_image_from_test_tiff(
            "Candidate protection fixture", width=64, height=64
        )
        return ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def bulk_label(
        self,
        segmentation: ImageSegmentation,
        geometries,
        *,
        label_state: str,
        source_model: str | None = None,
    ) -> None:
        """Put labeled objects in the database without going through the writer."""
        rows = []
        for geometry in geometries:
            min_x, min_y, max_x, max_y = geometry.bounds
            centroid = geometry.centroid
            rows.append(
                SegmentObject(
                    segmentation=segmentation,
                    geometry_wkb=shapely.to_wkb(geometry),
                    centroid_x=float(centroid.x),
                    centroid_y=float(centroid.y),
                    bbox_minx=float(min_x),
                    bbox_miny=float(min_y),
                    bbox_maxx=float(max_x),
                    bbox_maxy=float(max_y),
                    label_state=label_state,
                    source_model=source_model or self.source_model,
                    confidence_score=1.0,
                    features={},
                )
            )
        SegmentObject.objects.bulk_create(rows, batch_size=500)


class ProtectionIsUnchangedTests(_FixtureMixin, TestCase):
    """The index must retain and drop exactly what the nested loop did."""

    def setUp(self):
        self.segmentation = self.make_segmentation()

    def _oracle_geometries(self):
        confirmed = load_protected_geometries(
            SegmentObject.objects.filter(
                segmentation=self.segmentation, label_state="CONFIRMED"
            )
        )
        excluded = load_protected_geometries(
            SegmentObject.objects.filter(
                segmentation=self.segmentation,
                label_state="EXCLUDED",
                source_model__in=[self.source_model, SOURCE_MODEL_MANUAL],
            )
        )
        return confirmed, excluded

    def _assert_same_decisions(self, candidates: dict[str, Polygon]) -> None:
        confirmed_geoms, excluded_geoms = self._oracle_geometries()

        oracle_retained = {
            key
            for key, polygon in candidates.items()
            if not reference_suppresses(polygon, confirmed_geoms, excluded_geoms)
        }
        oracle_dropped = set(candidates) - oracle_retained
        # A fixture where nothing is dropped, or nothing kept, would pass
        # vacuously whatever the implementation did.
        self.assertTrue(oracle_dropped, "fixture must exercise suppression")
        self.assertTrue(oracle_retained, "fixture must exercise retention")

        one_at_a_time = build_protection_index(self.segmentation, self.source_model)
        scalar_retained = {
            key
            for key, polygon in candidates.items()
            if not one_at_a_time.suppresses(polygon)
        }
        self.assertEqual(
            scalar_retained,
            oracle_retained,
            "the indexed protection kept a different set of objects than the "
            "nested loop it replaced",
        )
        self.assertEqual(set(candidates) - scalar_retained, oracle_dropped)

        # The batched form is what a run actually uses, so it is asserted too
        # rather than assumed to agree with the single-polygon form.
        keys = list(candidates)
        batched = build_protection_index(self.segmentation, self.source_model)
        mask = batched.suppressed_mask([candidates[key] for key in keys])
        batched_retained = {
            key for key, dropped in zip(keys, mask, strict=True) if not dropped
        }
        self.assertEqual(batched_retained, oracle_retained)
        self.assertEqual(batched.stats(), one_at_a_time.stats())

    def test_seeded_random_overlaps_agree_object_for_object(self):
        rng = random.Random(20260810)
        confirmed = [
            square(rng.uniform(0, 900), rng.uniform(0, 900), rng.uniform(8, 60))
            for _ in range(400)
        ]
        excluded = [
            square(rng.uniform(0, 900), rng.uniform(0, 900), rng.uniform(8, 60))
            for _ in range(200)
        ]
        self.bulk_label(self.segmentation, confirmed, label_state="CONFIRMED")
        self.bulk_label(self.segmentation, excluded, label_state="EXCLUDED")

        candidates = {
            f"cand-{i}": square(
                rng.uniform(0, 900), rng.uniform(0, 900), rng.uniform(8, 60)
            )
            for i in range(800)
        }
        self._assert_same_decisions(candidates)

    def test_boundary_and_awkward_shapes_agree(self):
        # A confirmed 10x10 at the origin and an excluded 10x10 at x=100.
        self.bulk_label(
            self.segmentation, [square(0, 0, 10.0)], label_state="CONFIRMED"
        )
        self.bulk_label(
            self.segmentation, [square(100, 0, 10.0)], label_state="EXCLUDED"
        )
        # An excluded object belonging to a different model protects nothing.
        self.bulk_label(
            self.segmentation,
            [square(200, 0, 10.0)],
            label_state="EXCLUDED",
            source_model="omniem:mito",
        )
        # A manual rejection does protect.
        self.bulk_label(
            self.segmentation,
            [square(300, 0, 10.0)],
            label_state="EXCLUDED",
            source_model=SOURCE_MODEL_MANUAL,
        )

        candidates = {
            "confirmed-exactly-at-threshold": box(7.0, 0.0, 17.0, 10.0),
            "confirmed-just-under-threshold": box(7.01, 0.0, 17.01, 10.0),
            "confirmed-just-over-threshold": box(6.99, 0.0, 16.99, 10.0),
            # A huge candidate swallowing a small confirmed object: caught by the
            # other direction of the ratio, not by its own area.
            "confirmed-swallowed-by-giant": box(-500.0, -500.0, 500.0, 500.0),
            "excluded-exactly-at-threshold": box(102.0, 0.0, 112.0, 10.0),
            "excluded-below-threshold": box(105.0, 0.0, 115.0, 10.0),
            "other-model-exclusion-ignored": box(200.0, 0.0, 210.0, 10.0),
            "manual-exclusion-honoured": box(300.0, 0.0, 310.0, 10.0),
            # Envelopes touch, intersection has no area.
            "touching-edge-only": box(10.0, 0.0, 20.0, 10.0),
            "far-away": box(5000.0, 5000.0, 5010.0, 5010.0),
            # Degenerate: zero area, and a sliver.
            "zero-area": box(3.0, 3.0, 3.0, 3.0),
            "sliver-across-confirmed": box(0.0, 4.9, 10.0, 5.0),
        }
        self._assert_same_decisions(candidates)

    def test_an_unreadable_stored_geometry_is_skipped_by_both(self):
        self.bulk_label(
            self.segmentation, [square(0, 0, 10.0)], label_state="CONFIRMED"
        )
        broken = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=square(50, 50, 10.0),
            centroid=square(50, 50, 10.0).centroid,
            bbox=square(50, 50, 10.0),
            label_state="CONFIRMED",
            source_model=self.source_model,
            confidence_score=1.0,
            features={},
        )
        SegmentObject.objects.filter(pk=broken.pk).update(geometry_wkb=b"not-wkb")

        candidates = {
            "over-the-readable-one": box(1.0, 0.0, 11.0, 10.0),
            "over-the-unreadable-one": box(50.0, 50.0, 60.0, 60.0),
        }
        self._assert_same_decisions(candidates)
        index = build_protection_index(self.segmentation, self.source_model)
        self.assertEqual(index.confirmed_count, 1)

    def test_stats_attribute_each_drop_to_the_decision_that_caused_it(self):
        self.bulk_label(
            self.segmentation, [square(0, 0, 10.0)], label_state="CONFIRMED"
        )
        self.bulk_label(
            self.segmentation, [square(100, 0, 10.0)], label_state="EXCLUDED"
        )
        index = build_protection_index(self.segmentation, self.source_model)

        self.assertTrue(index.suppresses(square(1, 0, 10.0)))
        self.assertTrue(index.suppresses(square(101, 0, 10.0)))
        self.assertFalse(index.suppresses(square(900, 900, 10.0)))
        self.assertEqual(
            index.stats(), {"confirmed_hits": 1, "excluded_hits": 1}
        )

    def test_the_objects_that_reach_the_database_are_the_same_objects(self):
        """The end of the acceptance: identity of rows written, not a count.

        The decision tests above compare the protection index against the
        oracle. This one carries the comparison all the way to the table, so a
        writer that dropped or duplicated a candidate after the decision was
        made would fail here even though every decision agreed.
        """
        rng = random.Random(6589062)
        self.bulk_label(
            self.segmentation,
            [
                square(rng.uniform(0, 400), rng.uniform(0, 400), rng.uniform(10, 40))
                for _ in range(60)
            ],
            label_state="CONFIRMED",
        )
        self.bulk_label(
            self.segmentation,
            [
                square(rng.uniform(0, 400), rng.uniform(0, 400), rng.uniform(10, 40))
                for _ in range(30)
            ],
            label_state="EXCLUDED",
        )

        candidates = {}
        rows = []
        for i in range(150):
            x = rng.uniform(0, 400)
            y = rng.uniform(0, 400)
            size = rng.uniform(10, 40)
            key = f"cand-{i}"
            candidates[key] = square(x, y, size)
            rows.append(
                ExtractedSegment(
                    polygon_coords=[
                        (x, y),
                        (x + size, y),
                        (x + size, y + size),
                        (x, y + size),
                        (x, y),
                    ],
                    centroid_xy=(x + size / 2.0, y + size / 2.0),
                    bbox_xyxy=(x, y, x + size, y + size),
                    area=int(size * size),
                    features={"mito_generated": True, "fixture_id": key},
                    confidence_score=0.8,
                )
            )

        confirmed_geoms, excluded_geoms = self._oracle_geometries()
        oracle_retained = {
            key
            for key, polygon in candidates.items()
            if not reference_suppresses(polygon, confirmed_geoms, excluded_geoms)
        }
        self.assertTrue(oracle_retained)
        self.assertTrue(set(candidates) - oracle_retained)

        result = write_segments(
            self.segmentation,
            rows,
            run_identity=None,
            source_model=self.source_model,
            protection=build_protection_index(self.segmentation, self.source_model),
        )
        written = set(
            SegmentObject.objects.filter(
                segmentation=self.segmentation, label_state="CANDIDATE"
            ).values_list("features__fixture_id", flat=True)
        )
        self.assertEqual(written, oracle_retained)
        self.assertEqual(result.written, len(oracle_retained))
        self.assertEqual(
            result.suppressed, len(candidates) - len(oracle_retained)
        )

    def test_the_thresholds_are_unchanged(self):
        """Moving either threshold changes every user's candidate set.

        It may one day be the right change, but it is a scientific one and it
        cannot happen as a side effect of a refactor. This is the line that has
        to be edited deliberately.
        """
        self.assertEqual(CONFIRMED_OVERLAP_THRESHOLD, REFERENCE_CONFIRMED_THRESHOLD)
        self.assertEqual(EXCLUDED_OVERLAP_THRESHOLD, REFERENCE_EXCLUDED_THRESHOLD)

    def test_nothing_labeled_protects_nothing(self):
        index = build_protection_index(self.segmentation, self.source_model)
        self.assertEqual(index.confirmed_count, 0)
        self.assertEqual(index.excluded_count, 0)
        self.assertFalse(index.suppresses(square(0, 0, 10.0)))
        self.assertEqual(index.stats(), {"confirmed_hits": 0, "excluded_hits": 0})


class ProtectionBudgetTests(_FixtureMixin, TestCase):
    """3 000 confirmed against 5 000 new, inside the 200 ms budget."""

    CONFIRMED = 3000
    CANDIDATES = 5000

    def setUp(self):
        self.segmentation = self.make_segmentation()
        # A proofread image: confirmed objects on a 60 x 50 grid at 100 px pitch,
        # which is roughly a 30 MP field at this object density.
        confirmed = [
            square(100.0 * (i % 60), 100.0 * (i // 60), 20.0)
            for i in range(self.CONFIRMED)
        ]
        self.bulk_label(self.segmentation, confirmed, label_state="CONFIRMED")

        # A re-run finds the same objects again (offset by a couple of pixels,
        # as a re-run does) plus some genuinely new ones between them.
        self.candidates = [
            square(100.0 * (i % 60) + 2.0, 100.0 * (i // 60) + 2.0, 20.0)
            for i in range(self.CONFIRMED)
        ] + [
            square(100.0 * (i % 60) + 50.0, 100.0 * (i // 60) + 50.0, 20.0)
            for i in range(self.CANDIDATES - self.CONFIRMED)
        ]

    def test_three_thousand_confirmed_against_five_thousand_new(self):
        best = None
        for _ in range(3):
            started = time.perf_counter()
            index = build_protection_index(self.segmentation, self.source_model)
            suppressed = sum(index.suppressed_mask(self.candidates))
            elapsed = time.perf_counter() - started
            best = elapsed if best is None else min(best, elapsed)

        self.assertEqual(index.confirmed_count, self.CONFIRMED)
        self.assertEqual(suppressed, self.CONFIRMED)
        self.assertEqual(index.stats()["confirmed_hits"], self.CONFIRMED)
        self.assertLessEqual(
            best,
            PROTECTION_BUDGET_S,
            f"protecting {self.CONFIRMED} confirmed objects against "
            f"{self.CANDIDATES} new candidates took {best * 1000.0:.0f} ms, "
            f"budget is {PROTECTION_BUDGET_S * 1000.0:.0f} ms",
        )

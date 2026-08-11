"""Writing candidates has to be cheap, batched, and identical to a plain save.

Saving objects one at a time measured **8.97 ms each**, so the tail of a dense
run spent close to a minute writing rows it had already computed -- one SQLite
commit, one fsync, per object.
:func:`quantem.seg_core.db.segment_writer.write_segments` batches them instead.

Batching is only safe if two things hold, and both are asserted here rather than
asserted in prose:

1. **A bulk-written row is the same row.** ``bulk_create`` bypasses
   ``Model.save``, and ``SegmentObject.save`` derives ``status``, normalises
   ``source_model`` and repairs the geometry columns. If the writer ever drifts
   from the model, ``test_bulk_written_row_matches_a_saved_row`` fails.
2. **The batches stay small.** 150 rows per INSERT, in separate calls, because a
   500-row batch held the database long enough to stall a concurrent progress
   write for 117 ms -- and a tile counter that freezes for a tenth of a second
   reads on screen exactly like a hung run.

The invariant underneath all of it is the one the user actually cares about: a
model pass never destroys their corrections.
"""

from __future__ import annotations

import time
from unittest import mock

import shapely
from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from shapely.geometry import Polygon, box

from quantem.seg_core.db.candidate_protection import build_protection_index
from quantem.seg_core.db.segment_writer import (
    DEFAULT_BATCH_SIZE,
    WriteResult,
    build_row,
    prepare_row,
    write_segments,
)
from quantem.seg_core.types import ExtractedSegment
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.run_identity import RUN_FEATURE_KEY
from quantem.segmentation.source_models import (
    SOURCE_MODEL_MANUAL,
    SOURCE_MODEL_UNKNOWN,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

SOURCE_MODEL = "quantem:mito"

#: UX_PLAN's performance table: ``SegmentObject`` insert, 8.97 ms/row -> 0.8 ms.
PER_ROW_BUDGET_S = 0.0008

#: Fields whose value is generated per row and cannot match between two writes.
_NOT_COMPARABLE = {"id", "created_at", "updated_at"}


def ring(x: float, y: float, size: float = 10.0) -> list[tuple[float, float]]:
    """A closed square ring, the shape a segmenter hands over."""
    return [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
        (x, y),
    ]


def extracted(
    x: float,
    y: float,
    size: float = 10.0,
    *,
    confidence: float = 0.8,
    features: dict | None = None,
) -> ExtractedSegment:
    coords = ring(x, y, size)
    return ExtractedSegment(
        polygon_coords=coords,
        centroid_xy=(x + size / 2.0, y + size / 2.0),
        bbox_xyxy=(x, y, x + size, y + size),
        area=int(size * size),
        features={"mito_generated": True, **(features or {})},
        confidence_score=confidence,
    )


class _SegmentationMixin:
    def make_segmentation(self) -> ImageSegmentation:
        image = create_image_from_test_tiff("Segment writer fixture", width=64, height=64)
        return ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )


class BulkWriteMatchesSaveTests(_SegmentationMixin, TestCase):
    def setUp(self):
        self.segmentation = self.make_segmentation()

    def _write_both_ways(self, segment, run_identity, source_model):
        """The same object through the batched writer and through ``create``."""
        write_segments(
            self.segmentation,
            [segment],
            run_identity=run_identity,
            source_model=source_model,
        )
        bulk_row = SegmentObject.objects.get(segmentation=self.segmentation)

        # The pre-P1 write, verbatim: one ``create`` per object.
        geometry = Polygon(segment.polygon_coords)
        features = dict(segment.features)
        features.setdefault("source_model", source_model)
        if run_identity is not None:
            features[RUN_FEATURE_KEY] = dict(run_identity)
        min_x, min_y, max_x, max_y = segment.bbox_xyxy
        saved_row = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry_wkb=shapely.to_wkb(geometry),
            centroid_x=float(segment.centroid_xy[0]),
            centroid_y=float(segment.centroid_xy[1]),
            bbox_minx=float(min_x),
            bbox_miny=float(min_y),
            bbox_maxx=float(max_x),
            bbox_maxy=float(max_y),
            label_state="CANDIDATE",
            source_model=source_model,
            confidence_score=segment.confidence_score,
            features=features,
        )

        differences = {}
        for field in SegmentObject._meta.concrete_fields:
            if field.name in _NOT_COMPARABLE:
                continue
            bulk_value = getattr(bulk_row, field.attname)
            saved_value = getattr(saved_row, field.attname)
            if isinstance(bulk_value, memoryview):
                bulk_value = bytes(bulk_value)
            if isinstance(saved_value, memoryview):
                saved_value = bytes(saved_value)
            if bulk_value != saved_value:
                differences[field.name] = (bulk_value, saved_value)
        self.assertEqual(differences, {})
        return bulk_row

    def test_bulk_written_row_matches_a_saved_row(self):
        """The batched writer must not skip anything ``save`` does."""
        bulk_row = self._write_both_ways(
            extracted(4.0, 7.0, 12.0, confidence=0.42),
            {"run_id": "fixture-run", "threshold": 0.5},
            SOURCE_MODEL,
        )
        # The two fields ``save`` derives, spelled out so a silent default cannot
        # make the comparison above vacuous.
        self.assertEqual(bulk_row.status, SegmentObject.STATUS_CANDIDATE)
        self.assertEqual(bulk_row.source_model, SOURCE_MODEL)

    def test_a_source_model_the_model_must_derive_is_derived_the_same_way(self):
        """The case that actually exercises the lifecycle half of ``save``.

        With a source model already spelled correctly, ``sync_lifecycle_fields``
        changes nothing, so a writer that forgot to call it would still match.
        Here the row arrives as ``unknown`` and the model has to infer
        ``quantem:mito`` from the organelle and the extractor's marker -- and
        the batched path has to arrive at the same answer as ``save``.
        """
        bulk_row = self._write_both_ways(
            extracted(1.0, 1.0),
            None,
            SOURCE_MODEL_UNKNOWN,
        )
        self.assertEqual(bulk_row.source_model, SOURCE_MODEL)

    def test_a_mixed_case_source_model_is_normalised_the_same_way(self):
        bulk_row = self._write_both_ways(extracted(2.0, 2.0), None, "  QuantEM:Mito ")
        self.assertEqual(bulk_row.source_model, SOURCE_MODEL)

    def test_a_row_the_model_would_refuse_is_refused_here_too(self):
        """The geometry half of the prologue, pinned by what it rejects.

        A bow-tie ring repairs into more than one polygon, which
        ``SegmentObject.save`` treats as an error rather than storing a silently
        truncated shape. ``build_row`` is a public surface, so the batched path
        has to refuse it in the same words instead of writing an invalid
        geometry that no other row in the table could contain.

        ``write_segments`` itself never reaches this: ``to_valid_polygon``
        resolves a bow-tie to its dominant part first.
        """
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
        self.assertFalse(bowtie.is_valid)

        row = build_row(
            self.segmentation,
            extracted(0.0, 0.0),
            bowtie,
            run_identity=None,
            source_model=SOURCE_MODEL,
        )
        with self.assertRaises(ValueError) as batched:
            prepare_row(row)
        with self.assertRaises(ValueError) as singly:
            SegmentObject.objects.create(
                segmentation=self.segmentation,
                geometry_wkb=shapely.to_wkb(bowtie),
                centroid_x=5.0,
                centroid_y=5.0,
                bbox_minx=0.0,
                bbox_miny=0.0,
                bbox_maxx=10.0,
                bbox_maxy=10.0,
                label_state="CANDIDATE",
                source_model=SOURCE_MODEL,
                confidence_score=0.8,
                features={},
            )
        self.assertEqual(str(batched.exception), str(singly.exception))
        self.assertEqual(SegmentObject.objects.count(), 0)

    def test_run_identity_overrides_whatever_the_extractor_left(self):
        segment = extracted(0.0, 0.0, features={"run": {"run_id": "stale"}})
        write_segments(
            self.segmentation,
            [segment],
            run_identity={"run_id": "current"},
            source_model=SOURCE_MODEL,
        )
        row = SegmentObject.objects.get(segmentation=self.segmentation)
        self.assertEqual(row.features["run"], {"run_id": "current"})

    def test_no_run_identity_writes_no_run_key(self):
        write_segments(
            self.segmentation,
            [extracted(0.0, 0.0)],
            run_identity=None,
            source_model=SOURCE_MODEL,
        )
        row = SegmentObject.objects.get(segmentation=self.segmentation)
        self.assertNotIn("run", row.features)


class BatchingTests(_SegmentationMixin, TestCase):
    def setUp(self):
        self.segmentation = self.make_segmentation()

    def _spy_on_bulk_create(self):
        original = SegmentObject.objects.bulk_create
        sizes: list[int] = []

        def spy(objs, *args, **kwargs):
            objs = list(objs)
            sizes.append(len(objs))
            return original(objs, *args, **kwargs)

        return sizes, mock.patch.object(SegmentObject.objects, "bulk_create", side_effect=spy)

    def test_rows_go_out_in_separate_batches_of_at_most_150(self):
        rows = [extracted(20.0 * i, 0.0) for i in range(400)]
        sizes, patcher = self._spy_on_bulk_create()
        with patcher:
            result = write_segments(
                self.segmentation,
                rows,
                run_identity=None,
                source_model=SOURCE_MODEL,
            )

        self.assertEqual(result.written, 400)
        self.assertEqual(SegmentObject.objects.count(), 400)
        self.assertEqual(sum(sizes), 400)
        self.assertEqual(max(sizes), DEFAULT_BATCH_SIZE)
        self.assertEqual(sizes, [150, 150, 100])

    def test_four_hundred_candidates_do_not_cost_four_hundred_statements(self):
        """The structural form of the 8.97 ms/row finding.

        One ``create`` per object is one statement and one commit per object:
        401 statements for this input. Batched, it is a handful -- Django caps a
        multi-row INSERT at 999 bind parameters, which for this model's 18
        columns is 55 rows per statement, so the three 150-row transactions
        become eight INSERTs plus the one COUNT that reports how much manual
        work the pass left alone.

        Unlike the wall-clock budget in ``InsertCostTests``, this assertion does
        not depend on how fast or how loaded the machine is.
        """
        rows = [extracted(20.0 * i, 0.0) for i in range(400)]
        with CaptureQueriesContext(connection) as captured:
            write_segments(
                self.segmentation,
                rows,
                run_identity=None,
                source_model=SOURCE_MODEL,
            )
        self.assertLessEqual(
            len(captured.captured_queries),
            12,
            f"writing 400 candidates took {len(captured.captured_queries)} "
            f"database statements; one per row would be 401",
        )

    def test_the_default_batch_size_is_the_measured_one(self):
        self.assertEqual(DEFAULT_BATCH_SIZE, 150)
        self.assertEqual(WriteResult().batch_size, 150)

    def test_a_short_run_writes_once(self):
        sizes, patcher = self._spy_on_bulk_create()
        with patcher:
            write_segments(
                self.segmentation,
                [extracted(20.0 * i, 0.0) for i in range(7)],
                run_identity=None,
                source_model=SOURCE_MODEL,
            )
        self.assertEqual(sizes, [7])

    def test_nothing_extracted_writes_nothing(self):
        sizes, patcher = self._spy_on_bulk_create()
        details: list[str] = []
        with patcher:
            result = write_segments(
                self.segmentation,
                [],
                run_identity=None,
                source_model=SOURCE_MODEL,
                on_detail=details.append,
            )
        self.assertEqual(sizes, [])
        self.assertEqual(result.written, 0)
        self.assertEqual(details, ["No candidates to save after extraction"])


class InsertCostTests(_SegmentationMixin, TestCase):
    ROWS = 1500

    def setUp(self):
        self.segmentation = self.make_segmentation()

    def test_per_row_insert_cost(self):
        rows = [extracted(20.0 * (i % 200), 20.0 * (i // 200)) for i in range(self.ROWS)]
        started = time.perf_counter()
        result = write_segments(
            self.segmentation,
            rows,
            run_identity={"run_id": "cost"},
            source_model=SOURCE_MODEL,
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(result.written, self.ROWS)
        per_row = elapsed / self.ROWS
        self.assertLessEqual(
            per_row,
            PER_ROW_BUDGET_S,
            f"writing {self.ROWS} candidates cost {per_row * 1000.0:.2f} ms/row, "
            f"budget is {PER_ROW_BUDGET_S * 1000.0:.2f} ms/row",
        )


class WriteResultTests(_SegmentationMixin, TestCase):
    def setUp(self):
        self.segmentation = self.make_segmentation()

    def _label(self, geometry, *, label_state, source_model=SOURCE_MODEL):
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=geometry,
            centroid=geometry.centroid,
            bbox=geometry,
            label_state=label_state,
            source_model=source_model,
            confidence_score=1.0,
            features={},
        )

    def test_the_result_says_what_each_decision_absorbed(self):
        self._label(box(0, 0, 10, 10), label_state="CONFIRMED")
        self._label(box(100, 0, 110, 10), label_state="EXCLUDED")
        manual = self._label(
            box(500, 500, 510, 510),
            label_state="CONFIRMED",
            source_model=SOURCE_MODEL_MANUAL,
        )

        rows = [
            extracted(1.0, 0.0),  # lands on the confirmed one
            extracted(101.0, 0.0),  # lands on the rejected one
            extracted(900.0, 900.0),  # genuinely new
        ]
        # A ring that cannot be a polygon at all.
        rows.append(
            ExtractedSegment(
                polygon_coords=[(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
                centroid_xy=(0.5, 0.5),
                bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
                area=0,
                features={},
                confidence_score=0.1,
            )
        )

        protection = build_protection_index(self.segmentation, SOURCE_MODEL)
        result = write_segments(
            self.segmentation,
            rows,
            run_identity=None,
            source_model=SOURCE_MODEL,
            protection=protection,
            deleted=17,
        )

        self.assertEqual(result.written, 1)
        self.assertEqual(result.suppressed_confirmed, 1)
        self.assertEqual(result.suppressed_excluded, 1)
        self.assertEqual(result.suppressed, 2)
        self.assertEqual(result.unusable, 1)
        self.assertEqual(result.deleted, 17)
        self.assertEqual(result.manual_untouched, 1)
        self.assertGreater(result.elapsed_s, 0.0)
        self.assertTrue(SegmentObject.objects.filter(pk=manual.pk).exists())

    def test_no_protection_index_protects_nothing(self):
        self._label(box(0, 0, 10, 10), label_state="CONFIRMED")
        result = write_segments(
            self.segmentation,
            [extracted(1.0, 0.0)],
            run_identity=None,
            source_model=SOURCE_MODEL,
        )
        self.assertEqual(result.written, 1)
        self.assertEqual(result.suppressed, 0)

    def test_progress_reaches_the_end_of_its_range(self):
        statuses: list[tuple[str, float]] = []
        details: list[str] = []
        write_segments(
            self.segmentation,
            [extracted(20.0 * i, 0.0) for i in range(50)],
            run_identity=None,
            source_model=SOURCE_MODEL,
            on_status=lambda stage, pct: statuses.append((stage, pct)),
            on_detail=details.append,
        )
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1][0], "EXTRACTING_CANDIDATES")
        self.assertAlmostEqual(statuses[-1][1], 99.0)
        self.assertTrue(all(70.0 <= pct <= 99.0 for _, pct in statuses))
        self.assertTrue(details[-1].startswith("Saving candidate shapes: 50/50"))

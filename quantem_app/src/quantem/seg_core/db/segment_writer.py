"""Turning extracted shapes into rows, and saying what the write did.

Two jobs live here. The first is mechanical: build a
:class:`~quantem.segmentation.models.SegmentObject` for each shape a segmenter
found, drop the ones the user's own decisions protect, and get the rest onto
disk. The second is the honest report of what happened -- :class:`WriteResult`
carries the numbers a person needs to trust the pass, including the two that
say how many candidates their previous work absorbed and how much manual work
the pass left alone.

Why the write is batched
------------------------
Saving one row at a time measured **8.97 ms per object**, so a dense run's tail
spent close to a minute writing rows it had already computed. Each
``SegmentObject.objects.create`` is its own transaction, and on SQLite that is
a commit -- an fsync -- per object.

Rows go out in batches of :data:`DEFAULT_BATCH_SIZE` = 150. The batch size is a
measurement, not a round number: 500-row batches were faster still, but one of
them held the database long enough to stall a concurrent progress write for
117 ms, and a progress counter that freezes for a tenth of a second at a time
is indistinguishable on screen from a run that has hung. 150 keeps the write
cheap and keeps the lock short enough that the tile counter beside it stays
alive.

Batches are also flushed one call at a time rather than handed to Django as a
single ``bulk_create(..., batch_size=150)``, which would wrap every batch in one
transaction and hold the lock for the whole set -- reintroducing the stall the
batch size exists to avoid. Committing per batch also matches the old
per-row behaviour if the process dies mid-write: the work already done is
already on disk.

So 150 is the **transaction** size, not the statement size. Django caps a
multi-row INSERT at 999 bind parameters, which for this model's 18 columns is
55 rows, so each 150-row batch reaches SQLite as three INSERTs inside one
transaction. That is the right split: the statement count is Django's business,
the lock-hold duration is ours.

Why the model's ``save`` is reproduced here
-------------------------------------------
``bulk_create`` does not call ``Model.save``, and ``SegmentObject.save`` is not
a no-op: it derives ``status`` from ``label_state``/``refined`` and normalises
``source_model`` (:meth:`~quantem.segmentation.models.SegmentObject.sync_lifecycle_fields`),
then repairs and re-serialises the geometry columns
(:meth:`~quantem.segmentation.models.SegmentObject.prepare_shape_fields`).
Skipping either would write rows that differ from every other row in the table.
:func:`prepare_row` runs exactly that prologue, and
``test_segment_writer.py::test_bulk_written_row_matches_a_saved_row`` pins the
two paths together field by field so this file cannot drift from the model.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.run_identity import RUN_FEATURE_KEY
from quantem.segmentation.source_models import SOURCE_MODEL_MANUAL

from .candidate_protection import ProtectionIndex

logger = logging.getLogger(__name__)

#: Rows per transaction. See the module docstring: measured, not chosen for looks.
DEFAULT_BATCH_SIZE = 150


@dataclass
class WriteResult:
    """What one candidate write actually did.

    ``suppressed_confirmed`` and ``suppressed_excluded`` are the count of
    freshly extracted shapes the user's own confirmations and rejections
    absorbed; ``manual_untouched`` is how many hand-made objects the pass left
    exactly as they were. Together they are the evidence for the promise that a
    model pass never destroys a person's corrections.
    """

    written: int = 0
    suppressed_confirmed: int = 0
    suppressed_excluded: int = 0
    deleted: int = 0
    manual_untouched: int = 0
    elapsed_s: float = 0.0
    #: Shapes whose outline could not be made into a usable polygon at all.
    unusable: int = 0
    #: Rows per transaction actually used, kept so a receipt can explain timing.
    batch_size: int = DEFAULT_BATCH_SIZE

    @property
    def suppressed(self) -> int:
        """Every shape dropped because a labeled object already covered it."""
        return self.suppressed_confirmed + self.suppressed_excluded


@dataclass
class _Batcher:
    """Accumulates rows and flushes them one fixed-size transaction at a time."""

    batch_size: int
    pending: list[SegmentObject] = field(default_factory=list)
    written: int = 0

    def add(self, row: SegmentObject) -> None:
        self.pending.append(row)
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        batch = self.pending
        self.pending = []
        SegmentObject.objects.bulk_create(batch, batch_size=self.batch_size)
        self.written += len(batch)


def to_valid_polygon(coords: Sequence[tuple[float, float]]) -> ShapelyPolygon | None:
    """Build a shapely polygon from a closed ring, repairing self-intersections.

    Returns None when the ring cannot be turned into a usable polygon.
    """
    try:
        geometry = ShapelyPolygon(coords)
    except (ValueError, TypeError):
        return None
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
        if geometry.is_empty or not geometry.is_valid:
            return None
    if geometry.geom_type == "MultiPolygon":
        # buffer(0) on a bow-tie ring can split it; keep the dominant part.
        geometry = max(geometry.geoms, key=lambda part: part.area)
    if geometry.geom_type != "Polygon":
        return None
    return geometry


def prepare_row(row: SegmentObject) -> SegmentObject:
    """Run everything ``SegmentObject.save`` does before it reaches the database.

    ``bulk_create`` bypasses ``save``, so the lifecycle derivation and the
    geometry repair have to happen explicitly or a bulk-written row would not
    match a singly-saved one. Every check ``save`` makes is made here.

    The one difference is that a value the property just handed back is not
    written back: re-assigning it re-serialises to the identical bytes (or, for
    the bbox and centroid, the identical floats), so the stored row is the same
    and the work is not done twice per object.
    ``test_bulk_written_row_matches_a_saved_row`` compares the two paths field
    by field, which is what keeps that claim true.
    """
    row.sync_lifecycle_fields()
    geometry = row.geometry
    centroid = row.centroid
    bbox = row.bbox
    repaired_geometry, repaired_centroid, repaired_bbox = row.prepare_shape_fields(
        geometry=geometry,
        centroid=centroid,
        bbox=bbox,
    )
    if repaired_geometry is not geometry:
        row.geometry = repaired_geometry
    if repaired_centroid is not centroid:
        row.centroid = repaired_centroid
    if repaired_bbox is not bbox:
        row.bbox = repaired_bbox
    return row


def build_row(
    segmentation: ImageSegmentation,
    segment,
    geometry: BaseGeometry,
    *,
    run_identity: dict[str, object] | None,
    source_model: str,
) -> SegmentObject:
    """One unsaved ``SegmentObject`` for one extracted shape."""
    min_x, min_y, max_x, max_y = segment.bbox_xyxy
    features = dict(segment.features) if isinstance(segment.features, dict) else {}
    features.setdefault("source_model", source_model)
    if run_identity is not None:
        # Not setdefault: the run that just produced this object is the
        # authority on which settings made it, over anything an extractor
        # happened to leave in its features.
        features[RUN_FEATURE_KEY] = dict(run_identity)
    return SegmentObject(
        segmentation=segmentation,
        # The property, not the raw column: it serialises once and leaves the
        # shapely object cached, so ``prepare_row`` does not read it back.
        geometry=geometry,
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


def count_manual_objects(segmentation: ImageSegmentation) -> int:
    """Hand-made objects on this segmentation, which no model pass may touch."""
    return SegmentObject.objects.filter(
        segmentation=segmentation,
        source_model=SOURCE_MODEL_MANUAL,
    ).count()


def write_segments(
    segmentation: ImageSegmentation,
    rows: Iterable,
    *,
    run_identity: dict[str, object] | None,
    source_model: str,
    protection: ProtectionIndex | None = None,
    deleted: int = 0,
    on_status: Callable[[str, float], None] | None = None,
    on_detail: Callable[[str], None] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> WriteResult:
    """Write extracted shapes as CANDIDATE objects, skipping protected ground.

    Args:
        segmentation: The segmentation the new objects belong to.
        rows: The shapes a segmenter extracted
            (:class:`~quantem.seg_core.types.ExtractedSegment`).
        run_identity: The run that produced them, stamped onto every object.
            ``None`` writes no run key, which reads downstream as "not produced
            by a model".
        source_model: The model these objects are attributed to.
        protection: What the user has already decided, from
            :func:`~quantem.seg_core.db.candidate_protection.build_protection_index`.
            ``None`` protects nothing, which is only right when the caller has
            already established there is nothing to protect.
        deleted: Rows the caller removed before this write, carried into the
            result so one object describes the whole replacement.
        batch_size: Rows per INSERT. The default is measured; see the module
            docstring before changing it.
    """
    extracted = list(rows)
    total_extracted = len(extracted)
    result = WriteResult(deleted=int(deleted), batch_size=int(batch_size))
    started_at = time.perf_counter()

    progress_interval = max(1, total_extracted // 100) if total_extracted > 0 else 1
    detail_interval = max(1, total_extracted // 25) if total_extracted > 0 else 1

    if on_detail is not None and total_extracted == 0:
        on_detail("No candidates to save after extraction")

    # Outlines first, then one protection pass over all of them, then the write.
    # Asking the index about candidates one at a time cost more in per-call
    # shapely dispatch than the geometry work itself; see
    # :meth:`quantem.seg_core.db.candidate_protection.ProtectionIndex.suppressed_mask`.
    polygons = [to_valid_polygon(segment.polygon_coords) for segment in extracted]
    usable = [position for position, poly in enumerate(polygons) if poly is not None]
    result.unusable = total_extracted - len(usable)

    suppressed: set[int] = set()
    if protection is not None and usable:
        mask = protection.suppressed_mask([polygons[position] for position in usable])
        suppressed = {
            position for position, dropped in zip(usable, mask, strict=True) if dropped
        }

    batcher = _Batcher(batch_size=int(batch_size))
    loop_started_at = time.perf_counter()
    try:
        for idx, segment in enumerate(extracted, start=1):
            position = idx - 1
            geometry = polygons[position]
            if geometry is not None and position not in suppressed:
                batcher.add(
                    prepare_row(
                        build_row(
                            segmentation,
                            segment,
                            geometry,
                            run_identity=run_identity,
                            source_model=source_model,
                        )
                    )
                )

            if on_status is not None and (
                idx % progress_interval == 0 or idx == total_extracted
            ):
                on_status(
                    "EXTRACTING_CANDIDATES",
                    70.0 + (29.0 * idx / max(total_extracted, 1)),
                )
            if on_detail is not None and (
                idx % detail_interval == 0 or idx == total_extracted
            ):
                elapsed = time.perf_counter() - loop_started_at
                fraction = idx / max(total_extracted, 1)
                if elapsed > 0 and 0.0 < fraction < 1.0:
                    eta_seconds = elapsed * (1.0 - fraction) / fraction
                    on_detail(
                        f"Saving candidate shapes: {idx}/{total_extracted} ({fraction * 100.0:.0f}%, ETA ~{eta_seconds:.0f}s)"
                    )
                else:
                    on_detail(
                        f"Saving candidate shapes: {idx}/{total_extracted} ({fraction * 100.0:.0f}%)"
                    )
    finally:
        # Whatever was built before an unexpected failure is still real work;
        # the per-row writer this replaced would already have committed it.
        batcher.flush()
        result.written = batcher.written

    if protection is not None:
        hits = protection.stats()
        result.suppressed_confirmed = hits["confirmed_hits"]
        result.suppressed_excluded = hits["excluded_hits"]
    result.manual_untouched = count_manual_objects(segmentation)
    result.elapsed_s = time.perf_counter() - started_at
    return result

"""Re-measure stored objects under the pixel convention they are read in.

**Why this exists.** Until :mod:`quantem.seg_core.rasterize` landed, a
hand-drawn object was measured off a mask ``cv2.fillPoly`` had painted a
half-pixel larger all round than the outline: a square of side *s* stored
``(s + 1) ** 2``, which is +44% at 5 px, +21% at 10 px, +10% at 20 px and +2% at
100 px. A model-found object was measured off its own label mask and was always
right. The fix changes what is *written from now on*; it does not touch a number
already in the database.

So a database that predates the fix holds three populations and nothing on the
row tells them apart:

* model objects -- correct then, correct now;
* hand-drawn objects measured before the fix -- inflated, size-dependently;
* hand-drawn objects drawn or edited since -- correct.

The queued feature refresh does not clear this up. It sweeps for objects with no
``features["area"]`` at all (``jobs.handlers._unmeasured_segment_ids``), and a
stale measurement is not a missing one. This command is the deliberate pass:
re-measure every stored polygon through the same writer the app uses
(:func:`quantem.segmentation.features.measure.measure_segments`), so every row
in ``objects.csv`` means the same thing again.

It reports before doing anything. Run it with no flags to see the size of the
change per segmentation and overall; add ``--apply`` to write.

``geometry_changed`` is **not** set: no outline is being reshaped here, only
re-measured, so ``mean_prob`` and ``confidence_score`` -- which describe the
model's opinion of an outline that has not moved -- are correctly left alone.

Objects whose image cannot be read come back unmeasured, and
:func:`measure_segments` then *clears* their measurement keys rather than
leaving the old convention's numbers in place. That is the existing ruling
(absent means "not measured"; a number means "this is the measurement of this
shape") and it is also what makes them recoverable: an object with no ``area``
is what the queued refresh sweeps for. The command counts them separately and
says so.
"""

from __future__ import annotations

import logging

from django.core.management.base import BaseCommand, CommandError
from shapely.geometry import Polygon

from quantem.assets.asset_openable import get_asset_openable
from quantem.segmentation.features.measure import measure_polygon, measure_segments
from quantem.segmentation.models import ImageSegmentation, SegmentObject

logger = logging.getLogger(__name__)

#: Objects re-measured per database round trip. The asset is opened once per
#: segmentation by ``measure_segments``; this only bounds how many ORM objects
#: are held at a time on a segmentation with tens of thousands of them.
BATCH_SIZE = 500


class Command(BaseCommand):
    help = (
        "Re-measure stored objects so every area is in one pixel convention. "
        "Reports what would change; pass --apply to write."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--segmentation",
            type=str,
            default=None,
            help="Limit to one segmentation id (default: every segmentation).",
        )
        parser.add_argument(
            "--confirmed-only",
            action="store_true",
            help=(
                "Only objects in the CONFIRMED state. These are the ones that "
                "reach objects.csv, so this is the smallest pass that fixes "
                "what a paper cites."
            ),
        )
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Write the new measurements. Without it, nothing is changed.",
        )

    def handle(self, *args, **options):
        segmentation_id = options.get("segmentation")
        confirmed_only = bool(options.get("confirmed_only"))
        apply_changes = bool(options.get("apply"))

        segmentations = ImageSegmentation.objects.select_related("asset").order_by("created_at")
        if segmentation_id:
            segmentations = segmentations.filter(id=segmentation_id)
            if not segmentations.exists():
                raise CommandError(f"No segmentation with id {segmentation_id}.")

        mode = "apply" if apply_changes else "dry run (nothing will be written)"
        self.stdout.write(f"[remeasure] {mode}")

        totals = {"seen": 0, "changed": 0, "unmeasured": 0}
        worst_ratio = 1.0
        worst_id = ""

        for segmentation in segmentations:
            objects = SegmentObject.objects.filter(segmentation=segmentation)
            if confirmed_only:
                objects = objects.filter(label_state="CONFIRMED")

            seen = changed = unmeasured = 0
            batch: list[SegmentObject] = []
            for segment in objects.order_by("created_at").iterator(chunk_size=BATCH_SIZE):
                batch.append(segment)
                if len(batch) < BATCH_SIZE:
                    continue
                result = self._process(segmentation, batch, apply_changes)
                seen += result[0]
                changed += result[1]
                unmeasured += result[2]
                if result[3] > worst_ratio:
                    worst_ratio, worst_id = result[3], result[4]
                batch = []
            if batch:
                result = self._process(segmentation, batch, apply_changes)
                seen += result[0]
                changed += result[1]
                unmeasured += result[2]
                if result[3] > worst_ratio:
                    worst_ratio, worst_id = result[3], result[4]

            if seen:
                self.stdout.write(
                    f"[remeasure] segmentation={segmentation.id} "
                    f"objects={seen} area_changed={changed} "
                    f"unmeasurable={unmeasured}"
                )
            totals["seen"] += seen
            totals["changed"] += changed
            totals["unmeasured"] += unmeasured

        self.stdout.write(
            f"[remeasure] done objects={totals['seen']} "
            f"area_changed={totals['changed']} "
            f"unmeasurable={totals['unmeasured']}"
        )
        if worst_id:
            self.stdout.write(
                f"[remeasure] largest correction: {100 * (worst_ratio - 1):.1f}% "
                f"on object {worst_id}"
            )
        if totals["unmeasured"]:
            self.stdout.write(
                "[remeasure] objects that could not be measured have had their "
                "measurement keys cleared rather than left in the old "
                "convention; a feature refresh will pick them up once their "
                "image is readable again."
            )
        if not apply_changes and totals["changed"]:
            self.stdout.write("[remeasure] re-run with --apply to write these.")

    def _process(
        self,
        segmentation: ImageSegmentation,
        batch: list[SegmentObject],
        apply_changes: bool,
    ) -> tuple[int, int, int, float, str]:
        """Measure ``batch``; return (seen, changed, unmeasured, worst, id).

        ``worst`` is the largest ``old / new`` area ratio seen, which is how the
        dry run reports the size of the correction.
        """
        before = {str(segment.id): (segment.features or {}).get("area") for segment in batch}
        if apply_changes:
            measured, unmeasured = self._apply(segmentation, batch)
        else:
            measured, unmeasured = self._preview(segmentation, batch)

        changed = 0
        worst = 1.0
        worst_id = ""
        for segment in batch:
            key = str(segment.id)
            old = before.get(key)
            new = measured.get(key)
            if new is None or not isinstance(old, int | float) or isinstance(old, bool):
                continue
            if abs(float(old) - float(new)) > 1e-9:
                changed += 1
                if float(new) > 0:
                    ratio = float(old) / float(new)
                    if ratio > worst:
                        worst, worst_id = ratio, key

        return len(batch), changed, unmeasured, worst, worst_id

    @staticmethod
    def _apply(
        segmentation: ImageSegmentation, batch: list[SegmentObject]
    ) -> tuple[dict[str, float], int]:
        outcome = measure_segments(segmentation, batch)
        return (
            {str(segment.id): (segment.features or {}).get("area") for segment in batch},
            len(outcome.unmeasured),
        )

    @staticmethod
    def _preview(
        segmentation: ImageSegmentation, batch: list[SegmentObject]
    ) -> tuple[dict[str, float], int]:
        """What ``--apply`` would store, without storing it.

        Deliberately not "measure, compare, put it back": that would leave the
        database re-measured if the command were interrupted between the two
        writes, which is the one thing a dry run must not be able to do.
        """
        if not segmentation.asset_id:
            return {}, len(batch)
        try:
            target = get_asset_openable(segmentation.asset)
        except Exception:
            return {}, len(batch)

        areas: dict[str, float] = {}
        unmeasured = 0
        for segment in batch:
            polygon = segment.geometry
            if not isinstance(polygon, Polygon) or polygon.is_empty:
                unmeasured += 1
                continue
            try:
                area = measure_polygon(target, polygon, bbox=segment.bbox).get("area")
            except Exception:
                logger.warning("Could not measure segment %s", segment.id, exc_info=True)
                area = None
            if area is None:
                unmeasured += 1
            else:
                areas[str(segment.id)] = float(area)
        return areas, unmeasured

from __future__ import annotations

import logging
import time

from django.core.management.base import BaseCommand, CommandError
from django.db import OperationalError, transaction

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff import (
    merge_dirty_bboxes,
    register_overlay_mutation_all_bundles,
)
from quantem.segmentation.services.confirm_batch import (
    _ConfirmedFamily,
    _enqueue_segment_feature_refresh,
    _persist_confirmed_family,
)
from quantem.segmentation.services.confirm_batch.geometry import (
    extract_polygons,
    geometries_overlap,
    geometry_area,
    safe_intersection,
)
from quantem.segmentation.services.confirm_batch.overlap import (
    delete_manual_overlap_candidates,
    resolve_overlap_between_families,
)
from quantem.segmentation.type_definitions import MITOCHONDRIA

logger = logging.getLogger(__name__)

_SQLITE_LOCK_RETRY_ATTEMPTS = 5
_SQLITE_LOCK_RETRY_BASE_DELAY_SECONDS = 0.5


def _is_sqlite_lock_error(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


class Command(BaseCommand):
    help = (
        "Retroactively enforce non-overlapping confirmed objects and remove "
        "candidate-like objects that overlap confirmed geometry by more than 30%."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--segmentation-type",
            type=str,
            default=MITOCHONDRIA.internal_name,
            help=(
                "Segmentation type internal_name to normalize "
                f"(default: {MITOCHONDRIA.internal_name})."
            ),
        )
        parser.add_argument(
            "--image-names",
            nargs="*",
            help="Optional image display names to limit the run.",
        )

    def handle(self, *args, **options):
        segmentation_type = str(options["segmentation_type"]).strip()
        image_names = [
            str(name).strip()
            for name in (options.get("image_names") or [])
            if str(name).strip()
        ]

        qs = (
            ImageSegmentation.objects.select_related("asset", "segmentation_type")
            .filter(segmentation_type__internal_name=segmentation_type)
            .order_by("created_at")
        )
        if image_names:
            qs = qs.filter(asset__display_name__in=image_names)

        segmentations = list(qs)
        if not segmentations:
            raise CommandError(
                f"No segmentations found for type '{segmentation_type}'."
            )

        totals = {
            "created": 0,
            "updated": 0,
            "deleted": 0,
            "candidate_like_deleted": 0,
            "passes": 0,
        }
        self.stdout.write(
            "[normalize_confirmed_segments] "
            f"type={segmentation_type} count={len(segmentations)}"
        )

        for segmentation in segmentations:
            result = self._normalize_segmentation_with_retry(segmentation)
            totals["created"] += int(result["created"])
            totals["updated"] += int(result["updated"])
            totals["deleted"] += int(result["deleted"])
            totals["candidate_like_deleted"] += int(result["candidate_like_deleted"])
            totals["passes"] += int(result["passes"])
            self.stdout.write(
                "[normalize_confirmed_segments] "
                f"asset={segmentation.asset.display_name if segmentation.asset_id else segmentation.id} "
                f"segmentation={segmentation.id} "
                f"passes={result['passes']} "
                f"created={result['created']} "
                f"updated={result['updated']} "
                f"deleted={result['deleted']} "
                f"candidate_like_deleted={result['candidate_like_deleted']}"
            )

        self.stdout.write(
            "[normalize_confirmed_segments] "
            f"done created={totals['created']} updated={totals['updated']} "
            f"deleted={totals['deleted']} "
            f"candidate_like_deleted={totals['candidate_like_deleted']} "
            f"passes={totals['passes']}"
        )

    def _normalize_segmentation_with_retry(
        self,
        segmentation: ImageSegmentation,
    ) -> dict[str, int]:
        for attempt in range(_SQLITE_LOCK_RETRY_ATTEMPTS):
            try:
                return self._normalize_segmentation(segmentation)
            except OperationalError as exc:
                if not _is_sqlite_lock_error(exc) or attempt >= _SQLITE_LOCK_RETRY_ATTEMPTS - 1:
                    raise
                delay = _SQLITE_LOCK_RETRY_BASE_DELAY_SECONDS * (attempt + 1)
                self.stdout.write(
                    "[normalize_confirmed_segments] "
                    f"sqlite lock asset={segmentation.asset.display_name if segmentation.asset_id else segmentation.id} "
                    f"retry_in={delay:.1f}s attempt={attempt + 1}/{_SQLITE_LOCK_RETRY_ATTEMPTS}"
                )
                time.sleep(delay)

    def _normalize_segmentation(self, segmentation: ImageSegmentation) -> dict[str, int]:
        created = 0
        updated = 0
        deleted = 0
        candidate_like_deleted = 0
        passes = 0
        feature_refresh_ids: list[str] = []
        affected_geometries = []

        with transaction.atomic():
            confirmed_segments = list(
                SegmentObject.objects.filter(
                    segmentation=segmentation,
                    label_state="CONFIRMED",
                ).order_by("created_at")
            )
            families = [
                _ConfirmedFamily(
                    segment=segment,
                    polygons=extract_polygons(segment.geometry),
                    features=dict(segment.features)
                    if isinstance(segment.features, dict)
                    else {},
                )
                for segment in confirmed_segments
            ]

            changed = True
            while changed:
                changed = False
                passes += 1
                for index, first_family in enumerate(families):
                    first_geometry = first_family.union_geometry()
                    if first_geometry is None:
                        continue
                    first_bbox = first_geometry.envelope
                    for second_family in families[index + 1 :]:
                        second_geometry = second_family.union_geometry()
                        if second_geometry is None:
                            continue
                        if not geometries_overlap(first_bbox, second_geometry.envelope):
                            continue
                        if not geometries_overlap(first_geometry, second_geometry):
                            continue
                        overlap = safe_intersection(first_geometry, second_geometry)
                        if geometry_area(overlap) <= 1e-6:
                            continue
                        affected_geometries.append(first_geometry)
                        affected_geometries.append(second_geometry)
                        if resolve_overlap_between_families(first_family, second_family):
                            changed = True
                            affected_geometries.extend(first_family.polygons)
                            affected_geometries.extend(second_family.polygons)
                            first_geometry = first_family.union_geometry()
                            if first_geometry is None:
                                break
                            first_bbox = first_geometry.envelope

                if passes > 64:
                    raise CommandError(
                        f"Normalization did not converge for segmentation {segmentation.id}."
                    )

            candidate_deletes, candidate_geometries = delete_manual_overlap_candidates(
                segmentation=segmentation,
                manual_families=families,
            )
            deleted += candidate_deletes
            candidate_like_deleted += candidate_deletes
            affected_geometries.extend(candidate_geometries)

            for family in families:
                if family.segment is not None and not family.dirty:
                    continue
                persist_result = _persist_confirmed_family(
                    segmentation=segmentation,
                    family=family,
                )
                created += len(persist_result["created_ids"])
                updated += len(persist_result["updated_ids"])
                deleted += len(persist_result["deleted_ids"])
                feature_refresh_ids.extend(persist_result["refresh_ids"])
                affected_geometries.extend(family.polygons)

        if created > 0 or updated > 0 or deleted > 0:
            # Normalizes CONFIRMED-object geometry; confirmed objects belong to
            # every bundle, so the edit must fan out to all of them.
            register_overlay_mutation_all_bundles(
                segmentation,
                dirty_bbox=merge_dirty_bboxes(segmentation, affected_geometries),
            )

        try:
            _enqueue_segment_feature_refresh(
                segmentation_id=str(segmentation.id),
                segment_ids=feature_refresh_ids,
                recompute_features=(created > 0 or updated > 0 or deleted > 0),
            )
        except Exception as exc:
            # The normalization above is already committed and is the point of
            # this command, so a queueing failure must not undo it. But it is
            # not nothing either: the affected segments keep stale measurements
            # until something refreshes them, and saying so is the difference
            # between a known follow-up and a silent wrong number later.
            logger.warning(
                "Segments normalized, but the feature refresh for segmentation "
                "%s could not be queued (%s). Their measurements are stale "
                "until it runs.",
                segmentation.id,
                exc,
            )

        return {
            "created": created,
            "updated": updated,
            "deleted": deleted,
            "candidate_like_deleted": candidate_like_deleted,
            "passes": passes,
        }

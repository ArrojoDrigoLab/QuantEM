"""Marking a segmentation done, and undoing it.

"Mark image done" locks a segmentation *and* -- if the caller asks for it --
deletes every object a human has not confirmed. That second half is the most
destructive operation in the application: a full-image run is minutes of GPU
time and, once the candidates are gone, only another full run brings them back.

So the rules here are:

* **The destructive half never happens by default.** ``discard_unconfirmed``
  has to be sent, explicitly true, by a caller that meant it. A request that
  omits it locks the segmentation and keeps every object. There is no request
  shape that destroys work by accident, which is a stronger guarantee than a
  dialog: the dialog can be skipped, dismissed by a stray Enter, or simply not
  built yet by some future client.
* **The caller has to have counted.** ``acknowledged_discard_count`` must equal
  what is actually there. A client that shows "32 objects will be deleted" and
  then finds 40 is out of date -- an inference run finished while the dialog was
  open -- and gets a 409 with the fresh numbers instead of quietly destroying
  eight objects nobody was told about.
* **It is undoable.** The discarded objects are archived first, in the same
  transaction, and ``DELETE`` on the endpoint ("unlock segmentation") restores
  them. If the archive cannot be written the delete does not happen either.

:func:`completion_preview` is the read-only question "what would this destroy?",
which is what a confirmation dialog needs and had no way to ask.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

from django.db import transaction

from .models import ImageSegmentation, SegmentationCompletionArchive, SegmentObject

logger = logging.getLogger(__name__)

#: Objects with this label state survive completion. Everything else in the
#: segmentation is what ``discard_unconfirmed`` deletes.
KEPT_LABEL_STATE = "CONFIRMED"

#: ``ImageSegmentation.status_stage`` while a segmentation is marked done.
COMPLETED_STAGE = "COMPLETED"

#: What a refused mutation says. It states the rule, why the rule exists, and
#: the one action that lifts it -- a refusal that does not say how to proceed is
#: just a wall.
#: Shown to the user on every refusal, so it names the button and nothing else.
#: It used to end "...or DELETE this segmentation's /complete endpoint", which
#: is invariant I-12's exact failure -- an HTTP verb and a route offered as an
#: alternative to a control that is one click away. The route is still on the
#: payload, under ``unlock``, where a client can read it and a person cannot.
LOCKED_DETAIL = (
    "This segmentation is marked done, so it is locked: its objects, their "
    "labels and their measurements are final. Unlock it first if you need to "
    'change something ("Unlock segmentation" in the labeling header). '
    "Unlocking also restores whatever the completion discarded."
)


def is_locked(segmentation: ImageSegmentation) -> bool:
    """True when this segmentation refuses mutations.

    The dialog has always said *"Marking it done locks the segmentation"*, and
    for a while nothing enforced it: a ``COMPLETED`` segmentation still accepted
    relabelling every object, adding completed ROIs and re-running full
    segmentation, and only the button caption changed. "Done" is the state a lab
    relies on to mean the numbers are final, so it is enforced rather than
    reworded. It is not a one-way door: ``DELETE /complete`` unlocks.
    """
    return segmentation.status_stage == COMPLETED_STAGE


def locked_payload(segmentation: ImageSegmentation) -> dict:
    """Body for the refusal, carrying the way out in a machine-readable form."""
    return {
        "detail": LOCKED_DETAIL,
        "segmentation_id": str(segmentation.id),
        "status_stage": segmentation.status_stage,
        "locked": True,
        "unlock": {
            "method": "DELETE",
            "path": f"/api/segmentations/{segmentation.id}/complete",
        },
    }

#: Above this many objects the discarded set is not archived. A restore has to
#: hold the whole snapshot in memory and write it as one row; past some size
#: that trade is worse than the undo is worth. The API reports
#: ``restorable: false`` rather than implying an undo it cannot deliver.
DEFAULT_ARCHIVE_MAX_OBJECTS = 20000

#: Second, independent ceiling on the serialised snapshot. Object count is a
#: poor proxy for size -- one traced ER network can outweigh a thousand lipid
#: droplets -- so the snapshot is measured as it is built and abandoned if it
#: gets too big to be a sensible database row.
DEFAULT_ARCHIVE_MAX_BYTES = 32 * 1024 * 1024

_SNAPSHOT_FIELDS = (
    "id",
    "label_state",
    "refined",
    "status",
    "source_model",
    "confidence_score",
    "features",
    "base_segment_id",
    "geometry_wkb",
    "centroid_x",
    "centroid_y",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
)


def _env_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def archive_max_objects() -> int:
    return _env_int("QUANTEM_COMPLETION_ARCHIVE_MAX_OBJECTS", DEFAULT_ARCHIVE_MAX_OBJECTS)


def archive_max_bytes() -> int:
    return _env_int("QUANTEM_COMPLETION_ARCHIVE_MAX_BYTES", DEFAULT_ARCHIVE_MAX_BYTES)


def discardable_queryset(segmentation: ImageSegmentation):
    """Everything ``discard_unconfirmed`` would delete."""
    return SegmentObject.objects.filter(segmentation=segmentation).exclude(
        label_state=KEPT_LABEL_STATE
    )


def completion_preview(segmentation: ImageSegmentation) -> dict:
    """What marking this segmentation done would destroy. Changes nothing.

    Served by ``GET /api/segmentations/<id>/complete`` and embedded in every
    refusal from ``POST``, so a client never has to guess a count or make a
    second round trip to find out why it was refused.
    """
    label_states = [choice[0] for choice in SegmentObject.LABEL_STATE_CHOICES]
    by_label_state = {
        state: 0 for state in label_states if state != KEPT_LABEL_STATE
    }
    by_source_model: dict[str, int] = {}

    rows = (
        discardable_queryset(segmentation)
        .values_list("label_state", "source_model")
        .order_by()
    )
    discard_count = 0
    for label_state, source_model in rows.iterator():
        discard_count += 1
        by_label_state[str(label_state)] = by_label_state.get(str(label_state), 0) + 1
        key = str(source_model or "")
        if key:
            by_source_model[key] = by_source_model.get(key, 0) + 1

    confirmed_count = SegmentObject.objects.filter(
        segmentation=segmentation,
        label_state=KEPT_LABEL_STATE,
    ).count()

    archive = last_archive(segmentation)
    return {
        "segmentation_id": str(segmentation.id),
        "status_stage": segmentation.status_stage,
        "is_complete": segmentation.status_stage == "COMPLETED",
        "confirmed_count": int(confirmed_count),
        "discard_count": int(discard_count),
        "discard_by_label_state": by_label_state,
        "discard_by_source_model": by_source_model,
        # A prediction, not a promise: the snapshot is also size-capped, and the
        # POST response reports what actually happened.
        "restorable": discard_count <= archive_max_objects(),
        "archive_max_objects": archive_max_objects(),
        "restorable_count": int(archive.discarded_count) if archive is not None else 0,
    }


def last_archive(
    segmentation: ImageSegmentation,
) -> SegmentationCompletionArchive | None:
    """The most recent completion archive for this segmentation, if any."""
    return (
        SegmentationCompletionArchive.objects.filter(segmentation=segmentation)
        .order_by("-created_at")
        .first()
    )


def _snapshot_row(segment: SegmentObject) -> dict:
    wkb = segment.geometry_wkb
    return {
        "id": str(segment.id),
        "label_state": segment.label_state,
        "refined": segment.refined,
        "status": int(segment.status),
        "source_model": segment.source_model,
        "confidence_score": (
            float(segment.confidence_score)
            if segment.confidence_score is not None
            else None
        ),
        "features": segment.features if isinstance(segment.features, dict) else {},
        "base_segment_id": (
            str(segment.base_segment_id) if segment.base_segment_id else None
        ),
        "geometry_wkb": bytes(wkb).hex() if wkb else "",
        "centroid_x": float(segment.centroid_x),
        "centroid_y": float(segment.centroid_y),
        "bbox_minx": float(segment.bbox_minx),
        "bbox_miny": float(segment.bbox_miny),
        "bbox_maxx": float(segment.bbox_maxx),
        "bbox_maxy": float(segment.bbox_maxy),
    }


def build_snapshot(segmentation: ImageSegmentation) -> tuple[list[dict], bool]:
    """Serialise the discardable objects. Returns ``(rows, complete)``.

    ``complete`` is False when the set blew either ceiling; the partial rows are
    thrown away, because half an undo is worse than none -- it would silently
    return some of a user's objects and not others.
    """
    max_bytes = archive_max_bytes()
    max_objects = archive_max_objects()
    rows: list[dict] = []
    total_bytes = 0

    queryset = discardable_queryset(segmentation).only(*_SNAPSHOT_FIELDS)
    for segment in queryset.iterator(chunk_size=500):
        if len(rows) >= max_objects:
            return [], False
        row = _snapshot_row(segment)
        total_bytes += len(json.dumps(row))
        if total_bytes > max_bytes:
            return [], False
        rows.append(row)
    return rows, True


def _delete_by_id(ids: list[str]) -> int:
    """Delete exactly these objects, in batches SQLite will accept."""
    deleted = 0
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        removed, _ = SegmentObject.objects.filter(id__in=chunk).delete()
        deleted += int(removed)
    return deleted


def _restore_rows(
    segmentation: ImageSegmentation,
    rows: list[dict],
) -> int:
    """Recreate archived objects under their original ids. Returns the count.

    ``bulk_create`` deliberately, not ``save()``: the stored values are already
    normalised, and re-running lifecycle normalisation on the way back in could
    quietly change a label or a source model between what was archived and what
    is restored.
    """
    existing = set(
        SegmentObject.objects.filter(
            segmentation=segmentation,
            id__in=[row["id"] for row in rows],
        ).values_list("id", flat=True)
    )

    instances: list[SegmentObject] = []
    deferred_bases: list[tuple[str, str]] = []
    for row in rows:
        try:
            segment_id = uuid.UUID(str(row["id"]))
        except (TypeError, ValueError):
            logger.warning("Skipping archived segment with unusable id %r", row.get("id"))
            continue
        if segment_id in existing:
            # Something recreated this id since the archive was written. Leave
            # the live row alone rather than overwriting it.
            continue
        instances.append(
            SegmentObject(
                id=segment_id,
                segmentation=segmentation,
                geometry_wkb=bytes.fromhex(str(row.get("geometry_wkb") or "")),
                centroid_x=float(row["centroid_x"]),
                centroid_y=float(row["centroid_y"]),
                bbox_minx=float(row["bbox_minx"]),
                bbox_miny=float(row["bbox_miny"]),
                bbox_maxx=float(row["bbox_maxx"]),
                bbox_maxy=float(row["bbox_maxy"]),
                label_state=str(row["label_state"]),
                refined=str(row.get("refined") or "UNREFINED"),
                status=int(row["status"]),
                source_model=str(row.get("source_model") or ""),
                confidence_score=row.get("confidence_score"),
                features=row.get("features") or {},
                base_segment_id=None,
            )
        )
        if row.get("base_segment_id"):
            deferred_bases.append((str(row["id"]), str(row["base_segment_id"])))

    SegmentObject.objects.bulk_create(instances, batch_size=500)

    # Second pass so a family can be restored whatever order its members are in,
    # and so a link to an object that is genuinely gone becomes null rather than
    # an integrity error.
    if deferred_bases:
        present = set(
            SegmentObject.objects.filter(
                segmentation=segmentation,
                id__in=[base_id for _, base_id in deferred_bases],
            ).values_list("id", flat=True)
        )
        for segment_id, base_id in deferred_bases:
            try:
                base_uuid = uuid.UUID(base_id)
            except (TypeError, ValueError):
                continue
            if base_uuid in present:
                SegmentObject.objects.filter(id=segment_id).update(
                    base_segment_id=base_uuid
                )

    return len(instances)


@transaction.atomic
def archive_and_discard(segmentation: ImageSegmentation) -> dict:
    """Archive then delete every non-confirmed object. Returns what happened.

    Atomic on purpose: if the snapshot cannot be written, nothing is deleted.
    A completion that destroys work and fails to record it is the exact failure
    this whole module exists to prevent.
    """
    rows, complete = build_snapshot(segmentation)
    discarded_count = (
        len(rows) if complete else discardable_queryset(segmentation).count()
    )

    # One archive per segmentation: unlock restores the most recent, so older
    # ones are dead weight.
    SegmentationCompletionArchive.objects.filter(segmentation=segmentation).delete()

    archive = None
    if discarded_count:
        archive = SegmentationCompletionArchive.objects.create(
            segmentation=segmentation,
            discarded_count=discarded_count,
            objects_json=rows if complete else [],
        )
        if not complete:
            logger.warning(
                "Segmentation %s discarded %d object(s) without an archive "
                "(over the archive ceiling); this completion is not undoable.",
                segmentation.id,
                discarded_count,
            )

    if complete:
        # Delete exactly what was archived, by id. Deleting by predicate instead
        # would also take anything a worker wrote between the snapshot and the
        # delete -- objects that would then be gone with no record of them.
        deleted = _delete_by_id([str(row["id"]) for row in rows])
    else:
        deleted, _ = discardable_queryset(segmentation).delete()
    return {
        "discarded_count": int(discarded_count),
        "restorable": bool(complete and discarded_count and archive is not None),
        "archive_id": str(archive.id) if archive is not None else None,
        "deleted_rows": int(deleted),
    }


@transaction.atomic
def restore_last_archive(segmentation: ImageSegmentation) -> dict:
    """Put back what the last completion discarded, if it can be put back."""
    archive = last_archive(segmentation)
    if archive is None:
        return {"restored_count": 0, "restorable": False, "archived_count": 0}

    rows = archive.objects_json if isinstance(archive.objects_json, list) else []
    archived_count = int(archive.discarded_count)
    if not rows:
        # The completion was over the archive ceiling. Say so instead of
        # reporting a successful restore of nothing.
        return {
            "restored_count": 0,
            "restorable": False,
            "archived_count": archived_count,
        }

    restored = _restore_rows(segmentation, rows)
    archive.delete()
    return {
        "restored_count": int(restored),
        "restorable": True,
        "archived_count": archived_count,
    }


__all__ = [
    "COMPLETED_STAGE",
    "DEFAULT_ARCHIVE_MAX_BYTES",
    "DEFAULT_ARCHIVE_MAX_OBJECTS",
    "KEPT_LABEL_STATE",
    "LOCKED_DETAIL",
    "archive_and_discard",
    "archive_max_bytes",
    "archive_max_objects",
    "build_snapshot",
    "completion_preview",
    "discardable_queryset",
    "is_locked",
    "last_archive",
    "locked_payload",
    "restore_last_archive",
]

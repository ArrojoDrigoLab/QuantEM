"""Durable, recoverable annotation snapshots. Probability previews are not archived."""

import io
import zipfile

from django.core import serializers
from django.db import transaction
from django.utils import timezone

from quantem.core.config import DATA_DIR
from quantem.core.local_storage import storage_path, storage_relpath_for_path
from quantem.segmentation.models import (
    AnalysisMaskObject,
    CompletedROI,
    CorrectionRevision,
    GlobalMask,
    ImageSegmentation,
    SegmentationResultVersion,
    SegmentObject,
)

SNAPSHOT_MODELS = (SegmentObject, CompletedROI, AnalysisMaskObject, GlobalMask)


@transaction.atomic
def save_revision(segmentation, name):
    revision = CorrectionRevision.objects.create(
        segmentation=segmentation,
        name=name[:255],
        object_count=SegmentObject.objects.filter(
            segmentation=segmentation, superseded_at__isnull=True
        ).count(),
        include_level=segmentation.include_level,
    )
    directory = DATA_DIR / "correction_revisions" / str(segmentation.id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{revision.id}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for model in SNAPSHOT_MODELS:
            with archive.open(model._meta.model_name + ".jsonl", "w") as output:
                for row in (
                    model.objects.filter(segmentation=segmentation)
                    .order_by("id")
                    .iterator(chunk_size=500)
                ):
                    payload = serializers.serialize("json", [row])
                    output.write(payload.encode() + b"\n")
        mask = GlobalMask.objects.filter(segmentation=segmentation).first()
        if mask:
            source = storage_path(mask.file_path)
            if not source.is_file():
                raise ValueError("The mask is missing. Its current state cannot be saved safely.")
            archive.write(source, "global-mask.png")
    revision.archive_path = storage_relpath_for_path(path)
    revision.save(update_fields=["archive_path"])
    return revision


def has_revision_content(segmentation):
    return (
        SegmentObject.objects.filter(segmentation=segmentation).exists()
        or GlobalMask.objects.filter(segmentation=segmentation).exists()
        or CompletedROI.objects.filter(segmentation=segmentation).exists()
    )


@transaction.atomic
def restore_revision(revision):
    segmentation = ImageSegmentation.objects.select_for_update().get(id=revision.segmentation_id)
    path = storage_path(revision.archive_path)
    if not path.is_file():
        raise ValueError("This revision's archive is unavailable.")
    # A restore is reversible too; never throw away the work made since the snapshot.
    save_revision(segmentation, "Before restoring " + revision.name)
    with zipfile.ZipFile(path) as archive:
        SegmentObject.objects.filter(segmentation=segmentation, superseded_at__isnull=True).update(
            superseded_at=timezone.now()
        )
        for model in (CompletedROI, AnalysisMaskObject, GlobalMask):
            model.objects.filter(segmentation=segmentation).delete()
        for model in SNAPSHOT_MODELS:
            with archive.open(model._meta.model_name + ".jsonl") as stream:
                for line in io.TextIOWrapper(stream, encoding="utf-8"):
                    for restored in serializers.deserialize("json", line):
                        if str(restored.object.segmentation_id) != str(segmentation.id):
                            raise ValueError("This archive does not belong to the segmentation.")
                        restored.save()
        mask = GlobalMask.objects.filter(segmentation=segmentation).first()
        if mask:
            # Restore to an immutable revision-owned copy. Future edits write the
            # normal mask path, leaving this copy available for another restore.
            mask_path = path.with_suffix(".png")
            with archive.open("global-mask.png") as source, mask_path.open("wb") as output:
                import shutil

                shutil.copyfileobj(source, output)
            mask.file_path = storage_relpath_for_path(mask_path)
            mask.save(update_fields=["file_path", "updated_at"])
    segmentation.include_level = revision.include_level
    segmentation.status_stage = "CANDIDATES_READY"
    segmentation.status_error = ""
    segmentation.final_result_provenance = {}
    segmentation.save(
        update_fields=[
            "include_level",
            "status_stage",
            "status_error",
            "final_result_provenance",
            "updated_at",
        ]
    )
    SegmentationResultVersion.record_new_result(
        segmentation,
        run_identity={"restored_revision": str(revision.id)},
        include_level=revision.include_level,
    )
    from quantem.segmentation.overlay_ngff.dirty import full_image_dirty_bbox
    from quantem.segmentation.overlay_ngff.mutations import register_overlay_mutation_all_bundles

    return register_overlay_mutation_all_bundles(
        segmentation,
        dirty_bbox=full_image_dirty_bbox(segmentation),
        allow_sync_partial=False,
    )

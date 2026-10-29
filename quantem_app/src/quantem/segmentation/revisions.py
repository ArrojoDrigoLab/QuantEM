"""Durable, recoverable annotation snapshots. Probability previews are not archived."""

import io
import json
import shutil
import zipfile
from pathlib import Path

from django.core import serializers
from django.db import transaction
from django.utils import timezone

from quantem.assets.canonical_decode import decode_canonical_array
from quantem.core.config import DATA_DIR, GLOBAL_MASKS_DIR
from quantem.core.immutable_files import write_immutable_file
from quantem.core.local_storage import storage_path, storage_relpath_for_path
from quantem.segmentation.models import (
    AnalysisMaskObject,
    AnalysisMaskObjectAnnotation,
    CompletedROI,
    CorrectionRevision,
    GlobalMask,
    ImageSegmentation,
    RoiSegmentationStatus,
    SegmentationResultVersion,
    SegmentObject,
    SegmentObjectAnnotation,
)
from quantem.segmentation.mutations import record_change
from quantem.segmentation.revision_rois import restore_roi_reviews, save_roi_reviews

SNAPSHOT_MODELS = (SegmentObject, CompletedROI, AnalysisMaskObject, GlobalMask)
ANNOTATION_MODELS = (SegmentObjectAnnotation, AnalysisMaskObjectAnnotation)
MANIFEST = "revision.json"
FORMAT_VERSION = 2


@transaction.atomic
def save_revision(segmentation, name, *, allow_missing_mask=False):
    segmentation = ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
    mask = GlobalMask.objects.filter(segmentation=segmentation).first()
    source = storage_path(mask.file_path) if mask else None
    missing_mask = source is not None and not source.is_file()
    if missing_mask and not allow_missing_mask:
        raise ValueError("The mask is missing. Its current state cannot be saved safely.")
    if missing_mask:
        # A recovery must not depend on the damaged current file. Preserve all
        # remaining records in an explicitly incomplete automatic backup.
        name = name[:220] + " (mask unavailable)"
    revision = CorrectionRevision.objects.create(
        segmentation=segmentation,
        name=name[:255],
        object_count=SegmentObject.objects.filter(
            segmentation=segmentation, superseded_at__isnull=True
        ).count()
        + AnalysisMaskObject.objects.filter(segmentation=segmentation).count(),
        include_level=segmentation.include_level,
    )
    canonical_id = (
        segmentation.saved_results.filter(is_canonical=True).values_list("id", flat=True).first()
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "segmentation_id": str(segmentation.id),
        "manual_full_image": segmentation.manual_full_image,
        "area_vectors_authoritative": segmentation.area_vectors_authoritative
        or AnalysisMaskObject.objects.filter(segmentation=segmentation).exists(),
        "canonical_result_id": str(canonical_id) if canonical_id else None,
        "final_result_provenance": segmentation.final_result_provenance,
        "mask_unavailable": missing_mask,
    }

    def write(path):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST, json.dumps(manifest))
            save_roi_reviews(archive, segmentation)
            for model in (*SNAPSHOT_MODELS, *ANNOTATION_MODELS):
                owner = "object__segmentation" if model in ANNOTATION_MODELS else "segmentation"
                rows = model.objects.filter(**{owner: segmentation}).order_by("id")
                if model is SegmentObject:
                    rows = rows.filter(superseded_at__isnull=True)
                elif model is SegmentObjectAnnotation:
                    rows = rows.filter(object__superseded_at__isnull=True)
                with archive.open(model._meta.model_name + ".jsonl", "w") as output:
                    for row in rows.iterator(chunk_size=500):
                        output.write(serializers.serialize("json", [row]).encode() + b"\n")
            if source is not None and not missing_mask:
                archive.write(source, "global-mask.png")

    path = write_immutable_file(
        DATA_DIR / "correction_revisions" / str(segmentation.id),
        prefix=str(revision.id),
        suffix=".zip",
        write=write,
    )
    revision.archive_path = storage_relpath_for_path(path)
    revision.save(update_fields=["archive_path"])
    return revision


def has_revision_content(segmentation):
    return (
        any(model.objects.filter(segmentation=segmentation).exists() for model in SNAPSHOT_MODELS)
        or RoiSegmentationStatus.objects.filter(segmentation=segmentation).exists()
        or segmentation.manual_full_image
        or segmentation.area_vectors_authoritative
        or segmentation.saved_results.filter(is_canonical=True).exists()
    )


def _rows(archive, model):
    with archive.open(model._meta.model_name + ".jsonl") as stream:
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            for restored in serializers.deserialize("json", line):
                if type(restored.object) is not model:
                    raise ValueError("Unexpected record type in revision archive.")
                yield restored


def _manifest(archive, segmentation):
    if MANIFEST not in archive.namelist():
        # Legacy archives cannot establish whole-image/model coverage. They
        # still restore their explicit reviewed regions and materialized masks.
        return {}
    manifest = json.loads(archive.read(MANIFEST))
    if not isinstance(manifest, dict):
        raise ValueError("This revision's manifest is invalid.")
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("segmentation_id") != str(
        segmentation.id
    ):
        raise ValueError("Unsupported revision archive or mismatched segmentation.")
    return manifest


def restore_revision(revision):
    try:
        return _restore_revision(revision)
    except (zipfile.BadZipFile, KeyError, EOFError) as exc:
        raise ValueError("This revision's archive is damaged or incomplete.") from exc


@transaction.atomic
def _restore_revision(revision):
    segmentation = ImageSegmentation.objects.select_for_update().get(id=revision.segmentation_id)
    path = storage_path(revision.archive_path)
    if not path.is_file():
        raise ValueError("This revision's archive is unavailable.")
    previous_mask = GlobalMask.objects.filter(segmentation=segmentation).first()
    with zipfile.ZipFile(path) as archive:
        manifest = _manifest(archive, segmentation)
        if archive.testzip() is not None:
            raise ValueError("This revision's archive is damaged.")
        if manifest.get("mask_unavailable"):
            raise ValueError(
                "This backup's mask was unavailable when saved; its pixels cannot be restored."
            )
        # Validate the requested archive before writing another potentially
        # large backup. Recovery still works if the current mask has gone missing.
        save_revision(segmentation, "Before restoring " + revision.name, allow_missing_mask=True)
        record_change(segmentation, geometry=True)
        restore_roi_reviews(archive, segmentation)
        SegmentObject.objects.filter(segmentation=segmentation, superseded_at__isnull=True).update(
            superseded_at=timezone.now()
        )
        for model in (CompletedROI, AnalysisMaskObject, GlobalMask):
            model.objects.filter(segmentation=segmentation).delete()
        restored_ids = {}
        for model in SNAPSHOT_MODELS:
            restored_ids[model] = set()
            for restored in _rows(archive, model):
                obj = restored.object
                if (
                    str(obj.segmentation_id) != str(segmentation.id)
                    or model.objects.filter(pk=obj.pk).exclude(segmentation=segmentation).exists()
                ):
                    raise ValueError("This archive does not belong to the segmentation.")
                restored.save()
                restored_ids[model].add(obj.pk)
        for model in ANNOTATION_MODELS:
            parent = model._meta.get_field("object").related_model
            ids = restored_ids[parent]
            model.objects.filter(object_id__in=ids).delete()
            if model._meta.model_name + ".jsonl" not in archive.namelist():
                continue
            for restored in _rows(archive, model):
                obj = restored.object
                if (
                    obj.object_id not in ids
                    or model.objects.filter(pk=obj.pk).exclude(object_id__in=ids).exists()
                ):
                    raise ValueError("An annotation does not belong to this revision.")
                restored.save()
        mask = GlobalMask.objects.filter(segmentation=segmentation).first()
        if mask:

            def write_mask(destination):
                with archive.open("global-mask.png") as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                image = decode_canonical_array(destination)
                if image.shape != (mask.height, mask.width):
                    raise ValueError("The archived mask dimensions do not match its record.")

            # Every restore gets a new file, including repeated restores of the
            # same snapshot. A failed copy cannot damage a published mask.
            mask_path = write_immutable_file(
                GLOBAL_MASKS_DIR / str(segmentation.id),
                prefix="restore",
                suffix=".tif" if Path(mask.file_path).suffix.lower() == ".tif" else ".png",
                write=write_mask,
            )
            mask.file_path = storage_relpath_for_path(mask_path)
            mask.save(update_fields=["file_path", "updated_at"])
    if previous_mask is not None:
        from quantem.segmentation.artifact_cleanup import retire_artifact

        retire_artifact(storage_path(previous_mask.file_path))
    segmentation.saved_results.filter(is_canonical=True).update(is_canonical=False)
    canonical_id = manifest.get("canonical_result_id")
    if canonical_id and not segmentation.saved_results.filter(pk=canonical_id).update(
        is_canonical=True, deleted_at=None
    ):
        raise ValueError("The saved result required by this revision is unavailable.")
    segmentation.manual_full_image = manifest.get("manual_full_image", False)
    segmentation.area_vectors_authoritative = manifest.get(
        "area_vectors_authoritative", bool(restored_ids[AnalysisMaskObject])
    )
    segmentation.include_level = revision.include_level
    segmentation.status_stage = "CANDIDATES_READY"
    segmentation.status_error = ""
    segmentation.final_result_provenance = manifest.get("final_result_provenance", {})
    segmentation.save(
        update_fields=[
            "manual_full_image",
            "area_vectors_authoritative",
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
        force_full_rebuild=True,
        allow_sync_partial=False,
    )

"""Validate, then publish one complete measurement bundle for every runner."""

from copy import deepcopy
from uuid import uuid4

from django.db import transaction
from django.utils import timezone

from quantem.assets.models import Asset
from quantem.core.immutable_files import publish_directory
from quantem.segmentation.models import ImageSegmentation

from .identity import input_fingerprint
from .input_contract import dependency_ids
from .models import AnalysisRun


def input_guard(run):
    params = deepcopy(run.params)
    fingerprint = input_fingerprint(run.segmentation, params)
    if run.input_fingerprint and run.input_fingerprint != fingerprint:
        raise ValueError("Inputs changed while this analysis was queued. Start a new batch.")
    return params, fingerprint


def staging_directory(run):
    from .service import export_dir_for_run

    target = export_dir_for_run(run.id)
    return target.with_name(f".pending-{run.id}-{uuid4().hex}")


def publish(
    run, directory, *, expected, results, cancel_check=None, apply_output_selection=False, **updates
):
    from .export_selection import select_export_files
    from .service import export_dir_for_run

    # Filtering is part of preparation. The database still says RUNNING and
    # exposes no export while files or their input identity may still change.
    previous = run.export_dir
    previous_params = run.params
    run.export_dir = str(directory)
    run.params = expected[0]
    try:
        if apply_output_selection:
            select_export_files(run)
    finally:
        run.export_dir = previous
        run.params = previous_params
    target = export_dir_for_run(run.id)
    moved = False
    try:
        with transaction.atomic():
            params, fingerprint = expected
            ids = dependency_ids(run.segmentation_id, params)
            list(ImageSegmentation.objects.select_for_update().filter(id__in=ids).order_by("id"))
            asset = Asset.objects.select_for_update().get(pk=run.segmentation.asset_id)
            metadata = getattr(asset, "workflow_metadata", None)
            if metadata is not None:
                type(metadata).objects.select_for_update().get(pk=metadata.pk)
            current = AnalysisRun.objects.select_for_update().get(pk=run.pk)
            if current.status != AnalysisRun.STATUS_RUNNING:
                raise ValueError("This analysis is no longer running.")
            if input_fingerprint(run.segmentation, params) != fingerprint:
                raise ValueError(
                    "Inputs changed during measurement. Re-run to measure one consistent version."
                )
            if cancel_check is not None:
                cancel_check()
            if target.exists():
                raise ValueError("This run already has an export directory; start a new analysis.")
            publish_directory(directory, target)
            moved = True
            values = dict(
                status=AnalysisRun.STATUS_SUCCESS,
                results=results,
                export_dir=str(target),
                finished_at=timezone.now(),
                error="",
                input_fingerprint=(
                    fingerprint
                    if updates.get("params", params) == params
                    else input_fingerprint(run.segmentation, updates["params"])
                ),
                **updates,
            )
            AnalysisRun.objects.filter(pk=run.pk).update(**values)
        for key, value in values.items():
            setattr(run, key, value)
    except Exception:
        # A crash can leave an unreferenced directory, never a SUCCESS row
        # pointing at an incomplete one. The artifact audit reports leftovers.
        if moved:
            import shutil

            shutil.rmtree(target, ignore_errors=True)
        raise

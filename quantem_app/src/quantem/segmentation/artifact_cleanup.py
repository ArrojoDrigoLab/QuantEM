"""Retryable reclamation of files retired by scientific result transactions."""

import os
import shutil
from pathlib import Path

from django.db import transaction
from django.db.models import Q, Value
from django.db.models.functions import Replace

from quantem.analysis.models import AnalysisRun
from quantem.core.config import DATA_DIR, GLOBAL_MASKS_DIR, PROB_MAPS_DIR, STORAGE_DIR, TMP_DIR
from quantem.core.local_storage import storage_path, storage_relpath_for_path
from quantem.segmentation.models import (
    CorrectionRevision,
    GlobalMask,
    ProbabilityMap,
    RetiredResultArtifact,
)


def referenced_paths():
    for model, field in (
        (CorrectionRevision, "archive_path"),
        (GlobalMask, "file_path"),
        (ProbabilityMap, "file_path"),
    ):
        for value in model.objects.exclude(**{field: ""}).values_list(field, flat=True).iterator():
            yield storage_path(value).resolve()
    for value in (
        AnalysisRun.objects.exclude(export_dir="").values_list("export_dir", flat=True).iterator()
    ):
        yield Path(value).resolve()


def _is_referenced(path: Path, *, is_directory: bool) -> bool:
    """Ask the database about this path without resolving every saved file.

    Stored artifact paths are validated storage-relative paths; exports use
    absolute paths. Normalize Windows separators for older records as well.
    A directory match must end at a separator, never at a similar-name sibling.
    """
    for model, field, target in (
        (CorrectionRevision, "archive_path", storage_relpath_for_path(path)),
        (GlobalMask, "file_path", storage_relpath_for_path(path)),
        (ProbabilityMap, "file_path", storage_relpath_for_path(path)),
        (AnalysisRun, "export_dir", path.as_posix()),
    ):
        # Path equality on Windows is case insensitive, like Path.resolve.
        exact, prefix = ("iexact", "istartswith") if os.name == "nt" else ("exact", "startswith")
        query = Q(**{f"_artifact_path__{exact}": target})
        if is_directory:
            query |= Q(**{f"_artifact_path__{prefix}": target.rstrip("/") + "/"})
        if (
            model.objects.alias(_artifact_path=Replace(field, Value("\\"), Value("/")))
            .filter(query)
            .exists()
        ):
            return True
    return False


def managed_path(path):
    path = Path(path)
    resolved = path.resolve()
    root = STORAGE_DIR.resolve()
    if resolved == root or not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError("Result cleanup is restricted to owned storage paths.")
    roots = (
        GLOBAL_MASKS_DIR,
        PROB_MAPS_DIR,
        DATA_DIR / "correction_revisions",
        TMP_DIR / "prob_maps",
        TMP_DIR / "segmentation_overlays",
        STORAGE_DIR / "exports",
    )
    if not any(
        resolved != base.resolve() and resolved.is_relative_to(base.resolve()) for base in roots
    ):
        raise ValueError("Only individual result artifacts can be retired.")
    return resolved


@transaction.atomic
def retire_artifact(path, *, is_directory=False):
    path = managed_path(path)
    row, _ = RetiredResultArtifact.objects.get_or_create(
        path=storage_relpath_for_path(path), defaults={"is_directory": is_directory}
    )
    transaction.on_commit(lambda: drain_retired_artifacts(ids=[row.pk]), robust=True)


def drain_retired_artifacts(*, ids=None):
    queryset = RetiredResultArtifact.objects.all()
    if ids is not None:
        queryset = queryset.filter(pk__in=ids)
    removed = 0
    for pk in queryset.values_list("pk", flat=True):
        try:
            with transaction.atomic():
                row = RetiredResultArtifact.objects.select_for_update().filter(pk=pk).first()
                if row is None:
                    continue
                path = managed_path(storage_path(row.path))
                if _is_referenced(path, is_directory=row.is_directory):
                    continue
                if row.is_directory:
                    if path.exists():
                        shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
                row.delete()
                removed += 1
        except (OSError, ValueError) as exc:
            RetiredResultArtifact.objects.filter(pk=pk).update(last_error=str(exc))
    return removed

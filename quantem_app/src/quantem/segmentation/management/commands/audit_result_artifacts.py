"""Inspect scientific artifacts without interpreting derived files as results."""

import json
import time

from django.core.management.base import BaseCommand

from quantem.core.config import DATA_DIR, GLOBAL_MASKS_DIR, PROB_MAPS_DIR, STORAGE_DIR, TMP_DIR
from quantem.segmentation.artifact_cleanup import drain_retired_artifacts, referenced_paths
from quantem.segmentation.models import ImageSegmentation, RetiredResultArtifact
from quantem.segmentation.saved_results import cleanup_deleted_results


class Command(BaseCommand):
    help = (
        "Report missing/referenced/unreferenced result files; optionally retry committed deletions."
    )

    def add_arguments(self, parser):
        parser.add_argument("--retry-cleanup", action="store_true")
        parser.add_argument("--min-age-hours", type=float, default=24)

    def handle(self, *args, **options):
        if options["min_age_hours"] < 0:
            raise ValueError("Minimum age must be nonnegative.")
        if options["retry_cleanup"]:
            for segmentation in (
                ImageSegmentation.objects.filter(saved_results__deleted_at__isnull=False)
                .distinct()
                .iterator()
            ):
                cleanup_deleted_results(
                    segmentation,
                    result_ids=segmentation.saved_results.filter(
                        deleted_at__isnull=False
                    ).values_list("pk", flat=True),
                )
            drain_retired_artifacts()
        references = set(referenced_paths())
        # Mark referenced ancestors once. Repeating a full reference scan for
        # each candidate makes a large library's read-only audit quadratic.
        occupied = references.copy()
        root_path = STORAGE_DIR.resolve()
        for reference in references:
            occupied.update(
                parent for parent in reference.parents if parent.is_relative_to(root_path)
            )
        cutoff = time.time() - options["min_age_hours"] * 3600
        candidates = []
        for root, pattern in (
            (GLOBAL_MASKS_DIR, "*/*"),
            (DATA_DIR / "correction_revisions", "*/*"),
            (PROB_MAPS_DIR, "*/*"),
            (TMP_DIR / "prob_maps", "*/*/*"),
            (STORAGE_DIR / "exports", "*"),
        ):
            if root.is_dir():
                candidates.extend(root.glob(pattern))
        unreferenced = []
        for path in candidates:
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_relative_to(root_path):
                continue
            try:
                modified = path.stat().st_mtime
            except FileNotFoundError:
                # A committed deletion may finish while this audit is walking.
                continue
            if modified >= cutoff:
                continue
            if resolved not in occupied:
                unreferenced.append(str(path))
        self.stdout.write(
            json.dumps(
                {
                    "missing": sorted(str(path) for path in references if not path.exists()),
                    "unreferenced": sorted(unreferenced),
                    "pending_deletions": list(
                        RetiredResultArtifact.objects.values("path", "last_error")
                    ),
                    "note": "Unreferenced files may be active staging or derived caches; this audit never deletes them.",
                },
                indent=2,
            )
        )

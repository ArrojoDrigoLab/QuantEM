"""User-facing analysis notes; model and resolution details live in saved settings."""

from __future__ import annotations

import re
from typing import Any

PARTIAL_METRICS = re.compile(r"^\d+ of \d+ metrics (?:is|are) measured on fewer than")


def partial_metric_note(summary: dict[str, Any], total: int) -> str | None:
    # A missing probability map does not invalidate a manual annotation.
    # Its score and coverage remain blank/zero in the table.
    partial = {
        key: stats
        for key, stats in summary.items()
        if key != "mean_prob" and stats.get("n_missing")
    }
    if not partial:
        return None
    reasons: dict[str, list[str]] = {}
    for key, stats in partial.items():
        reason = stats.get("note") or stats.get("unreportable_reason")
        if reason and reason.startswith("Measured on "):
            reason = reason.partition("; ")[2] or reason
        reason = reason or f"{stats['n_missing']} objects have no saved value for this metric."
        reasons.setdefault(reason, []).append(key)
    explanations = [
        f"{keys[0] if len(keys) == 1 else f'{len(keys)} metrics'}: {reason}"
        for reason, keys in reasons.items()
    ]
    return (
        f"{len(partial)} of {len(summary)} metrics "
        f"{'is' if len(partial) == 1 else 'are'} measured on fewer than the {total} "
        f"confirmed objects ({', '.join(partial)}). " + " ".join(explanations)
    )


def analysis_notes(result: dict[str, Any] | None) -> list[str]:
    """Apply the current message allowlist without rewriting historical results."""
    result = result or {}
    objects = result.get("objects") or {}
    notes = []
    mixed: dict[str, set[str]] = {}
    for original in result.get("caveats") or []:
        note = None
        if original.startswith("No tissue mask was supplied:"):
            note = original
        elif PARTIAL_METRICS.match(original):
            note = partial_metric_note(objects.get("summary") or {}, objects.get("n", 0))
        elif match := re.match(
            r"(The objects in .+? come from \d+ different inference runs)[,.]", original
        ):
            note = match[1] + "."
        elif "were segmented under multiple models and settings." in original:
            note = original
        elif match := re.match(
            r"The (?:(.+?) )?objects in (.+?) were not all (?:produced|filtered) (?:under|at) the same",
            original,
        ):
            mixed.setdefault(match[2], set()).add(match[1] or "model-generated")
        elif original.startswith(("The tissue mask is empty:", "The measurement area is empty.")):
            note = "The measurement area is empty. Area fractions and point statistics are unavailable."
        elif match := re.match(
            r"(\d+) of (\d+) rows in the imported points CSV.*?: (.+?)\. A missing", original
        ):
            note = f"{match[1]} of {match[2]} imported points have invalid coordinates and were excluded (CSV {match[3]})."
        elif match := re.match(r"(\d+) of (\d+) points had a coordinate", original):
            note = f"{match[1]} of {match[2]} points have invalid coordinates and were excluded."
        elif match := re.match(r"(\d+) of (\d+) points fell outside the tissue mask", original):
            note = f"{match[1]} of {match[2]} points are outside the measurement area and were excluded from point fractions."
        elif match := re.match(r"(\d+) of (\d+) points lie outside the image", original):
            note = f"{match[1]} of {match[2]} points are outside the image; counts and distances use their nearest image-border positions."
        elif original.startswith("None of the ") and "points is on the tissue" in original:
            note = "No points are inside the measurement area. Enrichment and spatial statistics are unavailable."
        elif original.startswith("The points are the centroids of the objects that also define "):
            compartments = original.split("also define ", 1)[1].split(", so enrichment", 1)[0]
            note = f"The points also define {compartments}. Enrichment against their own compartment is not an independent measurement."
        elif match := re.match(r"Distance-to-(.+?) was requested but skipped: (.+)", original):
            target, reason = match.groups()
            if "not one of the compartments" in reason:
                reason = "the target structure is unavailable"
            elif "point set is empty" in reason:
                reason = "no points were supplied"
            else:
                reason = "physical distances are unavailable with the saved resolution settings"
            note = f"Distance to {target} was not calculated: {reason}."
        elif (
            original.startswith("The Monte-Carlo null ran ")
            or original.startswith("All ")
            and "Monte-Carlo replicates returned the same value" in original
        ):
            match = re.search(r"reported for (.+?)\.|same value for (.+?), so", original)
            names = (match[1] or match[2]) if match else "the affected metrics"
            note = f"Monte Carlo variation could not be estimated for {names}; z-scores and p-values are unavailable."
        elif original.startswith(
            (
                "The points also define ",
                "Distance to ",
                "No points are inside ",
                "Monte Carlo variation could not be estimated ",
            )
        ) or re.match(
            r"\d+ of \d+ (?:imported )?points (?:have invalid coordinates|are outside)", original
        ):
            note = original
        if note and note not in notes:
            notes.append(note)

    for compartment in result.get("segmentation_settings") or []:
        runs = compartment.get("runs") or []
        signatures = {
            tuple(
                str(run.get(key))
                for key in (
                    "pack_id",
                    "adapter_id",
                    "threshold",
                    "min_area",
                    "ran_at_nm",
                    "native_pixel_size_nm",
                    "prob_map_grid",
                )
            )
            for run in runs
            if run.get("run_id")
        }
        if len(signatures) > 1:
            names = {
                str(
                    (run.get("adapter") or {}).get("name")
                    or run.get("pack_id")
                    or "model-generated"
                )
                for run in runs
            }
            mixed[f"compartment '{compartment['compartment']}'"] = names
    for compartment, models in mixed.items():
        names = ", ".join(sorted(models - {"model-generated"})) or "model-generated"
        note = f"The {names} objects in {compartment} were segmented under multiple models and settings."
        if note not in notes:
            notes.append(note)
    return notes

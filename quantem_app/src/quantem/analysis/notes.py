"""User-facing notes for new and previously saved analysis results."""

from __future__ import annotations

import re
from typing import Any

HIDDEN_NOTE_PREFIXES = (
    "No area of this image is recorded as reviewed",
)
PARTIAL_METRICS = re.compile(r"^\d+ of \d+ metrics (?:is|are) measured on fewer than")


def partial_metric_note(summary: dict[str, Any], total: int) -> str | None:
    partial = {key: stats for key, stats in summary.items() if stats.get("n_missing")}
    if not partial:
        return None
    reasons: dict[str, list[str]] = {}
    for key, stats in partial.items():
        missing = stats["n_missing"]
        sources = stats.get("missing_by_source") or {}
        if key == "mean_prob" and sources.get("manual") == missing:
            reason = f"{missing} manually drawn objects have no model probability score."
        else:
            reason = stats.get("note") or stats.get("unreportable_reason")
            if reason and reason.startswith("Measured on "):
                reason = reason.partition("; ")[2] or reason
            reason = reason or f"{missing} objects have no saved value for this metric."
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
    """Apply current wording to saved results without rewriting the database."""
    result = result or {}
    objects = result.get("objects") or {}
    partial = partial_metric_note(objects.get("summary") or {}, objects.get("n", 0))
    notes = []
    for original in result.get("caveats") or []:
        if original.startswith(HIDDEN_NOTE_PREFIXES) or (
            original.startswith("The scale inference ran at in ")
            and "is unknown, not native:" in original
        ):
            continue
        if PARTIAL_METRICS.match(original):
            # Older manifests can lack metric details; still remove the obsolete
            # instruction to look in another table.
            original = partial or original.split(". ", 1)[0] + "."
        if original not in notes:
            notes.append(original)
    return notes

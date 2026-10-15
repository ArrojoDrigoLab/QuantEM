"""Structured analysis notes. Codes identify meaning; messages are saved at measurement time."""

from typing import Any, TypedDict


class AnalysisNote(TypedDict):
    code: str
    message: str
    details: dict[str, Any]


def note(code: str, message: str, **details: Any) -> AnalysisNote:
    return {"code": code, "message": message, "details": details}


def analysis_notes(result: dict[str, Any] | None) -> list[str]:
    result = result or {}
    if isinstance(result.get("notes"), list):
        # Unknown future codes remain visible. Wording never controls whether a
        # scientific qualification survives in the UI or a downloaded report.
        return list(
            dict.fromkeys(
                item["message"]
                for item in result["notes"]
                if isinstance(item, dict)
                and isinstance(item.get("message"), str)
                and item["message"]
            )
        )
    from .legacy_notes import legacy_analysis_notes

    return legacy_analysis_notes(result)


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


def settings_notes(settings: list[dict]) -> list[AnalysisNote]:
    notes = []
    for compartment in settings:
        runs = compartment.get("runs") or []
        where = f"compartment {compartment['compartment']!r}"
        run_ids = {run["run_id"] for run in runs if run.get("run_id")}
        if len(run_ids) > 1:
            notes.append(
                note(
                    "multiple_inference_runs",
                    f"The objects in {where} come from {len(run_ids)} different inference runs.",
                    compartment=compartment["compartment"],
                    count=len(run_ids),
                )
            )
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
            names = sorted(
                {
                    str(
                        (run.get("adapter") or {}).get("name")
                        or run.get("pack_id")
                        or "model-generated"
                    )
                    for run in runs
                }
            )
            notes.append(
                note(
                    "mixed_model_settings",
                    f"The {', '.join(names)} objects in {where} were segmented under multiple models and settings.",
                    compartment=compartment["compartment"],
                    models=names,
                )
            )
    return notes

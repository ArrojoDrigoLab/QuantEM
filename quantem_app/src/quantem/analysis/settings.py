"""Snapshot segmentation settings and read historical snapshots without guessing."""

from __future__ import annotations

import copy
import json


def global_settings(segmentation):
    from .loaders import _adapter_details

    record = segmentation.global_mask
    metadata = copy.deepcopy(record.metadata or {})
    history = metadata.get("inference_runs") or ([metadata] if record.source == "model" else [])
    runs = [
        {
            "run_id": value.get("run_id"),
            "pack_id": value.get("pack_id") or value.get("source_model"),
            "threshold": value.get("include_level", value.get("threshold")),
            "adapter_id": value.get("adapter_id"),
            "adapter": _adapter_details(str(value["adapter_id"]))
            if value.get("adapter_id")
            else None,
            "native_pixel_size_nm": value.get("native_pixel_size_nm"),
            "ran_at_nm": value.get("ran_at_nm"),
            "resampled": value.get("resampled"),
            "scope": value.get("scope", value.get("run_scope")),
            "device": value.get("device"),
        }
        for value in history
    ]
    return [
        {
            "compartment": segmentation.segmentation_type.long_name,
            "segmentation_id": str(segmentation.id),
            "runs": runs,
            "global_mask": {"source": record.source, "metadata": metadata},
        }
    ]


def segmentation_settings(provenance):
    settings = []
    for block in [*provenance.get("compartments", []), provenance.get("tissue"), provenance.get("subject")]:
        if not block:
            continue
        runs = copy.deepcopy((block.get("run") or {}).get("runs") or [])
        counts = block.get("n_confirmed_by_source") or {}
        for model, count in counts.items():
            if model == "manual":
                continue
            unstamped = count - sum(
                run.get("n_objects", 0) for run in runs if run.get("pack_id") == model
            )
            if unstamped > 0:
                # Current segmentation options are not historical run settings.
                runs.append({"pack_id": model, "n_objects": unstamped, "run_id": None})
        settings.append(
            {
                "compartment": block["compartment"],
                "segmentation_id": block.get("segmentation_id"),
                "manual_objects": counts.get("manual", 0),
                "runs": runs,
                "global_mask": block.get("mask_storage"),
                "probability_measurements": block.get("probability_measurements", []),
            }
        )
    return settings


def saved_segmentation_settings(run):
    result = run.results or {}
    if "segmentation_settings" in result:
        return result["segmentation_settings"]
    path = run.export_path
    if path is None:
        return []
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    saved = (manifest.get("result") or {}).get("segmentation_settings")
    return saved if saved is not None else segmentation_settings(manifest.get("models") or {})

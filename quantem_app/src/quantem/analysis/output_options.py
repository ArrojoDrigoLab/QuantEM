"""Stable checkbox vocabulary for guided analysis outputs."""

from __future__ import annotations

OUTPUT_DEFAULTS = {
    "whole_image": True,
    "sample_area": False,
    "all_cells": False,
    "per_cell": False,
    "individual_objects": False,
}


def normalise_output_selection(raw: object) -> dict[str, bool]:
    values = dict(OUTPUT_DEFAULTS)
    if isinstance(raw, dict):
        for key in values:
            if key in raw:
                values[key] = raw[key] is True
    values["whole_image"] = True
    return values

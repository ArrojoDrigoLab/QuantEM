"""Group-level aggregation endpoint.

``AnalysisRun.group`` is a free-text experimental label ("fasted", "control").
Every run stored it, the manifest wrote it out, and until this endpoint existed
nothing ever *used* it: :mod:`quantem.analysis.rollup` held the one rule that
matters and had no production caller. A stored field with no reader is a feature
that looks finished from the outside, so this completes it.

The per-run views live in ``quantem.segmentation.api_views.analysis`` alongside
every other ``/api/segmentations/<id>/...`` endpoint. This one is not about a
segmentation -- it is about a set of runs -- so it lives with the aggregation
code it is a thin wrapper over.

Three things this endpoint refuses to do quietly:

* **It states its aggregation rule in the payload.** Honesty rule 4: a
  group-level number is never shown without saying it is an unweighted mean over
  units. The rule travels with the number rather than living in a UI string that
  can be dropped in a redesign.
* **It counts each image once.** Only its latest successful, current run is
  included. Incompatible measurement recipes must be compared in separate
  batches rather than silently pooled.
* **It carries the runs' caveats up with their numbers.** Averaging is not a
  way of making a qualified number unqualified. Every run whose points are a
  compartment's own centroids says ``enrichment in that compartment is circular
  (it is 1 / area fraction by construction) and must not be reported as a
  result`` and names the exact columns; this rollup used to publish
  ``enrichment_mito`` and ``z_enrichment_mito`` as first-class metrics with a
  mean, an SD and an SEM and no trace of any of that. A group mean is the
  number that goes in a figure, so it is the *last* place a caveat may be
  dropped.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .identity import run_is_current
from .models import AnalysisRun
from .notes import analysis_notes
from .rollup import rollup
from .service import circular_columns, image_summary_row

#: The aggregation rule, returned with every payload. Same words as the export
#: manifest, for the same reason.
AGGREGATION_RULE = (
    "Unweighted mean over images (descriptive), using the latest successful run per image, with "
    "the sample SD (ddof=1) and SEM across units. Never weighted by point count "
    "— doing so produced a random-data enrichment of 0.73 instead of 1.0 in the "
    "reference implementation."
)

#: Row keys that describe a run rather than measure it. Everything else numeric
#: is a metric and gets aggregated. ``pixel_size_nm`` is numeric but belongs
#: here: it is an acquisition setting, and a "mean pixel size" is not a result --
#: the distinct values are reported per group instead. ``n_caveats`` for the same
#: reason in reverse: it counts sentences, and "these images averaged 5.3
#: caveats" is not a quantity. The caveats themselves are text and are skipped
#: by :func:`_is_metric` already.
IDENTITY_FIELDS = frozenset(
    {"image_key", "group", "calibrated", "run_id", "pixel_size_nm", "n_caveats"}
)

#: What one row of the rollup represents. Stated in the payload because "n = 3"
#: is meaningless without it.
UNIT_DESCRIPTION = "one image (latest successful run); biological independence is not assumed"

#: Said beside every circular metric in the payload, in the metric entry itself
#: rather than only in the group's warnings, because a client that reads
#: ``groups[i].metrics`` to draw a bar chart never looks at the sibling key.
CIRCULAR_NOTE = (
    "Circular by construction: the points in at least one run of this group are "
    "the centroids of the objects that define this very compartment, so the "
    "value is 1 / area fraction and says nothing about the biology. Averaging "
    "it does not make it a measurement — it must not be reported as a result."
)


def _recipe(run):
    """Compare scientific settings without requiring image-specific mask IDs to match."""
    from quantem.segmentation.models import ImageSegmentation

    params = run.params or {}
    mask_id = params.get("tissue_segmentation_id")
    mask = (
        ImageSegmentation.objects.select_related("segmentation_type").filter(id=mask_id).first()
        if mask_id
        else None
    )
    values = {
        "organelle": str(run.segmentation.segmentation_type_id) if run.segmentation_id else None,
        "compartments": sorted((params.get("compartments") or {}).keys()),
        "measurement_area": (mask.segmentation_type.internal_name, mask.display_name or "")
        if mask
        else None,
        "points_source": params.get("points_source"),
        "distance_target": params.get("distance_target"),
        "band_edges_nm": params.get("band_edges_nm"),
        "objects_in_measurement_area": params.get("objects_in_measurement_area", False),
    }
    return json.dumps(values, sort_keys=True)


def _is_metric(value: Any) -> bool:
    """Numbers are metrics; booleans and ``None`` are not."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _group_warnings(rows: list[dict[str, Any]], circular: list[str]) -> list[str]:
    """Everything about this group that would mislead if left unsaid."""
    warnings: list[str] = []
    image_keys = [str(row.get("image_key")) for row in rows]
    distinct = len(set(image_keys))
    if distinct < len(rows):
        warnings.append(
            f"{len(rows)} runs cover only {distinct} distinct "
            f"image{'s' if distinct != 1 else ''}. Repeated analyses of the same "
            "image are not independent experimental units; the SEM below is "
            "narrower than the biology supports."
        )
    if not rows[0].get("group"):
        warnings.append(
            "These runs carry no group label. They are listed together because "
            "they are ungrouped, not because they are one experimental group."
        )
    calibrated = [bool(row.get("calibrated")) for row in rows]
    if any(calibrated) and not all(calibrated):
        warnings.append(
            "Some runs in this group are uncalibrated. Metrics in physical units "
            "are averaged over the calibrated runs only — check each metric's "
            "n_units against the group's."
        )
    # Distinct pixel sizes are already listed per group, but a list of two
    # numbers is not a statement. Micron-denominated metrics averaged across
    # scales are averaged across acquisitions, which is a study design question,
    # not a rounding one.
    scales = sorted({row["pixel_size_nm"] for row in rows if row.get("pixel_size_nm") is not None})
    if len(scales) > 1:
        warnings.append(
            f"These runs were acquired at {len(scales)} different pixel sizes "
            f"({', '.join(f'{s:g}' for s in scales)} nm/px). Every metric in "
            "microns is averaged across those scales, and each image's own "
            "caveats say whether its calibration changed after its objects were "
            "produced."
        )
    if circular:
        one = len(circular) == 1
        warnings.append(
            f"{', '.join(circular)} {'is' if one else 'are'} circular in at "
            f"least one run here: {'that column' if one else 'those columns'} "
            "measure a compartment with its own objects' centroids, so the "
            "value is 1 / area fraction by construction and must not be "
            f"reported as a result. {'It is' if one else 'They are'} named in "
            "circular_metrics and flagged in the metric entry itself."
        )
    return warnings


def _group_caveats(
    rows: list[dict[str, Any]], caveats_by_run: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """The runs' own caveats, de-duplicated, each naming the runs it came from.

    De-duplicated because three images of the same uncalibrated experiment
    produce the same sentence three times and a reader stops reading; named per
    run because "one of these five runs" is not something a reader can act on.
    """
    order: list[str] = []
    by_text: dict[str, list[str]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        for text in caveats_by_run.get(run_id, []):
            if text not in by_text:
                by_text[text] = []
                order.append(text)
            by_text[text].append(run_id)
    return [
        {"text": text, "n_runs": len(by_text[text]), "run_ids": by_text[text]} for text in order
    ]


class AnalysisGroupRollupView(APIView):
    """``GET /api/analysis/groups/`` -- per-group means over completed runs.

    Query parameters, both optional:

    * ``segmentation=<uuid>`` restricts to one segmentation's runs.
    * ``group=<label>`` restricts to one group; pass an empty value for the
      ungrouped runs.

    Only ``SUCCESS`` runs are included -- a failed or still-running analysis has
    no numbers to average, and silently counting it as a unit would deflate
    every mean.
    """

    def get(self, request: Request) -> Response:
        runs = AnalysisRun.objects.filter(status=AnalysisRun.STATUS_SUCCESS)

        seg_id = (request.query_params.get("segmentation") or "").strip()
        if seg_id:
            try:
                uuid.UUID(seg_id)
            except (AttributeError, TypeError, ValueError):
                return Response(
                    {"error": f"{seg_id!r} is not a segmentation id."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            runs = runs.filter(segmentation_id=seg_id)

        wanted_group = request.query_params.get("group")
        if wanted_group is not None:
            runs = runs.filter(group=wanted_group)

        rows: list[dict[str, Any]] = []
        # Kept beside the rows rather than inside them: a row is the exported
        # CSV shape, shared with the bundle, and this endpoint must not be able
        # to change what a column means there.
        caveats_by_run: dict[str, list[str]] = {}
        circular_by_run: dict[str, list[str]] = {}
        seen_images = set()
        recipes = set()
        omitted_stale = []
        for run in runs.select_related("segmentation__asset").order_by("-created_at", "-id"):
            if not run.results:
                continue
            try:
                row = image_summary_row(run.results)
            except (KeyError, TypeError):
                # A run stored by an older build with a different result shape.
                # Skipping it is right -- guessing at its columns is not.
                continue
            image_key = (
                str(run.segmentation.asset_id)
                if run.segmentation_id
                else str(row.get("image_key") or run.id)
            )
            if image_key in seen_images:
                continue
            seen_images.add(image_key)
            if run.input_fingerprint and not run_is_current(run):
                omitted_stale.append(str(run.id))
                continue
            recipes.add(_recipe(run))
            row["run_id"] = str(run.id)
            caveats_by_run[str(run.id)] = analysis_notes(run.results)
            circular_by_run[str(run.id)] = circular_columns(run.results, row)
            rows.append(row)

        if len(recipes) > 1:
            return Response(
                {
                    "error": "These images use different organelles, masks or measurement settings. Use an experiment analysis batch to compare a consistent recipe.",
                    "omitted_outdated_runs": omitted_stale,
                },
                status=status.HTTP_409_CONFLICT,
            )
        metrics = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if key not in IDENTITY_FIELDS and _is_metric(value)
            }
        )
        grouped = rollup(rows, group_key="group", metrics=metrics)
        by_group: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_group.setdefault(str(row.get("group", "")), []).append(row)

        circular_per_group = {
            name: sorted(
                {
                    column
                    for row in members
                    for column in circular_by_run.get(str(row["run_id"]), [])
                }
            )
            for name, members in by_group.items()
        }

        payload = {
            "aggregation_rule": AGGREGATION_RULE,
            "unit": UNIT_DESCRIPTION,
            "scope": {
                "segmentation": seg_id or None,
                "group": wanted_group,
                "status": AnalysisRun.STATUS_SUCCESS,
                "n_runs": len(rows),
                "omitted_outdated_runs": omitted_stale,
            },
            "metrics": metrics,
            # The union over every group in this response, so a client that
            # reads the flat ``metrics`` list to decide what it can plot has the
            # exclusions in the same place.
            "circular_metrics": sorted(
                {column for columns in circular_per_group.values() for column in columns}
            ),
            "circular_note": CIRCULAR_NOTE,
            "groups": [
                {
                    "group": name,
                    "n_units": len(by_group[name]),
                    "unit": UNIT_DESCRIPTION,
                    "run_ids": [row["run_id"] for row in by_group[name]],
                    "image_keys": sorted({str(row.get("image_key")) for row in by_group[name]}),
                    # Distinct values, not a mean: more than one entry here says
                    # the group mixes acquisition scales, which a reader needs.
                    "pixel_sizes_nm": sorted(
                        {
                            row["pixel_size_nm"]
                            for row in by_group[name]
                            if row.get("pixel_size_nm") is not None
                        }
                    ),
                    "warnings": _group_warnings(by_group[name], circular_per_group[name]),
                    # Everything the runs behind these means said about
                    # themselves. A caveat that stops at the per-image screen is
                    # a caveat that never reaches the figure.
                    "caveats": _group_caveats(by_group[name], caveats_by_run),
                    "circular_metrics": circular_per_group[name],
                    "metrics": {
                        metric: {
                            **aggregated.as_dict(),
                            "circular": metric in circular_per_group[name],
                            **(
                                {"note": CIRCULAR_NOTE}
                                if metric in circular_per_group[name]
                                else {}
                            ),
                        }
                        for metric, aggregated in grouped[name].items()
                    },
                }
                for name in sorted(by_group)
            ],
        }
        return Response(payload, status=status.HTTP_200_OK)

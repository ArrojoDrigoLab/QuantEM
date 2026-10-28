"""Region-, cell-, and organelle-level outputs for the guided analysis builder."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from quantem.segmentation.models import ImageSegmentation

from .loaders import (
    analysis_mask_regions,
    confirmed_objects,
    ensure_confirmed_object_measurements,
    image_shape,
    segmentation_mask,
)
from .morphometrics import ObjectMetrics, derive, summarize
from .output_options import normalise_output_selection

REGION_SUMMARY_FIELDS = (
    "region_kind",
    "region_id",
    "region_name",
    "segmentation_type_id",
    "segmentation_id",
    "organelle",
    "representation",
    "region_area_px",
    "region_area_um2",
    "organelle_area_px",
    "organelle_area_um2",
    "area_fraction",
    "object_count",
)

CELL_SUMMARY_FIELDS = (
    "cell_id",
    "cell_name",
    "cell_area_px",
    "cell_area_um2",
    "measurement_area_px",
    "measurement_area_um2",
    "clipped_by_sample_area",
    "segmentation_type_id",
    "segmentation_id",
    "organelle",
    "organelle_area_px",
    "organelle_area_um2",
    "area_fraction",
    "object_count",
)

ORGANELLE_OBJECT_FIELDS = (
    "segmentation_type_id",
    "segmentation_id",
    "organelle",
    "object_id",
    "source_model",
    "centroid_x_px",
    "centroid_y_px",
    "in_sample_area",
    "assigned_cell_id",
    "assigned_cell_name",
)


def _physical_area(area_px: int, pixel_size_nm: float | None) -> float | None:
    if pixel_size_nm is None or pixel_size_nm <= 0:
        return None
    return float(area_px) * (float(pixel_size_nm) / 1000.0) ** 2


def _region_row(
    *,
    region_kind: str,
    region_id: str,
    region_name: str,
    region_mask: np.ndarray,
    organelle_name: str,
    organelle: ImageSegmentation,
    organelle_mask: np.ndarray,
    pixel_size_nm: float | None,
) -> dict[str, Any]:
    denominator = int(np.count_nonzero(region_mask))
    area = int(np.count_nonzero(organelle_mask & region_mask))
    return {
        "region_kind": region_kind,
        "region_id": region_id,
        "region_name": region_name,
        "segmentation_type_id": str(organelle.segmentation_type_id),
        "segmentation_id": str(organelle.id),
        "organelle": organelle_name,
        "representation": organelle.segmentation_type.representation,
        "region_area_px": denominator,
        "region_area_um2": _physical_area(denominator, pixel_size_nm),
        "organelle_area_px": area,
        "organelle_area_um2": _physical_area(area, pixel_size_nm),
        "area_fraction": (float(area) / float(denominator)) if denominator else None,
        "object_count": None,
    }


def _cell_contains(region, x: float, y: float, sample_mask: np.ndarray | None) -> bool:
    local_x = int(x) - int(region.x0)
    local_y = int(y) - int(region.y0)
    if not (0 <= local_y < region.mask.shape[0] and 0 <= local_x < region.mask.shape[1]):
        return False
    if not bool(region.mask[local_y, local_x]):
        return False
    if sample_mask is None:
        return True
    iy, ix = int(y), int(x)
    return (
        0 <= iy < sample_mask.shape[0]
        and 0 <= ix < sample_mask.shape[1]
        and bool(sample_mask[iy, ix])
    )


def _metric_row(metrics: ObjectMetrics) -> dict[str, Any]:
    row = metrics.as_row()
    # Centroids are database geometry, not a derived feature, and are added by
    # the caller. Keep every existing morphometric key under its established
    # name so this table can be joined with the legacy objects.csv vocabulary.
    return row


def _flatten_distribution(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric, statistics in summary.items():
        for statistic in (
            "n",
            "n_objects",
            "n_missing",
            "mean",
            "sd",
            "median",
            "iqr",
            "min",
            "max",
        ):
            if statistic in statistics:
                row[f"{metric}_{statistic}"] = statistics[statistic]
    return row


def build_structured_outputs(
    subject: ImageSegmentation,
    params: dict[str, Any],
    *,
    cancel_check=None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Build the optional output hierarchy selected in the analysis form."""
    outputs = normalise_output_selection(params.get("outputs"))
    shape = image_shape(subject)
    asset_id = subject.asset_id
    pixel_size_nm = (
        float(subject.asset.pixel_size_nm)
        if subject.asset and subject.asset.pixel_size_nm and subject.asset.pixel_size_nm > 0
        else None
    )

    organelles: list[tuple[str, ImageSegmentation]] = []
    for name, segmentation_id in (params.get("compartments") or {}).items():
        segmentation = (
            ImageSegmentation.objects.select_related("segmentation_type", "asset")
            .filter(id=segmentation_id, asset_id=asset_id, superseded_by__isnull=True)
            .first()
        )
        if (
            segmentation is not None
            and segmentation.segmentation_type.category
            == segmentation.segmentation_type.CATEGORY_ORGANELLE
        ):
            organelles.append((str(name), segmentation))

    sample = None
    sample_id = params.get("tissue_segmentation_id")
    if sample_id:
        sample = (
            ImageSegmentation.objects.select_related("segmentation_type")
            .filter(id=sample_id, asset_id=asset_id, superseded_by__isnull=True)
            .first()
        )
    sample_mask = (
        segmentation_mask(sample, shape, cancel_check=cancel_check) if sample is not None else None
    )

    cells = None
    cells_id = params.get("cell_segmentation_id")
    if cells_id:
        cells = (
            ImageSegmentation.objects.select_related("segmentation_type")
            .filter(id=cells_id, asset_id=asset_id, superseded_by__isnull=True)
            .first()
        )
    cell_regions = analysis_mask_regions(cells, shape) if cells is not None else ()

    organelle_masks = {
        str(segmentation.id): segmentation_mask(segmentation, shape, cancel_check=cancel_check)
        for _name, segmentation in organelles
    }
    whole = np.ones(shape, dtype=bool)
    cell_union = np.zeros(shape, dtype=bool)
    for region in cell_regions:
        y1 = region.y0 + region.mask.shape[0]
        x1 = region.x0 + region.mask.shape[1]
        cell_union[region.y0 : y1, region.x0 : x1] |= region.mask
    effective_cell_union = cell_union & sample_mask if sample_mask is not None else cell_union

    region_rows: list[dict[str, Any]] = []
    for organelle_name, organelle in organelles:
        mask = organelle_masks[str(organelle.id)]
        region_rows.append(
            _region_row(
                region_kind="whole_image",
                region_id=str(subject.asset_id),
                region_name="Whole image",
                region_mask=whole,
                organelle_name=organelle_name,
                organelle=organelle,
                organelle_mask=mask,
                pixel_size_nm=pixel_size_nm,
            )
        )
        if outputs["sample_area"] and sample_mask is not None:
            region_rows.append(
                _region_row(
                    region_kind="sample_area",
                    region_id=str(sample.id),
                    region_name=sample.segmentation_type.long_name,
                    region_mask=sample_mask,
                    organelle_name=organelle_name,
                    organelle=organelle,
                    organelle_mask=mask,
                    pixel_size_nm=pixel_size_nm,
                )
            )
        if outputs["all_cells"] and cells is not None:
            region_rows.append(
                _region_row(
                    region_kind="all_cells",
                    region_id=str(cells.id),
                    region_name="All cells",
                    region_mask=effective_cell_union,
                    organelle_name=organelle_name,
                    organelle=organelle,
                    organelle_mask=mask,
                    pixel_size_nm=pixel_size_nm,
                )
            )

    clipped_cells: list[dict[str, str]] = []
    cell_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region in cell_regions:
        full = region.mask.astype(bool, copy=False)
        y1 = region.y0 + full.shape[0]
        x1 = region.x0 + full.shape[1]
        effective = (
            full & sample_mask[region.y0 : y1, region.x0 : x1] if sample_mask is not None else full
        )
        cell_masks[region.identifier] = (full, effective)
        if sample_mask is not None and np.any(full & ~effective):
            clipped_cells.append({"id": region.identifier, "name": region.name})

    object_rows: list[dict[str, Any]] = []
    metrics_by_region_organelle: dict[tuple[str, str], list[ObjectMetrics]] = defaultdict(list)
    metrics_by_cell_organelle: dict[tuple[str, str], list[ObjectMetrics]] = defaultdict(list)
    for organelle_name, organelle in organelles:
        if (
            organelle.segmentation_type.representation
            != organelle.segmentation_type.REPRESENTATION_INSTANCES
        ):
            continue
        ensure_confirmed_object_measurements(organelle, cancel_check=cancel_check)
        for obj in confirmed_objects(organelle).iterator():
            if cancel_check is not None:
                cancel_check()
            metrics = derive(
                obj.features or {},
                object_id=str(obj.id),
                pixel_size_nm=pixel_size_nm,
                source_model=obj.source_model,
            )
            assigned = next(
                (
                    region
                    for region in cell_regions
                    if _cell_contains(region, obj.centroid_x, obj.centroid_y, sample_mask)
                ),
                None,
            )
            iy, ix = int(obj.centroid_y), int(obj.centroid_x)
            in_sample = sample_mask is None or (
                0 <= iy < shape[0] and 0 <= ix < shape[1] and bool(sample_mask[iy, ix])
            )
            if assigned is not None:
                metrics_by_cell_organelle[(assigned.identifier, str(organelle.id))].append(metrics)
            organelle_id = str(organelle.id)
            metrics_by_region_organelle[("whole_image", organelle_id)].append(metrics)
            if outputs["sample_area"] and sample_mask is not None and in_sample:
                metrics_by_region_organelle[("sample_area", organelle_id)].append(metrics)
            if outputs["all_cells"] and assigned is not None:
                metrics_by_region_organelle[("all_cells", organelle_id)].append(metrics)
            if outputs["individual_objects"]:
                object_rows.append(
                    {
                        "segmentation_type_id": str(organelle.segmentation_type_id),
                        "segmentation_id": str(organelle.id),
                        "organelle": organelle_name,
                        "centroid_x_px": float(obj.centroid_x),
                        "centroid_y_px": float(obj.centroid_y),
                        "in_sample_area": in_sample,
                        "assigned_cell_id": assigned.identifier if assigned else "",
                        "assigned_cell_name": assigned.name if assigned else "",
                        **_metric_row(metrics),
                    }
                )

    for row in region_rows:
        if row["representation"] != "instances":
            continue
        metrics = metrics_by_region_organelle.get(
            (str(row["region_kind"]), str(row["segmentation_id"])), []
        )
        row["object_count"] = len(metrics)
        row.update(_flatten_distribution(summarize(metrics)))

    cell_rows: list[dict[str, Any]] = []
    if outputs["per_cell"]:
        for region in cell_regions:
            full, effective = cell_masks[region.identifier]
            full_area = int(np.count_nonzero(full))
            effective_area = int(np.count_nonzero(effective))
            y1 = region.y0 + full.shape[0]
            x1 = region.x0 + full.shape[1]
            for organelle_name, organelle in organelles:
                organelle_crop = organelle_masks[str(organelle.id)][region.y0 : y1, region.x0 : x1]
                organelle_area = int(np.count_nonzero(organelle_crop & effective))
                metrics = metrics_by_cell_organelle.get((region.identifier, str(organelle.id)), [])
                cell_rows.append(
                    {
                        "cell_id": region.identifier,
                        "cell_name": region.name,
                        "cell_area_px": full_area,
                        "cell_area_um2": _physical_area(full_area, pixel_size_nm),
                        "measurement_area_px": effective_area,
                        "measurement_area_um2": _physical_area(effective_area, pixel_size_nm),
                        "clipped_by_sample_area": effective_area != full_area,
                        "segmentation_type_id": str(organelle.segmentation_type_id),
                        "segmentation_id": str(organelle.id),
                        "organelle": organelle_name,
                        "organelle_area_px": organelle_area,
                        "organelle_area_um2": _physical_area(organelle_area, pixel_size_nm),
                        "area_fraction": (
                            float(organelle_area) / float(effective_area)
                            if effective_area
                            else None
                        ),
                        "object_count": len(metrics),
                        **_flatten_distribution(summarize(metrics)),
                    }
                )

    warnings: list[dict[str, Any]] = []
    if clipped_cells:
        warnings.append(
            {
                "code": "cells_clipped_by_sample_area",
                "message": (
                    f"{len(clipped_cells)} cell(s) extend beyond the Sample Area. "
                    "Their all-cells and per-cell composition uses the intersection; "
                    "cell_area reports the full saved cell and measurement_area reports "
                    "the clipped area."
                ),
                "cells": clipped_cells,
            }
        )

    result = {
        "outputs": outputs,
        "warnings": warnings,
        "region_summary": region_rows,
        "cell_summary": cell_rows,
        "n_organelle_objects": len(object_rows),
        "n_cells": len(cell_regions),
    }
    tables = {
        "region_summary.csv": region_rows,
        "cell_summary.csv": cell_rows,
        "organelle_objects.csv": object_rows,
    }
    return result, tables

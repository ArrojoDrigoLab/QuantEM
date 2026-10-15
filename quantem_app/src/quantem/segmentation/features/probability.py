"""Measure an outline against a saved model probability map in image pixels."""

from __future__ import annotations

import math

import numpy as np

from quantem.core.local_storage import StorageError
from quantem.segmentation.models import ProbabilityMap, SegmentObject
from quantem.segmentation.prob_maps.features import rasterize_polygon_in_bbox
from quantem.segmentation.prob_maps.io import (
    open_probability_map_image,
    resolve_probability_map_path,
)
from quantem.segmentation.run_identity import read_run_identity


class OutlineProbability:
    """Reuse one decoded uint8 map at a time, including for large image batches.

    Manual objects use the latest model map covering their entire outline.
    Edited model objects must match their original run. Display composites are
    excluded: their zero-filled, unpredicted areas are not model predictions.
    """

    def __init__(self, segmentation):
        self.maps = list(
            ProbabilityMap.objects.filter(segmentation=segmentation).order_by("-created_at", "-id")
        )
        self.image = None
        self.map_id = None

    def close(self):
        if self.image is not None:
            self.image.close()
        self.image = None
        self.map_id = None

    def measure(self, segment, features):
        polygon = segment.geometry
        if polygon is None or polygon.is_empty:
            return features
        stamp = read_run_identity(segment.features)
        manual = segment.source_model == "manual"
        for prob_map in self.maps:
            metadata = prob_map.metadata or {}
            if metadata.get("composite") or not metadata.get("pack_id"):
                continue
            if not manual and (
                not stamp
                or metadata.get("run_id") != stamp.get("id")
                or metadata.get("pack_id") != (stamp.get("pack_id") or segment.source_model)
            ):
                continue
            roi = metadata.get("roi") or {}
            ox, oy = int(roi.get("x", 0)), int(roi.get("y", 0))
            try:
                if self.map_id != prob_map.id:
                    self.close()
                    self.image = open_probability_map_image(resolve_probability_map_path(prob_map))
                    self.map_id = prob_map.id
                width, height = self.image.size
                if self.image.mode != "L":
                    continue
                if roi and (width != roi.get("width") or height != roi.get("height")):
                    continue
                if not roi and (
                    width != segment.segmentation.asset.logical_width
                    or height != segment.segmentation.asset.logical_height
                ):
                    continue
                x0, y0, x1, y1 = polygon.bounds
                if x0 < ox or y0 < oy or x1 > ox + width or y1 > oy + height:
                    continue
                left, top = math.floor(x0) - ox, math.floor(y0) - oy
                right, bottom = math.ceil(x1) - ox, math.ceil(y1) - oy
                with self.image.crop((left, top, right, bottom)) as crop:
                    data = np.asarray(crop)
                    rr, cc = rasterize_polygon_in_bbox(
                        polygon, polygon.envelope, data.shape, (ox + left, oy + top)
                    )
                    if not len(rr):
                        continue
                    mean = float(np.mean(data[rr, cc], dtype=np.float64) / 255.0)
                return {
                    **features,
                    "mean_prob": mean,
                    "probability_source": {
                        "map_id": str(prob_map.id),
                        "model": metadata.get("pack_id"),
                        "run_id": metadata.get("run_id"),
                        "settings": metadata,
                        "measurement": "mean_inside_outline",
                    },
                }
            except (OSError, ValueError, StorageError):
                self.close()
        return features


def fill_missing_outline_probabilities(segmentation, *, cancel_check=None):
    """Fill scores for saved manual outlines without changing their geometry."""
    sampler = OutlineProbability(segmentation)
    if not sampler.maps:
        return
    pending = []
    try:
        rows = SegmentObject.objects.filter(
            segmentation=segmentation, label_state="CONFIRMED", source_model="manual"
        )
        for segment in rows.select_related("segmentation__asset").iterator(chunk_size=200):
            if cancel_check:
                cancel_check()
            if (segment.features or {}).get("mean_prob") is not None:
                continue
            updated = sampler.measure(segment, segment.features or {})
            if updated.get("mean_prob") is not None:
                segment.features = updated
                pending.append(segment)
            if len(pending) == 200:
                SegmentObject.objects.bulk_update(pending, ["features"], batch_size=200)
                pending.clear()
        if pending:
            SegmentObject.objects.bulk_update(pending, ["features"], batch_size=200)
    finally:
        sampler.close()

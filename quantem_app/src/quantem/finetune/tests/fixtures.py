"""A fully annotated image, built the way the app builds one.

Deliberately not a mock: an asset with a PNG on disk, a completed ROI, a
confirmed object, and a stored probability map. The probability map is graded —
confident inside the object, half-confident in a ring around it — so the
threshold sweep has something real to choose between and the calibrated
threshold is not degenerate.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from quantem.segmentation.models import CompletedROI, ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_mitochondria_type,
)
from quantem.testing import create_small_test_image, write_prob_map_png

SIZE = 256
ROI = (20, 20, 180, 180)
OBJECT = (60, 60, 120, 120)
#: Probability in the ring around the true object. Anything at or below this
#: threshold over-segments, so the sweep must climb past it.
RING_PROB = 0.55
RING_PX = 12


def square(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))


def graded_prob_map(size: int = SIZE, obj: tuple[int, int, int, int] = OBJECT) -> np.ndarray:
    x0, y0, x1, y1 = obj
    prob = np.full((size, size), 0.05, dtype=np.float32)
    prob[y0 - RING_PX : y1 + RING_PX, x0 - RING_PX : x1 + RING_PX] = RING_PROB
    prob[y0 : y1 + 1, x0 : x1 + 1] = 0.9
    return prob


def annotated_segmentation(
    display_name: str,
    *,
    with_roi: bool = True,
    with_object: bool = True,
    with_prob: bool = True,
    organelle: str = "mito",
    size: int = SIZE,
    roi: tuple[int, int, int, int] = ROI,
    obj: tuple[int, int, int, int] = OBJECT,
) -> ImageSegmentation:
    """One annotated image.

    ``organelle`` exists so a test can build a segmentation with *no* annotated
    siblings: crops are gathered across every segmentation of the same
    organelle, so a mito fixture is never alone once another mito image exists.
    ``size``/``roi``/``obj`` exist because head training needs a region big
    enough to cut a 512 px tile out of after resampling to 8 nm.
    """
    image = create_small_test_image(display_name, width=size, height=size, textured=True)
    segmentation = ImageSegmentation.objects.create(
        asset=image.asset,
        segmentation_type=(
            get_or_create_er_type() if organelle == "er" else get_or_create_mitochondria_type()
        ),
        status_stage="CANDIDATES_READY",
    )
    if with_roi:
        CompletedROI.objects.create(segmentation=segmentation, geometry=square(*roi))
    if with_object:
        polygon = square(*obj)
        SegmentObject.objects.create(
            segmentation=segmentation,
            label_state="CONFIRMED",
            confidence_score=1.0,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )
    if with_prob:
        write_prob_map_png(segmentation, graded_prob_map(size, obj), name="MITO_DINO")
    return segmentation


class FakeReporter:
    """Records what a job would have shown the user."""

    def __init__(self) -> None:
        self.updates: list[tuple[float | None, str | None]] = []
        self.logs: list[tuple[str, str]] = []

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        self.updates.append((progress, message))

    def log(self, level: str, message: str) -> None:
        self.logs.append((level, message))


class FakeCancel:
    """A cancel token that never fires (and one that always does)."""

    def __init__(self, cancelled: bool = False) -> None:
        self.cancelled = cancelled

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("Job cancellation requested.")

"""One box in, one object out.

The whole flow, in order: plan the crop window, read its pixels, encode it (or
find the encode in the cache), decode the box against that embedding, turn the
top mask into a polygon in global pixels, and store it through the segmentation
service that already knows how to reconcile a new confirmed object against the
existing ones.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from quantem.sam.backends import get_backend
from quantem.sam.embedding_cache import EMBEDDINGS, cache_key
from quantem.sam.geometry import (
    Box,
    Crop,
    box_to_crop,
    mask_to_global_polygon,
    plan_crop,
    polygon_coords,
)

logger = logging.getLogger(__name__)


class PromptRefused(ValueError):
    """The prompt cannot be answered, for a reason the user can act on."""


@dataclass(frozen=True)
class Candidate:
    """One mask SAM proposed, already placed in global image pixels."""

    polygon: Polygon
    score: float
    area: float


@dataclass(frozen=True)
class PromptResult:
    candidates: list[Candidate]
    crop: Crop
    cache_hit: bool
    device: str
    backend_identity: str
    encode_ms: float
    decode_ms: float

    @property
    def top(self) -> Candidate:
        return self.candidates[0]


def to_rgb(plane: np.ndarray) -> np.ndarray:
    """A 2-D 8-bit plane as the ``(H, W, 3)`` uint8 SAM expects.

    Replication, not a colormap. The app stores 8-bit greyscale, SAM was trained
    on three channels, and giving it the same channel three times is what every
    greyscale SAM pipeline does.
    """
    array = np.asarray(plane)
    if array.ndim == 3:
        return np.ascontiguousarray(array[..., :3].astype(np.uint8))
    return np.ascontiguousarray(np.stack([array, array, array], axis=-1).astype(np.uint8))


def run_prompt(
    *,
    segmentation_id: str,
    openable,
    box: Box,
    image_width: int,
    image_height: int,
) -> PromptResult:
    """Segment whatever ``box`` encloses.

    ``openable`` is an :class:`~quantem.assets.asset_openable.AssetOpenable`;
    the pixels come from ``load_image_roi_array``, which reads NGFF level 0 when
    the pyramid is published and the source file otherwise.
    """
    from quantem.assets.task_utils import load_image_roi_array

    if box.width <= 0 or box.height <= 0:
        raise PromptRefused("Draw a box with some width and height, then release.")

    crop = plan_crop(box, image_width, image_height)
    local_box = box_to_crop(box, crop)
    if local_box is None:
        raise PromptRefused("That box falls outside the image.")

    backend = get_backend()
    key = cache_key(segmentation_id, backend.identity, crop)

    encode_ms = 0.0
    embedding = EMBEDDINGS.get(key)
    cache_hit = embedding is not None
    if embedding is None:
        plane = load_image_roi_array(openable, crop.x, crop.y, crop.width, crop.height)
        started = time.perf_counter()
        embedding = backend.encode(to_rgb(plane))
        encode_ms = (time.perf_counter() - started) * 1000.0
        EMBEDDINGS.put(key, embedding)

    started = time.perf_counter()
    raw = backend.predict(embedding, local_box.as_tuple())
    decode_ms = (time.perf_counter() - started) * 1000.0

    candidates: list[Candidate] = []
    for item in raw:
        placed = mask_to_global_polygon(item.mask, crop)
        if placed is None:
            continue
        polygon, area = placed
        candidates.append(Candidate(polygon=polygon, score=item.score, area=area))

    if not candidates:
        raise PromptRefused(
            "No object was found in that box. Try drawing it tighter around "
            "one object."
        )

    return PromptResult(
        candidates=candidates,
        crop=crop,
        cache_hit=cache_hit,
        device=backend.device,
        backend_identity=backend.identity,
        encode_ms=round(encode_ms, 1),
        decode_ms=round(decode_ms, 1),
    )


def store_top_candidate(*, segmentation, result: PromptResult) -> dict[str, object]:
    """Persist the best-scoring mask as a confirmed object.

    Through ``confirm_segment_geometries`` rather than ``SegmentObject.objects
    .create``: that service is where overlap against existing confirmed objects
    is split along the shared boundary, where candidate objects the new one
    supersedes are removed, and where the new object is measured. A box-prompted
    object is a user-created object and has to behave like one.

    ``sam_score`` rides along in ``features``, which the service already
    understands -- it is the predicted-IoU the decoder returned, and it is worth
    keeping next to the geometry it justifies.
    """
    from quantem.segmentation.services.confirm_batch import confirm_segment_geometries

    top = result.top
    outcome = confirm_segment_geometries(
        segmentation=segmentation,
        incoming=[{"geometry": top.polygon, "sam_score": float(top.score)}],
        merge_overlaps=False,
        manual_creation=True,
    )
    return outcome


def candidates_payload(result: PromptResult, *, skip_top: bool = True) -> list[dict]:
    """The masks that were not stored, for a future accept-or-cycle affordance.

    Returned now, used later. The decoder produced them in the same pass that
    produced the stored one, so carrying them costs a few hundred coordinates
    and saves a whole round trip when the user wants the other interpretation.
    """
    items = result.candidates[1:] if skip_top else result.candidates
    return [
        {
            "geometry_coords": polygon_coords(candidate.polygon),
            "score": round(candidate.score, 4),
            "area": round(candidate.area, 1),
        }
        for candidate in items
    ]

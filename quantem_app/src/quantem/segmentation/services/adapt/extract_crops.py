"""Completed ROIs -> ``(em, gt, valid)`` crops, against the live schema.

Port of ``gk_gold_seg/scripts/finetune_cv/gk_gt_extract.py`` (and its generalised
sibling ``gold_pipeline/extract_ann.py``). The reference walked a research
database and wrote ``<name>_em.npy`` / ``_gt.npy`` / ``_valid.npy`` to disk; here
the same three arrays are produced in memory from the app's own models.

The contract, unchanged from the reference and the reason it exists:

* A :class:`~quantem.segmentation.models.CompletedROI` is the **exhaustively
  annotated unit**. Inside it, a ``CONFIRMED`` :class:`SegmentObject` is
  foreground and every other pixel is *true* background.
* Outside it, nothing is known. Those pixels are :data:`IGNORE` (255) and are
  excluded from every loss and every score.
* With no completed ROI there is no valid background anywhere, so Dice is
  meaningless — a model that finds a real object the user simply never got to
  would be punished for it. The reference refused in that state and so does
  this: :func:`require_crops` raises :class:`CompletedRoiRequired`.

Two departures from the reference, both deliberate:

1. **No resampling here.** The reference resampled 5 nm EM to the model's 8 nm
   canonical size while extracting. In QuantEM that is
   :mod:`quantem.inference.resample`'s job and it happens per model pack, so
   crops come out at the asset's native scale with ``pixel_size_nm`` attached.
2. **Crops are gathered across sibling segmentations.** A segmentation is one
   (asset, organelle) pair, so a single one can never give an image-disjoint
   split. Annotations for the *same organelle* on other assets are included, and
   that is what makes ``split_mode == "image-disjoint"`` reachable at all.

Two sources of ground truth, not one
------------------------------------
A :class:`~quantem.segmentation.models.CompletedROI` polygon is the original
source. A **ROI marked done** -- a
:class:`~quantem.segmentation.models.RoiSegmentationStatus` row with
``is_complete`` set -- is the second, and its own docstring already states the
identical contract: inside the rectangle the labels are dense, outside it they
are ``ignore``. So a done ROI is just another rectangle contributing an
:class:`AnnotatedCrop`, and it is produced by the same loop rather than a
parallel one.

**Overlap between the two is resolved in the completed area's favour.** Where a
done ROI covers ground a completed polygon already covers, those pixels are
struck out of the done ROI's ``valid`` mask, so the area is trained on once and
scored once. The polygon wins because it is the more specific statement -- the
user drew its outline, where the rectangle is a window they happened to be
working in -- and because a hole punched into a completed area by
``CompletedRoiSubtractView`` must not be quietly re-annotated by a rectangle
drawn over it. A done ROI wholly inside completed polygons contributes nothing
and is not counted.

Scope
-----
``collect_crops`` gathers from every sibling in the library by default, which is
what the labeling view wants. :func:`collect_crops_for_scope` gathers from an
explicit set of assets instead, which is what a named fine-tune over a chosen
dataset wants. Both run the same collection loop.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.task_utils import load_image_roi_array
from quantem.finetune.calibrate import split_crops
from quantem.seg_core.rasterize import paint_rings
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    ProbabilityMap,
    RoiSegmentationStatus,
    SegmentObject,
)
from quantem.segmentation.prob_maps.io import resolve_probability_map_path

logger = logging.getLogger(__name__)

#: Label for "the user did not tell us what is here". Matches the reference and
#: ``torch.nn.functional.cross_entropy(ignore_index=...)``.
IGNORE = 255

#: Completed ROIs smaller than this on either edge are not training data; the
#: reference used the same 16 px floor.
MIN_CROP_PX = 16

#: The one label state that counts as ground truth. Not ``INFERRED`` (the model
#: said so, the user did not) and not ``REFINED`` alone.
CONFIRMED = "CONFIRMED"

#: Which of the two ground-truth records a crop came from. Carried on the crop
#: rather than recomputed, because the dialog reports the two counts separately
#: and a crop that has been de-overlapped no longer looks like either one.
SOURCE_CONFIRMED_AREA = "confirmed_area"
SOURCE_DONE_ROI = "done_roi"

#: Said when nothing on disk covers the annotated area. This blocks threshold
#: calibration, which has nothing to sweep without a stored map, and nothing
#: else: head training predicts its own. It is a blocker only when the caller
#: asks for one (``require_probability=True``); otherwise it is a warning and a
#: per-mode note, because refusing every rung over it made guided fine-tuning
#: unreachable for head training too.
NO_PROBABILITY_MESSAGE = (
    "No probability map covers the completed area. Run the model on this image "
    "first; threshold calibration is scored against what the model currently "
    "predicts there."
)

#: Adaptation modes, mirrored from :mod:`quantem.finetune.job` rather than
#: imported: ``finetune`` already imports this module, and a cycle would break
#: both. The names are part of the wire contract, so they cannot drift silently.
MODE_THRESHOLD_ONLY = "threshold_only"
MODE_HEAD = "head"


class CompletedRoiRequired(ValueError):
    """No completed ROI, so there is nothing to score against.

    Carries a user-facing sentence; API views render ``str(exc)`` straight into
    ``{"error": ...}``.
    """


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------


def _ring_coords(ring) -> np.ndarray | None:
    """One shapely ring as an ``(N, 2)`` float array of image coordinates."""
    coords = np.asarray(ring.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 3:
        return None
    return coords[:, :2]


def _rasterize(polygons: Iterable[Polygon], x0: int, y0: int, shape: tuple[int, int]) -> np.ndarray:
    """Fill polygons into a ``uint8`` 0/1 mask, honouring interior rings.

    Holes matter: ``CompletedRoiSubtractView`` punches interior rings into the
    completed area, and treating a hole as annotated would score the model on
    pixels the user explicitly took back. :func:`~quantem.seg_core.rasterize
    .paint_rings` writes only the pixels of the shape, so a hole cannot erase a
    neighbouring object and no scratch buffer is needed to keep them apart.

    This is the ground truth a fine-tune is scored and trained against, so it
    uses the app's one pixel convention -- a pixel is the object's when its
    centre is inside the outline. Under ``cv2.fillPoly`` every drawn object here
    was a half-pixel fatter than the person drew it (a 30 px square supervised
    31 px), which is a boundary the model would have learnt.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        exterior = _ring_coords(polygon.exterior)
        if exterior is None:
            continue
        holes = [
            coords
            for coords in (_ring_coords(ring) for ring in polygon.interiors)
            if coords is not None
        ]
        paint_rings(mask, [exterior, *holes], 1, x0=x0, y0=y0)
    return mask


# ---------------------------------------------------------------------------
# One crop
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedCrop:
    """One completed ROI, with everything needed to train on it or score it.

    ``gt`` and ``valid`` are ``uint8`` 0/1 at the asset's native scale and are
    always populated. ``em`` and ``prob`` are loaded only when asked for, because
    threshold calibration needs no pixels of the image itself and head training
    needs no stored probability map.
    """

    id: str
    name: str
    #: Source asset id. Two crops with different ``image_key`` can be split
    #: image-disjointly; two with the same one cannot.
    image_key: str
    segmentation_id: str
    x: int
    y: int
    width: int
    height: int
    n_objects: int
    gt: np.ndarray
    valid: np.ndarray
    #: Which record this came from: :data:`SOURCE_CONFIRMED_AREA` or
    #: :data:`SOURCE_DONE_ROI`.
    source: str = SOURCE_CONFIRMED_AREA
    pixel_size_nm: float | None = None
    em: np.ndarray | None = None
    prob: np.ndarray | None = None
    #: True when the only probability map covering this crop was composited from
    #: ROI runs, so unrun areas read as confident background.
    prob_is_composite: bool = False
    has_probability: bool = False

    @property
    def annotated_px(self) -> int:
        return int(self.valid.sum())

    @property
    def foreground_px(self) -> int:
        return int((self.gt > 0).sum())

    def target(self, ignore_index: int = IGNORE) -> np.ndarray:
        """Training target: 0/1 inside the ROI, ``ignore_index`` outside it."""
        tgt = (self.gt > 0).astype(np.int64)
        tgt[self.valid == 0] = ignore_index
        return tgt

    def as_api_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "image_key": self.image_key,
            "segmentation_id": self.segmentation_id,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "n_objects": self.n_objects,
            "annotated_px": self.annotated_px,
            "foreground_px": self.foreground_px,
            "pixel_size_nm": self.pixel_size_nm,
            "has_probability": self.has_probability,
        }


@dataclass
class CropSet:
    """Every annotated crop reachable from one segmentation, plus the verdict."""

    crops: list[AnnotatedCrop] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers and bool(self.crops)

    @property
    def n_images(self) -> int:
        return len({c.image_key for c in self.crops})

    @property
    def confirmed_areas(self) -> int:
        """Completed-ROI polygons that produced a crop."""
        return sum(1 for c in self.crops if c.source == SOURCE_CONFIRMED_AREA)

    @property
    def done_rois(self) -> int:
        """ROIs marked done that produced a crop of their own."""
        return sum(1 for c in self.crops if c.source == SOURCE_DONE_ROI)

    @property
    def annotation_count(self) -> int:
        """The number the dialog leads with: records, not tiles.

        The owner's own example -- two images with three annotations each and a
        third with one shows **7** -- is this number. A region larger than one
        training tile is still one annotation here; the tile count is reported
        beside it and is a different fact.
        """
        return len(self.crops)

    def per_image_counts(self) -> dict[str, dict[str, int]]:
        """``{asset id: {"confirmed_areas", "done_rois", "annotation_count"}}``."""
        out: dict[str, dict[str, int]] = {}
        for crop in self.crops:
            entry = out.setdefault(
                crop.image_key,
                {"confirmed_areas": 0, "done_rois": 0, "annotation_count": 0},
            )
            if crop.source == SOURCE_DONE_ROI:
                entry["done_rois"] += 1
            else:
                entry["confirmed_areas"] += 1
            entry["annotation_count"] += 1
        return out

    @property
    def has_probability(self) -> bool:
        """True when at least one crop is covered by a stored probability map."""
        return any(c.has_probability for c in self.crops)

    @property
    def split_mode(self) -> str:
        return plan_split(self.crops)[2]

    def mode_blockers(self) -> dict[str, list[str]]:
        """Per-mode reasons, so a rung is greyed out instead of the whole page.

        ``blockers`` is what stops everything. This is what stops one mode:
        threshold calibration needs a stored probability map to sweep against,
        head training computes its own. Reporting them separately is what makes
        head training reachable on an image that has been annotated but whose
        map has not been written yet.
        """
        shared = list(self.blockers)
        threshold_only = list(shared)
        if self.crops and not self.has_probability:
            threshold_only.append(NO_PROBABILITY_MESSAGE)
        return {MODE_THRESHOLD_ONLY: threshold_only, MODE_HEAD: shared}

    def as_api_dict(self) -> dict[str, object]:
        train, heldout, mode = plan_split(self.crops)
        return {
            "crops": [c.as_api_dict() for c in self.crops],
            "split_mode": mode,
            "n_images": self.n_images,
            "confirmed_areas": self.confirmed_areas,
            "done_rois": self.done_rois,
            "annotation_count": self.annotation_count,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "has_probability": self.has_probability,
            "mode_blockers": self.mode_blockers(),
            # Honesty rule 2: the UI has to badge the crops the threshold will be
            # fit on, so it is told which they are before the run, not after.
            "train_crop_names": [c.name for c in train],
            "heldout_crop_names": [c.name for c in heldout],
        }


def plan_split(
    crops: Sequence[AnnotatedCrop],
) -> tuple[list[AnnotatedCrop], list[AnnotatedCrop], str]:
    """Split crops into fit / held-out, preferring an image-disjoint split.

    Delegates to :func:`quantem.finetune.calibrate.split_crops` so there is one
    implementation of the rule. That function only reads ``.name`` off each
    element, which is why an :class:`AnnotatedCrop` can stand in for a
    :class:`~quantem.finetune.calibrate.Crop` here — the arrays are not touched.
    """
    return split_crops(  # type: ignore[return-value]
        crops,  # type: ignore[arg-type]
        image_of={c.name: c.image_key for c in crops},
    )


# ---------------------------------------------------------------------------
# Probability maps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProbSource:
    """A stored probability map and the image window it covers."""

    path: object
    x: int
    y: int
    width: int
    height: int
    composite: bool

    def covers(self, x: int, y: int, width: int, height: int) -> bool:
        return (
            x >= self.x
            and y >= self.y
            and x + width <= self.x + self.width
            and y + height <= self.y + self.height
        )


def _prob_sources(
    segmentation: ImageSegmentation, asset_shape: tuple[int, int]
) -> list[_ProbSource]:
    """Every usable stored map for a segmentation, best first.

    Ordering is: real maps newest-first, then composites. A map is refused
    outright when its window cannot be established — an ROI-scoped map that
    records no offset would be read as if it started at (0, 0) and score the
    model against the wrong pixels. Runs written by
    :mod:`quantem.segmentation.prob_maps.persistence` record their window in
    ``metadata["roi"]`` precisely so they are usable here.

    *Composite* maps come last and carry a warning: they are ROI runs pasted
    into a black canvas, so everywhere the model was never run reads as
    confident background.

    A list rather than a single winner, because a user who runs the model over
    one ROI, annotates it, and then does the same on a second ROI has two real
    maps and neither covers the other's crop. Picking one and dropping the other
    silently halved the data calibration was fitted on.
    """
    asset_h, asset_w = asset_shape
    real: list[_ProbSource] = []
    composites: list[_ProbSource] = []
    for prob_map in ProbabilityMap.objects.filter(segmentation=segmentation).order_by(
        "-updated_at", "-created_at"
    ):
        try:
            path = resolve_probability_map_path(prob_map)
        except Exception:  # pragma: no cover - unresolvable storage path
            logger.debug("Cannot resolve probability map %s", prob_map.id, exc_info=True)
            continue
        if not path.exists():
            continue
        try:
            with Image.open(path) as handle:
                width, height = handle.size
        except Exception:  # pragma: no cover - unreadable PNG
            logger.debug("Cannot read probability map %s", path, exc_info=True)
            continue

        metadata = prob_map.metadata if isinstance(prob_map.metadata, dict) else {}
        roi = metadata.get("roi") if isinstance(metadata.get("roi"), dict) else None
        offset_x = int(roi.get("x", 0)) if roi else 0
        offset_y = int(roi.get("y", 0)) if roi else 0
        if roi is None and (int(height), int(width)) != (asset_h, asset_w):
            logger.debug(
                "Skipping probability map %s: %sx%s does not cover the image and "
                "carries no ROI window",
                prob_map.id,
                width,
                height,
            )
            continue
        source = _ProbSource(
            path=path,
            x=offset_x,
            y=offset_y,
            width=int(width),
            height=int(height),
            composite=bool(metadata.get("composite")),
        )
        (composites if source.composite else real).append(source)
    return real + composites


def _source_covering(
    sources: Sequence[_ProbSource], x: int, y: int, width: int, height: int
) -> _ProbSource | None:
    """The best-ranked stored map that fully contains a crop window."""
    for source in sources:
        if source.covers(x, y, width, height):
            return source
    return None


def _load_prob_window(source: _ProbSource, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop the stored map to a window, as float32 in ``[0, 1]``."""
    with Image.open(source.path) as handle:
        if handle.mode != "L":
            handle = handle.convert("L")
        left = x - source.x
        top = y - source.y
        window = handle.crop((left, top, left + width, top + height))
        return np.asarray(window, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _sibling_segmentations(
    segmentation: ImageSegmentation,
    include_siblings: bool,
    asset_ids: Collection[str] | None = None,
) -> list[ImageSegmentation]:
    """The requested segmentation first, then others of the same organelle.

    Ordering matters: :func:`plan_split` sorts by image key, but naming and the
    "which image did the user ask about" question both follow this order.

    ``asset_ids`` narrows the siblings to an explicit selection instead of the
    whole library. The requested segmentation is kept at the head whether or not
    its own asset is in the selection -- it is the one the caller asked about,
    and dropping it silently would make "adapt from here" return crops with no
    connection to *here*. Callers that want a pure scope use
    :func:`collect_crops_for_scope`, which has no privileged segmentation.
    """
    if not include_siblings:
        return [segmentation]
    others = (
        ImageSegmentation.objects.filter(
            segmentation_type_id=segmentation.segmentation_type_id,
            asset__isnull=False,
        )
        .exclude(id=segmentation.id)
        .exclude(asset__lifecycle_status="DELETED")
        .select_related("asset")
        .order_by("created_at")
    )
    if asset_ids is not None:
        others = others.filter(asset_id__in=[str(value) for value in asset_ids])
    return [segmentation, *others]


def _scope_segmentations(
    segmentation_type_id: str, asset_ids: Collection[str]
) -> list[ImageSegmentation]:
    """Every segmentation of one organelle over an explicit set of images."""
    if not asset_ids:
        return []
    return list(
        ImageSegmentation.objects.filter(
            segmentation_type_id=segmentation_type_id,
            asset_id__in=[str(value) for value in asset_ids],
        )
        .exclude(asset__lifecycle_status="DELETED")
        .select_related("asset")
        .order_by("created_at")
    )


def _confirmed_objects_in(
    segmentation: ImageSegmentation, x0: int, y0: int, x1: int, y1: int
) -> list[Polygon]:
    """Confirmed object polygons whose bbox meets a window, geometry loaded."""
    rows = SegmentObject.objects.filter(
        segmentation=segmentation,
        label_state=CONFIRMED,
        bbox_minx__lt=x1,
        bbox_maxx__gt=x0,
        bbox_miny__lt=y1,
        bbox_maxy__gt=y0,
    ).only(
        "geometry_wkb",
        "bbox_minx",
        "bbox_miny",
        "bbox_maxx",
        "bbox_maxy",
        "segmentation",
    )
    polygons: list[Polygon] = []
    for row in rows:
        geometry = row.geometry
        if isinstance(geometry, Polygon) and not geometry.is_empty:
            polygons.append(geometry)
    return polygons


def _unique(name: str, taken: set[str]) -> str:
    """Guard against two assets whose ids share their first eight characters.

    Crop names key the split, the per-crop table and the prediction cache, so a
    collision would silently merge two regions rather than fail.
    """
    candidate, suffix = name, 1
    while candidate in taken:
        candidate = f"{name}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


@dataclass(frozen=True)
class AnnotatedRegion:
    """One record that says "this area is exhaustively annotated", clipped.

    The unit both the counter and the cropper work in, so that "the dialog says
    7" and "seven crops were built" cannot come apart.
    """

    record_id: str
    source: str
    outline: Polygon
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def usable(self) -> bool:
        return self.width >= MIN_CROP_PX and self.height >= MIN_CROP_PX


def annotated_regions(
    segmentation: ImageSegmentation, asset_w: int, asset_h: int
) -> tuple[list[AnnotatedRegion], list[Polygon]]:
    """Every annotated region on one segmentation, and the polygons among them.

    Completed areas first, then the ROIs marked done that add something a
    completed area does not already cover. The second list is the completed
    polygons, which the caller needs to strike out of a done ROI's valid mask.

    Reads :class:`~quantem.segmentation.models.RoiSegmentationStatus`, whose own
    docstring states the ground-truth contract this module implements: inside a
    ROI marked complete for an organelle the labels are dense, so an unlabelled
    pixel there is background, and outside it nothing is known. That is the same
    contract a completed polygon carries, which is why the two are one list.

    Touches no pixels, so a count can be answered for the whole library without
    rasterising anything.
    """
    regions: list[AnnotatedRegion] = []
    polygons: list[Polygon] = []

    for roi in CompletedROI.objects.filter(segmentation=segmentation).order_by("created_at", "id"):
        polygon = roi.geometry
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            continue
        polygons.append(polygon)
        minx, miny, maxx, maxy = polygon.bounds
        regions.append(
            AnnotatedRegion(
                record_id=str(roi.id),
                source=SOURCE_CONFIRMED_AREA,
                outline=polygon,
                x0=max(0, int(np.floor(minx))),
                y0=max(0, int(np.floor(miny))),
                x1=min(asset_w, int(np.ceil(maxx))),
                y1=min(asset_h, int(np.ceil(maxy))),
            )
        )

    covered = unary_union(polygons) if polygons else None
    for row in (
        RoiSegmentationStatus.objects.filter(segmentation=segmentation, is_complete=True)
        .select_related("image_roi")
        .order_by("completed_at", "created_at", "id")
    ):
        roi = row.image_roi
        if roi is None:
            continue
        x0 = max(0, int(roi.x))
        y0 = max(0, int(roi.y))
        x1 = min(asset_w, int(roi.x) + int(roi.width))
        y1 = min(asset_h, int(roi.y) + int(roi.height))
        if x1 <= x0 or y1 <= y0:
            continue
        rectangle = box(x0, y0, x1, y1)
        # De-overlap, in the completed area's favour: a done ROI that says
        # nothing a hand-drawn area has not already said is not a second
        # annotation, and counting it would inflate the number on the dialog as
        # well as training on the same pixels twice.
        if covered is not None and rectangle.difference(covered).is_empty:
            continue
        regions.append(
            AnnotatedRegion(
                record_id=str(row.id),
                source=SOURCE_DONE_ROI,
                outline=rectangle,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
        )
    return regions, polygons


def count_annotations(
    segmentation_type_id: str, asset_ids: Collection[str] | None = None
) -> dict[str, dict[str, int]]:
    """``{asset id: {"confirmed_areas", "done_rois", "annotation_count"}}``.

    The number the dialog leads with, for a whole library or for a selection,
    without loading an image or rasterising a mask. Same rule as
    :func:`collect_crops_for_scope` produces crops by, so the count the user
    reads and the crops the run trains on are the same set.
    """
    segmentations = ImageSegmentation.objects.filter(
        segmentation_type_id=str(segmentation_type_id), asset__isnull=False
    ).exclude(asset__lifecycle_status="DELETED")
    if asset_ids is not None:
        segmentations = segmentations.filter(asset_id__in=[str(value) for value in asset_ids])

    counts: dict[str, dict[str, int]] = {}
    for seg in segmentations.select_related("asset"):
        asset = seg.asset
        asset_w = int(asset.logical_width or 0)
        asset_h = int(asset.logical_height or 0)
        if asset_w <= 0 or asset_h <= 0:
            continue
        regions, _polygons = annotated_regions(seg, asset_w, asset_h)
        usable = [region for region in regions if region.usable]
        if not usable:
            continue
        entry = counts.setdefault(
            str(asset.id),
            {"confirmed_areas": 0, "done_rois": 0, "annotation_count": 0},
        )
        for region in usable:
            if region.source == SOURCE_DONE_ROI:
                entry["done_rois"] += 1
            else:
                entry["confirmed_areas"] += 1
            entry["annotation_count"] += 1
    return counts


def _collect(
    segmentations: Sequence[ImageSegmentation],
    *,
    primary: ImageSegmentation | None,
    load_em: bool,
    load_prob: bool,
    require_probability: bool,
    nothing_annotated: str,
) -> CropSet:
    """The one collection loop, over whatever set of segmentations it is given.

    Both entry points below funnel through here so that the completed-ROI
    contract, the done-ROI contract, the probability-map resolution and the
    blocker vocabulary exist once.
    """
    crop_set = CropSet()
    names: set[str] = set()

    total_regions = 0
    total_objects = 0
    seen_any_region = False
    from_primary = 0

    for seg in segmentations:
        asset = seg.asset
        if asset is None:
            continue

        asset_w = int(asset.logical_width or 0)
        asset_h = int(asset.logical_height or 0)
        if asset_w <= 0 or asset_h <= 0:
            if CompletedROI.objects.filter(segmentation=seg).exists() or (
                RoiSegmentationStatus.objects.filter(segmentation=seg, is_complete=True).exists()
            ):
                seen_any_region = True
                crop_set.warnings.append(
                    f"{asset.display_name}: image dimensions are unknown, so its "
                    "annotated regions were skipped."
                )
            continue

        regions, polygons = annotated_regions(seg, asset_w, asset_h)
        if not regions:
            continue
        seen_any_region = True

        openable = get_asset_openable(asset, require=False) if load_em else None
        if load_em and openable is None:
            crop_set.warnings.append(
                f"{asset.display_name}: no local image file, so its annotated "
                "regions cannot be used for training."
            )
            continue

        # Always resolved, even when the arrays are not wanted: `has_probability`
        # is what tells the crops endpoint whether calibration can run at all.
        prob_sources = _prob_sources(seg, (asset_h, asset_w))
        used_a_composite = False
        if asset.pixel_size_nm is None:
            crop_set.warnings.append(
                f"{asset.display_name}: pixel size is unset, so the image cannot "
                "be resampled to the model's canonical resolution."
            )

        for index, region in enumerate(regions):
            total_regions += 1
            if not region.usable:
                crop_set.warnings.append(
                    f"{asset.display_name}: an annotated area of "
                    f"{region.width}x{region.height} px is too small to use "
                    f"(minimum {MIN_CROP_PX} px per side)."
                )
                continue

            x0, y0 = region.x0, region.y0
            width, height = region.width, region.height
            shape = (height, width)
            valid = _rasterize([region.outline], x0, y0, shape)
            if region.source == SOURCE_DONE_ROI and polygons:
                # De-overlap, in the completed area's favour: see the module
                # docstring. Struck-out pixels become `ignore` for this crop,
                # which is exactly right -- the area is still supervised, by the
                # completed-area crop that owns it, and it is neither trained on
                # nor scored twice.
                already = _rasterize(polygons, x0, y0, shape)
                valid[already > 0] = 0
            if not valid.any():
                continue

            objects = _confirmed_objects_in(seg, x0, y0, region.x1, region.y1)
            inside = [p for p in objects if p.intersects(region.outline)]
            gt = _rasterize(inside, x0, y0, shape)
            np.bitwise_and(gt, valid, out=gt)  # GT only inside the annotated area
            total_objects += len(inside)

            suffix = "done" if region.source == SOURCE_DONE_ROI else ""
            crop = AnnotatedCrop(
                id=region.record_id,
                name=_unique(f"{str(asset.id)[:8]}_{suffix}{index}", names),
                image_key=str(asset.id),
                segmentation_id=str(seg.id),
                source=region.source,
                x=x0,
                y=y0,
                width=width,
                height=height,
                n_objects=len(inside),
                gt=gt,
                valid=valid,
                pixel_size_nm=(float(asset.pixel_size_nm) if asset.pixel_size_nm else None),
            )

            covering = _source_covering(prob_sources, x0, y0, width, height)
            if covering is not None:
                crop.has_probability = True
                crop.prob_is_composite = covering.composite
                used_a_composite = used_a_composite or covering.composite
                if load_prob:
                    crop.prob = _load_prob_window(covering, x0, y0, width, height)
            if load_em and openable is not None:
                crop.em = load_image_roi_array(openable, x0, y0, width, height)

            crop_set.crops.append(crop)
            if primary is not None and seg.id == primary.id:
                from_primary += 1

        if used_a_composite:
            # Warned once per asset, and only when a crop actually fell back to
            # a composite: the map is ROI runs pasted into a black canvas, so
            # everywhere the model was never run scores as confident background.
            crop_set.warnings.append(
                f"{asset.display_name}: the probability map was composited from "
                "ROI runs, so anywhere the model was never run reads as "
                "background."
            )

    if primary is not None and crop_set.crops and not from_primary:
        crop_set.warnings.append(
            "This image has no annotated area of its own, so the adapter will be "
            "fitted on regions annotated in other images."
        )

    if not seen_any_region:
        crop_set.blockers.append(nothing_annotated)
    elif not crop_set.crops:
        crop_set.blockers.append(
            f"{total_regions} annotated area(s) found, but none of them could be "
            "used (too small, already covered by another area, or the image is "
            "unavailable)."
        )
    elif total_objects == 0:
        crop_set.blockers.append(
            "No confirmed objects inside the annotated area. Confirm the objects "
            "you want the model to learn — an empty region gives Dice nothing to "
            "measure."
        )
    elif not crop_set.has_probability:
        if require_probability:
            crop_set.blockers.append(NO_PROBABILITY_MESSAGE)
        else:
            crop_set.warnings.append(NO_PROBABILITY_MESSAGE)

    _add_split_warnings(crop_set)
    return crop_set


#: What the labeling view says when this organelle has nothing to learn from.
NOTHING_ANNOTATED_HERE = (
    "Nothing is marked as finished for this organelle yet. Mark the area you "
    "have finished annotating as complete, or tick a region as done: inside it "
    "every confirmed object is foreground and everything else is background, "
    "and without that there is no valid background to score against."
)

#: What the fine-tune dialog says when the chosen images have nothing to learn
#: from. Names the selection rather than "this image", because the user picked
#: a set and the answer is about the set.
NOTHING_ANNOTATED_IN_SCOPE = (
    "None of the images you chose has a finished area for this organelle. "
    "Mark an area complete, or tick a region as done, on at least one of them."
)


def collect_crops(
    segmentation: ImageSegmentation,
    *,
    include_siblings: bool = True,
    asset_ids: Collection[str] | None = None,
    load_em: bool = False,
    load_prob: bool = False,
    require_probability: bool = False,
) -> CropSet:
    """Gather every annotated region reachable from ``segmentation`` as a crop.

    Never raises for "the user has not annotated enough yet" — that state is
    reported in :attr:`CropSet.blockers` so the crops endpoint can render it.
    Use :func:`require_crops` when the answer must be usable.

    Args:
        segmentation: the segmentation the user is working in.
        include_siblings: also gather annotations for the same organelle on
            other assets. This is what makes an image-disjoint split possible.
        asset_ids: restrict the siblings to these assets. ``None``, the default,
            means the whole library, which is what the labeling view wants; an
            explicit set is what a fine-tune scoped to chosen images wants.
        load_em: load the EM pixels for each crop (head training needs them).
        load_prob: load the stored probability map window for each crop
            (threshold calibration needs it).
        require_probability: treat "no stored probability map" as a hard blocker
            rather than a warning. Only threshold calibration should ask for
            this — it has nothing to sweep without a map — and the default is
            ``False`` because head training computes its own. It used to default
            to ``True``, which meant the crops endpoint and the start endpoint
            refused *every* mode over a missing map.
    """
    return _collect(
        _sibling_segmentations(segmentation, include_siblings, asset_ids),
        primary=segmentation,
        load_em=load_em,
        load_prob=load_prob,
        require_probability=require_probability,
        nothing_annotated=NOTHING_ANNOTATED_HERE,
    )


def collect_crops_for_scope(
    segmentation_type_id: str,
    asset_ids: Collection[str],
    *,
    load_em: bool = False,
    load_prob: bool = False,
    require_probability: bool = False,
) -> CropSet:
    """Every annotated region for one organelle over an explicit set of images.

    The scope-first counterpart of :func:`collect_crops`: no image is privileged,
    so there is no "this image has none of its own" warning and the refusal
    names the selection. Same loop, same contract.
    """
    return _collect(
        _scope_segmentations(str(segmentation_type_id), asset_ids),
        primary=None,
        load_em=load_em,
        load_prob=load_prob,
        require_probability=require_probability,
        nothing_annotated=NOTHING_ANNOTATED_IN_SCOPE,
    )


def _add_split_warnings(crop_set: CropSet) -> None:
    """Say what kind of held-out number this data can produce, before it is run.

    Honesty rule 1: a within-image score and an image-disjoint score measure
    different things and must never be presented as the same number.
    """
    if not crop_set.crops:
        return
    mode = crop_set.split_mode
    if mode == "no-heldout":
        crop_set.warnings.append(
            "Only one annotated region, so everything is used to fit and there is "
            "no held-out score at all."
        )
    elif mode == "within-image":
        crop_set.warnings.append(
            "Only one image is annotated, so the held-out score is within-image: "
            "it does not measure generalisation to a new image. Annotate a "
            "region on a second image for an image-disjoint score."
        )


def require_crops(
    segmentation: ImageSegmentation,
    *,
    include_siblings: bool = True,
    asset_ids: Collection[str] | None = None,
    load_em: bool = False,
    load_prob: bool = False,
    require_probability: bool = False,
) -> CropSet:
    """:func:`collect_crops`, but refusing when the data cannot support a score.

    Raises:
        CompletedRoiRequired: with the first blocker as its message. The
            reference implementation refused in exactly this state rather than
            reporting a Dice that would have been meaningless.
    """
    return _require(
        collect_crops(
            segmentation,
            include_siblings=include_siblings,
            asset_ids=asset_ids,
            load_em=load_em,
            load_prob=load_prob,
            require_probability=require_probability,
        )
    )


def require_crops_for_scope(
    segmentation_type_id: str,
    asset_ids: Collection[str],
    *,
    load_em: bool = False,
    load_prob: bool = False,
    require_probability: bool = False,
) -> CropSet:
    """:func:`collect_crops_for_scope`, refusing when the data cannot support one."""
    return _require(
        collect_crops_for_scope(
            segmentation_type_id,
            asset_ids,
            load_em=load_em,
            load_prob=load_prob,
            require_probability=require_probability,
        )
    )


def _require(crop_set: CropSet) -> CropSet:
    if not crop_set.ready:
        raise CompletedRoiRequired(
            crop_set.blockers[0]
            if crop_set.blockers
            else "There is nothing annotated to adapt to yet."
        )
    return crop_set

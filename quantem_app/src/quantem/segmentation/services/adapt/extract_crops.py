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
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.task_utils import load_image_roi_array
from quantem.finetune.calibrate import split_crops
from quantem.seg_core.rasterize import paint_rings
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    ProbabilityMap,
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


def _rasterize(
    polygons: Iterable[Polygon], x0: int, y0: int, shape: tuple[int, int]
) -> np.ndarray:
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


def _load_prob_window(
    source: _ProbSource, x: int, y: int, width: int, height: int
) -> np.ndarray:
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
    segmentation: ImageSegmentation, include_siblings: bool
) -> list[ImageSegmentation]:
    """The requested segmentation first, then others of the same organelle.

    Ordering matters: :func:`plan_split` sorts by image key, but naming and the
    "which image did the user ask about" question both follow this order.
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
    return [segmentation, *others]


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


def collect_crops(
    segmentation: ImageSegmentation,
    *,
    include_siblings: bool = True,
    load_em: bool = False,
    load_prob: bool = False,
    require_probability: bool = False,
) -> CropSet:
    """Gather every completed ROI reachable from ``segmentation`` as a crop.

    Never raises for "the user has not annotated enough yet" — that state is
    reported in :attr:`CropSet.blockers` so the crops endpoint can render it.
    Use :func:`require_crops` when the answer must be usable.

    Args:
        segmentation: the segmentation the user is working in.
        include_siblings: also gather annotations for the same organelle on
            other assets. This is what makes an image-disjoint split possible.
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
    crop_set = CropSet()
    segmentations = _sibling_segmentations(segmentation, include_siblings)
    names: set[str] = set()

    total_rois = 0
    total_objects = 0
    seen_any_roi = False
    from_this_segmentation = 0

    for seg in segmentations:
        asset = seg.asset
        if asset is None:
            continue
        rois = list(
            CompletedROI.objects.filter(segmentation=seg).order_by("created_at", "id")
        )
        if not rois:
            continue
        seen_any_roi = True

        asset_w = int(asset.logical_width or 0)
        asset_h = int(asset.logical_height or 0)
        if asset_w <= 0 or asset_h <= 0:
            crop_set.warnings.append(
                f"{asset.display_name}: image dimensions are unknown, so its "
                "annotated regions were skipped."
            )
            continue

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

        for index, roi in enumerate(rois):
            total_rois += 1
            polygon = roi.geometry
            if not isinstance(polygon, Polygon) or polygon.is_empty:
                continue
            minx, miny, maxx, maxy = polygon.bounds
            x0 = max(0, int(np.floor(minx)))
            y0 = max(0, int(np.floor(miny)))
            x1 = min(asset_w, int(np.ceil(maxx)))
            y1 = min(asset_h, int(np.ceil(maxy)))
            width, height = x1 - x0, y1 - y0
            if width < MIN_CROP_PX or height < MIN_CROP_PX:
                crop_set.warnings.append(
                    f"{asset.display_name}: a completed area of {width}x{height} px "
                    f"is too small to use (minimum {MIN_CROP_PX} px per side)."
                )
                continue

            shape = (height, width)
            valid = _rasterize([polygon], x0, y0, shape)
            objects = _confirmed_objects_in(seg, x0, y0, x1, y1)
            inside = [p for p in objects if p.intersects(polygon)]
            gt = _rasterize(inside, x0, y0, shape)
            np.bitwise_and(gt, valid, out=gt)  # GT only inside the ROI
            total_objects += len(inside)

            crop = AnnotatedCrop(
                id=str(roi.id),
                name=_unique(f"{str(asset.id)[:8]}_{index}", names),
                image_key=str(asset.id),
                segmentation_id=str(seg.id),
                x=x0,
                y=y0,
                width=width,
                height=height,
                n_objects=len(inside),
                gt=gt,
                valid=valid,
                pixel_size_nm=(
                    float(asset.pixel_size_nm) if asset.pixel_size_nm else None
                ),
            )

            source = _source_covering(prob_sources, x0, y0, width, height)
            if source is not None:
                crop.has_probability = True
                crop.prob_is_composite = source.composite
                used_a_composite = used_a_composite or source.composite
                if load_prob:
                    crop.prob = _load_prob_window(source, x0, y0, width, height)
            if load_em and openable is not None:
                crop.em = load_image_roi_array(openable, x0, y0, width, height)

            crop_set.crops.append(crop)
            if seg.id == segmentation.id:
                from_this_segmentation += 1

        if used_a_composite:
            # Warned once per asset, and only when a crop actually fell back to
            # a composite: the map is ROI runs pasted into a black canvas, so
            # everywhere the model was never run scores as confident background.
            crop_set.warnings.append(
                f"{asset.display_name}: the probability map was composited from "
                "ROI runs, so anywhere the model was never run reads as "
                "background."
            )

    if crop_set.crops and not from_this_segmentation:
        crop_set.warnings.append(
            "This image has no completed area of its own, so the adapter will be "
            "fitted on regions annotated in other images."
        )

    if not seen_any_roi:
        crop_set.blockers.append(
            "No completed ROI on this image. Mark the area you have finished "
            "annotating as complete: inside it every confirmed object is "
            "foreground and everything else is background, and without that "
            "there is no valid background to score against."
        )
    elif not crop_set.crops:
        crop_set.blockers.append(
            f"{total_rois} completed area(s) found, but none of them could be used "
            "(too small, or the image is unavailable)."
        )
    elif total_objects == 0:
        crop_set.blockers.append(
            "No confirmed objects inside the completed area. Confirm the objects "
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
    crop_set = collect_crops(
        segmentation,
        include_siblings=include_siblings,
        load_em=load_em,
        load_prob=load_prob,
        require_probability=require_probability,
    )
    if not crop_set.ready:
        raise CompletedRoiRequired(
            crop_set.blockers[0]
            if crop_set.blockers
            else "There is nothing annotated to adapt to yet."
        )
    return crop_set

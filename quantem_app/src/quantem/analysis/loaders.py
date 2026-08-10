"""Database state -> the numpy inputs the analysis functions want.

The analysis suite deals in boolean masks, ``(x, y)`` point arrays and plain
feature dicts. QuantEM stores shapely polygons in a ``BinaryField``, centroids in
two indexed float columns and features in JSON. This module is the whole of the
translation, kept in one place so there is exactly one answer to "which objects
counted, and how were they rasterised?".

Three decisions worth stating, because they change every number downstream:

* **Only ``CONFIRMED`` objects count.** Candidate and inferred objects are model
  output the user has not accepted; including them would make a measurement
  depend on a threshold rather than on the data. The confirmed set is the one the
  user vouched for, so it is the one that goes in a paper.
* **Masks are rasterised with holes.** ``segmentation.utils.polygon_to_mask``
  reads ``polygon.exterior`` only, which silently fills every interior ring --
  and the tissue tool's whole purpose is to cut holes (resin, folds, tears) out
  of the denominator. :func:`rasterize_region` paints exteriors and holes and
  handles multipolygons, so it is used instead.
* **Centroids come from the stored columns**, not from recomputing them off the
  polygon. They are what the overlay and the object table already show; deriving
  a second, subtly different centroid here would put two different answers in
  front of the same user.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from django.db.models import Count, QuerySet

from quantem.segmentation.instance_params import (
    INSTANCE_PARAM_KEYS,
    supports_instance_params,
)
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff.render import geometry_to_rings, rasterize_region
from quantem.segmentation.run_identity import (
    RUN_FEATURE_KEY,
    RUN_IDENTITY_KEYS,
    read_run_identity,
)
from quantem.segmentation.source_models import (
    SOURCE_MODEL_MANUAL,
    SOURCE_MODEL_UNKNOWN,
    normalize_source_model,
)
from quantem.segmentation.type_definitions import (
    ER,
    LIPID_DROPLETS,
    MITOCHONDRIA,
    NUCLEUS,
    TISSUE,
)

from . import provenance
from .compartments import CompartmentSet
from .distances import DEFAULT_BAND_EDGES_NM
from .models import AnalysisRun
from .montecarlo import DEFAULT_REPLICATES, DEFAULT_SEED

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .service import AnalysisInputs

logger = logging.getLogger(__name__)

#: The only label state an analysis measures. See the module docstring.
CONFIRMED = "CONFIRMED"

#: Analysis vocabulary for the shipped segmentation types. Three things depend
#: on getting these exact strings: ``API_CONTRACT.md`` writes requests as
#: ``{"nucleus": ..., "mito": ...}``; :func:`quantem.analysis.compartments.
#: area_fractions` special-cases the literal ``"nucleus"`` and is what makes
#: ``cytoplasm`` derivable at all; and they become the ``area_fraction_<name>``
#: column headers a reader has to understand. ``quantem_internal_mito`` is a
#: database key, not a word for a paper.
BUILTIN_COMPARTMENT_NAMES = {
    MITOCHONDRIA.internal_name: "mito",
    ER.internal_name: "er",
    NUCLEUS.internal_name: "nucleus",
    LIPID_DROPLETS.internal_name: "ld",
    TISSUE.internal_name: "tissue",
}

#: Point sources the API accepts. ``centroids`` uses the run's own segmentation;
#: ``csv`` uses an imported ``x,y`` table (spot detections, immunolabels).
POINT_SOURCES = ("centroids", "csv")

#: Upper bound on Monte-Carlo replicates accepted from a request. 1000 replicates
#: on a 4k image is minutes of CPU; beyond that the user wants a batch script,
#: not a desktop button.
MAX_REPLICATES = 1000


class AnalysisInputError(ValueError):
    """A run cannot be assembled from what is in the database.

    Distinct from a programming error so the API can turn it into a 400 with the
    message shown to the user unchanged.
    """


@dataclass(frozen=True)
class LoadedAnalysis:
    """Everything :func:`quantem.analysis.service.run_for_segmentation` needs."""

    inputs: AnalysisInputs
    #: The request after :func:`normalise_params`, re-validated against the
    #: database as it is *now* -- a segmentation can be deleted between the
    #: request and the job that serves it.
    params: dict[str, Any]
    #: Mask provenance for the bundle manifest: which segmentation, which model,
    #: how many confirmed objects went into each compartment.
    provenance: dict[str, Any]
    #: Sentences the *provenance* raised that belong in the result's top-level
    #: ``caveats`` list, not only three levels down in the manifest JSON. A
    #: threshold that might not be the one that ran is exactly the kind of thing
    #: a reader has to be told in the yellow box rather than left to find.
    caveats: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Image geometry
# ---------------------------------------------------------------------------


def image_shape(segmentation: ImageSegmentation) -> tuple[int, int]:
    """``(height, width)`` of the image a segmentation belongs to."""
    asset = segmentation.asset
    if asset is None:
        raise AnalysisInputError(
            f"Segmentation {segmentation.id} is not attached to an image."
        )
    height = int(asset.logical_height or 0)
    width = int(asset.logical_width or 0)
    if height <= 0 or width <= 0:
        raise AnalysisInputError(
            f"Image {asset.display_name!r} has no recorded size, so its masks "
            "cannot be rasterised. Re-import the image."
        )
    return height, width


def compartment_name(segmentation: ImageSegmentation) -> str:
    """The analysis-vocabulary name for a segmentation's organelle.

    A type the user created themselves keeps its own internal name -- QuantEM
    does not know what they called it and will not rename it.
    """
    internal = segmentation.segmentation_type.internal_name
    return BUILTIN_COMPARTMENT_NAMES.get(internal, internal)


def pixel_size_nm(segmentation: ImageSegmentation) -> float | None:
    """Nanometres per pixel, or ``None`` when the image is uncalibrated.

    ``None`` is propagated rather than defaulted: assuming a pixel size produces
    wrong micron values for every image not acquired at that scale.
    """
    asset = segmentation.asset
    if asset is None or not asset.pixel_size_nm:
        return None
    value = float(asset.pixel_size_nm)
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


def confirmed_objects(segmentation: ImageSegmentation) -> QuerySet[SegmentObject]:
    """The confirmed ``SegmentObject`` rows of a segmentation, oldest first."""
    return (
        SegmentObject.objects.filter(
            segmentation=segmentation, label_state=CONFIRMED
        )
        .order_by("created_at", "id")
    )


def object_features(segmentation: ImageSegmentation) -> dict[str, dict[str, Any]]:
    """Stored per-object features, keyed by object id.

    The values are in pixels -- :func:`quantem.analysis.morphometrics.derive`
    converts them, and refuses to when ``pixel_size_nm`` is unknown.
    """
    return {
        str(object_id): (features or {})
        for object_id, features in confirmed_objects(segmentation).values_list(
            "id", "features"
        )
    }


def object_sources(segmentation: ImageSegmentation) -> dict[str, str]:
    """Which model produced each confirmed object, keyed by object id.

    ``"manual"`` is a person's own polygon. This is what makes a partly-filled
    column legible: a metric with n=4 out of 90 is not a mystery once the four
    are named as the hand-drawn ones.
    """
    return {
        str(object_id): normalize_source_model(source) or SOURCE_MODEL_UNKNOWN
        for object_id, source in confirmed_objects(segmentation).values_list(
            "id", "source_model"
        )
    }


def source_counts(segmentation: ImageSegmentation) -> dict[str, int]:
    """Confirmed objects per ``source_model``, e.g. ``{"manual": 4, "quantem:mito": 86}``."""
    counts: dict[str, int] = {}
    rows = (
        confirmed_objects(segmentation)
        .values_list("source_model")
        .annotate(n=Count("id"))
    )
    for source, n in rows:
        key = normalize_source_model(source) or SOURCE_MODEL_UNKNOWN
        counts[key] = counts.get(key, 0) + int(n)
    return dict(sorted(counts.items()))


def object_centroids(segmentation: ImageSegmentation) -> np.ndarray:
    """Confirmed object centroids as an ``(N, 2)`` array of ``(x, y)`` pixels."""
    rows = list(
        confirmed_objects(segmentation).values_list("centroid_x", "centroid_y")
    )
    if not rows:
        return np.empty((0, 2), dtype=float)
    return np.asarray(rows, dtype=float)


def segmentation_mask(
    segmentation: ImageSegmentation, shape: tuple[int, int]
) -> np.ndarray:
    """Union of a segmentation's confirmed object polygons as a boolean mask.

    Each object is painted into its own bounding box rather than into a
    full-image canvas, so a segmentation with ten thousand small objects costs
    the area of those objects and not ten thousand image-sized allocations.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)

    fields = (
        "geometry_wkb",
        "bbox_minx",
        "bbox_miny",
        "bbox_maxx",
        "bbox_maxy",
    )
    for obj in confirmed_objects(segmentation).only("id", *fields).iterator():
        rings = geometry_to_rings(obj.geometry)
        if not rings:
            continue
        x0 = max(0, int(math.floor(obj.bbox_minx)))
        y0 = max(0, int(math.floor(obj.bbox_miny)))
        x1 = min(width, int(math.ceil(obj.bbox_maxx)) + 1)
        y1 = min(height, int(math.ceil(obj.bbox_maxy)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        labels, _border = rasterize_region(
            [{"label": 1, "priority": 0, "area": 0.0, "rings": rings}],
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            border_width=1,
        )
        mask[y0:y1, x0:x1] |= labels != 0
    return mask


# ---------------------------------------------------------------------------
# Compartments
# ---------------------------------------------------------------------------


def _nesting(names: list[str]) -> dict[str, str]:
    """Which compartments are subsets of another, for the manifest.

    ``cytoplasm`` is derived as ``tissue AND NOT nucleus`` whenever a nucleus
    mask is present, so every other organelle sits *inside* cytoplasm rather than
    beside it. Recording that is what stops a reader adding the fractions up and
    expecting 1.0. With no nucleus mask there is no derived cytoplasm and nothing
    to record.
    """
    if "nucleus" not in names:
        return {}
    return {
        name: "cytoplasm"
        for name in names
        if name not in {"nucleus", "cytoplasm", "tissue"}
    }


def build_compartment_set(
    compartments: dict[str, ImageSegmentation],
    *,
    tissue: ImageSegmentation | None,
    shape: tuple[int, int],
) -> CompartmentSet:
    """Rasterise several segmentations of one image into a ``CompartmentSet``."""
    masks = {
        name: segmentation_mask(segmentation, shape)
        for name, segmentation in compartments.items()
    }
    tissue_mask = segmentation_mask(tissue, shape) if tissue is not None else None
    return CompartmentSet(
        masks=masks,
        tissue=tissue_mask,
        nested_in=_nesting(list(masks)),
    )


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


#: How many unreadable point rows are spelled out individually before the
#: caveat stops listing line numbers. A detection export can have hundreds; the
#: count is the fact, the first few line numbers are how the user finds them.
MAX_NAMED_BAD_POINT_LINES = 5


@dataclass(frozen=True)
class ParsedPoints:
    """The coordinates a points CSV actually contained, and what it did not."""

    #: ``(N, 2)`` float array of finite pixel coordinates. Never empty --
    #: :func:`parse_points_csv` raises when nothing readable survives.
    xy: np.ndarray
    #: ``(line_number, text)`` for each row whose coordinates could not be read
    #: as a position. In file order.
    unreadable: tuple[tuple[int, str], ...] = ()

    @property
    def n_unreadable(self) -> int:
        return len(self.unreadable)

    def caveat(self) -> str | None:
        """The sentence the bundle has to carry, or ``None`` if the file was clean."""
        if not self.unreadable:
            return None
        n = len(self.unreadable)
        named = self.unreadable[:MAX_NAMED_BAD_POINT_LINES]
        spelled = "; ".join(f"line {line} ({text})" for line, text in named)
        if n > len(named):
            spelled += f"; and {n - len(named)} more"
        total = int(self.xy.shape[0]) + n
        return (
            f"{n} of {total} rows in the imported points CSV had a coordinate "
            f"that is not a position and were dropped before any measurement: "
            f"{spelled}. A missing or infinite coordinate is missing, not (0, 0) "
            "— it used to round and clip to the image origin and be counted as "
            "a real point there. Every count, fraction and enrichment in this "
            f"bundle is over the {self.xy.shape[0]} rows that could be read."
        )


def parse_points_csv(text: str) -> ParsedPoints:
    """Parse an ``x,y`` CSV of pixel coordinates, and say what was unreadable.

    A header row is optional and detected by trying to read the first row as
    numbers. Coordinates are image pixels, matching everything else in the
    database; a CSV in nanometres would be silently wrong and there is no way to
    tell the two apart, so the API documents pixels and this refuses anything
    that is not a number.

    ``nan``, ``inf``, ``-inf`` and a literal that overflows a float on
    round-trip (``1e400``) all parse through :func:`float` without complaint and
    are **not** numbers a point can sit at. They used to reach
    :func:`~quantem.analysis.compartments.assign_points`, where
    ``np.round(nan).astype(int)`` is ``INT_MIN`` and the clip to the image turns
    that into 0: three unreadable rows in a four-row file produced a real
    observation at pixel (0, 0) each, and an enrichment of 31.0 at z = 13.2 in
    ``image_summary.csv``. ``numpy.savetxt`` writes ``nan`` for a missing value
    and a failed fit writes ``inf``, so an ordinary upstream tool produces this.

    Such a row is **dropped and reported**, not fatal: refusing a 5,000-row
    detection export because three fits failed leaves the user stripping rows by
    hand, which records nothing at all. The line numbers are kept here because
    this is the only place that still knows them --
    :attr:`ParsedPoints.unreadable` carries them and
    :meth:`ParsedPoints.caveat` is the sentence for the bundle. A file with
    *nothing* readable in it is still an error, exactly as an empty one is.
    """
    rows: list[tuple[float, float]] = []
    unreadable: list[tuple[int, str]] = []
    reader = csv.reader(io.StringIO(text))
    for line_number, raw in enumerate(reader, start=1):
        cells = [cell.strip() for cell in raw if cell.strip() != ""]
        if not cells:
            continue
        if len(cells) < 2:
            raise AnalysisInputError(
                f"Points CSV line {line_number} has fewer than two columns."
            )
        try:
            x, y = float(cells[0]), float(cells[1])
        except ValueError:
            if line_number == 1:
                continue  # a header row
            raise AnalysisInputError(
                f"Points CSV line {line_number} is not a pair of numbers: "
                f"{','.join(cells[:2])}"
            ) from None
        if not (math.isfinite(x) and math.isfinite(y)):
            unreadable.append((line_number, ",".join(cells[:2])))
            continue
        rows.append((x, y))
    if not rows:
        if unreadable:
            n = len(unreadable)
            raise AnalysisInputError(
                f"None of the {n} coordinate row(s) in the points CSV is a "
                f"position: every one has a missing or infinite value (line "
                f"{unreadable[0][0]} is {unreadable[0][1]}). There is nothing "
                "to measure."
            )
        raise AnalysisInputError("The points CSV contained no coordinates.")
    return ParsedPoints(
        xy=np.asarray(rows, dtype=float), unreadable=tuple(unreadable)
    )


def _points_provenance(
    points_source: str | None,
    points_xy: np.ndarray | None,
    parsed: ParsedPoints | None,
) -> dict[str, Any]:
    """Where the point set came from, and what was thrown out of it.

    A bundle whose enrichment column is built on 4,997 of 5,000 imported rows
    has to say so somewhere a reader can check, not only in a caveat sentence.
    The dropped rows are named by line number so the file can be fixed.
    """
    if points_source is None:
        return provenance.section(
            {"source": None},
            {
                "n_points": (
                    "This run analysed no point set: enrichment, distances and "
                    "the Monte-Carlo null were not computed."
                )
            },
        )
    values: dict[str, Any] = {
        "source": points_source,
        "n_points": int(points_xy.shape[0]) if points_xy is not None else 0,
        "coordinate_units": "image pixels, x then y",
    }
    if parsed is None:
        values["note"] = (
            "Object centroids from the segmentation being analysed, in image "
            "pixels; they cannot be out of the image or unreadable."
        )
        return provenance.section(values, {})
    values["n_rows_read"] = int(points_xy.shape[0]) + parsed.n_unreadable
    values["n_unreadable"] = parsed.n_unreadable
    values["unreadable_lines"] = [
        {"line": line, "text": text}
        for line, text in parsed.unreadable[:MAX_NAMED_BAD_POINT_LINES]
    ]
    values["note"] = (
        "n_unreadable rows had a missing or infinite coordinate and are in no "
        "count, fraction or enrichment in this bundle. unreadable_lines names "
        f"at most the first {MAX_NAMED_BAD_POINT_LINES}."
    )
    return provenance.section(values, {})


# ---------------------------------------------------------------------------
# Request normalisation
# ---------------------------------------------------------------------------


def _resolve_segmentation(
    raw_id: Any, *, asset_id: Any, role: str
) -> ImageSegmentation:
    try:
        # A malformed id is a bad request, not a 500: Django's UUIDField raises
        # rather than returning no rows when the value is not a UUID at all.
        wanted = uuid.UUID(str(raw_id))
    except (AttributeError, TypeError, ValueError):
        raise AnalysisInputError(f"{raw_id!r} is not a segmentation id.") from None

    segmentation = (
        ImageSegmentation.objects.select_related("asset", "segmentation_type")
        .filter(id=wanted)
        .first()
    )
    if segmentation is None:
        raise AnalysisInputError(f"No segmentation {raw_id} for {role}.")
    if segmentation.asset_id != asset_id:
        raise AnalysisInputError(
            f"The segmentation given for {role} is on a different image. "
            "Compartments must all come from the image being analysed."
        )
    return segmentation


def _reject_non_finite(values: list[float], *, field: str) -> None:
    """Refuse NaN and infinity in a numeric parameter, naming the offender.

    ``float()`` accepts far more than a number: ``"nan"``, ``"inf"``, ``"-inf"``
    and -- with no string involved at all -- a JSON literal like ``1e999``,
    which overflows to infinity while parsing. Every shape check downstream then
    agrees with it. NaN compares False against everything, so a strictly-
    increasing test *passes* ``[0, nan]``; infinity really is larger than any
    edge before it.

    The value therefore reached ``AnalysisRun.params``, where ``json.dumps``
    wrote a bare ``NaN``/``Infinity`` token -- which is not JSON -- and SQLite's
    ``JSON_VALID`` constraint rejected the insert. A bad request became an
    ``IntegrityError`` and an HTTP 500 quoting a database constraint at the user.
    The frontend's own parser has always refused these; this is the validator
    the API contract points at agreeing with it.
    """
    bad = [value for value in values if not math.isfinite(value)]
    if not bad:
        return
    one = len(bad) == 1
    raise AnalysisInputError(
        f"{field} must be finite numbers: "
        f"{', '.join(str(value) for value in bad)} "
        f"{'is not a length' if one else 'are not lengths'}."
    )


def normalise_params(raw: dict[str, Any], *, segmentation: ImageSegmentation) -> dict:
    """Validate an analysis request and fill in its defaults.

    Returns the dict stored on ``AnalysisRun.params`` and read back by
    :func:`load_inputs`. Doing the resolution once, here, is what keeps the API
    layer and the job layer from disagreeing about what was asked for: the job
    never re-reads the raw request.

    Raises :class:`AnalysisInputError` with a sentence meant for the user.
    """
    if segmentation.asset_id is None:
        raise AnalysisInputError(
            "This segmentation is not attached to an image and cannot be analysed."
        )

    raw_compartments = raw.get("compartments") or {}
    if not isinstance(raw_compartments, dict):
        raise AnalysisInputError("compartments must be an object of name -> id.")
    compartments = {str(k): str(v) for k, v in raw_compartments.items()}
    if not compartments:
        # Analysing one segmentation on its own is the common case; name the
        # compartment after the organelle so the exported columns read as
        # "area_fraction_mito" rather than "area_fraction_compartment_0".
        compartments = {compartment_name(segmentation): str(segmentation.id)}

    for name, seg_id in compartments.items():
        if not name.strip():
            raise AnalysisInputError("A compartment name cannot be blank.")
        _resolve_segmentation(seg_id, asset_id=segmentation.asset_id, role=name)

    tissue_id = raw.get("tissue_segmentation_id") or None
    if tissue_id:
        _resolve_segmentation(
            tissue_id, asset_id=segmentation.asset_id, role="the tissue mask"
        )
        tissue_id = str(tissue_id)

    points_source = raw.get("points_source") or None
    if points_source is not None and points_source not in POINT_SOURCES:
        raise AnalysisInputError(
            f"points_source must be one of {', '.join(POINT_SOURCES)}, or null."
        )
    points_csv = str(raw.get("points_csv") or "")
    if points_source == "csv":
        parse_points_csv(points_csv)  # fail now, not three minutes into the job
    else:
        points_csv = ""

    distance_target = raw.get("distance_target") or None
    if distance_target is not None:
        distance_target = str(distance_target)
        if distance_target not in compartments:
            raise AnalysisInputError(
                f"distance_target {distance_target!r} is not one of the "
                f"compartments ({', '.join(sorted(compartments))})."
            )
        if points_source is None:
            raise AnalysisInputError(
                "A distance target needs a point set; set points_source first."
            )

    raw_edges = raw.get("band_edges_nm") or DEFAULT_BAND_EDGES_NM
    try:
        band_edges = [float(edge) for edge in raw_edges]
    except (TypeError, ValueError):
        raise AnalysisInputError("band_edges_nm must be a list of numbers.") from None
    # Before the shape checks, because neither of them catches a non-finite
    # value: NaN fails every comparison, so the monotonic test below passes it,
    # and infinity genuinely increases.
    _reject_non_finite(band_edges, field="band_edges_nm")
    if len(band_edges) < 2:
        raise AnalysisInputError("band_edges_nm needs at least two edges.")
    if any(b <= a for a, b in zip(band_edges, band_edges[1:], strict=False)):
        raise AnalysisInputError("band_edges_nm must increase.")

    try:
        # OverflowError, not just ValueError: int(nan) is a ValueError but
        # int(inf) is an OverflowError, and an uncaught one here is a 500.
        replicates = int(raw.get("replicates", DEFAULT_REPLICATES))
        seed = int(raw.get("seed", DEFAULT_SEED))
    except (TypeError, ValueError, OverflowError):
        raise AnalysisInputError("replicates and seed must be whole numbers.") from None
    if not 1 <= replicates <= MAX_REPLICATES:
        raise AnalysisInputError(f"replicates must be between 1 and {MAX_REPLICATES}.")

    return {
        "compartments": compartments,
        "tissue_segmentation_id": tissue_id,
        "points_source": points_source,
        "points_csv": points_csv,
        "distance_target": distance_target,
        "band_edges_nm": band_edges,
        "replicates": replicates,
        "seed": seed,
        "group": str(raw.get("group") or ""),
    }


# ---------------------------------------------------------------------------
# The load itself
# ---------------------------------------------------------------------------


def _mask_provenance(
    name: str,
    segmentation: ImageSegmentation,
    n_objects: int,
    *,
    shape: tuple[int, int],
) -> dict[str, Any]:
    counts = source_counts(segmentation)
    hand_drawn = counts.get(SOURCE_MODEL_MANUAL, 0)
    return {
        "compartment": name,
        "segmentation_id": str(segmentation.id),
        "segmentation_type": segmentation.segmentation_type.internal_name,
        "n_confirmed_objects": n_objects,
        "source_models": sorted(k for k in counts if k != SOURCE_MODEL_UNKNOWN),
        # The split that explains every metric with a small n: a hand-drawn
        # polygon and a model-produced one do not carry the same measurements.
        "n_confirmed_by_source": counts,
        "n_hand_drawn": hand_drawn,
        "n_model_produced": sum(counts.values()) - hand_drawn,
        # What the confirmed count is the *survivor* of, and how much of the
        # image a person actually went through to produce it. Neither is
        # recoverable from a count of confirmed objects.
        "proofreading": {
            "n_confirmed": n_objects,
            "n_rejected": rejected_count(segmentation),
            "n_by_label_state": label_state_counts(segmentation),
            "reviewed_area": reviewed_area(segmentation, shape),
            "note": (
                "n_by_label_state counts every object row this segmentation "
                "has, not only the confirmed ones this bundle measures. "
                "EXCLUDED is a candidate a person looked at and threw away; "
                "CANDIDATE and INFERRED are model output nobody has ruled on, "
                "and they are in no number in this bundle either."
            ),
        },
        "run": run_provenance(segmentation, compartment=name),
    }


# ---------------------------------------------------------------------------
# Per-object run identity
# ---------------------------------------------------------------------------

#: Where inference records the identity of the run that produced an object, in
#: ``SegmentObject.features``. A hand-drawn polygon carries no such key, and the
#: absence means "not produced by a model" -- a different fact from "produced
#: with settings nobody wrote down". Re-exported from the module that *writes*
#: it rather than restated here: two definitions of a shared contract is one too
#: many, and this side only reads.
RUN_STAMP_KEY = RUN_FEATURE_KEY

#: The fields a stamp carries, in contract order.
RUN_STAMP_FIELDS: tuple[str, ...] = RUN_IDENTITY_KEYS


@dataclass(frozen=True)
class RunStamps:
    """Every confirmed object of one segmentation, with the run that made it.

    The pair, not just the stamps: an object that has no stamp is either a
    person's own polygon (nothing to record) or a model object made before
    QuantEM began stamping (settings unrecoverable), and the manifest has to
    tell those two apart.
    """

    #: ``(source_model, stamp or None)`` per confirmed object, oldest first.
    objects: tuple[tuple[str, dict[str, Any] | None], ...]
    #: Pixel area of each hand-drawn object that records one.
    #:
    #: Carried because the minimum-area floor is the one run setting that does
    #: not apply to every object in the bundle: ``filter_min_area`` runs inside
    #: inference, and a polygon a person drew never passes through it. Comparing
    #: these against the floor is what turns "the floor is per pack" from a
    #: schema detail into a statement about the object set actually exported.
    #: Read off the same query that reads the stamps, so it costs nothing.
    hand_drawn_areas: tuple[float, ...] = ()

    @property
    def stamps(self) -> list[dict[str, Any]]:
        return [stamp for _source, stamp in self.objects if stamp is not None]

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def n_hand_drawn(self) -> int:
        return sum(1 for source, _ in self.objects if source == SOURCE_MODEL_MANUAL)

    @property
    def n_model_produced(self) -> int:
        return self.n_objects - self.n_hand_drawn

    @property
    def n_unstamped(self) -> int:
        """Model-produced objects that do not say which run made them."""
        return sum(
            1
            for source, stamp in self.objects
            if source != SOURCE_MODEL_MANUAL and stamp is None
        )

    def packs(self) -> list[str]:
        """Every model pack behind an object here, by stamp or by source model."""
        found = {
            source for source, _ in self.objects
            if source not in {SOURCE_MODEL_MANUAL, SOURCE_MODEL_UNKNOWN}
        }
        found.update(
            str(stamp["pack_id"]) for stamp in self.stamps if stamp.get("pack_id")
        )
        return sorted(found)

    def for_pack(self, pack_id: str) -> list[dict[str, Any]]:
        return [
            stamp
            for source, stamp in self.objects
            if stamp is not None and str(stamp.get("pack_id") or source) == pack_id
        ]

    def n_for_pack(self, pack_id: str) -> int:
        """Confirmed objects attributed to one pack, stamped or not."""
        return sum(1 for source, _ in self.objects if source == pack_id)


def run_stamps(segmentation: ImageSegmentation) -> RunStamps:
    """Read the run stamp off every confirmed object of a segmentation."""
    rows = confirmed_objects(segmentation).values_list("features", "source_model")
    objects: list[tuple[str, dict[str, Any] | None]] = []
    hand_drawn_areas: list[float] = []
    for features, source in rows:
        normalised = normalize_source_model(source) or SOURCE_MODEL_UNKNOWN
        objects.append((normalised, read_run_identity(features)))
        if normalised != SOURCE_MODEL_MANUAL:
            continue
        area = (features or {}).get("area")
        if isinstance(area, (int, float)) and not isinstance(area, bool):
            hand_drawn_areas.append(float(area))
    return RunStamps(
        objects=tuple(objects), hand_drawn_areas=tuple(hand_drawn_areas)
    )


def _tally(stamps: list[dict[str, Any]], field_name: str) -> dict[Any, int]:
    """Recorded values of one stamp field and how many objects carry each.

    Missing and ``None`` are folded together into a single ``None`` key, because
    the contract uses ``None`` as a real value ("no adapter", "native scale")
    and a stamp that omits the key says the same thing.
    """
    counts: dict[Any, int] = {}
    for stamp in stamps:
        value = stamp.get(field_name)
        key = round(value, 12) if isinstance(value, float) else value
        try:
            counts[key] = counts.get(key, 0) + 1
        except TypeError:
            # A stamp is JSON out of the database and can hold anything. An
            # unhashable value is a broken stamp, not a reason to fail a run;
            # it is recorded as its own text so the reader sees the damage.
            key = repr(value)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """``"1 object"`` / ``"85 objects"``.

    Worth the three lines: every string these sections build is shown to a user
    in a yellow box, and "none of its 1 model-produced objects" reads as a bug
    in the sentence rather than a fact about their data.
    """
    if count == 1:
        return f"1 {singular}"
    return f"{count} {plural or singular + 's'}"


def _spell_tally(counts: dict[Any, int]) -> str:
    """``"85 at 0.5, 19 at 0.45"`` -- the sentence a disagreement is reported in."""
    return ", ".join(
        f"{n} at {'native scale' if value is None else value!r}"
        for value, n in sorted(counts.items(), key=lambda kv: (-kv[1], repr(kv[0])))
    )


@dataclass
class _RunReport:
    """One compartment's run provenance, and the caveats building it raised."""

    segmentation: ImageSegmentation
    compartment: str
    stamps: RunStamps
    caveats: list[str] = field(default_factory=list)

    @property
    def where(self) -> str:
        return f"compartment {self.compartment!r}" if self.compartment else "this segmentation"

    def caveat(self, text: str) -> None:
        if text not in self.caveats:
            self.caveats.append(text)


def run_provenance(
    segmentation: ImageSegmentation, *, compartment: str = ""
) -> dict[str, Any]:
    """The settings a segmentation's objects were *actually* produced under.

    Read from the objects themselves. Inference stamps the run that made each
    object into its ``features`` under :data:`RUN_STAMP_KEY`, so the threshold,
    adapter, scale and minimum area reported here are the ones that ran, not the
    ones configured now. The two are routinely different: proofread, fine-tune,
    apply, analyse is the obvious order to work in, and it leaves an adapter
    applied to a segmentation whose objects predate it.

    Where a stamp is missing -- objects made before QuantEM recorded them -- the
    section falls back to the current configuration, says so in every field it
    filled that way, and raises a caveat that :func:`load_inputs` lifts to the
    result's top-level ``caveats`` list. A qualification three levels down in
    the JSON is not a qualification.
    """
    report = _RunReport(
        segmentation=segmentation,
        compartment=compartment,
        stamps=run_stamps(segmentation),
    )
    unavailable: dict[str, str] = {}
    adapter_now = _adapter_provenance(segmentation, unavailable)

    values: dict[str, Any] = {
        "recorded_from": _recorded_from(report),
        "runs": _runs_seen(report),
        "adapter": _adapter_section(report, adapter_now),
        "adapter_applied_now": adapter_now,
        "foreground_threshold": _threshold_provenance(report, adapter_now),
        "min_area": _min_area_provenance(report),
        "instance_params": _instance_params(segmentation, unavailable),
        "scale": _run_scale(report),
        "inference_device": _device_provenance(report),
        "caveats": report.caveats,
    }
    return provenance.section(values, unavailable)


#: The stamp field a device would live in if inference recorded one. Read
#: forward-compatibly rather than assumed absent, so the day
#: :mod:`quantem.segmentation.run_identity` adds it this manifest reports it
#: without another change here.
DEVICE_STAMP_FIELD = "device"


def _device_provenance(report: _RunReport) -> dict[str, Any]:
    """What hardware the runs behind these objects used, if anything recorded it.

    ``cuda`` and ``cpu`` do not always agree to the last bit, so the device is a
    reproducibility variable like the library versions. The manifest's
    ``environment`` block can only report the machine that wrote the *bundle* --
    the analysis job routinely runs in another process, and on a shared install
    another box -- so the honest place for it is the run stamp, beside the
    threshold and the scale that run used.
    """
    tally = _tally(
        [s for s in report.stamps.stamps if s.get(DEVICE_STAMP_FIELD)],
        DEVICE_STAMP_FIELD,
    )
    if len(tally) == 1:
        value, n = next(iter(tally.items()))
        return {
            "value": value,
            "recorded_from": "the objects",
            "n_objects": n,
        }
    if len(tally) > 1:
        return provenance.section(
            {
                "values": sorted(tally, key=repr),
                "n_objects_by_value": {str(v): n for v, n in sorted(tally.items(), key=repr)},
                "recorded_from": "the objects",
            },
            {
                "value": (
                    f"These objects were produced on more than one device "
                    f"({_spell_tally(tally)}); a single value would be false for "
                    "some of them."
                )
            },
        )
    return provenance.section(
        {"recorded_from": "the objects"},
        {
            "value": (
                "The device an inference run used is not recorded anywhere "
                "QuantEM can read back. Every object produced by a model carries "
                "the run that made it (quantem.segmentation.run_identity), and "
                f"that contract's field list has no {DEVICE_STAMP_FIELD!r} entry, "
                "so there is nothing to read even for a fully stamped object "
                "set. It is read here the moment one is added. Until then, "
                "reproducing these numbers on different hardware may not give "
                "the last decimal place; environment.torch_devices_available "
                "records only what the machine that wrote this bundle offers, "
                "which is not necessarily the machine that ran the inference."
            )
        },
    )


def _recorded_from(report: _RunReport) -> str:
    """One sentence saying whether the numbers below are evidence or configuration."""
    stamps = report.stamps
    n = stamps.n_model_produced
    if n == 0:
        return (
            "Nothing here was produced by a model: there is no run to describe. "
            "Any value below is the segmentation's current configuration, "
            "recorded because the interface offers it."
        )
    if stamps.n_unstamped == 0:
        return (
            f"The {_plural(n, 'model-produced object')} "
            f"{'itself' if n == 1 else 'themselves'}. Each carries the identity "
            "and settings of the inference run that created it, so the values "
            "below are what ran — not what the segmentation is configured for "
            "now."
        )
    if not stamps.stamps:
        report.caveat(
            f"The inference settings recorded for {report.where} are the "
            f"segmentation's current ones, not the run's: its "
            f"{_plural(n, 'model-produced object')} "
            f"{'carries' if n == 1 else 'carry'} no record of the run that made "
            f"{'it' if n == 1 else 'them'}. If the threshold, adapter, pixel "
            "size or minimum area was changed after those objects were produced "
            "— applying an adapter after proofreading does exactly this — the "
            "manifest shows the later value. Re-run inference to get a "
            "self-describing object set."
        )
        return (
            f"The current configuration of {report.where}: its "
            f"{_plural(n, 'model-produced object')} "
            f"{'does' if n == 1 else 'do'} not record the run that made "
            f"{'it' if n == 1 else 'them'}. Those objects predate per-run "
            "stamping; a threshold, adapter, scale or minimum area changed since "
            "they were produced appears below in place of the one that ran."
        )
    report.caveat(
        f"{stamps.n_unstamped} of the {n} model-produced objects in "
        f"{report.where} carry no record of the run that made them, so the "
        "settings reported for them are the segmentation's current ones rather "
        f"than the run's. The other {len(stamps.stamps)} are self-describing."
    )
    return (
        f"A mixture. {len(stamps.stamps)} of the {n} model-produced objects "
        "record the run that made them; the remaining "
        f"{stamps.n_unstamped} predate per-run stamping and are described by the "
        "segmentation's current configuration."
    )


def _runs_seen(report: _RunReport) -> list[dict[str, Any]]:
    """Each distinct inference run behind these objects, newest last.

    A compartment whose objects came from two runs is the case the whole
    stamping contract exists for: a manifest that reports one threshold for it
    is wrong about at least one object.

    Each run carries the adapter *it* used, expanded in place. The id alone was
    a foreign key into a table the reader does not have; the base model, mode,
    steps, split mode, held-out Dice and head digest were in the manifest but
    only under ``adapter_applied_now``, which describes the adapter applied to
    the segmentation today and need not be the one that made these objects. The
    lookup is memoised because runs share adapters far more often than not.
    """
    by_id: dict[str, dict[str, Any]] = {}
    adapters: dict[str, dict[str, Any]] = {}
    for stamp in report.stamps.stamps:
        run_id = str(stamp.get("id") or "")
        adapter_id = stamp.get("adapter_id")
        entry = by_id.setdefault(
            run_id,
            {
                "run_id": run_id or None,
                "finished_at": stamp.get("finished_at"),
                "pack_id": stamp.get("pack_id"),
                "threshold": stamp.get("threshold"),
                "adapter_id": adapter_id,
                "adapter": (
                    adapters.setdefault(str(adapter_id), _adapter_details(str(adapter_id)))
                    if adapter_id
                    else None
                ),
                "ran_at_nm": stamp.get("ran_at_nm"),
                "min_area": stamp.get("min_area"),
                "n_objects": 0,
            },
        )
        entry["n_objects"] += 1
    runs = sorted(by_id.values(), key=lambda r: (str(r["finished_at"] or ""), str(r["run_id"] or "")))
    if len(runs) > 1:
        report.caveat(
            f"The objects in {report.where} come from {len(runs)} different "
            "inference runs, so no single threshold, adapter or scale describes "
            "all of them. Each run is listed separately in the manifest with the "
            "number of objects it produced."
        )
    return runs


def _adapter_section(
    report: _RunReport, adapter_now: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Which adapter, if any, was applied to the runs that made these objects.

    ``adapter_applied_now`` beside this records what is applied to the
    segmentation today. When they differ the objects win, and the difference is
    a caveat: an adapter applied after inference has calibrated nothing that is
    in this bundle.
    """
    stamps = report.stamps
    if not stamps.stamps:
        if adapter_now is None:
            return None
        out = dict(adapter_now)
        out["recorded_from"] = (
            "The segmentation's currently applied adapter, not the objects. "
            "None of them records the run that made it."
        )
        return out

    tally = _tally(stamps.stamps, "adapter_id")
    ids = [value for value in tally if value is not None]
    if not ids:
        if (adapter_now or {}).get("applied"):
            report.caveat(
                f"An adapter ({adapter_now.get('adapter_id')}) is applied to "
                f"{report.where} now, but none of its "
                f"{_plural(len(stamps.stamps), 'model-produced object')} was "
                "made under it: they were produced by the released model at its "
                "own threshold. "
                "Applying an adapter does not re-infer anything, so this bundle "
                "reports the threshold that ran, not the calibrated one. Re-run "
                "inference if you meant the adapter to apply to these objects."
            )
        return {
            "applied": False,
            "recorded_from": "the objects",
            "note": (
                f"None of the {_plural(len(stamps.stamps), 'stamped object')} "
                "here was produced under an adapter: the released model ran at "
                "its own threshold."
            ),
        }
    if len(tally) == 1:
        adapter_id = str(ids[0])
        out: dict[str, Any] = {
            "applied": True,
            "adapter_id": adapter_id,
            "recorded_from": "the objects",
            "n_objects": tally[ids[0]],
        }
        out.update(_adapter_details(adapter_id))
        if (adapter_now or {}).get("applied") and str(
            adapter_now.get("adapter_id")
        ) != adapter_id:
            report.caveat(
                f"The adapter applied to {report.where} now "
                f"({adapter_now.get('adapter_id')}) is not the one its objects "
                f"were produced under ({adapter_id}). The objects were not "
                "re-inferred; the bundle reports the adapter that ran."
            )
        return out

    spelled = ", ".join(
        f"{_plural(n, 'object')} under {'no adapter' if value is None else value}"
        for value, n in sorted(tally.items(), key=lambda kv: (-kv[1], repr(kv[0])))
    )
    report.caveat(
        f"The objects in {report.where} were not all produced under the same "
        f"adapter ({spelled}), so no single calibration describes this "
        "compartment."
    )
    return provenance.section(
        {
            "adapter_ids": sorted(str(value) for value in ids),
            "n_objects_by_adapter": {
                ("none" if value is None else str(value)): n
                for value, n in sorted(tally.items(), key=lambda kv: repr(kv[0]))
            },
            "recorded_from": "the objects",
        },
        {
            "applied": (
                "These objects were produced under more than one adapter "
                f"({spelled}); a single yes/no would be false for some of them."
            )
        },
    )


def adapter_facts(adapter: Any) -> dict[str, Any]:
    """Everything an adapter is, from the adapter row itself.

    One builder for every place an adapter appears in the manifest -- the run
    that used it, the compartment's ``adapter`` section, and
    ``adapter_applied_now`` -- because they were three partial lists and the run
    had the shortest. A reader who wanted to know what ``runs[1].adapter_id``
    *was* had to hope the same adapter was still the applied one.

    The head is identified by digest, not by where it sits: an adapter's weights
    are a file on the user's disk, they are as much a part of what produced the
    objects as the released pack's are, and ``head_path`` was the only mention of
    them in the bundle.
    """
    steps = None
    params = getattr(adapter, "params", None) or {}
    if isinstance(params, dict):
        steps = params.get("steps")

    facts: dict[str, Any] = {
        "adapter_id": str(adapter.id),
        "name": adapter.name,
        "base_model": adapter.base_model,
        "mode": adapter.mode,
        "status": adapter.status,
        "steps": steps,
        "calibrated_threshold": adapter.calibrated_threshold,
        "split_mode": adapter.split_mode,
        "heldout_dice": adapter.heldout_dice,
        "trainable_params": adapter.trainable_params,
        "verified_reload": adapter.verified_reload,
        "applied_at": adapter.applied_at.isoformat() if adapter.applied_at else None,
        "caveats": list(adapter.caveats()),
    }
    unavailable: dict[str, str] = {}
    if steps is None:
        unavailable["steps"] = (
            "This adapter's stored hyper-parameters record no step count. A "
            "threshold-only adapter fits no weights and takes none."
        )
    if adapter.trainable_params is None:
        unavailable["trainable_params"] = (
            f"This adapter's record does not say how many parameters it fitted. "
            f"A {adapter.mode!r} adapter that fits only a threshold has none to "
            "count; a head adapter records the count when it trains, so a null "
            "here means the fit predates that or did not finish."
        )
    if adapter.heldout_dice is None:
        unavailable["heldout_dice"] = (
            f"This adapter was fitted with split_mode={adapter.split_mode!r}, "
            "which leaves no held-out crops, so there is no held-out score. The "
            "threshold was chosen on every region that was annotated."
        )
    head_file = adapter.head_file
    if head_file is None:
        unavailable["head"] = (
            f"This is a {adapter.mode!r} adapter: it fits a threshold and no "
            "weights, so there is no head file to checksum. The calibrated "
            "threshold above is the whole of what it changed."
        )
    else:
        facts["head"] = provenance.file_identity(head_file, what="the adapter head")
    return provenance.section(facts, unavailable)


def _adapter_details(adapter_id: str) -> dict[str, Any]:
    """:func:`adapter_facts` for a stamped adapter id, if the record survives."""
    try:
        from quantem.finetune.models import Adapter

        adapter = Adapter.objects.filter(id=adapter_id).first()
    except Exception as exc:
        return {
            "unavailable": {
                "name": (
                    f"The adapter record {adapter_id} could not be read "
                    f"({exc.__class__.__name__}: {exc}); only the id the objects "
                    "carry is known."
                )
            },
            "name": None,
        }
    if adapter is None:
        return {
            "name": None,
            "unavailable": {
                "name": (
                    f"Adapter {adapter_id} produced these objects but no longer "
                    "exists in this database; only the id survives, on the "
                    "objects themselves."
                )
            },
        }
    return adapter_facts(adapter)


def _threshold_provenance(
    report: _RunReport, adapter_now: dict[str, Any] | None
) -> dict[str, Any]:
    """The probability above which a pixel was foreground, and where it came from.

    This is the number that decides the object set, so it is recorded per pack
    and with its origin: an adapter's calibrated threshold and a pack's
    published default are not interchangeable, and a bundle that shows only one
    number cannot be compared with a bundle from the released model.
    """
    per_pack: dict[str, Any] = {
        pack_id: _threshold_for_pack(report, pack_id, adapter_now)
        for pack_id in report.stamps.packs()
    }
    return {
        "by_pack": per_pack,
        "note": (
            "The foreground probability threshold each model ran at. It is "
            "applied on the model's own resampled grid; only the resulting "
            "binary mask is brought back to native pixels."
        ),
    }


def _threshold_for_pack(
    report: _RunReport, pack_id: str, adapter_now: dict[str, Any] | None
) -> dict[str, Any]:
    default = _default_threshold(pack_id)
    tally = _tally(
        [s for s in report.stamps.for_pack(pack_id) if s.get("threshold") is not None],
        "threshold",
    )
    if len(tally) == 1:
        value, n = next(iter(tally.items()))
        entry: dict[str, Any] = {
            "value": value,
            "source": (
                f"recorded on the {_plural(n, 'object')} {pack_id} produced "
                "here, by the run that produced it"
                if n == 1
                else f"recorded on the {n} objects {pack_id} produced here, by "
                "the run that produced them"
            ),
            "recorded_from": "the objects",
            "pack_default": default,
            "n_objects": n,
        }
        # The case that sent a wrong threshold to a manifest: an adapter applied
        # after inference. Its calibration is real, but it did not produce these
        # objects, so it is recorded beside the value rather than as it.
        calibrated_now = (
            (adapter_now or {}).get("calibrated_threshold")
            if (adapter_now or {}).get("applied")
            and (adapter_now or {}).get("base_model") == pack_id
            else None
        )
        if calibrated_now is not None and calibrated_now != value:
            entry["superseded_for_future_runs"] = {
                "adapter_id": adapter_now.get("adapter_id"),
                "calibrated_threshold": calibrated_now,
                "note": (
                    "This adapter is applied to the segmentation now and its "
                    "threshold will be used by the next run. It did not produce "
                    "the objects in this bundle."
                ),
            }
        return entry
    if len(tally) > 1:
        spelled = _spell_tally(tally)
        report.caveat(
            f"The {pack_id} objects in {report.where} were not all produced at "
            f"the same foreground threshold ({spelled}). Object count and object "
            "shape both depend on it, so this compartment is not a single "
            "measurement; re-run inference over the whole compartment before "
            "reporting it."
        )
        return provenance.section(
            {
                "values": sorted(tally, key=repr),
                "n_objects_by_value": {
                    str(v): n for v, n in sorted(tally.items(), key=repr)
                },
                "source": f"recorded on the objects, which disagree: {spelled}",
                "recorded_from": "the objects",
                "pack_default": default,
            },
            {
                "value": (
                    f"The objects {pack_id} produced here were made at more than "
                    f"one threshold ({spelled}); reporting one of them as the "
                    "run's threshold would be false for the rest."
                )
            },
        )

    # Nothing recorded: fall back to the configuration as it stands now.
    calibrated = (
        (adapter_now or {}).get("calibrated_threshold")
        if (adapter_now or {}).get("applied")
        else None
    )
    if calibrated is not None and (adapter_now or {}).get("base_model") == pack_id:
        return {
            "value": calibrated,
            "source": f"calibrated by adapter {adapter_now['adapter_id']}",
            "recorded_from": "the segmentation's current configuration",
            "pack_default": default,
            "n_objects": report.stamps.n_for_pack(pack_id),
            "caveat": (
                "The objects do not record what they were produced at; this is "
                "the adapter applied to the segmentation now. If it was applied "
                "after the objects were made, it did not produce them."
            ),
        }
    return {
        "value": default,
        "source": "the model pack's published default",
        "recorded_from": "the segmentation's current configuration",
        "n_objects": report.stamps.n_for_pack(pack_id),
        "caveat": (
            "The objects do not record what they were produced at; this is the "
            "pack's published default, which is what an un-adapted run uses."
        ),
    }


def _default_threshold(pack_id: str) -> float | None:
    try:
        from quantem.inference.specs import get_model_spec, parse_family

        organelle = pack_id.split(":", 1)[1] if ":" in pack_id else ""
        return get_model_spec(parse_family(pack_id), organelle).threshold
    except Exception:
        return None


def _default_min_area(pack_id: str) -> int | None:
    try:
        from quantem.inference.specs import get_model_spec, parse_family

        organelle = pack_id.split(":", 1)[1] if ":" in pack_id else ""
        spec = get_model_spec(parse_family(pack_id), organelle)
        return int(spec.organelle_spec.default_min_area)
    except Exception:
        return None


def _run_native_pixel_size(report: _RunReport) -> float | None:
    """The pixel size the image had when these objects were produced.

    ``None`` when at least one object was produced uncalibrated -- which is not
    the same as the image having no pixel size now, and is the distinction the
    whole "calibrated after the fact" guard turns on. Falls back to the asset's
    current value only when nothing is stamped, where there is nothing better.
    """
    tallied = _tally(report.stamps.stamps, "native_pixel_size_nm")
    if not tallied:
        return pixel_size_nm(report.segmentation)
    if None in tallied:
        return None
    recorded = sorted(value for value in tallied if value is not None)
    return recorded[0] if len(recorded) == 1 else None


def _min_area_provenance(report: _RunReport) -> dict[str, Any]:
    """The smallest object each run kept, in native pixels.

    Recorded because it moves the object count twice over. It is a per-organelle
    constant (mito 60, ER 100, LD 40, nucleus 8000) applied *after* the mask is
    mapped back to native pixels, so its value is stable in native pixels but
    its meaning is not: on a 5 nm image a pack whose canonical scale is 8 nm saw
    that area as ``(5/8)^2`` of it. The boundary itself is QuantEM's own and
    does not move with scikit-image -- see
    :func:`quantem.inference.postprocess.filter_min_area`.
    """
    native = pixel_size_nm(report.segmentation)
    per_pack: dict[str, Any] = {}
    for pack_id in report.stamps.packs():
        default = _default_min_area(pack_id)
        tally = _tally(
            [s for s in report.stamps.for_pack(pack_id) if s.get("min_area") is not None],
            "min_area",
        )
        if len(tally) == 1:
            value, n = next(iter(tally.items()))
            entry: dict[str, Any] = {
                "value": value,
                "source": (
                    f"recorded on the {_plural(n, 'object')} {pack_id} "
                    "produced here"
                ),
                "recorded_from": "the objects",
                "organelle_default": default,
                "n_objects": n,
            }
        elif len(tally) > 1:
            spelled = _spell_tally(tally)
            report.caveat(
                f"The {pack_id} objects in {report.where} were not all filtered "
                f"at the same minimum area ({spelled}), so the object count of "
                "this compartment mixes two different size floors."
            )
            entry = provenance.section(
                {
                    "values": sorted(tally, key=repr),
                    "n_objects_by_value": {
                        str(v): n for v, n in sorted(tally.items(), key=repr)
                    },
                    "source": f"recorded on the objects, which disagree: {spelled}",
                    "recorded_from": "the objects",
                    "organelle_default": default,
                },
                {
                    "value": (
                        f"The objects {pack_id} produced here were filtered at "
                        f"more than one minimum area ({spelled})."
                    )
                },
            )
        else:
            entry = {
                "value": default,
                "source": "the organelle's default minimum area for this build",
                "recorded_from": "the segmentation's current configuration",
                "organelle_default": default,
                "n_objects": report.stamps.n_for_pack(pack_id),
                "caveat": (
                    "The objects do not record the minimum area they were "
                    "filtered at; this is the default a run of this pack uses "
                    "unless one was passed."
                ),
            }
        value = entry.get("value")
        if isinstance(value, (int, float)):
            # Both restatements need the pixel size the run *had*, not the one
            # the image has now. Deriving them from the asset's current value
            # put `um2: 0.0015` into a bundle whose every other physical unit is
            # deliberately blank -- 60 x (5/1000)^2, from a pixel size the run
            # never saw -- and `model_grid_px: 23.438` one field above a `scale`
            # block correctly reading `resampled: false, ran_at: "native"`, i.e.
            # the model actually saw 60 native pixels. A reader quoting
            # "objects below 0.0015 um2 were discarded" as the study's size floor
            # would be quoting a number that never existed.
            ran_at = _run_native_pixel_size(report)
            if ran_at is None:
                entry["um2"] = None
                entry["model_grid_px"] = None
                entry["unavailable"] = {
                    "um2": (
                        "These objects were produced while the image had no "
                        "pixel size, so the minimum area has no physical value. "
                        f"The image records {native} nm/px now, but that was set "
                        "after the run and restating the filter in microns with "
                        "it would describe a run that never happened."
                    ),
                    "model_grid_px": (
                        "No resample happened -- the run had no pixel size to "
                        "resample from -- so the model saw these native pixels."
                    ),
                }
            else:
                entry["um2"] = value * (ran_at / 1000.0) ** 2
                canonical, _known = _canonical_nm(pack_id)
                if canonical:
                    factor = ran_at / canonical
                    entry["model_grid_px"] = round(value * factor * factor, 3)
        per_pack[pack_id] = entry

    return {
        "by_pack": per_pack,
        "units": "native image pixels",
        "applied": (
            "after the thresholded mask is mapped back from the model's grid to "
            "native pixels, together with the organelle's morphological closing"
        ),
        "note": (
            "Connected components below this area were discarded by the pack "
            "that produced them and are not in any count, area fraction or "
            "density here. The value is in native pixels; um2 and "
            "model_grid_px above restate it in physical area and in the pixels "
            "the model actually saw, because a pack that resamples sees a "
            "different number of them. The comparison is QuantEM's own — a "
            "component of exactly this many pixels is kept — and does not "
            "change with the scikit-image version; see "
            "environment.skimage_note."
        ),
        # The schema already scoped the floor under by_pack; the sentence above
        # did not, and read as a statement about the bundle. It is not one.
        "not_applied_to": _min_area_bypassed(report, per_pack),
    }


def _min_area_bypassed(
    report: _RunReport, per_pack: dict[str, Any]
) -> dict[str, Any]:
    """What the size floor did **not** filter, and what that does to a histogram.

    ``filter_min_area`` runs inside inference. A polygon a person drew never
    goes through it, so a hand-drawn object of 46 px is exported from a
    compartment whose model floor is 60 px, while a model-produced component of
    the same 46 px was discarded before it could become an object at all.

    That is not a wording problem. Pooling the two provenances gives a size
    distribution that is left-truncated for one of them and not the other: the
    mean area is biased upward for the model half, the minimum comes from the
    hand-drawn half, and the shape of the lower tail is an artefact of who drew
    what. The manifest names it here and, when it actually bites, raises a
    caveat that reaches the top-level list rather than sitting four levels down
    in the JSON.
    """
    n_hand_drawn = report.stamps.n_hand_drawn
    floors = [
        entry["value"]
        for entry in per_pack.values()
        if isinstance(entry.get("value"), (int, float))
    ]
    out: dict[str, Any] = {
        "hand_drawn_objects": n_hand_drawn,
        "why": (
            "quantem.inference.postprocess.filter_min_area runs inside "
            "inference. A polygon a person drew is stored directly and is "
            "never filtered, at any size."
        ),
    }
    if not floors or not n_hand_drawn:
        return out

    smallest = min(floors)
    below = sorted(a for a in report.stamps.hand_drawn_areas if a < smallest)
    out["smallest_floor_px"] = smallest
    out["n_hand_drawn_below_the_floor"] = len(below)
    if not below:
        return out

    out["smallest_hand_drawn_px"] = below[0]
    one = len(below) == 1
    report.caveat(
        f"The minimum-area floor in {report.where} applies to model-produced "
        f"objects only: the smallest in play is {smallest:g} px, and "
        f"{len(below)} of the {_plural(n_hand_drawn, 'hand-drawn object')} "
        f"here {'is' if one else 'are'} below it (smallest {below[0]:g} px). "
        f"{'It is' if one else 'They are'} in objects.csv, in the count, in "
        "the area fraction and in the density, while a model-produced object "
        "of the same size was discarded before it ever became one — "
        "filter_min_area runs inside inference and a drawn polygon never "
        "passes through it. A size distribution pooled over both provenances "
        "is therefore left-truncated for one of them and not the other, which "
        "biases the mean area upward for the model half and puts the minimum "
        "in the hand-drawn half. Split on the source_model column of "
        "objects.csv before quoting a mean area, a minimum or a histogram."
    )
    return out


#: Why ``instance_params`` are recorded as configuration and not as evidence.
#:
#: Nothing in ``quantem.seg_core`` or ``quantem.inference`` reads them:
#: ``organelle_tasks._build_segmenter_kwargs`` passes ``instance_params=`` to
#: the segmenter and ``DinoOrganelleSegmenter.__init__`` absorbs it into
#: ``**_ignored``. The threshold that actually ran is in
#: ``foreground_threshold`` above. Writing these into a manifest as "the
#: parameters used" would be the same class of untruth as a fabricated zero.
_INSTANCE_PARAMS_NOTE = (
    "The segmentation's configured instance-extraction settings, recorded "
    "because the interface offers them — not as evidence that this run used "
    "them. The DINO organelle segmenters take their foreground threshold from "
    "the model pack or from an applied adapter (see foreground_threshold) and "
    "do not read segmentation_threshold, center_min_distance, "
    "center_confidence_threshold or downsampling_factor."
)


def _instance_params(
    segmentation: ImageSegmentation, unavailable: dict[str, str]
) -> dict[str, Any] | None:
    """``segmentation_threshold`` and friends, as configured for this segmentation."""
    internal = segmentation.segmentation_type.internal_name
    if not supports_instance_params(internal):
        unavailable["instance_params"] = (
            f"Segmentation type {internal!r} takes no instance-extraction "
            "parameters; its objects come straight from the thresholded "
            "probability map."
        )
        return None
    config = getattr(segmentation, "config", None)
    if config is None:
        from quantem.segmentation.instance_params import instance_params_defaults

        params: dict[str, Any] = dict(instance_params_defaults())
        params["_configured"] = False
    else:
        params = {key: config.get_instance_params().get(key) for key in INSTANCE_PARAM_KEYS}
        params["_configured"] = True
    params["_note"] = _INSTANCE_PARAMS_NOTE
    return params


#: ``ran_at`` when no inference is recorded for a compartment at all. Not
#: ``"native"``: a pack that declares a canonical pixel size resamples, and
#: saying "native" about a run nobody recorded is the guess this module exists
#: to refuse.
SCALE_UNKNOWN = "unknown"


def _run_scale(report: _RunReport) -> dict[str, Any]:
    """The pixel size each model actually ran at, and whether it resampled.

    Six of the eight released packs declare a ``canonical_nm`` and resample the
    image to it before inference. A 5 nm/px image analysed by a pack whose
    canonical scale is 8.0 nm is seen by the model at 8 nm; an *uncalibrated*
    image cannot be resampled and is seen at whatever it happens to be. Those
    are different runs and they give different object counts.

    Read from the objects' stamps when they have them. Without stamps this can
    only be inferred from which packs produced the confirmed objects -- and a
    compartment with *no* model-produced objects supports no inference at all,
    so it reports :data:`SCALE_UNKNOWN` with the reason rather than defaulting
    to "native".
    """
    segmentation = report.segmentation
    stamps = report.stamps
    native = pixel_size_nm(segmentation)
    packs = stamps.packs()

    values: dict[str, Any] = {"native_pixel_size_nm": native}
    # One field name, two questions. On an object's run stamp
    # ``native_pixel_size_nm`` is the pixel size the image had *when that run
    # happened*; here it is what the image carries *now*, read off the asset.
    # They differ on exactly the runs this section exists to describe, and the
    # disclaimer used to appear in one branch of the note only
    # (:func:`_native_scale_note`), so a mixed-scale or resampled block asserted
    # the ambiguous name with nothing attached.
    values["native_pixel_size_nm_is"] = (
        "the image's pixel size as it is now, read from the image and not from "
        "these objects. The field of the same name on an object's run stamp is "
        "the pixel size the image had when that run happened; the two differ "
        "whenever an image was calibrated after its objects were produced."
    )
    unavailable: dict[str, str] = {}

    canonical: dict[str, Any] = {}
    for pack_id in packs:
        nm, known = _canonical_nm(pack_id)
        canonical[pack_id] = nm
        if not known:
            unavailable[f"canonical_nm_by_pack.{pack_id}"] = (
                f"{pack_id!r} is not one of the released packs known to this "
                "build, so the scale it declares — and therefore whether it "
                "resampled — cannot be looked up."
            )
    values["canonical_nm_by_pack"] = canonical

    _check_recalibration(report, native)

    ran = _tally(stamps.stamps, "ran_at_nm")
    if ran:
        values["recorded_from"] = "the objects"
        values["ran_at_nm_by_pack"] = {
            pack_id: sorted(
                _tally(stamps.for_pack(pack_id), "ran_at_nm"), key=repr
            )
            for pack_id in packs
        }
        if len(ran) == 1:
            only = next(iter(ran))
            values["ran_at_nm"] = only
            values["ran_at"] = "native" if only is None else "canonical"
            values["resampled"] = only is not None
            values["note"] = (
                _native_scale_note(canonical, native, report)
                if only is None
                else (
                    f"The image was resampled to {only} nm/px before inference "
                    "and the resulting masks were mapped back to native pixels "
                    "with nearest-neighbour interpolation, so every measurement "
                    "in this bundle is in native pixels."
                )
            )
        else:
            spelled = _spell_tally(ran)
            report.caveat(
                f"The objects in {report.where} were not all produced at the "
                f"same scale ({spelled}). A resampled run and a native one "
                "resolve different structures, so the object set is a mixture."
            )
            values["ran_at"] = "mixed"
            values["resampled"] = any(v is not None for v in ran)
            values["note"] = f"More than one scale ran here: {spelled}."
            unavailable["ran_at_nm"] = (
                f"These objects were produced at more than one scale ({spelled}); "
                "a single value would be false for some of them."
            )
        return provenance.section(values, unavailable)

    # --- No stamps: say what can be said, and refuse to guess the rest. ---
    values["recorded_from"] = "the packs that produced the confirmed objects"
    if not packs:
        why = (
            "it has no confirmed objects at all"
            if stamps.n_objects == 0
            else (
                f"all {stamps.n_hand_drawn} of its confirmed objects were drawn "
                "by hand, so no model output survives in it"
            )
        )
        reason = (
            f"No confirmed object in {report.where} was produced by a model: "
            f"{why}. Nothing here records whether a model resampled the image, "
            "or even whether inference ran, so the scale is unknown. It is not "
            "'native' — six of the eight released packs declare a canonical "
            "pixel size and resample to it, and a run at 8 nm on a 5 nm image "
            "resolves different objects than a native one."
        )
        report.caveat(
            f"The scale inference ran at in {report.where} is unknown, not "
            f"native: {why}. The manifest cannot say whether the image was "
            "resampled before inference, and this compartment must not be read "
            "as having run at the image's own resolution."
        )
        values["ran_at"] = SCALE_UNKNOWN
        for key in ("ran_at_nm", "resampled"):
            unavailable[key] = reason
        return provenance.section(values, unavailable)

    if native is None:
        values["ran_at"] = "native"
        values["ran_at_nm"] = None
        values["resampled"] = False
        values["note"] = (
            "The image has no recorded pixel size, so no pack could resample to "
            "its canonical scale: every model ran at the image's native "
            "resolution, whatever that is. This is a caveat on the object set, "
            "not only on the unit conversion."
        )
        return provenance.section(values, unavailable)

    unknown_packs = sorted(
        pack_id for pack_id in packs if not _canonical_nm(pack_id)[1]
    )
    resampling = {
        pack_id: nm
        for pack_id, nm in canonical.items()
        if nm and abs(nm - native) > 1e-9
    }
    values["resampled"] = bool(resampling)
    if resampling:
        values["ran_at"] = "canonical"
        values["ran_at_nm"] = resampling
        values["note"] = (
            "The image was resampled from its native pixel size to each pack's "
            "canonical scale before inference; the resulting masks were mapped "
            "back to native pixels with nearest-neighbour interpolation, so all "
            "measurements in this bundle are in native pixels."
        )
    elif unknown_packs:
        values["ran_at"] = SCALE_UNKNOWN
        reason = (
            f"{', '.join(unknown_packs)} produced objects here but is not a pack "
            "this build knows, so whether it resampled the image cannot be "
            "determined."
        )
        unavailable["ran_at_nm"] = reason
        unavailable["resampled"] = reason
        values.pop("resampled", None)
        report.caveat(
            f"The scale inference ran at in {report.where} is unknown: " + reason
        )
    else:
        values["ran_at"] = "native"
        values["ran_at_nm"] = native
        values["note"] = (
            "No pack used here declares a canonical pixel size different from "
            "the image's own, so inference ran at native resolution. Derived "
            "from which packs produced the confirmed objects, not from a record "
            "the runs left behind."
        )
    return provenance.section(values, unavailable)


def _native_scale_note(
    canonical: dict[str, Any], native: float | None, report: _RunReport
) -> str:
    """Why every object here ran at native scale -- benign, or a missed resample.

    ``ran_at_nm is None`` on every stamp has two very different causes, and the
    note used to state only the harmless one ("no pack that ran resampled it").
    A pack that declares a ``canonical_nm`` *would* have resampled, and did not
    only because the image had no pixel size at the time. Reporting that as a
    design fact -- beside a ``native_pixel_size_nm`` read from the asset's
    current value -- reads as reassurance in precisely the case that needs a
    warning.
    """
    wanted = {
        pack_id: nm
        for pack_id, nm in sorted(canonical.items())
        if isinstance(nm, int | float) and nm
    }
    if not wanted:
        return (
            "Every stamped object here was produced at the image's native "
            "resolution, and no pack that ran declares a canonical pixel size, "
            "so nothing was skipped."
        )
    spelled = ", ".join(f"{pack_id} at {nm} nm/px" for pack_id, nm in wanted.items())
    return (
        "Every stamped object here was produced at the image's native resolution "
        f"— but not because no resample was called for: {spelled} would have "
        "been resampled to its own scale had a pixel size been available when "
        "the run happened. It was not, so the model saw the raw pixels. "
        f"native_pixel_size_nm ({native}) is the image's value now, not the value "
        "these objects were produced under."
    )


def _packs_that_skipped_a_resample(report: _RunReport) -> list[str]:
    """Packs whose objects here were produced without the resample they need.

    A pack that declares a ``canonical_nm`` resamples the image to it before
    inference and cannot do so without a pixel size, so an uncalibrated run of
    one produced a different object set than a calibrated run would have.

    A pack that declares none is a different case entirely, and the manifest
    already says so where it can (``_native_scale_note``: "nothing was
    skipped"). ``quantem.inference.resample.resample_factor`` returns 1.0 when
    *either* the pixel size or the canonical scale is missing, and ``min_area``
    and ``close_radius`` are in native pixels either way -- so calibrating
    afterwards changes nothing about what that pack produced, and "re-run
    inference before reporting any count" is advice that would return the same
    objects.

    A pack this build does not recognise counts as having skipped one: what it
    would have done cannot be looked up, and an unlookupable scale is a reason
    to warn, not a reason to stay quiet.
    """
    skipped: list[str] = []
    for pack_id in report.stamps.packs():
        stamps = report.stamps.for_pack(pack_id)
        if not any(stamp.get("native_pixel_size_nm") is None for stamp in stamps):
            continue
        canonical, known = _canonical_nm(pack_id)
        if canonical or not known:
            skipped.append(pack_id)
    return sorted(skipped)


def _check_recalibration(report: _RunReport, native: float | None) -> None:
    """Warn when the image was recalibrated after these objects were produced."""
    tallied = _tally(report.stamps.stamps, "native_pixel_size_nm")
    recorded = {value for value in tallied if value is not None}
    if not recorded:
        # Uncalibrated -> calibrated: the one transition this check used to miss,
        # because filtering None out of `recorded` left nothing to compare and it
        # returned. It is also the transition the app recommends ("Set the
        # image's pixel size and re-run inference"), so a user who does the first
        # half and not the second lands exactly here -- and every other guard,
        # keyed on the image's current value, quietly starts reporting microns
        # over an object set produced without one.
        skipped = _packs_that_skipped_a_resample(report)
        if tallied and None in tallied and native is not None and skipped:
            report.caveat(
                f"The objects in {report.where} were produced while this image "
                f"had no pixel size; it is {native} nm/px now. "
                f"{', '.join(skipped)} would have resampled the image to its own "
                "scale and could not, so the object set is the one that run "
                "produced and setting a pixel size afterwards does not re-run "
                "inference. Every micron value, density and distance in this "
                "bundle is blank for that reason. Re-run inference before "
                "reporting any count, area fraction, density or distance from "
                "it — after discarding the objects it produced, which a re-run "
                "on its own will not replace because they are confirmed. The "
                "wrong-scale caveat on this run gives the route."
            )
        return
    if native is None:
        report.caveat(
            f"The objects in {report.where} were produced when this image had a "
            f"pixel size of {sorted(recorded)} nm/px; it has none now, so nothing "
            "in this bundle is in physical units even though the run that made "
            "the objects was calibrated."
        )
        return
    if any(abs(value - native) > 1e-9 for value in recorded):
        report.caveat(
            f"This image's pixel size has changed since the objects in "
            f"{report.where} were produced: they ran at {sorted(recorded)} nm/px "
            f"and it is {native} nm/px now. Every micron value in this bundle "
            "uses the current value, and the resample the model applied used the "
            "old one."
        )


def _canonical_nm(pack_id: str) -> tuple[float | None, bool]:
    """``(canonical_nm, is_a_pack_this_build_knows)``.

    The two nulls are different: a released pack may declare no canonical scale
    and genuinely run native, while an unrecognised pack tells us nothing. The
    caller has to be able to tell them apart, so the lookup reports which it is
    instead of collapsing both to ``None``.
    """
    try:
        from quantem.inference.specs import get_model_spec, parse_family

        organelle = pack_id.split(":", 1)[1] if ":" in pack_id else ""
        return get_model_spec(parse_family(pack_id), organelle).canonical_nm, True
    except Exception:
        return None, False


def _adapter_provenance(
    segmentation: ImageSegmentation, unavailable: dict[str, str]
) -> dict[str, Any] | None:
    """The guided-fine-tuning adapter applied to this segmentation, if any.

    An adapter replaces the published threshold with one calibrated on the
    user's own crops and may swap the head. A bundle that does not say so cannot
    be compared with one from the released model.
    """
    try:
        from quantem.finetune.models import active_adapter_for
    except Exception:
        unavailable["adapter"] = (
            "Guided fine-tuning is not installed in this build, so no adapter "
            "could have been applied."
        )
        return None
    try:
        adapter = active_adapter_for(segmentation)
    except Exception as exc:
        unavailable["adapter"] = (
            "Whether an adapter was applied could not be determined "
            f"({exc.__class__.__name__}: {exc}); treat the thresholds above as "
            "possibly superseded."
        )
        return None
    if adapter is None:
        return {
            "applied": False,
            "note": "No adapter is applied; the released model ran at its own threshold.",
        }
    out = {"applied": True, **adapter_facts(adapter)}
    out["note"] = (
        "This adapter's calibrated threshold replaced the pack's published "
        "default for every run made while it was applied. Whether it is the one "
        "that produced the objects in this bundle is answered by runs[].adapter "
        "above, which is read off the objects."
    )
    return out


def _what_rests_on_the_pixel_size(predates_the_objects: bool) -> str:
    """What the image's current pixel size does, or does not, hold up.

    The third of the "calibrated after the fact" leaks, and the same shape as
    the other two: prose keyed on the asset's *current* value describing what
    the *run* did. ``pixel_size_provenance`` said "Every micron column in this
    bundle, and the scale any model resampled to, rests on it" beside a bundle
    whose micron columns are all blank and whose models resampled to nothing,
    and told a reader who had overridden the file's value that "the
    measurements below were made at 5.0" when nothing below was made at
    anything. Both sentences are true of the ordinary run and false of exactly
    the run that needs a warning.
    """
    if predates_the_objects:
        return (
            " The objects in this bundle were produced before this value was "
            "set, so nothing here rests on it: no model resampled to it, every "
            "micron column is blank and calibrated is false. It is recorded "
            "because it is what the image says now, not because it is what "
            "these measurements are in."
        )
    return (
        " Every micron column in this bundle, and the scale any model resampled "
        "to, rests on it. The object count depends on the scale a model "
        "resampled to, so a wrong value here is not only a units error."
    )


def pixel_size_provenance(
    asset, *, produced_pixel_size_nm: frozenset[float | None] = frozenset()
) -> dict[str, Any]:
    """Who supplied the pixel size: the file, or a person.

    The manifest recorded ``pixel_size_nm: 10.0`` and nothing about where the
    10.0 came from. That is not a detail. A pixel size gates per-organelle
    resampling before inference, so the *object count* moves with it -- on one
    image the same pixels gave 0, 19, 120 and 233 objects at 5 nm/px, unset,
    10 nm/px and 20 nm/px -- and it is the conversion behind every micron column below.
    A number that a person typed and a number the microscope wrote are different
    evidence, and the bundle could not tell them apart.

    ``file_declared_nm`` is what the source file itself claimed, read off the
    rendition's stored metadata by the assets app. Comparing it with the
    effective value is the whole test:

    * equal -> ``read_from_file``
    * file silent, a value set -> ``entered_by_hand``
    * file declared one thing and the asset says another -> ``overridden_by_hand``

    ``produced_pixel_size_nm`` is what the objects' own stamps recorded (see
    :func:`_produced_pixel_sizes`). ``None`` among them means at least one was
    produced before this pixel size existed, which changes what the value holds
    up -- see :func:`_what_rests_on_the_pixel_size`. Empty (a notebook, or
    objects made before stamping) says nothing either way and is treated as the
    ordinary case, because inventing the warning would be the same guess in the
    other direction.
    """
    from quantem.assets.serializers import file_declared_pixel_size_nm

    predates = None in produced_pixel_size_nm
    effective = asset.pixel_size_nm
    try:
        declared = file_declared_pixel_size_nm(asset)
    except Exception as exc:  # pragma: no cover - provenance never fails a run
        return provenance.section(
            {"effective_nm": effective},
            {
                "source": (
                    "What the source file declared could not be read "
                    f"({exc.__class__.__name__}: {exc}), so whether this pixel "
                    "size was read from the file or typed in is unknown."
                )
            },
        )

    values: dict[str, Any] = {
        "effective_nm": effective,
        "file_declared_nm": declared,
    }
    unavailable: dict[str, str] = {}
    if not effective:
        values["source"] = "unset"
        values["note"] = (
            "No pixel size is set for this image. Nothing in this bundle is in "
            "physical units, and any model that resamples to a canonical scale "
            "ran on the pixels as they are."
        )
        values["applies_to_these_measurements"] = False
        return provenance.section(values, unavailable)
    if declared is None:
        values["source"] = "entered_by_hand"
        values["note"] = (
            "The source file declared no pixel size, so this value was typed by "
            "a person."
        )
    elif abs(float(declared) - float(effective)) <= 1e-9 * max(
        1.0, abs(float(effective))
    ):
        values["source"] = "read_from_file"
        values["note"] = (
            "The source file declared this pixel size and it has not been "
            "changed."
        )
    else:
        values["source"] = "overridden_by_hand"
        values["note"] = (
            f"The source file declared {declared} nm/px and this image is set to "
            f"{effective} nm/px, so a person overrode it. The two are not "
            "interchangeable: the object count depends on the scale a model "
            "resampled to."
        )
    values["note"] += _what_rests_on_the_pixel_size(predates)
    values["applies_to_these_measurements"] = not predates
    return provenance.section(values, unavailable)


def image_identity(
    asset, *, produced_pixel_size_nm: frozenset[float | None] = frozenset()
) -> dict[str, Any]:
    """What the image *is*, beyond a local database row.

    ``image_key`` is a UUID from this machine's SQLite file. It identifies
    nothing anywhere else, and it survives the underlying file being replaced.
    The checksum and the name the file was imported under do not.

    ``file`` is the digest of the **stored rendition** -- the file this run
    actually read pixels from. That is often not the file the user imported: an
    import converts, so ``original_filename`` can say ``.tif`` beside a
    ``file.filename`` of ``.png``. The two are named separately here, and when
    they differ the bundle says outright that it carries no digest of the source
    file, because nothing in the database stores one. A reader must not take the
    sha256 below as a fingerprint of their raw data.
    """
    values: dict[str, Any] = {
        "image_id": str(asset.id),
        "display_name": asset.display_name,
        "original_filename": asset.original_filename or None,
        "logical_size_px": [asset.logical_width, asset.logical_height],
        "pixel_size_nm": asset.pixel_size_nm,
        "pixel_size_nm_z": asset.pixel_size_nm_z,
        "pixel_size_provenance": pixel_size_provenance(
            asset, produced_pixel_size_nm=produced_pixel_size_nm
        ),
        "bit_depth": asset.bit_depth,
        "channels": asset.channels,
        "imported_at": asset.created_at.isoformat() if asset.created_at else None,
    }
    unavailable: dict[str, str] = {}
    if not asset.original_filename:
        unavailable["original_filename"] = (
            "This image was imported without its source filename being recorded."
        )
    if not asset.pixel_size_nm:
        unavailable["pixel_size_nm"] = (
            "No pixel size is set for this image, so nothing in this bundle is "
            "in physical units."
        )

    try:
        from quantem.assets.asset_openable import get_asset_openable

        openable = get_asset_openable(asset)
        path = openable.path
    except Exception as exc:
        path = None
        unavailable["file"] = (
            f"The stored image file could not be located ({exc.__class__.__name__}: {exc}), "
            "so it could not be checksummed."
        )
    values["file"] = provenance.file_identity(path, what="the image")
    values["file_is"] = (
        "The stored rendition this run read pixels from, which is what the "
        "sha256 above identifies."
    )

    stored_name = (values["file"] or {}).get("filename")
    if stored_name and asset.original_filename and stored_name != asset.original_filename:
        unavailable["source_file_sha256"] = (
            f"This bundle carries a digest of {stored_name}, the stored "
            f"rendition, not of {asset.original_filename}, the file that was "
            "imported — the import converted it and QuantEM does not record a "
            "checksum of the uploaded bytes. The bundle therefore cannot be "
            "tied to the raw file by digest; tie it by original_filename and "
            "imported_at, and keep the source alongside."
        )
    return provenance.section(values, unavailable)


def _produced_pixel_sizes(segmentation: ImageSegmentation) -> frozenset[float | None]:
    """Pixel sizes the image had when its objects were produced, from the stamps.

    ``None`` in the result means at least one object was produced while the
    image had no pixel size. Empty means nothing is stamped -- objects from
    before run identity, or a notebook -- and the caller must not infer either
    way from that.
    """
    stamps = run_stamps(segmentation).stamps
    if not stamps:
        return frozenset()
    return frozenset(_tally(stamps, "native_pixel_size_nm"))


def _masks_in_run(
    subject: ImageSegmentation,
    compartments: dict[str, ImageSegmentation],
    tissue: ImageSegmentation | None,
) -> list[ImageSegmentation]:
    """Every segmentation whose objects put a number in the bundle, deduplicated.

    The subject supplies the object rows and the count; a compartment supplies
    an area fraction and an enrichment; the tissue supplies the denominator of
    both, of ``objects_per_um2`` and of the Monte-Carlo domain. A run is only as
    well-scaled as the worst of them, and ``canonical_nm_by_pack`` has always
    been built over all three -- so the pixel sizes they were produced under
    have to be as well, or the caveat is drawn from a wider set than the guard.
    """
    by_id: dict[str, ImageSegmentation] = {}
    for segmentation in (subject, *compartments.values(), tissue):
        if segmentation is not None:
            by_id.setdefault(str(segmentation.id), segmentation)
    return list(by_id.values())


def load_inputs(run: AnalysisRun) -> LoadedAnalysis:
    """Assemble one :class:`AnalysisInputs` from an :class:`AnalysisRun` row."""
    from .service import AnalysisInputs  # local: keeps service.py Django-free

    segmentation = run.segmentation
    params = normalise_params(run.params or {}, segmentation=segmentation)
    shape = image_shape(segmentation)
    asset = segmentation.asset

    compartment_segmentations = {
        name: _resolve_segmentation(seg_id, asset_id=asset.id, role=name)
        for name, seg_id in params["compartments"].items()
    }
    tissue = None
    if params["tissue_segmentation_id"]:
        tissue = _resolve_segmentation(
            params["tissue_segmentation_id"],
            asset_id=asset.id,
            role="the tissue mask",
        )

    comp = build_compartment_set(
        compartment_segmentations, tissue=tissue, shape=shape
    )

    points_xy: np.ndarray | None = None
    # Rows the imported CSV could not be read from. Raised here rather than in
    # the service because this is the only layer that still knows which *line*
    # each one was on, and a line number is what makes a 5,000-row export
    # fixable.
    points_caveat: str | None = None
    parsed_points: ParsedPoints | None = None
    if params["points_source"] == "centroids":
        points_xy = object_centroids(segmentation)
    elif params["points_source"] == "csv":
        parsed_points = parse_points_csv(params["points_csv"])
        points_xy = parsed_points.xy
        points_caveat = parsed_points.caveat()

    features = object_features(segmentation)
    sources = object_sources(segmentation)

    compartment_provenance = [
        _mask_provenance(name, seg, confirmed_objects(seg).count(), shape=shape)
        for name, seg in compartment_segmentations.items()
    ]
    if tissue is not None:
        tissue_provenance = _mask_provenance(
            "tissue", tissue, confirmed_objects(tissue).count(), shape=shape
        )
    else:
        tissue_provenance = None

    # Every distinct model behind any mask in this run, so the manifest can
    # carry each one's digests once rather than per compartment.
    pack_ids = sorted(
        {
            source
            for block in [*compartment_provenance, tissue_provenance]
            if block
            for source in block["n_confirmed_by_source"]
            if source not in {SOURCE_MODEL_MANUAL, SOURCE_MODEL_UNKNOWN}
        }
    )

    # What the image's pixel size was *when these objects were made*, which is
    # not necessarily what it is now. `None` in here means at least one object
    # was produced uncalibrated -- in any mask this run measures with, not only
    # the subject: an area fraction and a tissue denominator are numbers in the
    # bundle too. Computed before the provenance block because the image's own
    # pixel-size record has to say whether anything here rests on it.
    produced_pixel_size_nm = frozenset(
        value
        for mask in _masks_in_run(segmentation, compartment_segmentations, tissue)
        for value in _produced_pixel_sizes(mask)
    )

    loaded_provenance = {
        "mask_source": (
            "Confirmed segment objects, rasterised from their stored polygons "
            "(exteriors and holes). Candidate and inferred objects are excluded."
        ),
        "image": image_identity(
            asset, produced_pixel_size_nm=produced_pixel_size_nm
        ),
        "image_id": str(asset.id),
        "image_name": asset.display_name,
        "image_shape": [shape[0], shape[1]],
        "subject_segmentation_id": str(segmentation.id),
        "compartments": compartment_provenance,
        "tissue": tissue_provenance,
        "points_source": params["points_source"],
        "points": _points_provenance(params["points_source"], points_xy, parsed_points),
        "model_packs": [provenance.model_pack(pack_id) for pack_id in pack_ids],
    }
    if not pack_ids:
        loaded_provenance["model_packs_note"] = (
            "No confirmed object in this run was produced by a model; every mask "
            "is hand-drawn, so there are no weights to identify."
        )

    # Provenance caveats belong beside the numbers, not three levels down in the
    # JSON: "this threshold may not be the one that ran" is precisely what a
    # reader has to be told before they quote the object count.
    provenance_caveats: list[str] = []
    if points_caveat:
        provenance_caveats.append(points_caveat)
    for block in [*compartment_provenance, tissue_provenance]:
        for caveat in (block or {}).get("run", {}).get("caveats", []):
            if caveat not in provenance_caveats:
                provenance_caveats.append(caveat)

    # The subject segmentation's own proofreading, for the caveats: it is the
    # one the confirmed objects being measured came out of. A compartment loaded
    # from a sibling segmentation carries its own under models.compartments[].
    subject_review = (
        reviewed_area(segmentation, shape)
        if not any(
            block["segmentation_id"] == str(segmentation.id)
            for block in compartment_provenance
        )
        else next(
            block["proofreading"]["reviewed_area"]
            for block in compartment_provenance
            if block["segmentation_id"] == str(segmentation.id)
        )
    )
    reviewed_px = subject_review.get("reviewed_px")

    # One union of the completed regions, used three ways: the per-object
    # in_reviewed_area column, the geometry recorded in the manifest, and the
    # region count beside it.
    review_union, n_review_regions = reviewed_geometry(segmentation)
    reviewed_regions = None
    if review_union is not None:
        _geom_values, _geom_unavailable = reviewed_regions_record(
            review_union, n_regions=n_review_regions
        )
        _geom_values["n_regions"] = n_review_regions
        reviewed_regions = provenance.section(_geom_values, _geom_unavailable)

    inputs = AnalysisInputs(
        image_key=str(asset.id),
        segmentation_id=str(segmentation.id),
        pixel_size_nm=pixel_size_nm(segmentation),
        compartments=comp,
        object_features=features,
        object_sources=sources,
        object_in_reviewed_area=objects_in_reviewed_area(
            segmentation, union=review_union
        ),
        reviewed_regions=reviewed_regions,
        # What the model was trained to see, so run_analysis can say when it did
        # not: an uncalibrated image is not resampled, and the dimensionless
        # numbers are affected even though no unit conversion touched them.
        canonical_nm_by_pack={
            pack_id: _canonical_nm(pack_id)[0] for pack_id in pack_ids
        },
        # ...and the half of `_canonical_nm`'s answer the map cannot hold. Both
        # "declares no canonical scale" and "not a pack this build knows" arrive
        # as None above, and they mean opposite things: the first is a run that
        # was identical with or without a pixel size, the second is a run nobody
        # can characterise. `_packs_that_skipped_a_resample` treats the second
        # as a skipped resample and `run_analysis` must too.
        unrecognised_packs=frozenset(
            pack_id for pack_id in pack_ids if not _canonical_nm(pack_id)[1]
        ),
        produced_pixel_size_nm=produced_pixel_size_nm,
        reviewed_px=(
            (int(reviewed_px), shape[0] * shape[1])
            if reviewed_px is not None
            else None
        ),
        n_rejected=rejected_count(segmentation),
        points_xy=points_xy,
        distance_target=params["distance_target"],
        band_edges_nm=tuple(params["band_edges_nm"]),
        replicates=params["replicates"],
        seed=params["seed"],
        group=params["group"],
    )
    return LoadedAnalysis(
        inputs=inputs,
        params=params,
        provenance=loaded_provenance,
        caveats=tuple(provenance_caveats),
    )


def circular_compartments(run: AnalysisRun, params: dict[str, Any]) -> list[str]:
    """Compartments whose enrichment is measured with their own centroids.

    Returned as names rather than only as a sentence, because the sentence goes
    in the caveat list and on screen while the *columns* it condemns --
    ``enrichment_<name>``, ``z_enrichment_<name>`` -- go in a spreadsheet, where
    there is nowhere to put a paragraph.
    """
    if params.get("points_source") != "centroids":
        return []
    own_id = str(run.segmentation_id)
    return sorted(
        name
        for name, seg_id in (params.get("compartments") or {}).items()
        if str(seg_id) == own_id
    )


def centroid_self_reference_caveat(
    run: AnalysisRun, params: dict[str, Any]
) -> str | None:
    """Warn when the point set is the centroids of one of the compartments.

    Enrichment of a compartment measured with that compartment's own centroids is
    ``1 / area_fraction`` by construction -- it says nothing about the biology.
    The result is still computed (the other compartments are informative) but the
    circular one has to be named, or someone will quote it.
    """
    circular = circular_compartments(run, params)
    if not circular:
        return None
    one = len(circular) == 1
    columns = ", ".join(
        f"enrichment_{name} and z_enrichment_{name}" for name in circular
    )
    return (
        "The points are the centroids of the objects that also define "
        f"{', '.join(circular)}, so enrichment in "
        f"{'that compartment' if one else 'those compartments'} is circular (it "
        "is 1 / area fraction by construction) and must not be reported as a "
        f"result. Those are the {columns} columns of image_summary.csv, named "
        "again in its circular_columns field."
    )


# ---------------------------------------------------------------------------
# What the human did
# ---------------------------------------------------------------------------

#: ``SegmentObject.label_state`` for a candidate a person looked at and threw
#: away. The Adapt wizard counts these to explain a held-out score; the analysis
#: manifest counts them because a confirmed set of 28 that started as 42
#: candidates is not the same evidence as one that started as 28.
REJECTED = "EXCLUDED"

#: Ceiling on the completed-region geometry inlined into a manifest. A hand-drawn
#: lasso over a 4k image can be tens of thousands of vertices, and a manifest is
#: a file people open in a text editor. Over the limit the geometry is omitted
#: *with a reason* rather than silently, per this module's first rule.
MAX_REVIEWED_GEOMETRY_WKT_CHARS = 200_000


def rejected_count(segmentation: ImageSegmentation) -> int:
    """Candidates rejected in this segmentation."""
    return SegmentObject.objects.filter(
        segmentation=segmentation, label_state=REJECTED
    ).count()


def label_state_counts(segmentation: ImageSegmentation) -> dict[str, int]:
    """Every object of this segmentation by label state, rejected ones included."""
    rows = (
        SegmentObject.objects.filter(segmentation=segmentation)
        .values_list("label_state")
        .annotate(n=Count("id"))
    )
    return {str(state): int(n) for state, n in sorted(rows)}


def reviewed_area(
    segmentation: ImageSegmentation, shape: tuple[int, int]
) -> dict[str, Any]:
    """How much of the image a person marked as exhaustively reviewed.

    A :class:`~quantem.segmentation.models.CompletedROI` is the user's own
    statement that they went through a region and every object in it is either
    confirmed or rejected. Outside it, what is in the object set is whatever
    inference produced and nobody checked -- and every count, area fraction and
    density in this bundle covers the whole image either way.

    Measured off the polygons with shapely rather than by rasterising: the ROIs
    are stored merged into disjoint polygons, the union is exact, and a mask the
    size of the image is a lot of memory to allocate for one number.
    """
    height, width = shape
    image_px = int(height) * int(width)
    values: dict[str, Any] = {"image_px": image_px}
    unavailable: dict[str, str] = {}

    from quantem.segmentation.services.completed_rois import list_completed_rois

    polygons = [
        roi.geometry
        for roi in list_completed_rois(segmentation).only(
            "geometry_wkb", "segmentation_id"
        )
        if roi.geometry is not None and not roi.geometry.is_empty
    ]
    if not polygons:
        reason = (
            "No completed area is recorded for this segmentation. Either nobody "
            "marked a region as reviewed, or the proofreading was done without "
            "one; the two are indistinguishable here, so the reviewed area is "
            "unknown rather than zero."
        )
        for key in ("reviewed_px", "reviewed_fraction", "n_regions", "bbox_px"):
            unavailable[key] = reason
        return provenance.section(values, unavailable)

    try:
        from shapely.ops import unary_union

        union = unary_union(polygons)
        reviewed = float(union.area)
        minx, miny, maxx, maxy = union.bounds
    except Exception as exc:  # pragma: no cover - shapely is a hard dependency
        reason = (
            f"The {len(polygons)} completed region(s) recorded here could not be "
            f"combined to measure them ({exc.__class__.__name__}: {exc})."
        )
        values["n_regions"] = len(polygons)
        for key in ("reviewed_px", "reviewed_fraction", "bbox_px"):
            unavailable[key] = reason
        return provenance.section(values, unavailable)

    values["n_regions"] = len(polygons)
    values["reviewed_px"] = int(round(reviewed))
    values["bbox_px"] = [minx, miny, maxx, maxy]
    if image_px:
        values["reviewed_fraction"] = reviewed / image_px
    else:  # pragma: no cover - image_shape() rejects a zero-sized image first
        unavailable["reviewed_fraction"] = "The image has no area, so this is 0/0."

    geometry_values, geometry_unavailable = reviewed_regions_record(
        union, n_regions=len(polygons)
    )
    values.update(geometry_values)
    unavailable.update(geometry_unavailable)
    values["note"] = (
        "The union of the regions a person marked complete, in image pixels. "
        "Objects outside it are model output that has not been reviewed, and "
        "they are counted and measured here on the same footing as the rest."
    )
    return provenance.section(values, unavailable)


def reviewed_regions_record(
    union, *, n_regions: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """The completed regions themselves, as ``(values, unavailable)``.

    An area and a bounding box cannot be turned back into "which pixels were
    reviewed", so the reviewed/unreviewed split -- the thing several caveats in
    every bundle point at -- was unreconstructible from the bundle. WKT because
    it is the one geometry spelling shapely, GEOS, PostGIS and QGIS all read
    unaided.
    """
    values: dict[str, Any] = {
        "geometry_note": (
            "regions_wkt is the union of the regions a person marked complete, "
            "in image pixel coordinates (x right, y down, matching every other "
            "coordinate in this bundle). objects.csv gives the same split per "
            "object, in its in_reviewed_area column."
        )
    }
    wkt = union.wkt
    if len(wkt) <= MAX_REVIEWED_GEOMETRY_WKT_CHARS:
        values["regions_wkt"] = wkt
        return values, {}
    return values, {
        "regions_wkt": (
            f"The union of the {n_regions} completed region(s) is {len(wkt):,} "
            f"characters of WKT, over this manifest's "
            f"{MAX_REVIEWED_GEOMETRY_WKT_CHARS:,}-character limit, so the "
            "geometry is not inlined. reviewed_px, bbox_px and the "
            "in_reviewed_area column of objects.csv still describe it."
        )
    }


def reviewed_geometry(segmentation: ImageSegmentation) -> tuple[Any, int]:
    """``(union, n_regions)`` for this segmentation's completed regions.

    ``(None, 0)`` means no completed area is recorded -- nobody said which parts
    were reviewed -- and callers must carry that through as unknown rather than
    collapsing it to "not reviewed".
    """
    from shapely.ops import unary_union

    from quantem.segmentation.services.completed_rois import list_completed_rois

    polygons = [
        roi.geometry
        for roi in list_completed_rois(segmentation).only(
            "geometry_wkb", "segmentation_id"
        )
        if roi.geometry is not None and not roi.geometry.is_empty
    ]
    if not polygons:
        return None, 0
    try:
        return unary_union(polygons), len(polygons)
    except Exception:  # pragma: no cover - shapely is a hard dependency
        logger.exception("could not union the completed regions of %s", segmentation.id)
        return None, len(polygons)


def objects_in_reviewed_area(
    segmentation: ImageSegmentation, *, union: Any = None
) -> dict[str, bool] | None:
    """Which confirmed objects sit in a region a person marked as reviewed.

    ``None`` -- not an empty dict, and not a dict of ``False`` -- when no
    completed area is recorded at all. Every count in an export bundle is over
    the whole image whatever fraction of it was actually gone through, and the
    bundle said so in a caveat while leaving the two indistinguishable in
    ``objects.csv``. This is what makes them distinguishable.

    Membership is by **centroid**: an object is inside if its centroid is in the
    reviewed union. Objects straddling a boundary exist, but a completed region
    is a statement about an area a person swept, and a single unambiguous rule
    beats a third state in a spreadsheet column.

    Pass ``union`` from :func:`reviewed_geometry` to reuse one that has already
    been built.
    """
    if union is None:
        union, _n = reviewed_geometry(segmentation)
    if union is None:
        return None

    from shapely.geometry import Point
    from shapely.prepared import prep

    prepared = prep(union)
    out: dict[str, bool] = {}
    for object_id, cx, cy in confirmed_objects(segmentation).values_list(
        "id", "centroid_x", "centroid_y"
    ):
        if cx is None or cy is None:
            continue
        out[str(object_id)] = bool(prepared.intersects(Point(float(cx), float(cy))))
    return out

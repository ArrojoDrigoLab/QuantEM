"""Per-object morphometrics in calibrated units.

Raw per-object features — area, perimeter, eccentricity, axis lengths, Feret
diameter and intensity percentiles — are stored in pixels. This module turns
those raw pixel values into the numbers a paper reports.

Everything needing a physical unit requires ``pixel_size_nm``. Without it the
function returns pixel values and marks them uncalibrated rather than guessing —
assuming a pixel size silently produces wrong micron values for every image not
acquired at that scale.

**The stored spelling is not a private detail of this module.** :func:`derive`
reads exactly :data:`~quantem.seg_core.extraction.SEGMENT_FEATURE_KEYS`, the
vocabulary the extractor writes, imported rather than retyped. It was retyped
once, with ``intensity_mean`` here against ``mean_intensity`` there, and the
result was ten blank columns out of twenty-seven for every model-produced object
— measurements that had been computed, stored and then looked up under a name
nobody wrote.

**A derived quantity is answerable for its own range.** Most columns here are a
stored measurement in different units, and a unit conversion cannot invent an
impossible value. ``circularity`` is not: it combines ``area`` and ``perimeter``,
which are measured under different pixel conventions
(:mod:`quantem.segmentation.features.geometry`), and the combination runs
**above its own ceiling of 1.0** for small compact objects — a shipped
``objects.csv`` carried 1.2272 and the summary above it read ``MAX 1.227``. The
rule this module follows is that an impossible number is not a measurement:
values over :data:`CIRCULARITY_REPORT_CEILING` are blanked with the reason
recorded. Which perimeter estimator to report is a separate decision and is not
taken here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantem.seg_core.extraction import SEGMENT_FEATURE_KEYS

#: Feature keys stored on SegmentObject.features, in pixels.
PIXEL_AREA_KEY = "area"
PIXEL_PERIMETER_KEY = "perimeter"

#: The exported shape descriptor derived from the two above, and the only number
#: in this bundle with a hard mathematical ceiling.
CIRCULARITY_KEY = "circularity"

#: Circularity's ceiling. ``4 pi A / P^2`` is 1.0 for a perfect circle and below
#: it for every other shape — that is the isoperimetric inequality, not a
#: convention, a unit choice or a rounding tolerance. A value above it therefore
#: says something about the *estimator* and nothing about the object, so
#: :func:`derive` refuses to export it.
CIRCULARITY_MAX = 1.0

#: The value above which a circularity is withheld as an estimator failure.
#:
#: This is NOT the theoretical ceiling, and the gap between the two is the
#: point. Under ``perimeter_crofton`` a *genuinely round* object scatters
#: within about +/-1.5% of 1.0 -- measured across rasterised and polygon-drawn
#: discs from r=10 to r=100 px, the largest value a true circle produced was
#: 1.011. Blanking at exactly 1.0 would therefore censor roughly half of the
#: roundest objects, truncating the estimator's sampling distribution from
#: above and biasing every group mean of a round population downward -- by more
#: the rounder the group, which is precisely the shape of artefact the
#: estimator switch was made to remove. So values in ``(1.0, 1.015]`` are
#: reported as measured ("as round as this estimator can resolve"), and only
#: values beyond the measured envelope -- which discs of r < ~10 px and tiny
#: cornered shapes produce -- are withheld. The envelope is from measurement,
#: not preference; re-derive it if the estimator ever changes again.
CIRCULARITY_REPORT_CEILING = 1.015

#: Why a circularity cell is blank when ``area`` and ``perimeter`` are both
#: filled in. Reported once per metric rather than once per row, the way
#: ``mean_prob``'s absence is.
CIRCULARITY_ABOVE_CEILING_REASON = (
    f"Their 4*pi*area/perimeter^2 came out above {CIRCULARITY_REPORT_CEILING:g}. "
    f"The theoretical ceiling is {CIRCULARITY_MAX:g}, and the estimator's "
    "measured envelope for a genuinely round object reaches about 1.011 — so a "
    "value beyond 1.015 measures the estimator failing on a small object, not "
    "the object, and is left blank rather than exported as a roundness."
)

#: The estimator description included with exported column documentation.
#:
#: Blanking the impossible values is the floor, not the fix. The estimator is
#: biased in one direction over the whole range, by an amount that depends on
#: how big the object is, so a circularity that is *below* 1.0 is still not
#: comparable between two populations of different size — and "the treated group
#: is rounder" is exactly what a treatment that shrinks mitochondria produces out
#: of a perfectly correct segmentation. The bias is identical for model-found
#: and hand-drawn objects (both are measured off one mask), so it is not a
#: provenance problem and the parity between the two does not touch it.
#:
#: The numbers are measured, not estimated: squares and discs swept from 3 to
#: 100 px through the app's own ``compute_segment_features``.
CIRCULARITY_ESTIMATOR_NOTE = (
    "circularity is 4*pi*area/perimeter^2; 1.0 is its theoretical ceiling, "
    "reached only by a perfect circle. The perimeter here is scikit-image's "
    "perimeter_crofton on the pixel mask. Bundles "
    "whose environment.perimeter_estimator field is absent or names "
    "regionprops.perimeter used the earlier estimator, whose bias grew as "
    "objects shrank and could turn a pure size change into a roundness "
    "difference — perimeter and circularity are not comparable across that "
    "boundary. Crofton is close to unbiased on round shapes: a disc measures "
    "0.995 at r=10 px, 1.008 at r=20 and 1.001 at r=80, against a true 1.0. It "
    "is still an estimator, not geometry. A genuinely round object scatters "
    "within about 1.5% of 1.0, on both sides, so a value slightly above 1.0 "
    "(up to 1.015) is reported as measured — it means as round as this "
    "estimator can resolve, and withholding it would censor the roundest "
    "objects and bias a round population's mean downward. Values beyond 1.015 "
    "are estimator failures — discs below about r=10 px and tiny cornered "
    "shapes produce them — and are blank. On cornered shapes crofton reads "
    "high by a roughly constant factor (a square measures 0.900 at 20 px and "
    "0.879 at 100 px against a true 0.785), which cancels between groups of "
    "the same shape class. Below ~10 px radius the estimator is unreliable in "
    "both directions; for populations dominated by such objects, report the "
    "size distribution beside any circularity comparison."
)

#: Exported column -> the key it is read from on ``SegmentObject.features``.
#:
#: The column names carry their unit (``_px``) and the stored names are
#: scikit-image's; the two differ, so the mapping is written out rather than
#: inferred. Every value on the right must be in
#: :data:`~quantem.seg_core.extraction.SEGMENT_FEATURE_KEYS` and every member of
#: that tuple must appear here — asserted below, because a typo on either side
#: is silent and empties a column.
STORED_FEATURE_FOR_METRIC: dict[str, str] = {
    "area_px": PIXEL_AREA_KEY,
    "perimeter_px": PIXEL_PERIMETER_KEY,
    "eccentricity": "eccentricity",
    "solidity": "solidity",
    "elongation": "elongation",
    "major_axis_px": "major_axis_length",
    "minor_axis_px": "minor_axis_length",
    "feret_max_px": "feret_diameter_max",
    "intensity_mean": "intensity_mean",
    "intensity_p10": "intensity_p10",
    "intensity_p50": "intensity_p50",
    "intensity_p90": "intensity_p90",
    "mean_prob": "mean_prob",
}

#: Shape descriptors :func:`derive` computes rather than reads. Ratios and a
#: diameter derived from the area, all scale-free or in pixels.
DERIVED_PIXEL_METRIC_KEYS: tuple[str, ...] = (
    "aspect_ratio",
    CIRCULARITY_KEY,
    "equivalent_diameter_px",
)

#: Everything :func:`derive` produces without a pixel size, in export order.
PIXEL_METRIC_KEYS: tuple[str, ...] = (
    *STORED_FEATURE_FOR_METRIC,
    *DERIVED_PIXEL_METRIC_KEYS,
)

#: The extra keys :func:`derive` adds once the image has a pixel size.
CALIBRATED_METRIC_KEYS: tuple[str, ...] = (
    "area_um2",
    "perimeter_um",
    "major_axis_um",
    "minor_axis_um",
    "feret_max_um",
    "equivalent_diameter_um",
    "pixel_size_nm",
)

#: Every measurement column, pixel and calibrated, in export order.
METRIC_KEYS: tuple[str, ...] = (*PIXEL_METRIC_KEYS, *CALIBRATED_METRIC_KEYS)

#: Every column :meth:`ObjectMetrics.as_row` can emit, in export order.
#:
#: Declared rather than discovered so ``objects.csv`` still has a header row
#: when a run confirmed no objects at all. A zero-byte CSV is not an empty
#: table: ``pandas.read_csv`` raises ``EmptyDataError`` on it, and the person
#: hitting that error is reading the bundle a paper cites. The drift between
#: this list and what :func:`derive` actually returns is caught by a test, not
#: by a comment.
OBJECT_ROW_FIELDS: tuple[str, ...] = (
    "object_id",
    "source_model",
    "calibrated",
    # Which side of the proofreading line this object is on. The bundle already
    # says, in words, that "every count, area fraction and density here is over
    # the whole image, including the 27% that was never gone through, where the
    # objects are unreviewed model output" -- and then said that the two were
    # not distinguishable in objects.csv. This is the column that distinguishes
    # them, so the sentence is actionable rather than only true. Blank means no
    # completed area is recorded at all: nobody said, which is not False.
    "in_reviewed_area",
    *METRIC_KEYS,
)

#: ``SegmentObject.source_model`` for an object a person drew by hand.
MANUAL_SOURCE = "manual"

#: ``SegmentObject.source_model`` before anything could be inferred for it.
UNKNOWN_SOURCE = "unknown"

# A typo in either vocabulary empties a column silently, so refuse to import.
_unknown = set(STORED_FEATURE_FOR_METRIC.values()) - set(SEGMENT_FEATURE_KEYS)
_unread = set(SEGMENT_FEATURE_KEYS) - set(STORED_FEATURE_FOR_METRIC.values())
if _unknown or _unread:  # pragma: no cover - a broken build, not a runtime path
    raise ImportError(
        "quantem.analysis.morphometrics and quantem.seg_core.extraction disagree "
        f"about stored feature names: read-but-never-written={sorted(_unknown)}, "
        f"written-but-never-read={sorted(_unread)}."
    )
del _unknown, _unread


@dataclass(frozen=True)
class ObjectMetrics:
    object_id: str
    calibrated: bool
    values: dict[str, float | None] = field(default_factory=dict)
    #: ``SegmentObject.source_model``: ``"manual"`` for a hand-drawn object,
    #: otherwise the model that produced it. Carried through to ``objects.csv``
    #: and to the reason attached to every partly-populated summary metric.
    source_model: str = ""
    #: Whether this object sits inside a region a person marked as reviewed.
    #: ``None`` when no completed area is recorded at all -- that is "nobody
    #: said", which is not the same as "outside the reviewed area". See
    #: :data:`OBJECT_ROW_FIELDS` for why this column exists.
    in_reviewed_area: bool | None = None
    #: metric -> why this object's value for it was computed and then refused.
    #:
    #: A blank cell in ``objects.csv`` means "not measured" everywhere else in
    #: this module. This is the narrower case where the input *was* there and
    #: the output was not usable -- today only ``circularity`` above
    #: :data:`CIRCULARITY_MAX` -- and it is kept apart from a plain absence so
    #: :func:`summarize` can say which of the two a reader is looking at
    #: instead of reporting a refusal as a missing measurement.
    unreportable: dict[str, str] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "object_id": self.object_id,
            "source_model": self.source_model,
            "calibrated": self.calibrated,
            "in_reviewed_area": self.in_reviewed_area,
        }
        row.update(self.values)
        return row


def derive(
    features: dict[str, Any],
    *,
    object_id: str,
    pixel_size_nm: float | None,
    source_model: str = "",
    in_reviewed_area: bool | None = None,
) -> ObjectMetrics:
    """Turn one object's stored pixel features into calibrated measurements.

    Adds the quantities the stored set lacks but every reader expects:
    equivalent diameter, circularity, aspect ratio, and the micron-unit versions
    of area and perimeter.

    A feature the object does not carry stays ``None``. It is never defaulted to
    zero: 0.0 is a real area, a real intensity and a real probability, and a
    column of fabricated zeros reads as a measurement.

    **Nor is a value the estimator cannot have produced.** ``circularity`` above
    :data:`CIRCULARITY_MAX` is arithmetic on two measurements taken under
    different pixel conventions, not a rounder object, and it is blanked with
    the reason recorded in :attr:`ObjectMetrics.unreportable` rather than
    exported. A shipped ``objects.csv`` carried ``circularity = 1.2272``, and
    the screen above it reported ``MAX 1.227`` for a quantity whose ceiling is
    1.0.
    """
    out: dict[str, float | None] = {
        metric: _f(features.get(stored)) for metric, stored in STORED_FEATURE_FOR_METRIC.items()
    }
    area_px = out["area_px"]
    perim_px = out["perimeter_px"]
    major = out["major_axis_px"]
    minor = out["minor_axis_px"]
    unreportable: dict[str, str] = {}

    # Shape descriptors that need no calibration (ratios are scale-free).
    out["aspect_ratio"] = (major / minor) if (major and minor) else None
    circularity = (4.0 * math.pi * area_px / (perim_px**2)) if (area_px and perim_px) else None
    if circularity is not None and circularity > CIRCULARITY_REPORT_CEILING:
        # Not clamped to 1.0 and not rounded down: both would put a number in
        # the column that no measurement produced. The object keeps its area and
        # its perimeter, which are what was actually measured.
        unreportable[CIRCULARITY_KEY] = CIRCULARITY_ABOVE_CEILING_REASON
        circularity = None
    out[CIRCULARITY_KEY] = circularity
    out["equivalent_diameter_px"] = math.sqrt(4.0 * area_px / math.pi) if area_px else None

    if not pixel_size_nm or pixel_size_nm <= 0:
        return ObjectMetrics(
            object_id=object_id,
            calibrated=False,
            values=out,
            source_model=source_model,
            in_reviewed_area=in_reviewed_area,
            unreportable=unreportable,
        )

    nm = float(pixel_size_nm)
    um = nm / 1000.0
    out.update(
        {
            "area_um2": area_px * um * um if area_px is not None else None,
            "perimeter_um": perim_px * um if perim_px is not None else None,
            "major_axis_um": major * um if major is not None else None,
            "minor_axis_um": minor * um if minor is not None else None,
            "feret_max_um": (out["feret_max_px"] * um if out["feret_max_px"] is not None else None),
            "equivalent_diameter_um": (
                out["equivalent_diameter_px"] * um
                if out["equivalent_diameter_px"] is not None
                else None
            ),
            "pixel_size_nm": nm,
        }
    )
    return ObjectMetrics(
        object_id=object_id,
        calibrated=True,
        values=out,
        source_model=source_model,
        in_reviewed_area=in_reviewed_area,
        unreportable=unreportable,
    )


def summarize(
    metrics: list[ObjectMetrics], *, keys: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Distribution summary per metric: n, mean, sd, median, IQR, min, max.

    Reported alongside every field-level number so a mean is never quoted
    without its spread or its n.

    **A metric whose n is below the object count carries the reason.** ``n`` on
    its own is not enough: "feret_max_um, mean 1.601, sd 8.6e-5" over four
    objects looks like the Feret diameter of the ninety mitochondria in the
    image, and the only thing separating those two readings is a number in a
    different column. ``n_objects``, ``n_missing``, ``missing_by_source`` and a
    plain-English ``note`` travel with the number itself, so whatever renders it
    cannot show one without the other.

    Values withheld after measurement are counted separately from values that
    were never stored, allowing the interface to explain the two cases without
    displaying a full caveat paragraph beneath the table.
    """
    if not metrics:
        return {}
    total = len(metrics)
    if keys is None:
        present = {k for m in metrics for k in m.values}
        keys = [k for k in METRIC_KEYS if k in present]
        keys += sorted(present - set(METRIC_KEYS))
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        vals: list[float] = []
        missing: list[ObjectMetrics] = []
        for m in metrics:
            value = m.values.get(key)
            if value is not None and np.isfinite(float(value)):
                vals.append(float(value))
            else:
                missing.append(m)

        entry: dict[str, Any]
        if not vals:
            entry = {"n": 0}
        else:
            arr = np.asarray(vals, dtype=float)
            q1, q3 = np.percentile(arr, [25, 75])
            entry = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
                "median": float(np.median(arr)),
                "iqr": float(q3 - q1),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        entry["n_objects"] = total
        entry["n_missing"] = len(missing)
        refused = [m for m in missing if key in m.unreportable]
        notes: list[str] = []
        if missing:
            by_source = _count_by_source(missing)
            entry["missing_by_source"] = by_source
            if refused:
                # Split out from n_missing rather than folded into it: "not
                # measured" and "measured, and the answer was impossible" are
                # different facts about the same blank cell, and only the second
                # one is a statement about the estimator.
                entry["n_unreportable"] = len(refused)
                entry["unreportable_reason"] = refused[0].unreportable[key]
            notes.append(
                _coverage_note(
                    key,
                    entry["n"],
                    total,
                    by_source,
                    n_unreportable=len(refused),
                    unreportable_reason=(refused[0].unreportable[key] if refused else ""),
                )
            )
        if notes:
            entry["note"] = " ".join(notes)
        out[key] = entry
    return out


def density(
    n_objects: int, *, tissue_area_px: int, pixel_size_nm: float | None
) -> dict[str, float | None]:
    """Object count per unit tissue area — the number that makes counts comparable."""
    if tissue_area_px <= 0:
        return {"count": n_objects, "per_um2": None, "tissue_um2": None}
    if not pixel_size_nm or pixel_size_nm <= 0:
        return {"count": n_objects, "per_um2": None, "tissue_um2": None}
    um2 = tissue_area_px * (pixel_size_nm / 1000.0) ** 2
    return {
        "count": n_objects,
        "tissue_um2": um2,
        "per_um2": (n_objects / um2) if um2 else None,
    }


def count_by_source(metrics: list[ObjectMetrics]) -> dict[str, int]:
    """How many objects came from each source, hand-drawn included.

    The manifest records this: it is the split that explains every small n, and
    without it "n=4 of 90" is a mystery rather than "the four a person drew".
    """
    return _count_by_source(metrics)


def _count_by_source(metrics: list[ObjectMetrics]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in metrics:
        source = m.source_model or UNKNOWN_SOURCE
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _source_phrase(source: str, count: int) -> str:
    if source == MANUAL_SOURCE:
        return f"{count} hand-drawn"
    if source == UNKNOWN_SOURCE or not source:
        return f"{count} of unrecorded origin"
    return f"{count} from {source}"


def _coverage_note(
    key: str,
    n: int,
    total: int,
    missing_by_source: dict[str, int],
    *,
    n_unreportable: int = 0,
    unreportable_reason: str = "",
) -> str:
    """One sentence saying what the n of a partly-populated metric refers to.

    ``n_unreportable`` are the blanks this module *created*: the value was
    computed and refused (:attr:`ObjectMetrics.unreportable`). They are named
    apart from the ones that were simply never stored, because "the model does
    not produce a probability for a hand-drawn outline" and "the estimator
    returned a number the metric cannot take" are different things for a reader
    to do something about.
    """
    breakdown = ", ".join(
        _source_phrase(source, count) for source, count in missing_by_source.items()
    )
    missing = sum(missing_by_source.values())
    absent = missing - n_unreportable
    clauses: list[str] = []
    if absent:
        clauses.append(
            f"{absent} carr{'ies' if absent == 1 else 'y'} no stored value for this metric"
        )
    if n_unreportable:
        clauses.append(
            f"{n_unreportable} {'was' if n_unreportable == 1 else 'were'} "
            "measured and could not be reported"
        )
    note = (
        f"Measured on {n} of {total} confirmed object{'s' if total != 1 else ''}; "
        f"{' and '.join(clauses)}. Missing sources: {breakdown}."
    )
    if n_unreportable and unreportable_reason:
        note += f" {unreportable_reason}"
    if key == "mean_prob":
        note += " User-drawn or defined objects have no model probability behind them."
    return note


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None

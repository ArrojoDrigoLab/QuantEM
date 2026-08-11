"""Compartment masks, area fractions, and point enrichment.

Ported and generalised from the Figure-4 pipeline
(``gk_gold_seg/scripts/gold_pipeline/analyze2.py``), which computed these for
immunogold particles in liver. Nothing here is gold- or liver-specific: a
"point set" is any array of ``(x, y)`` positions — object centroids, an imported
CSV of spot detections, immunolabels — and a "compartment" is any binary mask.

The definitions, unchanged from the reference:

* Every organelle mask is first restricted to the tissue mask. Anything outside
  the delineated tissue is excluded from both numerator and denominator, and
  counted separately.
* ``cytoplasm = tissue AND NOT nucleus``. Mitochondria are a *subset* of
  cytoplasm, not a sibling — they are reported both ways.
* ``enrichment(c) = (fraction of on-tissue points in c) / (fraction of tissue
  area occupied by c)``. 1.0 is chance.

That last ratio is the Figure-4C axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Reported when a compartment has zero area — the enrichment ratio is undefined
#: rather than infinite, and callers must render it as such.
UNDEFINED = None


def readable_points(points_xy: np.ndarray) -> np.ndarray:
    """Boolean per row: both coordinates are finite numbers.

    ``~readable_points(pts)`` selects the rows that have no position at all.
    A coordinate that is ``nan`` or ``±inf`` is missing, not ``(0, 0)`` --
    :func:`assign_points` says at length what happened when it was treated as a
    position -- and every function that reads a point set has to make the same
    cut, or two of them will report different populations from one array.
    """
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    return np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])


def out_of_image(xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Boolean per coordinate: the position is not inside the image.

    The bound is ``[-1, w]`` rather than ``[0, w - 1]``: pixel indices run to
    ``w - 1`` but a coordinate is a position, and whether the image spans
    ``[0, w)`` (corner convention, as the stored polygons use) or
    ``[-0.5, w - 0.5]`` (centre convention, which rounding implies) it does not
    end at ``w - 1``. Testing the index range would report an object touching
    the right edge as out of the image.
    """
    h, w = shape
    return (xs < -1.0) | (xs > w) | (ys < -1.0) | (ys > h)


def pixel_indices(
    xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Round and clip finite coordinates onto valid ``(x, y)`` pixel indices.

    **Clip as floats, cast after.** ``np.round(1e30).astype(int)`` is
    ``INT_MIN``, and ``np.clip(INT_MIN, 0, w - 1)`` is ``0``, so casting first
    turns any coordinate outside the int64 range into a real observation at the
    image origin -- the same fabrication the non-finite rows produced, except
    that ``1e30`` is a finite number and passes every "is this a number" test on
    the way in. Clipping the float first bounds it before the cast can overflow.

    Every site that turns a coordinate into a pixel index calls this. There were
    two, written the two different ways, and one image's point set was reported
    as sitting on the far edge by :func:`assign_points` and at the origin by
    :func:`quantem.analysis.distances.distance_to_boundary` at the same time.

    Rows that are not finite must be removed first -- see
    :func:`readable_points`. This clips, it does not repair.
    """
    h, w = shape
    return (
        np.clip(np.round(xs), 0, w - 1).astype(int),
        np.clip(np.round(ys), 0, h - 1).astype(int),
    )


@dataclass(frozen=True)
class CompartmentSet:
    """Binary masks over one image, all the same shape.

    ``tissue`` is the denominator for everything. If the user has not painted a
    tissue mask, pass ``tissue=None`` and the whole image is treated as tissue —
    but every fraction then includes empty resin, so the app should say so.
    """

    masks: dict[str, np.ndarray]
    tissue: np.ndarray | None = None
    #: Compartments that are subsets of another, e.g. {"mito": "cytoplasm"}.
    #: Recorded so a reader can tell which fractions are meant to sum to 1.
    nested_in: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shapes = {m.shape for m in self.masks.values()}
        if self.tissue is not None:
            shapes.add(self.tissue.shape)
        if len(shapes) > 1:
            raise ValueError(f"compartment masks disagree on shape: {sorted(shapes)}")

    @property
    def shape(self) -> tuple[int, int]:
        """The image shape every mask here agrees on.

        With no tissue mask and no compartments there is nothing to read it
        from. That cannot happen through the app -- ``normalise_params`` names
        the subject segmentation as a compartment when the caller names none --
        but this module is documented as usable from a notebook, and a bare
        ``CompartmentSet({})`` used to raise ``StopIteration`` from inside a
        property, which is both unreadable and, inside a generator, silently a
        ``StopIteration`` someone else's ``for`` loop would swallow.
        """
        if self.tissue is not None:
            return self.tissue.shape  # type: ignore[return-value]
        for mask in self.masks.values():
            return mask.shape  # type: ignore[return-value]
        raise ValueError(
            "This CompartmentSet has neither a tissue mask nor any compartment "
            "mask, so it does not know the shape of the image it covers. Pass "
            "tissue=<mask>, or at least one compartment."
        )

    def tissue_mask(self) -> np.ndarray:
        if self.tissue is not None:
            return self.tissue.astype(bool)
        return np.ones(self.shape, dtype=bool)

    def restricted(self) -> dict[str, np.ndarray]:
        """Each mask ANDed with tissue — the form every calculation uses."""
        tis = self.tissue_mask()
        return {name: (m.astype(bool) & tis) for name, m in self.masks.items()}


@dataclass(frozen=True)
class AreaFractions:
    tissue_px: int
    tissue_um2: float | None
    fractions: dict[str, float]
    areas_px: dict[str, int]
    areas_um2: dict[str, float] | None


def area_fractions(comp: CompartmentSet, *, pixel_size_nm: float | None = None) -> AreaFractions:
    """Area of each compartment as a fraction of tissue area.

    ``cytoplasm`` is derived as ``tissue AND NOT nucleus`` when a nucleus mask is
    present and no explicit cytoplasm mask was supplied, matching
    ``analyze2.py:31``.

    The derived fraction is computed from the derived *area*, under the same
    ``if total`` guard as every other compartment, and not as ``1 - nucleus``.
    With an empty tissue mask the subtraction reads ``1.0 - 0.0`` and asserts
    that the whole of a zero-pixel tissue is cytoplasm; because 1.0 is truthy it
    then gives that compartment a *defined* enrichment of 0.0 -- maximal
    depletion of a compartment that has no area -- while the nucleus beside it is
    correctly :data:`UNDEFINED`. A zero tissue makes every fraction zero, which
    is what the caveat printed next to these numbers already says.

    A **non-positive** ``pixel_size_nm`` produces no micron values at all, the
    same as ``None``. ``(-5/1000)**2`` is ``(5/1000)**2``, so a truthiness test
    alone turned an impossible calibration into micron areas that look right and
    are not attributable to any real scale. :func:`quantem.analysis.
    morphometrics.derive` and :func:`~quantem.analysis.morphometrics.density`
    already refused it; this is the third place that had to.
    """
    tis = comp.tissue_mask()
    total = int(tis.sum())
    restricted = comp.restricted()

    areas_px = {name: int(m.sum()) for name, m in restricted.items()}
    if "nucleus" in areas_px and "cytoplasm" not in areas_px:
        areas_px["cytoplasm"] = total - areas_px["nucleus"]
    fractions = {name: (px / total if total else 0.0) for name, px in areas_px.items()}

    um2 = None
    tissue_um2 = None
    if pixel_size_nm and pixel_size_nm > 0:
        px_um2 = (pixel_size_nm / 1000.0) ** 2
        um2 = {name: px * px_um2 for name, px in areas_px.items()}
        tissue_um2 = total * px_um2

    return AreaFractions(
        tissue_px=total,
        tissue_um2=tissue_um2,
        fractions=fractions,
        areas_px=areas_px,
        areas_um2=um2,
    )


@dataclass(frozen=True)
class PointAssignment:
    """Where each point landed, and the resulting enrichment per compartment.

    ``n_total == n_on_tissue + n_off_tissue + n_unreadable``. The three are
    disjoint, and the third is not a kind of the second: a point that could not
    be read has no position at all, so it is not "outside the tissue".
    """

    n_total: int
    n_on_tissue: int
    n_off_tissue: int
    #: Rows of ``points_xy`` whose x or y was not a finite number. They are in
    #: no count, fraction or enrichment below. See :func:`assign_points`.
    n_unreadable: int
    #: Readable points whose rounded position fell outside the image and were
    #: clipped onto its border. They *are* counted in everything below, because
    #: clipping is the documented behaviour; the number is here so a caller can
    #: say that a fifth of the point set was pinned to one edge pixel.
    n_out_of_bounds: int
    counts: dict[str, int]
    fractions: dict[str, float]
    enrichment: dict[str, float | None]
    #: Boolean mask over the input points, True where on tissue. False for an
    #: unreadable row, which is on no tissue and off no tissue.
    on_tissue: np.ndarray
    #: Boolean mask over the input points, True where both coordinates were
    #: finite. ``~readable`` selects exactly the rows counted by
    #: :attr:`n_unreadable`, so a caller can name the offending rows.
    readable: np.ndarray
    #: Per-compartment boolean masks over **all** the input points -- the same
    #: length and order as ``points_xy`` and as :attr:`on_tissue`, already ANDed
    #: with it so an off-tissue point is False everywhere. Index them against the
    #: full point array, never against ``points_xy[on_tissue]``.
    membership: dict[str, np.ndarray]


def assign_points(
    points_xy: np.ndarray,
    comp: CompartmentSet,
    *,
    areas: AreaFractions | None = None,
) -> PointAssignment:
    """Assign points to compartments and compute area-normalised enrichment.

    Points are rounded to the nearest pixel and clipped to the image. Points
    outside the tissue mask are excluded from every fraction and reported as
    ``n_off_tissue`` — the Figure-4 tables carry exactly this column, because
    silently dropping them would inflate every enrichment ratio.

    **A coordinate that is not a finite number is missing, not (0, 0).** The
    clip above is the reason this has to be said: ``np.round(nan).astype(int)``
    and the same on ``±inf`` are ``INT_MIN``, which ``np.clip(..., 0, w - 1)``
    turns into 0. Every unusable row therefore became a genuine observation at
    the image origin, and the only trace was a ``RuntimeWarning: invalid value
    encountered in cast`` on stderr. Measured: four points -- one real at
    (80, 80) plus ``nan,nan``, ``inf,0`` and ``-inf,0`` -- against a mask
    covering 2.42% of the tissue reported ``n_on_tissue 4``, ``counts
    {'mito': 3}`` and an **enrichment of 31.0** at z = 13.2, from three
    coordinates that could not be read. ``numpy.savetxt`` writes ``nan`` for a
    missing value and a failed fit writes ``inf``, so this is what an ordinary
    upstream tool produces. Such rows are dropped from every count and reported
    as :attr:`~PointAssignment.n_unreadable`; the caller must say so.

    **With no point on the tissue there is no enrichment**, not an enrichment of
    zero. ``fractions`` is then 0/0 and reported as 0.0, exactly as
    :func:`area_fractions` reports a zero tissue -- but the ratio built on it is
    :data:`UNDEFINED`, for the reason spelled out there: a *defined* 0.0 is
    maximal depletion, a real and extreme finding, and it reaches
    ``image_summary.csv`` where someone sorts on it. It also fed a Monte-Carlo
    null of twenty identical zeros that reported ``p = 1.0`` as a statistic.
    Coordinates given in nanometres instead of pixels land every point off the
    tissue and produce exactly this.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points_xy must have shape (N, 2+)")

    h, w = comp.shape
    tis = comp.tissue_mask()
    restricted = comp.restricted()
    areas = areas or area_fractions(comp)

    readable = readable_points(pts)
    n_unreadable = int((~readable).sum())

    # Round and clip only what can be rounded and clipped. The unreadable rows
    # keep a False everywhere rather than an index, so nothing they touch has a
    # pixel to be assigned to.
    # Counted on the raw coordinates and before the clip, because after it there
    # is nothing left to see. A point at x = 800 on a 100 px image is not an
    # observation of this image; it is what a CSV in nanometres looks like, and
    # clipping puts every one of them on the same border pixel. Clipping is
    # still what happens -- it is the documented behaviour -- but the caller is
    # told how many needed it.
    ux = pts[readable, 0]
    uy = pts[readable, 1]
    n_out_of_bounds = int(out_of_image(ux, uy, (h, w)).sum())

    px, py = pixel_indices(ux, uy, (h, w))

    def _full(values: np.ndarray) -> np.ndarray:
        """A readable-subset boolean array widened back to one row per point."""
        out = np.zeros(pts.shape[0], dtype=bool)
        out[readable] = values
        return out

    on = _full(tis[py, px])
    n_on = int(on.sum())

    counts: dict[str, int] = {}
    membership: dict[str, np.ndarray] = {}
    for name, mask in restricted.items():
        hit = _full(mask[py, px]) & on
        membership[name] = hit
        counts[name] = int(hit.sum())

    if "nucleus" in counts and "cytoplasm" not in counts:
        cyto = on & ~membership["nucleus"]
        membership["cytoplasm"] = cyto
        counts["cytoplasm"] = int(cyto.sum())

    fractions = {name: (c / n_on if n_on else 0.0) for name, c in counts.items()}
    enrichment: dict[str, float | None] = {}
    for name, frac in fractions.items():
        a = areas.fractions.get(name)
        enrichment[name] = (frac / a) if (a and n_on) else UNDEFINED

    return PointAssignment(
        n_total=int(pts.shape[0]),
        n_on_tissue=n_on,
        n_off_tissue=int(pts.shape[0] - n_on - n_unreadable),
        n_unreadable=n_unreadable,
        n_out_of_bounds=n_out_of_bounds,
        counts=counts,
        fractions=fractions,
        enrichment=enrichment,
        on_tissue=on,
        readable=readable,
        membership=membership,
    )

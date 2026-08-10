"""Distance from points to the nearest boundary of a mask.

Ported from ``gk_gold_seg/scripts/gold_pipeline/analyze2.py:53-61`` and
``analyze3.py:53-57``: erode the mask, take ``mask AND NOT eroded`` as its
boundary, then query a ``cKDTree`` of boundary pixels. Distances are exact
Euclidean in pixels, converted with the image's ``pixel_size_nm``.

**One implementation, deliberately.** The published Figure-4D used *two*: exact
KD-tree distances for the observed points, but a 96 nm binary dilation
(``RADIUS_PX = round(100/(8*2)) = 6`` px on a 16 nm grid) for the Monte-Carlo
null it was compared against. Those are not the same measurement. Everything
here — observed and simulated alike — goes through :func:`distance_to_boundary`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .compartments import out_of_image, pixel_indices, readable_points

#: Default bands in nanometres. The upper edge is open.
DEFAULT_BAND_EDGES_NM: tuple[float, ...] = (0.0, 50.0, 100.0, 200.0)


def band_labels(edges_nm: tuple[float, ...] = DEFAULT_BAND_EDGES_NM) -> list[str]:
    labels = []
    for i in range(len(edges_nm) - 1):
        labels.append(f"{edges_nm[i]:g}-{edges_nm[i + 1]:g} nm")
    labels.append(f">{edges_nm[-1]:g} nm")
    if edges_nm[0] == 0.0 and labels:
        labels[0] = f"<{edges_nm[1]:g} nm"
    return labels


def boundary_pixels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(xs, ys)`` of the mask's inner boundary — ``mask & ~binary_erosion(mask)``."""
    m = mask.astype(bool)
    ys, xs = np.where(m & ~ndi.binary_erosion(m))
    return xs, ys


@dataclass(frozen=True)
class DistanceResult:
    #: Signed distance in nm: negative inside the mask, positive outside.
    #: The reference reported unsigned distance plus a separate inside flag;
    #: a signed value carries both and cannot be mismatched.
    #:
    #: **One entry per readable point, not per input point.** A row whose
    #: coordinate is not a number has no distance to anything; there is no
    #: "missing" value to put in its place that a median or a histogram would
    #: not then swallow. :attr:`readable` maps these back onto the input.
    distances_nm: np.ndarray
    inside: np.ndarray
    band_edges_nm: tuple[float, ...]
    band_labels: list[str]
    band_counts: list[int]
    band_fractions: list[float | None]
    median_nm: float | None
    #: Boolean mask over the *input* points, True where both coordinates were
    #: finite. The same cut :func:`quantem.analysis.compartments.assign_points`
    #: makes, so the two report the same population from one array.
    readable: np.ndarray
    #: Input rows with a coordinate that is not a number. In no distance, band
    #: or median here.
    n_unreadable: int
    #: Readable rows whose position lies outside the image. They *are* measured,
    #: from the border position they were clipped onto -- which is the pixel
    #: ``assign_points`` counts them at, so the two agree about where they are.
    n_out_of_image: int

    @property
    def n(self) -> int:
        """Points measured. ``n + n_unreadable`` is the number handed in."""
        return int(self.distances_nm.size)


def distance_to_boundary(
    points_xy: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_size_nm: float,
    band_edges_nm: tuple[float, ...] = DEFAULT_BAND_EDGES_NM,
    signed: bool = True,
) -> DistanceResult:
    """Distance from each point to the nearest boundary pixel of ``mask``.

    Points inside the mask get a negative distance when ``signed`` is set, so
    "inside" and "within 100 nm of the outside" are distinguishable in one array.
    Banding always uses the *absolute* distance, matching the reference.

    **The point set is filtered exactly as
    :func:`~quantem.analysis.compartments.assign_points` filters it**, and this
    has to be spelled out because for a while it was not. Two things follow.

    A row whose coordinate is not a finite number is dropped
    (:attr:`~DistanceResult.n_unreadable`) rather than measured. Handed to the
    KD-tree it raises; handed to the older index arithmetic here it became a
    point at pixel (0, 0), which is a position nobody supplied.

    A readable coordinate that lies outside the image is measured from the
    border position it is clipped onto -- the pixel ``assign_points`` counts it
    at -- and counted in :attr:`~DistanceResult.n_out_of_image`. Measured from
    where it claims to be instead, a point at ``x = 1e30`` on a 256 px image
    reported a median distance to the nearest mitochondrion of 3.5e+30 nm,
    which is 3.5e+21 metres, and the run succeeded and wrote it to
    ``image_summary.csv``. Clipping bounds every distance by the image
    diagonal; the caller still has to say the clipping happened, because a
    distance from a border pixel is not a distance from where the file said.
    """
    if not pixel_size_nm or pixel_size_nm <= 0:
        raise ValueError(
            "pixel_size_nm is required to express distances in nanometres; "
            "set it on the image first"
        )

    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    m = mask.astype(bool)
    xs, ys = boundary_pixels(m)
    h, w = m.shape

    readable = readable_points(pts)
    n_unreadable = int((~readable).sum())
    ux = pts[readable, 0]
    uy = pts[readable, 1]
    n_out_of_image = int(out_of_image(ux, uy, (h, w)).sum())

    labels = band_labels(band_edges_nm)
    if ux.size == 0 or xs.size == 0:
        empty = np.empty(0, dtype=float)
        return DistanceResult(
            distances_nm=empty,
            inside=np.empty(0, dtype=bool),
            band_edges_nm=band_edges_nm,
            band_labels=labels,
            band_counts=[0] * len(labels),
            band_fractions=[None] * len(labels),
            median_nm=None,
            readable=readable,
            n_unreadable=n_unreadable,
            n_out_of_image=n_out_of_image,
        )

    # Query from the clipped position, as floats, so an in-image point keeps its
    # exact sub-pixel coordinate and an out-of-image one is measured from the
    # same border pixel it is counted at. Clipping the *query* is also what
    # keeps ``d_px`` finite: (1e300)**2 overflows to inf, and the sign step below
    # then turns inf into nan, which np.histogram drops silently -- band_counts
    # summed to 1 of 2 points while band_fractions read 1.0 -- and which no
    # JSON, and therefore no database row, will hold.
    qx_f = np.clip(ux, 0.0, float(w - 1))
    qy_f = np.clip(uy, 0.0, float(h - 1))
    tree = cKDTree(np.column_stack([xs, ys]))
    d_px, _ = tree.query(np.column_stack([qx_f, qy_f]))
    d_nm = d_px * float(pixel_size_nm)

    qx, qy = pixel_indices(ux, uy, (h, w))
    inside = m[qy, qx]

    signed_nm = -d_nm * inside + d_nm * ~inside if signed else d_nm

    edges = list(band_edges_nm) + [np.inf]
    counts, _ = np.histogram(np.abs(signed_nm), bins=edges)
    total = int(counts.sum())
    if total != signed_nm.size:  # pragma: no cover - defended, not expected
        raise ValueError(
            f"{signed_nm.size - total} of {signed_nm.size} distances fell in no "
            "band, which means they are not finite numbers. Nothing downstream "
            "of here would show them as missing: a histogram drops them and the "
            "fractions beside it would still add to 1."
        )
    fractions = [(int(c) / total if total else None) for c in counts]

    return DistanceResult(
        distances_nm=signed_nm,
        inside=inside,
        band_edges_nm=band_edges_nm,
        band_labels=labels,
        band_counts=[int(c) for c in counts],
        band_fractions=fractions,
        median_nm=float(np.median(np.abs(signed_nm))),
        readable=readable,
        n_unreadable=n_unreadable,
        n_out_of_image=n_out_of_image,
    )


def nearest_neighbour_nm(
    points_xy: np.ndarray, *, pixel_size_nm: float, other_xy: np.ndarray | None = None
) -> np.ndarray:
    """Nearest-neighbour distance for each point, in nm.

    With ``other_xy`` omitted this is within-set spacing (self-matches excluded),
    which is the standard clustering/dispersion readout. With ``other_xy`` given
    it is cross-set: e.g. every lipid droplet to its nearest mitochondrion.
    """
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty(0, dtype=float)

    if other_xy is None:
        if pts.shape[0] < 2:
            return np.full(pts.shape[0], np.nan)
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)  # k=1 is the point itself
        return d[:, 1] * float(pixel_size_nm)

    other = np.asarray(other_xy, dtype=float).reshape(-1, 2)
    if other.shape[0] == 0:
        return np.full(pts.shape[0], np.nan)
    d, _ = cKDTree(other).query(pts)
    return d * float(pixel_size_nm)


def contact_fraction(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    pixel_size_nm: float,
    within_nm: float = 30.0,
) -> dict[str, float]:
    """Fraction of ``mask_a``'s perimeter lying within ``within_nm`` of ``mask_b``.

    The ER-mitochondria contact-site measurement. Reported as both a coverage
    fraction and a contact length, because a fraction alone is not comparable
    between objects of different size.
    """
    if not pixel_size_nm or pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm is required for contact analysis")

    ax, ay = boundary_pixels(mask_a)
    bx, by = boundary_pixels(mask_b)
    if ax.size == 0:
        return {"perimeter_um": 0.0, "contact_um": 0.0, "coverage": 0.0}

    perimeter_um = ax.size * pixel_size_nm / 1000.0
    if bx.size == 0:
        return {"perimeter_um": perimeter_um, "contact_um": 0.0, "coverage": 0.0}

    d, _ = cKDTree(np.column_stack([bx, by])).query(np.column_stack([ax, ay]))
    in_contact = int((d * pixel_size_nm <= within_nm).sum())
    return {
        "perimeter_um": perimeter_um,
        "contact_um": in_contact * pixel_size_nm / 1000.0,
        "coverage": in_contact / ax.size,
    }

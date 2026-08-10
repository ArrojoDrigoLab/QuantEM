"""Polygon rasterisation, and the pixel convention every measurement uses.

**The convention.** A polygon lives in the same continuous plane as the image.
Pixel ``(row r, col c)`` is the unit square *centred* on ``(x=c, y=r)`` — it
covers ``[c-0.5, c+0.5) x [r-0.5, r+0.5)``. A pixel belongs to a polygon when
**its centre does**, with a point exactly on the boundary counted once by the
half-open rule (a span ``[a, b)`` takes the sample at ``a`` and not the one at
``b``). Nothing else is a pixel of that shape.

That single rule is what makes ``len(mask)`` a measurement rather than a
rendering: the pixel count of the rasterised mask equals ``polygon.area`` for
any axis-aligned rectangle and is unbiased for everything else, so

* a shape a person **drew** and
* a shape a model **found** (``skimage.measure.regionprops`` counts the pixels
  of its label mask, and ``find_contours`` hands back the outline of exactly
  those pixels, at the half-integers this convention puts their edges on)

measure the same when they *are* the same, and ``perimeter`` — measured off the
same mask — describes the same outline as ``area``.

**What this replaces.** Every mask in this app used to come from
``cv2.fillPoly``, which is a *drawing* primitive: it rounds each vertex to the
nearest pixel centre and then paints **both** boundaries of the span, so a
polygon spanning *s* pixels covered *s+1*. A hand-drawn square measured
``(s+1)**2`` instead of ``s**2`` — +44% on a 5 px object, +21% on 10 px, +10% on
a 20 px lipid droplet at 8 nm/px, +2% on 100 px — while the same object found by
a model was counted correctly off its label mask, because that path never
rasterised a polygon at all. One ``objects.csv`` could hold a model object and
the hand-drawn correction of the same organelle whose ``area`` differed by 21%
purely by provenance. The bias is size-dependent, so it does not divide out: it
reached ``area_fraction_*`` (small objects inflate proportionally more than the
large ``tissue_px`` denominator), ``areas_um2``, ``equivalent_diameter_*`` and
``circularity``.

``cv2.fillPoly`` cannot be made to do this. ``shift=`` looks like subpixel
support but only scales the rounding — filling ``[5.0, 15.0]``, ``[5.25, 15.25]``
and ``[4.5, 14.5]`` at ``shift=8`` all yield the same 11-pixel-wide span. So the
scanline fill is implemented here instead, in numpy.

**Cost.** Each edge is visited once per scanline it actually crosses, so the
work is ``O(perimeter + bbox area)`` per ring — the same order as OpenCV, and
spent only inside the ring's own bounding box. It is nonetheless numpy against
C: in isolation it runs 6x-12x ``cv2.fillPoly``, ~100 us for a 24 px object and
~0.5 ms for a 400 px one. Neither place that matters notices. Measuring one
object is dominated by the image read and the ~5 ms ``regionprops`` call this
feeds. A full overlay rebuild of 1500 objects in a 2048x2048 image, timed
through ``rebuild_overlay_full`` itself, went 2.24 s to 2.07 s — the zarr writes
and the pyramid dominate it, not the fill.

The obvious formulation, a ``(rows x edges)`` crossing matrix, *is* slow enough
to matter (24x OpenCV, and superlinear in object size), which is why the fill
below buckets crossings by scanline instead.

Pure numpy, no Django, no OpenCV: importable from a rasterisation worker
process and from :mod:`quantem.analysis`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_ring(ring: np.ndarray) -> np.ndarray | None:
    """``ring`` as an ``(N, 2)`` float64 array, or None when it cannot be filled.

    A non-finite coordinate is rejected outright. The scanline fill leans on
    every scanline meeting a closed ring an even number of times, and a NaN
    silently breaks that -- it compares False against everything, so the edge it
    sits on contributes no crossing while its neighbour contributes one, and the
    spans after it pair up shifted by one. An outline with a NaN in it is not a
    shape; measuring nothing is the honest answer.
    """
    pts = np.asarray(ring, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return None
    if pts.shape[1] > 2:
        pts = pts[:, :2]
    if not np.isfinite(pts).all():
        return None
    return pts


def ring_window(
    ring: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, int, int, int] | None:
    """The part of ``[x0,x1) x [y0,y1)`` that ``ring`` can possibly cover.

    Follows straight from the convention: the sample points inside a span
    ``[lo, hi)`` are the integers from ``ceil(lo)`` to ``ceil(hi) - 1``, so a
    ring bounded by ``[min, max]`` can only touch rows/columns
    ``ceil(min) .. ceil(max) - 1``.

    Returns None when that is empty. Nothing is ever clipped in *geometry*: the
    ring keeps its true coordinates and only the pixels outside the window are
    dropped, so an object running off the edge of the image is measured on the
    part of it that is there rather than on a shape flattened against the
    border. Clipping the coordinates instead -- what the OpenCV path had to do
    to stay in bounds -- bends the outline along the edge and then measures the
    bent shape.
    """
    pts = _as_ring(ring)
    if pts is None:
        return None
    low = pts.min(axis=0)
    high = pts.max(axis=0)
    wx0 = max(int(np.ceil(low[0])), x0)
    wx1 = min(int(np.ceil(high[0])), x1)
    wy0 = max(int(np.ceil(low[1])), y0)
    wy1 = min(int(np.ceil(high[1])), y1)
    if wx1 <= wx0 or wy1 <= wy0:
        return None
    return wx0, wy0, wx1, wy1


def _scanline(
    pts: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    """Even-odd scanline fill of ``pts`` over ``[x0,x1) x [y0,y1)``."""
    height = y1 - y0
    width = x1 - x0
    out = np.zeros((height, width), dtype=bool)
    if height <= 0 or width <= 0:
        return out

    count = pts.shape[0]
    ax = pts[:, 0]
    ay = pts[:, 1]
    # The next vertex round the ring. Written by hand rather than with np.roll,
    # which copies through concatenate and costs more than the fill for a small
    # object.
    bx = np.empty(count)
    by = np.empty(count)
    bx[:-1] = ax[1:]
    bx[-1] = ax[0]
    by[:-1] = ay[1:]
    by[-1] = ay[0]

    # A horizontal edge crosses no scanline. Dropping it is what keeps the
    # crossing count even at a vertex a scanline passes exactly through.
    rising = ay != by
    if not rising.all():
        ax, ay, bx, by = ax[rising], ay[rising], bx[rising], by[rising]
        if ax.size == 0:
            return out

    y_lo = np.minimum(ay, by)
    y_hi = np.maximum(ay, by)

    # Half-open in y: an edge owns the scanline at its lower end and not the one
    # at its upper end, so a vertex two edges share is crossed exactly once.
    # Rows an edge crosses are ceil(y_lo) .. ceil(y_hi) - 1, clipped to the
    # window -- clipping rows, never coordinates.
    first_row = np.minimum(np.maximum(np.ceil(y_lo), y0), y1).astype(np.int64)
    stop_row = np.minimum(np.maximum(np.ceil(y_hi), y0), y1).astype(np.int64)
    spans = stop_row - first_row
    total = int(spans.sum())
    if total <= 0:
        return out

    # One entry per (edge, scanline it crosses): O(perimeter), not O(rows x
    # edges). An edge is never looked at on a scanline it does not reach.
    edge_of = np.repeat(np.arange(spans.size), spans)
    starts = np.cumsum(spans)
    starts -= spans
    sample_row = first_row[edge_of] + (np.arange(total) - starts[edge_of])

    e_ay = ay[edge_of]
    crossing = ax[edge_of] + (sample_row - e_ay) * (bx[edge_of] - ax[edge_of]) / (
        by[edge_of] - e_ay
    )

    local_row = sample_row - y0
    order = np.lexsort((crossing, local_row))
    local_row = local_row[order]
    crossing = crossing[order]

    # Crossings are now grouped by row and sorted within each row, so the 0th,
    # 2nd, 4th ... crossing of a row opens a span and the one after it closes
    # it. Every row's count is even (a scanline meets a closed ring an even
    # number of times), so every row starts at an even offset and the opening
    # crossings are exactly the even *global* indices -- no per-row bookkeeping.
    if total % 2:  # pragma: no cover - not reachable for a finite closed ring
        return out
    open_at = crossing[0::2]
    close_at = crossing[1::2]
    span_row = local_row[0::2]

    # Half-open in x, by the same rule: ceil(lo) .. ceil(hi) - 1.
    lo = np.minimum(np.maximum(np.ceil(open_at) - x0, 0), width).astype(np.int64)
    hi = np.minimum(np.maximum(np.ceil(close_at) - x0, 0), width).astype(np.int64)
    non_empty = hi > lo
    if not non_empty.all():
        lo, hi, span_row = lo[non_empty], hi[non_empty], span_row[non_empty]
        if lo.size == 0:
            return out

    # Difference array: +1 where a span opens, -1 where it closes.
    diff = np.zeros((height, width + 1), dtype=np.int32)
    np.add.at(diff, (span_row, lo), 1)
    np.add.at(diff, (span_row, hi), -1)
    np.cumsum(diff, axis=1, out=diff)
    np.greater(diff[:, :width], 0, out=out)
    return out


def fill_ring(
    ring: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    """Boolean mask of ``[x0,x1) x [y0,y1)`` covered by one closed ring.

    ``ring`` is an ``(N, 2)`` array of ``(x, y)`` vertices, open or closed.
    Row ``i`` of the result is image row ``y0 + i``; column ``j`` is image
    column ``x0 + j``.

    Even-odd fill, so a self-intersecting outline resolves the way
    ``cv2.fillPoly`` resolved it.
    """
    pts = _as_ring(ring)
    if pts is None:
        return np.zeros((max(0, y1 - y0), max(0, x1 - x0)), dtype=bool)
    return _scanline(pts, x0=x0, y0=y0, x1=x1, y1=y1)


def fill_rings(
    rings: Sequence[np.ndarray],
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    """``rings[0]`` filled, ``rings[1:]`` punched out of it.

    An exterior and its holes, as
    :func:`quantem.segmentation.geometry.extract_polygons` yields them.
    """
    height = max(0, y1 - y0)
    width = max(0, x1 - x0)
    mask = np.zeros((height, width), dtype=bool)
    if height == 0 or width == 0 or not len(rings):
        return mask

    for index, ring in enumerate(rings):
        pts = _as_ring(ring)
        if pts is None:
            if index == 0:
                return mask
            continue
        window = ring_window(pts, x0=x0, y0=y0, x1=x1, y1=y1)
        if window is None:
            continue
        wx0, wy0, wx1, wy1 = window
        filled = _scanline(pts, x0=wx0, y0=wy0, x1=wx1, y1=wy1)
        view = mask[wy0 - y0 : wy1 - y0, wx0 - x0 : wx1 - x0]
        if index == 0:
            view |= filled
        else:
            view &= ~filled
    return mask


def paint_ring(
    target: np.ndarray,
    ring: np.ndarray,
    value: int,
    *,
    x0: int,
    y0: int,
) -> None:
    """Write ``value`` into ``target`` everywhere ``ring`` covers.

    ``target`` is an array whose ``[0, 0]`` is image pixel ``(y0, x0)``. Only
    the ring's own bounding box is touched, so painting a thousand small objects
    into one tile costs their combined area and not a thousand tile-sized
    passes.
    """
    pts = _as_ring(ring)
    if pts is None:
        return
    height, width = target.shape
    window = ring_window(pts, x0=x0, y0=y0, x1=x0 + width, y1=y0 + height)
    if window is None:
        return
    wx0, wy0, wx1, wy1 = window
    filled = _scanline(pts, x0=wx0, y0=wy0, x1=wx1, y1=wy1)
    view = target[wy0 - y0 : wy1 - y0, wx0 - x0 : wx1 - x0]
    view[filled] = value


def paint_rings(
    target: np.ndarray,
    rings: Sequence[np.ndarray],
    value: int,
    *,
    x0: int,
    y0: int,
) -> None:
    """Write ``value`` where ``rings[0]`` covers and ``rings[1:]`` do not.

    Only pixels of the shape are written, so a hole in one object cannot erase a
    neighbour that happens to lie under it -- there is no separate scratch
    buffer to keep them apart. Use :func:`paint_ring` instead where a hole is
    meant to punch through to background whatever is beneath it, which is what
    the overlay's label map does.
    """
    if not len(rings):
        return
    exterior = _as_ring(rings[0])
    if exterior is None:
        return
    height, width = target.shape
    window = ring_window(exterior, x0=x0, y0=y0, x1=x0 + width, y1=y0 + height)
    if window is None:
        return
    wx0, wy0, wx1, wy1 = window
    filled = fill_rings([exterior, *rings[1:]], x0=wx0, y0=wy0, x1=wx1, y1=wy1)
    view = target[wy0 - y0 : wy1 - y0, wx0 - x0 : wx1 - x0]
    view[filled] = value

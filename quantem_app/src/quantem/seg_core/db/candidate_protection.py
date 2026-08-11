"""What a model pass is not allowed to overwrite.

A re-run deletes its own previous candidates and writes fresh ones. It must
never take back a decision a person made: an object the user confirmed stays,
and an object the user rejected stays rejected rather than reappearing as a new
candidate on the next pass. That is enforced here, by dropping any freshly
extracted polygon that lands on top of a labeled one.

Two thresholds, not one
-----------------------
* A new candidate overlapping a **CONFIRMED** object by >= 30 % is dropped. The
  bar is low because the confirmed object is already the answer for that piece
  of the image; a second polygon over it is a duplicate, not a discovery.
* A new candidate overlapping an **EXCLUDED** object needs >= 80 % before it is
  dropped. The bar is high because "not this shape" is a narrower statement than
  "yes, this shape": a genuinely different object that merely touches a rejected
  one deserves to be offered again.

Overlap is measured in **both** directions -- intersection over the candidate's
area *or* over the labeled object's area -- so a large candidate swallowing a
small confirmed object is caught as well as a small candidate sitting inside a
large one.

Why there is an index in here
-----------------------------
This used to be two Python lists and a nested loop: every new candidate was
tested against every labeled object. On the images this app is for that is the
wrong shape of work. A well-proofread 60 MP image can hold 3 000 confirmed
objects, and a re-run on it extracts a few thousand fresh candidates, so the
loop performed roughly 15 million shapely calls and the tail of the run took
minutes -- and it got *slower the more careful the user had been*, which is
exactly backwards.

The set of labeled objects a candidate can possibly overlap is tiny and local,
so the geometries go into a :class:`shapely.STRtree` once and each candidate
asks the tree for its neighbours. The arithmetic that decides a hit is
unchanged: same thresholds, same both-directions rule, same
intersection-area comparison. Only the *search* for which pairs are worth
testing is different, and the accompanying benchmark asserts set equality
against the old nested loop on a fixed fixture rather than trusting that
sentence.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.source_models import SOURCE_MODEL_MANUAL

logger = logging.getLogger(__name__)

#: A new candidate overlapping a CONFIRMED object this much is a duplicate.
CONFIRMED_OVERLAP_THRESHOLD = 0.3
#: A new candidate overlapping an EXCLUDED object this much is the same refusal.
EXCLUDED_OVERLAP_THRESHOLD = 0.8


def load_protected_geometries(queryset) -> list[BaseGeometry]:
    """Materialize shapely geometries for a ``SegmentObject`` queryset.

    A row whose WKB cannot be read is skipped with a warning rather than
    failing the run: one corrupt geometry must not cost the user a whole
    segmentation pass. The consequence is stated honestly -- that object
    protects nothing on this pass.

    Parsing is one vectorised call rather than three thousand: ``on_invalid``
    turns a row GEOS refuses into ``None`` instead of an exception, which is the
    same skip the per-row ``try``/``except`` performed, at a twentieth of the
    cost.
    """
    payloads = [bytes(wkb) for wkb in queryset.values_list("geometry_wkb", flat=True) if wkb]
    if not payloads:
        return []
    parsed = shapely.from_wkb(payloads, on_invalid="ignore")
    geometries = [geometry for geometry in parsed if geometry is not None]
    unreadable = len(payloads) - len(geometries)
    if unreadable:
        logger.warning(
            "Skipping %d unreadable segment geometries; they protect nothing on this pass.",
            unreadable,
        )
    return geometries


def _geometry_array(polygons: Sequence[BaseGeometry]) -> np.ndarray:
    """A 1-D object array of geometries, which is what the shapely bulk API takes."""
    array = np.empty(len(polygons), dtype=object)
    array[:] = polygons
    return array


def _ratio_at_least(
    numerators: np.ndarray, denominators: np.ndarray, threshold: float
) -> np.ndarray:
    """``numerator / denominator >= threshold``, with a zero denominator meaning no."""
    positive = denominators > 0
    ratios = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators),
        where=positive,
    )
    return positive & (ratios >= threshold)


class _ProtectedLayer:
    """One protected geometry set, its STRtree, and the threshold it applies.

    The arithmetic that decides a hit is the nested loop's, pair by pair and
    unchanged: intersection area over the candidate's area, or over the labeled
    object's, against the same threshold. Two things about the *shape* of the
    work changed.

    The tree answers "which protected objects could this polygon possibly
    touch" -- the same question the hand-written bounding-box reject asked, and
    with the same answer, since ``STRtree.query`` returns exactly the
    envelope-intersecting set. And the surviving pairs are evaluated in one
    vectorised call instead of several Python-level shapely calls each: at this
    scale the per-call dispatch cost was larger than the geometry work, 154 ms
    of it against 64 ms for the batched form on the 3 000 x 5 000 fixture.
    """

    __slots__ = ("threshold", "_tree", "_geometries", "_areas", "_count")

    def __init__(self, geometries: list[BaseGeometry], threshold: float) -> None:
        self.threshold = float(threshold)
        self._count = len(geometries)
        if not geometries:
            self._tree = None
            self._geometries = None
            self._areas = None
            return
        self._tree = STRtree(geometries)
        # ``STRtree.geometries`` is the indexed set as a numpy object array, in
        # the order the query indices refer to. Areas are computed once up
        # front because a dense image asks for the same ones thousands of times.
        self._geometries = self._tree.geometries
        self._areas = shapely.area(self._geometries)

    def __len__(self) -> int:
        return self._count

    def hits(self, polygons: np.ndarray) -> np.ndarray:
        """Which of ``polygons`` this layer suppresses, as a boolean mask."""
        hit = np.zeros(len(polygons), dtype=bool)
        if self._tree is None or hit.size == 0:
            return hit

        left, right = self._tree.query(polygons)
        if left.size == 0:
            return hit

        candidates = polygons.take(left)
        neighbours = self._geometries.take(right)
        try:
            intersection_areas = shapely.area(shapely.intersection(candidates, neighbours))
        except Exception:
            intersection_areas = self._pairwise_areas(candidates, neighbours)

        pair_hit = _ratio_at_least(
            intersection_areas, shapely.area(candidates), self.threshold
        ) | _ratio_at_least(intersection_areas, self._areas.take(right), self.threshold)
        hit[left[pair_hit]] = True
        return hit

    @staticmethod
    def _pairwise_areas(candidates: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
        """Intersection areas one pair at a time, so one bad pair costs one pair.

        GEOS refusing on a single stored geometry must not cost the user the
        whole pass, which is why the loop this replaced wrapped every pair in
        ``try``/``except``. A pair that cannot be intersected contributes zero
        area, which is the same as the ``continue`` it used to take.
        """
        areas = np.zeros(candidates.size, dtype=float)
        for position, (candidate, neighbour) in enumerate(zip(candidates, neighbours, strict=True)):
            try:
                areas[position] = candidate.intersection(neighbour).area
            except Exception:
                logger.warning(
                    "Skipping an unusable geometry pair while protecting labeled objects.",
                    exc_info=True,
                )
        return areas


class ProtectionIndex:
    """The labeled objects a run may not overwrite, indexed for lookup.

    Build it with :func:`build_protection_index`, then ask
    :meth:`suppressed_mask` about a batch of freshly extracted polygons, or
    :meth:`suppresses` about one.
    """

    __slots__ = ("_confirmed", "_excluded", "_confirmed_hits", "_excluded_hits")

    def __init__(
        self,
        confirmed: list[BaseGeometry],
        excluded: list[BaseGeometry],
    ) -> None:
        self._confirmed = _ProtectedLayer(confirmed, CONFIRMED_OVERLAP_THRESHOLD)
        self._excluded = _ProtectedLayer(excluded, EXCLUDED_OVERLAP_THRESHOLD)
        self._confirmed_hits = 0
        self._excluded_hits = 0

    @property
    def confirmed_count(self) -> int:
        """How many confirmed objects are protecting this segmentation."""
        return len(self._confirmed)

    @property
    def excluded_count(self) -> int:
        """How many rejected objects are protecting this segmentation."""
        return len(self._excluded)

    def suppressed_mask(self, polygons: Sequence[BaseGeometry]) -> list[bool]:
        """Which of these candidates must be dropped rather than written.

        Counts the reason as it goes, so :meth:`stats` can say afterwards how
        many candidates the user's own decisions accounted for. Confirmed is
        tested first and wins the attribution when both would fire, and the
        rejected set is only consulted for the candidates confirmation did not
        already take -- the same order, and the same short circuit, as testing
        one polygon at a time.
        """
        array = _geometry_array(list(polygons))
        confirmed = self._confirmed.hits(array)
        excluded = np.zeros(array.size, dtype=bool)
        remaining = ~confirmed
        if remaining.any():
            excluded[remaining] = self._excluded.hits(array[remaining])

        self._confirmed_hits += int(confirmed.sum())
        self._excluded_hits += int(excluded.sum())
        return (confirmed | excluded).tolist()

    def suppresses(self, polygon: BaseGeometry) -> bool:
        """True when this one candidate must be dropped rather than written."""
        return self.suppressed_mask([polygon])[0]

    def stats(self) -> dict[str, int]:
        """Candidates suppressed so far, split by which decision suppressed them."""
        return {
            "confirmed_hits": self._confirmed_hits,
            "excluded_hits": self._excluded_hits,
        }


def build_protection_index(
    segmentation: ImageSegmentation,
    source_model: str,
) -> ProtectionIndex:
    """Load and index everything this segmentation's user has already decided.

    CONFIRMED objects protect regardless of which model produced them: a
    confirmation is the user's statement about the image, not about the model.
    EXCLUDED objects protect only against the model that is running and against
    the user's own manual work, because a rejection of one model's proposal is
    not a rejection of another model's.
    """
    confirmed = load_protected_geometries(
        SegmentObject.objects.filter(
            segmentation=segmentation,
            label_state="CONFIRMED",
        )
    )
    excluded = load_protected_geometries(
        SegmentObject.objects.filter(
            segmentation=segmentation,
            label_state="EXCLUDED",
            source_model__in=[source_model, SOURCE_MODEL_MANUAL],
        )
    )
    return ProtectionIndex(confirmed, excluded)

"""One answer to "how sure is the model about this object?".

``SegmentObject.confidence_score`` is the column, and it is NULL more often than
it looks: it is only written when the extractor had a probability map to average
under the outline (:mod:`quantem.seg_core.extraction` sets it *from*
``features["mean_prob"]``), and never for an object a person drew.

Two readers had two different fallbacks for that NULL, and returned two
different answers for the same object:

* ``serializers/segments.py`` fell back to ``features["mean_prob"]`` -- which is
  the same measurement the column is filled from, so this one was right;
* ``api_views/segments/query.py`` fell back to ``features["sam_score"]`` only,
  under a comment claiming it returned null "when the object has no score of any
  kind" -- while ``features`` sat in its own ``.only(...)`` field list.

Observed on one object with ``confidence_score=NULL`` and
``features["mean_prob"]=0.82``: ``GET /segments/at-point`` answered 0.82 and
``POST /segments/query-region`` answered null. The click-ranking key in the same
module had the same gap, so that object sorted as unscored under the cursor.

The order here is the one rule. ``mean_prob`` first because it *is* the column's
own measurement; ``sam_score`` after it because it is a score a caller supplied
for the rare object that carries one, and a real score beats none. ``None`` last
and honestly: **missing is not zero.** A ``0.0`` here reads as "the model was
certain this is background", which is the strongest claim the number can make
about an outline a human drew by hand.
"""

from __future__ import annotations

#: Feature keys that can stand in for a NULL ``confidence_score``, best first.
CONFIDENCE_FALLBACK_FEATURE_KEYS: tuple[str, ...] = ("mean_prob", "sam_score")


def _as_score(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if score == score else None  # drop NaN


def confidence_from_features(features: object) -> float | None:
    """The stored confidence on a ``features`` dict, or None if there is none."""
    if not isinstance(features, dict):
        return None
    for key in CONFIDENCE_FALLBACK_FEATURE_KEYS:
        score = _as_score(features.get(key))
        if score is not None:
            return score
    return None


def segment_confidence_score(segment: object) -> float | None:
    """The confidence to report for one object. ``None`` means "no score".

    Every endpoint that reports a confidence goes through here, so two of them
    cannot answer differently about the same row again.
    """
    score = _as_score(getattr(segment, "confidence_score", None))
    if score is not None:
        return score
    return confidence_from_features(getattr(segment, "features", None))


__all__ = [
    "CONFIDENCE_FALLBACK_FEATURE_KEYS",
    "confidence_from_features",
    "segment_confidence_score",
]

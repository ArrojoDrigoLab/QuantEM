"""Pure annotation ownership rules shared by publication and comparison."""

from shapely.geometry import GeometryCollection


def protected_annotation(geometry, protected, label_state):
    """Return the retained geometry and label inside completed annotation.

    Confirmed foreground and explicit rejections retain every intersecting
    polygon and hole. A proposal wholly inside completed work is accepted;
    an unconfirmed proposal crossing that boundary owns no protected pixels.
    """
    kept = geometry.intersection(protected)
    if label_state in {"CANDIDATE", "INFERRED"}:
        if kept.equals(geometry):
            return kept, "CONFIRMED"
        return GeometryCollection(), label_state
    return kept, label_state

"""Lifecycle status constants for non-cell segment objects."""

from __future__ import annotations

SEGMENT_STATUS_CANDIDATE = 0
SEGMENT_STATUS_CONFIRMED = 1
SEGMENT_STATUS_REFINED = 10

SEGMENT_STATUS_LABELS = {
    SEGMENT_STATUS_CANDIDATE: "CANDIDATE",
    SEGMENT_STATUS_CONFIRMED: "CONFIRMED",
    SEGMENT_STATUS_REFINED: "REFINED",
}

SEGMENT_STATUS_CHOICES = (
    (SEGMENT_STATUS_CANDIDATE, SEGMENT_STATUS_LABELS[SEGMENT_STATUS_CANDIDATE]),
    (SEGMENT_STATUS_CONFIRMED, SEGMENT_STATUS_LABELS[SEGMENT_STATUS_CONFIRMED]),
    (SEGMENT_STATUS_REFINED, SEGMENT_STATUS_LABELS[SEGMENT_STATUS_REFINED]),
)


def segment_status_label(status: int | str | None) -> str:
    try:
        normalized = int(status)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        normalized = SEGMENT_STATUS_CANDIDATE
    return SEGMENT_STATUS_LABELS.get(normalized, SEGMENT_STATUS_LABELS[SEGMENT_STATUS_CANDIDATE])


def normalize_segment_status(status: int | str | None) -> int:
    try:
        normalized = int(status)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported segment status: {status!r}") from exc
    if normalized not in SEGMENT_STATUS_LABELS:
        raise ValueError(f"Unsupported segment status: {status!r}")
    return normalized


def status_for_segment_lifecycle(*, label_state: str, refined: str) -> int:
    if label_state == "CONFIRMED":
        if refined != "UNREFINED":
            return SEGMENT_STATUS_REFINED
        return SEGMENT_STATUS_CONFIRMED
    return SEGMENT_STATUS_CANDIDATE


__all__ = [
    "SEGMENT_STATUS_CANDIDATE",
    "SEGMENT_STATUS_CHOICES",
    "SEGMENT_STATUS_CONFIRMED",
    "SEGMENT_STATUS_LABELS",
    "SEGMENT_STATUS_REFINED",
    "normalize_segment_status",
    "segment_status_label",
    "status_for_segment_lifecycle",
]

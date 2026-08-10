"""Background user-feedback processing.

There is no per-organelle handler for a point click yet. The
capture + status machinery (model, endpoint, job, polling) is kept intact so the
proofreading screen keeps recording clicks and reporting their state.
"""

from __future__ import annotations

import logging
from typing import Any

from quantem.segmentation.models import UserFeedback

logger = logging.getLogger(__name__)


def _apply_user_feedback(feedback: UserFeedback) -> dict[str, Any]:
    """Act on one captured point click.

    TODO(quantem): implement add/remove-at-point against the DINO segmenters.
    """
    raise NotImplementedError(
        "Point feedback has no organelle handler in QuantEM yet."
    )


def process_user_feedback(user_feedback_id: str) -> dict[str, Any]:
    """Process one UserFeedback item and always return an outcome payload."""
    feedback = (
        UserFeedback.objects.select_related("segmentation__segmentation_type")
        .filter(id=user_feedback_id)
        .first()
    )
    if feedback is None:
        logger.warning("UserFeedback %s not found", user_feedback_id)
        return {
            "user_feedback_id": user_feedback_id,
            "utilized_status": "FAILED",
            "detail": "not_found",
        }

    feedback.utilized_status = UserFeedback.STATUS_PROCESSING
    feedback.save(update_fields=["utilized_status", "updated_at"])

    try:
        if feedback.input_type != UserFeedback.INPUT_TYPE_POINT:
            raise ValueError(f"Unsupported input_type: {feedback.input_type}")
        result = _apply_user_feedback(feedback)
    except Exception as exc:  # noqa: BLE001 - this must not fail the job
        logger.exception(
            "User feedback processing failed for %s: %s",
            feedback.id,
            exc,
        )
        feedback.utilized_status = UserFeedback.STATUS_FAILED
        feedback.save(update_fields=["utilized_status", "updated_at"])
        return {
            "user_feedback_id": str(feedback.id),
            "utilized_status": feedback.utilized_status,
            "detail": str(exc),
        }

    feedback.utilized_status = UserFeedback.STATUS_SUCCESS
    feedback.save(update_fields=["utilized_status", "updated_at"])
    return {
        "user_feedback_id": str(feedback.id),
        "utilized_status": feedback.utilized_status,
        "result": result,
    }

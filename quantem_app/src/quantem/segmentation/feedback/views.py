"""Views for async user feedback capture and polling."""

from __future__ import annotations

import logging
import os
import time
from datetime import timedelta

from django.db import OperationalError
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.jobs.constants import QUEUE_P1_INTERACTIVE
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, UserFeedback
from quantem.segmentation.serializers import (
    UserFeedbackCreateSerializer,
    UserFeedbackSerializer,
)

logger = logging.getLogger(__name__)
_SQLITE_READ_RETRY_ATTEMPTS = 3
_SQLITE_READ_RETRY_BASE_DELAY_SECONDS = 0.1
_ACTIVE_FEEDBACK_STATUSES = (
    UserFeedback.STATUS_QUEUED,
    UserFeedback.STATUS_PROCESSING,
)


def _is_sqlite_lock_error(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _stale_user_feedback_after() -> timedelta:
    raw = (os.environ.get("USER_FEEDBACK_STALE_AFTER_SECONDS", "300") or "300").strip()
    try:
        seconds = int(raw)
    except ValueError:
        seconds = 300
    return timedelta(seconds=max(30, seconds))


def _expire_stale_user_feedback(segmentation_id: str) -> int:
    cutoff = timezone.now() - _stale_user_feedback_after()
    expired_count = UserFeedback.objects.filter(
        segmentation_id=segmentation_id,
        utilized_status__in=_ACTIVE_FEEDBACK_STATUSES,
        updated_at__lt=cutoff,
    ).update(
        utilized_status=UserFeedback.STATUS_FAILED,
        updated_at=timezone.now(),
    )
    if expired_count:
        logger.warning(
            "Expired %s stale user feedback rows for segmentation %s",
            expired_count,
            segmentation_id,
        )
    return expired_count


class SegmentationUserFeedbackView(APIView):
    """List and create user feedback for a segmentation."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        _expire_stale_user_feedback(str(segmentation.id))
        feedback_qs = UserFeedback.objects.filter(segmentation=segmentation).order_by(
            "created_at"
        )

        ids_param = request.query_params.get("ids")
        if ids_param:
            ids = [item.strip() for item in ids_param.split(",") if item.strip()]
            if ids:
                feedback_qs = feedback_qs.filter(id__in=ids)

        statuses_param = request.query_params.get("utilized_statuses")
        if statuses_param:
            statuses = [
                item.strip().upper()
                for item in statuses_param.split(",")
                if item.strip()
            ]
            valid_statuses = {
                UserFeedback.STATUS_QUEUED,
                UserFeedback.STATUS_PROCESSING,
                UserFeedback.STATUS_FAILED,
                UserFeedback.STATUS_SUCCESS,
            }
            statuses = [status_value for status_value in statuses if status_value in valid_statuses]
            if statuses:
                feedback_qs = feedback_qs.filter(utilized_status__in=statuses)
        for attempt in range(_SQLITE_READ_RETRY_ATTEMPTS):
            try:
                serializer = UserFeedbackSerializer(feedback_qs, many=True)
                return Response(serializer.data, status=status.HTTP_200_OK)
            except OperationalError as exc:
                if (
                    not _is_sqlite_lock_error(exc)
                    or attempt >= _SQLITE_READ_RETRY_ATTEMPTS - 1
                ):
                    logger.exception(
                        "Failed reading user feedback for segmentation %s", seg_id
                    )
                    return Response(
                        {"error": "Database busy while reading user feedback"},
                        status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    )
                delay = _SQLITE_READ_RETRY_BASE_DELAY_SECONDS * (attempt + 1)
                time.sleep(delay)

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        create_serializer = UserFeedbackCreateSerializer(data=request.data)
        create_serializer.is_valid(raise_exception=True)

        feedback = UserFeedback.objects.create(
            segmentation=segmentation,
            **create_serializer.to_feedback_kwargs(),
        )

        try:
            job = Job.enqueue(
                job_type="process_user_feedback",
                payload={
                    "user_feedback_id": str(feedback.id),
                    "segmentation_id": str(segmentation.id),
                },
                priority="high",
                resource_class="cpu",
                queue_name=QUEUE_P1_INTERACTIVE,
                max_attempts=1,
                tags=[
                    f"segmentation:{segmentation.id}",
                    f"user_feedback:{feedback.id}",
                ],
            )
        except Exception as exc:
            logger.exception(
                "Failed to enqueue feedback processing for %s: %s",
                feedback.id,
                exc,
            )
            feedback.utilized_status = UserFeedback.STATUS_FAILED
            feedback.save(update_fields=["utilized_status", "updated_at"])
            serializer = UserFeedbackSerializer(feedback)
            return Response(
                {
                    **serializer.data,
                    "detail": "Feedback captured, but failed to enqueue processing.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        serializer = UserFeedbackSerializer(feedback)
        return Response(
            {
                **serializer.data,
                "job_id": str(job.id),
            },
            status=status.HTTP_201_CREATED,
        )

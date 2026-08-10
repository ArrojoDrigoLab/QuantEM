from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.models import Asset
from quantem.jobs.apps import scheduler_is_running
from quantem.jobs.constants import (
    JOB_TYPE_LABELS,
    QUEUE_DISPLAY_NAMES,
    QUEUE_PRIORITY_ORDER,
)
from quantem.jobs.failure_reconcile import reconcile_domain_objects_for_removed_job
from quantem.jobs.models import Job
from quantem.jobs.serializers import JobCreateSerializer, JobSerializer
from quantem.segmentation.models import ImageSegmentation

DONE_JOB_STATUSES = ("SUCCESS", "FAILED", "CANCELLED")
FAILED_JOB_STATUSES = ("FAILED", "CANCELLED")
MANUAL_RETRY_JOB_STATUSES = FAILED_JOB_STATUSES
TERMINAL_JOBS_LIMIT = 100


def _build_task_label(job: Job) -> str:
    return JOB_TYPE_LABELS.get(job.type, f"Job: {job.type}")


def _queue_sort_key(queue_name: str) -> tuple[int, str]:
    return (QUEUE_PRIORITY_ORDER.get(queue_name, 99), queue_name)


def _collect_context_maps(
    jobs: list[Job],
) -> tuple[dict[str, Asset], dict[str, ImageSegmentation]]:
    asset_ids: set[str] = set()
    segmentation_ids: set[str] = set()
    for job in jobs:
        payload = job.payload_json or {}
        asset_id = payload.get("asset_id")
        if asset_id:
            asset_ids.add(str(asset_id))
        segmentation_id = payload.get("segmentation_id")
        if segmentation_id:
            segmentation_ids.add(str(segmentation_id))

    assets_by_id = {
        str(asset.id): asset for asset in Asset.objects.filter(id__in=asset_ids)
    }
    segmentations_by_id = {
        str(segmentation.id): segmentation
        for segmentation in ImageSegmentation.objects.filter(
            id__in=segmentation_ids
        ).select_related("asset", "segmentation_type")
    }
    return assets_by_id, segmentations_by_id


def _serialize_job_status(
    job: Job,
    assets_by_id: dict[str, Asset],
    segmentations_by_id: dict[str, ImageSegmentation],
) -> dict:
    payload = job.payload_json or {}
    asset = None
    segmentation = None
    asset_id = payload.get("asset_id")
    segmentation_id = payload.get("segmentation_id")
    if segmentation_id:
        segmentation = segmentations_by_id.get(str(segmentation_id))
        if segmentation:
            asset = segmentation.asset
    if asset is None and asset_id:
        asset = assets_by_id.get(str(asset_id))

    segmentation_payload = None
    if segmentation:
        segmentation_payload = {
            "id": str(segmentation.id),
            "name": segmentation.segmentation_type.long_name,
            "internal_name": segmentation.segmentation_type.internal_name,
            "short_name": segmentation.segmentation_type.short_name,
            "long_name": segmentation.segmentation_type.long_name,
        }

    image_payload = None
    if asset:
        image_payload = {"id": str(asset.id), "display_name": asset.display_name}

    return {
        "id": str(job.id),
        "type": job.type,
        "task_label": _build_task_label(job),
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "cancel_requested": job.cancel_requested,
        "queue_name": job.queue_name,
        "resource_class": job.resource_class,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "image": image_payload,
        "segmentation": segmentation_payload,
    }


def _worker_status() -> dict:
    """Whether anything in this process will actually drain the queue.

    Single-user desktop: the scheduler runs on a thread inside the server
    process. There is no PID file, no process probing, and no way to restart a
    worker from the API — that is operator tooling and a shipped app has no
    operator.
    """
    return {"scheduler_in_process": scheduler_is_running()}


class JobSubmitView(APIView):
    def post(self, request):
        serializer = JobCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        job = Job.enqueue(
            job_type=data["type"],
            payload=data["payload"],
            priority=data["priority"],
            resource_class=data["resource_class"],
            queue_name=data["queue_name"],
            max_attempts=data["max_attempts"],
            tags=data.get("tags", []),
        )
        return Response(JobSerializer(job).data, status=status.HTTP_201_CREATED)


class JobDetailView(APIView):
    def get(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        return Response(JobSerializer(job).data)

    def delete(self, request, job_id):
        """Remove a queued job — and conclude whatever it was carrying.

        This is the only exit a queued job has: ``JobCancelView`` refuses
        anything that is not RUNNING with a 409. It also *deletes the row*,
        which puts the job beyond every safety net in
        :mod:`quantem.jobs.failure_reconcile` --
        ``JobScheduler._recover_orphaned_jobs`` iterates ``status="RUNNING"``
        and cannot reach a row that no longer exists. So the reconciliation has
        to happen here, on the way out, or never.

        Reached from one click: "Remove" on a queued job in ``JobQueueSidebar``,
        or "Cancel all" on a whole queue.
        """
        job = get_object_or_404(Job, id=job_id)
        if job.status not in {"PENDING", "RETRY"}:
            return Response(
                {"detail": "Only queued jobs can be deleted."},
                status=status.HTTP_409_CONFLICT,
            )
        job_type = job.type
        payload = job.payload_json or {}
        with transaction.atomic():
            job.delete()
            reconcile_domain_objects_for_removed_job(job_type, payload)
        return Response(status=status.HTTP_204_NO_CONTENT)


class JobCancelView(APIView):
    def post(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        if job.status != "RUNNING":
            return Response(
                {"detail": "Only running jobs can be cancelled."},
                status=status.HTTP_409_CONFLICT,
            )
        job.cancel_requested = True
        job.message = "cancelling"
        job.save(update_fields=["cancel_requested", "message"])
        return Response({"status": "cancel_requested"}, status=status.HTTP_200_OK)


class JobRetryView(APIView):
    def post(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        if job.status not in MANUAL_RETRY_JOB_STATUSES:
            detail = (
                f"Only failed or cancelled jobs can be retried; this one is "
                f"{job.status.lower()}."
            )
            if job.status == "RUNNING":
                # Naming the way out matters: a running job whose worker is gone
                # is exactly the state a user reaches this endpoint from, and
                # cancelling it is what makes it retryable.
                detail += (
                    " Cancel it first (POST /api/jobs/"
                    f"{job.id}/cancel/); a job whose worker is gone is cancelled "
                    "within a few seconds."
                )
            return Response(
                {"detail": detail, "job_id": str(job.id), "job_status": job.status},
                status=status.HTTP_409_CONFLICT,
            )

        now = timezone.now()
        Job.objects.filter(id=job.id).update(
            status="PENDING",
            progress=0.0,
            message="retry queued",
            cancel_requested=False,
            started_at=None,
            finished_at=None,
            next_run_at=now,
            attempts=0,
            result_json=None,
            error_traceback="",
            updated_at=now,
        )
        return Response(
            {
                "status": "queued",
                "job_id": str(job.id),
            },
            status=status.HTTP_200_OK,
        )


class JobQueueStatusView(APIView):
    def get(self, request):
        worker_status = _worker_status()
        running_jobs = list(
            Job.objects.filter(status="RUNNING").order_by("started_at", "created_at")
        )
        queued_jobs = list(
            Job.objects.filter(status__in=["PENDING", "RETRY"]).order_by("created_at")
        )
        failed_jobs = list(
            Job.objects.filter(status__in=FAILED_JOB_STATUSES).order_by(
                "-finished_at", "-updated_at", "-created_at"
            )[:TERMINAL_JOBS_LIMIT]
        )
        completed_jobs = list(
            Job.objects.filter(status="SUCCESS").order_by(
                "-finished_at", "-updated_at", "-created_at"
            )[:TERMINAL_JOBS_LIMIT]
        )

        all_jobs = running_jobs + queued_jobs + failed_jobs + completed_jobs
        assets_by_id, segmentations_by_id = _collect_context_maps(all_jobs)

        running_payload = [
            _serialize_job_status(job, assets_by_id, segmentations_by_id)
            for job in running_jobs
        ]
        failed_payload = [
            _serialize_job_status(job, assets_by_id, segmentations_by_id)
            for job in failed_jobs
        ]
        completed_payload = [
            _serialize_job_status(job, assets_by_id, segmentations_by_id)
            for job in completed_jobs
        ]

        queues: dict[str, list[dict]] = {}
        for job in queued_jobs:
            queues.setdefault(job.queue_name, []).append(
                _serialize_job_status(job, assets_by_id, segmentations_by_id)
            )

        queue_payload = [
            {
                "queue_name": queue_name,
                "display_name": QUEUE_DISPLAY_NAMES.get(queue_name, queue_name.title()),
                "pending": jobs,
            }
            for queue_name, jobs in queues.items()
        ]
        queue_payload.sort(key=lambda entry: _queue_sort_key(entry["queue_name"]))

        return Response(
            {
                "running": running_payload,
                "queues": queue_payload,
                "failed": failed_payload,
                "completed": completed_payload,
                "worker": worker_status,
                "generated_at": timezone.now(),
            },
            status=status.HTTP_200_OK,
        )


class JobClearDoneView(APIView):
    def post(self, request):
        done_jobs = Job.objects.filter(status__in=DONE_JOB_STATUSES)
        deleted = done_jobs.count()
        if deleted > 0:
            done_jobs.delete()
        return Response(
            {
                "deleted": deleted,
                "cleared_statuses": list(DONE_JOB_STATUSES),
            },
            status=status.HTTP_200_OK,
        )

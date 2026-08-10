import uuid

from django.db import models
from django.utils import timezone


class Job(models.Model):
    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("RUNNING", "Running"),
        ("SUCCESS", "Success"),
        ("FAILED", "Failed"),
        ("CANCELLED", "Cancelled"),
        ("RETRY", "Retry"),
    ]
    PRIORITY_CHOICES = [
        ("high", "High"),
        ("default", "Default"),
    ]
    RESOURCE_CHOICES = [
        ("cpu", "CPU"),
        ("gpu", "GPU"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    type = models.CharField(max_length=128)
    priority = models.CharField(
        max_length=16, choices=PRIORITY_CHOICES, default="default"
    )
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="PENDING")
    progress = models.FloatField(default=0.0)
    # Byte-level progress, for jobs whose work is moving bytes (model pack
    # downloads). Null means "this job does not report bytes", which is not the
    # same as 0 of anything: the percent bar above is the generic surface, and
    # these two are what lets the Models screen show "435 of 1243 MB" for an
    # in-flight install without parsing it back out of ``message``.
    progress_current_bytes = models.BigIntegerField(null=True, blank=True)
    progress_total_bytes = models.BigIntegerField(null=True, blank=True)
    message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    attempts = models.PositiveIntegerField(default=0)
    max_attempts = models.PositiveIntegerField(default=3)
    next_run_at = models.DateTimeField(default=timezone.now)
    payload_json = models.JSONField(default=dict)
    result_json = models.JSONField(null=True, blank=True)
    error_traceback = models.TextField(blank=True)
    cancel_requested = models.BooleanField(default=False)
    resource_class = models.CharField(
        max_length=16, choices=RESOURCE_CHOICES, default="cpu"
    )
    queue_name = models.CharField(max_length=64, default="default")
    tags = models.JSONField(default=list, blank=True)
    claimed_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "priority"]),
            models.Index(fields=["next_run_at"]),
            models.Index(fields=["resource_class"]),
            models.Index(fields=["queue_name"]),
        ]

    @classmethod
    def enqueue(
        cls,
        *,
        job_type: str,
        payload: dict,
        priority: str = "default",
        resource_class: str = "cpu",
        queue_name: str = "default",
        max_attempts: int = 3,
        tags: list | None = None,
    ) -> "Job":
        return cls.objects.create(
            type=job_type,
            payload_json=payload,
            priority=priority,
            resource_class=resource_class,
            queue_name=queue_name,
            max_attempts=max_attempts,
            tags=tags or [],
        )


class JobLog(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="logs")
    timestamp = models.DateTimeField(auto_now_add=True)
    level = models.CharField(max_length=16, default="info")
    message = models.TextField()

    class Meta:
        ordering = ["timestamp"]


class JobArtifact(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="artifacts")
    kind = models.CharField(max_length=64)
    file_path = models.CharField(max_length=1024)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-id"]


class StorageArtifactLease(models.Model):
    STATUS_ACTIVE = "ACTIVE"
    STATUS_RELEASED = "RELEASED"
    STATUS_CHOICES = [
        (STATUS_ACTIVE, "Active"),
        (STATUS_RELEASED, "Released"),
    ]

    artifact_path = models.CharField(max_length=1024, unique=True)
    job = models.ForeignKey(
        Job,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="storage_artifact_leases",
    )
    status = models.CharField(
        max_length=16,
        choices=STATUS_CHOICES,
        default=STATUS_ACTIVE,
    )
    acquired_at = models.DateTimeField(default=timezone.now)
    expires_at = models.DateTimeField()
    released_at = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["artifact_path"]
        indexes = [
            models.Index(fields=["status", "expires_at"]),
            models.Index(fields=["job", "status"]),
        ]

import uuid

from django.db import models
from django.utils import timezone

from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)
from quantem.jobs.tile_plan import planned_units_for

#: Statuses in which a job still has work ahead of it. A run wave stays open
#: while any of its jobs is in one of these.
OPEN_JOB_STATUSES = ("PENDING", "RETRY", "RUNNING")

#: Statuses in which a job will never do any more work.
TERMINAL_JOB_STATUSES = ("SUCCESS", "FAILED", "CANCELLED")

#: Job types that report their progress in countable units of work. Anything not
#: listed here is enqueued with null unit columns: it is honest for a job that
#: cannot say how much work it will do to say nothing rather than to claim 0 of
#: 0, which reads as finished.
UNIT_PROGRESS_JOB_TYPES = frozenset(
    {
        JOB_TYPE_RUN_SEGMENTATION_ROI,
        JOB_TYPE_RUN_SEGMENTATION_FULL,
        # One row, one denominator: the tiles of every organelle in the run,
        # added up at enqueue. See :func:`quantem.jobs.tile_plan.planned_units_for`.
        JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
        # Training **steps**, across every round the fine-tune will run. Not
        # tiles: see BATCH_ROLLUP_JOB_TYPES below for why that difference had to
        # be written down rather than left implied by the label.
        JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    }
)

#: The subset of the above that takes part in a per-image **run wave** rollup
#: ("X of Y tiles across this image").
#:
#: Counting units and belonging to a wave used to be the same membership, and
#: adding a job that counts *steps* broke the identity. A fine-tune launched
#: from a labeling view carries that view's ``segmentation_id``, so it would
#: have joined the image's run wave and mixed steps into a tile total.
#: ``aggregate_batch_progress`` refuses a wave with two unit labels in it, so
#: the visible effect would have been the whole image's run progress
#: disappearing for the length of a fine-tune. A fine-tune is not one of the
#: runs the user pressed go on for this image; it is its own thing, with its own
#: bar.
BATCH_ROLLUP_JOB_TYPES = frozenset(
    {
        JOB_TYPE_RUN_SEGMENTATION_ROI,
        JOB_TYPE_RUN_SEGMENTATION_FULL,
        JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
    }
)

# --- progress_stage vocabulary ------------------------------------------------
# Machine keys, not prose. The UI maps them to copy; nothing here is rendered
# verbatim and nothing here is ever a shell command (invariant I-12).

#: Waiting for a worker.
STAGE_QUEUED = "queued"
#: Resolving and loading model weights into the worker process. This is the
#: 4-20 s of dead air before tile 1 that used to read as a frozen 5 %.
STAGE_LOADING_MODEL = "loading_model"
#: Sliding windows over the image. This is the stage that reports tiles.
STAGE_INFERENCE = "inference"
#: Saving the result that makes the threshold preview available.
STAGE_PREPARING_THRESHOLD = "preparing_threshold"
#: Turning the probability map into objects.
STAGE_EXTRACTING = "extracting"
#: Writing objects and overlays.
STAGE_SAVING = "saving"
#: Moving a model pack over the network. Deliberately a *different* stage from
#: everything above, and it reports **bytes**, never tiles: "downloading 1.2 GB"
#: and "segmenting 858 tiles" are different facts about the machine and the
#: owner asked for them as separate indicators.
STAGE_DOWNLOADING_MODEL = "downloading_model"

# --- fine-tuning ---
#: Reading the chosen images' annotations and cutting them into training tiles.
STAGE_PREPARING = "preparing"
#: Taking optimiser steps. This is the stage that reports steps.
STAGE_TRAINING = "training"
#: Running the just-trained model over the held-out area and scoring it. One
#: round is one training pass plus one of these.
STAGE_EVALUATING = "evaluating"

PROGRESS_STAGES = (
    STAGE_QUEUED,
    STAGE_LOADING_MODEL,
    STAGE_INFERENCE,
    STAGE_PREPARING_THRESHOLD,
    STAGE_EXTRACTING,
    STAGE_SAVING,
    STAGE_DOWNLOADING_MODEL,
    STAGE_PREPARING,
    STAGE_TRAINING,
    STAGE_EVALUATING,
)

#: ``progress_unit_label`` value for sliding-window inference. Singular, lower
#: case; the UI pluralises.
UNIT_TILE = "tile"

#: ``progress_unit_label`` for head training. One optimiser step. Counted across
#: every round of a cross-validated run, so the bar is monotone over the whole
#: job rather than restarting per fold.
UNIT_STEP = "step"


def batch_key_for_payload(payload: dict | None) -> str | None:
    """The image a job's payload is about, as a batch grouping key.

    Segmentation payloads already carry ``asset_id``; the ``segmentation_id``
    fallback exists for callers that do not, and costs one indexed lookup at
    enqueue time. Returns None when the payload is not about an image, which is
    not an error -- it means this job is not part of a per-image run wave.
    """
    payload = payload or {}
    asset_id = payload.get("asset_id")
    if asset_id:
        return str(asset_id)
    segmentation_id = payload.get("segmentation_id")
    if not segmentation_id:
        return None
    try:
        from quantem.segmentation.models import (  # noqa: PLC0415 -- app-registry cycle
            ImageSegmentation,
        )

        asset_id = (
            ImageSegmentation.objects.filter(id=segmentation_id)
            .values_list("asset_id", flat=True)
            .first()
        )
    except Exception:
        return None
    return str(asset_id) if asset_id else None


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
    priority = models.CharField(max_length=16, choices=PRIORITY_CHOICES, default="default")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="PENDING")
    progress = models.FloatField(default=0.0)
    # Byte-level progress, for jobs whose work is moving bytes (model pack
    # downloads). Null means "this job does not report bytes", which is not the
    # same as 0 of anything: the percent bar above is the generic surface, and
    # these two are what lets the Models screen show "435 of 1243 MB" for an
    # in-flight install without parsing it back out of ``message``.
    progress_current_bytes = models.BigIntegerField(null=True, blank=True)
    progress_total_bytes = models.BigIntegerField(null=True, blank=True)
    # Countable units of work, for jobs that know how much work there is before
    # they start. For a segmentation run that is **tiles**: the sliding-window
    # plan is laid out before the first forward pass, so "531 of 858 tiles" is a
    # fact and not an extrapolation from a percentage.
    #
    # Null means "this job does not count units", which is not 0 of anything.
    # These are deliberately separate from the byte columns above: a model
    # download moves bytes and a run walks tiles, and calling both "progress"
    # would be a lie about what the machine is doing.
    progress_units_done = models.PositiveIntegerField(null=True, blank=True)
    progress_units_total = models.PositiveIntegerField(null=True, blank=True)
    #: Singular lower-case noun for one unit, e.g. ``"tile"``. Empty when the
    #: job reports no units.
    progress_unit_label = models.CharField(max_length=32, blank=True, default="")
    #: Which phase of the job is running; one of :data:`PROGRESS_STAGES`.
    progress_stage = models.CharField(max_length=32, blank=True, default="")
    #: Structured extras for the current stage (``eta_seconds``, the tile grid,
    #: the pack being loaded). Machine-readable; never rendered verbatim.
    progress_detail_json = models.JSONField(default=dict, blank=True)
    #: The run wave this job belongs to, or "" for a job that is not part of
    #: one. See :meth:`Job.resolve_batch`.
    batch_id = models.CharField(max_length=128, blank=True, default="", db_index=True)
    #: Position of this job within its wave, in enqueue order, starting at 0.
    batch_seq = models.PositiveIntegerField(default=0)
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
    resource_class = models.CharField(max_length=16, choices=RESOURCE_CHOICES, default="cpu")
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
            models.Index(fields=["batch_id", "status"]),
        ]

    @classmethod
    def resolve_batch(cls, job_type: str, payload: dict | None) -> tuple[str, int]:
        """The run wave a new job joins, and its position in it.

        A **run wave** is the set of runs the user thinks of as one thing: tick
        mitochondria and nucleus and press go, and both belong to one wave, so
        the app can answer "across all organelle runs for this image, X of Y
        tiles done". The rule is deliberately written at enqueue time rather
        than inferred at read time, because reading cannot distinguish "these
        two overlapped" from "these two were separate runs that happened to be
        adjacent":

        * if any run for this image is still open (PENDING / RETRY / RUNNING),
          the new job **joins that wave** -- covering both "ticked three at
          once" and "started a second organelle while the first was running";
        * otherwise it **starts a new wave**, so yesterday's finished runs never
          count towards today's aggregate.

        A job that failed or was cancelled stays in its wave. It was part of the
        work the user started, and the rollup has to be able to say so rather
        than quietly forget it (see
        :func:`quantem.jobs.serializers.aggregate_batch_progress`).

        Returns ``("", 0)`` for job types that are not part of a run wave --
        which is not the same set as the types that count units; see
        :data:`BATCH_ROLLUP_JOB_TYPES`.
        """
        if job_type not in BATCH_ROLLUP_JOB_TYPES:
            return "", 0
        key = batch_key_for_payload(payload)
        if not key:
            return "", 0
        prefix = f"asset:{key}:"
        open_batch = (
            cls.objects.filter(batch_id__startswith=prefix, status__in=OPEN_JOB_STATUSES)
            .order_by("created_at")
            .values_list("batch_id", flat=True)
            .first()
        )
        batch_id = open_batch or f"{prefix}{uuid.uuid4().hex[:12]}"
        return batch_id, cls.objects.filter(batch_id=batch_id).count()

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
        batch_id: str | None = None,
    ) -> "Job":
        """Create a queued job.

        ``batch_id`` overrides the wave :meth:`resolve_batch` would pick. It
        exists for a caller that already knows the grouping -- a set-wide
        re-apply over forty images -- and must not be used to smuggle unrelated
        jobs into one rollup.

        **The job is created carrying its tiling plan**, not waiting for the run
        to discover it (:func:`quantem.jobs.tile_plan.planned_units_for`). That
        is what lets a queued run say how big it is, and what makes the wave
        rollup count work the user has asked for rather than only work that has
        already begun -- see
        :func:`quantem.jobs.serializers.aggregate_batch_progress`, which
        reported a three-run wave as 100 % complete while two of the three runs
        had never started.
        """
        if batch_id is None:
            batch_id, batch_seq = cls.resolve_batch(job_type, payload)
        else:
            batch_seq = cls.objects.filter(batch_id=batch_id).count()
        planned = planned_units_for(job_type, payload)
        units_total, unit_label = planned if planned else (None, "")
        return cls.objects.create(
            type=job_type,
            payload_json=payload,
            priority=priority,
            resource_class=resource_class,
            queue_name=queue_name,
            max_attempts=max_attempts,
            tags=tags or [],
            batch_id=batch_id,
            batch_seq=batch_seq,
            # Null when the plan is not knowable, which reads as "this job does
            # not count units" -- never 0 of 0, which reads as finished.
            progress_units_total=units_total,
            progress_units_done=0 if units_total is not None else None,
            progress_unit_label=unit_label,
            progress_stage=STAGE_QUEUED if units_total is not None else "",
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

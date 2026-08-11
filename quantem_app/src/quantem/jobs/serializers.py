from rest_framework import serializers

from quantem.jobs.constants import (
    ALLOWED_JOB_TYPES,
    ALLOWED_QUEUE_NAMES,
    JOB_DEFAULTS,
)
from quantem.jobs.models import (
    OPEN_JOB_STATUSES,
    Job,
    JobArtifact,
    JobLog,
)


def _percent(done: int, total: int) -> float | None:
    if not total:
        return None
    return round(min(max(100.0 * done / total, 0.0), 100.0), 1)


def download_progress(job: Job) -> dict | None:
    """Bytes moving over the network for this job, or None.

    Structurally separate from the tile numbers on purpose. A caller cannot
    accidentally render "downloading 1.2 GB" as segmentation progress, because
    the two never share a field.
    """
    total = job.progress_total_bytes
    if total is None and job.progress_current_bytes is None:
        return None
    current = int(job.progress_current_bytes or 0)
    return {
        "current_bytes": current,
        "total_bytes": int(total) if total is not None else None,
        "percent": _percent(current, int(total or 0)),
    }


def unit_progress(job: Job) -> dict | None:
    """Countable work for this job -- tiles -- or None if it counts none."""
    total = job.progress_units_total
    if total is None:
        return None
    done = int(job.progress_units_done or 0)
    detail = job.progress_detail_json or {}
    return {
        "done": done,
        "total": int(total),
        "label": job.progress_unit_label or "",
        "percent": _percent(done, int(total)),
        "stage": job.progress_stage or "",
        "eta_seconds": detail.get("eta_seconds"),
    }


def run_legs(job: Job) -> list[dict] | None:
    """The organelles inside one run, or None when the job is not one of those.

    A run over three organelles is one job row with one tile count, which is
    what makes the whole-image bar honest -- and it would make the *per*-organelle
    lines disappear, because the run panel builds one line per job. So the run
    publishes its legs as it goes and they are carried here, structurally
    separate from the job's own numbers, one entry per organelle with its own
    status and its own share of the tiles.

    Read from ``progress_detail_json``, which is machine-readable and never
    rendered verbatim; the names are the segmentation types' own long names, the
    same words the rest of the app calls those organelles.
    """
    legs = (job.progress_detail_json or {}).get("legs")
    if not isinstance(legs, list) or not legs:
        return None
    rows = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        total = int(leg.get("units_total") or 0)
        done = min(int(leg.get("units_done") or 0), total) if total else 0
        rows.append(
            {
                "segmentation_id": leg.get("segmentation_id") or "",
                "name": leg.get("name") or "",
                "status": leg.get("status") or "PENDING",
                "units_done": done,
                "units_total": total,
                "unit_label": leg.get("unit_label") or "",
                "percent": _percent(done, total),
            }
        )
    return rows or None


def overall_percent(job: Job, units: dict | None = None) -> float:
    """The one percentage the wire quotes for this job.

    ``Job.progress`` and ``unit_progress.percent`` are two divisors over one
    run. ``progress`` is laid out over the tiles *plus* the stages either side
    of them -- 57 "units" for a 56-tile plan, the extra one standing for
    finding and saving the objects -- so the same instant reads ``7/57 =
    12.3`` on one field and ``7/56 = 12.5`` on the other. Measured on the wire
    throughout a real run. Nothing renders both today, and that is the only
    reason it has not been on screen: two numbers for one fact, one careless
    field access apart.

    So a job that counts units quotes **one** number, and it is the tiling
    plan's: that is the divisor the tile line has to use (``32 of 56 tiles``
    beside 57ths would be the same defect in one sentence), and 57 is not a
    count of anything -- the ``+1`` is a placeholder for a phase, not a unit of
    work. When the tiles are walked the run is at 100 % of its tiles and its
    ``progress_stage`` says what is still happening; that is exactly what the
    run panel already draws, a full bar with "finding objects" beside it.

    A finished job is 100 %, whatever its tile count says: a run that reused an
    earlier result walks no tiles and is still complete.

    The stored column is untouched -- this is what the API answers, not what
    the run writes.
    """
    if job.status == "SUCCESS":
        return 100.0
    if units is None:
        units = unit_progress(job)
    if units is not None and units["percent"] is not None:
        return float(units["percent"])
    return round(float(job.progress or 0.0), 1)


def aggregate_batch_progress(jobs) -> dict | None:
    """Roll a run wave up into one answer: X of Y tiles across this image.

    ``jobs`` is every job sharing a ``batch_id`` (see
    :meth:`quantem.jobs.models.Job.resolve_batch`). **Every** job in the wave
    takes part in the run counts, whether or not it has begun and whether or
    not it knows its own size, because the question this answers is "how far
    through the thing I pressed go on", and a run the user asked for is part of
    that thing from the moment they asked.

    Three cases the arithmetic has to get right, and how:

    * **Runs that started at different times.** Nothing here is time-based. A
      queued run carries its full planned total (written at enqueue by
      :func:`quantem.jobs.tile_plan.planned_units_for`), so the denominator is
      the whole wave from the moment the last run joins it. This used to be a
      promise the code did not keep: the filter below was
      ``progress_units_total is not None`` and that column was first written by
      the run itself, so a wave of three runs over one image reported the first
      one as the whole of the work -- ``runs_total 1``, ``percent 100.0`` --
      while two runs had not started and one of them would fail.
    * **A run that failed**, and
    * **a run that was cancelled.** Those tiles will never be walked. They stay
      in the denominator: the user asked for 118 tiles of work, 25 happened,
      and 25 of 118 is the answer to how much of it was done. Dropping them
      instead -- which is what this function used to do -- lets the bar reach
      100 % on a wave that ran a fifth of itself, and makes the percentage jump
      *upwards* the moment a run dies. They are reported separately as
      ``units_abandoned`` and the run is counted in ``runs_failed`` /
      ``runs_cancelled``, so the UI can say out loud that part of the wave will
      never happen and that the bar is not going to fill.

    ``units_done`` only ever grows for a fixed set of jobs; ``units_total``
    grows when the user adds a run to a wave that is still open, which is real
    new work and is why the wave also reports ``runs_total``.

    Returns None when no job in the wave counts units at all -- a wave with no
    countable work has no rollup, which is not the same as a rollup of zero.
    """
    jobs = list(jobs)
    planned = [job for job in jobs if job.progress_units_total is not None]
    if not planned:
        return None

    labels = {job.progress_unit_label or "" for job in planned} - {""}
    if len(labels) > 1:
        # Summing tiles with something that is not tiles would be a number with
        # no meaning. Refuse rather than invent one.
        return None

    units_done = 0
    units_total = 0
    units_abandoned = 0
    runs_unplanned = 0
    counts = {
        "PENDING": 0,
        "RETRY": 0,
        "RUNNING": 0,
        "SUCCESS": 0,
        "FAILED": 0,
        "CANCELLED": 0,
    }
    etas: list[float] = []
    unknown_eta = False
    runs = []

    for job in sorted(jobs, key=lambda j: (j.batch_seq, j.created_at)):
        total = int(job.progress_units_total or 0)
        # Clamped: a run whose plan was corrected downwards mid-flight must not
        # push the wave past 100 %.
        done = min(int(job.progress_units_done or 0), total) if total else 0
        units_done += done
        units_total += total
        if job.progress_units_total is None:
            runs_unplanned += 1
        counts[job.status] = counts.get(job.status, 0) + 1
        if job.status in ("FAILED", "CANCELLED") and done < total:
            units_abandoned += total - done
        if job.status in OPEN_JOB_STATUSES:
            eta = (job.progress_detail_json or {}).get("eta_seconds")
            if job.status == "RUNNING" and isinstance(eta, (int, float)):
                etas.append(float(eta))
            else:
                # A queued run's wait depends on the queue, not on itself.
                unknown_eta = True
        runs.append(
            {
                "job_id": str(job.id),
                "segmentation_id": (job.payload_json or {}).get("segmentation_id"),
                "status": job.status,
                "batch_seq": job.batch_seq,
                "units_done": done,
                "units_total": total,
                "stage": job.progress_stage or "",
            }
        )

    units_reachable = max(units_total - units_abandoned, 0)
    open_runs = counts["PENDING"] + counts["RETRY"] + counts["RUNNING"]
    return {
        "batch_id": planned[0].batch_id,
        "unit_label": next(iter(labels), ""),
        "units_done": units_done,
        "units_total": units_total,
        "units_abandoned": units_abandoned,
        # How many of the planned tiles can still happen. Reported because it
        # is a different and useful fact, and deliberately *not* the
        # denominator of ``percent``: see the docstring.
        "units_reachable": units_reachable,
        # None, not 0, only when the wave's size genuinely is not known -- one
        # of its runs could not be planned, so any percentage would be a
        # fraction of an unknown.
        "percent": None if runs_unplanned else _percent(units_done, units_total),
        "runs_total": len(jobs),
        "runs_unplanned": runs_unplanned,
        "runs_pending": counts["PENDING"] + counts["RETRY"],
        "runs_running": counts["RUNNING"],
        "runs_succeeded": counts["SUCCESS"],
        "runs_failed": counts["FAILED"],
        "runs_cancelled": counts["CANCELLED"],
        "complete": open_runs == 0,
        "eta_seconds": (
            None if (unknown_eta or not etas) else round(max(etas), 1)
        ),
        "runs": runs,
    }


def batch_progress_for(job: Job) -> dict | None:
    """The wave rollup for the wave ``job`` belongs to, or None."""
    if not job.batch_id:
        return None
    return aggregate_batch_progress(Job.objects.filter(batch_id=job.batch_id))


def batch_progress_map(batch_ids) -> dict[str, dict]:
    """``{batch_id: rollup}`` for several waves in one query.

    :func:`batch_progress_for` is one query per job, which is fine for
    ``GET /api/jobs/<id>/`` and is not fine for ``queue-status``, where a wave
    of four organelle runs would otherwise cost four identical queries every
    three seconds. The rollup needs *every* job in the wave -- including ones
    that finished long enough ago to be off the endpoint's terminal-job page --
    so it cannot be assembled from the rows the view already has.
    """
    wanted = sorted({str(batch_id) for batch_id in batch_ids if batch_id})
    if not wanted:
        return {}
    grouped: dict[str, list[Job]] = {batch_id: [] for batch_id in wanted}
    for job in Job.objects.filter(batch_id__in=wanted):
        grouped[str(job.batch_id)].append(job)
    rollups = {}
    for batch_id, jobs in grouped.items():
        rollup = aggregate_batch_progress(jobs)
        if rollup is not None:
            rollups[batch_id] = rollup
    return rollups


def job_progress_block(job: Job, *, batch: dict | None = None) -> dict:
    """The three kinds of progress, as the wire carries them.

    One function so that every endpoint reporting a job's progress reports the
    same fields with the same names. ``GET /api/jobs/<id>/`` and
    ``GET /api/jobs/queue-status/`` disagreeing about what a job is doing is
    exactly how tile progress came to exist on the API and nowhere on screen.

    The three stay structurally apart, as they do on the model: a percentage, a
    count of tiles, and a count of bytes are three different facts, and a caller
    that wants to draw one of them cannot reach for another by accident.

    ``progress`` is here too, and it is :func:`overall_percent` rather than the
    stored column, so that the job's percentage and its tile percentage are the
    same number rather than two that disagree by a point.
    """
    units = unit_progress(job)
    return {
        "progress": overall_percent(job, units),
        "progress_stage": job.progress_stage or "",
        "unit_progress": units,
        "download": download_progress(job),
        "batch_id": job.batch_id or "",
        "batch_progress": batch,
        # One entry per organelle when this job runs several. Null otherwise,
        # which is what keeps a single-organelle row exactly what it was.
        "run_legs": run_legs(job),
    }


class JobSerializer(serializers.ModelSerializer):
    download = serializers.SerializerMethodField()
    unit_progress = serializers.SerializerMethodField()
    batch_progress = serializers.SerializerMethodField()
    run_legs = serializers.SerializerMethodField()
    #: Not the stored column: see :func:`overall_percent`. Both endpoints answer
    #: with the same number so that a consumer reading ``progress`` and a
    #: consumer reading ``unit_progress.percent`` cannot draw two bars that
    #: disagree.
    progress = serializers.SerializerMethodField()

    def get_progress(self, job: Job) -> float:
        return overall_percent(job)

    def get_download(self, job: Job) -> dict | None:
        return download_progress(job)

    def get_unit_progress(self, job: Job) -> dict | None:
        return unit_progress(job)

    def get_batch_progress(self, job: Job) -> dict | None:
        return batch_progress_for(job)

    def get_run_legs(self, job: Job) -> list[dict] | None:
        return run_legs(job)

    class Meta:
        model = Job
        fields = [
            "id",
            "type",
            "priority",
            "status",
            "progress",
            "progress_current_bytes",
            "progress_total_bytes",
            "progress_units_done",
            "progress_units_total",
            "progress_unit_label",
            "progress_stage",
            "progress_detail_json",
            "batch_id",
            "batch_seq",
            "download",
            "unit_progress",
            "batch_progress",
            "run_legs",
            "message",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "attempts",
            "max_attempts",
            "next_run_at",
            "payload_json",
            "result_json",
            "error_traceback",
            "cancel_requested",
            "resource_class",
            "queue_name",
            "tags",
            "claimed_at",
            "heartbeat_at",
        ]


class JobCreateSerializer(serializers.Serializer):
    """Validate one enqueue request.

    ``type`` is checked twice, against two different lists, and the second
    check is the one that matters.

    :data:`~quantem.jobs.constants.ALLOWED_JOB_TYPES` is the *declaration*: the
    types this build knows the name of. It is deliberately allowed to run ahead
    of the implementation, because ``registry.job_handler`` refuses to register
    a handler for an undeclared type, so a type has to be declared before the
    package that implements it can land. During a release where several
    packages are built side by side, that list therefore contains work that
    cannot yet run.

    Validating only against the declaration would let a client enqueue one of
    those: the request would 201, a row would be written, the scheduler would
    claim it, and it would die at dispatch with "no handler registered" -- a
    failure with no cause the user can see and nothing they can do. So
    :meth:`validate_type` checks the *registry*, which is the set of types
    something can actually run, and refuses the gap at the door.
    """

    type = serializers.ChoiceField(choices=sorted(ALLOWED_JOB_TYPES))
    payload = serializers.JSONField(default=dict)
    priority = serializers.ChoiceField(choices=["high", "default"], required=False)
    resource_class = serializers.ChoiceField(
        choices=["cpu", "gpu"], required=False
    )
    queue_name = serializers.ChoiceField(
        choices=sorted(ALLOWED_QUEUE_NAMES),
        required=False,
    )
    max_attempts = serializers.IntegerField(default=3, min_value=0)
    tags = serializers.ListField(child=serializers.CharField(), required=False)

    def validate_type(self, value: str) -> str:
        # Imported here rather than at module scope: the registry is populated
        # by import side effect from ``JobsConfig.ready``, and reading it at
        # request time is what makes this reflect the handlers that are
        # actually loaded rather than the ones loaded when this module was
        # first imported.
        from quantem.jobs.registry import _HANDLERS  # noqa: PLC0415

        if value not in _HANDLERS:
            raise serializers.ValidationError(
                "This version of QuantEM cannot do that kind of work yet. "
                "Starting it would add a task that never finishes."
            )
        return value

    def validate(self, attrs):
        job_type = attrs["type"]
        defaults = JOB_DEFAULTS[job_type]
        if attrs.get("priority") is None:
            attrs["priority"] = defaults["priority"]

        resource_class = attrs.get("resource_class")
        if resource_class is None:
            attrs["resource_class"] = defaults["resource_class"]
        elif resource_class != defaults["resource_class"]:
            raise serializers.ValidationError(
                {
                    "resource_class": (
                        f"{job_type} must use resource_class={defaults['resource_class']}."
                    )
                }
            )

        queue_name = attrs.get("queue_name")
        if queue_name is None:
            attrs["queue_name"] = defaults["queue_name"]
        elif queue_name != defaults["queue_name"]:
            raise serializers.ValidationError(
                {
                    "queue_name": (
                        f"{job_type} must use queue_name={defaults['queue_name']}."
                    )
                }
            )

        return attrs


class JobLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobLog
        fields = ["timestamp", "level", "message"]


class JobArtifactSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobArtifact
        fields = ["kind", "file_path", "metadata"]

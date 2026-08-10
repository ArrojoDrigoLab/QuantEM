"""Persistent record of one quantitative analysis.

An analysis is started by an HTTP request, executed minutes later by the job
queue, and read back whenever the user reopens the results panel. Nothing in
that sequence shares memory, so the run has to be a row: it exists before the
work starts (a poller needs something to poll) and it outlives the work (a
result must be reopenable without recomputing it).

``params`` and ``results`` are stored verbatim, for the same reason the export
bundle carries a manifest -- a number in the UI has to be traceable to the
inputs that produced it.
"""

from __future__ import annotations

from pathlib import Path

from django.db import models

from quantem.assets.models import TimeStampedModel


class AnalysisRun(TimeStampedModel):
    """One analysis of one segmentation, and the export bundle it wrote."""

    STATUS_PENDING = "PENDING"
    STATUS_RUNNING = "RUNNING"
    STATUS_SUCCESS = "SUCCESS"
    STATUS_FAILED = "FAILED"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_SUCCESS, "Success"),
        (STATUS_FAILED, "Failed"),
    ]

    #: The segmentation the run is *about*: its confirmed objects supply the
    #: morphometrics and, when ``points_source`` is ``centroids``, the point set.
    #: Other segmentations may contribute compartment masks (see ``params``).
    #: The lazy reference keeps this app importable regardless of the order
    #: ``INSTALLED_APPS`` happens to load its models in.
    #:
    #: ``SET_NULL``, not ``CASCADE``: a run's numbers and its export bundle are
    #: the scientific record of an analysis that happened, and deleting the
    #: segmentation it measured (``DELETE /api/segmentations/<id>/``) does not
    #: un-happen it. Every run is created with a segmentation, so a null here
    #: has exactly one meaning -- *that segmentation was deleted* -- and the
    #: serializer says so as ``segmentation_deleted`` rather than leaving the
    #: reader to infer it from a missing id.
    segmentation = models.ForeignKey(
        "segmentation.ImageSegmentation",
        on_delete=models.SET_NULL,
        related_name="analysis_runs",
        null=True,
        blank=True,
    )

    status = models.CharField(
        max_length=16,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True,
    )
    #: Experimental group label, e.g. "fasted". Free text: QuantEM does not know
    #: the user's design and will not invent one. Group-level statistics are the
    #: unweighted mean over the runs sharing this label.
    group = models.CharField(max_length=255, blank=True, default="")

    #: The validated, normalised request. See
    #: :func:`quantem.analysis.loaders.normalise_params`.
    params = models.JSONField(default=dict, blank=True)
    #: What :func:`quantem.analysis.service.run_analysis` returned, minus the
    #: per-object rows -- those are in ``objects.csv``, which is the file a paper
    #: should cite. Keeping tens of thousands of rows in a JSON column would make
    #: every list query expensive to serve a table nobody reads on screen.
    results = models.JSONField(default=dict, blank=True)

    #: Absolute path to the export bundle. Absolute rather than relative because
    #: the user is told this path so they can open it in Excel.
    export_dir = models.CharField(max_length=1024, blank=True, default="")

    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["segmentation", "-created_at"]),
            models.Index(fields=["status"]),
        ]

    @property
    def export_path(self) -> Path | None:
        """The export bundle as a path, or ``None`` if nothing was written."""
        return Path(self.export_dir) if self.export_dir else None

    def __str__(self) -> str:
        return f"AnalysisRun {self.id} ({self.status}) for {self.segmentation_id}"

"""Persistent batch membership; a task may be retried without duplicating its peers."""

from django.db import models

from quantem.assets.models import TimeStampedModel


class AnalysisBatch(TimeStampedModel):
    experiment = models.ForeignKey("library.Experiment", null=True, on_delete=models.SET_NULL)
    name = models.CharField(max_length=255)
    settings = models.JSONField(default=dict)
    experiment_name = models.CharField(max_length=255)


class AnalysisBatchItem(TimeStampedModel):
    batch = models.ForeignKey(AnalysisBatch, on_delete=models.CASCADE, related_name="items")
    segmentation = models.ForeignKey(
        "segmentation.ImageSegmentation", null=True, on_delete=models.SET_NULL
    )
    run = models.ForeignKey("analysis.AnalysisRun", null=True, on_delete=models.SET_NULL)
    job = models.ForeignKey("jobs.Job", null=True, on_delete=models.SET_NULL)
    identity = models.JSONField(default=dict)
    params = models.JSONField(default=dict)
    fingerprint = models.CharField(max_length=64, blank=True)
    skip_reason = models.TextField(blank=True)
    reused = models.BooleanField(default=False)

"""Optional sample identity, independent of overlapping dataset membership."""

from django.db import models

from quantem.assets.models import TimeStampedModel


class ImageWorkflowMetadata(TimeStampedModel):
    asset = models.OneToOneField(
        "assets.Asset", on_delete=models.CASCADE, related_name="workflow_metadata"
    )
    sample_id = models.CharField(max_length=255, blank=True, default="")
    group = models.CharField(max_length=255, blank=True, default="")

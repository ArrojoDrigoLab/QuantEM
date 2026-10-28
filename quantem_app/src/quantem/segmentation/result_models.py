"""Saved model/threshold results, independent of annotation completion."""

from django.db import models

from quantem.assets.models import TimeStampedModel


class ModelApplicationBatch(TimeStampedModel):
    experiment = models.ForeignKey("library.Experiment", on_delete=models.CASCADE)
    settings = models.JSONField(default=dict)
    items = models.JSONField(default=list)


class SavedSegmentationResult(TimeStampedModel):
    segmentation = models.ForeignKey(
        "segmentation.ImageSegmentation", on_delete=models.CASCADE, related_name="saved_results"
    )
    name = models.CharField(max_length=255)
    source_model = models.CharField(max_length=100, blank=True)
    adapter_id = models.CharField(max_length=36, blank=True)
    threshold = models.FloatField(null=True, blank=True)
    probability_map = models.ForeignKey(
        "segmentation.ProbabilityMap",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="saved_results",
    )
    provenance = models.JSONField(default=dict, blank=True)
    object_count = models.PositiveIntegerField(default=0)
    full_image = models.BooleanField(default=False)
    is_canonical = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        constraints = [
            models.UniqueConstraint(
                fields=["segmentation"],
                condition=models.Q(is_canonical=True),
                name="one_canonical_segmentation_result",
            ),
        ]

import uuid

from django.db import models


class TimeStampedModel(models.Model):
    """Abstract base model with UUID primary key and timestamps."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


PREPROCESS_STAGE_CHOICES = [
    ("NONE", "Not started"),
    ("ENCODING", "Encoding image / metadata"),
    ("FEATURES", "Computing base features"),
    ("DONE", "Completed"),
    ("FAILED", "Failed"),
    ("CANCELLED", "Cancelled"),
    ("SKIPPED", "Skipped"),
]


class Asset(TimeStampedModel):
    """One logical image (2D plane or 3D volume) in the local image library."""

    LIFECYCLE_ACTIVE = "ACTIVE"
    LIFECYCLE_DELETED = "DELETED"
    LIFECYCLE_STATUS_CHOICES = [
        (LIFECYCLE_ACTIVE, "Active"),
        (LIFECYCLE_DELETED, "Deleted"),
    ]

    lifecycle_status = models.CharField(
        max_length=16,
        choices=LIFECYCLE_STATUS_CHOICES,
        default=LIFECYCLE_ACTIVE,
        db_index=True,
    )
    deleted_at = models.DateTimeField(null=True, blank=True)
    display_name = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255, blank=True, default="")
    notes = models.TextField(blank=True, default="")
    logical_width = models.PositiveIntegerField(null=True, blank=True)
    logical_height = models.PositiveIntegerField(null=True, blank=True)
    logical_depth = models.PositiveIntegerField(null=True, blank=True)
    channels = models.PositiveSmallIntegerField(null=True, blank=True)
    bit_depth = models.PositiveSmallIntegerField(null=True, blank=True)
    # Numeric xy pixel size, in nanometres per pixel.
    #
    # A free-text ``resolution`` string ("4 nm", "2x2 nm/pixel", "~1.2nm", "") is
    # fine for a catalog facet label but cannot be used as a number: pixel size
    # gates per-organelle resampling before inference and is the unit conversion
    # behind every analysis result (areas, diameters, densities), so it is stored
    # numerically here.
    # ``None`` means "unknown" - callers must treat measurements as unavailable
    # rather than silently assuming 1 nm/px.
    pixel_size_nm = models.FloatField(null=True, blank=True)
    # Numeric z spacing (nm between stored planes) for volumes; ``None`` for 2D
    # assets and for volumes whose source carried no z spacing.
    pixel_size_nm_z = models.FloatField(null=True, blank=True)
    preprocess_stage = models.CharField(
        max_length=16,
        choices=PREPROCESS_STAGE_CHOICES,
        default="NONE",
    )
    preprocess_progress = models.FloatField(default=0.0)
    preprocess_error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["display_name", "created_at"]

    def __str__(self):
        return self.display_name


class Rendition(TimeStampedModel):
    """Typed storage artifact for one logical Asset."""

    TYPE_FULL = "FULL"
    TYPE_SUBSET = "SUBSET"
    TYPE_NGFF = "NGFF"
    TYPE_CHOICES = [
        (TYPE_FULL, "Full"),
        (TYPE_SUBSET, "Subset"),
        (TYPE_NGFF, "NGFF"),
    ]

    asset = models.ForeignKey(
        Asset,
        on_delete=models.PROTECT,
        related_name="renditions",
    )
    derived_from = models.ForeignKey(
        "self",
        on_delete=models.PROTECT,
        related_name="derived_renditions",
        null=True,
        blank=True,
    )
    type = models.CharField(max_length=16, choices=TYPE_CHOICES, db_index=True)
    storage_root = models.CharField(max_length=64, blank=True, default="")
    stored_path = models.CharField(max_length=2048, blank=True, default="")
    path_exists = models.BooleanField(default=False)
    is_directory = models.BooleanField(default=False)
    stored_width = models.PositiveIntegerField(null=True, blank=True)
    stored_height = models.PositiveIntegerField(null=True, blank=True)
    stored_depth = models.PositiveIntegerField(null=True, blank=True)
    stored_channels = models.PositiveSmallIntegerField(null=True, blank=True)
    stored_bit_depth = models.PositiveSmallIntegerField(null=True, blank=True)
    z_plane_indices = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["asset__display_name", "type", "stored_path"]
        indexes = [
            models.Index(fields=["asset", "type"], name="asset_rend_asset_type_idx"),
        ]

    def __str__(self):
        return f"{self.asset_id} {self.type}"


class ImageROI(TimeStampedModel):
    """Represents a ROI crop stored as its own image."""

    ROI_SOURCE_CHOICES = [
        ("AUTO", "Auto"),
        ("MANUAL", "Manual"),
    ]

    asset = models.ForeignKey(
        Asset,
        on_delete=models.PROTECT,
        related_name="rois",
        null=True,
        blank=True,
    )
    display_name = models.CharField(max_length=255, blank=True)

    x = models.PositiveIntegerField()
    y = models.PositiveIntegerField()
    width = models.PositiveIntegerField()
    height = models.PositiveIntegerField()
    source = models.CharField(max_length=16, choices=ROI_SOURCE_CHOICES, default="AUTO")
    is_active = models.BooleanField(default=False)
    is_complete = models.BooleanField(default=False)

    segmentations = models.ManyToManyField(
        # Lazy model references are "<app_label>.<ModelName>"; the app label is
        # the last component of the app's dotted path ("quantem.segmentation"),
        # so it stays "segmentation" here.
        "segmentation.ImageSegmentation",
        related_name="rois",
        blank=True,
    )

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.display_name} ({self.width}x{self.height})"

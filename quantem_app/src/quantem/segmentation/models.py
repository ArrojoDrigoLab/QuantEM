import logging

from django.db import models
from django.utils import timezone
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from quantem.assets.models import Asset, TimeStampedModel

from .geometry import (
    bbox_field_names,
    bbox_property,
    expand_update_fields,
    point_property,
    repair_geometry,
    wkb_geometry_property,
)
from .instance_params import coerce_instance_params, instance_params_defaults
from .segment_status import (
    SEGMENT_STATUS_CANDIDATE,
    SEGMENT_STATUS_CHOICES,
    SEGMENT_STATUS_CONFIRMED,
    SEGMENT_STATUS_REFINED,
    segment_status_label,
    status_for_segment_lifecycle,
)
from .source_models import (
    SOURCE_MODEL_UNKNOWN,
    infer_source_model_from_features,
    normalize_source_model,
)

logger = logging.getLogger(__name__)

# Geometry is stored as shapely WKB in image pixel space, plus indexed float
# columns for the derived bbox and centroid. See ``segmentation/geometry/fields.py``.


class SegmentationTypeTag(TimeStampedModel):
    """Tag for categorizing segmentation types (e.g., 'organelle')."""

    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["name"]


class SegmentationType(TimeStampedModel):
    """Global segmentation type (e.g., 'mitochondria', 'nucleus')."""

    internal_name = models.CharField(max_length=100, unique=True)
    short_name = models.CharField(max_length=100, unique=True)
    long_name = models.CharField(max_length=100, unique=True)
    default_color = models.CharField(max_length=7, blank=True)  # '#RRGGBB'
    tags = models.ManyToManyField(
        SegmentationTypeTag, related_name="segmentation_types", blank=True
    )

    def __str__(self):
        return self.long_name

    class Meta:
        ordering = ["long_name"]


def _build_check_constraint(*, expression, name: str) -> models.CheckConstraint:
    """``CheckConstraint`` across the rename of its first argument.

    ``check=`` was renamed to ``condition=`` in Django 5.1 and is removed in
    6.0. Try the current spelling first: preferring the deprecated one put a
    ``RemovedInDjango60Warning`` in the output of every test run and every
    server start, which is how a warning that matters gets missed.
    """
    try:
        return models.CheckConstraint(condition=expression, name=name)
    except TypeError:
        return models.CheckConstraint(check=expression, name=name)


class ImageSegmentation(TimeStampedModel):
    """Represents a segmentation instance for a specific image and segmentation type."""

    asset = models.ForeignKey(
        Asset,
        on_delete=models.PROTECT,
        related_name="segmentations",
        null=True,
        blank=True,
    )
    segmentation_type = models.ForeignKey(
        SegmentationType,
        on_delete=models.PROTECT,
        related_name="image_segmentations",
    )

    # Status of the segmentation pipeline
    STATUS_STAGE_CHOICES = [
        ("UNSTARTED", "Unstarted"),
        ("RUNNING_INFERENCE", "Running inference"),
        ("EXTRACTING_CANDIDATES", "Extracting candidates"),
        ("CANDIDATES_READY", "Candidates ready"),
        ("UPDATING", "Updating with feedback"),
        ("COMPUTING_FEATURES", "Computing features"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]
    status_stage = models.CharField(
        max_length=50,
        choices=STATUS_STAGE_CHOICES,
        default="UNSTARTED",
    )
    status_progress = models.FloatField(default=0.0)  # 0–100
    status_error = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["asset", "segmentation_type"],
                condition=models.Q(asset__isnull=False),
                name="unique_segmentation_per_asset",
            ),
        ]
        ordering = ["created_at"]

    def __str__(self):
        target_name = (
            self.asset.display_name
            if self.asset_id
            else str(self.id)
        )
        return f"{target_name} - {self.segmentation_type.long_name}"


class SegmentObject(TimeStampedModel):
    """
    Represents a single candidate shape (segment) within an ImageSegmentation.

    Each SegmentObject has geometry (polygon), centroid, bounding box, label state,
    a confidence score, and computed features stored as JSON.

    Geometry lives in ``geometry_wkb`` as shapely WKB in image pixel coordinates;
    ``centroid`` and ``bbox`` are properties over the indexed float columns that
    replace the old PostGIS point/polygon columns. Reading and writing all three
    still deals in shapely geometries.
    """

    segmentation = models.ForeignKey(
        ImageSegmentation, on_delete=models.CASCADE, related_name="segments"
    )

    # Geometry storage (image pixel coordinate system).
    geometry_wkb = models.BinaryField()
    centroid_x = models.FloatField()
    centroid_y = models.FloatField()
    bbox_minx = models.FloatField()
    bbox_miny = models.FloatField()
    bbox_maxx = models.FloatField()
    bbox_maxy = models.FloatField()

    geometry = wkb_geometry_property(
        "geometry_wkb", doc="Segment outline as a shapely Polygon in image pixels."
    )
    centroid = point_property(
        "centroid_x",
        "centroid_y",
        name="SegmentObject.centroid",
        doc="Segment centroid as a shapely Point in image pixels.",
    )
    bbox = bbox_property(
        "bbox", doc="Axis-aligned bounding box as a shapely box in image pixels."
    )

    _GEOMETRY_UPDATE_FIELDS = {
        "geometry": ("geometry_wkb",),
        "centroid": ("centroid_x", "centroid_y"),
        "bbox": bbox_field_names("bbox"),
    }

    # Label state
    LABEL_STATE_CHOICES = [
        ("CONFIRMED", "Confirmed"),
        ("EXCLUDED", "Excluded"),
        ("INFERRED", "Inferred"),
        ("CANDIDATE", "Candidate"),
    ]
    REFINEMENT_STATUS_CHOICES = [
        ("UNREFINED", "Unrefined"),
        ("MANUAL", "Manual"),
        ("AUTOMATIC", "Automatic"),
    ]
    label_state = models.CharField(
        max_length=10, choices=LABEL_STATE_CHOICES, default="INFERRED"
    )
    refined = models.CharField(
        max_length=10,
        choices=REFINEMENT_STATUS_CHOICES,
        default="UNREFINED",
    )
    STATUS_CANDIDATE = SEGMENT_STATUS_CANDIDATE
    STATUS_CONFIRMED = SEGMENT_STATUS_CONFIRMED
    STATUS_REFINED = SEGMENT_STATUS_REFINED
    STATUS_CHOICES = SEGMENT_STATUS_CHOICES
    status = models.PositiveSmallIntegerField(
        choices=STATUS_CHOICES,
        default=STATUS_CANDIDATE,
    )
    source_model = models.CharField(
        max_length=128,
        default=SOURCE_MODEL_UNKNOWN,
        db_index=True,
    )

    # Score - 100 for confirmed, 0 for excluded, model score for inferred
    confidence_score = models.FloatField(null=True, blank=True)

    # All computed metrics (regionprops, intensity percentiles, probability map stats, etc.).
    # Morphometrics here are in PIXELS; converting them to physical units requires
    # ``Asset.pixel_size_nm``.
    features = models.JSONField(default=dict, blank=True)

    base_segment = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="derived_segments",
    )

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["segmentation", "label_state"]),
            models.Index(fields=["segmentation", "confidence_score"]),
            models.Index(fields=["segmentation", "refined"]),
            models.Index(fields=["segmentation", "source_model", "status"]),
            models.Index(fields=["segmentation", "source_model", "label_state"]),
            # Numeric replacements for the PostGIS spatial indexes: a viewport
            # query is a range filter on these columns plus a shapely refine step.
            models.Index(fields=["segmentation", "bbox_minx", "bbox_maxx"]),
            models.Index(fields=["segmentation", "bbox_miny", "bbox_maxy"]),
            models.Index(fields=["segmentation", "centroid_x", "centroid_y"]),
        ]

    @property
    def status_label(self) -> str:
        return segment_status_label(self.status)

    def _infer_segmentation_type_internal_name(self) -> str | None:
        try:
            return self.segmentation.segmentation_type.internal_name
        except Exception:
            return None

    def sync_lifecycle_fields(self) -> list[str]:
        changed_fields: list[str] = []
        next_status = status_for_segment_lifecycle(
            label_state=self.label_state,
            refined=self.refined,
        )
        if self.status != next_status:
            self.status = next_status
            changed_fields.append("status")

        normalized_source = normalize_source_model(self.source_model)
        if not normalized_source or normalized_source == SOURCE_MODEL_UNKNOWN:
            normalized_source = infer_source_model_from_features(
                segmentation_type_internal_name=self._infer_segmentation_type_internal_name(),
                features=self.features,
                label_state=self.label_state,
            )
        if self.source_model != normalized_source:
            self.source_model = normalized_source
            changed_fields.append("source_model")
        return changed_fields

    def resolve_base_segment_or_self(self) -> "SegmentObject":
        if not self.base_segment_id:
            return self
        try:
            return self.base_segment
        except type(self).DoesNotExist:
            stale_base_segment_id = self.base_segment_id
            if self.pk is not None:
                type(self).objects.filter(pk=self.pk).update(base_segment=None)
            self.base_segment = None
            logger.warning(
                "Segment %s references missing base segment %s; treating it as the family root.",
                self.pk,
                stale_base_segment_id,
            )
            return self

    @staticmethod
    def _normalize_polygon_shape(
        value: BaseGeometry,
        *,
        field_name: str,
    ) -> Polygon:
        return repair_geometry(value, subject=f"SegmentObject.{field_name}")

    @staticmethod
    def _normalize_point_shape(
        value: BaseGeometry,
        *,
        field_name: str,
    ) -> Point:
        if not isinstance(value, BaseGeometry):
            raise ValueError(f"SegmentObject.{field_name} must be a shapely geometry.")
        if value.is_empty or value.geom_type != "Point":
            raise ValueError(
                f"SegmentObject.{field_name} must be a non-empty Point, got {value.geom_type}."
            )
        return value

    @classmethod
    def prepare_shape_fields(
        cls,
        *,
        geometry: BaseGeometry,
        centroid: BaseGeometry,
        bbox: BaseGeometry,
    ) -> tuple[Polygon, Point, Polygon]:
        return (
            cls._normalize_polygon_shape(geometry, field_name="geometry"),
            cls._normalize_point_shape(centroid, field_name="centroid"),
            cls._normalize_polygon_shape(bbox, field_name="bbox"),
        )

    def save(self, *args, **kwargs):
        lifecycle_fields = self.sync_lifecycle_fields()
        update_fields = expand_update_fields(
            kwargs.get("update_fields"), self._GEOMETRY_UPDATE_FIELDS
        )
        if update_fields is not None:
            if lifecycle_fields:
                update_fields = list(
                    dict.fromkeys([*update_fields, *lifecycle_fields])
                )
            kwargs["update_fields"] = update_fields
        self.geometry, self.centroid, self.bbox = self.prepare_shape_fields(
            geometry=self.geometry,
            centroid=self.centroid,
            bbox=self.bbox,
        )
        return super().save(*args, **kwargs)

    def __str__(self):
        return (
            f"Segment {self.id} ({self.label_state}) - "
            f"{self.segmentation} / {self.segmentation.segmentation_type.long_name}"
        )


class CompletedROI(TimeStampedModel):
    """User-marked completed polygon area for a segmentation/image pair."""

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="completed_rois",
    )
    geometry_wkb = models.BinaryField()
    bbox_minx = models.FloatField()
    bbox_miny = models.FloatField()
    bbox_maxx = models.FloatField()
    bbox_maxy = models.FloatField()

    geometry = wkb_geometry_property(
        "geometry_wkb", doc="Completed area as a shapely Polygon in image pixels."
    )
    bbox = bbox_property(
        "bbox", doc="Axis-aligned bounding box of ``geometry``, kept in sync on save."
    )

    _GEOMETRY_UPDATE_FIELDS = {
        "geometry": ("geometry_wkb",),
        "bbox": bbox_field_names("bbox"),
    }

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [
            models.Index(fields=["segmentation"]),
            models.Index(fields=["segmentation", "bbox_minx", "bbox_maxx"]),
            models.Index(fields=["segmentation", "bbox_miny", "bbox_maxy"]),
        ]

    @staticmethod
    def _normalize_polygon_shape(
        value: BaseGeometry,
        *,
        field_name: str,
    ) -> Polygon:
        return repair_geometry(value, subject=f"CompletedROI.{field_name}")

    def save(self, *args, **kwargs):
        update_fields = expand_update_fields(
            kwargs.get("update_fields"), self._GEOMETRY_UPDATE_FIELDS
        )
        if update_fields is not None:
            kwargs["update_fields"] = update_fields
        self.geometry = self._normalize_polygon_shape(
            self.geometry,
            field_name="geometry",
        )
        self.bbox = self.geometry
        return super().save(*args, **kwargs)

    def __str__(self):
        return (
            f"Completed ROI {self.id} - "
            f"{self.segmentation} / {self.segmentation.segmentation_type.long_name}"
        )


class RoiSegmentationStatus(TimeStampedModel):
    """Per-(ROI, segmentation) completion state.

    Records that a specific ROI window has been exhaustively annotated for a
    specific organelle, where ``segmentation`` is the ``ImageSegmentation``
    (asset x segmentation_type). This is the per-organelle analogue of
    ``ImageSegmentation.status_stage == "COMPLETED"`` ("mark image done") and a
    finer-grained companion to the flat ``ImageROI.is_complete`` flag.

    Its primary use is the ROI-scoped ground-truth contract: a ROI marked
    complete for an organelle is treated as exhaustively labeled, so for
    training the region inside the ROI is dense GT and everything outside it is
    ``ignore`` rather than background.
    """

    image_roi = models.ForeignKey(
        "assets.ImageROI",
        on_delete=models.CASCADE,
        related_name="segmentation_statuses",
    )
    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="roi_statuses",
    )
    is_complete = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["created_at", "id"]
        constraints = [
            models.UniqueConstraint(
                fields=["image_roi", "segmentation"],
                name="uniq_roi_segmentation_status",
            )
        ]
        indexes = [
            models.Index(fields=["segmentation", "is_complete"]),
            models.Index(fields=["image_roi"]),
        ]

    def set_complete(self, is_complete: bool) -> None:
        """Set completion state, stamping/clearing ``completed_at`` to match."""
        self.is_complete = is_complete
        self.completed_at = timezone.now() if is_complete else None

    def __str__(self):
        state = "complete" if self.is_complete else "incomplete"
        return f"ROI {self.image_roi_id} / segmentation {self.segmentation_id}: {state}"


class SegmentationOverlayState(TimeStampedModel):
    """Tracks the Viv-ready overlay pyramid lifecycle for a segmentation."""

    STATUS_MISSING = "MISSING"
    STATUS_READY = "READY"
    STATUS_DIRTY = "DIRTY"
    STATUS_BUILDING = "BUILDING"
    STATUS_FAILED = "FAILED"
    STATUS_CHOICES = [
        (STATUS_MISSING, "Missing"),
        (STATUS_READY, "Ready"),
        (STATUS_DIRTY, "Dirty"),
        (STATUS_BUILDING, "Building"),
        (STATUS_FAILED, "Failed"),
    ]

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="overlay_states",
    )
    candidate_source_model = models.CharField(max_length=128, blank=True, default="")
    status = models.CharField(
        max_length=16,
        choices=STATUS_CHOICES,
        default=STATUS_MISSING,
    )
    bundle_version = models.PositiveIntegerField(default=0)
    applied_revision = models.PositiveIntegerField(default=0)
    desired_revision = models.PositiveIntegerField(default=0)
    # Bumped on state-only changes (confirm/recolour/show-hide). Unlike the
    # revision fields above (which track the raster), a lut_revision change never
    # rebuilds the pyramid -- the client just refetches the colour/state LUT.
    lut_revision = models.PositiveIntegerField(default=0)
    pending_full_rebuild = models.BooleanField(default=False)
    dirty_chunk_runs = models.JSONField(default=list, blank=True)
    last_error = models.TextField(blank=True)
    last_built_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["segmentation", "candidate_source_model"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["segmentation", "candidate_source_model"],
                name="unique_overlay_state_per_segmentation_source",
            )
        ]

    def __str__(self):
        source_suffix = (
            f" source={self.candidate_source_model}"
            if self.candidate_source_model
            else ""
        )
        return (
            f"OverlayState<{self.segmentation_id}{source_suffix}> "
            f"{self.status} rev={self.applied_revision}/{self.desired_revision}"
        )


class SegmentationOverlayLabel(models.Model):
    """Maps a bundle's dense raster labels (1..N) to live objects.

    The ``labels`` raster stores these dense ids per pixel; this table is the
    stable label -> object mapping the render-time LUT joins against. Because the
    mapping survives state changes, confirming/recolouring an object never
    rewrites the raster -- only the resolved LUT colour changes.
    """

    overlay_state = models.ForeignKey(
        SegmentationOverlayState,
        on_delete=models.CASCADE,
        related_name="labels",
    )
    label = models.PositiveIntegerField()
    object_uuid = models.UUIDField()

    class Meta:
        indexes = [
            models.Index(fields=["overlay_state", "label"], name="seg_ovl_label_state_lbl_idx"),
            models.Index(
                fields=["overlay_state", "object_uuid"], name="seg_ovl_label_state_uuid_idx"
            ),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["overlay_state", "label"],
                name="unique_overlay_label_per_state",
            )
        ]

    def __str__(self):
        return f"OverlayLabel<{self.overlay_state_id}> {self.label} -> {self.object_uuid}"


class UserFeedback(TimeStampedModel):
    """User-submitted segmentation feedback input and utilization status."""

    INPUT_TYPE_POINT = "point"
    INPUT_TYPE_POLYGON = "polygon"
    INPUT_TYPE_CHOICES = [
        (INPUT_TYPE_POINT, "Point"),
        (INPUT_TYPE_POLYGON, "Polygon"),
    ]

    FEEDBACK_TYPE_CONFIRMED = "CONFIRMED"
    FEEDBACK_TYPE_REJECTED = "REJECTED"
    FEEDBACK_TYPE_CHOICES = [
        (FEEDBACK_TYPE_CONFIRMED, "Confirmed"),
        (FEEDBACK_TYPE_REJECTED, "Rejected"),
    ]

    STATUS_QUEUED = "QUEUED"
    STATUS_PROCESSING = "PROCESSING"
    STATUS_FAILED = "FAILED"
    STATUS_SUCCESS = "SUCCESS"
    UTILIZED_STATUS_CHOICES = [
        (STATUS_QUEUED, "Queued"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_FAILED, "Failed"),
        (STATUS_SUCCESS, "Success"),
    ]

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="user_feedback",
    )
    input_type = models.CharField(
        max_length=16,
        choices=INPUT_TYPE_CHOICES,
        default=INPUT_TYPE_POINT,
    )
    pt_x = models.FloatField(null=True, blank=True)
    pt_y = models.FloatField(null=True, blank=True)
    polygon_wkb = models.BinaryField(null=True, blank=True)

    pt_coordinates = point_property(
        "pt_x",
        "pt_y",
        name="UserFeedback.pt_coordinates",
        doc="Clicked point as a shapely Point in image pixels, or None.",
    )
    polygon = wkb_geometry_property(
        "polygon_wkb", doc="Drawn polygon as a shapely Polygon in image pixels, or None."
    )

    feedback_type = models.CharField(max_length=16, choices=FEEDBACK_TYPE_CHOICES)
    utilized_status = models.CharField(
        max_length=16,
        choices=UTILIZED_STATUS_CHOICES,
        default=STATUS_QUEUED,
    )

    _GEOMETRY_UPDATE_FIELDS = {
        "pt_coordinates": ("pt_x", "pt_y"),
        "polygon": ("polygon_wkb",),
    }

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["segmentation", "utilized_status"]),
            models.Index(fields=["segmentation", "feedback_type"]),
        ]
        constraints = [
            _build_check_constraint(
                expression=(
                    ~models.Q(input_type="point")
                    | models.Q(pt_x__isnull=False, pt_y__isnull=False)
                ),
                name="user_feedback_point_requires_coordinates",
            ),
            _build_check_constraint(
                expression=(
                    ~models.Q(input_type="polygon")
                    | models.Q(polygon_wkb__isnull=False)
                ),
                name="user_feedback_polygon_requires_polygon",
            ),
        ]

    def save(self, *args, **kwargs):
        update_fields = expand_update_fields(
            kwargs.get("update_fields"), self._GEOMETRY_UPDATE_FIELDS
        )
        if update_fields is not None:
            kwargs["update_fields"] = update_fields
        return super().save(*args, **kwargs)

    def __str__(self):
        return (
            f"UserFeedback {self.id} ({self.feedback_type}/{self.utilized_status}) "
            f"for {self.segmentation_id}"
        )


class ProbabilityMap(TimeStampedModel):
    """
    Represents a probability map file associated with an ImageSegmentation.

    Probability maps are the per-pixel foreground scores produced by a
    segmentation model. These maps are used to compute probability-based
    features for each segment (e.g., mean probability within the segment's
    polygon).

    Multiple probability maps can be associated with a single segmentation,
    allowing comparison of different model outputs or different channels.
    """

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="probability_maps",
    )
    name = models.CharField(max_length=255, blank=True)  # Optional display name
    file_path = models.CharField(
        max_length=1024
    )  # Relative path within container/data directory
    channel_index = models.PositiveSmallIntegerField(
        default=0
    )  # For multi-channel probability maps
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        name_str = self.name if self.name else "Unnamed"
        return f"{name_str} - {self.segmentation}"


class SegmentationConfig(TimeStampedModel):
    """Per-segmentation instance-extraction configuration."""

    segmentation = models.OneToOneField(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="config",
    )

    instance_params = models.JSONField(default=instance_params_defaults, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def get_instance_params(self) -> dict[str, int | float | None]:
        return coerce_instance_params(self.instance_params)

    def __str__(self):
        return f"Config for {self.segmentation}"



class SegmentationCompletionArchive(TimeStampedModel):
    """What one "mark image done" discarded, kept so unlock can put it back.

    ``POST /api/segmentations/<id>/complete`` can delete every non-CONFIRMED
    object in a segmentation. Before this existed, ``DELETE`` on the same
    endpoint ("unlock segmentation") only flipped ``status_stage`` back and
    restored nothing: a user who pruned a 32-object run with no confirmations
    could recover it only by paying for a whole new inference pass. Creating a
    segmentation is cheap and reversible; destroying a run's output was neither,
    and had a lower bar.

    One row holds the whole discarded set as JSON. At most one archive is kept
    per segmentation -- a new completion replaces the previous archive, and a
    successful restore deletes it -- so this cannot grow without bound.

    What a restore preserves and what it does not
    ---------------------------------------------
    Preserved: the primary key, geometry, centroid, bbox, label state,
    refinement, status, source model, confidence and the whole ``features``
    dict (including the run identity that says which run produced the object).
    Restoring under the original ids is what keeps ``base_segment`` links and
    any id a client is still holding valid.

    Not preserved: ``created_at`` / ``updated_at``, which Django stamps on
    insert. A restored object is the same object; its timestamps say when it
    came back.

    A very large discard is not archived at all -- see
    :data:`quantem.segmentation.completion.ARCHIVE_MAX_OBJECTS` -- and the API
    says so in its response rather than implying an undo that is not there.
    """

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="completion_archives",
    )
    #: How many objects the completion discarded. Equals ``len(objects_json)``
    #: whenever the set was archivable, and is still recorded when it was not,
    #: so the count a user was shown survives even without the objects.
    discarded_count = models.PositiveIntegerField(default=0)
    objects_json = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["segmentation", "-created_at"]),
        ]

    def __str__(self):
        return (
            f"Completion archive for {self.segmentation_id}: "
            f"{self.discarded_count} object(s)"
        )

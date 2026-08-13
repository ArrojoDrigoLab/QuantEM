import logging

from django.db import IntegrityError, models, transaction
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
    SOURCE_MODEL_MANUAL,
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

    MEASUREMENT_MODE_OBJECTS = "objects"
    MEASUREMENT_MODE_GLOBAL = "global"
    MEASUREMENT_MODE_CHOICES = [
        (MEASUREMENT_MODE_OBJECTS, "Object-based"),
        (MEASUREMENT_MODE_GLOBAL, "Global"),
    ]

    internal_name = models.CharField(max_length=100, unique=True)
    short_name = models.CharField(max_length=100, unique=True)
    long_name = models.CharField(max_length=100, unique=True)
    default_color = models.CharField(max_length=7, blank=True)  # '#RRGGBB'
    # This setting belongs to the reusable type, not one image's instance.
    measurement_mode = models.CharField(
        max_length=16,
        choices=MEASUREMENT_MODE_CHOICES,
        default=MEASUREMENT_MODE_OBJECTS,
    )
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
    # Most segmentations are named by their reusable type. Analysis masks are
    # deliberately different: one image can carry several named masks (for
    # example, a tissue mask and a cells mask) that must not become reusable
    # custom types on every other image.
    display_name = models.CharField(max_length=100, blank=True, default="")

    # Status of the segmentation pipeline
    STATUS_STAGE_CHOICES = [
        ("UNSTARTED", "Unstarted"),
        ("RUNNING_INFERENCE", "Running inference"),
        ("THRESHOLD_READY", "Threshold ready"),
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

    #: How many rows of the live preview raster have been written for the run
    #: currently in flight. Counts up during inference so the viewer can wash
    #: the finished band over the image before the overlay exists; reset to 0
    #: when a run starts. Zero also means "nothing to wash", which is the state
    #: of every segmentation that has never run.
    preview_rows_ready = models.PositiveIntegerField(default=0)

    #: The include level (0-1) the current object set was extracted at, or
    #: ``None`` when no dial movement has been recorded against it. ``None`` is
    #: not "0.5": the run's own threshold lives in each object's run identity,
    #: and inventing a level here would claim a dial position the user never
    #: chose. Set when objects are re-extracted from the stored probability map.
    include_level = models.FloatField(null=True, blank=True)

    #: Immutable while this segmentation is completed. Explicitly unlocking
    #: clears the note because the result is no longer final; the next
    #: completion writes a new immutable description. Probability maps are
    #: working artifacts and are reclaimed at completion, so this compact note
    #: is the durable answer to which model produced the final pixels.
    final_result_provenance = models.JSONField(default=dict, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["asset", "segmentation_type", "display_name"],
                condition=models.Q(asset__isnull=False),
                name="unique_segmentation_name_per_asset",
            ),
        ]
        ordering = ["created_at"]

    def __str__(self):
        target_name = self.asset.display_name if self.asset_id else str(self.id)
        name = self.display_name or self.segmentation_type.long_name
        return f"{target_name} - {name}"


class GlobalMask(TimeStampedModel):
    """The one binary foreground mask for a global-mode segmentation.

    Global results deliberately have no ``SegmentObject`` representation: a
    disconnected foreground and its holes are properties of one mask, not a
    collection of instances. ``file_path`` is storage-root-relative so the
    database remains portable with the rest of the QuantEM data directory.
    """

    segmentation = models.OneToOneField(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="global_mask",
    )
    file_path = models.CharField(max_length=1024)
    width = models.PositiveIntegerField()
    height = models.PositiveIntegerField()
    foreground_pixels = models.PositiveBigIntegerField(default=0)
    source = models.CharField(max_length=32, blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"Global mask for {self.segmentation_id}"


class AnalysisMaskObject(TimeStampedModel):
    """One named, independently editable object inside an analysis mask.

    Analysis masks are measured globally, so their authoritative analysis
    representation remains :class:`GlobalMask`.  This compact vector record is
    the editing representation: one object may be a polygon, a multipolygon,
    or a polygon with holes after any number of Include/Exclude brush strokes
    and polygon operations.
    """

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="analysis_mask_objects",
    )
    name = models.CharField(max_length=100)
    color = models.CharField(max_length=7)
    sort_order = models.PositiveIntegerField(default=0)
    geometry_wkb = models.BinaryField(null=True, blank=True)
    geometry = wkb_geometry_property(
        "geometry_wkb",
        doc="Polygonal analysis-mask geometry in image pixels.",
    )

    class Meta:
        ordering = ["sort_order", "created_at"]
        indexes = [
            models.Index(
                fields=["segmentation", "sort_order"],
                name="seg_analysis_obj_sort_idx",
            )
        ]

    def __str__(self):
        return f"{self.segmentation_id}: {self.name}"


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
    bbox = bbox_property("bbox", doc="Axis-aligned bounding box as a shapely box in image pixels.")

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
    label_state = models.CharField(max_length=10, choices=LABEL_STATE_CHOICES, default="INFERRED")
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

    #: Which result version produced this object. Objects written before result
    #: versions existed are version 1, which is why the default is 1 and not 0:
    #: there has always been a first result, and calling it version 1 keeps the
    #: numbering a user sees ("Version 2") the same as the numbering stored.
    run_version = models.PositiveIntegerField(default=1)

    #: When a later version replaced this object, or ``None`` while it is live.
    #: A superseded object is kept rather than deleted so a revert is exact and
    #: so a user's own corrections survive a model pass; every read path that
    #: means "the current objects" filters on ``superseded_at__isnull=True``.
    superseded_at = models.DateTimeField(null=True, blank=True)

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
            # Result versions. Both are composite and lead with ``segmentation``
            # rather than being bare single-column indexes on the two new
            # columns, because neither is ever queried on its own -- "the live
            # objects" and "the objects of version N" are always scoped to one
            # segmentation first. Two indexes instead of four keeps the
            # per-object insert cost of an extraction run where it was.
            models.Index(fields=["segmentation", "superseded_at"]),
            models.Index(fields=["segmentation", "run_version"]),
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
                update_fields = list(dict.fromkeys([*update_fields, *lifecycle_fields]))
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
            f" source={self.candidate_source_model}" if self.candidate_source_model else ""
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
                    ~models.Q(input_type="point") | models.Q(pt_x__isnull=False, pt_y__isnull=False)
                ),
                name="user_feedback_point_requires_coordinates",
            ),
            _build_check_constraint(
                expression=(~models.Q(input_type="polygon") | models.Q(polygon_wkb__isnull=False)),
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

    The four provenance columns below say what grid the stored array is on and
    how it was got there. They exist because owner ruling R11 made the stored
    map the authority every threshold reads, rather than a by-product kept for
    fine-tuning: once a dial re-thresholds this array, "which grid, quantised
    how, resampled with what" stops being a curiosity and becomes part of the
    record that says whether two object sets are the same computation.
    """

    #: The stored array is on the original image's pixel grid.
    GRID_NATIVE = "native"
    #: The stored array is on whatever grid the model predicted at. Maps written
    #: before R11 are this, and they are not replayable: re-thresholding one
    #: would decide on the wrong grid. Recorded rather than assumed so a reader
    #: can tell an old map from a new one.
    GRID_MODEL = "model"
    GRID_CHOICES = [
        (GRID_NATIVE, "Original image pixels"),
        (GRID_MODEL, "Model prediction grid"),
    ]

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="probability_maps",
    )
    name = models.CharField(max_length=255, blank=True)  # Optional display name
    file_path = models.CharField(max_length=1024)  # Relative path within container/data directory
    channel_index = models.PositiveSmallIntegerField(
        default=0
    )  # For multi-channel probability maps
    metadata = models.JSONField(default=dict, blank=True)

    #: Which pixel grid the stored array is on. Defaults to ``"native"``
    #: because that is what every map written from now on is; the rows that
    #: predate R11 are corrected by the data migration that adds this column,
    #: which reads each map's recorded metadata rather than guessing.
    grid = models.CharField(
        max_length=16,
        choices=GRID_CHOICES,
        default=GRID_NATIVE,
    )
    #: How the continuous field was quantised for storage, e.g. ``"uint8"``.
    #: Bounds the granularity of any threshold read off this map -- uint8 is
    #: about 1/255 -- which is a number the dial has to state rather than imply.
    quantisation = models.CharField(max_length=16, default="uint8")
    #: The interpolation used to bring probabilities back to native pixels
    #: (``"area"``, ``"bilinear"``), or blank when no resampling happened
    #: because the model already predicted at native scale. Never ``"nearest"``
    #: for a continuous field.
    resample_kernel = models.CharField(max_length=32, blank=True, default="")
    #: The float range the quantised levels stand for, as ``[low, high]``.
    #: ``None`` means the range was not recorded, which is not the same as
    #: ``[0, 1]``: a reader must be able to tell an unrecorded range from a
    #: recorded full one before converting a level back to a probability.
    value_range = models.JSONField(null=True, blank=True)

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
        return f"Completion archive for {self.segmentation_id}: {self.discarded_count} object(s)"


#: How many times :meth:`SegmentationResultVersion.record_new_result` will
#: recompute the next version number after losing the unique constraint to
#: another pass. Small on purpose: two extractions finishing on the same
#: segmentation in the same instant is already unusual, and a long spin here
#: would hold a write transaction open behind a completed inference run.
_VERSION_ALLOCATION_ATTEMPTS = 5


class SegmentationResultVersion(TimeStampedModel):
    """One numbered result for a segmentation: what it was, and when.

    A segmentation's objects are replaced whenever the model runs again or the
    include level moves. Before this existed, the previous set was gone, so
    "put it back" was another inference pass and two versions of the same image
    could not be told apart in an analysis. A row here is the header of one
    such set; the objects themselves carry the matching
    :attr:`SegmentObject.run_version` and are marked
    :attr:`SegmentObject.superseded_at` rather than deleted, so a revert is
    exact rather than approximate.

    ``version`` starts at 1 because objects written before result versions
    existed default to 1 -- there has always been a first result, and giving it
    a number here rather than a special case keeps every later count honest.

    **An analysis counts one image once.** Two versions of the same
    segmentation are two descriptions of the same pixels, not two samples, and
    the read path in :mod:`quantem.analysis` is required to use the live set
    only.
    """

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="result_versions",
    )
    version = models.PositiveIntegerField(default=1)
    #: The include level this set was extracted at, or ``None`` when the set
    #: came straight from a run and no dial position was chosen. Not defaulted
    #: to the run's threshold: the two are different facts and conflating them
    #: would show a user a dial position they never set.
    include_level = models.FloatField(null=True, blank=True)
    #: The run identity every object in this set carries, copied here so the
    #: version list can name the run without loading an object.
    run_identity = models.JSONField(default=dict, blank=True)
    object_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["segmentation", "version"]
        constraints = [
            models.UniqueConstraint(
                fields=["segmentation", "version"],
                name="uniq_result_version_per_segmentation",
            ),
        ]
        indexes = [
            models.Index(fields=["segmentation", "-version"]),
        ]

    @classmethod
    def current_version_for(cls, segmentation) -> int:
        """The result version a new judgement is about.

        One definition, in one place, because three features ask the question
        and a disagreement between them would silently mis-scope a quality
        estimate: the highest numbered version if any is recorded, else the
        highest version stamped on a live object, else 1. Never 0 -- a
        segmentation with no objects at all is still on its first result, and
        answering 0 would make every check written against it look stale the
        moment the first run landed.
        """
        latest = (
            cls.objects.filter(segmentation=segmentation)
            .order_by("-version")
            .values_list("version", flat=True)
            .first()
        )
        if latest:
            return int(latest)
        stamped = (
            SegmentObject.objects.filter(
                segmentation=segmentation,
                superseded_at__isnull=True,
            )
            .order_by("-run_version")
            .values_list("run_version", flat=True)
            .first()
        )
        return int(stamped) if stamped else 1

    @classmethod
    def _next_version_after(cls, segmentation, floor: int) -> int:
        """The number to give the next result. Never reuses one already taken.

        ``floor`` is what the caller knows about the set being replaced; the
        highest row already recorded is what the database knows. Taking the
        larger of the two and adding one is what makes the retry in
        :meth:`record_new_result` terminate: a lost race raises the row floor,
        so the next attempt asks for a strictly larger number.
        """
        latest = (
            cls.objects.filter(segmentation=segmentation)
            .order_by("-version")
            .values_list("version", flat=True)
            .first()
        )
        return max(int(floor or 0), int(latest or 0)) + 1

    @classmethod
    def record_new_result(
        cls,
        segmentation,
        *,
        after_version: int | None = None,
        run_identity: dict | None = None,
        include_level: float | None = None,
    ) -> "SegmentationResultVersion | None":
        """Number the result a pass has just produced, and move the objects onto it.

        **This is the writer that makes the version mean something.** The
        schema, the constraints and every read path were landed together and
        nothing ever incremented the number, so
        :meth:`current_version_for` could only ever answer 1: a quality
        estimate taken against one candidate set went on feeding the headline
        after the model had been re-run at a different threshold, and the
        "compare with the previous version" surface was permanently empty
        because there was never a previous version to compare with.

        Called at the one moment that makes a stored quality estimate untrue:
        a model pass has replaced the candidate set (see
        :func:`quantem.seg_core.db.extraction.run_extraction`). Answering a
        question, drawing a hand-made outline or clearing labels are
        deliberately **not** such moments -- they are the user's own judgements
        about the same objects, and :class:`QualityCheck` says in as many words
        that a later clear-labels pass does not unmake one.

        **Every live object is stamped, not only the new rows.** A pass deletes
        this model's own untouched candidates and writes fresh ones, but a
        CONFIRMED or EXCLUDED object survives it untouched -- and those objects
        are still on screen and still in the analysis, so they are part of the
        new result. Leaving them on the old number would drop them out of
        :func:`~quantem.segmentation.quality_sampling.live_model_objects`, and
        the count the spot check quotes ("12 of the 511") would silently stop
        counting everything the user had already confirmed. ``run_version`` is
        the number of the *result set*; which run produced any individual
        object is a separate fact and is recorded per object in
        ``features["run"]``, which this never touches.

        Args:
            segmentation: the segmentation whose objects were just replaced.
            after_version: the version of the set this one replaces, or ``0``
                when there was no result here before. **Read before the pass
                ran**, because afterwards the question cannot be answered: the
                objects on the table are this pass's own, and asking then would
                make a brand-new segmentation's first result "version 2".
                ``None`` falls back to :meth:`current_version_for`, which is
                right for a caller that genuinely does not know.
            run_identity: the run behind the new set, copied onto the row so
                the version list can name the run without loading an object.
            include_level: the dial position this set was extracted at.
                ``None`` -- the ordinary case -- means the set came straight
                from a run and nobody chose a level; it is deliberately not
                defaulted to the run's threshold, because showing a user a dial
                position they never set is worse than showing none.

        Returns:
            The new row, or ``None`` when a version number could not be
            allocated. ``None`` rather than an exception: this is bookkeeping
            beside a completed inference run, and a failure to number the
            result must not throw away the objects the run just wrote.
        """
        stamp = dict(run_identity) if isinstance(run_identity, dict) else {}
        level = None if include_level is None else float(include_level)
        floor = (
            cls.current_version_for(segmentation) if after_version is None else int(after_version)
        )

        for _attempt in range(_VERSION_ALLOCATION_ATTEMPTS):
            version = cls._next_version_after(segmentation, floor)
            try:
                with transaction.atomic():
                    row = cls.objects.create(
                        segmentation=segmentation,
                        version=version,
                        include_level=level,
                        run_identity=stamp,
                    )
                    # ``update`` rather than a save loop: this is one statement
                    # over an indexed (segmentation, superseded_at) range, and
                    # it deliberately leaves ``updated_at`` alone -- numbering
                    # the result is not an edit to any object in it.
                    SegmentObject.objects.filter(
                        segmentation=segmentation,
                        superseded_at__isnull=True,
                    ).update(run_version=version)
                    row.object_count = (
                        SegmentObject.objects.filter(
                            segmentation=segmentation,
                            superseded_at__isnull=True,
                            run_version=version,
                        )
                        .exclude(source_model=SOURCE_MODEL_MANUAL)
                        .count()
                    )
                    row.save(update_fields=["object_count", "updated_at"])
            except IntegrityError:
                # Two passes over the same segmentation finished together and
                # both claimed the same number. The loser recomputes and takes
                # the next one; the unique constraint is what makes that safe.
                continue
            return row

        logger.warning(
            "Could not allocate a result version for segmentation %s after %d "
            "attempts; its objects stay on the previous version.",
            getattr(segmentation, "pk", segmentation),
            _VERSION_ALLOCATION_ATTEMPTS,
        )
        return None

    def __str__(self):
        return (
            f"Result version {self.version} of {self.segmentation_id}: "
            f"{self.object_count} object(s)"
        )


class QualityCheck(TimeStampedModel):
    """One question the app asked about one object, and the user's answer.

    Half of the two-number quality answer. Precision is estimated from a random
    spot check: a sample of the run's own untouched objects, each shown once,
    each answered yes / wrong shape / not the thing / not sure. This table is
    the sample *and* the answers, which is what makes the sample stable -- the
    rows are written when the sample is drawn, so the same twelve objects come
    back after a reload, a restart, and a second person opening the same image.

    **Why ``sample_seed`` and ``ordinal`` are stored rather than recomputed.**
    The draw is deterministic (a hash of the object id and the seed), so it
    could in principle be recomputed on every request. It is not, because the
    set it draws from moves: answering a question labels the object, which
    takes it out of the "untouched" pool, so a recomputed sample would reshuffle
    itself as the user worked through it. Storing the draw fixes the twelve
    before the first answer changes anything.

    **``unsure`` is excluded from the denominator**, and the readout says so.
    An answer of "not sure" is a real answer -- it is the honest one when the
    image does not settle the question -- and counting it either way would
    invent a judgement the user declined to make.

    A row survives the deletion of its object (``segment`` goes null): the user
    made that judgement, and a later clear-labels pass does not unmake it. The
    estimate still reports the answer, because dropping answered rows would
    quietly shrink the denominator that the sample size in the sentence quotes.
    """

    KIND_RANDOM_SAMPLE = "random_sample"
    KIND_COUNT_BOX = "count_box"
    KIND_CHOICES = [
        (KIND_RANDOM_SAMPLE, "Random spot check"),
        (KIND_COUNT_BOX, "Inside the count box"),
    ]

    ANSWER_YES = "yes"
    ANSWER_WRONG_SHAPE = "wrong_shape"
    ANSWER_NOT_THE_THING = "not_the_thing"
    ANSWER_UNSURE = "unsure"
    ANSWER_CHOICES = [
        (ANSWER_YES, "Yes"),
        (ANSWER_WRONG_SHAPE, "Wrong shape"),
        (ANSWER_NOT_THE_THING, "Not the thing"),
        (ANSWER_UNSURE, "Not sure"),
    ]

    #: The answers that count as the object being good. ``wrong_shape`` is not
    #: here: an outline that is wrong is a wrong object for a measurement, even
    #: though the thing under it is real.
    POSITIVE_ANSWERS = frozenset({ANSWER_YES})
    #: The answers that count at all. ``unsure`` is deliberately absent.
    SCORED_ANSWERS = frozenset({ANSWER_YES, ANSWER_WRONG_SHAPE, ANSWER_NOT_THE_THING})

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="quality_checks",
    )
    #: The result version this check is about. A check never carries over to a
    #: new version: the objects changed, so the judgement is about objects that
    #: are no longer on screen.
    run_version = models.PositiveIntegerField(default=1)
    kind = models.CharField(
        max_length=16,
        choices=KIND_CHOICES,
        default=KIND_RANDOM_SAMPLE,
    )
    #: Fixed once per (segmentation, result version) so the draw is repeatable
    #: and so extending a sample from 12 to 36 takes the next 24 of the same
    #: order rather than reshuffling the first 12.
    sample_seed = models.BigIntegerField(default=0)
    segment = models.ForeignKey(
        SegmentObject,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="quality_checks",
    )
    #: Position in the draw, from 0. Also the order the questions are asked in,
    #: so "3 of 12" means the same thing on every device.
    ordinal = models.PositiveIntegerField(default=0)
    #: Blank until answered. Blank is not an answer and is not counted.
    answer = models.CharField(
        max_length=16,
        choices=ANSWER_CHOICES,
        blank=True,
        default="",
    )
    answered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["segmentation", "run_version", "kind", "ordinal"]
        constraints = [
            models.UniqueConstraint(
                fields=["segmentation", "run_version", "kind", "ordinal"],
                name="uniq_quality_check_slot",
            ),
            _build_check_constraint(
                # An answer and its timestamp are one fact. Half of it is a row
                # that either counts with no record of when, or records a
                # moment with nothing decided at it.
                expression=(
                    models.Q(answer="", answered_at__isnull=True)
                    | (~models.Q(answer="") & models.Q(answered_at__isnull=False))
                ),
                name="quality_check_answer_and_time_together",
            ),
        ]
        indexes = [
            models.Index(fields=["segmentation", "run_version", "kind"]),
        ]

    @property
    def is_answered(self) -> bool:
        return bool(self.answer)

    def record_answer(self, answer: str, *, now=None) -> None:
        """Set the answer and its timestamp together.

        In one method because the check constraint above requires both, and
        because a re-answer must move the timestamp: the user changed their
        mind, and the record should say when they did.
        """
        self.answer = answer
        self.answered_at = now or timezone.now()

    def __str__(self):
        state = self.answer or "unanswered"
        return (
            f"Quality check {self.ordinal} ({self.kind}/{state}) "
            f"for {self.segmentation_id} v{self.run_version}"
        )


class CountBox(TimeStampedModel):
    """One small box the user marked up exhaustively, and what it showed.

    The other half of the quality answer, and the half that cannot be got any
    other way. A spot check can only ask about objects the model already found,
    so it measures precision and is blind to misses -- a model that finds 511
    of 1 300 real mitochondria scores beautifully on a spot check while the
    user's counts are 60 % low. The only way to see the misses is to have a
    human mark every object in a small area and compare.

    **The app places the box, not the user.** A user-chosen box is a biased
    box: people put it where the segmentation looks interesting, which is
    exactly where it is not representative. ``seed`` records the draw so the
    same box comes back on reload and so the placement can be re-derived.

    **One box, and it is rough.** ``n_marked`` and ``n_matched`` come from a
    single small window, so the recall they imply carries a wide interval and
    the readout says so in words. Two boxes would be better; the price is the
    user's time, and this plan spends about three minutes of it.
    """

    #: The app scored windows for tissue content and drew among the good ones.
    PLACEMENT_TISSUE_SCORED = "tissue_scored"
    #: The app could not read the image to score it, so the box went in the
    #: middle. A distinct value rather than a silent fallback: a box in the
    #: centre of an image whose centre is empty resin measures nothing, and the
    #: recall it implies is not the recall of the tissue.
    PLACEMENT_CENTRED = "centred"

    segmentation = models.ForeignKey(
        ImageSegmentation,
        on_delete=models.CASCADE,
        related_name="count_boxes",
    )
    #: The result version this box was marked against, for the same reason
    #: :attr:`QualityCheck.run_version` exists: a new version is new objects,
    #: so ``n_matched`` no longer describes anything on screen.
    run_version = models.PositiveIntegerField(default=1)

    # The rectangle, in the image's own pixel coordinates -- the same space
    # segment geometry lives in, so no conversion sits between the box and the
    # objects it is compared against.
    x = models.FloatField()
    y = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()

    #: The draw that placed the box.
    seed = models.BigIntegerField(default=0)
    #: How the box got where it is: :attr:`PLACEMENT_TISSUE_SCORED` or
    #: :attr:`PLACEMENT_CENTRED`. Blank on a row written before this was
    #: recorded, which is "not recorded" and not a third kind of placement.
    #: Stored rather than re-derived because re-deriving it needs the image,
    #: and the reason to know it is precisely that the image could not be read.
    placement = models.CharField(max_length=16, blank=True, default="")
    #: How many objects the user marked inside the box.
    n_marked = models.PositiveIntegerField(default=0)
    #: How many of those the model had already found.
    n_matched = models.PositiveIntegerField(default=0)
    #: Set when the user pressed done. An unfinished box is not an answer: it
    #: reads as "0 marked", which would say the model missed nothing.
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["segmentation", "run_version"]
        constraints = [
            models.UniqueConstraint(
                fields=["segmentation", "run_version"],
                name="uniq_count_box_per_result_version",
            ),
            _build_check_constraint(
                expression=models.Q(width__gt=0) & models.Q(height__gt=0),
                name="count_box_has_area",
            ),
            _build_check_constraint(
                # The matched objects are a subset of the marked ones, so a row
                # claiming otherwise is arithmetic that would report recall
                # above 1.
                expression=models.Q(n_matched__lte=models.F("n_marked")),
                name="count_box_matched_within_marked",
            ),
        ]
        indexes = [
            models.Index(fields=["segmentation", "run_version"]),
        ]

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    def __str__(self):
        state = "complete" if self.is_complete else "in progress"
        return (
            f"Count box for {self.segmentation_id} v{self.run_version} "
            f"({state}): {self.n_matched} of {self.n_marked} found"
        )

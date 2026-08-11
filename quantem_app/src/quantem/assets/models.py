import hashlib
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
    # SHA-256 of the *source file's* bytes, recorded at import. This is how the
    # library recognises an image it already has; see
    # :func:`refuse_duplicate_import` for why that matters and what is compared.
    #
    # Not derivable later: the upload pipeline deletes the source file once it
    # has written the canonical PNG (``assets/tasks.py``), so the only moment
    # these bytes exist is the import itself. Blank means "imported before this
    # was recorded" and never matches anything -- an honest "unknown", not a
    # false "different".
    source_sha256 = models.CharField(
        max_length=64,
        blank=True,
        default="",
        db_index=True,
    )
    # Set only when the user was told "you already have this" and chose to
    # import it again anyway. A second copy that says what it is a copy of can
    # be shown as one, counted as one, and cleaned up later; a second copy that
    # says nothing is the silent duplicate this whole mechanism exists to
    # prevent. ``None`` is the ordinary case -- a first import is not a copy of
    # anything.
    #
    # ``SET_NULL`` rather than ``PROTECT``: the original being deleted must not
    # make the copy undeletable, and "this was a copy of an image that is gone"
    # is not worth blocking anything over.
    duplicate_of = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        related_name="duplicate_copies",
        null=True,
        blank=True,
    )

    # Where this image sits in the library. Both optional: an unorganised
    # library is the normal starting state, not a problem. Deleting an
    # experiment orphans its images rather than destroying them, which is why
    # this is SET_NULL and not CASCADE.
    experiment = models.ForeignKey(
        "library.Experiment",
        on_delete=models.SET_NULL,
        related_name="assets",
        null=True,
        blank=True,
    )
    datasets = models.ManyToManyField(
        "library.Dataset",
        related_name="assets",
        blank=True,
    )

    class Meta:
        ordering = ["display_name", "created_at"]

    def __str__(self):
        return self.display_name


# ---------------------------------------------------------------------------
# Importing the same image twice
# ---------------------------------------------------------------------------
#
# Dropping a file that is already in the library used to create a second Asset,
# silently. Three drops of one montage gave three rows and three copies on disk.
# That is refused now, for a reason that is about the science and not about the
# disk: the library is the denominator. A duplicated field of view is counted
# twice in every per-group number and weighted twice in every mean, and the
# duplicate is invisible in a list of forty file names that all look alike. It
# also splits the user's proofreading in two, with no way to say which copy is
# the answer.
#
# The refusal names the image that is already there, so the client can offer to
# open it, and it is *before* the bytes are staged, so nothing is copied for an
# import that will not happen.

#: Machine-readable tag for the refusal, for a client that wants to render its
#: own copy and a link rather than the sentence below.
DUPLICATE_IMPORT_ERROR_CODE = "duplicate_image"

#: Read a megabyte at a time: an EM image is routinely hundreds of megabytes and
#: this runs inside the import request.
_HASH_CHUNK_BYTES = 1024 * 1024

#: Preprocessing outcomes that mean the existing row is *not* a usable copy of
#: the image. Re-dropping a file whose first import failed is a retry, and
#: answering a retry with "you already have this" would be both wrong and
#: infuriating -- the user does not have it, that is why they are back.
_UNUSABLE_PREPROCESS_STAGES = ("FAILED", "CANCELLED")


def sha256_of_upload(uploaded_file) -> str:
    """SHA-256 of an upload's bytes, streamed.

    Content, not file name. Two exports of the same field of view under
    different names are one image; two different images that a microscope
    called ``Image_001.tif`` are two. A name comparison gets both cases wrong,
    and the second one wrong in the dangerous direction.

    ``File.chunks()`` seeks to the start and leaves the file re-readable, so the
    caller can still save the upload afterwards.
    """
    digest = hashlib.sha256()
    for chunk in uploaded_file.chunks(_HASH_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def find_imported_asset_with_same_bytes(sha256: str):
    """The oldest live image whose source file had this digest, or ``None``.

    Deleted images do not count (deleting one is how a user says "let me start
    again with that file"), and neither do failed or cancelled imports.
    """
    if not sha256:
        return None
    return (
        Asset.objects.filter(
            source_sha256=sha256,
            lifecycle_status=Asset.LIFECYCLE_ACTIVE,
        )
        .exclude(preprocess_stage__in=_UNUSABLE_PREPROCESS_STAGES)
        .order_by("created_at")
        .first()
    )


def _imported_when(asset: "Asset") -> str:
    """``"9 August 2026 at 14:32"`` -- local time, for a sentence."""
    from django.utils import timezone

    when = asset.created_at
    if when is None:  # an unsaved row; nothing useful to say
        return ""
    local = timezone.localtime(when) if timezone.is_aware(when) else when
    return f"{local.day} {local:%B %Y} at {local:%H:%M}"


def duplicate_import_message(existing: "Asset") -> str:
    """What the user is told: which image this already is, and what to do.

    Kept to one short sentence and a clause. The client renders a server
    refusal in the row it belongs to, and a paragraph does not fit there --
    and ``frontend/src/utils/apiErrors.ts`` treats an error body longer than
    200 characters as a document rather than a message on any path where the
    JSON envelope is lost. Naming the image, the time and the action is all a
    user needs; the identity for a link is in
    :attr:`DuplicateImageError.payload`.
    """
    name = (existing.display_name or existing.original_filename or "").strip()
    named = f'"{name}"' if name else "an image with no name"
    when = _imported_when(existing)
    imported = f", imported {when}" if when else ""
    return (
        f"This is the same file as {named}{imported}. Nothing was imported "
        f"— open that image to carry on with it."
    )


class DuplicateImageError(ValueError):
    """The bytes being imported are already in the library.

    A ``ValueError`` on purpose: the upload view already turns one into a 400
    carrying ``str(exc)``, so the sentence reaches the user with no change to
    the view. :attr:`payload` is there for a view that wants to answer 409 with
    the identity of the image it is pointing at.
    """

    def __init__(self, existing: "Asset", sha256: str) -> None:
        super().__init__(duplicate_import_message(existing))
        self.existing_asset = existing
        self.existing_asset_id = str(existing.id)
        self.sha256 = sha256
        self.error_code = DUPLICATE_IMPORT_ERROR_CODE

    @property
    def payload(self) -> dict:
        existing = self.existing_asset
        return {
            "error": str(self),
            "error_code": self.error_code,
            "duplicate_of": {
                "id": str(existing.id),
                "display_name": existing.display_name,
                "created_at": (
                    existing.created_at.isoformat().replace("+00:00", "Z")
                    if existing.created_at
                    else None
                ),
            },
        }


def resolve_duplicate_import(
    uploaded_file, *, allow_duplicate: bool = False
) -> tuple[str, "Asset | None"]:
    """``(digest, the image these bytes already are)`` -- or a refusal.

    The decision point the import call site uses. It answers both questions in
    one read of the upload, because the import needs both answers: the digest
    goes on the new row's :attr:`Asset.source_sha256` (without it the *next*
    import cannot recognise this one either), and the matched image goes on
    :attr:`Asset.duplicate_of` when the user has deliberately asked for a
    second copy.

    With ``allow_duplicate`` false -- the default, and what an ordinary
    re-drop does -- a match raises :class:`DuplicateImageError` and nothing is
    imported. With it true, the same match is returned instead of raised, so
    the copy that gets created knows what it is a copy of. Returning
    ``(digest, None)`` is the ordinary "this image is new" answer.

    :func:`refuse_duplicate_import` is the older, narrower form of this and is
    kept because it reads better where only the refusal matters.
    """
    sha256 = sha256_of_upload(uploaded_file)
    existing = find_imported_asset_with_same_bytes(sha256)
    if existing is not None and not allow_duplicate:
        raise DuplicateImageError(existing, sha256)
    return sha256, existing


def refuse_duplicate_import(uploaded_file, *, allow_duplicate: bool = False) -> str:
    """Hash the upload, refuse it if the library already holds those bytes.

    Returns the digest, which the caller stores on the new
    :class:`Asset` -- so the check costs one read of the upload and the answer
    is kept, rather than being recomputed on the next import.

    Call this **before** the upload is staged: hashing streams from wherever the
    upload already is, while staging copies the whole image, and copying a
    hundred megabytes for an import that is about to be refused is the kind of
    waiting a user blames on the microscope.

    ``allow_duplicate`` is the deliberate second copy. Nothing in the shipped
    client sends it yet; it exists so the refusal is a question the product can
    let the user answer, not a wall.

    Two honest limits. Images imported before this column existed carry no
    digest and cannot be recognised -- the source file they came from was
    deleted at the end of their own import, so there is nothing left to hash.
    And two identical uploads racing each other can both pass this check; on a
    single-user application over loopback that is a theoretical window, and
    closing it with a database constraint would take the ``allow_duplicate``
    escape hatch away.
    """
    sha256, _existing = resolve_duplicate_import(uploaded_file, allow_duplicate=allow_duplicate)
    return sha256


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

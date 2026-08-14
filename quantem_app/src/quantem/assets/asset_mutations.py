from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path
from typing import Literal, NamedTuple

from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.utils import timezone

from quantem.assets.models import Asset, Rendition, resolve_duplicate_import
from quantem.assets.serializers import serialize_asset_detail
from quantem.assets.upload_staging import discard_upload_if_unreferenced
from quantem.assets.utils import (
    PNG_UPLOAD_SUFFIXES,
    TIFF_UPLOAD_SUFFIXES,
    extract_image_metadata,
    save_uploaded_file_to_path,
    validate_upload_file,
)
from quantem.core.config import DATA_DIR, UPLOADS_DIR
from quantem.core.local_storage import normalize_stored_path_value
from quantem.jobs.constants import (
    JOB_TYPE_ENSURE_IMAGE_NGFF,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    QUEUE_P2_UPLOAD,
)
from quantem.jobs.models import Job

PIXEL_SIZE_FIELDS = ("pixel_size_nm", "pixel_size_nm_z")


# ---------------------------------------------------------------------------
# Where this import goes in the library
# ---------------------------------------------------------------------------
#
# Both form fields are optional, but an imported image's experiment and dataset
# are not. An importer that names no experiment gets a new one named after the
# image. An importer that names no dataset gets the next ``Dataset N`` inside
# that experiment. This keeps the simple one-file import simple while giving
# every image a stable library home from the moment its row exists.
#
# The *shape* is checked here, before a byte is claimed, because "you named a
# dataset but no experiment" is the user's to fix and needs no file on disk to
# discover. The rows themselves are resolved inside the same transaction that
# writes the asset, so an import that fails later does not leave behind an
# experiment nobody has any images in.


class ImportGrouping(NamedTuple):
    """The four cleaned fields an import may carry."""

    experiment_id: str
    experiment_name: str
    dataset_id: str
    dataset_name: str

    @property
    def is_empty(self) -> bool:
        return not any(self)


def normalise_import_grouping(
    *,
    experiment_id=None,
    experiment_name=None,
    dataset_id=None,
    dataset_name=None,
) -> ImportGrouping:
    """Clean the grouping fields an import form posted, and refuse the one
    combination that cannot mean anything.

    A dataset belongs to exactly one experiment, so naming a dataset without
    naming an experiment describes nothing. Every other combination is legal,
    including all four blank.
    """
    grouping = ImportGrouping(
        experiment_id=str(experiment_id or "").strip(),
        experiment_name=str(experiment_name or "").strip(),
        dataset_id=str(dataset_id or "").strip(),
        dataset_name=str(dataset_name or "").strip(),
    )
    named_dataset = grouping.dataset_id or grouping.dataset_name
    named_experiment = grouping.experiment_id or grouping.experiment_name
    if named_dataset and not named_experiment:
        raise ValueError(
            "A dataset lives inside an experiment. Choose or name an "
            "experiment as well, then this dataset can be created in it."
        )
    return grouping


def _resolve_import_grouping(
    grouping: ImportGrouping | None,
    *,
    display_name: str,
):
    """Resolve the experiment and dataset for a new image.

    Called before the asset row is inserted, inside that row's transaction.
    This avoids creating a temporary per-image experiment and then replacing
    it when the import form named a shared experiment.
    """
    from django.core.exceptions import ValidationError as DjangoValidationError

    from quantem.library.grouping import resolve_dataset, resolve_experiment
    from quantem.library.models import create_default_dataset, create_image_experiment

    grouping = grouping or ImportGrouping("", "", "", "")

    try:
        experiment = resolve_experiment(
            experiment_id=grouping.experiment_id or None,
            experiment_name=grouping.experiment_name or None,
        )
        dataset = resolve_dataset(
            experiment=experiment,
            dataset_id=grouping.dataset_id or None,
            dataset_name=grouping.dataset_name or None,
        )
        if experiment is None:
            experiment = create_image_experiment(display_name)
        if dataset is None:
            dataset = create_default_dataset(experiment)
        return experiment, dataset
    except DjangoValidationError as exc:
        # The import view turns a ValueError into a 400 carrying the sentence;
        # a Django ValidationError would fall through to the 500 branch and be
        # reported as an internal fault over what is a typing mistake.
        messages = getattr(exc, "messages", None)
        raise ValueError(messages[0] if messages else str(exc)) from exc


# ---------------------------------------------------------------------------
# Is this file whole?
# ---------------------------------------------------------------------------
#
# A PNG cut off part-way through used to be accepted (201, a row in the
# library) and to fail minutes later, inside the background pipeline, with a
# sentence about a decoder. The acceptance check was a header parse, and a
# PNG's header is its first 33 bytes, so any file truncated after that passed.
# MEASURED in wave 0c: ``real_png[:1000]`` and ``real_png[:100000]`` both
# answered 201.
#
# What is checked here is the *container*, not the picture. A PNG is a chain of
# length-prefixed chunks, so the file can be walked by reading eight bytes and
# seeking over the rest: the cost is the number of chunks, never the size of
# the image, which is what lets this run at the door on a 2-3 GB import on an
# 8 GB laptop (owner ruling R3). A TIFF is checked the same way in spirit --
# every strip or tile the directory promises must actually lie inside the file.
#
# Deliberately not checked: per-chunk CRCs and the compressed stream. Both need
# every byte, and both would refuse an image some third-party exporter merely
# mis-stamped -- a worse failure than the one being fixed, because there would
# be no way to get that image in at all. Truncation is the class that was
# measured, and it is the class a half-finished copy off a share, an
# interrupted download or a full disk actually produces.

#: PNG's fixed 8-byte signature.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: 4 bytes of length + 4 of type, then the data, then a 4-byte CRC.
_PNG_CHUNK_HEADER_BYTES = 8
_PNG_CHUNK_TRAILER_BYTES = 4

#: A stop for noise that happens to read as an endless run of tiny chunks. A
#: 3 GB PNG written in 8 kB blocks is about 400 000 chunks, so this is an order
#: of magnitude past anything real.
_PNG_MAX_CHUNKS = 4_000_000

#: Byte-order marks for classic TIFF and BigTIFF, either endianness.
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")

#: StripOffsets/StripByteCounts and TileOffsets/TileByteCounts: between them
#: they say where every byte of pixel data in a directory lives.
_TIFF_DATA_TAG_PAIRS = ((273, 279), (324, 325))
_TIFF_DATA_TAGS = frozenset(tag for pair in _TIFF_DATA_TAG_PAIRS for tag in pair)

#: Bytes per TIFF value type, by the type code in a directory entry.
_TIFF_TYPE_BYTES = {
    1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4,
    10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8,
}  # fmt: skip

#: Stops for a directory chain that is really noise. A 10 000-plane volume is
#: already an extreme file; a directory with more than a few thousand entries
#: is not a TIFF anyone wrote.
_TIFF_MAX_DIRECTORIES = 100_000
_TIFF_MAX_ENTRIES = 4_096

#: Read at most this much of one strip or tile table. Bigger than any real
#: one (a 2-3 GB image striped at 8 kB is ~1.5 MB of table) and small enough
#: that a corrupt count cannot turn the door check into an allocation.
_TIFF_MAX_VALUE_BYTES = 8 * 1024 * 1024

#: What the user is told when the file stops before the picture does. No file
#: name, no byte counts, no decoder: the one thing they can act on is that the
#: copy they have is not all of it. Under 200 characters so the client renders
#: it as a sentence (``frontend/src/utils/apiErrors.ts``).
INCOMPLETE_SOURCE_MESSAGE = (
    "This image is incomplete: the file ends part-way through the picture "
    "data. Nothing was imported. Copy or export the image again, then import "
    "the new file."
)

#: Nothing at all arrived. Distinguished from truncation because the answer is
#: different: this is usually the wrong file, not a damaged one.
EMPTY_SOURCE_MESSAGE = (
    "This file is empty: there is no image in it. Nothing was imported. "
    "Check you picked the right file, then try again."
)


def _wrong_format_message(kind: str) -> str:
    """The name says one thing and the bytes say another."""
    return (
        f"This file is not a {kind} image, whatever its name says. Nothing was "
        f"imported. Check you picked the right file, then try again."
    )


def _damaged_message(kind: str) -> str:
    """A container that starts right and then does not make sense."""
    return (
        f"This {kind} image could not be read: it looks damaged or incomplete. "
        f"Nothing was imported. Copy or export the image again, then import "
        f"the new file."
    )


def verify_source_is_complete(uploaded_file) -> None:
    """Refuse an upload whose container is damaged, before anything reads it.

    Raises :class:`ValueError` carrying one plain sentence -- the upload view
    turns that into a 400 with the sentence in ``error``, which is what the
    import panel renders.

    Costs a handful of eight-byte reads and seeks; never a decode, never a
    buffer the size of the image. An upload that cannot be seeked at all is
    passed through rather than refused: an unverifiable file is not a damaged
    one, and refusing it would be a new way to lose a good image.

    The file is left rewound, so the hash and the staging copy that follow read
    it from the beginning.
    """

    if not _is_seekable(uploaded_file):
        return
    suffix = Path(str(getattr(uploaded_file, "name", "") or "")).suffix.lower()
    try:
        total = int(uploaded_file.seek(0, os.SEEK_END))
        if total <= 0:
            raise ValueError(EMPTY_SOURCE_MESSAGE)
        if suffix in PNG_UPLOAD_SUFFIXES:
            _verify_png_container(uploaded_file, total)
        elif suffix in TIFF_UPLOAD_SUFFIXES:
            _verify_tiff_container(uploaded_file, total)
    finally:
        with contextlib.suppress(AttributeError, OSError, ValueError):
            uploaded_file.seek(0)


def _is_seekable(uploaded_file) -> bool:
    seek = getattr(uploaded_file, "seek", None)
    if not callable(seek):
        return False
    seekable = getattr(uploaded_file, "seekable", None)
    try:
        return bool(seekable()) if callable(seekable) else True
    except (AttributeError, OSError, ValueError):  # pragma: no cover
        return False


def _is_png_chunk_type(chunk_type: bytes) -> bool:
    """PNG chunk types are exactly four ASCII letters; case carries meaning."""
    return len(chunk_type) == 4 and all(
        0x41 <= byte <= 0x5A or 0x61 <= byte <= 0x7A for byte in chunk_type
    )


def _verify_png_container(uploaded_file, total: int) -> None:
    """Walk the chunk chain to IEND without reading any picture data."""
    uploaded_file.seek(0)
    if uploaded_file.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
        raise ValueError(_wrong_format_message("PNG"))

    offset = len(_PNG_SIGNATURE)
    expect_header_chunk = True
    for _ in range(_PNG_MAX_CHUNKS):
        if offset + _PNG_CHUNK_HEADER_BYTES > total:
            raise ValueError(INCOMPLETE_SOURCE_MESSAGE)
        uploaded_file.seek(offset)
        header = uploaded_file.read(_PNG_CHUNK_HEADER_BYTES)
        if len(header) < _PNG_CHUNK_HEADER_BYTES:  # pragma: no cover - see above
            raise ValueError(INCOMPLETE_SOURCE_MESSAGE)
        length = int.from_bytes(header[:4], "big")
        chunk_type = header[4:]
        if not _is_png_chunk_type(chunk_type):
            raise ValueError(_wrong_format_message("PNG"))
        if expect_header_chunk and chunk_type != b"IHDR":
            raise ValueError(_wrong_format_message("PNG"))
        expect_header_chunk = False
        end = offset + _PNG_CHUNK_HEADER_BYTES + length + _PNG_CHUNK_TRAILER_BYTES
        if end > total:
            raise ValueError(INCOMPLETE_SOURCE_MESSAGE)
        if chunk_type == b"IEND":
            return
        offset = end
    raise ValueError(_damaged_message("PNG"))  # pragma: no cover - see the cap


class _SourceIsTruncated(Exception):
    """Internal: a TIFF directory pointing past the end of its own file."""


def _verify_tiff_container(uploaded_file, total: int) -> None:
    """Every strip or tile the directory promises must be inside the file.

    Reads the image file directories by hand rather than through ``tifffile``,
    for two reasons that point the same way. ``assets/tests/
    test_ngff_decode_chokepoint.py`` forbids a new decoder anywhere outside
    ``canonical_decode`` -- correctly: four saturating decodes have been found
    in this tree, each added by someone who only meant to read a header. And a
    decoder is the wrong tool anyway. What is wanted here is arithmetic on the
    directory: a few hundred bytes read, no pixel path entered at all, and no
    dependence on how a library chooses to be lenient.

    Only the first and last directories are inspected. Truncation removes the
    tail, so the last directory is where it shows, and checking every plane of
    a thousand-plane volume would buy no new class of defect.
    """
    uploaded_file.seek(0)
    header = uploaded_file.read(16)
    if len(header) < 8 or header[:4] not in _TIFF_MAGIC:
        raise ValueError(_wrong_format_message("TIFF"))
    order: Literal["little", "big"] = "little" if header[:2] == b"II" else "big"
    version = int.from_bytes(header[2:4], order)
    if version == 42:
        big = False
        directory_offset = int.from_bytes(header[4:8], order)
    elif version == 43:
        # BigTIFF: an 8-byte offset size, two reserved bytes, then the offset.
        if len(header) < 16 or int.from_bytes(header[4:6], order) != 8:
            raise ValueError(_wrong_format_message("TIFF"))
        big = True
        directory_offset = int.from_bytes(header[8:16], order)
    else:
        raise ValueError(_wrong_format_message("TIFF"))

    try:
        first: dict | None = None
        last: dict | None = None
        seen: set[int] = set()
        for _ in range(_TIFF_MAX_DIRECTORIES):
            if not directory_offset or directory_offset in seen:
                break
            seen.add(directory_offset)
            entries, directory_offset = _read_tiff_directory(
                uploaded_file, directory_offset, total, order, big
            )
            if first is None:
                first = entries
            last = entries
        if first is None:
            raise ValueError(_damaged_message("TIFF"))
        for entries in ({id(first): first, id(last): last}).values():
            _check_tiff_data_fits(uploaded_file, entries, total, order, big)
    except _SourceIsTruncated:
        raise ValueError(INCOMPLETE_SOURCE_MESSAGE) from None


def _read_tiff_directory(
    uploaded_file, offset: int, total: int, order, big: bool
) -> tuple[dict, int]:
    """``({tag: (type, count, value field)}, next directory offset)``.

    Only the four tags that describe where the pixel data lives are kept; the
    rest of the directory is skipped over without being interpreted.
    """
    count_bytes = 8 if big else 2
    entry_bytes = 20 if big else 12
    next_bytes = 8 if big else 4
    minimum = 16 if big else 8
    if offset < minimum or offset + count_bytes > total:
        raise _SourceIsTruncated
    uploaded_file.seek(offset)
    raw_count = uploaded_file.read(count_bytes)
    if len(raw_count) < count_bytes:
        raise _SourceIsTruncated
    entry_count = int.from_bytes(raw_count, order)
    if entry_count <= 0 or entry_count > _TIFF_MAX_ENTRIES:
        raise ValueError(_damaged_message("TIFF"))
    body_bytes = entry_count * entry_bytes + next_bytes
    if offset + count_bytes + body_bytes > total:
        raise _SourceIsTruncated
    body = uploaded_file.read(body_bytes)
    if len(body) < body_bytes:
        raise _SourceIsTruncated

    entries: dict[int, tuple[int, int, bytes]] = {}
    for index in range(entry_count):
        start = index * entry_bytes
        tag = int.from_bytes(body[start : start + 2], order)
        if tag not in _TIFF_DATA_TAGS:
            continue
        value_type = int.from_bytes(body[start + 2 : start + 4], order)
        if big:
            value_count = int.from_bytes(body[start + 4 : start + 12], order)
            field = body[start + 12 : start + 20]
        else:
            value_count = int.from_bytes(body[start + 4 : start + 8], order)
            field = body[start + 8 : start + 12]
        entries[tag] = (value_type, value_count, field)
    return entries, int.from_bytes(body[entry_count * entry_bytes :], order)


def _tiff_values(uploaded_file, entry, total: int, order, big: bool) -> list[int]:
    """The integers one directory entry holds, inline or by reference."""
    value_type, value_count, field = entry
    type_bytes = _TIFF_TYPE_BYTES.get(value_type)
    if not type_bytes or value_count <= 0:
        return []
    payload = value_count * type_bytes
    inline = 8 if big else 4
    if payload <= inline:
        data = field[:payload]
    else:
        pointer = int.from_bytes(field[:inline], order)
        if pointer + payload > total:
            raise _SourceIsTruncated
        if payload > _TIFF_MAX_VALUE_BYTES:
            # The array is inside the file, which is what truncation would
            # break; reading megabytes of strip table to check each entry is
            # not what a door check is for.
            return []
        uploaded_file.seek(pointer)
        data = uploaded_file.read(payload)
        if len(data) < payload:
            raise _SourceIsTruncated
    return [
        int.from_bytes(data[at : at + type_bytes], order) for at in range(0, payload, type_bytes)
    ]


def _check_tiff_data_fits(uploaded_file, entries: dict, total: int, order, big) -> None:
    for offset_tag, count_tag in _TIFF_DATA_TAG_PAIRS:
        if offset_tag not in entries or count_tag not in entries:
            continue
        starts = _tiff_values(uploaded_file, entries[offset_tag], total, order, big)
        lengths = _tiff_values(uploaded_file, entries[count_tag], total, order, big)
        # Not ``strict``: a malformed directory can declare unequal offset and
        # byte-count arrays, and that is a file to refuse, not an exception to
        # raise out of a validator.
        for start, length in zip(starts, lengths):  # noqa: B905
            if start + length > total:
                raise _SourceIsTruncated


def parse_pixel_size_nm(value) -> float | None:
    """Coerce a user-supplied pixel size to a positive float (or ``None``).

    Blank input means "unknown"; a non-numeric or non-positive value is a hard
    error, because a wrong pixel size silently corrupts every analysis number.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Pixel size must be a number, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError("Pixel size must be greater than zero.")
    return parsed


def create_uploaded_asset(
    *,
    uploaded_file: UploadedFile,
    display_name: str | None = None,
    pixel_size_nm=None,
    notes: str | None = None,
    segment_mito: bool = False,
    segment_er: bool = False,
    segment_nucleus: bool = False,
    segment_ld: bool = False,
    defer_processing: bool = False,
    swallow_enqueue_errors: bool = False,
    allow_duplicate: bool = False,
    grouping: ImportGrouping | None = None,
) -> dict:
    """Import one uploaded image and return its serialized detail.

    The order of the first four steps is the whole of this function's
    behaviour under a bad import, and each step is where it is on purpose:

    1. the extension, because it costs nothing and rules out most mis-drops;
    2. the pixel size, because a typo there is the user's to fix and no bytes
       need moving to find it;
    3. :func:`verify_source_is_complete`, because a file that stops half way
       through the picture is not importable and finding that out at the door
       is the difference between a sentence the user can act on and a
       background failure four minutes later;
    4. the content digest, which both answers "do we already have this?" and
       is recorded on the row so the *next* import can ask the same question.

    Only then are the bytes claimed into the staging directory. Everything
    after the claim runs under a guard that releases them again if the import
    does not complete, so a refusal never costs the user disk.

    ``allow_duplicate`` is the user having been told the library already holds
    these bytes and having said "import it anyway". The copy that results
    records what it is a copy of (:attr:`Asset.duplicate_of`); a duplicate that
    says so can be shown, counted and cleaned up, and a silent one cannot.
    """
    is_valid, error_message = validate_upload_file(uploaded_file)
    if not is_valid:
        raise ValueError(error_message)

    pixel_size = parse_pixel_size_nm(pixel_size_nm)
    verify_source_is_complete(uploaded_file)
    source_sha256, duplicate_of = resolve_duplicate_import(
        uploaded_file, allow_duplicate=allow_duplicate
    )
    asset_id = uuid.uuid4()
    original_filename = uploaded_file.name
    display_name = display_name or original_filename
    file_ext = (
        uploaded_file.name.rsplit(".", 1)[-1] if "." in uploaded_file.name else "tif"
    ).lower()
    staged_path = UPLOADS_DIR / f"{asset_id}.{file_ext}"
    save_uploaded_file_to_path(uploaded_file, staged_path)
    try:
        return _record_uploaded_asset(
            asset_id=asset_id,
            staged_path=staged_path,
            display_name=display_name,
            original_filename=original_filename,
            notes=notes,
            pixel_size=pixel_size,
            source_sha256=source_sha256,
            duplicate_of=duplicate_of,
            segment_mito=segment_mito,
            segment_er=segment_er,
            segment_nucleus=segment_nucleus,
            segment_ld=segment_ld,
            defer_processing=defer_processing,
            swallow_enqueue_errors=swallow_enqueue_errors,
            grouping=grouping,
        )
    except BaseException:
        # The claim happens before anything reads the file, so every failure
        # from here on -- an unreadable header, a reader raising, a database
        # error -- used to answer the user with an error and keep the whole
        # body forever. ``StagedUploadedFile.close`` is the backstop, but it
        # only runs once the response has been handed to the socket, which is
        # after the user has been told the import failed and after a client
        # that immediately lists the directory can see the bytes. This is the
        # hook ``upload_staging`` documents for saying so from inside the
        # request. It keeps anything a rendition or asset row points at, so an
        # accepted import is never touched.
        discard_upload_if_unreferenced(staged_path)
        raise


def _record_uploaded_asset(
    *,
    asset_id,
    staged_path,
    display_name: str,
    original_filename: str,
    notes: str | None,
    pixel_size,
    source_sha256: str,
    duplicate_of,
    segment_mito: bool,
    segment_er: bool,
    segment_nucleus: bool,
    segment_ld: bool,
    defer_processing: bool,
    swallow_enqueue_errors: bool,
    grouping: ImportGrouping | None = None,
) -> dict:
    """Read the staged file and write the library rows for it.

    Split out of :func:`create_uploaded_asset` so that everything which can
    fail *after* the bytes have been claimed sits inside one guard, rather than
    the guard being re-remembered at each new failure path -- which is how the
    two leaks before this one were introduced.
    """
    metadata = extract_image_metadata(staged_path)

    # A pixel size the user typed always wins; otherwise take whatever the file
    # declares. Leaving this null means no resampling and no calibrated
    # measurement, so it is worth reading even though many EM TIFFs omit it.
    if pixel_size is None:
        pixel_size = metadata.get("pixel_size_nm")

    # Free text the importer typed, stored on the column the library's search
    # already covers (``_filtered_asset_queryset`` matches display name, filename
    # and notes). The import form had a "Tags" box posting ``tag_names``, which
    # nothing here read: ``Asset`` has no tag field and there is no tag model in
    # the tree, so the text was accepted and dropped. ``notes`` is the field that
    # exists, and it is already patchable through :func:`update_asset` -- upload
    # was simply the one door that could not set it.
    notes_text = "" if notes is None else str(notes).strip()

    with transaction.atomic():
        experiment, dataset = _resolve_import_grouping(
            grouping,
            display_name=display_name,
        )
        asset = Asset.objects.create(
            id=asset_id,
            display_name=display_name,
            original_filename=original_filename,
            notes=notes_text,
            logical_width=int(metadata["width"]),
            logical_height=int(metadata["height"]),
            channels=int(metadata["channels"]),
            bit_depth=int(metadata["bit_depth"]),
            pixel_size_nm=pixel_size,
            preprocess_stage="ENCODING",
            preprocess_progress=0.0,
            preprocess_error="",
            # Recorded here and nowhere else: the upload pipeline deletes the
            # source file once the canonical PNG exists (``assets/tasks.py``),
            # so this import is the only moment these bytes are in reach. A
            # blank column is not "different", it is "unknown", and every row
            # that carries one is a row no future import can recognise.
            source_sha256=source_sha256,
            duplicate_of=duplicate_of,
            experiment=experiment,
        )
        Rendition.objects.create(
            asset=asset,
            type=Rendition.TYPE_FULL,
            storage_root="DATA_DIR",
            stored_path=normalize_stored_path_value(staged_path, relative_to=DATA_DIR),
            path_exists=staged_path.exists(),
            is_directory=False,
            stored_width=int(metadata["width"]),
            stored_height=int(metadata["height"]),
            stored_channels=int(metadata["channels"]),
            stored_bit_depth=int(metadata["bit_depth"]),
            metadata={
                "upload_state": "staged",
                # A multi-image import clears this only in the same transaction
                # that creates all of its job rows. If the app closes first,
                # the next home-page load can recover the stranded asset.
                "processing_deferred": defer_processing,
                "original_filename": original_filename,
                # Asset.raw_metadata/normalized_metadata were corpus-curation
                # fields and are gone; the source file's own metadata belongs to
                # the rendition that was read from.
                "source_metadata": _json_safe_metadata(metadata),
            },
        )
        asset.datasets.add(dataset)

    if not defer_processing:
        try:
            enqueue_upload_pipeline(
                asset,
                segment_mito=segment_mito,
                segment_er=segment_er,
                segment_nucleus=segment_nucleus,
                segment_ld=segment_ld,
            )
        except Exception:
            if not swallow_enqueue_errors:
                raise

    asset.refresh_from_db()
    return serialize_asset_detail(asset)


def enqueue_upload_pipeline(
    asset: Asset,
    *,
    segment_mito: bool = False,
    segment_er: bool = False,
    segment_nucleus: bool = False,
    segment_ld: bool = False,
) -> Job:
    """Queue initial processing once for an imported image.

    Multi-image imports deliberately create every asset before calling this
    function. The open-job check makes the batch-start endpoint safe to retry
    if its response is lost after the transaction commits.
    """
    existing_job = (
        Job.objects.filter(
            type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            payload_json__asset_id=str(asset.id),
        )
        .order_by("-created_at")
        .first()
    )
    if existing_job is not None:
        job = existing_job
    else:
        job = Job.enqueue(
            job_type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            payload={
                "asset_id": str(asset.id),
                "segment_mito": segment_mito,
                "segment_er": segment_er,
                "segment_nucleus": segment_nucleus,
                "segment_ld": segment_ld,
            },
            priority="high",
            resource_class="cpu",
            queue_name=QUEUE_P2_UPLOAD,
            tags=[f"asset:{asset.id}"],
        )

    full = asset.renditions.filter(type=Rendition.TYPE_FULL).first()
    if full is not None and full.metadata.get("processing_deferred"):
        full.metadata = {**full.metadata, "processing_deferred": False}
        full.save(update_fields=["metadata", "updated_at"])
    return job


def _json_safe_metadata(metadata: dict) -> dict:
    payload = dict(metadata)
    dtype = payload.get("dtype")
    if dtype is not None:
        payload["dtype"] = str(dtype)
    shape = payload.get("shape")
    if shape is not None:
        payload["shape"] = [int(value) for value in shape]
    return payload


def update_asset(asset: Asset, payload: dict, *, inside_transaction: bool = False) -> dict:
    allowed = {"display_name", "notes", *PIXEL_SIZE_FIELDS}
    updates = {key: payload[key] for key in allowed if key in payload}
    for field in PIXEL_SIZE_FIELDS:
        if field in updates:
            updates[field] = parse_pixel_size_nm(updates[field])
    if not updates:
        return serialize_asset_detail(asset)

    def apply_updates() -> None:
        for field, value in updates.items():
            setattr(asset, field, value)
        asset.save(update_fields=[*updates.keys(), "updated_at"])

    if inside_transaction:
        apply_updates()
    else:
        with transaction.atomic():
            apply_updates()

    asset.refresh_from_db()
    return serialize_asset_detail(asset)


def enqueue_ngff_for_asset(asset: Asset) -> Job:
    active_job = (
        Job.objects.filter(
            type=JOB_TYPE_ENSURE_IMAGE_NGFF,
            status__in={"PENDING", "RUNNING", "RETRY"},
            payload_json__asset_id=str(asset.id),
        )
        .order_by("-created_at")
        .first()
    )
    if active_job is not None:
        return active_job
    return Job.enqueue(
        job_type=JOB_TYPE_ENSURE_IMAGE_NGFF,
        payload={"asset_id": str(asset.id)},
        priority="high",
        resource_class="cpu",
        queue_name=QUEUE_P2_UPLOAD,
        tags=[f"asset:{asset.id}"],
    )


def tombstone_asset(asset: Asset) -> None:
    if asset.lifecycle_status == Asset.LIFECYCLE_DELETED:
        return

    active_jobs = Job.objects.filter(
        status__in={"PENDING", "RUNNING", "RETRY"},
        payload_json__asset_id=str(asset.id),
    )
    for job in active_jobs:
        job.cancel_requested = True
        if job.status in {"PENDING", "RETRY"}:
            job.status = "CANCELLED"
            job.finished_at = timezone.now()
            job.message = "cancelled"
        job.save(
            update_fields=[
                "cancel_requested",
                "status",
                "finished_at",
                "message",
                "updated_at",
            ]
        )

    asset.lifecycle_status = Asset.LIFECYCLE_DELETED
    asset.deleted_at = timezone.now()
    asset.save(update_fields=["lifecycle_status", "deleted_at", "updated_at"])

"""Importing the same file twice is refused, and the refusal says which image.

The defect (wave 0b, V9): dropping a file the library already held created a
second image, silently. The verifier ended a session with three rows and three
copies on disk of one 175 MB montage.

Why that is worth a refusal rather than a warning. The library is the
denominator. A field of view that is in it twice is counted twice in every
per-group number and weighted twice in every mean, and among forty file names
that all look alike nobody spots it. It also splits the user's proofreading
across two rows with no way to say which one is the answer. A warning arrives
after the row exists, which is after the damage.

What is compared is the bytes, not the name -- see
:func:`quantem.assets.models.sha256_of_upload` for why the name is wrong in
both directions.
"""

from __future__ import annotations

import hashlib
from uuid import uuid4

import tifffile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.utils import timezone

from quantem.assets.asset_mutations import create_uploaded_asset
from quantem.assets.models import (
    DUPLICATE_IMPORT_ERROR_CODE,
    Asset,
    DuplicateImageError,
    find_imported_asset_with_same_bytes,
    refuse_duplicate_import,
    sha256_of_upload,
)
from quantem.core.config import STORAGE_DIR
from quantem.registry.tests.copy_gate import find_violations
from quantem.testing import make_em_like_array


def _tiff_bytes(width: int = 64, height: int = 48, *, seed: int = 0) -> bytes:
    """A real TIFF, so nothing here passes on bytes a reader would refuse."""
    path = STORAGE_DIR / "tmp" / f"dup_{uuid4().hex}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    array = make_em_like_array(width, height)
    if seed:
        array = (array.astype("uint16") + seed).astype("uint8")
    tifffile.imwrite(str(path), array, photometric="minisblack")
    try:
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _upload(name: str, payload: bytes) -> SimpleUploadedFile:
    return SimpleUploadedFile(name, payload, content_type="image/tiff")


class DigestTests(TestCase):
    def test_the_digest_is_of_the_bytes(self):
        payload = _tiff_bytes()
        self.assertEqual(
            sha256_of_upload(_upload("scan.tif", payload)),
            hashlib.sha256(payload).hexdigest(),
        )

    def test_hashing_leaves_the_upload_readable(self):
        """The import saves the file after this; hashing must not consume it."""
        payload = _tiff_bytes()
        uploaded = _upload("scan.tif", payload)

        sha256_of_upload(uploaded)

        self.assertEqual(b"".join(uploaded.chunks()), payload)

    def test_two_different_images_of_the_same_size_hash_differently(self):
        """Guards against anyone reducing this to a cheap size comparison."""
        first = _tiff_bytes(seed=0)
        second = _tiff_bytes(seed=57)
        self.assertEqual(len(first), len(second))
        self.assertNotEqual(
            sha256_of_upload(_upload("a.tif", first)),
            sha256_of_upload(_upload("b.tif", second)),
        )


class RefusalTests(TestCase):
    """The check itself, against rows created the way an import creates them."""

    def _import(self, payload: bytes, name: str = "scan.tif") -> Asset:
        """Import for real, and record the digest the call site records."""
        uploaded = _upload(name, payload)
        digest = refuse_duplicate_import(uploaded)
        detail = create_uploaded_asset(
            uploaded_file=uploaded,
            display_name=name,
            swallow_enqueue_errors=True,
        )
        asset = Asset.objects.get(id=detail["id"])
        asset.source_sha256 = digest
        asset.save(update_fields=["source_sha256", "updated_at"])
        return asset

    def test_the_same_bytes_under_a_different_name_are_refused(self):
        payload = _tiff_bytes()
        self._import(payload, "grid2_cell04.tif")

        with self.assertRaises(DuplicateImageError) as caught:
            refuse_duplicate_import(_upload("copy of grid2_cell04.tif", payload))

        self.assertEqual(caught.exception.error_code, DUPLICATE_IMPORT_ERROR_CODE)
        self.assertEqual(Asset.objects.count(), 1)

    def test_the_refusal_names_the_image_that_is_already_there(self):
        payload = _tiff_bytes()
        existing = self._import(payload, "grid2_cell04.tif")

        with self.assertRaises(DuplicateImageError) as caught:
            refuse_duplicate_import(_upload("grid2_cell04.tif", payload))

        message = str(caught.exception)
        self.assertIn('"grid2_cell04.tif"', message)
        self.assertIn("Nothing was imported", message)
        local = timezone.localtime(existing.created_at)
        self.assertIn(f"{local.day} {local:%B %Y}", message)
        # It has to fit in the row it is rendered in, and stay under the
        # length at which the client stops treating a body as a sentence
        # (frontend/src/utils/apiErrors.ts, MAX_PLAIN_TEXT_ERROR_LENGTH).
        self.assertLess(len(message), 200, message)

    def test_the_refusal_carries_the_identity_of_the_existing_image(self):
        """So a client can offer to open it instead of only printing a sentence."""
        payload = _tiff_bytes()
        existing = self._import(payload)

        with self.assertRaises(DuplicateImageError) as caught:
            refuse_duplicate_import(_upload("again.tif", payload))

        payload_body = caught.exception.payload
        self.assertEqual(payload_body["error_code"], DUPLICATE_IMPORT_ERROR_CODE)
        self.assertEqual(payload_body["duplicate_of"]["id"], str(existing.id))
        self.assertEqual(payload_body["duplicate_of"]["display_name"], existing.display_name)
        self.assertTrue(payload_body["duplicate_of"]["created_at"])
        self.assertEqual(payload_body["error"], str(caught.exception))

    def test_a_different_image_is_imported_normally(self):
        self._import(_tiff_bytes(seed=0))

        digest = refuse_duplicate_import(_upload("other.tif", _tiff_bytes(seed=57)))

        self.assertEqual(len(digest), 64)

    def test_an_import_still_being_prepared_already_counts(self):
        """The reported case: the same file dropped twice in one batch.

        The first row is nowhere near DONE when the second drop arrives, so a
        check that only looked at finished images would let the pair through.
        """
        payload = _tiff_bytes()
        first = self._import(payload)
        Asset.objects.filter(id=first.id).update(preprocess_stage="ENCODING")

        with self.assertRaises(DuplicateImageError):
            refuse_duplicate_import(_upload("scan.tif", payload))

    def test_a_failed_import_does_not_block_a_retry(self):
        """A user re-dropping a file that failed does not have that image."""
        payload = _tiff_bytes()
        first = self._import(payload)
        Asset.objects.filter(id=first.id).update(
            preprocess_stage="FAILED", preprocess_error="Error reading TIFF file"
        )

        digest = refuse_duplicate_import(_upload("scan.tif", payload))

        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_a_cancelled_import_does_not_block_a_retry(self):
        payload = _tiff_bytes()
        first = self._import(payload)
        Asset.objects.filter(id=first.id).update(preprocess_stage="CANCELLED")

        refuse_duplicate_import(_upload("scan.tif", payload))

    def test_a_deleted_image_does_not_block_importing_the_file_again(self):
        payload = _tiff_bytes()
        first = self._import(payload)
        Asset.objects.filter(id=first.id).update(
            lifecycle_status=Asset.LIFECYCLE_DELETED, deleted_at=timezone.now()
        )

        refuse_duplicate_import(_upload("scan.tif", payload))

    def test_an_image_with_no_recorded_digest_never_matches(self):
        """Images imported before the digest existed say "unknown", not "same".

        Their source file was deleted at the end of their own import, so there
        is nothing left to hash and a blank column is the honest answer.
        """
        payload = _tiff_bytes()
        first = self._import(payload)
        Asset.objects.filter(id=first.id).update(source_sha256="")

        refuse_duplicate_import(_upload("scan.tif", payload))
        self.assertIsNone(find_imported_asset_with_same_bytes(""))

    def test_asking_for_a_second_copy_on_purpose_is_allowed(self):
        payload = _tiff_bytes()
        self._import(payload)

        digest = refuse_duplicate_import(_upload("scan.tif", payload), allow_duplicate=True)

        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_the_oldest_matching_image_is_the_one_named(self):
        payload = _tiff_bytes()
        first = self._import(payload, "first.tif")
        second = refuse_duplicate_import(_upload("second.tif", payload), allow_duplicate=True)
        Asset.objects.create(
            display_name="second.tif",
            original_filename="second.tif",
            source_sha256=second,
        )

        with self.assertRaises(DuplicateImageError) as caught:
            refuse_duplicate_import(_upload("third.tif", payload))

        self.assertEqual(caught.exception.existing_asset_id, str(first.id))

    def test_the_refusal_is_app_copy(self):
        """I-12: no command, no module path, no endpoint in what a user reads."""
        payload = _tiff_bytes()
        self._import(payload, "grid2_cell04.tif")

        with self.assertRaises(DuplicateImageError) as caught:
            refuse_duplicate_import(_upload("grid2_cell04.tif", payload))

        violations = find_violations(str(caught.exception), "duplicate refusal")
        self.assertEqual(violations, [], "\n".join(str(v) for v in violations))

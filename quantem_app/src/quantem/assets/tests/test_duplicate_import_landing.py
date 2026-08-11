"""The duplicate check, at the door a user actually walks through.

``assets/tests/test_duplicate_import.py`` proves the *mechanism* works when it
is called. This file proves it is called. Wave 0c measured the difference:
four imports of one montage through the real endpoint produced **four images**,
and ``source_sha256`` was NULL on 25 of 25 rows, so nothing in the database
could even have found the duplicates afterwards. The whole apparatus --
:func:`~quantem.assets.models.sha256_of_upload`,
:func:`~quantem.assets.models.find_imported_asset_with_same_bytes`,
:class:`~quantem.assets.models.DuplicateImageError`, and the column migration
``0003`` -- sat in ``models.py`` with no caller.

**What a scientist experiences, and why.** The second import does not happen.
The refusal names the image that is already in the library, says nothing was
imported, and carries that image's identity so the client can offer to open it.
A researcher who genuinely wants a second copy passes ``allow_duplicate`` and
gets one, recorded as a copy of the first (:attr:`Asset.duplicate_of`).

The asymmetry is what settles it. A silent duplicate is invisible among forty
file names that all look alike, is counted twice in every per-group number, and
splits the user's proofreading across two rows with no way to say which one is
the answer -- and undoing it means noticing it, deciding which copy is real,
and deleting the other with its labels. Being told "you already have this" is
undone by one click, and the user's actual intent -- to have that image -- is
already satisfied.
"""

from __future__ import annotations

import hashlib
import io
import tracemalloc
from pathlib import Path
from uuid import uuid4

import numpy as np
import tifffile
from django.core.files.uploadedfile import SimpleUploadedFile, UploadedFile
from django.test import TestCase
from PIL import Image

from quantem.assets.asset_mutations import create_uploaded_asset
from quantem.assets.models import (
    DUPLICATE_IMPORT_ERROR_CODE,
    Asset,
    DuplicateImageError,
    sha256_of_upload,
)
from quantem.core.config import STORAGE_DIR, UPLOADS_DIR
from quantem.registry.tests.copy_gate import find_violations
from quantem.testing import make_em_like_array

UPLOAD_URL = "/api/assets/upload/"


def _tiff_bytes(*, seed: int = 0, width: int = 64, height: int = 48) -> bytes:
    """A real TIFF, so nothing here passes on bytes a reader would refuse."""
    array = make_em_like_array(width, height)
    if seed:
        array = np.clip(array.astype("int16") + seed, 0, 255).astype("uint8")
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array, photometric="minisblack")
    return buffer.getvalue()


def _png_bytes(*, seed: int = 0, width: int = 64, height: int = 48) -> bytes:
    array = make_em_like_array(width, height)
    if seed:
        array = np.clip(array.astype("int16") + seed, 0, 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, "PNG")
    return buffer.getvalue()


def _upload(name: str, payload: bytes) -> SimpleUploadedFile:
    return SimpleUploadedFile(name, payload, content_type="image/tiff")


def _snapshot() -> dict[str, int]:
    if not UPLOADS_DIR.exists():
        return {}
    return {p.name: p.stat().st_size for p in UPLOADS_DIR.iterdir() if p.is_file()}


class UploadsDirectoryTestCase(TestCase):
    """Photographs the staging directory and removes what the test added."""

    def setUp(self):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.before = _snapshot()

    def tearDown(self):
        for name in set(_snapshot()) - set(self.before):
            (UPLOADS_DIR / name).unlink(missing_ok=True)

    def assertNoNewBytes(self, message: str = ""):
        after = _snapshot()
        added = {n: s for n, s in after.items() if self.before.get(n) != s}
        self.assertEqual(
            added,
            {},
            f"{message}: {sum(added.values())} bytes no asset owns: {added}",
        )


class DigestIsRecordedTests(TestCase):
    """The column the whole feature depends on is written, on every import."""

    def test_an_imported_asset_records_the_digest_of_its_source_bytes(self):
        payload = _tiff_bytes()

        detail = create_uploaded_asset(
            uploaded_file=_upload("scan.tif", payload),
            swallow_enqueue_errors=True,
        )

        asset = Asset.objects.get(id=detail["id"])
        self.assertEqual(asset.source_sha256, hashlib.sha256(payload).hexdigest())

    def test_a_png_import_records_its_digest_too(self):
        payload = _png_bytes()

        detail = create_uploaded_asset(
            uploaded_file=SimpleUploadedFile("scan.png", payload, "image/png"),
            swallow_enqueue_errors=True,
        )

        asset = Asset.objects.get(id=detail["id"])
        self.assertEqual(asset.source_sha256, hashlib.sha256(payload).hexdigest())

    def test_no_new_asset_is_left_without_a_digest(self):
        """The wave-0c measurement was 25 of 25 NULL; the answer is 0 of N."""
        for seed in (0, 11, 23):
            create_uploaded_asset(
                uploaded_file=_upload(f"scan{seed}.tif", _tiff_bytes(seed=seed)),
                swallow_enqueue_errors=True,
            )

        self.assertEqual(Asset.objects.count(), 3)
        self.assertEqual(Asset.objects.filter(source_sha256="").count(), 0)
        self.assertEqual(Asset.objects.filter(source_sha256__isnull=True).count(), 0)


class SecondImportTests(UploadsDirectoryTestCase):
    """Dropping the same file twice."""

    def _import(self, name: str, payload: bytes, **kwargs) -> Asset:
        detail = create_uploaded_asset(
            uploaded_file=_upload(name, payload),
            display_name=name,
            swallow_enqueue_errors=True,
            **kwargs,
        )
        return Asset.objects.get(id=detail["id"])

    def test_the_same_bytes_twice_make_one_image(self):
        payload = _tiff_bytes()
        self._import("grid2_cell04.tif", payload)

        with self.assertRaises(DuplicateImageError):
            self._import("grid2_cell04.tif", payload)

        self.assertEqual(Asset.objects.count(), 1)

    def test_the_same_bytes_under_a_different_name_are_still_one_image(self):
        payload = _tiff_bytes()
        self._import("grid2_cell04.tif", payload)

        with self.assertRaises(DuplicateImageError):
            self._import("copy of grid2_cell04.tif", payload)

        self.assertEqual(Asset.objects.count(), 1)

    def test_different_bytes_under_the_same_name_are_two_images(self):
        """The dangerous direction: a microscope that names everything alike."""
        self._import("Image_001.tif", _tiff_bytes(seed=0))
        self._import("Image_001.tif", _tiff_bytes(seed=57))

        self.assertEqual(Asset.objects.count(), 2)
        digests = set(Asset.objects.values_list("source_sha256", flat=True))
        self.assertEqual(len(digests), 2)

    def test_the_refusal_names_the_image_and_points_at_it(self):
        payload = _tiff_bytes()
        existing = self._import("grid2_cell04.tif", payload)

        with self.assertRaises(DuplicateImageError) as caught:
            self._import("again.tif", payload)

        message = str(caught.exception)
        self.assertIn('"grid2_cell04.tif"', message)
        self.assertIn("Nothing was imported", message)
        body = caught.exception.payload
        self.assertEqual(body["error_code"], DUPLICATE_IMPORT_ERROR_CODE)
        self.assertEqual(body["duplicate_of"]["id"], str(existing.id))

    def test_the_refusal_costs_the_user_no_disk(self):
        payload = _tiff_bytes()
        self._import("grid2_cell04.tif", payload)
        before = _snapshot()

        with self.assertRaises(DuplicateImageError):
            self._import("grid2_cell04.tif", payload)

        after = _snapshot()
        self.assertEqual(
            {n: s for n, s in after.items() if before.get(n) != s},
            {},
            "the refused import staged its bytes anyway",
        )

    def test_a_failed_first_import_does_not_block_the_retry(self):
        """The user re-dropping a file that failed does not have that image."""
        payload = _tiff_bytes()
        first = self._import("scan.tif", payload)
        Asset.objects.filter(id=first.id).update(preprocess_stage="FAILED")

        self._import("scan.tif", payload)

        self.assertEqual(Asset.objects.count(), 2)


class DeliberateSecondCopyTests(TestCase):
    """ "I know, import it anyway" -- the door in the refusal."""

    def _import(self, name: str, payload: bytes, **kwargs) -> Asset:
        detail = create_uploaded_asset(
            uploaded_file=_upload(name, payload),
            display_name=name,
            swallow_enqueue_errors=True,
            **kwargs,
        )
        return Asset.objects.get(id=detail["id"])

    def test_a_second_copy_can_be_asked_for_on_purpose(self):
        payload = _tiff_bytes()
        self._import("scan.tif", payload)

        second = self._import("scan.tif", payload, allow_duplicate=True)

        self.assertEqual(Asset.objects.count(), 2)
        self.assertEqual(second.source_sha256, hashlib.sha256(payload).hexdigest())

    def test_the_second_copy_records_what_it_is_a_copy_of(self):
        """Otherwise the library has two identical rows and no way to say so."""
        payload = _tiff_bytes()
        first = self._import("scan.tif", payload)

        second = self._import("scan.tif", payload, allow_duplicate=True)

        self.assertEqual(second.duplicate_of_id, first.id)
        self.assertEqual(list(first.duplicate_copies.all()), [second])

    def test_a_first_import_is_not_marked_as_a_copy_of_anything(self):
        asset = self._import("scan.tif", _tiff_bytes())
        self.assertIsNone(asset.duplicate_of_id)

    def test_allowing_duplicates_does_not_stop_the_next_refusal(self):
        payload = _tiff_bytes()
        self._import("first.tif", payload)
        self._import("second.tif", payload, allow_duplicate=True)

        with self.assertRaises(DuplicateImageError):
            self._import("third.tif", payload)


class RefusalCopyTests(TestCase):
    """I-12: what the refusal says is fit to put in front of a biologist."""

    def test_the_refusal_carries_no_command_path_or_endpoint(self):
        payload = _tiff_bytes()
        create_uploaded_asset(
            uploaded_file=_upload("grid2_cell04.tif", payload),
            display_name="grid2_cell04.tif",
            swallow_enqueue_errors=True,
        )

        with self.assertRaises(DuplicateImageError) as caught:
            create_uploaded_asset(
                uploaded_file=_upload("grid2_cell04.tif", payload),
                swallow_enqueue_errors=True,
            )

        message = str(caught.exception)
        violations = find_violations(message, "duplicate refusal")
        self.assertEqual(violations, [], "\n".join(str(v) for v in violations))
        # No absolute path, no raw identifier, and short enough that the client
        # still treats it as a sentence (apiErrors.ts, 200 characters).
        self.assertNotIn(str(STORAGE_DIR), message)
        self.assertNotIn(":\\", message)
        self.assertLess(len(message), 200, message)


class UploadEndpointTests(UploadsDirectoryTestCase):
    """Through the real door, which is where wave 0c measured four images."""

    def test_the_second_upload_of_the_same_bytes_is_refused(self):
        payload = _tiff_bytes()

        first = self.client.post(UPLOAD_URL, {"file": _upload("grid2_cell04.tif", payload)})
        second = self.client.post(UPLOAD_URL, {"file": _upload("grid2_cell04.tif", payload)})

        self.assertEqual(first.status_code, 201, first.content[:300])
        # 400 today; 409 once the view answers with the identity payload. Both
        # carry the sentence in ``error``, which is what the client renders.
        self.assertIn(second.status_code, (400, 409), second.content[:300])
        body = second.json()
        self.assertIn("Nothing was imported", body["error"])
        self.assertIn('"grid2_cell04.tif"', body["error"])
        self.assertEqual(Asset.objects.count(), 1)

    def test_the_refused_upload_leaves_nothing_on_disk(self):
        payload = _tiff_bytes()
        self.client.post(UPLOAD_URL, {"file": _upload("scan.tif", payload)})
        before = _snapshot()

        self.client.post(UPLOAD_URL, {"file": _upload("scan.tif", payload)})

        after = _snapshot()
        self.assertEqual({n: s for n, s in after.items() if before.get(n) != s}, {})

    def test_an_upload_through_the_endpoint_records_its_digest(self):
        payload = _tiff_bytes()

        response = self.client.post(UPLOAD_URL, {"file": _upload("scan.tif", payload)})

        asset = Asset.objects.get(id=response.json()["id"])
        self.assertEqual(asset.source_sha256, hashlib.sha256(payload).hexdigest())

    def test_four_drops_of_one_file_make_one_image(self):
        """The wave-0c reproduction, verbatim: four imports, four images."""
        payload = _tiff_bytes()
        names = ["montage.tif", "montage.tif", "montage copy.tif", "montage (1).tif"]

        codes = [
            self.client.post(UPLOAD_URL, {"file": _upload(name, payload)}).status_code
            for name in names
        ]

        self.assertEqual(codes[0], 201)
        self.assertTrue(all(code in (400, 409) for code in codes[1:]), codes)
        self.assertEqual(Asset.objects.count(), 1)


class HashingIsStreamedTests(TestCase):
    """A 2-3 GB import must not gain a full-size buffer (owner ruling R3)."""

    def test_hashing_a_file_backed_upload_does_not_buffer_it(self):
        payload = b"".join(
            bytes([(index * 7 + 13) % 256]) * 65536 for index in range(128)
        )  # 8 MiB, on disk, read through a real file handle
        path = Path(STORAGE_DIR) / "tmp" / f"hashstream_{uuid4().hex}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        try:
            with open(path, "rb") as handle:
                uploaded = UploadedFile(handle, "big.tif", "image/tiff", len(payload), None)
                tracemalloc.start()
                try:
                    digest = sha256_of_upload(uploaded)
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        # One chunk at a time, not one image at a time. Generous headroom over
        # the 1 MiB chunk so this asserts the shape, not an allocator detail.
        self.assertLess(
            peak,
            len(payload) // 2,
            f"hashing peaked at {peak} bytes for an {len(payload)} byte file",
        )

    def test_hashing_leaves_the_upload_readable_for_the_import(self):
        payload = _tiff_bytes()
        uploaded = _upload("scan.tif", payload)

        sha256_of_upload(uploaded)

        self.assertEqual(b"".join(uploaded.chunks()), payload)

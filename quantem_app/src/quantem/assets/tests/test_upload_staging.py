"""The upload lands in the staging directory once, and is claimed by rename."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import tifffile
from django.core.files.uploadedfile import SimpleUploadedFile, TemporaryUploadedFile
from django.test import TestCase

from quantem.assets.models import Asset
from quantem.assets.upload_staging import (
    STAGING_PREFIX,
    StagedFileUploadHandler,
    StagedUploadedFile,
    staged_upload_handlers,
)
from quantem.assets.utils import save_uploaded_file_to_path
from quantem.core.config import STORAGE_DIR, UPLOADS_DIR
from quantem.testing import make_em_like_array


def _staged_files() -> list[Path]:
    if not UPLOADS_DIR.exists():
        return []
    return sorted(UPLOADS_DIR.glob(f"{STAGING_PREFIX}*"))


class StagedUploadedFileTests(TestCase):
    def setUp(self):
        for leftover in _staged_files():
            leftover.unlink(missing_ok=True)

    def test_writes_into_the_staging_directory_not_the_system_temp_dir(self):
        """Where the bytes land, stated as a property of the storage layout.

        The assertion this replaces was ``gettempdir() not in path.parents``.
        That is not the claim the handler makes, and it fails on a perfectly
        healthy configuration: any run whose ``QUANTEM_DATA_DIR`` happens to
        sit *under* the system temp directory -- which is what a verification
        harness that puts its whole scratch tree in ``TEMP`` produces --
        tripped it with no product defect involved.

        The real claim is narrower and stronger, so it is asserted directly:
        the file is created inside the application's own staging directory,
        on the same volume as where it is going (that is what makes the claim
        a rename rather than a copy), and *not* loose in the system temp
        directory the way Django's own ``TemporaryUploadedFile`` creates it.
        The last clause is checked against a real ``TemporaryUploadedFile``
        built with stock settings rather than against a path expression, so
        it keeps meaning if Django changes where its default handler writes.
        """
        system_temp = Path(tempfile.gettempdir()).resolve()
        staged = StagedUploadedFile("scan.TIF", "image/tiff", 0, "utf-8")
        try:
            path = Path(staged.temporary_file_path()).resolve()
            self.assertEqual(path.parent, UPLOADS_DIR.resolve())
            self.assertEqual(path.suffix, ".tif")
            self.assertTrue(
                path.is_relative_to(Path(STORAGE_DIR).resolve()),
                "the upload must be staged inside the app's own storage",
            )
            self.assertEqual(
                path.drive,
                UPLOADS_DIR.resolve().drive,
                "staging on another volume turns the claim into a copy",
            )
            self.assertNotEqual(
                path.parent,
                system_temp,
                "the upload must not be dropped loose in the system temp dir",
            )
        finally:
            staged.close()

        # What "not the system temp dir" is measured against: Django's own
        # handler, with this application's redirect switched off.
        with self.settings(FILE_UPLOAD_TEMP_DIR=None):
            default = TemporaryUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
            try:
                default_path = Path(default.temporary_file_path()).resolve()
            finally:
                default.close()
        self.assertEqual(
            default_path.parent,
            system_temp,
            "guard: Django's stock handler is expected to write to the system "
            "temp directory, and the comparison below is pointless if it does not",
        )
        self.assertNotEqual(path.parent, default_path.parent)

    def test_claim_moves_the_bytes_and_survives_close(self):
        staged = StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
        staged.write(b"electron microscopy")
        source_path = Path(staged.temporary_file_path())
        target = UPLOADS_DIR / f"{uuid4()}.tif"

        staged.claim(target)

        self.assertFalse(source_path.exists())
        self.assertTrue(target.exists())
        staged.close()
        self.assertTrue(target.exists(), "close must not delete a claimed upload")
        self.assertEqual(target.read_bytes(), b"electron microscopy")
        target.unlink()

    def test_an_unclaimed_upload_is_removed_on_close(self):
        staged = StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
        staged.write(b"abandoned")
        path = Path(staged.temporary_file_path())
        self.assertTrue(path.exists())

        staged.close()

        self.assertFalse(path.exists())
        self.assertEqual(_staged_files(), [])

    def test_claiming_twice_is_refused(self):
        staged = StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
        staged.write(b"x")
        target = UPLOADS_DIR / f"{uuid4()}.tif"
        staged.claim(target)
        try:
            with self.assertRaises(RuntimeError):
                staged.claim(UPLOADS_DIR / f"{uuid4()}.tif")
        finally:
            staged.close()
            target.unlink(missing_ok=True)


class SaveUploadedFileToPathTests(TestCase):
    def test_a_staged_upload_is_claimed_rather_than_copied(self):
        staged = StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
        staged.write(b"payload")
        source_path = Path(staged.temporary_file_path())
        target = UPLOADS_DIR / f"{uuid4()}.tif"

        save_uploaded_file_to_path(staged, target)

        self.assertFalse(source_path.exists(), "the bytes should have moved, not copied")
        self.assertEqual(target.read_bytes(), b"payload")
        staged.close()
        self.assertTrue(target.exists())
        target.unlink()

    def test_a_plain_uploaded_file_is_still_copied(self):
        uploaded = SimpleUploadedFile("scan.tif", b"payload", content_type="image/tiff")
        target = STORAGE_DIR / "tmp" / f"copied_{uuid4().hex}.tif"
        try:
            save_uploaded_file_to_path(uploaded, target)
            self.assertEqual(target.read_bytes(), b"payload")
        finally:
            target.unlink(missing_ok=True)

    def test_djangos_own_temporary_file_is_copied_not_moved(self):
        # The regression this guards: os.replace on an O_TEMPORARY handle can
        # succeed and then be undone by close(), losing the upload.
        uploaded = TemporaryUploadedFile("scan.tif", "image/tiff", 7, "utf-8")
        uploaded.write(b"payload")
        uploaded.file.flush()
        source_path = Path(uploaded.temporary_file_path())
        target = STORAGE_DIR / "tmp" / f"copied_{uuid4().hex}.tif"
        try:
            save_uploaded_file_to_path(uploaded, target)
            self.assertTrue(source_path.exists(), "Django's temp file must not be moved")
            self.assertEqual(target.read_bytes(), b"payload")
            uploaded.close()
            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), b"payload")
        finally:
            target.unlink(missing_ok=True)


class UploadEndpointStagingTests(TestCase):
    """End to end: the handler is installed and nothing is left behind."""

    def setUp(self):
        for leftover in _staged_files():
            leftover.unlink(missing_ok=True)

    def _tiff_bytes(self, width: int = 64, height: int = 48) -> bytes:
        path = STORAGE_DIR / "tmp" / f"upload_{uuid4().hex}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            str(path), make_em_like_array(width, height), photometric="minisblack"
        )
        try:
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)

    def test_staging_is_the_only_handler(self):
        # This used to be [MemoryFileUploadHandler, StagedFileUploadHandler],
        # which meant every upload under FILE_UPLOAD_MAX_MEMORY_SIZE took a
        # path with no cleanup on it. See test_upload_leak_sizes.py.
        handlers = staged_upload_handlers(None)
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0], StagedFileUploadHandler)

    def test_upload_creates_an_asset_and_leaves_no_incoming_file(self):
        payload = self._tiff_bytes()
        with self.settings(FILE_UPLOAD_MAX_MEMORY_SIZE=0):
            response = self.client.post(
                "/api/assets/upload/",
                {
                    "file": SimpleUploadedFile(
                        "scan.tif", payload, content_type="image/tiff"
                    ),
                    "display_name": "staged upload",
                },
            )
        self.assertEqual(response.status_code, 201, response.content[:400])
        asset_id = response.json()["id"]
        asset = Asset.objects.get(id=asset_id)
        staged = UPLOADS_DIR / f"{asset.id}.tif"
        self.assertTrue(staged.exists(), "the claimed upload should be the asset's file")
        self.assertEqual(staged.read_bytes(), payload)
        self.assertEqual(_staged_files(), [], "no unclaimed staging files left behind")

    def test_a_rejected_upload_leaves_nothing_in_the_staging_directory(self):
        before = set(UPLOADS_DIR.glob("*")) if UPLOADS_DIR.exists() else set()
        with self.settings(FILE_UPLOAD_MAX_MEMORY_SIZE=0):
            response = self.client.post(
                "/api/assets/upload/",
                {
                    "file": SimpleUploadedFile(
                        "notes.mrc", b"not an image", content_type="application/octet-stream"
                    )
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(_staged_files(), [])
        after = set(UPLOADS_DIR.glob("*")) if UPLOADS_DIR.exists() else set()
        self.assertEqual(after, before)

    def test_the_staged_bytes_survive_the_end_of_the_request(self):
        # The whole point: Django closes every uploaded file when the request
        # ends, and the claimed file must not be caught by that.
        payload = self._tiff_bytes(80, 60)
        with self.settings(FILE_UPLOAD_MAX_MEMORY_SIZE=0):
            response = self.client.post(
                "/api/assets/upload/",
                {"file": SimpleUploadedFile("scan.tif", payload, content_type="image/tiff")},
            )
        asset_id = response.json()["id"]
        staged = UPLOADS_DIR / f"{asset_id}.tif"
        self.assertTrue(staged.exists())
        with tifffile.TiffFile(str(staged)) as tif:
            self.assertEqual(tif.series[0].shape, (60, 80))
        self.assertEqual(os.path.getsize(staged), len(payload))
        np.testing.assert_array_equal(
            tifffile.imread(str(staged)), make_em_like_array(80, 60)
        )

"""Rejected, interrupted and abandoned uploads must not keep their bytes.

Two leaks, both measured on a real session (wave-0 verification, finding F3):

* A ``.tif`` whose *contents* are not a TIFF passes the extension check, so the
  staged bytes are **claimed** into ``<asset id>.tif`` before
  ``extract_image_metadata`` rejects the file. The request answers 400 and the
  full body stays on disk forever -- 3 000 000 B and 500 008 B were still there
  six minutes and one server restart later.
* Killing the server mid-import left ``incoming-489aa6a9….tif``, **2 074 034
  677 B**, and nothing in the tree ever swept ``UPLOADS_DIR``.

Storage lives with the installation, so both leaks eat the volume the user
installed onto with no trace in any screen.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from unittest import mock

import tifffile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from quantem.assets.models import Asset, Rendition
from quantem.assets.upload_staging import (
    DEFAULT_ABANDONED_UPLOAD_MAX_AGE_SECONDS,
    STAGING_PREFIX,
    StagedUploadedFile,
    abandoned_upload_max_age_seconds,
    sweep_abandoned_uploads,
)
from quantem.core.config import DATA_DIR, STORAGE_DIR, UPLOADS_DIR
from quantem.core.local_storage import normalize_stored_path_value
from quantem.testing import make_em_like_array


def _upload_dir_entries() -> set[Path]:
    if not UPLOADS_DIR.exists():
        return set()
    return set(UPLOADS_DIR.glob("*"))


def _age_file(path: Path, seconds: float) -> None:
    """Backdate ``path`` so an age-gated sweep considers it abandoned."""
    when = time.time() - seconds
    os.utime(path, (when, when))


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class RejectedUploadLeavesNoBytesTests(TestCase):
    """Every rejection path, not only the one the extension check catches."""

    def setUp(self):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.before = _upload_dir_entries()

    def _new_entries(self) -> set[Path]:
        return _upload_dir_entries() - self.before

    def _tiff_bytes(self, width: int = 64, height: int = 48) -> bytes:
        path = STORAGE_DIR / "tmp" / f"sweep_src_{uuid.uuid4().hex}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path), make_em_like_array(width, height), photometric="minisblack")
        try:
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)

    def test_a_tif_that_is_not_a_tiff_leaves_no_bytes_behind(self):
        # The measured leak. The extension passes, so the bytes are claimed to
        # <asset id>.tif and only then does the reader reject them.
        payload = b"NOT-A-TIFF" * 300_000  # 3 000 000 B, the size in the finding
        with self.settings(FILE_UPLOAD_MAX_MEMORY_SIZE=0):
            response = self.client.post(
                "/api/assets/upload/",
                {"file": SimpleUploadedFile("scan.tif", payload, content_type="image/tiff")},
            )

        self.assertEqual(response.status_code, 400, response.content[:400])
        leaked = self._new_entries()
        self.assertEqual(
            leaked,
            set(),
            "a rejected upload kept its full staged body: "
            + ", ".join(f"{p.name} ({p.stat().st_size} B)" for p in sorted(leaked)),
        )

    def test_an_upload_that_fails_after_the_claim_leaves_no_bytes_behind(self):
        # Not a 400 this time: anything raising after the bytes are claimed used
        # to leak them, including the 500 arm of the view. The fix has to cover
        # the rejection paths nobody has written yet, not just the known one.
        payload = self._tiff_bytes()
        with (
            self.settings(FILE_UPLOAD_MAX_MEMORY_SIZE=0),
            mock.patch(
                "quantem.assets.asset_mutations.extract_image_metadata",
                side_effect=RuntimeError("reader exploded"),
            ),
        ):
            response = self.client.post(
                "/api/assets/upload/",
                {"file": SimpleUploadedFile("scan.tif", payload, content_type="image/tiff")},
            )

        self.assertEqual(response.status_code, 500, response.content[:400])
        self.assertEqual(self._new_entries(), set())

    def test_a_successful_upload_keeps_its_bytes(self):
        # The control: cleanup that also deletes accepted uploads would pass the
        # two tests above and destroy the library.
        payload = self._tiff_bytes(80, 60)
        with self.settings(FILE_UPLOAD_MAX_MEMORY_SIZE=0):
            response = self.client.post(
                "/api/assets/upload/",
                {"file": SimpleUploadedFile("scan.tif", payload, content_type="image/tiff")},
            )

        self.assertEqual(response.status_code, 201, response.content[:400])
        asset_id = response.json()["id"]
        kept = UPLOADS_DIR / f"{asset_id}.tif"
        self.assertTrue(kept.exists(), "the accepted upload's bytes were deleted")
        self.assertEqual(kept.read_bytes(), payload)


class SweepAbandonedUploadsTests(TestCase):
    """The sweeper: age plus liveness plus 'is anything pointing at it'."""

    def setUp(self):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.old = DEFAULT_ABANDONED_UPLOAD_MAX_AGE_SECONDS + 60
        # The staging directory is shared with every other test in this run and
        # with whatever the last one left. Clear anything already old enough to
        # sweep so that `freed_bytes` below counts this test's file and nothing
        # else -- including a file an earlier test kept because of a rendition
        # row the test database has since rolled back.
        sweep_abandoned_uploads()

    def _asset_with_rendition(self, path: Path) -> Asset:
        asset = Asset.objects.create(
            display_name="referenced",
            original_filename=path.name,
            logical_width=4,
            logical_height=4,
            channels=1,
            bit_depth=8,
        )
        Rendition.objects.create(
            asset=asset,
            type=Rendition.TYPE_FULL,
            storage_root="DATA_DIR",
            stored_path=normalize_stored_path_value(path, relative_to=DATA_DIR),
            path_exists=True,
            is_directory=False,
        )
        return asset

    def test_an_old_interrupted_staging_file_is_swept(self):
        # The 2 GB file the killed server left behind, in miniature.
        path = _write(UPLOADS_DIR / f"{STAGING_PREFIX}{uuid.uuid4().hex}.tif", b"x" * 4096)
        _age_file(path, self.old)

        result = sweep_abandoned_uploads()

        self.assertFalse(path.exists())
        self.assertIn(path, result.removed)
        self.assertEqual(result.freed_bytes, 4096)

    def test_a_staging_file_that_could_still_be_in_flight_is_kept(self):
        path = _write(UPLOADS_DIR / f"{STAGING_PREFIX}{uuid.uuid4().hex}.tif", b"still arriving")

        result = sweep_abandoned_uploads()

        self.assertTrue(path.exists(), "a fresh upload must never be swept")
        self.assertNotIn(path, result.removed)
        path.unlink()

    def test_a_staging_file_open_in_this_process_is_kept_however_old(self):
        # Age alone is not enough: a slow body can be written by a handle whose
        # last-write time has not moved for minutes.
        staged = StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
        path = Path(staged.temporary_file_path())
        staged.write(b"chunk")
        _age_file(path, self.old)
        try:
            result = sweep_abandoned_uploads()
            self.assertTrue(path.exists(), "swept an upload this process is writing")
            self.assertNotIn(path, result.removed)
        finally:
            staged.close()

    def test_an_old_orphan_no_rendition_points_at_is_swept(self):
        # Bytes claimed into <asset id>.tif by a request that then died: not
        # prefixed, so only the database can say they are nobody's.
        path = _write(UPLOADS_DIR / f"{uuid.uuid4()}.tif", b"y" * 512)
        _age_file(path, self.old)

        result = sweep_abandoned_uploads()

        self.assertFalse(path.exists())
        self.assertIn(path, result.removed)

    def test_an_old_file_a_rendition_points_at_is_never_swept(self):
        # The library's imported originals live here forever. Sweeping one is
        # data loss, not housekeeping.
        path = _write(UPLOADS_DIR / f"{uuid.uuid4()}.tif", b"the user's image")
        self.addCleanup(path.unlink, True)
        self._asset_with_rendition(path)
        _age_file(path, self.old)

        result = sweep_abandoned_uploads()

        self.assertTrue(path.exists(), "swept an image the library still lists")
        self.assertNotIn(path, result.removed)
        self.assertEqual(path.read_bytes(), b"the user's image")

    def test_a_directory_is_left_alone(self):
        directory = UPLOADS_DIR / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
        directory.mkdir(parents=True, exist_ok=True)
        _age_file(directory, self.old)

        result = sweep_abandoned_uploads()

        self.assertTrue(directory.is_dir())
        self.assertNotIn(directory, result.removed)
        directory.rmdir()

    def test_the_age_threshold_is_configurable_and_has_a_floor(self):
        with mock.patch.dict(os.environ, {"QUANTEM_UPLOAD_SWEEP_MAX_AGE_SECONDS": "900"}):
            self.assertEqual(abandoned_upload_max_age_seconds(), 900)
        with mock.patch.dict(os.environ, {"QUANTEM_UPLOAD_SWEEP_MAX_AGE_SECONDS": "1"}):
            # A threshold shorter than a plausible upload would delete bytes out
            # from under the request writing them.
            self.assertGreaterEqual(abandoned_upload_max_age_seconds(), 300)
        with mock.patch.dict(os.environ, {"QUANTEM_UPLOAD_SWEEP_MAX_AGE_SECONDS": "not a number"}):
            self.assertEqual(
                abandoned_upload_max_age_seconds(),
                DEFAULT_ABANDONED_UPLOAD_MAX_AGE_SECONDS,
            )

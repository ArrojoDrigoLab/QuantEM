"""A refused import leaves nothing behind, at every size.

The defect these tests exist for, raised by two verifiers in a row. The import
endpoint used to install ``[MemoryFileUploadHandler, StagedFileUploadHandler]``,
so a body under ``FILE_UPLOAD_MAX_MEMORY_SIZE`` (2 621 440 B) never became a
:class:`~quantem.assets.upload_staging.StagedUploadedFile` -- while
``create_uploaded_asset`` still claimed it into the staging directory under the
id of an asset that, for a file the readers cannot open, was never created. The
whole cleanup lived in ``StagedUploadedFile.close``, which that path never
reached. MEASURED before the fix, from an empty staging directory: fifteen
refused imports across five sizes left **15 600 000 B** with no asset row,
surviving a restart, because the hourly sweep will not touch a file younger
than an hour.

The bug is not "small files leak". The bug is that a size threshold sat in
front of a cleanup path, so the cleanup was only ever exercised at the sizes
the tests happened to use -- and the suite used one size, above the threshold.
Hence the shape of this file: every case is parametrised across sizes that
straddle 2 621 440 B, **at the real setting**. A test that pins
``FILE_UPLOAD_MAX_MEMORY_SIZE`` cannot see this class of defect at all, which
is exactly how it survived the first fix.

What is asserted throughout is bytes, not file counts: the staging directory is
photographed before and after and the two are compared entry by entry.
"""

from __future__ import annotations

import io
import os
import threading
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.files.uploadhandler import MemoryFileUploadHandler
from django.test import TestCase

from quantem.assets.models import Asset
from quantem.assets.upload_staging import (
    STAGING_PREFIX,
    StagedFileUploadHandler,
    StagedUploadedFile,
    discard_upload_if_unreferenced,
    staged_upload_handlers,
    sweep_abandoned_uploads,
)
from quantem.core.config import UPLOADS_DIR

#: Bodies either side of Django's in-memory threshold. The first four are under
#: it and are the sizes that leaked; the last is over it. Keep both sides.
MEMORY_THRESHOLD_BYTES = 2_621_440
UPLOAD_SIZES = (100_000, 500_000, 2_000_000, 2_600_000, 5_000_000)


def _snapshot() -> dict[str, int]:
    """``{name: size}`` for every file in the staging directory."""
    if not UPLOADS_DIR.exists():
        return {}
    return {p.name: p.stat().st_size for p in UPLOADS_DIR.iterdir() if p.is_file()}


def _readable_tiff(target_bytes: int) -> bytes:
    """A TIFF the readers accept, of roughly ``target_bytes``."""
    side = max(16, int((target_bytes / 2) ** 0.5))
    rng = np.random.default_rng(11)
    array = rng.integers(0, 65535, size=(side, side), dtype=np.uint16)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array, photometric="minisblack")
    return buffer.getvalue()


def _tiff_magic_then_noise(target_bytes: int) -> bytes:
    """Passes the suffix check and the magic number, fails the reader.

    This is the shape of a genuine mis-drop: a truncated or wrong-endian
    acquisition file that still opens far enough to fail late.
    """
    return b"II*\x00" + os.urandom(max(0, target_bytes - 4))


def _noise(target_bytes: int) -> bytes:
    return os.urandom(target_bytes)


#: ``(label, filename, body builder, extra form fields)``. Every refusal this
#: endpoint can be made to give, on both sides of the claim: the first two are
#: refused before the bytes are claimed, the rest after.
REJECTION_CASES = (
    ("unsupported suffix", "acquisition.mrc", _noise, {}),
    ("no suffix at all", "acquisition", _noise, {}),
    ("pixel size not a number", "scan.tif", _readable_tiff, {"pixel_size_nm": "abc"}),
    ("pixel size not positive", "scan.tif", _readable_tiff, {"pixel_size_nm": "-5"}),
    ("tiff magic, unreadable body", "scan.tif", _tiff_magic_then_noise, {}),
    ("not a tiff at all", "scan.tif", _noise, {}),
    ("not a png at all", "scan.png", _noise, {}),
)


class UploadsDirectoryTestCase(TestCase):
    """Photographs the staging directory and cleans up what the test added."""

    def setUp(self):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.before = _snapshot()

    def tearDown(self):
        for name in set(_snapshot()) - set(self.before):
            (UPLOADS_DIR / name).unlink(missing_ok=True)

    def assertNoNewBytes(self, message: str = ""):
        after = _snapshot()
        added = {name: n for name, n in after.items() if self.before.get(name) != n}
        self.assertEqual(
            added,
            {},
            f"{message}: the staging directory grew by "
            f"{sum(added.values())} bytes that no asset owns",
        )

    def post_upload(self, filename: str, body: bytes, fields: dict | None = None):
        payload = {
            "file": SimpleUploadedFile(filename, body, content_type="application/octet-stream"),
            **(fields or {}),
        }
        return self.client.post("/api/assets/upload/", payload)


class HandlerContractTests(TestCase):
    """The handler list is the fix; assert it directly so it cannot come back."""

    def test_the_import_endpoint_installs_exactly_one_handler(self):
        handlers = staged_upload_handlers(None)
        self.assertEqual(
            [type(handler).__name__ for handler in handlers],
            [StagedFileUploadHandler.__name__],
        )

    def test_no_handler_keeps_an_upload_in_memory(self):
        # Named explicitly because this is the exact regression: the handler
        # that answers "is this small?" by keeping the body out of
        # StagedUploadedFile, and therefore out of the only cleanup there is.
        for handler in staged_upload_handlers(None):
            self.assertNotIsInstance(
                handler,
                MemoryFileUploadHandler,
                "an upload small enough for this handler would have no "
                "cleanup on its rejection path",
            )

    def test_the_sizes_under_test_really_do_straddle_the_threshold(self):
        # If a future settings change moved the threshold, the parametrisation
        # below would quietly stop covering both sides of it and this file
        # would go back to testing one path.
        self.assertEqual(settings.FILE_UPLOAD_MAX_MEMORY_SIZE, MEMORY_THRESHOLD_BYTES)
        self.assertTrue(any(size < MEMORY_THRESHOLD_BYTES for size in UPLOAD_SIZES))
        self.assertTrue(any(size > MEMORY_THRESHOLD_BYTES for size in UPLOAD_SIZES))


class RejectedUploadLeavesNoBytesTests(UploadsDirectoryTestCase):
    """Every refusal, at every size, costs zero bytes of the user's disk."""

    def test_every_rejection_reason_at_every_size_leaves_nothing(self):
        for size in UPLOAD_SIZES:
            for label, filename, build_body, fields in REJECTION_CASES:
                with self.subTest(size=size, reason=label):
                    before = _snapshot()
                    response = self.post_upload(filename, build_body(size), fields)
                    self.assertEqual(
                        response.status_code,
                        400,
                        f"{label} at {size} B: {response.content[:300]!r}",
                    )
                    after = _snapshot()
                    added = {name: n for name, n in after.items() if before.get(name) != n}
                    self.assertEqual(
                        added,
                        {},
                        f"{label} at {size} B left {sum(added.values())} bytes behind: {added}",
                    )
                    self.assertEqual(
                        Asset.objects.count(),
                        0,
                        f"{label} at {size} B created an asset row",
                    )

    def test_a_refusal_says_something_a_person_can_act_on(self):
        # Adjacent to the leak and cheap to guard: the refusal must not hand the
        # user a request method, an address, or an import path.
        response = self.post_upload("acquisition.mrc", _noise(100_000))
        message = response.json()["error"]
        self.assertIn(".tif", message)
        for forbidden in ("POST", "/api/", "quantem.", "Traceback"):
            self.assertNotIn(forbidden, message)

    def test_four_mis_drops_in_a_row_cost_nothing(self):
        # The verifier's exact reproduction: half a megabyte, four times over.
        for _ in range(4):
            response = self.post_upload("scan.tif", _tiff_magic_then_noise(500_000))
            self.assertEqual(response.status_code, 400)
        self.assertNoNewBytes("four refused half-megabyte imports")

    def test_a_failure_that_is_not_a_refusal_also_costs_nothing(self):
        # A 500 is the path nobody enumerates. The cleanup does not ask why the
        # import stopped, only whether anything now points at the bytes.
        with mock.patch(
            "quantem.assets.asset_mutations.extract_image_metadata",
            side_effect=RuntimeError("the reader fell over"),
        ):
            response = self.post_upload("scan.tif", _readable_tiff(500_000))
        self.assertEqual(response.status_code, 500)
        self.assertNoNewBytes("an import that failed after the claim")

    def test_a_second_file_field_is_not_left_on_disk(self):
        # Only "file" is read. Anything else posted alongside it is still
        # streamed to disk by the handler and must still be released.
        response = self.client.post(
            "/api/assets/upload/",
            {
                "file": SimpleUploadedFile(
                    "scan.tif", _tiff_magic_then_noise(500_000), content_type="image/tiff"
                ),
                "sidecar": SimpleUploadedFile(
                    "notes.tif", _noise(500_000), content_type="image/tiff"
                ),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertNoNewBytes("a refused import posted with a second file field")


class AcceptedUploadTests(UploadsDirectoryTestCase):
    """The success path is unchanged, at every size, byte for byte."""

    def test_an_accepted_upload_keeps_exactly_its_own_bytes(self):
        for size in UPLOAD_SIZES:
            with self.subTest(size=size):
                payload = _readable_tiff(size)
                before = _snapshot()
                response = self.post_upload(f"scan_{size}.tif", payload)
                self.assertEqual(response.status_code, 201, response.content[:300])
                asset_id = response.json()["id"]
                added = {name: n for name, n in _snapshot().items() if before.get(name) != n}
                self.assertEqual(
                    added,
                    {f"{asset_id}.tif": len(payload)},
                    "an accepted upload must keep its body and nothing else",
                )
                kept = UPLOADS_DIR / f"{asset_id}.tif"
                self.assertEqual(kept.read_bytes(), payload)
                with tifffile.TiffFile(str(kept)) as tif:
                    self.assertEqual(len(tif.series[0].shape), 2)

    def test_no_unclaimed_staging_file_survives_a_success(self):
        response = self.post_upload("scan.tif", _readable_tiff(500_000))
        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            sorted(UPLOADS_DIR.glob(f"{STAGING_PREFIX}*")),
            [],
            "the claimed upload should have been renamed, not copied",
        )


class MixedSessionAccountingTests(UploadsDirectoryTestCase):
    """Every byte in the staging directory is accounted for after a session."""

    def test_accepted_and_rejected_interleaved(self):
        expected: dict[str, int] = {}
        for size in UPLOAD_SIZES:
            refused = self.post_upload("scan.tif", _tiff_magic_then_noise(size))
            self.assertEqual(refused.status_code, 400)

            payload = _readable_tiff(size)
            accepted = self.post_upload(f"scan_{size}.tif", payload)
            self.assertEqual(accepted.status_code, 201, accepted.content[:300])
            expected[f"{accepted.json()['id']}.tif"] = len(payload)

            refused_again = self.post_upload("acquisition.mrc", _noise(size))
            self.assertEqual(refused_again.status_code, 400)

        added = {name: n for name, n in _snapshot().items() if self.before.get(name) != n}
        self.assertEqual(added, expected)
        self.assertEqual(
            sum(added.values()),
            sum(expected.values()),
            "the directory holds exactly the accepted images and nothing else",
        )

    def test_a_restart_sweep_removes_nothing_it_should_not(self):
        """What the scheduler runs at start-up, over this session's files.

        Two things a restart must not do: cost the user an accepted image, and
        find leaked bytes still waiting to be collected. Only files this test
        made are judged -- the suite's staging directory is shared with every
        other test process on this machine, so a sweep with the age guard
        switched off would be reaching into other people's work.
        """
        payload = _readable_tiff(500_000)
        accepted = self.post_upload("scan.tif", payload)
        self.assertEqual(accepted.status_code, 201)
        kept = UPLOADS_DIR / f"{accepted.json()['id']}.tif"

        refused = self.post_upload("scan.tif", _tiff_magic_then_noise(500_000))
        self.assertEqual(refused.status_code, 400)

        mine = {name for name in _snapshot() if name not in self.before}
        self.assertEqual(
            mine,
            {kept.name},
            "a refused import was still on disk when the restart began",
        )
        # Old enough for the sweep to consider, which a fresh upload is not.
        two_hours_ago = kept.stat().st_mtime - 7200
        os.utime(kept, (two_hours_ago, two_hours_ago))

        result = sweep_abandoned_uploads()

        self.assertTrue(kept.exists(), "the sweep deleted an accepted image")
        self.assertEqual(kept.read_bytes(), payload)
        self.assertEqual(
            [path for path in result.removed if path.name in mine],
            [],
            "the sweep still had leaked bytes from this session to collect, "
            "so a refusal did not clean up after itself",
        )


class InterruptedUploadTests(UploadsDirectoryTestCase):
    """A body that never finishes arriving does not become a permanent file."""

    def test_an_interrupted_upload_releases_its_partial_body(self):
        handler = StagedFileUploadHandler(None)
        handler.new_file(
            field_name="file",
            file_name="scan.tif",
            content_type="image/tiff",
            content_length=None,
            charset="utf-8",
            content_type_extra={},
        )
        handler.receive_data_chunk(_noise(500_000), 0)
        partial = Path(handler.file.temporary_file_path())
        self.assertTrue(partial.exists())

        handler.upload_interrupted()

        self.assertFalse(partial.exists(), "an aborted body was left on disk")
        self.assertNoNewBytes("an aborted upload")

    def test_closing_twice_is_harmless(self):
        staged = StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
        staged.write(b"partial")
        staged.close()
        staged.close()
        self.assertNoNewBytes("a staged upload closed twice")


class DiscardHelperTests(UploadsDirectoryTestCase):
    """The hook the import path needs, and everything it refuses to touch."""

    def test_it_removes_an_orphaned_claim(self):
        orphan = UPLOADS_DIR / f"{uuid.uuid4()}.tif"
        orphan.write_bytes(_noise(1024))

        self.assertTrue(discard_upload_if_unreferenced(orphan))

        self.assertFalse(orphan.exists())

    def test_it_keeps_a_file_an_asset_still_owns(self):
        payload = _readable_tiff(100_000)
        response = self.post_upload("scan.tif", payload)
        self.assertEqual(response.status_code, 201)
        kept = UPLOADS_DIR / f"{response.json()['id']}.tif"

        self.assertFalse(discard_upload_if_unreferenced(kept))

        self.assertTrue(kept.exists())
        self.assertEqual(kept.read_bytes(), payload)

    def test_it_keeps_a_file_this_module_did_not_name(self):
        by_hand = UPLOADS_DIR / "notes-from-the-microscope.tif"
        by_hand.write_bytes(b"someone put this here")
        try:
            self.assertFalse(discard_upload_if_unreferenced(by_hand))
            self.assertTrue(by_hand.exists())
        finally:
            by_hand.unlink(missing_ok=True)

    def test_it_keeps_a_body_this_process_is_still_writing(self):
        staged = StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
        arriving = Path(staged.temporary_file_path())
        staged.write(b"still coming")
        try:
            self.assertFalse(discard_upload_if_unreferenced(arriving))
            self.assertTrue(arriving.exists())
        finally:
            staged.close()

    def test_it_refuses_a_path_outside_the_staging_directory(self):
        from quantem.core.config import STORAGE_DIR

        elsewhere = STORAGE_DIR / "tmp" / f"{uuid.uuid4()}.tif"
        elsewhere.parent.mkdir(parents=True, exist_ok=True)
        elsewhere.write_bytes(b"not the staging directory")
        try:
            self.assertFalse(discard_upload_if_unreferenced(elsewhere))
            self.assertTrue(elsewhere.exists())
        finally:
            elsewhere.unlink(missing_ok=True)

    def test_a_database_that_cannot_be_asked_counts_as_referenced(self):
        orphan = UPLOADS_DIR / f"{uuid.uuid4()}.tif"
        orphan.write_bytes(_noise(1024))
        try:
            with mock.patch(
                "quantem.assets.upload_staging.upload_is_referenced",
                return_value=None,
            ):
                self.assertFalse(discard_upload_if_unreferenced(orphan))
            self.assertTrue(
                orphan.exists(),
                "bytes were deleted on a database error; an accepted image "
                "could be destroyed the same way",
            )
        finally:
            orphan.unlink(missing_ok=True)


class ConcurrentStagingTests(UploadsDirectoryTestCase):
    """Simultaneous imports do not delete each other's bytes.

    Object level on purpose: the endpoint's concurrency was exercised over real
    HTTP against a running server (eight simultaneous imports, mixed accepted
    and refused). What is worth pinning in the suite is the part that has
    shared mutable state -- the staged names and the live-path registry.
    """

    def test_ten_simultaneous_stagings_keep_only_the_claimed_ones(self):
        claimed: list[Path] = []
        errors: list[BaseException] = []
        lock = threading.Lock()
        start = threading.Barrier(10)

        def one_upload(index: int) -> None:
            try:
                start.wait(timeout=30)
                staged = StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8")
                body = _noise(50_000)
                staged.write(body)
                if index % 2 == 0:
                    target = UPLOADS_DIR / f"{uuid.uuid4()}.tif"
                    staged.claim(target)
                    with lock:
                        claimed.append(target)
                staged.close()
            except BaseException as exc:  # pragma: no cover - reported below
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=one_upload, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertEqual(len(claimed), 5)
        try:
            # A claim made outside a request keeps its bytes: close() only
            # second-guesses a claim it made for an HTTP request.
            for target in claimed:
                self.assertTrue(target.exists(), f"{target.name} was deleted")
                self.assertEqual(target.stat().st_size, 50_000)
            self.assertEqual(
                sorted(UPLOADS_DIR.glob(f"{STAGING_PREFIX}*")),
                [],
                "an unclaimed body survived a concurrent run",
            )
        finally:
            for target in claimed:
                target.unlink(missing_ok=True)

    def test_the_live_path_registry_is_empty_once_everything_is_closed(self):
        # A stale entry here would make the sweeper keep an abandoned body for
        # the life of the process, which is the leak this module exists to stop.
        from quantem.assets.upload_staging import _live_paths

        staged = [StagedUploadedFile("scan.tif", "image/tiff", 0, "utf-8") for _ in range(5)]
        paths = [Path(item.temporary_file_path()) for item in staged]
        for item in staged:
            item.write(b"body")
        for item in staged:
            item.close()

        live = _live_paths()
        for path in paths:
            self.assertNotIn(path, live)
            self.assertFalse(path.exists())

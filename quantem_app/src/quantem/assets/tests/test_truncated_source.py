"""An incomplete image file is refused at the door, not four minutes later.

The defect (wave 0c, W9). ``real_png[:1000]`` and ``real_png[:100_000]`` were
both answered **201**. A row appeared in the library, the user watched it queue,
and the import failed in the background pipeline with *"Error decoding PNG to
8-bit grayscale: image file is truncated"* -- a sentence about a decoder,
arriving after the point where the user could still fix it by exporting the file
again. The acceptance check was ``Image.open``, which parses the header and
stops; a PNG's header is the first 33 bytes, so a file cut anywhere after that
passed.

What is checked here is *structure*, not pixels: the container is walked by
seeking over its chunk table, so a 2 GB image costs a few dozen 8-byte reads and
never a decode (owner ruling R3 -- this has to run on an 8 GB laptop). That
catches the whole truncation class, which is what a half-finished copy off a
microscope share or an interrupted download actually looks like, and it catches
"named .png, isn't one" on the way past.

Deliberately *not* checked: per-chunk CRCs and the compressed stream itself. A
CRC pass would need every byte and would refuse a file some third-party exporter
merely mis-stamped, which is a worse failure than the one being fixed -- there
would be no way for the user to get their image in at all.
"""

from __future__ import annotations

import io
import tracemalloc
from pathlib import Path
from uuid import uuid4

import numpy as np
import tifffile
from django.core.files.uploadedfile import SimpleUploadedFile, UploadedFile
from django.test import TestCase
from PIL import Image

from quantem.assets.asset_mutations import (
    create_uploaded_asset,
    verify_source_is_complete,
)
from quantem.assets.models import Asset
from quantem.core.config import STORAGE_DIR, UPLOADS_DIR
from quantem.jobs.models import Job
from quantem.registry.tests.copy_gate import find_violations

UPLOAD_URL = "/api/assets/upload/"


def _real_png_bytes(width: int = 512, height: int = 512) -> bytes:
    """A PNG big enough that a 100 000 byte prefix still has a valid header."""
    rng = np.random.default_rng(4)
    array = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, "PNG", compress_level=1)
    return buffer.getvalue()


def _real_tiff_bytes(width: int = 512, height: int = 512) -> bytes:
    rng = np.random.default_rng(4)
    array = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array, photometric="minisblack")
    return buffer.getvalue()


def _upload(name: str, payload: bytes) -> SimpleUploadedFile:
    return SimpleUploadedFile(name, payload, content_type="application/octet-stream")


def _snapshot() -> dict[str, int]:
    if not UPLOADS_DIR.exists():
        return {}
    return {p.name: p.stat().st_size for p in UPLOADS_DIR.iterdir() if p.is_file()}


class TruncatedPngTests(TestCase):
    """The reported case, at the two sizes that were measured accepted."""

    def setUp(self):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.before = _snapshot()
        self.png = _real_png_bytes()
        self.assertGreater(len(self.png), 150_000, "fixture too small to truncate")

    def tearDown(self):
        for name in set(_snapshot()) - set(self.before):
            (UPLOADS_DIR / name).unlink(missing_ok=True)

    def test_a_truncated_png_is_refused(self):
        for cut in (1_000, 100_000, len(self.png) - 1):
            with self.subTest(bytes_kept=cut):
                with self.assertRaises(ValueError) as caught:
                    create_uploaded_asset(
                        uploaded_file=_upload("cell.png", self.png[:cut]),
                        swallow_enqueue_errors=True,
                    )
                self.assertIn("incomplete", str(caught.exception).lower())

    def test_a_truncated_png_creates_no_library_row_and_no_work(self):
        with self.assertRaises(ValueError):
            create_uploaded_asset(
                uploaded_file=_upload("cell.png", self.png[:100_000]),
                swallow_enqueue_errors=True,
            )

        self.assertEqual(Asset.objects.count(), 0)
        self.assertEqual(Job.objects.count(), 0)

    def test_a_truncated_png_leaves_no_bytes_behind(self):
        before = _snapshot()

        with self.assertRaises(ValueError):
            create_uploaded_asset(
                uploaded_file=_upload("cell.png", self.png[:100_000]),
                swallow_enqueue_errors=True,
            )

        after = _snapshot()
        self.assertEqual({n: s for n, s in after.items() if before.get(n) != s}, {})

    def test_a_complete_png_is_imported(self):
        detail = create_uploaded_asset(
            uploaded_file=_upload("cell.png", self.png),
            swallow_enqueue_errors=True,
        )

        self.assertEqual(Asset.objects.get(id=detail["id"]).logical_width, 512)


class RefusalWordingTests(TestCase):
    """A sentence a biologist can act on, with nothing internal in it."""

    def _message(self, name: str, payload: bytes) -> str:
        with self.assertRaises(ValueError) as caught:
            verify_source_is_complete(_upload(name, payload))
        return str(caught.exception)

    def test_the_truncation_sentence_is_plain_and_actionable(self):
        message = self._message("cell.png", _real_png_bytes()[:100_000])

        self.assertIn("incomplete", message.lower())
        self.assertIn("Nothing was imported", message)
        # It has to tell the user what to do next, not only what went wrong.
        self.assertIn("again", message.lower())

    def test_no_refusal_leaks_a_path_a_command_or_an_exception_class(self):
        cases = {
            "truncated png": ("cell.png", _real_png_bytes()[:100_000]),
            "png magic then noise": (
                "cell.png",
                b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8,
            ),
            "not a png at all": ("cell.png", bytes(range(256)) * 400),
            "empty png": ("cell.png", b""),
            "truncated tiff": ("cell.tif", _real_tiff_bytes()[: 512 * 100]),
            "not a tiff at all": ("cell.tif", bytes(range(256)) * 400),
        }
        for label, (name, payload) in cases.items():
            with self.subTest(case=label):
                message = self._message(name, payload)
                violations = find_violations(message, label)
                self.assertEqual(violations, [], "\n".join(str(v) for v in violations))
                self.assertNotIn(str(STORAGE_DIR), message)
                self.assertNotIn(":\\", message)
                self.assertNotIn("Error:", message)
                self.assertNotIn("Traceback", message)
                # apiErrors.ts stops treating a body as a sentence at 200.
                self.assertLess(len(message), 200, message)
                self.assertTrue(message.endswith("."), message)


class WrongFormatTests(TestCase):
    """A file whose name says PNG and whose bytes do not."""

    def test_bytes_that_are_not_a_png_are_refused_at_the_door(self):
        with self.assertRaises(ValueError) as caught:
            create_uploaded_asset(
                uploaded_file=_upload("cell.png", bytes(range(256)) * 400),
                swallow_enqueue_errors=True,
            )

        self.assertIn("not a PNG", str(caught.exception))
        self.assertEqual(Asset.objects.count(), 0)

    def test_an_empty_file_is_refused(self):
        with self.assertRaises(ValueError):
            create_uploaded_asset(
                uploaded_file=_upload("cell.png", b""),
                swallow_enqueue_errors=True,
            )
        with self.assertRaises(ValueError):
            create_uploaded_asset(
                uploaded_file=_upload("cell.tif", b""),
                swallow_enqueue_errors=True,
            )

        self.assertEqual(Asset.objects.count(), 0)


class TruncatedTiffTests(TestCase):
    """Same defect, other container: the data the file promises is not there."""

    def test_a_truncated_tiff_is_refused(self):
        payload = _real_tiff_bytes()

        with self.assertRaises(ValueError) as caught:
            create_uploaded_asset(
                uploaded_file=_upload("cell.tif", payload[: len(payload) // 2]),
                swallow_enqueue_errors=True,
            )

        self.assertIn("incomplete", str(caught.exception).lower())
        self.assertEqual(Asset.objects.count(), 0)

    def test_a_complete_tiff_is_imported(self):
        detail = create_uploaded_asset(
            uploaded_file=_upload("cell.tif", _real_tiff_bytes()),
            swallow_enqueue_errors=True,
        )

        self.assertEqual(Asset.objects.get(id=detail["id"]).logical_width, 512)


class CheckIsCheapTests(TestCase):
    """The check may not become the reason a big import is slow or fat."""

    def test_validating_a_png_does_not_read_the_picture_into_memory(self):
        payload = _real_png_bytes(2048, 2048)
        path = Path(STORAGE_DIR) / "tmp" / f"complete_{uuid4().hex}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        try:
            with open(path, "rb") as handle:
                uploaded = UploadedFile(
                    handle, "cell.png", "image/png", len(payload), None
                )
                tracemalloc.start()
                try:
                    verify_source_is_complete(uploaded)
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
        finally:
            path.unlink(missing_ok=True)

        self.assertGreater(len(payload), 1_000_000, "fixture too small to matter")
        self.assertLess(
            peak,
            len(payload) // 8,
            f"validating peaked at {peak} bytes for a {len(payload)} byte PNG",
        )

    def test_the_check_leaves_the_upload_readable(self):
        payload = _real_png_bytes()
        uploaded = _upload("cell.png", payload)

        verify_source_is_complete(uploaded)

        self.assertEqual(b"".join(uploaded.chunks()), payload)


class UploadEndpointTests(TestCase):
    """Through the real door, with the real upload handler."""

    def setUp(self):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.before = _snapshot()

    def tearDown(self):
        for name in set(_snapshot()) - set(self.before):
            (UPLOADS_DIR / name).unlink(missing_ok=True)

    def test_a_truncated_png_is_answered_400_with_a_sentence(self):
        payload = _real_png_bytes()[:100_000]

        response = self.client.post(UPLOAD_URL, {"file": _upload("cell.png", payload)})

        self.assertEqual(response.status_code, 400, response.content[:300])
        message = response.json()["error"]
        self.assertIn("incomplete", message.lower())
        self.assertEqual(Asset.objects.count(), 0)
        after = _snapshot()
        self.assertEqual({n: s for n, s in after.items() if self.before.get(n) != s}, {})

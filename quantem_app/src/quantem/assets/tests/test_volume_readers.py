"""Reader dispatch is narrowed to TIFF + PNG (owner ruling 2026-08-06).

These tests pin the two halves of that contract: the formats that must keep
working, and the refusal an unsupported file gets -- a user-facing message
naming what *is* accepted, never a KeyError or a silent ``None``.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
from django.test import SimpleTestCase
from PIL import Image

from quantem.assets.volume_readers import (
    UnsupportedVolumeSource,
    read_volume_source,
)


class UnsupportedSourceTests(SimpleTestCase):
    def test_dropped_formats_are_refused_with_a_readable_message(self):
        # mrc/nd2/dm3/dm4/avi are not supported in v1.
        for name in ("tomo.mrc", "stack.nd2", "scan.dm4", "series.dm3", "sbfsem.avi"):
            with self.subTest(name=name):
                with self.assertRaises(UnsupportedVolumeSource) as ctx:
                    read_volume_source(name)
                message = str(ctx.exception)
                self.assertIn(Path(name).suffix, message)
                self.assertIn(".tif", message)
                self.assertIn(".png", message)

    def test_unsupported_source_is_a_value_error(self):
        # The API layer maps ValueError to a 400 with the message verbatim.
        self.assertTrue(issubclass(UnsupportedVolumeSource, ValueError))


class PngSourceTests(SimpleTestCase):
    def test_single_png_reads_as_a_one_plane_volume(self):
        plane = np.arange(20, dtype=np.uint8).reshape(4, 5)
        with TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "image.png"
            Image.fromarray(plane, mode="L").save(png_path)

            with read_volume_source(png_path) as source:
                self.assertEqual(source.metadata.source_format, "png")
                self.assertEqual(source.metadata.depth, 1)
                self.assertEqual(source.metadata.height, 4)
                self.assertEqual(source.metadata.width, 5)
                self.assertEqual(source.metadata.bit_depth, 8)
                np.testing.assert_array_equal(source.read_plane(0), plane)


class ImageSequenceTests(SimpleTestCase):
    def test_sequence_accepts_tiff_and_png_and_ignores_everything_else(self):
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            Image.fromarray(np.full((4, 5), 10, dtype=np.uint8), mode="L").save(
                directory / "slice_2.png"
            )
            tifffile.imwrite(
                str(directory / "slice_10.tif"), np.full((4, 5), 20, dtype=np.uint8)
            )
            (directory / "notes.txt").write_text("not a slice", encoding="utf-8")
            (directory / "preview.avi").write_bytes(b"not a slice either")

            with read_volume_source(directory) as source:
                self.assertEqual(source.metadata.depth, 2)
                # Natural sort: slice_2 precedes slice_10.
                self.assertEqual(int(source.read_plane(0)[0, 0]), 10)
                self.assertEqual(int(source.read_plane(1)[0, 0]), 20)

    def test_directory_without_readable_slices_is_refused(self):
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "stack.mrc").write_bytes(b"0" * 16)

            with self.assertRaises(UnsupportedVolumeSource) as ctx:
                read_volume_source(directory)
            self.assertIn(".png", str(ctx.exception))

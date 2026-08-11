from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from PIL import Image

from quantem.assets.utils import (
    convert_tiff_to_png,
    extract_image_metadata,
    extract_tiff_metadata,
    validate_upload_file,
)


class ConvertTiffToPngTests(TestCase):
    def test_convert_tiff_to_png_collapses_grayscale_stack_to_first_plane(self):
        stack = np.zeros((3, 4, 5), dtype=np.uint8)
        stack[0, :, :] = 11
        stack[1, :, :] = 99

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tiff_path = tmpdir_path / "stack.tif"
            png_path = tmpdir_path / "stack.png"
            tifffile.imwrite(str(tiff_path), stack)

            metadata = extract_tiff_metadata(tiff_path)
            converted_path = convert_tiff_to_png(tiff_path, metadata, png_path)

            self.assertEqual(converted_path, png_path)
            self.assertTrue(png_path.exists())

            with Image.open(png_path) as converted:
                self.assertEqual(converted.mode, "L")
                self.assertEqual(converted.size, (5, 4))
                pixels = np.array(converted)

            np.testing.assert_array_equal(pixels, stack[0])


class ValidateUploadFileTests(SimpleTestCase):
    """Upload accepts TIFF and PNG only (owner ruling 2026-08-06)."""

    def test_accepts_tiff_and_png(self):
        for name in ("scan.tif", "scan.TIFF", "scan.png"):
            with self.subTest(name=name):
                is_valid, error = validate_upload_file(SimpleUploadedFile(name, b"bytes"))
                self.assertTrue(is_valid)
                self.assertIsNone(error)

    def test_rejects_dropped_formats_by_naming_the_accepted_ones(self):
        for name in ("tomo.mrc", "stack.nd2", "scan.dm4", "sbfsem.avi", "photo.jpg"):
            with self.subTest(name=name):
                is_valid, error = validate_upload_file(SimpleUploadedFile(name, b"bytes"))
                self.assertFalse(is_valid)
                self.assertIn(".tif", error)
                self.assertIn(".png", error)


class ExtractImageMetadataTests(SimpleTestCase):
    def test_reads_png_geometry_without_decoding_through_tifffile(self):
        with TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "image.png"
            Image.fromarray(np.zeros((4, 5), dtype=np.uint8), mode="L").save(png_path)

            metadata = extract_image_metadata(png_path)

            self.assertEqual(metadata["width"], 5)
            self.assertEqual(metadata["height"], 4)
            self.assertEqual(metadata["channels"], 1)
            self.assertEqual(metadata["bit_depth"], 8)

    def test_rejects_an_unsupported_suffix(self):
        with self.assertRaises(ValueError):
            extract_image_metadata(Path("tomo.mrc"))

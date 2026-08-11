"""The one decode, checked branch by branch.

Every test here corresponds to a defect a verification round found in the
decoder, or to a requirement in owner rulings R1/R6/R8. The four defects, in
the verifier's numbering:

* **F2** -- an 8-bit palette PNG imported as its raw palette indices: max
  difference 255, every pixel wrong, reported as success.
* **F3** -- signed integers scaled against the *unsigned* range, so an int16
  picture came out at half brightness.
* **F4** -- a three-row interleaved RGB TIFF decoded transposed and failed the
  import.
* **R8** -- a 16-bit source whose data occupies a narrow band of the declared
  range was flattened into a handful of grey levels by the fixed full-range
  map, with nothing recorded and nothing said.

The R8 numbers here are not invented: they are the measured distribution of a
real 16-bit EM corpus (92 TIFFs; the pancreas mosaics live between raw values
28481 and 30076 and the fixed full-range map renders them as 6 to 16 distinct
greys).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from quantem.assets.canonical_decode import (
    DECODER_VERSION,
    MIN_FULL_RANGE_LEVELS,
    CanonicalPlane,
    UnsupportedPixelType,
    decode_canonical_plane,
    standard_window,
)

Image.MAX_IMAGE_PIXELS = None


class DecodeCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="quantem-decode-")
        self.directory = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def decode(self, path: Path) -> CanonicalPlane:
        return decode_canonical_plane(path)


class PalettesResolveToPixels(DecodeCase):
    """F2. A palette image has no band; it has a lookup."""

    def _palette_png(self, name: str, palette: list[int]) -> tuple[Path, np.ndarray]:
        indices = np.arange(256, dtype=np.uint8).reshape(16, 16)
        image = Image.fromarray(indices, mode="P")
        image.putpalette(palette)
        path = self.directory / name
        image.save(path)
        return path, indices

    def test_a_grey_palette_png_decodes_to_the_greys_not_the_indices(self):
        # The fixture's palette maps index i to grey 255 - i, so reading the
        # indices as intensities returns exactly the inverse of the picture.
        palette: list[int] = []
        for i in range(256):
            palette += [255 - i, 255 - i, 255 - i]
        path, indices = self._palette_png("grey_palette.png", palette)

        plane = self.decode(path)

        expected = (255 - indices).astype(np.uint8)
        self.assertEqual(int(np.abs(plane.array.astype(int) - expected).max()), 0)
        self.assertIn("palette", plane.provenance)

    def test_a_palette_png_and_the_same_picture_as_rgb_decode_identically(self):
        palette = []
        for i in range(256):
            palette += [i, 255 - i, (i * 7) % 256]
        path, indices = self._palette_png("colour_palette.png", palette)

        lookup = np.asarray(palette, dtype=np.uint8).reshape(256, 3)
        rgb_path = self.directory / "as_rgb.png"
        Image.fromarray(lookup[indices], mode="RGB").save(rgb_path)

        self.assertEqual(
            int(
                np.abs(
                    self.decode(path).array.astype(int)
                    - self.decode(rgb_path).array.astype(int)
                ).max()
            ),
            0,
        )

    def test_a_palette_tiff_resolves_its_colormap(self):
        # ImageJ's "8-bit colour" writes exactly this shape.
        indices = np.arange(256, dtype=np.uint8).reshape(16, 16)
        colormap = np.zeros((3, 256), dtype=np.uint16)
        for i in range(256):
            colormap[:, i] = (255 - i) * 257
        path = self.directory / "palette.tif"
        tifffile.imwrite(str(path), indices, photometric="palette", colormap=colormap)

        plane = self.decode(path)

        expected = (255 - indices).astype(np.uint8)
        self.assertLessEqual(int(np.abs(plane.array.astype(int) - expected).max()), 1)
        self.assertIn("palette", plane.provenance)


class SignedIntegersScaleAgainstTheirOwnRange(DecodeCase):
    """F3. ``int16`` holds 32767, not 65535."""

    def test_an_int16_picture_is_not_halved(self):
        values = np.linspace(0, 32767, 64 * 64).reshape(64, 64).astype(np.int16)
        path = self.directory / "int16.tif"
        tifffile.imwrite(str(path), values)

        plane = self.decode(path)

        self.assertEqual(int(plane.array.max()), 255)

    def test_the_same_picture_as_int16_and_uint16_decodes_the_same(self):
        signed = np.linspace(0, 32767, 128 * 128).reshape(128, 128).astype(np.int16)
        unsigned = (signed.astype(np.int32) * 2).astype(np.uint16)
        signed_path = self.directory / "signed.tif"
        unsigned_path = self.directory / "unsigned.tif"
        tifffile.imwrite(str(signed_path), signed)
        tifffile.imwrite(str(unsigned_path), unsigned)

        difference = np.abs(
            self.decode(signed_path).array.astype(int)
            - self.decode(unsigned_path).array.astype(int)
        )
        self.assertLessEqual(int(difference.max()), 1)

    def test_int32_scales_against_the_signed_maximum(self):
        top = 2**31 - 1
        values = np.array([[0, top // 2, top]], dtype=np.int32)
        path = self.directory / "int32.tif"
        tifffile.imwrite(str(path), values)

        plane = self.decode(path)

        self.assertEqual(int(plane.array[0, 0]), 0)
        self.assertEqual(int(plane.array[0, 2]), 255)
        self.assertAlmostEqual(int(plane.array[0, 1]), 127, delta=1)

    def test_negative_signed_data_is_still_refused_by_name(self):
        path = self.directory / "negative.tif"
        tifffile.imwrite(str(path), np.array([[-5, 5]], dtype=np.int16))
        with self.assertRaises(UnsupportedPixelType) as caught:
            self.decode(path)
        self.assertIn("negative", str(caught.exception))


class BandsComeFromTheAxisTheFileDeclares(DecodeCase):
    """F4. The shape heuristic was ambiguous; the container is not."""

    def test_a_three_row_interleaved_rgb_tiff_keeps_its_geometry(self):
        array = np.arange(3 * 300 * 3, dtype=np.uint8).reshape(3, 300, 3)
        path = self.directory / "three_rows.tif"
        tifffile.imwrite(str(path), array, photometric="rgb")

        plane = self.decode(path)

        self.assertEqual(plane.shape, (3, 300))
        self.assertEqual(int(np.abs(plane.array.astype(int) - array[:, :, 0]).max()), 0)

    def test_a_planar_rgb_tiff_still_takes_the_first_plane(self):
        array = np.arange(3 * 40 * 30, dtype=np.uint8).reshape(3, 40, 30)
        path = self.directory / "planar.tif"
        tifffile.imwrite(str(path), array, photometric="rgb", planarconfig="separate")

        plane = self.decode(path)

        self.assertEqual(plane.shape, (40, 30))
        self.assertEqual(int(np.abs(plane.array.astype(int) - array[0]).max()), 0)

    def test_an_interleaved_rgb_tiff_still_takes_band_zero(self):
        array = np.arange(40 * 30 * 3, dtype=np.uint8).reshape(40, 30, 3)
        path = self.directory / "interleaved.tif"
        tifffile.imwrite(str(path), array, photometric="rgb")

        plane = self.decode(path)

        self.assertEqual(plane.shape, (40, 30))
        self.assertEqual(int(np.abs(plane.array.astype(int) - array[:, :, 0]).max()), 0)

    def test_a_multipage_tiff_still_takes_the_first_page(self):
        stack = np.arange(6 * 20 * 30, dtype=np.uint8).reshape(6, 20, 30)
        path = self.directory / "stack.tif"
        tifffile.imwrite(str(path), stack)

        plane = self.decode(path)

        self.assertEqual(plane.shape, (20, 30))
        self.assertEqual(int(np.abs(plane.array.astype(int) - stack[0]).max()), 0)

    def test_an_rgba_png_takes_band_zero(self):
        array = np.arange(20 * 30 * 4, dtype=np.uint8).reshape(20, 30, 4)
        path = self.directory / "rgba.png"
        Image.fromarray(array, mode="RGBA").save(path)

        plane = self.decode(path)

        self.assertEqual(plane.shape, (20, 30))
        self.assertEqual(int(np.abs(plane.array.astype(int) - array[:, :, 0]).max()), 0)


class StandardWindows(unittest.TestCase):
    """R8 requirement 2: a fixed ladder, never a per-image fit."""

    def test_the_window_is_the_smallest_grid_interval_that_contains_the_data(self):
        # The first pair is Collagen.tif's measured robust interval.
        self.assertEqual(standard_window(28947, 29789, 65535), (28672, 30719))
        self.assertEqual(standard_window(0, 65535, 65535), (0, 65535))
        self.assertEqual(standard_window(100, 200, 65535), (0, 1023))
        self.assertEqual(standard_window(65535, 65535, 65535), (64512, 65535))

    def test_two_images_whose_data_sits_in_the_same_cells_get_the_same_window(self):
        # Beta cell.tif and Collagen.tif from the measured corpus. 14 of the 26
        # narrow images in that corpus share this one window.
        self.assertEqual(
            standard_window(28848, 29759, 65535), standard_window(28947, 29789, 65535)
        )

    def test_a_pair_straddling_a_grid_line_gets_two_windows_and_that_is_recorded(self):
        # The honest cost of a fixed grid. Not a defect to be smoothed away:
        # the two conversions really are different, so they are visibly
        # different rather than quietly different.
        self.assertNotEqual(
            standard_window(28900, 29700, 65535), standard_window(28950, 29650, 65535)
        )

    def test_the_window_never_escapes_the_declared_range(self):
        for low, high in ((0, 0), (65535, 65535), (1, 65534), (32768, 32768)):
            window = standard_window(low, high, 65535)
            self.assertGreaterEqual(window[0], 0)
            self.assertLessEqual(window[1], 65535)
            self.assertLessEqual(window[0], low)
            self.assertGreaterEqual(window[1], high)


class SixteenBitConversion(DecodeCase):
    """R6 by default, R8's fallback only when the criterion says so."""

    def _write(self, name: str, array: np.ndarray) -> Path:
        path = self.directory / name
        tifffile.imwrite(str(path), array)
        return path

    def test_a_full_range_16_bit_image_keeps_the_fixed_full_range_map(self):
        values = np.linspace(0, 65535, 256 * 256).reshape(256, 256).astype(np.uint16)
        path = self._write("full.tif", values)

        plane = self.decode(path)

        self.assertEqual(plane.conversion.strategy, "full-range")
        self.assertEqual(plane.conversion.window, (0, 65535))
        self.assertIn("full-range", plane.provenance)

    def test_the_full_range_map_is_bit_identical_to_the_shipped_arithmetic(self):
        # The default branch must not be a behaviour change, only a cheaper
        # one: every one of the 65 536 possible values, not a sample.
        values = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        path = self._write("every_value.tif", values)

        plane = self.decode(path)

        shipped = np.nan_to_num(
            np.clip(values.astype(np.float32) * (255.0 / 65535.0), 0, 255), nan=0
        ).astype(np.uint8)
        self.assertEqual(int(np.abs(plane.array.astype(int) - shipped.astype(int)).max()), 0)
        self.assertEqual(plane.conversion.strategy, "full-range")

    def test_a_real_narrow_band_em_image_falls_back_to_a_standard_window(self):
        # The measured pancreas distribution: everything between 28481 and
        # 30076, which the full-range map renders as about eight greys.
        rng = np.random.default_rng(20260810)
        values = rng.integers(28481, 30077, size=(256, 256)).astype(np.uint16)
        path = self._write("narrow.tif", values)

        plane = self.decode(path)

        self.assertEqual(plane.conversion.strategy, "standard-window")
        self.assertIsNotNone(plane.conversion.window)
        low, high = plane.conversion.window
        self.assertLessEqual(low, 28481)
        self.assertGreaterEqual(high, 30076)
        self.assertLess(plane.conversion.full_range_levels, MIN_FULL_RANGE_LEVELS)
        # The point of the fallback: the picture keeps its tonal resolution.
        distinct = int(np.unique(plane.array).size)
        self.assertGreater(distinct, 100)

    def test_the_full_range_map_would_have_destroyed_that_same_image(self):
        rng = np.random.default_rng(20260810)
        values = rng.integers(28481, 30077, size=(256, 256)).astype(np.uint16)
        flattened = np.clip(
            values.astype(np.float32) * (255.0 / 65535.0), 0, 255
        ).astype(np.uint8)
        self.assertLess(int(np.unique(flattened).size), MIN_FULL_RANGE_LEVELS)

    def test_the_branch_is_deterministic_for_the_same_file(self):
        rng = np.random.default_rng(7)
        values = rng.integers(4000, 4600, size=(128, 128)).astype(np.uint16)
        path = self._write("repeat.tif", values)

        first = self.decode(path)
        second = self.decode(path)

        self.assertEqual(first.conversion.strategy, second.conversion.strategy)
        self.assertEqual(first.conversion.window, second.conversion.window)
        self.assertEqual(int(np.abs(first.array.astype(int) - second.array.astype(int)).max()), 0)

    def test_two_narrow_images_in_the_same_cells_stay_comparable(self):
        """Comparable means: the same source value becomes the same grey."""

        rng = np.random.default_rng(11)
        first_values = rng.integers(28850, 29760, size=(256, 256)).astype(np.uint16)
        second_values = rng.integers(28950, 29790, size=(256, 256)).astype(np.uint16)
        # An identical patch in both images. If the two conversions agree, this
        # patch has to come out identical; if either had been fitted to its own
        # image it would not.
        patch = rng.integers(29000, 29500, size=(8, 8)).astype(np.uint16)
        first_values[:8, :8] = patch
        second_values[:8, :8] = patch

        first = self.decode(self._write("a.tif", first_values))
        second = self.decode(self._write("b.tif", second_values))

        self.assertEqual(first.conversion.strategy, "standard-window")
        self.assertEqual(first.conversion.window, second.conversion.window)
        self.assertEqual(
            int(
                np.abs(
                    first.array[:8, :8].astype(int) - second.array[:8, :8].astype(int)
                ).max()
            ),
            0,
        )

    def test_a_hot_pixel_does_not_drag_the_window_open(self):
        # Percentiles rather than min/max, precisely so one saturated pixel
        # cannot claim the whole range for the image.
        rng = np.random.default_rng(3)
        values = rng.integers(28900, 29700, size=(256, 256)).astype(np.uint16)
        values[0, 0] = 65535
        path = self._write("hot.tif", values)

        plane = self.decode(path)

        self.assertEqual(plane.conversion.strategy, "standard-window")
        self.assertLess(plane.conversion.window[1], 65535)

    def test_the_conversion_is_recorded_in_provenance_and_on_the_plane(self):
        rng = np.random.default_rng(5)
        values = rng.integers(28481, 30077, size=(128, 128)).astype(np.uint16)
        path = self._write("recorded.tif", values)

        plane = self.decode(path)
        record = plane.conversion.as_dict()

        self.assertEqual(record["strategy"], "standard-window")
        self.assertEqual(record["depth"], "16-bit")
        self.assertIsNotNone(record["window"])
        self.assertIsNotNone(record["observed"])
        self.assertIsNotNone(record["robust_interval"])
        self.assertIn("window", plane.provenance)
        self.assertIn("range/", plane.provenance)

    def test_the_user_is_told_in_plain_language(self):
        rng = np.random.default_rng(5)
        values = rng.integers(28481, 30077, size=(128, 128)).astype(np.uint16)
        plane = self.decode(self._write("notice.tif", values))

        notice = plane.notice

        self.assertIn("16-bit", notice)
        self.assertIn("narrow-range", notice)
        self.assertIn("not directly comparable", notice)
        # I-12: user-facing copy carries no machinery.
        for forbidden in (
            "canonical_decode",
            "quantem.",
            "uint16",
            "numpy",
            "ndarray",
            "ValueError",
            "GET ",
            "POST ",
            "/api/",
            "C:\\",
            "D:\\",
        ):
            self.assertNotIn(forbidden, notice)

    def test_a_full_range_image_says_it_is_comparable(self):
        values = np.linspace(0, 65535, 128 * 128).reshape(128, 128).astype(np.uint16)
        plane = self.decode(self._write("comparable.tif", values))

        self.assertIn("comparable", plane.notice)
        self.assertNotIn("narrow-range", plane.notice)

    def test_an_8_bit_source_says_nothing_because_nothing_happened(self):
        values = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        plane = self.decode(self._write("eight.tif", values))

        self.assertEqual(plane.notice, "")
        self.assertEqual(plane.conversion.strategy, "identity")

    def test_a_uniform_16_bit_image_does_not_explode(self):
        values = np.full((64, 64), 30000, dtype=np.uint16)
        plane = self.decode(self._write("flat.tif", values))

        self.assertEqual(plane.array.dtype, np.uint8)
        self.assertEqual(int(np.unique(plane.array).size), 1)

    def test_an_all_black_and_an_all_white_16_bit_image_stay_black_and_white(self):
        black = self.decode(self._write("black.tif", np.zeros((32, 32), dtype=np.uint16)))
        white = self.decode(
            self._write("white.tif", np.full((32, 32), 65535, dtype=np.uint16))
        )

        self.assertEqual(int(black.array.max()), 0)
        self.assertEqual(int(white.array.min()), 255)


class UnchangedBehaviour(DecodeCase):
    """The branches this change did not touch still answer the same way."""

    def test_uint8_is_the_identity(self):
        values = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        path = self.directory / "eight.png"
        Image.fromarray(values, mode="L").save(path)

        plane = self.decode(path)

        self.assertEqual(int(np.abs(plane.array.astype(int) - values).max()), 0)

    def test_floats_are_clipped_not_rescaled(self):
        values = np.array([[-3.0, 0.0, 12.5, 300.0, np.nan]], dtype=np.float32)
        path = self.directory / "float.tif"
        tifffile.imwrite(str(path), values)

        plane = self.decode(path)

        self.assertEqual(plane.array.tolist(), [[0, 0, 12, 255, 0]])

    def test_bilevel_becomes_0_and_255(self):
        values = np.array([[True, False], [False, True]])
        path = self.directory / "bool.tif"
        tifffile.imwrite(str(path), values)

        plane = self.decode(path)

        self.assertEqual(plane.array.tolist(), [[255, 0], [0, 255]])

    def test_complex_data_is_refused_by_name(self):
        path = self.directory / "complex.tif"
        tifffile.imwrite(str(path), np.ones((4, 4), dtype=np.complex64))
        with self.assertRaises(UnsupportedPixelType) as caught:
            self.decode(path)
        self.assertIn("complex", str(caught.exception))

    def test_dispatch_is_on_the_bytes_not_the_suffix(self):
        values = np.linspace(0, 65535, 32 * 32).reshape(32, 32).astype(np.uint16)
        honest = self.directory / "honest.tif"
        tifffile.imwrite(str(honest), values)
        liar = self.directory / "liar.png"
        liar.write_bytes(honest.read_bytes())

        self.assertEqual(
            int(
                np.abs(
                    self.decode(honest).array.astype(int)
                    - self.decode(liar).array.astype(int)
                ).max()
            ),
            0,
        )

    def test_the_decoder_version_is_stamped_on_every_plane(self):
        path = self.directory / "stamp.tif"
        tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8))
        self.assertEqual(self.decode(path).decoder_version, DECODER_VERSION)

    def test_the_source_file_is_not_left_open(self):
        # Windows refuses to rename a file another handle still holds, and the
        # importer renames staged uploads straight after decoding them. The
        # palette branch makes a second Pillow image, which is where a handle
        # would leak.
        indices = np.arange(256, dtype=np.uint8).reshape(16, 16)
        image = Image.fromarray(indices, mode="P")
        image.putpalette([v for i in range(256) for v in (i, i, i)])
        path = self.directory / "held.png"
        image.save(path)

        self.decode(path)

        moved = self.directory / "moved.png"
        path.rename(moved)
        self.assertTrue(moved.exists())

"""The fast pyramid must be the *same* pyramid.

Import used to decode the source three times: once to write a canonical PNG,
once more to fill NGFF level 0 from that PNG, and once per pyramid level to
read back the level below out of zarr. That is now a single decode feeding an
in-memory build, and the whole point of these tests is that nothing about the
resulting pixels changed.

``_LegacyPyramid`` below is the previous algorithm, kept verbatim in behaviour
(chunked crops off a Pillow-decoded PNG for level 0; each coarser level read
back out of the store just written; the old Blosc settings). Every comparison
is exact equality, not a tolerance: a resampling change would be a change to
what segmentation and analysis measure, and would have to be argued for
separately rather than smuggled in behind a speed-up.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import tifffile
import zarr
from django.test import TestCase
from numcodecs import Blosc
from PIL import Image

from quantem.assets import ngff as ngff_module
from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.canonical_decode import decode_canonical_array
from quantem.assets.models import Asset, Rendition
from quantem.assets.ngff import (
    NGFF_CHUNK_SIZE,
    _chunk_bounds,
    _downsample_region,
    _is_valid_ngff_store,
    _iter_level_chunks,
    _level_shapes,
)
from quantem.assets.pyramid_authority import (
    Intent,
    PublishedPyramid,
    resolve_pyramid,
)
from quantem.assets.tasks import prepare_asset_renditions
from quantem.assets.utils import (
    convert_png_to_8bit_grayscale,
    convert_tiff_to_png,
    create_roi_image_from_image,
    extract_image_metadata,
    extract_tiff_metadata,
    load_source_plane_uint8,
)
from quantem.core.config import DATA_DIR, IMAGES_DIR, STORAGE_DIR, UPLOADS_DIR
from quantem.core.local_storage import normalize_stored_path_value
from quantem.testing import make_em_like_array

LEGACY_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def build_standalone(image, plane) -> Path:
    """Build one generation for a duck-typed image, with no database row.

    The pyramid *arithmetic* is worth testing on its own, and it does not need
    an asset: ``build_pyramid`` takes a ticket and a plane, so a test can mint
    a ticket pointing at scratch. That the builder cannot be handed a *path* is
    the point -- see ``test_ngff_decode_chokepoint``.
    """

    from quantem.assets.ngff import build_pyramid
    from quantem.assets.pyramid_authority import BuildTicket

    root = _scratch_dir() / f"gen-{uuid4().hex[:12]}"
    ticket = BuildTicket(
        asset_id=str(image.id),
        attempt_token="standalone",
        generation_id=root.name,
        root=root,
        from_generation="",
    )
    build_pyramid(ticket, image, plane)
    return root


def published_root(image):
    """The published generation for ``image``, or ``None``.

    ``get_ngff_paths`` is gone. A pyramid's location is no longer something a
    caller derives from an id -- twelve predicates over a derived path is the
    defect this change removed -- so a test asks the authority like everything
    else does.
    """

    resolved = resolve_pyramid(image, intent=Intent.SERVE)
    return resolved.root if isinstance(resolved, PublishedPyramid) else None


class _FakeImage:
    """The duck type ``ngff`` needs: an id, a size and a display name."""

    def __init__(self, width: int, height: int):
        self.id = uuid4()
        self.width = width
        self.height = height
        self.display_name = "legacy comparison"
        self.has_stored_z_stack = False


class _LegacyPyramid:
    """The pre-change NGFF builder, preserved as the reference implementation."""

    @staticmethod
    def create_store(image, ngff_root: Path) -> list[zarr.Array]:
        if ngff_root.exists():
            shutil.rmtree(ngff_root)
        level_shapes = _level_shapes(int(image.height), int(image.width))
        ngff_root.parent.mkdir(parents=True, exist_ok=True)
        zarr_root = zarr.open_group(str(ngff_root), mode="w", zarr_format=2)
        arrays: list[zarr.Array] = []
        for level_idx, (height, width) in enumerate(level_shapes):
            arrays.append(
                zarr_root.create_array(
                    str(level_idx),
                    shape=(1, height, width),
                    chunks=(1, min(NGFF_CHUNK_SIZE, height), min(NGFF_CHUNK_SIZE, width)),
                    dtype=np.uint8,
                    compressor=LEGACY_COMPRESSOR,
                    overwrite=True,
                    fill_value=0,
                )
            )
        return arrays

    @staticmethod
    def write_level0_from_png(source_path: Path, level0_array: zarr.Array) -> None:
        level_height = int(level0_array.shape[1])
        level_width = int(level0_array.shape[2])
        pil_image = Image.open(source_path)
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        try:
            for chunk_x, chunk_y in _iter_level_chunks(level_width, level_height):
                x_min, y_min, x_max, y_max = _chunk_bounds(
                    chunk_x, chunk_y, width=level_width, height=level_height
                )
                region = pil_image.crop((x_min, y_min, x_max, y_max))
                level0_array[0, y_min:y_max, x_min:x_max] = np.asarray(
                    region, dtype=np.uint8
                )
        finally:
            pil_image.close()

    @staticmethod
    def write_downsampled_level(child_array: zarr.Array, parent_array: zarr.Array) -> None:
        child_height = int(child_array.shape[1])
        child_width = int(child_array.shape[2])
        parent_height = int(parent_array.shape[1])
        parent_width = int(parent_array.shape[2])
        for chunk_x, chunk_y in _iter_level_chunks(parent_width, parent_height):
            x_min, y_min, x_max, y_max = _chunk_bounds(
                chunk_x, chunk_y, width=parent_width, height=parent_height
            )
            source_region = np.asarray(
                child_array[
                    0,
                    y_min * 2 : min(child_height, y_max * 2),
                    x_min * 2 : min(child_width, x_max * 2),
                ],
                dtype=np.uint8,
            )
            parent_array[0, y_min:y_max, x_min:x_max] = _downsample_region(
                source_region,
                target_height=y_max - y_min,
                target_width=x_max - x_min,
            )

    @classmethod
    def build(cls, image, source_png: Path, ngff_root: Path) -> list[zarr.Array]:
        arrays = cls.create_store(image, ngff_root)
        cls.write_level0_from_png(source_png, arrays[0])
        for level_idx in range(1, len(arrays)):
            cls.write_downsampled_level(arrays[level_idx - 1], arrays[level_idx])
        return arrays


def _reference_half(plane: np.ndarray) -> np.ndarray:
    """A 2x2 box mean, written out longhand.

    ``_LegacyPyramid`` deliberately reuses ``_downsample_region`` from the
    module under test -- it is the tiling and the level chaining that changed,
    not the kernel -- which means that comparison alone cannot notice the
    kernel itself drifting. This is the independent statement of what a
    pyramid level *is*: pad the odd row/column by repeating the edge, average
    each 2x2 block, and round halves to even (numpy's ``rint``, and what the
    stores in the field were built with).
    """

    height, width = plane.shape
    target_height = max(1, math.ceil(height / 2))
    target_width = max(1, math.ceil(width / 2))
    padded = np.pad(
        plane,
        ((0, target_height * 2 - height), (0, target_width * 2 - width)),
        mode="edge",
    )
    blocks = padded.astype(np.float64).reshape(target_height, 2, target_width, 2)
    return np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)


def _scratch_dir() -> Path:
    path = STORAGE_DIR / "tmp" / f"pyramid_test_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_levels(ngff_root: Path) -> list[np.ndarray]:
    level_count = len(list(ngff_root.glob("[0-9]*")))
    return [
        np.asarray(zarr.open_array(str(ngff_root / str(idx)), mode="r")[:])
        for idx in range(level_count)
    ]


class PyramidEquivalenceTests(TestCase):
    """The in-memory build reproduces the old store level for level."""

    def _assert_same_pyramid(self, width: int, height: int, *, seed: int = 0) -> None:
        scratch = _scratch_dir()
        try:
            plane = make_em_like_array(width, height, seed=seed)
            png_path = scratch / "canonical.png"
            Image.fromarray(plane, mode="L").save(
                str(png_path), "PNG", compress_level=3, optimize=False
            )

            legacy_image = _FakeImage(width, height)
            legacy_root = scratch / "legacy.zarr"
            _LegacyPyramid.build(legacy_image, png_path, legacy_root)

            new_image = _FakeImage(width, height)
            try:
                new_root = build_standalone(new_image, decode_canonical_array(png_path))

                legacy_levels = _read_levels(legacy_root)
                new_levels = _read_levels(new_root)
                self.assertEqual(
                    len(new_levels),
                    len(legacy_levels),
                    f"level count changed for {width}x{height}",
                )
                self.assertEqual(len(legacy_levels), len(_level_shapes(height, width)))
                for idx, (legacy, new) in enumerate(
                    zip(legacy_levels, new_levels, strict=True)
                ):
                    self.assertEqual(legacy.shape, new.shape, f"level {idx} shape")
                    np.testing.assert_array_equal(
                        new,
                        legacy,
                        err_msg=f"level {idx} differs for {width}x{height}",
                    )
            finally:
                shutil.rmtree(new_root, ignore_errors=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_single_chunk_image(self):
        self._assert_same_pyramid(300, 200)

    def test_multi_chunk_image_with_ragged_edges(self):
        # 2101 x 1537 -> a 3x2 chunk grid at level 0 whose right and bottom
        # tiles are partial, and level shapes that keep hitting odd sizes, so
        # every edge-padding branch in _downsample_region is exercised.
        self._assert_same_pyramid(2101, 1537, seed=3)

    def test_exact_chunk_multiple(self):
        self._assert_same_pyramid(2048, 1024, seed=7)

    def test_odd_by_one_image(self):
        self._assert_same_pyramid(1025, 1023, seed=11)

    def test_metadata_and_chunking_are_unchanged(self):
        width, height = 2101, 1537
        scratch = _scratch_dir()
        try:
            plane = make_em_like_array(width, height, seed=5)
            image = _FakeImage(width, height)
            try:
                root = build_standalone(image, plane)
                attrs_path = root / ".zattrs"
                self.assertTrue(_is_valid_ngff_store(root))
                attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
                multiscale = attrs["multiscales"][0]
                self.assertEqual(multiscale["version"], "0.4")
                self.assertEqual(
                    [axis["name"] for axis in multiscale["axes"]], ["c", "y", "x"]
                )
                expected_shapes = _level_shapes(height, width)
                self.assertEqual(len(multiscale["datasets"]), len(expected_shapes))
                for idx, dataset in enumerate(multiscale["datasets"]):
                    self.assertEqual(dataset["path"], str(idx))
                    self.assertEqual(
                        dataset["coordinateTransformations"][0]["scale"],
                        [1, 2**idx, 2**idx],
                    )
                    array = zarr.open_array(str(root / str(idx)), mode="r")
                    level_height, level_width = expected_shapes[idx]
                    self.assertEqual(array.shape, (1, level_height, level_width))
                    self.assertEqual(
                        array.chunks,
                        (
                            1,
                            min(NGFF_CHUNK_SIZE, level_height),
                            min(NGFF_CHUNK_SIZE, level_width),
                        ),
                    )
                    self.assertEqual(array.dtype, np.uint8)
            finally:
                shutil.rmtree(root, ignore_errors=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_every_level_is_a_two_by_two_box_mean_of_the_level_above(self):
        for width, height, seed in ((2101, 1537, 41), (1025, 1023, 43), (300, 200, 47)):
            with self.subTest(width=width, height=height):
                plane = make_em_like_array(width, height, seed=seed)
                image = _FakeImage(width, height)
                try:
                    root = build_standalone(image, plane)
                    levels = _read_levels(root)
                    np.testing.assert_array_equal(levels[0][0], plane)
                    expected = plane
                    for idx in range(1, len(levels)):
                        expected = _reference_half(expected)
                        np.testing.assert_array_equal(
                            levels[idx][0],
                            expected,
                            err_msg=f"level {idx} is not a box mean of level {idx - 1}",
                        )
                finally:
                    shutil.rmtree(root, ignore_errors=True)

    def test_refuses_a_plane_that_contradicts_the_recorded_geometry(self):
        image = _FakeImage(64, 48)
        with self.assertRaises(ValueError):
            build_standalone(image, np.zeros((48, 63), dtype=np.uint8))
        with self.assertRaises(ValueError):
            build_standalone(image, np.zeros((48, 64), dtype=np.uint16))


class BoxMeanKernelTests(TestCase):
    """The integer shortcut has to be the float expression, not an approximation."""

    def test_agrees_with_the_float_mean_on_every_possible_block_sum(self):
        # A 2x2 block of uint8 can sum to anything in 0..1020. Materialise one
        # block per sum and check the two formulations agree on all of them,
        # which is the whole domain of the function.
        sums = np.arange(0, 1021, dtype=np.int64)
        blocks = np.zeros((1021, 2, 2), dtype=np.uint8)
        blocks[:, 0, 0] = np.minimum(sums, 255)
        blocks[:, 0, 1] = np.clip(sums - 255, 0, 255)
        blocks[:, 1, 0] = np.clip(sums - 510, 0, 255)
        blocks[:, 1, 1] = np.clip(sums - 765, 0, 255)
        np.testing.assert_array_equal(blocks.sum(axis=(1, 2)), sums)

        region = blocks.transpose(1, 0, 2).reshape(2, 1021 * 2)
        fast = _downsample_region(region, target_height=1, target_width=1021)
        slow = np.rint(
            region.reshape(1, 2, 1021, 2).mean(axis=(1, 3)).astype(np.float64)
        ).astype(np.uint8)
        np.testing.assert_array_equal(fast, slow)

    def test_agrees_with_the_float_mean_on_random_regions(self):
        rng = np.random.default_rng(101)
        for shape in ((2, 2), (16, 24), (255, 129), (1024, 1024)):
            with self.subTest(shape=shape):
                region = rng.integers(0, 256, size=shape, dtype=np.uint8)
                target_height = shape[0] // 2
                target_width = shape[1] // 2
                fast = _downsample_region(
                    region, target_height=target_height, target_width=target_width
                )
                slow = np.rint(
                    region[: target_height * 2, : target_width * 2]
                    .reshape(target_height, 2, target_width, 2)
                    .mean(axis=(1, 3))
                ).astype(np.uint8)
                np.testing.assert_array_equal(fast, slow)

    def test_pads_a_ragged_region_by_repeating_the_edge(self):
        region = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        result = _downsample_region(region, target_height=2, target_width=2)
        padded = np.pad(region, ((0, 1), (0, 1)), mode="edge")
        expected = np.rint(padded.reshape(2, 2, 2, 2).mean(axis=(1, 3))).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_non_uint8_input_still_goes_through_the_float_path(self):
        region = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
        result = _downsample_region(region, target_height=1, target_width=1)
        self.assertEqual(result.dtype, np.uint8)
        # 2500 clipped into uint8 by astype, exactly as the previous code did.
        self.assertEqual(int(result[0, 0]), int(np.uint8(np.rint(2500.0))))


class SourcePlaneTests(TestCase):
    """One decode has to yield exactly the pixels the PNG encoder used to."""

    def _assert_plane_matches_legacy_png(self, array: np.ndarray, **write_kwargs) -> None:
        scratch = _scratch_dir()
        try:
            tiff_path = scratch / "source.tif"
            tifffile.imwrite(str(tiff_path), array, **write_kwargs)
            metadata = extract_tiff_metadata(tiff_path)

            legacy_png = scratch / "legacy.png"
            convert_tiff_to_png(tiff_path, metadata, legacy_png)
            with Image.open(legacy_png) as opened:
                legacy_pixels = np.asarray(opened.convert("L"), dtype=np.uint8)

            plane = load_source_plane_uint8(tiff_path, metadata)
            np.testing.assert_array_equal(plane, legacy_pixels)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_eight_bit_grayscale(self):
        self._assert_plane_matches_legacy_png(
            make_em_like_array(133, 97), photometric="minisblack"
        )

    def test_sixteen_bit_grayscale_uses_native_bit_depth_scaling(self):
        rng = np.random.default_rng(5)
        array = rng.integers(0, 65536, size=(97, 133), dtype=np.uint16)
        self._assert_plane_matches_legacy_png(array, photometric="minisblack")

    def test_thirty_two_bit_grayscale(self):
        rng = np.random.default_rng(9)
        array = rng.integers(0, 2**32, size=(41, 53), dtype=np.uint32)
        self._assert_plane_matches_legacy_png(array, photometric="minisblack")

    def test_multi_channel_keeps_the_first_channel(self):
        rng = np.random.default_rng(13)
        array = rng.integers(0, 256, size=(37, 59, 3), dtype=np.uint8)
        self._assert_plane_matches_legacy_png(array, photometric="rgb")

    def test_grayscale_stack_keeps_the_first_plane(self):
        stack = np.zeros((3, 24, 32), dtype=np.uint8)
        stack[0] = 11
        stack[1] = 99
        self._assert_plane_matches_legacy_png(stack)

    def test_png_source_matches_the_previous_canonicaliser(self):
        # ``convert_png_to_8bit_grayscale`` is what a staged PNG upload used to
        # go through. It is the reference for the PNG branch of
        # ``load_source_plane_uint8`` exactly as ``convert_tiff_to_png`` is for
        # the TIFF branch.
        scratch = _scratch_dir()
        try:
            for mode, source in (
                ("L", Image.fromarray(make_em_like_array(61, 43), mode="L")),
                (
                    "RGB",
                    Image.fromarray(
                        np.random.default_rng(19)
                        .integers(0, 256, size=(43, 61, 3))
                        .astype(np.uint8),
                        mode="RGB",
                    ),
                ),
            ):
                with self.subTest(mode=mode):
                    png_path = scratch / f"source_{mode}.png"
                    source.save(str(png_path))
                    legacy_path = scratch / f"legacy_{mode}.png"
                    convert_png_to_8bit_grayscale(png_path, legacy_path)
                    with Image.open(legacy_path) as opened:
                        legacy_pixels = np.asarray(opened.convert("L"), dtype=np.uint8)
                    np.testing.assert_array_equal(
                        load_source_plane_uint8(png_path, {}), legacy_pixels
                    )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_sixteen_bit_png_source(self):
        scratch = _scratch_dir()
        try:
            rng = np.random.default_rng(17)
            values = rng.integers(0, 65536, size=(29, 31), dtype=np.uint16)
            png_path = scratch / "source.png"
            Image.fromarray(values.astype(np.uint32), mode="I").convert("I;16").save(
                str(png_path)
            )
            plane = load_source_plane_uint8(png_path, {})
            with Image.open(png_path) as opened:
                raw = np.asarray(opened, dtype=np.uint16)
            expected = np.clip(
                np.asarray(raw, dtype=np.float32) * (255.0 / 65535.0), 0, 255
            ).astype(np.uint8)
            np.testing.assert_array_equal(plane, expected)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _stage_upload(
    array: np.ndarray, *, name: str = "import.tif", as_png: bool = False
) -> Asset:
    """Create the asset/rendition pair the upload endpoint creates."""

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    asset_id = uuid4()
    if as_png:
        staged = UPLOADS_DIR / f"{asset_id}.png"
        Image.fromarray(array, mode="L").save(str(staged))
        metadata = extract_image_metadata(staged)
    else:
        staged = UPLOADS_DIR / f"{asset_id}.tif"
        tifffile.imwrite(str(staged), array, photometric="minisblack")
        metadata = extract_tiff_metadata(staged)
    asset = Asset.objects.create(
        id=asset_id,
        display_name=name,
        original_filename=name,
        logical_width=int(metadata["width"]),
        logical_height=int(metadata["height"]),
        channels=int(metadata["channels"]),
        bit_depth=int(metadata["bit_depth"]),
        pixel_size_nm=metadata.get("pixel_size_nm"),
        preprocess_stage="ENCODING",
        preprocess_progress=0.0,
    )
    Rendition.objects.create(
        asset=asset,
        type=Rendition.TYPE_FULL,
        storage_root="DATA_DIR",
        stored_path=normalize_stored_path_value(staged, relative_to=DATA_DIR),
        path_exists=True,
        is_directory=False,
        stored_width=int(metadata["width"]),
        stored_height=int(metadata["height"]),
        stored_channels=int(metadata["channels"]),
        stored_bit_depth=int(metadata["bit_depth"]),
        metadata={"upload_state": "staged", "original_filename": name},
    )
    return asset


class PrepareAssetRenditionsTests(TestCase):
    def test_produces_the_canonical_png_and_a_matching_pyramid(self):
        array = make_em_like_array(1300, 1100, seed=21)
        asset = _stage_upload(array)
        staged_path = get_asset_openable(asset).path

        prepare_asset_renditions(str(asset.id))

        asset.refresh_from_db()
        openable = get_asset_openable(asset)
        self.assertEqual(openable.path.suffix, ".png")
        self.assertTrue(openable.path.exists())
        self.assertFalse(staged_path.exists(), "staged upload should be removed")
        self.assertEqual(asset.channels, 1)
        self.assertEqual(asset.bit_depth, 8)
        self.assertEqual(openable.channels, 1)
        self.assertEqual(openable.bit_depth, 8)

        with Image.open(openable.path) as png:
            self.assertEqual(png.mode, "L")
            png_pixels = np.asarray(png, dtype=np.uint8)
        np.testing.assert_array_equal(png_pixels, array)

        ngff_root = published_root(openable)
        self.assertTrue(_is_valid_ngff_store(ngff_root))
        level0 = np.asarray(zarr.open_array(str(ngff_root / "0"), mode="r")[0])
        np.testing.assert_array_equal(level0, array)

        ngff_rendition = asset.renditions.get(type=Rendition.TYPE_NGFF)
        self.assertEqual(ngff_rendition.stored_channels, 1)
        self.assertEqual(ngff_rendition.stored_bit_depth, 8)
        self.assertTrue(ngff_rendition.is_directory)
        self.assertEqual(asset.preprocess_stage, "ENCODING")
        self.assertEqual(asset.preprocess_progress, 55.0)

    def test_a_staged_png_upload_is_canonicalised_and_pyramided(self):
        # PNG is an accepted upload format, and its decode returns a
        # Pillow-backed read-only array -- which both the PNG encoder and the
        # zarr writer have to accept.
        array = make_em_like_array(1300, 1100, seed=37)
        asset = _stage_upload(array, name="import.png", as_png=True)
        staged_path = get_asset_openable(asset).path

        prepare_asset_renditions(str(asset.id))

        openable = get_asset_openable(asset)
        self.assertEqual(openable.path.parent.name, str(asset.id))
        self.assertFalse(staged_path.exists())
        with Image.open(openable.path) as png:
            self.assertEqual(png.mode, "L")
            np.testing.assert_array_equal(np.asarray(png, dtype=np.uint8), array)
        ngff_root = published_root(openable)
        self.assertTrue(_is_valid_ngff_store(ngff_root))
        level0 = np.asarray(zarr.open_array(str(ngff_root / "0"), mode="r")[0])
        np.testing.assert_array_equal(level0, array)

    def test_canonical_png_is_byte_identical_to_the_previous_encoder(self):
        array = make_em_like_array(700, 500, seed=23)
        asset = _stage_upload(array)
        staged_path = get_asset_openable(asset).path
        reference_source = staged_path.parent / f"reference_{uuid4().hex}.tif"
        shutil.copyfile(staged_path, reference_source)
        try:
            metadata = extract_tiff_metadata(reference_source)
            reference_png = IMAGES_DIR / f"reference_{uuid4().hex}.png"
            convert_tiff_to_png(reference_source, metadata, reference_png)

            prepare_asset_renditions(str(asset.id))
            produced = get_asset_openable(asset).path
            self.assertEqual(produced.read_bytes(), reference_png.read_bytes())
        finally:
            reference_source.unlink(missing_ok=True)

    def test_is_reentrant_when_the_source_is_already_canonical(self):
        array = make_em_like_array(600, 400, seed=27)
        asset = _stage_upload(array)
        prepare_asset_renditions(str(asset.id))
        canonical = get_asset_openable(asset).path
        first_bytes = canonical.read_bytes()

        from quantem.assets.task_utils import _open_generation_level_cache_clear

        _open_generation_level_cache_clear()
        shutil.rmtree(published_root(get_asset_openable(asset)), ignore_errors=True)

        prepare_asset_renditions(str(asset.id))

        self.assertTrue(canonical.exists())
        self.assertEqual(canonical.read_bytes(), first_bytes)
        # A rebuild is a *new* generation: nothing is written over, so the old
        # name never comes back.
        ngff_root = published_root(get_asset_openable(asset))
        self.assertTrue(_is_valid_ngff_store(ngff_root))
        level0 = np.asarray(zarr.open_array(str(ngff_root / "0"), mode="r")[0])
        np.testing.assert_array_equal(level0, array)

    def test_a_half_built_generation_is_never_published(self):
        """The completeness proof is at publish time, not at read time.

        A store missing a level used to be discovered by a reader, which is too
        late and was never sound anyway: ``.zattrs``, ``.zgroup`` and every
        ``<level>/.zarray`` are all written before the first pixel. Now the
        builder counts the chunk files of every level against the geometry --
        exact, because the writes are dense -- and a generation that cannot
        prove it finished is not sealed and not published.
        """

        from quantem.assets.canonical_decode import decode_canonical_plane
        from quantem.assets.ngff import build_pyramid
        from quantem.assets.pyramid_authority import (
            Intent,
            PublishedPyramid,
            request_build,
            resolve_pyramid,
        )

        array = make_em_like_array(1300, 1100, seed=29)
        asset = _stage_upload(array)
        prepare_asset_renditions(str(asset.id))
        openable = get_asset_openable(asset)
        good = published_root(openable)

        ticket = request_build(asset)
        plane = decode_canonical_plane(openable.path)
        real_write = ngff_module._write_levels_from_plane

        def _write_all_but_the_last_level(image, plane_arg, arrays, **kwargs):
            real_write(image, plane_arg, arrays[:-1], **kwargs)

        with patch.object(
            ngff_module, "_write_levels_from_plane", _write_all_but_the_last_level
        ):
            with self.assertRaises(RuntimeError) as caught:
                build_pyramid(ticket, openable, plane)
        self.assertIn("chunk files", str(caught.exception))
        self.assertFalse((ticket.root / "manifest.json").exists())

        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, PublishedPyramid)
        self.assertEqual(
            resolved.root, good, "a half-built generation replaced the published one"
        )
        rebuilt = np.asarray(zarr.open_array(str(resolved.root / "0"), mode="r")[0])
        np.testing.assert_array_equal(rebuilt, array)

    def test_roi_crop_from_the_pyramid_matches_the_canonical_png(self):
        array = make_em_like_array(1400, 1200, seed=31)
        asset = _stage_upload(array)
        prepare_asset_renditions(str(asset.id))
        openable = get_asset_openable(asset)

        roi = create_roi_image_from_image(
            openable, x=300, y=250, width=700, height=600, source="AUTO"
        )
        from quantem.core.config import ROIS_DIR

        with Image.open(ROIS_DIR / f"{roi.id}.png") as opened:
            crop = np.asarray(opened.convert("L"), dtype=np.uint8)
        np.testing.assert_array_equal(crop, array[250:850, 300:1000])

        with Image.open(openable.path) as png:
            png_crop = np.asarray(png.convert("L"), dtype=np.uint8)[250:850, 300:1000]
        np.testing.assert_array_equal(crop, png_crop)

    def test_tile_endpoint_serves_chunks_the_viewer_can_decode(self):
        array = make_em_like_array(2101, 1537, seed=33)
        asset = _stage_upload(array)
        prepare_asset_renditions(str(asset.id))
        openable = get_asset_openable(asset)
        ngff_root = published_root(openable)

        base = f"/ngff/assets/{asset.id}.zarr"
        root_response = self.client.get(base)
        self.assertEqual(root_response.status_code, 200)
        attrs = json.loads(b"".join(root_response.streaming_content))
        self.assertEqual(len(attrs["multiscales"][0]["datasets"]), len(_level_shapes(1537, 2101)))

        zarray_response = self.client.get(f"{base}/0/.zarray")
        self.assertEqual(zarray_response.status_code, 200)
        zarray = json.loads(b"".join(zarray_response.streaming_content))
        self.assertEqual(zarray["chunks"], [1, NGFF_CHUNK_SIZE, NGFF_CHUNK_SIZE])
        compressor = zarray["compressor"]
        self.assertEqual(compressor["id"], "blosc")

        codec = Blosc(
            cname=compressor["cname"],
            clevel=compressor["clevel"],
            shuffle=compressor["shuffle"],
        )
        chunk_grid_x = math.ceil(2101 / NGFF_CHUNK_SIZE)
        chunk_grid_y = math.ceil(1537 / NGFF_CHUNK_SIZE)
        for chunk_y in range(chunk_grid_y):
            for chunk_x in range(chunk_grid_x):
                response = self.client.get(f"{base}/0/0.{chunk_y}.{chunk_x}")
                self.assertEqual(response.status_code, 200)
                raw = codec.decode(b"".join(response.streaming_content))
                tile = np.frombuffer(raw, dtype=np.uint8).reshape(
                    NGFF_CHUNK_SIZE, NGFF_CHUNK_SIZE
                )
                y0 = chunk_y * NGFF_CHUNK_SIZE
                x0 = chunk_x * NGFF_CHUNK_SIZE
                y1 = min(1537, y0 + NGFF_CHUNK_SIZE)
                x1 = min(2101, x0 + NGFF_CHUNK_SIZE)
                np.testing.assert_array_equal(
                    tile[: y1 - y0, : x1 - x0],
                    array[y0:y1, x0:x1],
                    err_msg=f"tile {chunk_y}.{chunk_x} differs",
                )
        self.assertTrue(ngff_root.is_dir())

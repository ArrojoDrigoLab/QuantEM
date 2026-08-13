"""Openable has to mean the same pixels as finished.

Import now registers the NGFF pyramid -- the thing that flips ``ngff_ready``
and makes the viewer reachable -- while the canonical PNG is still being
encoded on a background thread. That window was measured at 13.4 s on a 475 MP
TIFF and up to ~70 s on a busy disk, and the library page navigates into the
viewer the instant the flag flips, so a user can now start work inside it.

Three properties are pinned here, one per defect the window exposed:

1. **Read agreement.** Every uint8 reader must hand back the *same pixels*
   during the window as it does after ``DONE``, for every supported source bit
   depth and channel count. It did not: ``load_image_array`` and friends
   ``Image.open(...).convert("L")`` the staged upload, which saturates an
   ``I;16`` TIFF and luma-blends an RGB one, while the canonical PNG is built
   by native bit-depth scaling of the first band.
2. **A failure says which stage failed.** The decode path let tifffile's bare
   ``failed to read N bytes, got M`` through, so a user saw a byte count with
   no hint that their image had failed to decode.
3. **An asset is never openable *and* FAILED.** The pyramid is registered
   before the PNG thread is joined; a failure in that join used to leave
   ``ngff_ready`` true on an asset the card renders as failed.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import tifffile
from django.test import TestCase
from PIL import Image

from quantem.assets import ngff as ngff_module
from quantem.assets import tasks as asset_tasks
from quantem.assets.asset_openable import (
    asset_ngff_ready,
    get_asset_openable,
)
from quantem.assets.models import Asset, Rendition
from quantem.assets.ngff import (
    _is_valid_ngff_store,
    regenerate_ngff_from_plane,
)
from quantem.assets.preprocess_status import set_stage
from quantem.assets.serializers import serialize_asset_entry
from quantem.assets.task_utils import (
    load_image_array,
    load_image_preview_array,
    load_image_roi_array,
)
from quantem.assets.tasks import prepare_asset_renditions
from quantem.assets.utils import (
    create_roi_image_from_image,
    extract_image_metadata,
    extract_tiff_metadata,
    load_source_plane_uint8,
)
from quantem.core.config import DATA_DIR, IMAGES_DIR, ROIS_DIR, UPLOADS_DIR
from quantem.core.local_storage import normalize_stored_path_value
from quantem.testing import make_em_like_array

#: Big enough to span several 1024^2 pyramid chunks in x and to leave the
#: bottom/right chunk ragged, small enough to keep the suite quick.
_WIDTH = 1300
_HEIGHT = 900

#: The window the ROI/measurement readers ask for. Deliberately not chunk
#: aligned, and not at the origin.
_ROI = (517, 233, 640, 480)


def _stage_upload(array: np.ndarray, *, as_png: bool = False, name: str | None = None) -> Asset:
    """Create exactly the asset/rendition pair the upload endpoint creates.

    The FULL rendition names the *staged* file and carries the source's own
    channel count and bit depth -- which is the state ``prepare_asset_renditions``
    starts from and, crucially, the state it is still in throughout the
    openable-but-still-encoding window.
    """

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    asset_id = uuid4()
    if as_png:
        staged = UPLOADS_DIR / f"{asset_id}.png"
        Image.fromarray(array).save(str(staged))
        metadata = extract_image_metadata(staged)
    else:
        staged = UPLOADS_DIR / f"{asset_id}.tif"
        photometric = "rgb" if array.ndim == 3 and array.shape[-1] == 3 else "minisblack"
        tifffile.imwrite(str(staged), array, photometric=photometric)
        metadata = extract_tiff_metadata(staged)
    asset = Asset.objects.create(
        id=asset_id,
        display_name=name or "window.tif",
        original_filename=name or "window.tif",
        logical_width=int(metadata["width"]),
        logical_height=int(metadata["height"]),
        channels=int(metadata["channels"]),
        bit_depth=int(metadata["bit_depth"]),
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
        metadata={"upload_state": "staged"},
    )
    return asset


def _source_cases() -> list[tuple[str, np.ndarray, bool]]:
    """(label, array, staged_as_png) over every supported source form."""

    base = make_em_like_array(_WIDTH, _HEIGHT, seed=11)
    sixteen_bit = (base.astype(np.uint32) * 257).astype(np.uint16)
    thirty_two_bit = (base.astype(np.uint64) * 16843009).astype(np.uint32)
    rgb = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 13, axis=0)], axis=-1).astype(
        np.uint8
    )
    stack = np.stack([base, np.roll(base, 3, axis=0), 255 - base]).astype(np.uint8)
    return [
        ("8-bit grayscale TIFF", base, False),
        ("16-bit grayscale TIFF", sixteen_bit, False),
        ("32-bit grayscale TIFF", thirty_two_bit, False),
        ("8-bit RGB TIFF", rgb, False),
        ("3-plane 8-bit grayscale TIFF", stack, False),
        ("8-bit PNG", base, True),
        ("16-bit PNG", sixteen_bit, True),
    ]


def _canonical_png_path(asset: Asset) -> Path:
    return IMAGES_DIR / str(asset.id) / f"{asset.original_filename.split('.')[0]}.png"


def _read_everything(openable) -> dict[str, np.ndarray]:
    """Every uint8 read path a user's work can go through, materialised."""

    x, y, width, height = _ROI
    full, _ = load_image_array(openable)
    return {
        "load_image_array": np.array(full, dtype=np.uint8, copy=True),
        "load_image_roi_array": np.array(
            load_image_roi_array(openable, x, y, width, height),
            dtype=np.uint8,
            copy=True,
        ),
        "load_image_preview_array": np.array(
            load_image_preview_array(openable, max_size=128), dtype=np.uint8, copy=True
        ),
    }


def _published_root(image):
    """Where this asset's published generation lives, or ``None``.

    ``get_ngff_paths`` is gone: a path is no longer something a caller derives,
    it is what the authority hands back with a published generation.
    """

    from quantem.assets.pyramid_authority import (
        Intent,
        PublishedPyramid,
        resolve_pyramid,
    )

    resolved = resolve_pyramid(image, intent=Intent.SERVE)
    return resolved.root if isinstance(resolved, PublishedPyramid) else None


def _enter_the_window(asset: Asset) -> None:
    """Put the asset in the openable-but-still-encoding state, exactly.

    That state is not a fiction and it is not raced for here: it is precisely
    "the NGFF rendition is registered, the FULL rendition still names the
    staged upload". ``prepare_asset_renditions`` reaches it at
    ``upsert_ngff_rendition`` and leaves it at the ``Rendition.objects.filter(...)
    .update(...)`` after ``png_writer.join()``. Reconstructing it directly
    keeps this test single-threaded and deterministic;
    ``EncodingWindowIsRealTests`` below is what proves the real pipeline
    actually passes through it.
    """

    openable = get_asset_openable(asset)
    plane = load_source_plane_uint8(
        openable.path,
        {"channels": openable.channels, "bit_depth": openable.bit_depth},
    )
    # Publishing *is* the registration now: one database UPDATE, no second
    # step that could disagree with it.
    regenerate_ngff_from_plane(openable, plane)


class InitialImportBorderTrimTests(TestCase):
    def test_the_canonical_image_is_trimmed_once_and_retries_do_not_crop_it_again(self):
        source = np.full((12, 14), 255, dtype=np.uint8)
        source[2:10, 3:11] = make_em_like_array(8, 8, seed=31)
        asset = _stage_upload(source, name="bordered.tif")

        prepare_asset_renditions(str(asset.id))

        asset.refresh_from_db()
        rendition = asset.renditions.get(type=Rendition.TYPE_FULL)
        canonical_path = get_asset_openable(asset).path
        first_canonical = np.asarray(Image.open(canonical_path))
        self.assertEqual(first_canonical.shape, (8, 8))
        self.assertEqual((asset.logical_width, asset.logical_height), (8, 8))
        self.assertEqual(rendition.metadata["upload_state"], "canonical")
        self.assertEqual(
            rendition.metadata["border_trim"],
            {
                "left": 3,
                "top": 2,
                "right": 3,
                "bottom": 2,
                "original_width": 14,
                "original_height": 12,
            },
        )

        prepare_asset_renditions(str(asset.id))

        second_canonical = np.asarray(Image.open(get_asset_openable(asset).path))
        self.assertTrue(np.array_equal(second_canonical, first_canonical))


class WindowReadAgreementTests(TestCase):
    """F1: openable must not mean "different pixels for a while"."""

    def test_every_reader_agrees_across_the_window_for_every_source_form(self):
        for label, array, as_png in _source_cases():
            with self.subTest(source=label):
                asset = _stage_upload(array, as_png=as_png)
                staged_path = get_asset_openable(asset).path

                _enter_the_window(asset)

                asset.refresh_from_db()
                self.assertTrue(
                    asset_ngff_ready(asset),
                    "the asset must be openable at this point or there is no window",
                )
                window_openable = get_asset_openable(asset)
                self.assertEqual(
                    window_openable.path,
                    staged_path,
                    "the FULL rendition still names the staged upload in the window",
                )
                during = _read_everything(window_openable)

                prepare_asset_renditions(str(asset.id))

                asset.refresh_from_db()
                done_openable = get_asset_openable(asset)
                self.assertEqual(done_openable.path.suffix, ".png")
                after = _read_everything(done_openable)

                for reader, pixels in during.items():
                    np.testing.assert_array_equal(
                        pixels,
                        after[reader],
                        err_msg=(
                            f"{reader} returned different pixels during the "
                            f"encoding window than after DONE, for a {label}"
                        ),
                    )

                # And the pixels both states agree on are the canonical ones,
                # not merely each other's.
                with Image.open(done_openable.path) as canonical:
                    self.assertEqual(canonical.mode, "L")
                    canonical_pixels = np.asarray(canonical, dtype=np.uint8)
                np.testing.assert_array_equal(
                    during["load_image_array"],
                    canonical_pixels,
                    err_msg=f"window read is not the canonical plane, for a {label}",
                )
                x, y, width, height = _ROI
                np.testing.assert_array_equal(
                    during["load_image_roi_array"],
                    canonical_pixels[y : y + height, x : x + width],
                    err_msg=f"window ROI is not the canonical crop, for a {label}",
                )

    def test_the_preview_is_the_same_thumbnail_the_canonical_png_gives(self):
        """The AUTO ROI heuristic scores this preview; it must not drift.

        Sourcing the preview from the pyramid instead of the canonical PNG
        would be a silent change to which 3000^2 window a user is handed to
        label unless the pixels and the resampling are both identical. They
        are, and this is the test that keeps them so.
        """

        asset = _stage_upload(make_em_like_array(_WIDTH, _HEIGHT, seed=12))
        prepare_asset_renditions(str(asset.id))
        asset.refresh_from_db()
        openable = get_asset_openable(asset)

        for max_size in (128, 1024):
            with self.subTest(max_size=max_size):
                with Image.open(openable.path) as png:
                    reference = png.copy()
                reference.thumbnail((max_size, max_size))
                np.testing.assert_array_equal(
                    load_image_preview_array(openable, max_size=max_size),
                    np.asarray(reference, dtype=np.uint8),
                )

    def test_a_window_outside_the_image_is_padded_as_before(self):
        """Falling off the edge keeps the requested shape, pyramid or not."""

        asset = _stage_upload(make_em_like_array(600, 400, seed=13))
        prepare_asset_renditions(str(asset.id))
        asset.refresh_from_db()
        openable = get_asset_openable(asset)

        window = load_image_roi_array(openable, 500, 300, 200, 200)

        self.assertEqual(window.shape, (200, 200))
        with Image.open(openable.path) as png:
            expected = np.asarray(png.crop((500, 300, 700, 500)), dtype=np.uint8)
        np.testing.assert_array_equal(window, expected)

    def test_readers_still_work_with_no_pyramid_at_all(self):
        """The pyramid is a fast path, not a dependency."""

        array = (make_em_like_array(320, 240, seed=14).astype(np.uint32) * 257).astype(np.uint16)
        asset = _stage_upload(array)
        openable = get_asset_openable(asset)
        self.assertIsNone(_published_root(openable))

        expected = load_source_plane_uint8(openable.path, {"channels": 1, "bit_depth": 16})
        full, _ = load_image_array(openable)

        np.testing.assert_array_equal(full, expected)
        np.testing.assert_array_equal(
            load_image_roi_array(openable, 40, 30, 100, 80),
            expected[30:110, 40:140],
        )

    def test_the_roi_png_is_the_canonical_crop_even_with_no_pyramid(self):
        """The saved ROI is what a user labels; it must be the same picture.

        With the pyramid present this already came out of level 0. Without one
        it went through ``convert("L")`` on the source file, which for a 16-bit
        TIFF is a saturated crop of an image the viewer will never show.
        """

        array = (make_em_like_array(400, 300, seed=25).astype(np.uint32) * 257).astype(np.uint16)
        asset = _stage_upload(array)
        openable = get_asset_openable(asset)
        self.assertIsNone(_published_root(openable))
        expected = load_source_plane_uint8(openable.path, {"channels": 1, "bit_depth": 16})

        roi = create_roi_image_from_image(openable, x=60, y=40, width=180, height=120)

        with Image.open(ROIS_DIR / f"{roi.id}.png") as saved:
            self.assertEqual(saved.mode, "L")
            np.testing.assert_array_equal(
                np.asarray(saved, dtype=np.uint8), expected[40:160, 60:240]
            )

    def test_a_pyramid_whose_geometry_disagrees_is_not_used(self):
        """A stale store must not be silently served in place of the image."""

        asset = _stage_upload(make_em_like_array(400, 300, seed=15))
        prepare_asset_renditions(str(asset.id))
        asset.refresh_from_db()
        openable = get_asset_openable(asset)

        # A store built for a different image, registered against this asset.
        other = _stage_upload(make_em_like_array(260, 180, seed=16))
        other_openable = get_asset_openable(other)
        wrong_root = regenerate_ngff_from_plane(
            other_openable,
            load_source_plane_uint8(other_openable.path, {"channels": 1, "bit_depth": 8}),
        )
        _publish_someone_elses_store(asset, wrong_root)
        asset.refresh_from_db()

        full, _ = load_image_array(get_asset_openable(asset))

        with Image.open(openable.path) as png:
            np.testing.assert_array_equal(full, np.asarray(png, dtype=np.uint8))


def _publish_someone_elses_store(asset: Asset, root):
    """Point this asset's state row at a store built for a different image.

    Only a test can do this now -- the builder writes into a generation
    directory the authority hands it -- but the *reader* must still refuse the
    wrong picture, so the geometry check is worth keeping honest.
    """

    from quantem.assets.models import Rendition
    from quantem.core.config import NGFF_TMP_DIR

    row = Rendition.objects.get(asset=asset, type=Rendition.TYPE_NGFF)
    metadata = dict(row.metadata)
    pyramid = dict(metadata.get("pyramid") or {})
    pyramid["published_generation"] = root.name
    manifest = dict(pyramid.get("published_manifest") or {})
    import json as _json

    manifest.update(_json.loads((root / "manifest.json").read_text(encoding="utf-8")))
    pyramid["published_manifest"] = manifest
    metadata["pyramid"] = pyramid
    row.metadata = metadata
    row.stored_path = root.relative_to(NGFF_TMP_DIR).as_posix()
    row.path_exists = True
    row.is_directory = True
    row.save(update_fields=["metadata", "stored_path", "path_exists", "is_directory"])


class EncodingWindowIsRealTests(TestCase):
    """The window this file is about is not hypothetical.

    Characterisation, not regression: this passes before and after the fix.
    It exists so ``WindowReadAgreementTests`` cannot be dismissed as testing a
    state the pipeline never reaches.
    """

    def test_the_pyramid_is_registered_while_the_canonical_png_is_still_writing(self):
        asset = _stage_upload(make_em_like_array(_WIDTH, _HEIGHT, seed=17))
        get_asset_openable(asset)
        observed: dict[str, object] = {}
        registered = threading.Event()
        real_save = asset_tasks.save_plane_as_canonical_png
        real_publish = ngff_module.publish

        def _publish_and_release_the_png(ticket, manifest):
            # Main thread. The compare-and-swap is what flips ``ngff_ready``
            # now; record what exists on disk at exactly that instant.
            observed["store_valid_at_registration"] = _is_valid_ngff_store(ticket.root)
            observed["png_written_at_registration"] = _canonical_png_path(asset).exists()
            registered.set()
            return real_publish(ticket, manifest)

        def _save_only_after_registration(plane, target_file_path):
            # Background thread. Deliberately no DB and no store reads from
            # here -- a TestCase's rows are invisible to a second connection,
            # and polling the store races zarr's atomic metadata rename.
            observed["png_thread"] = threading.current_thread().name
            observed["png_released"] = registered.wait(timeout=60.0)
            return real_save(plane, target_file_path)

        with (
            patch.object(asset_tasks, "save_plane_as_canonical_png", _save_only_after_registration),
            patch.object(ngff_module, "publish", _publish_and_release_the_png),
        ):
            prepare_asset_renditions(str(asset.id))

        self.assertTrue(observed["png_released"], "the canonical PNG thread never ran")
        self.assertTrue(
            observed["store_valid_at_registration"],
            "the pyramid must be complete when the asset becomes openable",
        )
        self.assertFalse(
            observed["png_written_at_registration"],
            "if the PNG already existed there would be no window to defend",
        )
        self.assertIn("canonical-png-", str(observed["png_thread"]))

        asset.refresh_from_db()
        self.assertTrue(asset_ngff_ready(asset))
        self.assertTrue(_canonical_png_path(asset).exists())


class ImportFailureNamesTheStageTests(TestCase):
    """F2: "failed to read 1050000 bytes, got 419846" is not an explanation."""

    def test_a_truncated_tiff_says_the_decode_failed(self):
        asset = _stage_upload(make_em_like_array(_WIDTH, _HEIGHT, seed=18))
        staged = get_asset_openable(asset).path
        raw = staged.read_bytes()
        staged.write_bytes(raw[: len(raw) // 3])

        with self.assertRaises(ValueError) as caught:
            prepare_asset_renditions(str(asset.id))

        message = str(caught.exception)
        self.assertIn("Error decoding TIFF", message)
        cause = caught.exception.__cause__
        self.assertIsNotNone(cause, "the underlying decode error must be chained")
        self.assertIn(
            str(cause),
            message,
            "naming the stage must not throw away what the decoder said",
        )

    def test_a_truncated_png_says_the_decode_failed(self):
        asset = _stage_upload(make_em_like_array(_WIDTH, _HEIGHT, seed=19), as_png=True)
        staged = get_asset_openable(asset).path
        raw = staged.read_bytes()
        staged.write_bytes(raw[: len(raw) // 3])

        with self.assertRaises(ValueError) as caught:
            prepare_asset_renditions(str(asset.id))

        message = str(caught.exception)
        self.assertIn("Error decoding PNG", message)
        cause = caught.exception.__cause__
        self.assertIsNotNone(cause)
        self.assertIn(str(cause), message)

    def test_a_canonical_png_write_failure_says_so(self):
        asset = _stage_upload(make_em_like_array(400, 300, seed=20))

        with patch.object(
            asset_tasks,
            "save_plane_as_canonical_png",
            side_effect=OSError("injected canonical-PNG write failure"),
        ):
            with self.assertRaises(ValueError) as caught:
                prepare_asset_renditions(str(asset.id))

        message = str(caught.exception)
        self.assertIn("canonical PNG", message)
        self.assertIn("injected canonical-PNG write failure", message)

    def test_a_pyramid_failure_says_so(self):
        asset = _stage_upload(make_em_like_array(400, 300, seed=21))

        with patch.object(
            asset_tasks,
            "build_and_publish",
            side_effect=OSError("injected pyramid failure"),
        ):
            with self.assertRaises(ValueError) as caught:
                prepare_asset_renditions(str(asset.id))

        message = str(caught.exception)
        self.assertIn("pyramid", message)
        self.assertIn("injected pyramid failure", message)


class FailedImportIsNotOpenableTests(TestCase):
    """F3: the card and the viewer must not disagree."""

    def test_a_canonical_png_failure_leaves_a_closed_asset_not_a_half_open_one(self):
        asset = _stage_upload(make_em_like_array(_WIDTH, _HEIGHT, seed=22))
        staged = get_asset_openable(asset).path

        with patch.object(
            asset_tasks,
            "save_plane_as_canonical_png",
            side_effect=OSError("injected canonical-PNG write failure"),
        ):
            with self.assertRaises(ValueError) as caught:
                prepare_asset_renditions(str(asset.id))

        # What jobs.failure_reconcile._reconcile_asset_preprocessing does to
        # the asset when the job it belongs to dies.
        asset.refresh_from_db()
        set_stage(asset, "FAILED", progress=0.0, error=str(caught.exception))
        asset.refresh_from_db()

        self.assertFalse(asset_ngff_ready(asset))
        entry = serialize_asset_entry(asset)
        self.assertEqual(entry["preprocess_stage"], "FAILED")
        self.assertFalse(entry["ngff_ready"])
        self.assertFalse(entry["can_view"])
        self.assertFalse(entry["can_segment"])
        self.assertIsNone(entry["ngff_url"])

        # The source is untouched, so the retry has something to work from.
        self.assertEqual(get_asset_openable(asset).path, staged)
        self.assertTrue(staged.exists())

    def test_the_retry_after_such_a_failure_still_recovers(self):
        asset = _stage_upload(make_em_like_array(_WIDTH, _HEIGHT, seed=23))
        source = np.asarray(
            load_source_plane_uint8(get_asset_openable(asset).path, {"channels": 1, "bit_depth": 8})
        )

        with patch.object(
            asset_tasks,
            "save_plane_as_canonical_png",
            side_effect=OSError("injected canonical-PNG write failure"),
        ):
            with self.assertRaises(ValueError):
                prepare_asset_renditions(str(asset.id))
        asset.refresh_from_db()
        set_stage(asset, "FAILED", progress=0.0, error="injected")

        prepare_asset_renditions(str(asset.id))

        asset.refresh_from_db()
        self.assertTrue(asset_ngff_ready(asset))
        openable = get_asset_openable(asset)
        self.assertEqual(openable.path.suffix, ".png")
        full, _ = load_image_array(openable)
        np.testing.assert_array_equal(full, source)

    def test_a_decode_failure_does_not_leave_an_earlier_pyramid_openable(self):
        """A retry that dies before the pyramid must also close the asset.

        Otherwise a second failure on an asset that had been openable leaves
        the same contradiction by a different route.
        """

        asset = _stage_upload(make_em_like_array(400, 300, seed=24))
        prepare_asset_renditions(str(asset.id))
        asset.refresh_from_db()
        self.assertTrue(asset_ngff_ready(asset))

        with patch.object(
            asset_tasks,
            "decode_canonical_plane",
            side_effect=ValueError("Error decoding TIFF: injected"),
        ):
            with self.assertRaises(ValueError):
                prepare_asset_renditions(str(asset.id))

        asset.refresh_from_db()
        set_stage(asset, "FAILED", progress=0.0, error="injected")
        self.assertFalse(asset_ngff_ready(asset))
        self.assertFalse(serialize_asset_entry(asset)["can_view"])

"""The source matrix: every container x dtype x bands x staging state.

Three rounds of hand-enumerated tests missed the defect; both verifiers found
it by fuzzing a matrix. So the cells here are *generated* from one table, and
every pixel assertion is against ``test_ngff_reference`` -- an oracle that
imports nothing from the application.

Two cells are named because each is a finding:

* **``png / uint16 / 1 band / lazy rebuild``** (and its
  canonical-PNG-write-fails variant in ``test_ngff_failed_reopen_race``). Round
  3 *had* a test aimed at this exact vector,
  ``test_even_running_the_ngff_task_directly_cannot_reopen_it``, and it passed
  -- because its ``_stage_upload`` hard-coded ``.tif``. The
  **staging-state x container axis is the single axis that would have caught
  it**, and it is why this file exists.
* **``tif / uint16 / 1 band / pyvips present``**. No verification round has
  ever executed the libvips arm, because libvips is not installed on this box;
  the design review found it *clamps* where the canonical path *scales*
  (127.646 against 120.625, 260 200 of 262 144 pixels). The cells below install
  a stub ``pyvips`` that behaves that way and require the pyramid to be
  unchanged -- which it is, because the one decoder never asks libvips
  anything.

The one sentence every assertion here reduces to:

    a read either returns bytes equal to the independent oracle or raises a
    named exception, and the asset's advertised state is true exactly when a
    read would return oracle bytes.
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from django.test import Client, TestCase
from PIL import Image

from quantem.assets import canonical_decode, task_utils
from quantem.assets.canonical_decode import (
    DECODER_VERSION,
    UnrecognisedContainer,
    UnsupportedPixelType,
    decode_canonical_plane,
)
from quantem.assets.models import Asset, Rendition
from quantem.assets.ngff import render_lowest_resolution_ngff_png_from_root
from quantem.assets.pyramid_authority import (
    Intent,
    PublishedPyramid,
    Reason,
    Unavailable,
    resolve_pyramid,
)
from quantem.assets.serializers import serialize_asset_entry
from quantem.assets.tasks import ensure_ngff_for_asset_task, prepare_asset_renditions
from quantem.assets.utils import (
    PNG_COMPRESS_LEVEL,
    TIFF_UPLOAD_SUFFIXES,
    UPLOAD_SUFFIXES,
    extract_image_metadata,
)
from quantem.core.config import DATA_DIR, IMAGES_DIR, UPLOADS_DIR
from quantem.core.local_storage import normalize_stored_path_value

from . import test_ngff_reference as ref

Image.MAX_IMAGE_PIXELS = None

#: Small enough that ~90 cells run in seconds; ``BIG`` crosses the 1024 chunk
#: boundary in both axes and gives more than one pyramid level with real
#: downsampling, which is where an off-by-one in the box mean would show.
SMALL = (300, 260)
BIG = (1300, 1100)


def _pattern(shape: tuple[int, int]) -> np.ndarray:
    """Values in (0, 1): a gradient plus structure, never flat, never clipped."""

    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    values = 0.18 + 0.55 * (xx / max(width - 1, 1))
    values += 0.18 * np.sin(yy / 17.0) * np.cos(xx / 23.0)
    return np.clip(values, 0.02, 0.94)


def _as_dtype(shape: tuple[int, int], dtype: str) -> np.ndarray:
    base = _pattern(shape)
    if dtype == "uint8":
        return (base * 255).astype(np.uint8)
    if dtype == "uint16":
        return (base * 65535).astype(np.uint16)
    if dtype == "uint32":
        return (base * 4294967295).astype(np.uint32)
    if dtype == "float32":
        return (base * 255.0).astype(np.float32)
    if dtype == "int32neg":
        return (base * 1000.0 - 500.0).astype(np.int32)
    if dtype == "int32pos":
        return (base * 1000.0).astype(np.int32)
    if dtype == "complex64":
        return (base + 1j * base).astype(np.complex64)
    raise AssertionError(f"unknown dtype {dtype}")


def _with_bands(plane: np.ndarray, bands: str) -> np.ndarray:
    if bands == "1":
        return plane
    others = [plane, (plane / 2).astype(plane.dtype), (plane / 3).astype(plane.dtype)]
    if bands == "3i":
        return np.stack(others, axis=-1)
    return np.stack(others, axis=0)


#: Names a real EM image, or a directory of them (``*_em.png`` is taken, first
#: by name). Real EM images are not part of this distribution and there is
#: deliberately no default here -- a default would ship one laboratory's mount
#: point to everyone who downloads the source distribution. Unset, the real-
#: image cell skips; see ``test_a_real_em_image_imports_to_oracle_bytes``.
_REAL_EM_ENV_VAR = "QUANTEM_TEST_EM_IMAGE"


def _real_em_image() -> Path | None:
    setting = os.environ.get(_REAL_EM_ENV_VAR, "").strip()
    if not setting:
        return None
    candidate = Path(setting)
    if candidate.is_file():
        return candidate
    try:
        for found in sorted(candidate.glob("*_em.png")):
            return found
    except OSError:
        return None
    return None


@dataclass(frozen=True)
class Cell:
    container: str  # "tif" | "tiff" | "png"
    dtype: str
    bands: str  # "1" | "3i" | "3p"
    size: str  # "small" | "big"
    suffix: str  # the *name* it is written under, which may disagree
    pyvips: bool = False

    @property
    def name(self) -> str:
        vips = "+vips" if self.pyvips else ""
        return f"{self.container}/{self.dtype}/{self.bands}/{self.size}/as{self.suffix}{vips}"

    @property
    def shape(self) -> tuple[int, int]:
        return BIG if self.size == "big" else SMALL


def _write_cell(cell: Cell, directory: Path) -> Path:
    array = _with_bands(_as_dtype(cell.shape, cell.dtype), cell.bands)
    path = directory / f"cell-{uuid.uuid4().hex[:8]}{cell.suffix}"
    if cell.container == "png":
        if cell.dtype == "uint16":
            Image.fromarray(array.astype("<u2")).save(path, format="PNG")
        elif cell.bands == "1":
            Image.fromarray(array, mode="L").save(path, format="PNG")
        else:
            Image.fromarray(array, mode="RGB").save(path, format="PNG")
    else:
        tifffile.imwrite(str(path), array)
    return path


def _build_matrix() -> list[Cell]:
    cells: list[Cell] = []
    tif_dtypes = ["uint8", "uint16", "uint32", "float32", "int32pos", "int32neg", "complex64"]
    for dtype in tif_dtypes:
        for bands in ("1", "3i", "3p"):
            cells.append(Cell("tif", dtype, bands, "small", ".tif"))
    # A second TIFF suffix, so the dispatch cannot start depending on ".tif".
    for dtype in ("uint8", "uint16"):
        cells.append(Cell("tiff", dtype, "1", "small", ".tiff"))
    # PNG holds exactly these three.
    cells.append(Cell("png", "uint8", "1", "small", ".png"))
    cells.append(Cell("png", "uint8", "3i", "small", ".png"))
    cells.append(Cell("png", "uint16", "1", "small", ".png"))
    # Containers whose bytes disagree with their name -- the class round 3's
    # ``suffix == ".png"`` test got wrong.
    cells.append(Cell("tif", "uint16", "1", "small", ".png"))
    cells.append(Cell("png", "uint16", "1", "small", ".tif"))
    cells.append(Cell("png", "uint8", "3i", "small", ".tif"))
    # Multi-chunk, multi-level: every scaling dtype and every band layout.
    for dtype in ("uint8", "uint16", "uint32", "float32"):
        cells.append(Cell("tif", dtype, "1", "big", ".tif"))
    cells.append(Cell("tif", "uint16", "3i", "big", ".tif"))
    cells.append(Cell("tif", "uint8", "3p", "big", ".tif"))
    cells.append(Cell("png", "uint16", "1", "big", ".png"))
    cells.append(Cell("png", "uint8", "3i", "big", ".png"))
    # The libvips axis, on the cells where a clamp-instead-of-scale would show.
    for dtype in ("uint16", "uint32"):
        cells.append(Cell("tif", dtype, "1", "small", ".tif", pyvips=True))
    cells.append(Cell("tif", "uint16", "3i", "small", ".tif", pyvips=True))
    cells.append(Cell("png", "uint16", "1", "small", ".png", pyvips=True))
    return cells


MATRIX = _build_matrix()


class _ClampingVips:
    """A stand-in for the libvips arm nobody has ever executed.

    ``pyvips`` is not installed here, which is exactly why three verification
    rounds never saw that ``image.cast("uchar")`` *clamps* 16-bit data where the
    canonical decode *scales* it. This stub reproduces that behaviour, so a cell
    that reaches for libvips would come back visibly wrong -- and the assertion
    is that no cell does.
    """

    class Image:  # noqa: D106 - mirrors the pyvips namespace shape
        @staticmethod
        def new_from_file(path, **kwargs):  # pragma: no cover - must never be called
            raise AssertionError(
                f"something asked libvips to decode {path}; the canonical decoder "
                "is the only decode in the tree and it never does"
            )


def stage_asset(path: Path) -> Asset:
    """Create the Asset + FULL rendition an upload would, without a job.

    Module level so the race harnesses share one fixture with the matrix: a
    second, subtly different staging helper is how round 3 ended up testing
    only ``.tif``.
    """

    metadata = extract_image_metadata(path)
    asset_id = uuid.uuid4()
    staged = UPLOADS_DIR / f"{asset_id}{path.suffix}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, staged)
    asset = Asset.objects.create(
        id=asset_id,
        display_name=path.name,
        original_filename=path.name,
        logical_width=int(metadata["width"]),
        logical_height=int(metadata["height"]),
        channels=int(metadata["channels"]),
        bit_depth=int(metadata["bit_depth"]),
        preprocess_stage="ENCODING",
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
        metadata={"source_metadata": {}},
    )
    return asset


def _run_import_as_the_job_would(asset) -> Exception | None:
    """``prepare_asset_renditions`` plus the job layer's failure reconciler.

    Calling the task alone would leave a failed import in ``ENCODING``: the
    terminal ``FAILED`` transition is written by ``jobs.failure_reconcile`` when
    the job exhausts its attempts. The tests need the state the *user* ends up
    looking at, so they run both halves -- which is also what makes the
    ``failure_detail`` versus ``preprocess_error`` split testable, because the
    reconciler writes only the latter.
    """

    from quantem.jobs.constants import JOB_TYPE_UPLOAD_IMAGE_PIPELINE
    from quantem.jobs.failure_reconcile import reconcile_domain_objects_for_failed_job

    try:
        prepare_asset_renditions(str(asset.id))
    except Exception as exc:  # noqa: BLE001 - the job runner does exactly this
        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            {"asset_id": str(asset.id)},
            f"failed: {type(exc).__name__}: {exc}",
            supersede_stale_failure=True,
        )
        return exc
    return None


class SourceMatrixTests(TestCase):
    """Every cell, through a fresh import and through a lazy rebuild."""

    maxDiff = None

    def setUp(self):
        self.sources = Path(DATA_DIR) / "matrix-sources"
        self.sources.mkdir(parents=True, exist_ok=True)
        self.client = Client()

    # -- fixtures ---------------------------------------------------------

    def _stage(self, path: Path) -> Asset:
        return stage_asset(path)

    # -- assertions -------------------------------------------------------

    def _assert_every_reader_agrees(self, cell: Cell, asset: Asset, expected: np.ndarray):
        from quantem.assets.asset_openable import get_asset_openable

        openable = get_asset_openable(asset)
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, PublishedPyramid, f"{cell.name}: {resolved}")

        # 1. level 0 is the oracle plane, bit for bit.
        level0 = np.asarray(resolved.open_level(0)[0])
        self.assertEqual(
            int(np.abs(level0.astype(np.int32) - expected.astype(np.int32)).max()),
            0,
            f"{cell.name}: level 0 differs from the oracle",
        )

        # 2. the blunt saturation check -- one line, and it is the one that
        #    would have caught round 3's all-white pyramid without knowing
        #    anything about the transform.
        if float(expected.mean()) not in (0.0, 255.0):
            self.assertNotIn(
                round(float(level0.mean()), 6),
                (0.0, 255.0),
                f"{cell.name}: the published pyramid is uniformly black or white "
                f"but the oracle mean is {expected.mean():.3f}",
            )

        # 3. every level equals the oracle's own downsample chain.
        oracle_levels = ref.pyramid(expected)
        self.assertEqual(resolved.level_count, len(oracle_levels), cell.name)
        for index, oracle_level in enumerate(oracle_levels):
            got = np.asarray(resolved.open_level(index)[0])
            self.assertEqual(
                int(np.abs(got.astype(np.int32) - oracle_level.astype(np.int32)).max()),
                0,
                f"{cell.name}: pyramid level {index} differs from the oracle",
            )

        # 4. the manifest's chunk count is exact, and every chunk file is there.
        for level in resolved.manifest["levels"]:
            files = [
                child.name
                for child in (resolved.root / level["path"]).iterdir()
                if not child.name.startswith(".")
            ]
            self.assertEqual(
                len(files),
                level["chunk_count"],
                f"{cell.name}: level {level['path']} has {len(files)} chunk files, "
                f"manifest says {level['chunk_count']}",
            )
        self.assertTrue(resolved.manifest["dense"], cell.name)
        self.assertEqual(resolved.manifest["decoder_version"], DECODER_VERSION, cell.name)

        # 5. every reader, on three windows: interior, chunk-straddling and
        #    edge-clipped.
        height, width = expected.shape
        windows = [
            (5, 5, 32, 32),
            (max(0, width - 1050), max(0, height - 1050), 64, 64),
            (width - 20, height - 20, 40, 40),
        ]
        plane, _ = task_utils.load_image_array(openable)
        np.testing.assert_array_equal(plane, expected, err_msg=f"{cell.name}: load_image_array")
        np.testing.assert_array_equal(
            task_utils.load_image_preview_array(openable, 64),
            ref.thumbnail(expected, 64),
            err_msg=f"{cell.name}: load_image_preview_array",
        )
        for x, y, w, h in windows:
            crop = np.array(
                Image.fromarray(expected, mode="L").crop((x, y, x + w, y + h)), dtype=np.uint8
            )
            np.testing.assert_array_equal(
                task_utils.load_image_roi_array(openable, x, y, w, h),
                crop,
                err_msg=f"{cell.name}: load_image_roi_array at {(x, y, w, h)}",
            )
        x, y, w, h = windows[0]
        np.testing.assert_array_equal(
            task_utils.load_image_ngff_level0_roi_array(openable, x, y, w, h),
            expected[y : y + h, x : x + w],
            err_msg=f"{cell.name}: load_image_ngff_level0_roi_array",
        )

        # 6. the ROI PNG writer, which crops straight out of level 0.
        from quantem.assets.utils import _save_roi_png_from_ngff

        roi_png = Path(DATA_DIR) / f"roi-{uuid.uuid4().hex[:8]}.png"
        self.assertTrue(_save_roi_png_from_ngff(openable, roi_png, x=x, y=y, width=w, height=h))
        np.testing.assert_array_equal(
            np.array(ref.read_array(roi_png), dtype=np.uint8),
            expected[y : y + h, x : x + w],
            err_msg=f"{cell.name}: _save_roi_png_from_ngff",
        )

        # 7. the thumbnail renderer and the HTTP routes.
        self.assertTrue(render_lowest_resolution_ngff_png_from_root(resolved.root))
        for url in (f"/ngff/assets/{asset.id}.zarr/.zattrs", f"/ngff/assets/{asset.id}.zarr/0/0.0.0"):
            response = self.client.get(url)
            try:
                self.assertEqual(response.status_code, 200, f"{cell.name} {url}")
                self.assertEqual(response["ETag"], f'"{resolved.generation_id}"', cell.name)
                self.assertTrue(b"".join(response.streaming_content), cell.name)
            finally:
                # Windows will not delete a directory a FileResponse still holds
                # open, and the lazy-rebuild half of this cell deletes exactly
                # that directory. Closing here is the test being tidy; the
                # product's own answer to a held handle is the sweeper's
                # ``still_held`` count, exercised in test_ngff_publish_race.
                response.close()

        # 8. what the library card says is what a read would give.
        entry = serialize_asset_entry(Asset.objects.get(id=asset.id))
        self.assertTrue(entry["ngff_ready"], cell.name)
        self.assertTrue(entry["can_view"] and entry["can_segment"], cell.name)
        self.assertEqual(entry["ngff_url"], f"/ngff/assets/{asset.id}.zarr", cell.name)

    def _assert_refused_by_name(self, cell: Cell, asset: Asset, expected_error: Exception):
        asset.refresh_from_db()
        self.assertEqual(asset.preprocess_stage, "FAILED", cell.name)
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, Unavailable, cell.name)
        self.assertEqual(resolved.reason, Reason.TERMINAL_FAILURE, cell.name)
        entry = serialize_asset_entry(Asset.objects.get(id=asset.id))
        self.assertFalse(entry["ngff_ready"] or entry["can_view"] or entry["can_segment"])
        self.assertIsNone(entry["ngff_url"], cell.name)
        for url in (
            f"/ngff/assets/{asset.id}.zarr",
            f"/ngff/assets/{asset.id}.zarr/.zattrs",
            f"/ngff/assets/{asset.id}.zarr/0/0.0.0",
        ):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 409, f"{cell.name} {url}")
            response.close()
        # ``load_image_array`` must refuse rather than hand back the staged
        # upload's pixels: a failed import has no pixels to give.
        with self.assertRaises(task_utils.PyramidUnavailable):
            task_utils.load_image_array(asset)

    # -- the size axis the design names: one real EM image ----------------

    def test_a_real_em_image_imports_to_oracle_bytes(self):
        """The matrix's real-image cell, when a real image is reachable.

        The design names "one real EM image" as a matrix axis. The suite may
        not *depend* on one -- a developer without a copy would see a red gate
        for no reason -- so this skips unless ``QUANTEM_TEST_EM_IMAGE``
        resolves. When it does resolve the cell is the full one: every reader,
        every level, the canonical PNG, the routes and the saturation check,
        all against the oracle.
        """

        source = _real_em_image()
        if source is None:
            self.skipTest(
                "no real EM image available; set QUANTEM_TEST_EM_IMAGE to one "
                "(the synthesised cells above cover the same code paths)"
            )
        local = self.sources / f"real-{uuid.uuid4().hex[:8]}{source.suffix}"
        shutil.copy2(source, local)
        cell = Cell("real", "as-found", "1", "real", source.suffix)
        expected = ref.decode(local)
        asset = stage_asset(local)
        self.assertIsNone(_run_import_as_the_job_would(asset))
        asset.refresh_from_db()
        self.assertEqual(asset.preprocess_stage, "ENCODING", asset.preprocess_error)
        self._assert_every_reader_agrees(cell, asset, expected)

    # -- the cells --------------------------------------------------------

    def _run_cell(self, cell: Cell):
        path = _write_cell(cell, self.sources)

        oracle_error = None
        try:
            expected = ref.decode(path)
        except ref.ReferenceRefusal as exc:
            expected, oracle_error = None, exc

        installed = None
        if cell.pyvips:
            installed = sys.modules.get("pyvips")
            sys.modules["pyvips"] = _ClampingVips  # type: ignore[assignment]
        try:
            self._run_cell_inner(cell, path, expected, oracle_error)
        finally:
            if cell.pyvips:
                if installed is None:
                    sys.modules.pop("pyvips", None)
                else:  # pragma: no cover - libvips is not installed here
                    sys.modules["pyvips"] = installed

    def _run_cell_inner(self, cell, path, expected, oracle_error):
        # Upload-time refusal is allowed only for a container whose bytes
        # disagree with its name, and only with a message that says so.
        try:
            asset = self._stage(path)
        except ValueError as exc:
            self.assertNotEqual(
                cell.container, Path(path).suffix.lstrip(".").replace("tiff", "tif"),
                f"{cell.name}: a matching container was refused at upload: {exc}",
            )
            self.assertIn("Error reading", str(exc), cell.name)
            return

        import_error = _run_import_as_the_job_would(asset)
        asset.refresh_from_db()

        if oracle_error is not None:
            # complex / negative-signed: the product must refuse by name too,
            # rather than discarding the imaginary part or clipping the negative
            # half to black, which is what both decoders used to do in silence.
            self.assertIsNotNone(import_error, f"{cell.name}: imported without complaint")
            self.assertIsInstance(import_error, UnsupportedPixelType, cell.name)
            self._assert_refused_by_name(cell, asset, oracle_error)
            self.assertIn(
                "UnsupportedPixelType",
                asset.preprocess_error,
                f"{cell.name}: refused, but not by name: {asset.preprocess_error}",
            )
            return
        self.assertIsNone(import_error, f"{cell.name}: {import_error}")

        self.assertEqual(
            asset.preprocess_stage,
            "ENCODING",
            f"{cell.name}: import did not complete: {asset.preprocess_error}",
        )
        self._assert_every_reader_agrees(cell, asset, expected)

        # The canonical PNG is byte-identical to the oracle's.
        from quantem.assets.asset_openable import get_asset_openable

        canonical_png = get_asset_openable(asset).path
        self.assertEqual(canonical_png.parent.parent, IMAGES_DIR, cell.name)
        self.assertEqual(
            canonical_png.read_bytes(),
            ref.canonical_png_bytes(expected, compress_level=PNG_COMPRESS_LEVEL),
            f"{cell.name}: the canonical PNG is not what the oracle would write",
        )

        # -- staging state 2: the lazy rebuild ----------------------------
        published_before = resolve_pyramid(asset, intent=Intent.SERVE)
        task_utils._open_generation_level_cache_clear()
        shutil.rmtree(published_before.root)
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, Unavailable, f"{cell.name}: lazy state")
        response = self.client.get(f"/ngff/assets/{asset.id}.zarr/0/0.0.0")
        self.assertEqual(response.status_code, 202, cell.name)
        response.close()
        result = ensure_ngff_for_asset_task(str(asset.id))
        self.assertEqual(result["status"], "published", f"{cell.name}: {result}")
        rebuilt = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(rebuilt, PublishedPyramid, cell.name)
        self.assertNotEqual(rebuilt.generation_id, published_before.generation_id, cell.name)
        self._assert_every_reader_agrees(cell, asset, expected)


def _make_case(cell: Cell):
    def case(self):
        self._run_cell(cell)

    case.__name__ = "test_" + cell.name.replace("/", "_").replace(".", "").replace("+", "_")
    case.__doc__ = f"{cell.name}: pyramid, readers, routes and lazy rebuild vs the oracle"
    return case


for _cell in MATRIX:
    _case = _make_case(_cell)
    setattr(SourceMatrixTests, _case.__name__, _case)


class MatrixCoverageTests(TestCase):
    """The matrix has to keep up with what the importer claims to accept."""

    def test_every_accepted_container_and_dtype_has_a_cell(self):
        covered_suffixes = {cell.suffix for cell in MATRIX}
        missing = set(UPLOAD_SUFFIXES) - covered_suffixes
        self.assertEqual(
            missing,
            set(),
            f"the importer accepts {sorted(missing)} and the matrix has no cell for them",
        )
        covered = {(cell.container, cell.dtype, cell.bands) for cell in MATRIX}
        required = {
            ("tif", dtype, bands)
            for dtype in ("uint8", "uint16", "uint32", "float32", "int32neg", "complex64")
            for bands in ("1", "3i", "3p")
        } | {("png", "uint8", "1"), ("png", "uint8", "3i"), ("png", "uint16", "1")}
        self.assertEqual(
            required - covered,
            set(),
            "extract_tiff_metadata accepts dtypes the matrix does not exercise",
        )
        self.assertTrue(any(cell.pyvips for cell in MATRIX))
        self.assertTrue(any(cell.size == "big" for cell in MATRIX))
        self.assertTrue(TIFF_UPLOAD_SUFFIXES)

    def test_a_container_is_chosen_by_its_bytes_and_never_by_its_name(self):
        directory = Path(DATA_DIR) / "sniff"
        directory.mkdir(parents=True, exist_ok=True)
        array = (_pattern((64, 64)) * 65535).astype(np.uint16)
        as_tiff = directory / "really-a-tiff.png"
        tifffile.imwrite(str(as_tiff), array)
        as_png = directory / "really-a-png.tif"
        Image.fromarray(array.astype("<u2")).save(as_png, format="PNG")
        self.assertEqual(canonical_decode.sniff_container(as_tiff), "tiff")
        self.assertEqual(canonical_decode.sniff_container(as_png), "png")
        for path in (as_tiff, as_png):
            np.testing.assert_array_equal(decode_canonical_plane(path).array, ref.decode(path))
        nonsense = directory / "nonsense.tif"
        nonsense.write_bytes(b"not an image at all")
        with self.assertRaises(UnrecognisedContainer):
            decode_canonical_plane(nonsense)

    def test_the_decoder_refuses_complex_and_negative_signed_by_name(self):
        directory = Path(DATA_DIR) / "refusals"
        directory.mkdir(parents=True, exist_ok=True)
        for dtype, fragment in (("complex64", "complex"), ("int32neg", "negative")):
            path = directory / f"{dtype}.tif"
            tifffile.imwrite(str(path), _as_dtype((32, 32), dtype))
            with self.assertRaises(UnsupportedPixelType) as caught:
                decode_canonical_plane(path)
            self.assertIn(fragment, str(caught.exception))

    def test_a_canonical_plane_cannot_be_built_from_the_wrong_thing(self):
        """The type is the lock that holds when the linters are edited around."""

        from quantem.assets.canonical_decode import CanonicalPlane

        for bad in (
            np.zeros((4, 4), dtype=np.uint16),
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8)[::2],
        ):
            with self.assertRaises((ValueError, TypeError)):
                CanonicalPlane(
                    array=bad, decoder_version="x", provenance="x", source_fingerprint="x"
                )
        with self.assertRaises(TypeError):
            CanonicalPlane(
                array=[[1, 2]], decoder_version="x", provenance="x", source_fingerprint="x"
            )

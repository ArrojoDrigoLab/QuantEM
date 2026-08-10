"""ImageJ/Fiji TIFF calibration: the unit lives in the ImageDescription.

Fiji's convention for a calibrated TIFF is *not* the baseline TIFF one. It
writes ``ResolutionUnit=NONE`` (1) and stores the real unit as a ``unit=<u>``
line inside an ImageJ-style ``ImageDescription`` block, with ``XResolution`` as
pixels-per-``<u>``. This is the most common calibrated-EM format in the wild,
and the pixel-size reader originally only saw the unit when ``tifffile``
happened to parse the block -- which requires the description to begin exactly
``ImageJ=``. A real pancreas TEM (vendor writer) begins its block with a bare
``ImageJ`` line instead, declares ``unit=micrometer`` and 140.78 px/um
(= 7.103 nm/px, agreeing with its own PixelScaleX vendor tag), and imported
with ``file_declared_pixel_size_nm: null``: the whole wrong-scale cascade the
app warns about, on a file that *did* declare its scale.

These tests build tiny TIFFs with the exact tag layouts in-test and verify both
the helper (``extract_tiff_metadata``) and the real upload path end to end.
They also pin the precedence rule documented on
``volume_readers._resolution_tag_nm_and_conflict``: a recognised ImageJ unit
wins over a disagreeing baseline ``ResolutionUnit`` tag, but the disagreement
is surfaced as ``pixel_size_caveat`` / ``file_declared_pixel_size_caveat``
rather than silently swallowed.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from quantem.assets.utils import extract_tiff_metadata
from quantem.assets.volume_readers import probe_volume_source

# An optional real-world sample, if one is pointed at by the environment. The
# vendor layout below is copied from such a file either way, so the format is
# pinned even when no sample is present.
_SAMPLE_ENV = os.environ.get("QUANTEM_IMAGEJ_SAMPLE", "")
USER_SAMPLE = Path(_SAMPLE_ENV) if _SAMPLE_ENV else Path("nonexistent-imagej-sample.tif")

#: Exact ImageDescription layout of the user's pancreas TEM: a bare ``ImageJ``
#: first line (no ``=version``, so ``tifffile`` does not parse it), the unit in
#: the ImageJ block, then vendor keys whose PixelScale agrees with XResolution.
VENDOR_DESCRIPTION = (
    "ImageJ\n"
    "min=6726.0\n"
    "max=8211.1\n"
    "unit=micrometer\n"
    "AppFive\n"
    "PixelScaleX=7.10311E-009m\n"
    "PixelScaleY=7.10311E-009m\n"
    "Magnification=2000 X"
)

#: XResolution of the user's file as the exact rational it stores:
#: 295244384 / 2097152 = 140.78 px/um -> 7.10310527637452 nm/px.
VENDOR_RESOLUTION = (295244384, 2097152)
VENDOR_NM_PER_PX = 2097152 / 295244384 * 1000.0


def _tiff_bytes(
    *,
    description: str | None = None,
    resolution=None,
    resolutionunit=None,
    shape=(16, 16),
    dtype=np.uint16,
    pages: int = 1,
) -> bytes:
    data = np.zeros((pages, *shape) if pages > 1 else shape, dtype)
    kwargs = {"photometric": "minisblack"}
    if description is not None:
        kwargs["description"] = description
    if resolution is not None:
        kwargs["resolution"] = resolution
    if resolutionunit is not None:
        kwargs["resolutionunit"] = resolutionunit
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, data, **kwargs)
    return buffer.getvalue()


def _fiji_description(unit: str | None) -> str:
    lines = ["ImageJ=1.54f", "images=1"]
    if unit is not None:
        lines.append(f"unit={unit}")
    return "\n".join(lines)


class ImageJCalibrationHelperTests(SimpleTestCase):
    """The exact tag layouts, through ``extract_tiff_metadata``."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)

    def _write(self, name: str, **kwargs) -> Path:
        path = self.tmp / name
        path.write_bytes(_tiff_bytes(**kwargs))
        return path

    def test_fiji_micron_description_calibrates(self):
        # Standard Fiji: ImageJ=<version> block, unit=micron, ResolutionUnit=NONE.
        p = self._write(
            "fiji_um.tif",
            description=_fiji_description("micron"),
            resolution=(140.78, 140.78),
            resolutionunit="NONE",
        )
        meta = extract_tiff_metadata(p)
        self.assertAlmostEqual(meta["pixel_size_nm"], 1000.0 / 140.78, delta=1e-3)
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_fiji_nanometer_description_calibrates(self):
        # 0.5 px/nm -> 2 nm/px.
        p = self._write(
            "fiji_nm.tif",
            description=_fiji_description("nm"),
            resolution=(0.5, 0.5),
            resolutionunit="NONE",
        )
        self.assertAlmostEqual(
            extract_tiff_metadata(p)["pixel_size_nm"], 2.0, delta=1e-6
        )

    def test_vendor_imagej_block_without_version_calibrates(self):
        """The user's exact layout: bare ``ImageJ`` first line.

        ``tifffile`` only parses descriptions beginning ``ImageJ=``, so this is
        the case the reader used to miss entirely -- the file that motivated
        this module importing as ``file_declared_pixel_size_nm: null``.
        """
        p = self._write(
            "vendor.tif",
            description=VENDOR_DESCRIPTION,
            resolution=(VENDOR_RESOLUTION, VENDOR_RESOLUTION),
            resolutionunit="NONE",
        )
        meta = extract_tiff_metadata(p)
        self.assertIsNotNone(meta["pixel_size_nm"])
        self.assertAlmostEqual(meta["pixel_size_nm"], VENDOR_NM_PER_PX, delta=1e-6)
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_resolutionunit_only_still_calibrates(self):
        # No ImageJ block at all: the baseline tag path is unchanged.
        p = self._write(
            "cm.tif", resolution=(2e6, 2e6), resolutionunit="CENTIMETER"
        )
        meta = extract_tiff_metadata(p)
        self.assertAlmostEqual(meta["pixel_size_nm"], 5.0, delta=1e-6)
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_agreeing_imagej_unit_and_resolutionunit_carry_no_caveat(self):
        # unit=cm in the ImageJ block and ResolutionUnit=CENTIMETER say the
        # same thing; agreement is not a conflict.
        p = self._write(
            "agree.tif",
            description=_fiji_description("cm"),
            resolution=(2e6, 2e6),
            resolutionunit="CENTIMETER",
        )
        meta = extract_tiff_metadata(p)
        self.assertAlmostEqual(meta["pixel_size_nm"], 5.0, delta=1e-6)
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_conflicting_imagej_unit_and_resolutionunit_keep_imagej_and_say_so(self):
        """The precedence rule, and the refusal to apply it silently.

        The ImageJ unit is the writer's deliberate calibration statement;
        ResolutionUnit=INCH here contradicts it by 25400x. The ImageJ value is
        used, and the disagreement travels with the asset as a caveat.
        """
        p = self._write(
            "conflict.tif",
            description=_fiji_description("micron"),
            resolution=(140.78, 140.78),
            resolutionunit="INCH",
        )
        meta = extract_tiff_metadata(p)
        self.assertAlmostEqual(meta["pixel_size_nm"], 1000.0 / 140.78, delta=1e-3)
        caveat = meta["pixel_size_caveat"]
        self.assertIsInstance(caveat, str)
        self.assertIn("micron", caveat)
        self.assertIn("inch", caveat.lower())
        self.assertIn("ImageJ", caveat)

    def test_imagej_block_with_no_unit_invents_nothing(self):
        # An ImageJ block without a unit line plus ResolutionUnit=NONE makes no
        # physical claim; reporting nothing beats inventing nanometres.
        p = self._write(
            "no_unit.tif",
            description=_fiji_description(None),
            resolution=(300, 300),
            resolutionunit="NONE",
        )
        meta = extract_tiff_metadata(p)
        self.assertIsNone(meta["pixel_size_nm"])
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_imagej_pixel_unit_is_not_a_length(self):
        # unit=pixel is ImageJ saying "uncalibrated". Treating it as
        # nanometres would fabricate a scale.
        p = self._write(
            "pixel_unit.tif",
            description=_fiji_description("pixel"),
            resolution=(300, 300),
            resolutionunit="NONE",
        )
        self.assertIsNone(extract_tiff_metadata(p)["pixel_size_nm"])


class ImageJCalibrationVolumeReaderTests(SimpleTestCase):
    """The volume reader shares the helper, so a Fiji stack calibrates too."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)

    def test_vendor_imagej_unit_reaches_voxel_size(self):
        p = self.tmp / "stack.tif"
        p.write_bytes(
            _tiff_bytes(
                description=VENDOR_DESCRIPTION,
                resolution=(VENDOR_RESOLUTION, VENDOR_RESOLUTION),
                resolutionunit="NONE",
                pages=3,
            )
        )
        meta = probe_volume_source(p)
        _z, y, x = meta.voxel_size_nm
        self.assertAlmostEqual(x, VENDOR_NM_PER_PX, delta=1e-6)
        self.assertAlmostEqual(y, VENDOR_NM_PER_PX, delta=1e-6)
        self.assertNotIn("calibration_conflict", meta.extra)

    def test_volume_conflict_lands_in_extra(self):
        p = self.tmp / "conflict_stack.tif"
        p.write_bytes(
            _tiff_bytes(
                description=_fiji_description("micron"),
                resolution=(140.78, 140.78),
                resolutionunit="INCH",
                pages=3,
            )
        )
        meta = probe_volume_source(p)
        _z, y, x = meta.voxel_size_nm
        self.assertAlmostEqual(x, 1000.0 / 140.78, delta=1e-3)
        self.assertAlmostEqual(y, 1000.0 / 140.78, delta=1e-3)
        conflict = meta.extra.get("calibration_conflict")
        self.assertIsInstance(conflict, str)
        self.assertIn("inch", conflict.lower())


class ImageJCalibrationUploadTests(TestCase):
    """End to end through the real upload path, not just the helper."""

    def setUp(self):
        self.client = APIClient()

    def _upload(self, content: bytes, name: str = "scan.tif", **extra):
        payload = {
            "file": SimpleUploadedFile(name, content, content_type="image/tiff"),
        }
        payload.update(extra)
        response = self.client.post("/api/assets/upload/", payload, format="multipart")
        self.assertEqual(response.status_code, 201, response.data)
        return response.data

    def test_fiji_style_upload_lands_calibrated_with_file_provenance(self):
        body = self._upload(
            _tiff_bytes(
                description=VENDOR_DESCRIPTION,
                resolution=(VENDOR_RESOLUTION, VENDOR_RESOLUTION),
                resolutionunit="NONE",
            )
        )
        self.assertAlmostEqual(body["pixel_size_nm"], VENDOR_NM_PER_PX, delta=1e-6)
        # Provenance "from file": the declared value exists and matches the
        # effective one, which is exactly how the library card decides.
        self.assertAlmostEqual(
            body["file_declared_pixel_size_nm"], VENDOR_NM_PER_PX, delta=1e-6
        )
        self.assertIsNone(body["file_declared_pixel_size_caveat"])

    def test_conflicting_upload_surfaces_the_caveat_on_the_asset(self):
        body = self._upload(
            _tiff_bytes(
                description=_fiji_description("micron"),
                resolution=(140.78, 140.78),
                resolutionunit="INCH",
            )
        )
        caveat = body["file_declared_pixel_size_caveat"]
        self.assertIsInstance(caveat, str)
        self.assertIn("inch", caveat.lower())

        # And it survives to the list payload the library actually renders.
        listed = self.client.get("/api/assets/")
        self.assertEqual(listed.status_code, 200)
        entries = {entry["id"]: entry for entry in listed.data}
        self.assertEqual(
            entries[body["id"]]["file_declared_pixel_size_caveat"], caveat
        )

    def test_a_typed_pixel_size_still_wins_over_the_imagej_block(self):
        body = self._upload(
            _tiff_bytes(
                description=VENDOR_DESCRIPTION,
                resolution=(VENDOR_RESOLUTION, VENDOR_RESOLUTION),
                resolutionunit="NONE",
            ),
            pixel_size_nm="4.2",
        )
        self.assertEqual(body["pixel_size_nm"], 4.2)
        # The file's own claim is still recorded, so the UI can say
        # "entered by hand" for the effective value.
        self.assertAlmostEqual(
            body["file_declared_pixel_size_nm"], VENDOR_NM_PER_PX, delta=1e-6
        )

    @unittest.skipUnless(USER_SAMPLE.exists(), "no sample at $QUANTEM_IMAGEJ_SAMPLE")
    def test_the_users_real_pancreas_tem_imports_calibrated(self):
        """The file that motivated this module, byte for byte."""
        body = self._upload(USER_SAMPLE.read_bytes(), name=USER_SAMPLE.name)
        self.assertIsNotNone(body["pixel_size_nm"])
        self.assertAlmostEqual(body["pixel_size_nm"], 7.10310527637452, delta=1e-6)
        self.assertAlmostEqual(
            body["file_declared_pixel_size_nm"], 7.10310527637452, delta=1e-6
        )
        # ResolutionUnit=NONE is the Fiji convention, not a conflict.
        self.assertIsNone(body["file_declared_pixel_size_caveat"])

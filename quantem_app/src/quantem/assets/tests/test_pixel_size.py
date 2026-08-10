"""Pixel-size extraction from TIFF resolution tags.

``Asset.pixel_size_nm`` gates per-organelle resampling and every calibrated
number the analysis suite produces, so a wrong value here is worse than a
missing one: it silently mislabels micrometres.

The bug these tests pin: ``_resolution_tag_nm`` originally read only the ImageJ
``unit`` string and ignored the baseline TIFF ``ResolutionUnit`` tag, so a plain
TIFF written in pixels-per-centimetre came back off by 1e7 -- reported as
5e-07 nm instead of 5 nm.
"""

from __future__ import annotations

import numpy as np
import tifffile
from django.test import SimpleTestCase

from quantem.assets.utils import extract_tiff_metadata


def _write(path, *, resolution, unit):
    kwargs = {"photometric": "minisblack"}
    if resolution is not None:
        kwargs["resolution"] = resolution
        kwargs["resolutionunit"] = unit
    tifffile.imwrite(str(path), np.zeros((16, 16), np.uint8), **kwargs)
    return path


class TiffPixelSizeTests(SimpleTestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)

    def test_centimetre_resolution(self):
        # 2e6 px/cm -> 5 nm/px
        p = _write(self.tmp / "cm.tif", resolution=(2e6, 2e6), unit="CENTIMETER")
        self.assertAlmostEqual(extract_tiff_metadata(p)["pixel_size_nm"], 5.0, places=6)

    def test_inch_resolution(self):
        # 2540 px/inch -> 10 um/px -> 10000 nm
        p = _write(self.tmp / "in.tif", resolution=(2540, 2540), unit="INCH")
        self.assertAlmostEqual(
            extract_tiff_metadata(p)["pixel_size_nm"], 10000.0, places=3
        )

    def test_unitless_resolution_is_refused(self):
        """ResolutionUnit=1 is an aspect ratio, not a scale. Inventing nanometres
        from it would be worse than reporting nothing."""
        p = _write(self.tmp / "none.tif", resolution=(300, 300), unit="NONE")
        self.assertIsNone(extract_tiff_metadata(p)["pixel_size_nm"])

    def test_absent_resolution_is_none(self):
        p = _write(self.tmp / "bare.tif", resolution=None, unit=None)
        self.assertIsNone(extract_tiff_metadata(p)["pixel_size_nm"])

    def test_geometry_is_still_extracted(self):
        p = _write(self.tmp / "geo.tif", resolution=(2e6, 2e6), unit="CENTIMETER")
        meta = extract_tiff_metadata(p)
        self.assertEqual((meta["width"], meta["height"]), (16, 16))
        self.assertEqual(meta["channels"], 1)
        self.assertEqual(meta["bit_depth"], 8)

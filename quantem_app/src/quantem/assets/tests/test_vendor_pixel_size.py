"""Vendor pixel size: the calibration this lab's own microscope writes.

A Zeiss/Fibics ATLAS export carries no ``XResolution`` worth the name. It sets
``ResolutionUnit = NONE`` (or writes ``XResolution = (1, 1)``, which says the
same nothing) and puts the real scan geometry in a private TIFF tag,
**51023 ``FibicsXML``**, as an XML block:

    <Scan><FOV_X units="um">254.315880514301</FOV_X>
          <Ux>0.00522874872557056</Ux><Uy>0</Uy>
          <Vx>0</Vx><Vy>-0.00522874872557056</Vy></Scan>

Until this module's reader existed, every one of those imported *uncalibrated*.
That is not cosmetic. ``Asset.pixel_size_nm`` decides the resample factor
between an asset and a model's canonical nm/px, so an uncalibrated import
either blocks inference or -- once someone types a number to unblock it --
silently rescales every micron in the analysis. MEASURED over a corpus of
Atlas exports from one microscope, of which most carry tag 51023 and every one
that does has ``ResolutionUnit = NONE``.

Two things the real files teach that the tag's name does not:

* **``<Uy>`` is zero.** ``U`` and ``V`` are the two scan *step vectors* in
  micrometres, not the x and y sizes. Reading ``<Ux>``/``<Uy>`` as x and y --
  the obvious reading, and the one the design sketch suggested -- gives ``0``
  for the y pixel size on every file in the corpus. The y size is ``|V|``.
* **The block can outlive the raster it describes.** It records its own
  ``<Width>``/``<Height>``, so a binned or resized copy that kept the tag can
  be caught and refused instead of being calibrated to a scale it no longer
  has.

The synthetic cases below reproduce the exact tag layout of the real files
(verified against them byte for byte at the field level), so the format stays
pinned on any machine. The ``AtlasCorpus*`` cases run against the real exports
when the lab volume is mounted and skip when it is not.

Nothing here decodes a pixel: every read is a TIFF header read, which is what
makes reading the calibration affordable inside the upload request on a 2 GB
file.
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
from quantem.assets.volume_readers import (
    PIXEL_SIZE_SOURCE_FIBICS,
    PIXEL_SIZE_SOURCE_IMAGEJ,
    PIXEL_SIZE_SOURCE_OME,
    PIXEL_SIZE_SOURCE_RESOLUTION_UNIT,
    probe_volume_source,
)

#: The step vector of ``25_0586-5nm-R4.tif``, in micrometres, to the last digit
#: the file stores. 0.00522874872557056 um = 5.22874872557056 nm/pixel -- the
#: "5.229 nm/px" this stage exists to recover.
ATLAS_UX_UM = 0.00522874872557056
ATLAS_NM = ATLAS_UX_UM * 1000.0


def fibics_xml(
    *,
    ux: float | None = ATLAS_UX_UM,
    uy: float | None = 0.0,
    vx: float | None = 0.0,
    vy: float | None = -ATLAS_UX_UM,
    width: int | None = 16,
    height: int | None = 16,
    fov_x: float | None = None,
    fov_y: float | None = None,
    scan_rot: float = 0.0,
) -> str:
    """A FibicsXML block with the element names and nesting of the real files.

    Copied from ``25_0586-5nm-R4.tif`` (ZEISS Atlas 5 Client v5.5.5.35), pruned
    to the elements a calibration reader looks at plus the surrounding
    structure that could confuse a naive match -- ``<OriginalWidth>`` next to
    ``<Width>``, ``units`` attributes on the field-of-view elements.
    """
    image = ""
    if width is not None and height is not None:
        image = (
            f"<Image><Width>{width}</Width><Height>{height}</Height>"
            f"<OriginalWidth>0</OriginalWidth><OriginalHeight>0</OriginalHeight>"
            f"<Cropped>false</Cropped></Image>"
        )
    scan_parts = [f'<ScanRot units="deg">{scan_rot}</ScanRot>']
    if fov_x is not None:
        scan_parts.insert(0, f'<FOV_X units="um">{fov_x}</FOV_X>')
    if fov_y is not None:
        scan_parts.insert(1, f'<FOV_Y units="um">{fov_y}</FOV_Y>')
    for name, value in (("Ux", ux), ("Uy", uy), ("Vx", vx), ("Vy", vy)):
        if value is not None:
            scan_parts.append(f"<{name}>{value!r}</{name}>")
    return (
        '<?xml version="1.0"?>\r\n'
        '<Fibics version="1.2" format="Image">'
        "<Application><Version>ZEISS Atlas 5 Client v5.5.5.35 PRE-RELEASE</Version>"
        "<SupportsTransparency>true</SupportsTransparency></Application>"
        f"{image}<Scan>{''.join(scan_parts)}</Scan>"
        '<Stage><X units="um">0</X></Stage>'
        "</Fibics>"
    )


def atlas_tiff_bytes(
    *,
    xml: str | None = None,
    description: str | None = None,
    resolution=None,
    resolutionunit="NONE",
    shape=(16, 16),
    pages: int = 1,
) -> bytes:
    """A TIFF laid out like an Atlas export: tag 51023 and no real resolution.

    ``resolutionunit="NONE"`` and no ``XResolution`` is the exact combination
    every Fibics file in the corpus has, and the combination that made them
    import uncalibrated.
    """
    data = np.zeros((pages, *shape) if pages > 1 else shape, np.uint8)
    kwargs: dict = {"photometric": "minisblack"}
    if description is not None:
        kwargs["description"] = description
    if resolution is not None:
        kwargs["resolution"] = resolution
    if resolutionunit is not None:
        kwargs["resolutionunit"] = resolutionunit
    if xml is not None:
        kwargs["extratags"] = [(51023, "s", 0, xml, True)]
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, data, **kwargs)
    return buffer.getvalue()


def _fiji_description(unit: str | None) -> str:
    lines = ["ImageJ=1.54f", "images=1"]
    if unit is not None:
        lines.append(f"unit={unit}")
    return "\n".join(lines)


class _TempTiffCase(SimpleTestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)

    def write(self, name: str, **kwargs) -> Path:
        path = self.tmp / name
        path.write_bytes(atlas_tiff_bytes(**kwargs))
        return path


class FibicsTagReaderTests(_TempTiffCase):
    """Tag 51023 read through the real 2D import helper."""

    def test_atlas_export_imports_calibrated_and_names_the_tag(self):
        """The headline case. Without the reader this returns ``None``."""
        path = self.write("atlas.tif", xml=fibics_xml())
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], ATLAS_NM, delta=1e-9)
        self.assertEqual(meta["pixel_size_source"], PIXEL_SIZE_SOURCE_FIBICS)
        self.assertIn("51023", meta["pixel_size_source"])
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_a_bare_xresolution_of_one_does_not_beat_the_vendor_tag(self):
        """Half the corpus writes ``XResolution = (1, 1)`` beside the tag.

        With ``ResolutionUnit = NONE`` that is not a scale, so it must not
        suppress or contradict the vendor record.
        """
        path = self.write(
            "atlas_res1.tif", xml=fibics_xml(), resolution=(1, 1), resolutionunit="NONE"
        )
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], ATLAS_NM, delta=1e-9)
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_the_y_size_comes_from_v_not_from_uy(self):
        """``<Uy>`` is 0 in every real file; the y step is ``<Vy>``.

        Read as "Ux is x and Uy is y" this file calibrates y at 0 nm/pixel --
        which is why this test asserts on the *volume* reader, the one path
        that reads a Y size at all.
        """
        path = self.tmp / "atlas_y.tif"
        path.write_bytes(atlas_tiff_bytes(xml=fibics_xml(), pages=3))
        _z, y, x = probe_volume_source(path).voxel_size_nm
        self.assertAlmostEqual(x, ATLAS_NM, delta=1e-9)
        self.assertAlmostEqual(y, ATLAS_NM, delta=1e-9)

    def test_a_rotated_scan_uses_the_vector_length(self):
        """With ``<ScanRot>`` non-zero the step vector's components mix.

        Ux alone would report 0.866 of the true size here; |U| is exact.
        """
        import math

        step = ATLAS_UX_UM
        path = self.write(
            "rotated.tif",
            xml=fibics_xml(
                ux=step * math.cos(math.radians(30)),
                uy=step * math.sin(math.radians(30)),
                vx=step * math.sin(math.radians(30)),
                vy=-step * math.cos(math.radians(30)),
                scan_rot=30.0,
            ),
        )
        self.assertAlmostEqual(extract_tiff_metadata(path)["pixel_size_nm"], ATLAS_NM, delta=1e-9)

    def test_field_of_view_is_the_fallback_when_the_step_vectors_are_absent(self):
        """``FOV_X / Width`` equals ``|U|`` to 15 digits in every corpus file.

        Keeping it as a fallback means a writer that emits the field of view
        but not the step vectors still calibrates.
        """
        path = self.write(
            "fov.tif",
            xml=fibics_xml(
                ux=None,
                uy=None,
                vx=None,
                vy=None,
                width=16,
                height=16,
                fov_x=ATLAS_UX_UM * 16,
                fov_y=ATLAS_UX_UM * 16,
            ),
        )
        self.assertAlmostEqual(extract_tiff_metadata(path)["pixel_size_nm"], ATLAS_NM, delta=1e-9)

    def test_a_file_with_no_calibration_at_all_stays_uncalibrated(self):
        """The honest-silence case: no vendor tag, no unit, no scale.

        ``processed_25_0586-5nm-R4.tif`` in the corpus is exactly this -- an
        11 GB re-export whose tag 51023 was dropped -- and it must keep
        importing as "not calibrated" rather than acquiring a guess.
        """
        path = self.write("silent.tif", xml=None, resolutionunit="NONE")
        meta = extract_tiff_metadata(path)
        self.assertIsNone(meta["pixel_size_nm"])
        self.assertIsNone(meta["pixel_size_source"])
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_a_zero_step_vector_is_not_a_pixel_size(self):
        """A block that says the scan step is 0 declares nothing usable."""
        path = self.write("zero.tif", xml=fibics_xml(ux=0.0, uy=0.0, vx=0.0, vy=0.0))
        meta = extract_tiff_metadata(path)
        self.assertIsNone(meta["pixel_size_nm"])
        self.assertIsNone(meta["pixel_size_source"])

    def test_a_malformed_block_is_ignored_rather_than_failing_the_import(self):
        """Garbage in a private tag must not cost the user their upload."""
        path = self.write(
            "broken.tif",
            xml='<?xml version="1.0"?><Fibics><Scan><Ux>not-a-number</Ux></Scan>',
        )
        meta = extract_tiff_metadata(path)
        self.assertEqual(meta["width"], 16)
        self.assertIsNone(meta["pixel_size_nm"])

    def test_a_tag_that_is_not_a_fibics_block_is_ignored(self):
        path = self.write("foreign.tif", xml="<SomeOtherVendor><Ux>0.5</Ux>")
        self.assertIsNone(extract_tiff_metadata(path)["pixel_size_nm"])


class FibicsStaleRasterTests(_TempTiffCase):
    """The block records its own geometry, so a resized copy can be caught."""

    def test_a_block_describing_a_different_raster_is_refused_and_explained(self):
        path = self.write(
            "binned.tif",
            shape=(16, 16),
            xml=fibics_xml(width=32, height=32),
        )
        meta = extract_tiff_metadata(path)
        self.assertIsNone(meta["pixel_size_nm"])
        self.assertIsNone(meta["pixel_size_source"])
        caveat = meta["pixel_size_caveat"]
        self.assertIsInstance(caveat, str)
        self.assertIn("51023", caveat)
        self.assertIn("32 x 32", caveat)
        self.assertIn("16 x 16", caveat)

    def test_a_block_with_no_declared_geometry_is_still_trusted(self):
        """Older writers omit ``<Image>``; absence is not a mismatch."""
        path = self.write("nogeom.tif", xml=fibics_xml(width=None, height=None))
        self.assertAlmostEqual(extract_tiff_metadata(path)["pixel_size_nm"], ATLAS_NM, delta=1e-9)


class CalibrationPrecedenceTests(_TempTiffCase):
    """What happens when a file declares its scale twice and disagrees."""

    def test_an_imagej_unit_still_wins_over_the_vendor_tag(self):
        """Precedence is unchanged: a ``unit=`` line is a statement about *this*
        raster, made after acquisition, and a Fiji re-save is exactly the
        operation that can resize an image while leaving the vendor block
        behind. The vendor value is not discarded silently."""
        path = self.write(
            "both.tif",
            xml=fibics_xml(),
            description=_fiji_description("nm"),
            resolution=(0.5, 0.5),  # 2 nm/px, vs the tag's 5.229
            resolutionunit="NONE",
        )
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], 2.0, delta=1e-9)
        self.assertEqual(meta["pixel_size_source"], PIXEL_SIZE_SOURCE_IMAGEJ)
        caveat = meta["pixel_size_caveat"]
        self.assertIsInstance(caveat, str)
        self.assertIn("51023", caveat)
        self.assertIn("ImageJ", caveat)
        self.assertIn("2 nm/pixel", caveat)
        self.assertIn("5.229 nm/pixel", caveat)

    def test_an_agreeing_imagej_unit_and_vendor_tag_carry_no_caveat(self):
        path = self.write(
            "agree.tif",
            xml=fibics_xml(),
            description=_fiji_description("nm"),
            resolution=(1.0 / ATLAS_NM, 1.0 / ATLAS_NM),
            resolutionunit="NONE",
        )
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], ATLAS_NM, delta=1e-6)
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_the_vendor_tag_beats_a_disagreeing_baseline_tag_and_says_so(self):
        """``ResolutionUnit = INCH`` at 72 dpi is what an image editor leaves
        behind, not a microscope's claim. Two real corpus files carry exactly
        that after a Photoshop round trip. The microscope wins, out loud."""
        path = self.write(
            "photoshop.tif",
            xml=fibics_xml(),
            resolution=(72, 72),
            resolutionunit="INCH",
        )
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], ATLAS_NM, delta=1e-9)
        self.assertEqual(meta["pixel_size_source"], PIXEL_SIZE_SOURCE_FIBICS)
        caveat = meta["pixel_size_caveat"]
        self.assertIsInstance(caveat, str)
        self.assertIn("51023", caveat)
        self.assertIn("ResolutionUnit", caveat)

    def test_an_agreeing_baseline_tag_carries_no_caveat(self):
        # 2e6 px/cm = 5 nm/px; make the vendor block say the same thing.
        path = self.write(
            "agree_cm.tif",
            xml=fibics_xml(ux=0.005, uy=0.0, vx=0.0, vy=-0.005),
            resolution=(2e6, 2e6),
            resolutionunit="CENTIMETER",
        )
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], 5.0, delta=1e-9)
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_ome_physical_size_still_outranks_everything(self):
        path = self.tmp / "ome.tif"
        with tifffile.TiffWriter(str(path), ome=True) as writer:
            writer.write(
                np.zeros((16, 16), np.uint8),
                photometric="minisblack",
                resolution=(1e5, 1e5),
                resolutionunit="CENTIMETER",
                extratags=[(51023, "s", 0, fibics_xml(), True)],
                metadata={"PhysicalSizeX": 0.003, "PhysicalSizeXUnit": "\u00b5m"},
            )
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], 3.0, delta=1e-9)
        self.assertEqual(meta["pixel_size_source"], PIXEL_SIZE_SOURCE_OME)


class UnchangedCalibrationTests(_TempTiffCase):
    """No file without tag 51023 changes its value, only gains a source label.

    These duplicate the numbers ``test_pixel_size.py`` and
    ``test_imagej_calibration.py`` already pin. They are here because "no
    existing calibrated file changes its value" is this stage's acceptance
    criterion, and a criterion nobody asserts is a hope.
    """

    def test_centimetre_resolution_is_unchanged(self):
        path = self.write("cm.tif", resolution=(2e6, 2e6), resolutionunit="CENTIMETER")
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], 5.0, delta=1e-9)
        self.assertEqual(meta["pixel_size_source"], PIXEL_SIZE_SOURCE_RESOLUTION_UNIT)

    def test_inch_resolution_is_unchanged(self):
        path = self.write("inch.tif", resolution=(2540, 2540), resolutionunit="INCH")
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], 10000.0, delta=1e-6)
        self.assertEqual(meta["pixel_size_source"], PIXEL_SIZE_SOURCE_RESOLUTION_UNIT)

    def test_a_unitless_resolution_is_still_refused(self):
        path = self.write("none.tif", resolution=(300, 300), resolutionunit="NONE")
        meta = extract_tiff_metadata(path)
        self.assertIsNone(meta["pixel_size_nm"])
        self.assertIsNone(meta["pixel_size_source"])

    def test_the_imagej_unit_path_is_unchanged(self):
        path = self.write(
            "fiji.tif",
            description=_fiji_description("micron"),
            resolution=(140.78, 140.78),
            resolutionunit="NONE",
        )
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], 1000.0 / 140.78, delta=1e-6)
        self.assertEqual(meta["pixel_size_source"], PIXEL_SIZE_SOURCE_IMAGEJ)
        self.assertIsNone(meta["pixel_size_caveat"])

    def test_the_imagej_versus_resolutionunit_conflict_is_unchanged(self):
        path = self.write(
            "conflict.tif",
            description=_fiji_description("micron"),
            resolution=(140.78, 140.78),
            resolutionunit="INCH",
        )
        meta = extract_tiff_metadata(path)
        self.assertAlmostEqual(meta["pixel_size_nm"], 1000.0 / 140.78, delta=1e-6)
        caveat = meta["pixel_size_caveat"]
        self.assertIn("micron", caveat)
        self.assertIn("inch", caveat.lower())
        self.assertNotIn("51023", caveat)


class VendorPixelSizeVolumeTests(_TempTiffCase):
    """A stack of Atlas pages calibrates the same way a single page does.

    The 2D and 3D readers share one composed helper precisely so that the same
    file cannot come back with two different scales depending on which door it
    was imported through.
    """

    def test_a_vendor_stack_calibrates_and_records_the_source(self):
        path = self.tmp / "stack.tif"
        path.write_bytes(atlas_tiff_bytes(xml=fibics_xml(), pages=4))
        meta = probe_volume_source(path)
        _z, y, x = meta.voxel_size_nm
        self.assertAlmostEqual(x, ATLAS_NM, delta=1e-9)
        self.assertAlmostEqual(y, ATLAS_NM, delta=1e-9)
        self.assertEqual(meta.extra.get("calibration_source"), PIXEL_SIZE_SOURCE_FIBICS)
        self.assertNotIn("calibration_conflict", meta.extra)

    def test_a_stale_vendor_block_reaches_extra_as_a_conflict(self):
        path = self.tmp / "stale_stack.tif"
        path.write_bytes(atlas_tiff_bytes(xml=fibics_xml(width=999, height=999), pages=4))
        meta = probe_volume_source(path)
        _z, y, x = meta.voxel_size_nm
        self.assertIsNone(x)
        self.assertIsNone(y)
        self.assertIn("51023", meta.extra.get("calibration_conflict", ""))
        self.assertNotIn("calibration_source", meta.extra)


class VendorPixelSizeUploadTests(TestCase):
    """End to end through the real upload path, not just the helper."""

    def setUp(self):
        self.client = APIClient()

    def _upload(self, content: bytes, name: str = "atlas.tif", **extra):
        payload = {"file": SimpleUploadedFile(name, content, content_type="image/tiff")}
        payload.update(extra)
        response = self.client.post("/api/assets/upload/", payload, format="multipart")
        self.assertEqual(response.status_code, 201, response.data)
        return response.data

    def test_an_atlas_upload_lands_calibrated_with_the_tag_named(self):
        body = self._upload(atlas_tiff_bytes(xml=fibics_xml()))
        self.assertAlmostEqual(body["pixel_size_nm"], ATLAS_NM, delta=1e-9)
        self.assertAlmostEqual(body["file_declared_pixel_size_nm"], ATLAS_NM, delta=1e-9)
        self.assertEqual(body["file_declared_pixel_size_source"], PIXEL_SIZE_SOURCE_FIBICS)
        self.assertIsNone(body["file_declared_pixel_size_caveat"])

        # And the provenance survives to the list payload the library renders,
        # which is the only place a user sees where a number came from.
        listed = self.client.get("/api/assets/")
        self.assertEqual(listed.status_code, 200)
        entries = {entry["id"]: entry for entry in listed.data}
        self.assertEqual(
            entries[body["id"]]["file_declared_pixel_size_source"],
            PIXEL_SIZE_SOURCE_FIBICS,
        )

    def test_a_hand_typed_pixel_size_still_wins_but_the_tag_is_recorded(self):
        """The distinction the provenance field exists to make.

        The effective value is the typed one; the file's own claim and the tag
        that made it are still on the payload, so the UI can say which is which.
        """
        body = self._upload(atlas_tiff_bytes(xml=fibics_xml()), pixel_size_nm="4.2")
        self.assertEqual(body["pixel_size_nm"], 4.2)
        self.assertAlmostEqual(body["file_declared_pixel_size_nm"], ATLAS_NM, delta=1e-9)
        self.assertEqual(body["file_declared_pixel_size_source"], PIXEL_SIZE_SOURCE_FIBICS)

    def test_an_uncalibrated_upload_says_so_rather_than_guessing(self):
        body = self._upload(atlas_tiff_bytes(xml=None), name="silent.tif")
        self.assertIsNone(body["pixel_size_nm"])
        self.assertIsNone(body["file_declared_pixel_size_nm"])
        self.assertIsNone(body["file_declared_pixel_size_source"])
        self.assertIsNone(body["file_declared_pixel_size_caveat"])

    def test_a_stale_vendor_block_reaches_the_user_as_a_caveat(self):
        body = self._upload(
            atlas_tiff_bytes(xml=fibics_xml(width=4096, height=4096)),
            name="binned.tif",
        )
        self.assertIsNone(body["pixel_size_nm"])
        caveat = body["file_declared_pixel_size_caveat"]
        self.assertIsInstance(caveat, str)
        self.assertIn("51023", caveat)


# --------------------------------------------------------------------------- #
# The real exports
# --------------------------------------------------------------------------- #
# Three Zeiss Atlas files one microscope wrote, named individually because
# their pixel sizes differ and each one is its own regression.
#
# The expected values are transcribed from each file's own ``<Ux>`` (recorded
# here as constants, so the assertion is not the reader checking itself) and
# independently cross-checked below against ``FOV_X / Width``, which the reader
# does not consult when a step vector is present.
#
# The files themselves are microscope exports, not part of this distribution.
# Point the environment variable below at a directory holding them to run these
# three tests. There is deliberately **no default**: a default would ship one
# laboratory's mount point to everyone who downloads the source distribution.
# Unset, the corpus is simply absent and the class skips, exactly as it does on
# a machine where the volume is not mounted -- and the synthetic cases above
# still pin the format either way.
_CORPUS_ROOT_ENV_VAR = "QUANTEM_ATLAS_SAMPLE_ROOT"
_corpus_setting = os.environ.get(_CORPUS_ROOT_ENV_VAR, "")
_CORPUS_ROOT: Path | None = Path(_corpus_setting) if _corpus_setting else None

ATLAS_SAMPLES: list[tuple[str, float, int, int]] = [
    (r"20250516-ImmunoEM-exports\25_0586-5nm-R4.tif", 5.22874872557056, 48638, 56890),
    (
        r"20250516-ImmunoEM-exports\73_6hrfast_M1\25-0073_5nm_Region12_bsd.tif",
        5.151788097986,
        64640,
        32629,
    ),
    (r"human_liver\24-0616-5nm-Region4.tif", 5.00140686035156, 27396, 38329),
]

#: An 11 GB re-export of the first sample whose tag 51023 did not survive. It
#: is the control: the reader must not invent a scale for it.
UNCALIBRATED_SAMPLE = r"20250516-ImmunoEM-exports\processed_25_0586-5nm-R4.tif"


def _sample(relative: str) -> Path:
    """One corpus file. Only reached once :func:`_corpus_available` is true."""
    if _CORPUS_ROOT is None:
        raise RuntimeError(f"{_CORPUS_ROOT_ENV_VAR} is not set")
    return _CORPUS_ROOT / relative


def _corpus_available() -> bool:
    if _CORPUS_ROOT is None:
        return False
    return all(_sample(rel).exists() for rel, _nm, _w, _h in ATLAS_SAMPLES)


def _uncalibrated_control_available() -> bool:
    return _corpus_available() and _sample(UNCALIBRATED_SAMPLE).exists()


@unittest.skipUnless(
    _corpus_available(),
    f"Atlas corpus not available; set {_CORPUS_ROOT_ENV_VAR} to a directory holding it",
)
class AtlasCorpusTests(SimpleTestCase):
    """The real 0.8-2 GB exports, read from the lab volume. Headers only."""

    def test_every_real_atlas_export_imports_calibrated_from_tag_51023(self):
        for rel, expected_nm, width, height in ATLAS_SAMPLES:
            with self.subTest(file=rel):
                meta = extract_tiff_metadata(_sample(rel))
                self.assertEqual((meta["width"], meta["height"]), (width, height))
                self.assertIsNotNone(
                    meta["pixel_size_nm"],
                    "a calibrated Atlas export imported uncalibrated",
                )
                self.assertAlmostEqual(meta["pixel_size_nm"], expected_nm, delta=1e-9)
                self.assertEqual(meta["pixel_size_source"], PIXEL_SIZE_SOURCE_FIBICS)
                self.assertIsNone(meta["pixel_size_caveat"])

    def test_the_step_vector_agrees_with_the_field_of_view_in_every_file(self):
        """An independent read of the same block, through a path the reader
        does not take when ``<Ux>`` is present. If these two ever disagree the
        vendor block is not saying what this module thinks it says."""
        import re

        for rel, expected_nm, _w, _h in ATLAS_SAMPLES:
            with self.subTest(file=rel):
                with tifffile.TiffFile(str(_sample(rel))) as tif:
                    xml = tif.pages[0].tags.get(51023).value
                fov = float(re.search(r"<FOV_X[^>]*>([^<]+)</FOV_X>", xml).group(1))
                cols = float(re.search(r"<Width>([^<]+)</Width>", xml).group(1))
                self.assertAlmostEqual(fov / cols * 1000.0, expected_nm, delta=1e-6)

    @unittest.skipUnless(_uncalibrated_control_available(), "no uncalibrated control in the corpus")
    def test_the_re_export_that_lost_its_tag_stays_uncalibrated(self):
        meta = extract_tiff_metadata(_sample(UNCALIBRATED_SAMPLE))
        self.assertIsNone(meta["pixel_size_nm"])
        self.assertIsNone(meta["pixel_size_source"])
        self.assertIsNone(meta["pixel_size_caveat"])

"""ROI image I/O must never hold SQLite's single writer lock."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.db import connection
from django.test import TransactionTestCase

from quantem.assets import utils
from quantem.assets.models import ImageROI


class RoiTransactionScopeTests(TransactionTestCase):
    def test_file_crop_precedes_the_short_database_transaction(self):
        """A slow source decode leaves unrelated database writers available."""

        states: list[tuple[str, bool]] = []
        roi_root = Path(utils.ROIS_DIR).parent / "test-roi-transaction-scope"
        self.addCleanup(shutil.rmtree, roi_root, ignore_errors=True)
        real_create = ImageROI.objects.create

        def save_roi(_image, roi_path, **_bounds):
            states.append(("file", connection.in_atomic_block))
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_path.write_bytes(b"roi")
            return True

        def create_row(**kwargs):
            states.append(("database", connection.in_atomic_block))
            return real_create(**kwargs)

        image = SimpleNamespace(asset=None, id="image-1", display_name="Large image")
        with (
            patch.object(utils, "ROIS_DIR", roi_root),
            patch.object(utils, "get_file_absolute_path", return_value=roi_root / "source.tif"),
            patch.object(utils, "_save_roi_png_from_ngff", side_effect=save_roi),
            patch.object(ImageROI.objects, "create", side_effect=create_row),
        ):
            roi = utils.create_roi_image_from_image(
                image,
                x=0,
                y=0,
                width=32,
                height=32,
            )

        self.assertEqual(
            states,
            [("file", False), ("database", True)],
            "filesystem work acquired SQLite's writer transaction",
        )
        self.assertTrue((roi_root / f"{roi.id}.png").exists())

    def test_failed_database_commit_removes_the_unreferenced_png(self):
        roi_root = Path(utils.ROIS_DIR).parent / "test-roi-failed-commit"
        self.addCleanup(shutil.rmtree, roi_root, ignore_errors=True)
        written_path: list[Path] = []

        def save_roi(_image, roi_path, **_bounds):
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_path.write_bytes(b"roi")
            written_path.append(roi_path)
            return True

        image = SimpleNamespace(asset=None, id="image-1", display_name="Large image")
        with (
            patch.object(utils, "ROIS_DIR", roi_root),
            patch.object(utils, "get_file_absolute_path", return_value=roi_root / "source.tif"),
            patch.object(utils, "_save_roi_png_from_ngff", side_effect=save_roi),
            patch.object(ImageROI.objects, "create", side_effect=RuntimeError("write failed")),
            self.assertRaisesRegex(RuntimeError, "write failed"),
        ):
            utils.create_roi_image_from_image(
                image,
                x=0,
                y=0,
                width=32,
                height=32,
            )

        self.assertEqual(len(written_path), 1)
        self.assertFalse(written_path[0].exists())

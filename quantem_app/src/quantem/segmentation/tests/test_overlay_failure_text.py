"""The overlay's failure reason, as the person labeling actually reads it.

Findings V5 and V8 of the wave-0b verification. Both are about the same thing:
the overlay build is the one background job whose failure the user is expected
to *fix*, and it can only be fixed if the message names what went wrong and
where, in a form that can be pasted into a file browser.

V5: ``last_error=str(exc)``. ``OSError.__str__`` renders the filename with
``repr()``, so the path reached the screen with every separator doubled and
wrapped in quotes.
V8: ``_remove_tree`` raised "Failed to remove overlay path: <path>" and threw
away the ``OSError`` that said *why* -- almost always another program holding a
file open, which the user can neither guess nor see.

Every exception used below is raised by a real filesystem operation rather than
built by hand: the doubling in V5 is a property of how the OS populates
``strerror``/``filename``, and a fabricated ``OSError("...")`` would not have
reproduced the defect at all.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase
from shapely.geometry import Polygon

from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationOverlayState,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff import OverlayStoreError
from quantem.segmentation.overlay_ngff.failure_text import (
    describe_failure,
    describe_os_error,
)
from quantem.segmentation.overlay_ngff.mutations import run_overlay_rebuild_job
from quantem.segmentation.overlay_ngff.paths import _remove_tree, get_overlay_root
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

on_windows = os.name == "nt"


def _collision_error(directory: Path) -> OSError:
    """The exception a stray file in the staging slot produces, for real."""
    target = directory / "staging"
    target.mkdir()
    try:
        target.mkdir()
    except OSError as exc:
        return exc
    raise AssertionError("mkdir over an existing directory did not fail")


class _TempDirMixin:
    def temp_dir(self) -> Path:
        # `TMPDIR`/`TEMP`/`TMP` are pointed at repo scratch by the suite's
        # `_pytest_env` plugin, so this never lands on the system drive.
        return Path(self.enterContext(tempfile.TemporaryDirectory(ignore_cleanup_errors=True)))


class DescribeFailureTests(_TempDirMixin, TestCase):
    def test_the_reason_and_the_real_path_and_nothing_else(self):
        exc = _collision_error(self.temp_dir())

        described = describe_failure(exc)

        self.assertEqual(described, f"{exc.strerror}: {exc.filename}")
        # The `repr()` artefacts that made V5 unpasteable.
        self.assertNotIn("'", described)
        # The numeric code is the one part no user can act on.
        self.assertNotIn("WinError", described)
        self.assertNotIn("Errno", described)

    @unittest.skipUnless(on_windows, "separator doubling is a Windows path artefact")
    def test_a_windows_path_keeps_single_separators(self):
        exc = _collision_error(self.temp_dir())

        # State the defect as an assertion so the test explains itself.
        self.assertIn("\\\\", str(exc))
        self.assertNotIn("\\\\", describe_failure(exc))

    def test_the_described_path_is_the_path_on_disk(self):
        """Pasteable means pasteable: it round-trips to the real directory."""
        exc = _collision_error(self.temp_dir())

        recovered = describe_failure(exc).split(": ", 1)[1]

        self.assertTrue(Path(recovered).is_dir())
        self.assertEqual(Path(recovered), Path(str(exc.filename)))

    def test_a_two_path_operation_names_both(self):
        base = self.temp_dir()
        source = base / "source"
        source.write_text("x", encoding="utf-8")
        try:
            os.rename(source, base / "missing" / "target")
        except OSError as exc:
            described = describe_os_error(exc)
        else:  # pragma: no cover - the rename must fail
            raise AssertionError("rename into a missing directory did not fail")

        self.assertIn(str(source), described)
        self.assertIn("target", described)
        self.assertNotIn("'", described)

    def test_our_own_exceptions_keep_their_english(self):
        self.assertEqual(
            describe_failure(OverlayStoreError("The label store is malformed.")),
            "The label store is malformed.",
        )

    def test_an_exception_with_no_message_still_says_something(self):
        self.assertEqual(describe_failure(KeyError()), "KeyError")


class RemoveTreeReasonTests(_TempDirMixin, TestCase):
    """V8: say what went wrong, not only where."""

    def _assert_names_reason_and_blocker(self, message: str, blocker: Path) -> None:
        self.assertIn("overlay folder", message)
        # The half V8 said was thrown away: a description of the obstacle, not
        # just the directory it is in.
        reason = message.split("overlay folder. ", 1)[1]
        self.assertTrue(reason.endswith(str(blocker)), reason)
        self.assertGreater(len(reason) - len(str(blocker)), 8, reason)
        self.assertNotIn("'", message)

    @unittest.skipUnless(on_windows, "only Windows refuses to unlink an open file")
    def test_a_file_another_program_holds_open_is_named_with_its_reason(self):
        """The field scenario, with a real handle rather than a simulation."""
        doomed = self.temp_dir() / "labels.zarr"
        doomed.mkdir()
        chunk = doomed / "0"
        chunk.write_bytes(b"chunk")

        with chunk.open("rb"):
            with self.assertRaises(OverlayStoreError) as caught:
                _remove_tree(doomed)

        message = str(caught.exception)
        self.assertIn("used by another process", message)
        self._assert_names_reason_and_blocker(message, chunk)
        self.assertNotIn("\\\\", message)

    @unittest.skipIf(on_windows, "chmod does not block deletion on Windows")
    def test_a_directory_we_may_not_write_to_is_named_with_its_reason(self):
        doomed = self.temp_dir() / "labels.zarr"
        doomed.mkdir()
        chunk = doomed / "0"
        chunk.write_bytes(b"chunk")
        doomed.chmod(0o500)
        self.addCleanup(doomed.chmod, 0o700)

        with self.assertRaises(OverlayStoreError) as caught:
            _remove_tree(doomed)

        self._assert_names_reason_and_blocker(str(caught.exception), chunk)

    def test_a_directory_that_does_go_away_raises_nothing(self):
        doomed = self.temp_dir() / "labels.zarr"
        doomed.mkdir()
        (doomed / "0").write_bytes(b"chunk")

        _remove_tree(doomed)

        self.assertFalse(doomed.exists())


class OverlayStateFailureTextTests(TestCase):
    """V5 end to end: what the manifest carries after a real failed build."""

    def setUp(self):
        self.image = create_image_from_test_tiff("Overlay Failure Text Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10)))
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
        )

    def _state(self) -> SegmentationOverlayState:
        return SegmentationOverlayState.objects.get(
            segmentation=self.segmentation, candidate_source_model=""
        )

    def test_a_stray_file_in_the_staging_slot_is_recorded_as_a_real_path(self):
        """The exact fault the verifier reproduced, through the real job."""
        root = get_overlay_root(str(self.segmentation.id))
        root.mkdir(parents=True, exist_ok=True)
        # A file where the staging *directory* belongs. Nothing exotic: a
        # half-finished copy, a synchronisation client's placeholder.
        (root / "staging").write_bytes(b"x")

        with self.assertRaises(OSError):
            run_overlay_rebuild_job(self.segmentation, mode="full")

        state = self._state()
        self.assertEqual(state.status, SegmentationOverlayState.STATUS_FAILED)
        self.assertIn(str(root / "staging"), state.last_error)
        self.assertNotIn("'", state.last_error)
        if on_windows:
            self.assertNotIn("\\\\", state.last_error)
            self.assertNotIn("WinError", state.last_error)

    def test_a_failure_of_our_own_making_is_recorded_unchanged(self):
        with patch(
            "quantem.segmentation.overlay_ngff.mutations.rebuild_overlay_full",
            side_effect=OverlayStoreError("The label store is malformed."),
        ):
            with self.assertRaises(OverlayStoreError):
                run_overlay_rebuild_job(self.segmentation, mode="full")

        self.assertEqual(self._state().last_error, "The label store is malformed.")

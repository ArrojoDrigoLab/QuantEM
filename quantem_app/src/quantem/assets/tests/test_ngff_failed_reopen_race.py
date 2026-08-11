"""A failed import can never re-open itself, on any container.

Round 3 guarded ``assets/views.py`` and was falsified in two runs by one
ordinary polling client: ``tasks.ensure_ngff_for_asset_task`` had no guard,
the job reaching it had been enqueued a few seconds earlier by a *legitimate*
202 while the asset was still ``ENCODING``, and
``ngff.regenerate_ngff_for_image`` refused only sources whose suffix was not
``.png`` -- so a staged 16-bit PNG upload was treated as the canonical PNG,
rebuilt through ``Image.open(...).convert("L")``, and republished. **725
contradiction samples in 913**, and the pyramid the user could then open and
segment was uniformly 255: 16 777 216 of 16 777 216 pixels wrong, with the
correct one sitting in ``withdrawn/``.

Their test for exactly this vector passed, because its ``_stage_upload``
hard-coded ``.tif``. So this file runs **every container** and, in each round,
calls the dangerous operation directly rather than hoping a poll happens to
race it -- which is strictly stronger than the sustained polling that found the
bug.

What makes it impossible now is not a fourth guard:

* the failure and the fence move in **one transaction**
  (``record_attempt_failure``), so a build already running under the old token
  cannot publish afterwards, and one that starts later is refused a ticket;
* the terminal message goes to ``failure_detail``, which the job layer's retry
  note does not write, so a storage-lease conflict can no longer replace the
  real cause on the 409 body.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import tifffile
from django.test import Client, SimpleTestCase, TestCase
from PIL import Image

from quantem.assets import tasks as tasks_module
from quantem.assets.canonical_decode import decode_canonical_plane
from quantem.assets.models import Asset
from quantem.assets.ngff import PyramidBuildRefused, build_pyramid
from quantem.assets.pyramid_authority import (
    Intent,
    PublishedPyramid,
    Reason,
    Unavailable,
    begin_attempt,
    failure_detail,
    publish,
    record_attempt_failure,
    request_build,
    request_lazy_build,
    resolve_pyramid,
)
from quantem.assets.serializers import serialize_asset_entry
from quantem.assets.tasks import ensure_ngff_for_asset_task, prepare_asset_renditions
from quantem.core.config import DATA_DIR
from quantem.jobs.constants import JOB_TYPE_ENSURE_IMAGE_NGFF, JOB_TYPE_UPLOAD_IMAGE_PIPELINE
from quantem.jobs.failure_reconcile import (
    reconcile_domain_objects_for_failed_job,
    reconcile_domain_objects_for_retrying_job,
)

from .test_ngff_source_matrix import stage_asset as _stage

Image.MAX_IMAGE_PIXELS = None

INJECTED = "[WinError 5] Access is denied: the canonical image store"

#: One cell per container the importer accepts, at the dtypes where a wrong
#: decode is visible. ``.png``/uint16 is the cell round 3 shipped broken.
CELLS = [
    ("gray8.tif", "tif", "uint8"),
    ("gray16.tif", "tif", "uint16"),
    ("gray32.tif", "tif", "uint32"),
    ("rgb8.tiff", "tiff", "uint8-3band"),
    ("gray8.png", "png", "uint8"),
    ("gray16.png", "png", "uint16"),
    ("tifbytes.png", "tif-bytes-named-png", "uint16"),
]

SIDE = 256


def _plane(dtype: str) -> np.ndarray:
    yy, xx = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
    base = np.clip(0.2 + 0.5 * (xx / (SIDE - 1)) + 0.15 * np.sin(yy / 11.0), 0.02, 0.94)
    if dtype == "uint8":
        return (base * 255).astype(np.uint8)
    if dtype == "uint16":
        return (base * 65535).astype(np.uint16)
    if dtype == "uint32":
        return (base * 4294967295).astype(np.uint32)
    if dtype == "uint8-3band":
        eight = (base * 255).astype(np.uint8)
        return np.stack([eight, eight // 2, eight // 3], axis=-1)
    raise AssertionError(dtype)


def _write_source(name: str, kind: str, dtype: str, directory: Path) -> Path:
    path = directory / f"{uuid.uuid4().hex[:8]}-{name}"
    array = _plane(dtype)
    if kind == "png":
        if dtype == "uint16":
            Image.fromarray(array.astype("<u2")).save(path, format="PNG")
        else:
            Image.fromarray(array, mode="L").save(path, format="PNG")
    else:
        tifffile.imwrite(str(path), array)
    return path


def _import_with_a_broken_png_write(asset: Asset) -> Exception:
    """The verifier's fault, injected where it lands rather than with an ACL.

    The pyramid completes and is published; only the canonical-PNG thread
    fails. That is the state that used to leave a FAILED asset openable, with
    real-looking pixels behind it and no canonical PNG on disk at all.
    """

    def refuse(_plane, target):
        raise PermissionError(f"{INJECTED}: {target}")

    original = tasks_module.save_plane_as_canonical_png
    tasks_module.save_plane_as_canonical_png = refuse
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
    finally:
        tasks_module.save_plane_as_canonical_png = original
    raise AssertionError("the import was supposed to fail")


class FailedAssetNeverReopensTests(TestCase):
    """One test per container. Each hammers the vector that broke round 3."""

    ROUNDS = 40

    def setUp(self):
        self.client = Client()
        self.sources = Path(DATA_DIR) / "reopen-sources"
        self.sources.mkdir(parents=True, exist_ok=True)

    def _poll_once(self, asset: Asset) -> dict:
        detail = self.client.get(f"/api/assets/{asset.id}/")
        entry = detail.json()
        statuses = {}
        for label, url in (
            ("root", f"/ngff/assets/{asset.id}.zarr"),
            ("attrs", f"/ngff/assets/{asset.id}.zarr/.zattrs"),
            ("tile", f"/ngff/assets/{asset.id}.zarr/0/0.0.0"),
            ("thumb", f"/api/assets/{asset.id}/ngff-thumbnail/"),
        ):
            response = self.client.get(url)
            statuses[label] = response.status_code
            response.close()
        return {"entry": entry, "statuses": statuses}

    def _run_cell(self, name: str, kind: str, dtype: str):
        source = _write_source(name, kind, dtype, self.sources)
        asset = _stage(source)
        error = _import_with_a_broken_png_write(asset)
        self.assertIn("WinError 5", str(error), f"{name}: wrong fault injected")

        contradictions: list[dict] = []
        samples = 0
        for _ in range(self.ROUNDS):
            # The exact operation that re-opened the asset in round 3: the job
            # a legitimate 202 enqueued while the import was still running.
            result = ensure_ngff_for_asset_task(str(asset.id))
            self.assertEqual(
                result["status"],
                "refused",
                f"{name}: the NGFF task rebuilt a failed asset: {result}",
            )
            self.assertEqual(result["reason"], Reason.TERMINAL_FAILURE.value, name)
            sample = self._poll_once(asset)
            samples += 1
            entry = sample["entry"]
            if entry["preprocess_stage"] in {"FAILED", "CANCELLED"} and (
                entry["ngff_ready"]
                or entry["can_view"]
                or entry["can_segment"]
                or entry["ngff_url"]
                or 200 in sample["statuses"].values()
            ):
                contradictions.append(sample)

        self.assertEqual(
            contradictions,
            [],
            f"{name}: {len(contradictions)} of {samples} samples had a FAILED asset "
            f"advertising itself as openable; first: {contradictions[:1]}",
        )
        final = self._poll_once(asset)
        self.assertEqual(final["entry"]["preprocess_stage"], "FAILED", name)
        self.assertEqual(final["statuses"], {"root": 409, "attrs": 409, "tile": 409, "thumb": 404})

        # Nothing is published, and no stray generation is left behind that a
        # later request could adopt.
        asset.refresh_from_db()
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, Unavailable, name)
        self.assertEqual(resolved.reason, Reason.TERMINAL_FAILURE, name)

        # The 409 body carries the real cause, not a job-layer artefact.
        body = self.client.get(f"/ngff/assets/{asset.id}.zarr").json()
        self.assertIn("WinError 5", body["preprocess_error"], f"{name}: {body}")
        self.assertNotIn("lease", body["preprocess_error"].lower(), f"{name}: {body}")
        entry = serialize_asset_entry(Asset.objects.get(id=asset.id))
        self.assertFalse(entry["ngff_ready"] or entry["can_view"] or entry["can_segment"])


def _make_case(name: str, kind: str, dtype: str):
    def case(self):
        self._run_cell(name, kind, dtype)

    case.__name__ = f"test_a_failed_{kind.replace('-', '_')}_{dtype.replace('-', '_')}_import_stays_shut"
    case.__doc__ = f"{kind}/{dtype}: a failed import can never re-open itself"
    return case


for _name, _kind, _dtype in CELLS:
    _case = _make_case(_name, _kind, _dtype)
    setattr(FailedAssetNeverReopensTests, _case.__name__, _case)


class TheFenceTests(TestCase):
    """The property that replaces the guards: a stale build cannot publish."""

    def setUp(self):
        self.sources = Path(DATA_DIR) / "fence-sources"
        self.sources.mkdir(parents=True, exist_ok=True)

    def _asset(self):
        source = _write_source("gray16.tif", "tif", "uint16", self.sources)
        asset = _stage(source)
        from quantem.assets.asset_openable import get_asset_openable

        return asset, get_asset_openable(asset), decode_canonical_plane(source)

    def test_a_build_that_started_before_a_failure_cannot_publish_after_it(self):
        asset, openable, plane = self._asset()
        begin_attempt(asset)
        ticket = request_build(asset)
        self.assertNotIsInstance(ticket, Unavailable)
        manifest = build_pyramid(ticket, openable, plane)
        # The import fails while that build is finishing -- the verifier's exact
        # timing, and the one the between-attempt ENCODING stretch made a
        # stage-based guard blind to.
        record_attempt_failure(asset, f"ValueError: {INJECTED}")
        self.assertFalse(
            publish(ticket, manifest),
            "a build from a superseded attempt published over a failed import",
        )
        self.assertIsInstance(resolve_pyramid(asset, intent=Intent.SERVE), Unavailable)

    def test_a_build_from_attempt_n_cannot_publish_over_attempt_n_plus_1(self):
        asset, openable, plane = self._asset()
        begin_attempt(asset)
        stale = request_build(asset)
        begin_attempt(asset)  # the retry edge, where the stage is still ENCODING
        fresh = request_build(asset)
        fresh_manifest = build_pyramid(fresh, openable, plane)
        self.assertTrue(publish(fresh, fresh_manifest))
        stale_manifest = build_pyramid(stale, openable, plane)
        self.assertFalse(publish(stale, stale_manifest))
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(resolved, PublishedPyramid)
        self.assertEqual(resolved.generation_id, fresh.generation_id)

    def test_two_builds_of_one_attempt_cannot_both_publish(self):
        asset, openable, plane = self._asset()
        begin_attempt(asset)
        first = request_build(asset)
        second = request_build(asset)
        self.assertTrue(publish(first, build_pyramid(first, openable, plane)))
        self.assertFalse(publish(second, build_pyramid(second, openable, plane)))
        resolved = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertEqual(resolved.generation_id, first.generation_id)

    def test_a_terminal_import_is_refused_a_ticket_at_all(self):
        asset, _openable, _plane = self._asset()
        record_attempt_failure(asset, "ValueError: boom")
        asset.preprocess_stage = "FAILED"
        asset.save(update_fields=["preprocess_stage"])
        refusal = request_build(asset)
        self.assertIsInstance(refusal, Unavailable)
        self.assertEqual(refusal.reason, Reason.TERMINAL_FAILURE)
        with self.assertRaises(PyramidBuildRefused):
            from quantem.assets.ngff import build_and_publish

            build_and_publish(asset, np.zeros((4, 4), dtype=np.uint8))

    def test_a_retry_note_cannot_overwrite_the_real_cause(self):
        """FINDING 4, at its root rather than at its symptom.

        Three concurrent GETs raced one enqueue into three jobs; their
        ``StorageLeaseConflict`` reconciled last and replaced the real
        ``WinError 5`` on the one field the 409 body reads. The terminal message
        now lives on ``failure_detail``, which the retry note may not write.
        """

        asset, _openable, _plane = self._asset()
        record_attempt_failure(asset, f"ValueError: {INJECTED}")
        reconcile_domain_objects_for_retrying_job(
            JOB_TYPE_ENSURE_IMAGE_NGFF,
            {"asset_id": str(asset.id)},
            "Attempt 1 of 3 failed; retrying automatically. StorageLeaseConflict: "
            "Storage artifact is leased by another active job",
        )
        asset.refresh_from_db()
        self.assertIn("lease", asset.preprocess_error.lower(), "the note did not land")
        self.assertIn("WinError 5", failure_detail(asset))
        self.assertNotIn("lease", failure_detail(asset).lower())

    def test_a_rebuild_keeps_serving_the_previous_generation_until_it_publishes(self):
        asset, openable, plane = self._asset()
        begin_attempt(asset)
        first = request_build(asset)
        self.assertTrue(publish(first, build_pyramid(first, openable, plane)))
        second = request_build(asset)
        during = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertIsInstance(during, PublishedPyramid)
        self.assertEqual(during.generation_id, first.generation_id)
        self.assertTrue(publish(second, build_pyramid(second, openable, plane)))
        after = resolve_pyramid(asset, intent=Intent.SERVE)
        self.assertEqual(after.generation_id, second.generation_id)


class ConcurrentEnqueueTests(SimpleTestCase):
    """Twenty simultaneous tile GETs must produce one job, not three."""

    def test_twenty_simultaneous_lazy_builds_collapse_to_one_job(self):
        from .test_ngff_publish_race import run_driver

        result = run_driver("concurrent_enqueue", 20)
        try:
            self.assertEqual(result["ready_before_go"], 20, result)
            self.assertEqual(set(result["child_exit_codes"]), {0}, result)
            self.assertEqual(
                result["job_count"],
                1,
                "concurrent GETs created more than one NGFF job; they fight for the "
                "storage lease and their conflict reconciles over the real error. "
                f"{result}",
            )
            self.assertEqual(len(result["job_tokens"]), 1, result)
        finally:
            shutil.rmtree(result["_workdir"], ignore_errors=True)


class TheAttemptTokenIsTheEnqueueKeyTests(TestCase):
    """A new attempt gets its own job; a stale one can never be revived."""

    def setUp(self):
        self.sources = Path(DATA_DIR) / "enqueue-sources"
        self.sources.mkdir(parents=True, exist_ok=True)

    def test_a_new_attempt_gets_its_own_job_and_a_stale_one_is_not_revived(self):
        source = _write_source("gray8.tif", "tif", "uint8", self.sources)
        asset = _stage(source)
        begin_attempt(asset)
        first = request_lazy_build(asset)
        self.assertEqual(request_lazy_build(asset).id, first.id)
        begin_attempt(asset)
        second = request_lazy_build(asset)
        self.assertNotEqual(second.id, first.id)
        self.assertNotEqual(
            second.payload_json["attempt_token"], first.payload_json["attempt_token"]
        )

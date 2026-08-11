"""An overlay with more object than the pool threshold builds, or fails loudly.

Two properties, both of which the shipped product got wrong on every real EM
image:

1. **It builds.** Above :data:`RASTER_POOL_MIN_OBJECTS` the rasteriser fans out
   to a :class:`~concurrent.futures.ProcessPoolExecutor`. On Windows that pool
   spawns, and a spawned child unpickles ``rasterize_tile_worker`` by importing
   :mod:`quantem.segmentation.overlay_ngff.render` -- whose package ``__init__``
   pulls Django models. With no ``django.setup()`` in the child that import
   raises ``AppRegistryNotReady`` before the first task runs, every worker dies,
   and the pool comes back as ``BrokenProcessPool``. Every real image has more
   than 2 000 objects, so no real image ever produced an overlay.

2. **A build that fails, fails.** The old failure was not just a crash: the
   manifest endpoint reset the FAILED state back to BUILDING and re-queued the
   same doomed job on every poll, so the user watched "Overlay updating..."
   forever and nothing anywhere said why.
"""

from __future__ import annotations

import time
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import zarr
from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.assets.models import Asset, Rendition
from quantem.jobs.constants import JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY
from quantem.jobs.models import Job
from quantem.jobs.registry import get_handler
from quantem.jobs.reporter import CancelToken, JobReporter
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationOverlayLabel,
    SegmentationOverlayState,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff import (
    ensure_overlay_manifest,
    get_overlay_active_bundle_path,
    get_overlay_root,
)
from quantem.segmentation.overlay_ngff.constants import (
    ACTIVE_OVERLAY_JOB_STATUSES,
    RASTER_POOL_MIN_OBJECTS,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type


def _create_sized_asset(*, width: int, height: int) -> Asset:
    """An asset of the given logical size, with no pixels behind it.

    The overlay build reads ``logical_width``/``logical_height`` and the object
    rows -- never the image itself (see ``overlay_ngff.dimensions``). Writing a
    real 8192-square TIFF would add ~64 MB of I/O to a test that would not look
    at a single one of those bytes.
    """
    asset = Asset.objects.create(
        display_name=f"Overlay scale asset {uuid4().hex[:8]}",
        original_filename="overlay_scale.tif",
        logical_width=width,
        logical_height=height,
        channels=1,
        bit_depth=8,
        pixel_size_nm=5.0,
        preprocess_stage="DONE",
        preprocess_progress=100.0,
    )
    Rendition.objects.create(
        asset=asset,
        type=Rendition.TYPE_FULL,
        storage_root="DATA_DIR",
        stored_path=f"images/overlay_scale_{asset.id}.png",
        path_exists=False,
        is_directory=False,
        stored_width=width,
        stored_height=height,
        stored_channels=1,
        stored_bit_depth=8,
    )
    return asset


def _seed_objects(segmentation: ImageSegmentation, *, count: int, extent: int) -> None:
    """``count`` small square objects spread over an ``extent``-square image."""
    side = 24
    columns = int(count**0.5) + 1
    step = max(side + 2, extent // columns)
    rows = []
    for index in range(count):
        column = index % columns
        row = index // columns
        x0 = min(extent - side - 1, column * step)
        y0 = min(extent - side - 1, row * step)
        polygon = box(x0, y0, x0 + side, y0 + side)
        centroid = polygon.centroid
        rows.append(
            SegmentObject(
                segmentation=segmentation,
                geometry=polygon,
                centroid=centroid,
                bbox=polygon.envelope,
                label_state="INFERRED",
                confidence_score=0.8,
                features={},
            )
        )
    SegmentObject.objects.bulk_create(rows, batch_size=500)


def _run_queued_overlay_job(job: Job) -> Job:
    """Execute one queued overlay job through its real handler, in-process.

    The handler is what the spawned job worker calls; running it here exercises
    the same rasteriser and the same process pool, and lets the test assert on
    the SUCCESS/FAILED transition the runner would write.
    """
    handler = get_handler(job.type)
    reporter = JobReporter(str(job.id))
    cancel = CancelToken(str(job.id))
    try:
        result = handler(job.payload_json, reporter, cancel)
    except Exception as exc:  # mirrors runner._run_job_in_subprocess
        Job.objects.filter(id=job.id).update(
            status="FAILED",
            message=f"failed: {exc.__class__.__name__}: {exc}",
        )
    else:
        Job.objects.filter(id=job.id).update(
            status="SUCCESS",
            progress=100.0,
            result_json=result or {},
            message="completed",
        )
    job.refresh_from_db()
    return job


class OverlayProcessPoolBuildTests(TestCase):
    """The acceptance case from the plan: 4 000 objects on an 8192-square asset."""

    OBJECT_COUNT = 4000
    EXTENT = 8192
    DEADLINE_SECONDS = 120.0

    def setUp(self):
        self.client = APIClient()
        self.asset = _create_sized_asset(width=self.EXTENT, height=self.EXTENT)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def test_full_rebuild_of_4000_objects_reaches_success_and_ready(self):
        _seed_objects(self.segmentation, count=self.OBJECT_COUNT, extent=self.EXTENT)
        self.assertGreater(self.OBJECT_COUNT, RASTER_POOL_MIN_OBJECTS)

        started = time.monotonic()
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/overlay-rebuild/",
            {"mode": "full"},
            format="json",
        )
        self.assertEqual(response.status_code, 202)

        job = Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY).latest("created_at")
        job = _run_queued_overlay_job(job)
        elapsed = time.monotonic() - started

        self.assertEqual(
            job.status,
            "SUCCESS",
            f"overlay rebuild job did not succeed: {job.message}",
        )
        manifest = ensure_overlay_manifest(self.segmentation)
        self.assertEqual(manifest["status"], SegmentationOverlayState.STATUS_READY)
        self.assertIsNotNone(manifest["ngff_url"])
        self.assertEqual(manifest["last_error"], "")
        self.assertLess(elapsed, self.DEADLINE_SECONDS, f"took {elapsed:.1f}s")

        # READY with a store of the right shape is not proof that the pool
        # workers *drew* anything -- a pool returning blank tiles passes both.
        # So check pixels: every object the tile workers were handed has to be
        # in the raster under the label the LUT gave it.
        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        labels = zarr.open_array(
            str(get_overlay_active_bundle_path(state) / "labels" / "0"), mode="r"
        )
        sampled = list(SegmentObject.objects.filter(segmentation=self.segmentation)[:200])
        self.assertEqual(len(sampled), 200)
        for obj in sampled:
            expected = SegmentationOverlayLabel.objects.get(
                overlay_state=state, object_uuid=obj.id
            ).label
            centroid = obj.centroid
            painted = int(np.asarray(labels[int(centroid.y), int(centroid.x)]))
            self.assertEqual(
                painted,
                int(expected),
                f"object {obj.id} is missing from the raster at "
                f"({int(centroid.x)}, {int(centroid.y)})",
            )


class OverlayBuildFailureIsHonestTests(TestCase):
    """A failed build fails: it never re-queues itself into a forever spinner."""

    def setUp(self):
        self.client = APIClient()
        self.asset = _create_sized_asset(width=1024, height=1024)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _fail_the_build(self, message: str = "pool exploded") -> Job:
        self.client.post(
            f"/api/segmentations/{self.segmentation.id}/overlay-rebuild/",
            {"mode": "full"},
            format="json",
        )
        job = Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY).latest("created_at")
        with patch(
            "quantem.segmentation.overlay_ngff.mutations.rebuild_overlay_full",
            side_effect=RuntimeError(message),
        ):
            return _run_queued_overlay_job(job)

    def test_failed_build_fails_the_job_with_the_real_reason(self):
        job = self._fail_the_build("pool exploded")

        self.assertEqual(job.status, "FAILED")
        self.assertIn("pool exploded", job.message)
        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        self.assertEqual(state.status, SegmentationOverlayState.STATUS_FAILED)
        self.assertIn("pool exploded", state.last_error)

    def test_manifest_reports_the_failure_instead_of_building_forever(self):
        self._fail_the_build("pool exploded")
        job_count_after_failure = Job.objects.filter(
            type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY
        ).count()

        for _ in range(3):
            manifest = ensure_overlay_manifest(self.segmentation)

        self.assertEqual(manifest["status"], SegmentationOverlayState.STATUS_FAILED)
        self.assertIn("pool exploded", manifest["last_error"])
        self.assertEqual(
            Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY).count(),
            job_count_after_failure,
            "the manifest endpoint re-queued a rebuild that had already failed",
        )

    def test_a_real_storage_failure_reaches_the_manifest_intact(self):
        """No mock: a stray file where the overlay directory belongs.

        The build then fails inside ``_create_empty_label_store`` with an OS
        error, which is a fair stand-in for the disk-shaped failures a desktop
        install actually meets (a sync client holding a name, a leftover file
        from a crashed version). Two things have to survive it: the endpoint
        must still answer (writing the *debug* manifest fails on the same path,
        and used to turn every poll into an empty HTTP 500), and the answer
        must carry the real reason.
        """
        blocker = get_overlay_root(str(self.segmentation.id))
        blocker.parent.mkdir(parents=True, exist_ok=True)
        blocker.write_bytes(b"not a directory")
        self.addCleanup(blocker.unlink, True)

        self.client.post(
            f"/api/segmentations/{self.segmentation.id}/overlay-rebuild/",
            {"mode": "full"},
            format="json",
        )
        job = Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY).latest("created_at")
        job = _run_queued_overlay_job(job)
        self.assertEqual(job.status, "FAILED")

        response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], SegmentationOverlayState.STATUS_FAILED)
        self.assertTrue(
            response.data["last_error"].strip(),
            "the manifest reported FAILED with no reason on it",
        )

    def _kill_the_worker_silently(self, *, times: int) -> None:
        """Jobs that FAILED without the handler ever writing to the state.

        The other half of the forever spinner: a worker terminated by the OS
        (out of memory on a dense image, a machine going to sleep) never runs
        ``run_overlay_rebuild_job``'s except arm, so the state stays BUILDING
        with nothing wrong recorded on it -- which is precisely the shape the
        manifest endpoint used to treat as "nobody is building this, start
        one".
        """
        for _ in range(times):
            self.client.get(
                f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
            )
            Job.objects.filter(
                type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
                status__in=ACTIVE_OVERLAY_JOB_STATUSES,
            ).update(status="FAILED", message="failed: worker exited with code 3221225477")

    def test_a_worker_that_dies_silently_still_stops_after_a_few_attempts(self):
        self._kill_the_worker_silently(times=3)

        manifest = ensure_overlay_manifest(self.segmentation)

        self.assertEqual(manifest["status"], SegmentationOverlayState.STATUS_FAILED)
        self.assertIn("3221225477", manifest["last_error"])
        self.assertFalse(
            Job.objects.filter(
                type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
                status__in=ACTIVE_OVERLAY_JOB_STATUSES,
            ).exists(),
            "a fourth doomed rebuild was queued",
        )

    def test_a_couple_of_silent_deaths_are_still_retried(self):
        """The budget is for a loop, not for the first sign of trouble."""
        self._kill_the_worker_silently(times=2)

        manifest = ensure_overlay_manifest(self.segmentation)

        self.assertEqual(manifest["status"], SegmentationOverlayState.STATUS_BUILDING)
        self.assertTrue(
            Job.objects.filter(
                type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
                status__in=ACTIVE_OVERLAY_JOB_STATUSES,
            ).exists()
        )

    def test_an_explicit_rebuild_clears_the_failure_and_retries(self):
        self._fail_the_build("pool exploded")

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/overlay-rebuild/",
            {"mode": "full"},
            format="json",
        )

        self.assertEqual(response.status_code, 202)
        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        self.assertNotEqual(state.status, SegmentationOverlayState.STATUS_FAILED)
        self.assertEqual(state.last_error, "")
        self.assertTrue(
            Job.objects.filter(
                type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
                status__in=ACTIVE_OVERLAY_JOB_STATUSES,
            ).exists()
        )

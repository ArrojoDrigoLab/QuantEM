from __future__ import annotations

from unittest.mock import patch

import numpy as np
import zarr
from django.test import TestCase
from django.utils import timezone
from numcodecs import Blosc
from rest_framework.test import APIClient
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon

from quantem.jobs.constants import JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY
from quantem.jobs.models import Job
from quantem.jobs.reporter import JobCancelledError
from quantem.segmentation import overlay_ngff
from quantem.segmentation.geometry import extract_polygons
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationOverlayLabel,
    SegmentationOverlayState,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff import (
    DirtyBBox,
    apply_partial_overlay_update,
    build_label_lut_binary,
    get_overlay_active_bundle_path,
    overlay_jobs_for_bundle,
    queue_full_overlay_rebuild,
    queue_overlay_rebuild,
    rebuild_overlay_full,
    register_overlay_mutation,
    run_overlay_rebuild_job,
)
from quantem.segmentation.overlay_ngff.constants import (
    ACTIVE_OVERLAY_JOB_STATUSES,
    COLOR_CONFIRMED,
    COLOR_EXCLUDED,
)
from quantem.segmentation.overlay_ngff.manifest import OVERLAY_CANCELLED_MESSAGE
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _open_labels_level0(state: SegmentationOverlayState):
    """Open the level-0 ``labels`` array (uint32, 2D) of a built bundle."""
    root = get_overlay_active_bundle_path(state)
    return zarr.open_array(str(root / "labels" / "0"), mode="r")


def _open_border_level0(state: SegmentationOverlayState):
    """Open the level-0 ``border`` array (uint8, 2D) of a built bundle."""
    root = get_overlay_active_bundle_path(state)
    return zarr.open_array(str(root / "border" / "0"), mode="r")


def _label_value_at(arr, y: int, x: int) -> int:
    return int(np.asarray(arr[y, x]))


def _border_max(arr, y0: int, y1: int, x0: int, x1: int) -> int:
    return int(np.asarray(arr[y0:y1, x0:x1]).max())


def _label_for_object(state: SegmentationOverlayState, object_uuid) -> int:
    """Look up the dense label assigned to an object's uuid for this bundle."""
    row = SegmentationOverlayLabel.objects.get(overlay_state=state, object_uuid=object_uuid)
    return int(row.label)


def _lut_rgba_for_label(state: SegmentationOverlayState, label: int) -> tuple[int, int, int, int]:
    rgba_bytes, max_label = build_label_lut_binary(state)
    assert label <= max_label, f"label {label} exceeds max_label {max_label}"
    buffer = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape((max_label + 1, 4))
    return tuple(int(channel) for channel in buffer[label])


class SegmentationOverlayManifestTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Manifest Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def test_manifest_endpoint_queues_initial_build(self):
        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/overlay-manifest/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "BUILDING")
        self.assertIsNone(response.data["ngff_url"])
        # The ID-map overlay exposes two integer arrays + a render-time LUT,
        # replacing the legacy pre-colored channel-index map.
        self.assertEqual(response.data["arrays"], ["labels", "border"])
        self.assertEqual(response.data["label_dtype"], "uint32")
        self.assertEqual(response.data["display_role"], "confirmed")
        self.assertTrue(response.data["data_ready"])
        self.assertEqual(response.data["update_job"]["status"], "PENDING")
        self.assertIsNotNone(response.data["lut_url"])
        self.assertNotIn("channel_indices", response.data)
        self.assertTrue(Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY).exists())

    def test_manifest_endpoint_requeues_dirty_valid_overlay(self):
        state = rebuild_overlay_full(self.segmentation, desired_revision=1)
        state.status = SegmentationOverlayState.STATUS_DIRTY
        state.applied_revision = 1
        state.desired_revision = 2
        state.pending_full_rebuild = True
        state.save(
            update_fields=[
                "status",
                "applied_revision",
                "desired_revision",
                "pending_full_rebuild",
                "updated_at",
            ]
        )

        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/overlay-manifest/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "BUILDING")
        queued_job = Job.objects.get(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY)
        self.assertEqual(
            queued_job.payload_json,
            {"segmentation_id": str(self.segmentation.id), "mode": "full"},
        )

    def test_one_bundle_is_queued_once_however_often_it_is_asked_for(self):
        url = f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
        for _ in range(3):
            self.assertEqual(self.client.get(url).status_code, 200)
        for _ in range(3):
            self.assertEqual(
                self.client.get(url, {"source_model": "quantem:nucleus"}).status_code,
                200,
            )
        # Differently cased: normalised, so it must not start a third build.
        self.assertEqual(self.client.get(url, {"source_model": "QuantEM:Nucleus"}).status_code, 200)

        jobs = list(Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY))
        self.assertEqual(
            len(jobs),
            2,
            "expected exactly one rebuild per bundle: confirmed display and quantem:nucleus",
        )
        self.assertEqual(
            sorted(job.payload_json.get("source_model", "") for job in jobs),
            ["", "quantem:nucleus"],
        )

    def test_the_confirmed_display_and_a_per_source_bundle_are_separate_builds(self):
        """Reported as duplicate work; it is not. Do not collapse these.

        A segmentation carries a confirmed-display overlay and one overlay per model
        that produced objects in it, each its own zarr store with its own
        revisions. Opening a nucleus segmentation asks for both, so two jobs is
        the right number -- merging them would leave one bundle stale for good.

        What made the pair look like duplicate work is that both queue rows
        render as ``"Rebuild segmentation overlay"``
        (``jobs.constants.JOB_TYPE_LABELS``) with nothing to tell them apart.
        The distinguishing value is on the job, in the payload and in the tag,
        and this pins it there so the queue view has something to show.
        """
        url = f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
        self.client.get(url)
        self.client.get(url, {"source_model": "quantem:nucleus"})

        jobs = {
            job.payload_json.get("source_model", ""): job
            for job in Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY)
        }
        self.assertEqual(set(jobs), {"", "quantem:nucleus"})
        self.assertIn("source_model:quantem:nucleus", jobs["quantem:nucleus"].tags)
        self.assertNotIn(
            "source_model:quantem:nucleus",
            jobs[""].tags,
            "the confirmed-display build must not be tagged with one model's name",
        )


class SegmentationOverlayRebuildStateTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Overlay Rebuild State Test Image")
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
            confidence_score=None,
            features={"sam_score": 0.9},
        )

    def test_full_rebuild_preserves_newer_revision_bumped_mid_build(self):
        initial_state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        self.assertEqual(initial_state.applied_revision, 0)

        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        state.status = SegmentationOverlayState.STATUS_BUILDING
        state.applied_revision = 0
        state.desired_revision = 1
        state.pending_full_rebuild = False
        state.dirty_chunk_runs = []
        state.save(
            update_fields=[
                "status",
                "applied_revision",
                "desired_revision",
                "pending_full_rebuild",
                "dirty_chunk_runs",
                "updated_at",
            ]
        )

        original_rasterize_tile_worker = overlay_ngff.render.rasterize_tile_worker
        mutated = False

        def rasterize_with_revision_bump(*args, **kwargs):
            nonlocal mutated
            if not mutated:
                mutated = True
                current_state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
                current_state.status = SegmentationOverlayState.STATUS_DIRTY
                current_state.desired_revision = 2
                current_state.pending_full_rebuild = True
                current_state.save(
                    update_fields=[
                        "status",
                        "desired_revision",
                        "pending_full_rebuild",
                        "updated_at",
                    ]
                )
            return original_rasterize_tile_worker(*args, **kwargs)

        with patch(
            "quantem.segmentation.overlay_ngff.render.rasterize_tile_worker",
            side_effect=rasterize_with_revision_bump,
        ):
            rebuilt_state = rebuild_overlay_full(self.segmentation, desired_revision=1)

        rebuilt_state.refresh_from_db()
        self.assertEqual(rebuilt_state.applied_revision, 1)
        self.assertEqual(rebuilt_state.desired_revision, 2)
        self.assertTrue(rebuilt_state.pending_full_rebuild)
        self.assertEqual(rebuilt_state.status, SegmentationOverlayState.STATUS_DIRTY)

    def test_partial_rebuild_preserves_later_dirty_runs_and_applied_revision(self):
        rebuild_overlay_full(self.segmentation, desired_revision=0)
        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        state.status = SegmentationOverlayState.STATUS_BUILDING
        state.applied_revision = 0
        state.desired_revision = 1
        state.pending_full_rebuild = False
        state.dirty_chunk_runs = []
        state.save(
            update_fields=[
                "status",
                "applied_revision",
                "desired_revision",
                "pending_full_rebuild",
                "dirty_chunk_runs",
                "updated_at",
            ]
        )

        original_rasterize_tile_worker = overlay_ngff.render.rasterize_tile_worker
        mutated = False
        later_dirty_run = {
            "revision": 2,
            "bbox": {
                "x_min": 0,
                "y_min": 0,
                "x_max": 64,
                "y_max": 64,
            },
            "chunk_x_min": 0,
            "chunk_x_max": 0,
            "chunk_y_min": 0,
            "chunk_y_max": 0,
        }

        def rasterize_with_dirty_run_bump(*args, **kwargs):
            nonlocal mutated
            if not mutated:
                mutated = True
                current_state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
                current_state.status = SegmentationOverlayState.STATUS_DIRTY
                current_state.desired_revision = 2
                current_state.dirty_chunk_runs = [later_dirty_run]
                current_state.save(
                    update_fields=[
                        "status",
                        "desired_revision",
                        "dirty_chunk_runs",
                        "updated_at",
                    ]
                )
            return original_rasterize_tile_worker(*args, **kwargs)

        with patch(
            "quantem.segmentation.overlay_ngff.render.rasterize_tile_worker",
            side_effect=rasterize_with_dirty_run_bump,
        ):
            updated_state = apply_partial_overlay_update(
                self.segmentation,
                dirty_bbox=DirtyBBox(x_min=0, y_min=0, x_max=64, y_max=64),
                desired_revision=1,
            )

        updated_state.refresh_from_db()
        self.assertEqual(updated_state.applied_revision, 1)
        self.assertEqual(updated_state.desired_revision, 2)
        self.assertFalse(updated_state.pending_full_rebuild)
        self.assertEqual(updated_state.dirty_chunk_runs, [later_dirty_run])
        self.assertEqual(updated_state.status, SegmentationOverlayState.STATUS_DIRTY)


class SegmentationOverlayCancellationTests(TestCase):
    """What has to survive the user pressing Cancel on an overlay rebuild.

    Two things, and neither was true when responsive cancellation first landed.

    *The cancel has to stick.* The handler recorded a cancelled build as DIRTY,
    which is indistinguishable from "there is pending work and nobody is
    building it" -- the exact condition ``ensure_overlay_manifest`` answers by
    enqueueing a rebuild. The labelling screen polls every 1.5 s, so the job the
    user had just cancelled was back in the Tasks drawer, from zero, before they
    could look away, and a long build on a large image could not be stopped at
    all.

    *A full rebuild owed by someone else has to survive it.* The flag is the
    only record that one is owed -- the ``async_full`` registration path adds no
    dirty run to fall back on -- and anything that clears it downgrades the
    follow-up to a partial, which repaints one bbox and then marks the bundle
    READY over geometry that was never rasterised. Analysis reuses a settled
    raster as-is, so that would silently drop objects from exported
    measurements.
    """

    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Cancellation Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = Polygon(((10, 10), (40, 10), (40, 40), (10, 40), (10, 10)))
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=None,
            features={"sam_score": 0.9},
        )
        # Build once, so the cancellations under test are the interesting kind:
        # a usable-but-stale picture stays on screen behind them.
        rebuild_overlay_full(self.segmentation, desired_revision=1)

    def _mark_partial_work_pending(self) -> SegmentationOverlayState:
        """Put the bundle where a queued partial rebuild would find it."""
        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        state.status = SegmentationOverlayState.STATUS_DIRTY
        state.applied_revision = 1
        state.desired_revision = 2
        state.pending_full_rebuild = False
        state.dirty_chunk_runs = [
            {
                "revision": 2,
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 64, "y_max": 64},
                "chunk_x_min": 0,
                "chunk_x_max": 0,
                "chunk_y_min": 0,
                "chunk_y_max": 0,
            }
        ]
        state.save(
            update_fields=[
                "status",
                "applied_revision",
                "desired_revision",
                "pending_full_rebuild",
                "dirty_chunk_runs",
                "updated_at",
            ]
        )
        return state

    def _active_rebuild_count(self, segmentation: ImageSegmentation) -> int:
        return (
            overlay_jobs_for_bundle(str(segmentation.id))
            .filter(status__in=ACTIVE_OVERLAY_JOB_STATUSES)
            .count()
        )

    def test_cancel_keeps_a_full_rebuild_registered_while_the_job_ran(self):
        """The race the handler must survive, and used to lose.

        A completed extraction (``seg_core.db.extraction``) or the rebuild
        button asks for a full rebuild while a partial job is mid-flight; the
        request lands on ``pending_full_rebuild`` and nowhere else, because
        ``queue_overlay_rebuild`` is a no-op while that job holds the bundle.
        The handler then recomputed the flag from the ``state`` it had loaded
        *before* the build began and wrote ``False`` over it.
        """
        self._mark_partial_work_pending()
        registered = False

        def cancel_check():
            nonlocal registered
            if not registered:
                registered = True
                queue_full_overlay_rebuild(self.segmentation)
                return
            raise JobCancelledError("cancelled from the tasks drawer")

        with self.assertRaises(JobCancelledError):
            run_overlay_rebuild_job(
                self.segmentation,
                mode="partial",
                cancel_check=cancel_check,
            )

        self.assertTrue(registered, "the mid-build full rebuild was never registered")
        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        self.assertTrue(
            state.pending_full_rebuild,
            "a full rebuild requested during the build must outlive the cancellation",
        )
        self.assertEqual(state.status, SegmentationOverlayState.STATUS_FAILED)
        self.assertEqual(state.last_error, OVERLAY_CANCELLED_MESSAGE)

    def test_an_incremental_edit_does_not_discharge_an_owed_full_rebuild(self):
        """The other end of the same flag: an edit must not consume it either.

        ``overlay_rebuild_policy`` answers "async_partial" precisely *because* a
        full rebuild is already owed, so recomputing the flag from that answer
        cleared it -- and the partial that followed painted this edit's bbox and
        called the bundle settled.
        """
        state = self._mark_partial_work_pending()
        state.pending_full_rebuild = True
        state.save(update_fields=["pending_full_rebuild", "updated_at"])

        register_overlay_mutation(
            self.segmentation,
            dirty_bbox=DirtyBBox(x_min=0, y_min=0, x_max=32, y_max=32),
        )

        state.refresh_from_db()
        self.assertTrue(state.pending_full_rebuild)
        queued = overlay_jobs_for_bundle(str(self.segmentation.id)).get()
        self.assertEqual(
            queued.payload_json.get("mode"),
            "full",
            "the owed full rebuild, not a partial, is what has to be queued",
        )

    def test_a_cancelled_rebuild_is_not_requeued_by_the_next_manifest_poll(self):
        self._mark_partial_work_pending()
        job = queue_overlay_rebuild(self.segmentation, mode="partial")

        def cancel_check():
            raise JobCancelledError("cancelled from the tasks drawer")

        with self.assertRaises(JobCancelledError):
            run_overlay_rebuild_job(
                self.segmentation,
                mode="partial",
                cancel_check=cancel_check,
            )
        job.status = "CANCELLED"
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "finished_at"])

        url = f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
        for _ in range(3):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)

        self.assertEqual(
            self._active_rebuild_count(self.segmentation),
            0,
            "the poll re-queued the rebuild the user had just cancelled",
        )
        self.assertEqual(response.data["status"], "FAILED")
        self.assertEqual(response.data["last_error"], OVERLAY_CANCELLED_MESSAGE)
        self.assertIsNotNone(
            response.data["ngff_url"],
            "the previously built overlay must keep being served behind the stop",
        )

    def test_a_cancel_that_killed_the_worker_is_not_requeued_either(self):
        """The path where the handler above never runs.

        When the runner stops waiting it terminates the worker process, so
        nothing writes the stop to the state: it is left mid-build, which is the
        shape the manifest answers by enqueueing another job. The cancelled
        queue row is the only surviving record, which is why the brake reads it
        rather than the state.
        """
        state = self._mark_partial_work_pending()
        job = queue_overlay_rebuild(self.segmentation, mode="partial")
        state.status = SegmentationOverlayState.STATUS_BUILDING
        state.save(update_fields=["status", "updated_at"])
        job.status = "CANCELLED"
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "finished_at"])

        url = f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
        for _ in range(3):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)

        self.assertEqual(self._active_rebuild_count(self.segmentation), 0)
        self.assertEqual(response.data["status"], "FAILED")
        self.assertEqual(response.data["last_error"], OVERLAY_CANCELLED_MESSAGE)

    def test_a_cancelled_first_build_stops_asking_but_an_edit_resumes_it(self):
        """No bundle to fall back on, and no failure budget spent by a cancel.

        The first-ever build of a segmentation leaves nothing on disk, so this
        path re-queued unconditionally: ``_failed_rebuilds_since_last_success``
        counts ``FAILED`` jobs and a cancellation is not one, so the build the
        user stopped came back on every poll for as long as the screen was open.
        The stop still has to be temporary -- the user's next edit clears it.
        """
        image = create_image_from_test_tiff("Overlay Cancellation First Build Image")
        segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        url = f"/api/segmentations/{segmentation.id}/overlay-manifest/"
        self.assertEqual(self.client.get(url).status_code, 200)

        job = overlay_jobs_for_bundle(str(segmentation.id)).get()
        job.status = "CANCELLED"
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "finished_at"])

        for _ in range(3):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)

        self.assertEqual(self._active_rebuild_count(segmentation), 0)
        self.assertEqual(response.data["status"], "FAILED")
        self.assertEqual(response.data["last_error"], OVERLAY_CANCELLED_MESSAGE)
        self.assertIsNone(response.data["ngff_url"])

        queue_full_overlay_rebuild(segmentation)

        self.assertEqual(
            self._active_rebuild_count(segmentation),
            1,
            "the user's own retry has to lift the stop the cancellation put in place",
        )
        state = SegmentationOverlayState.objects.get(segmentation=segmentation)
        self.assertNotEqual(state.status, SegmentationOverlayState.STATUS_FAILED)
        self.assertEqual(state.last_error, "")


class SegmentationOverlayQueryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Query Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.confirmed = self._create_segment(10, 10, 20, 20, "CONFIRMED")
        self.candidate = self._create_segment(30, 30, 42, 42, "CANDIDATE")

    def _create_segment(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label_state: str,
    ) -> SegmentObject:
        polygon = Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            confidence_score=0.8 if label_state != "CONFIRMED" else None,
            features={"sam_score": 0.9},
        )

    def test_at_point_respects_states_filter(self):
        response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/segments/at-point",
            {"x": 15, "y": 15, "states": "CONFIRMED"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["id"], str(self.confirmed.id))

    def test_query_region_returns_exact_state_filtered_hits(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/query-region",
            {
                "bbox": {"x0": 0, "y0": 0, "x1": 50, "y1": 50},
                "states": ["CANDIDATE"],
                "include_geometry": True,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["segments"]), 1)
        self.assertEqual(response.data["segments"][0]["id"], str(self.candidate.id))
        self.assertGreaterEqual(len(response.data["segments"][0]["geometry_coords"]), 4)


class SegmentationOverlaySyncPartialTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Sync Partial Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.segment = self._create_segment("CANDIDATE")

    def _create_segment(self, label_state: str) -> SegmentObject:
        polygon = Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            confidence_score=0.8,
            features={"sam_score": 0.9},
        )

    def test_label_update_defers_the_raster_and_recolours_immediately(self):
        rebuild_overlay_full(self.segmentation, desired_revision=0)

        response = self.client.post(
            f"/api/segments/{self.segment.id}/label/",
            {"label_state": "CONFIRMED"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        # An answer registers the edit and leaves the raster to the queue (see
        # api_views.segments.labels): the response is not waiting on a disk
        # write. What the reviewer sees is still correct at once, because the
        # colour comes from the LUT rather than the raster -- so this asserts
        # both that the dense label survived and that the LUT already resolves
        # the confirmed colour, with `applied_revision` still behind.
        self.assertEqual(response.data["overlay"]["rebuild_mode"], "async_partial")
        self.assertFalse(response.data["overlay"]["sync_applied"])
        self.assertGreater(
            response.data["overlay"]["desired_revision"],
            response.data["overlay"]["applied_revision"],
        )

        state = SegmentationOverlayState.objects.get(
            segmentation=self.segmentation, candidate_source_model=""
        )
        labels0 = _open_labels_level0(state)
        label = _label_for_object(state, self.segment.id)
        self.assertEqual(_label_value_at(labels0, 15, 15), label)

        rgba = _lut_rgba_for_label(state, label)
        self.assertEqual(rgba[:3], _hex_to_rgb(COLOR_CONFIRMED))
        # CONFIRMED is visible by default.
        self.assertEqual(rgba[3], 255)

    def test_batch_reject_moves_candidate_out_of_candidate_channels(self):
        rebuild_overlay_full(self.segmentation, desired_revision=0)

        response = self.client.post(
            "/api/segments/labels/batch/",
            {"labels": [{"id": str(self.segment.id), "label_state": "EXCLUDED"}]},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(str(self.segmentation.id), response.data["overlays"])

        state = SegmentationOverlayState.objects.get(
            segmentation=self.segmentation, candidate_source_model=""
        )
        labels0 = _open_labels_level0(state)
        label = _label_for_object(state, self.segment.id)
        # The raster keeps the object's dense label; only its LUT colour/alpha
        # changes to the excluded state.
        self.assertEqual(_label_value_at(labels0, 15, 15), label)

        rgba = _lut_rgba_for_label(state, label)
        self.assertEqual(rgba[:3], _hex_to_rgb(COLOR_EXCLUDED))
        # EXCLUDED is hidden by default, so the LUT alpha is zeroed.
        self.assertEqual(rgba[3], 0)


class SegmentationOverlaySparseChunkTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Sparse Chunk Test Image")
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
            confidence_score=None,
            features={"sam_score": 0.9},
        )

    def test_missing_sparse_chunk_returns_zero_filled_bytes(self):
        state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        store_root = get_overlay_active_bundle_path(state)
        # The only object is at (10..22, 10..22); chunk (cy=2, cx=0) covers
        # pixels y=512..768 and is guaranteed never written.
        missing_chunk_path = store_root / "labels" / "0" / "2.0"
        self.assertFalse(missing_chunk_path.exists())

        response = self.client.get(
            f"/segmentation-overlays/{self.segmentation.id}.zarr/labels/0/2.0?rev=0"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/octet-stream")
        self.assertEqual(
            response["Cache-Control"],
            "public, max-age=31536000, immutable",
        )

        labels0 = zarr.open_array(str(store_root / "labels" / "0"), mode="r")
        chunk_h = min(256, int(labels0.shape[0]))
        chunk_w = min(256, int(labels0.shape[1]))
        # Decode using the same codec the labels array was created with
        # (Blosc zstd, level 5, byte-shuffle) and assert all-background zeros.
        decoded = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE).decode(response.content)
        chunk = np.frombuffer(decoded, dtype=np.uint32).reshape((chunk_h, chunk_w))
        self.assertEqual(int(chunk.max()), 0)

    def test_missing_sparse_chunk_matches_encode_zero_chunk(self):
        state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        store_root = get_overlay_active_bundle_path(state)
        labels0 = zarr.open_array(str(store_root / "labels" / "0"), mode="r")
        chunk_h = min(256, int(labels0.shape[0]))
        chunk_w = min(256, int(labels0.shape[1]))

        response = self.client.get(
            f"/segmentation-overlays/{self.segmentation.id}.zarr/labels/0/2.0?rev=0"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.content,
            overlay_ngff.encode_zero_chunk("labels", (chunk_h, chunk_w)),
        )


class SegmentationOverlayRasterizationTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Overlay Rasterization Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _create_segment(
        self,
        polygon: Polygon,
        *,
        label_state: str = "CONFIRMED",
        source_model: str = "manual",
    ) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            source_model=source_model,
            confidence_score=0.8 if label_state != "CONFIRMED" else None,
            features={"sam_score": 0.9},
        )

    def test_source_overlay_renders_the_selected_model_and_hand_drawn_work(self):
        # A model bundle carries this model's objects plus the hand-drawn ones.
        # Another model's objects -- confirmed or not -- live in the separate
        # source-less display bundle, which is what keeps a confirmation a
        # LUT-only update. Hand-drawn objects cannot be moved out with them:
        # owner ruling R13 forbids hiding what the user annotated, and the
        # candidate layer that reads this bundle is the only layer that paints
        # an outline the user drew but has not confirmed.
        active_candidate = self._create_segment(
            Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10))),
            label_state="CANDIDATE",
            source_model="quantem:mito",
        )
        other_candidate = self._create_segment(
            Polygon(((40, 10), (52, 10), (52, 22), (40, 22), (40, 10))),
            label_state="CANDIDATE",
            source_model="omniem:mito",
        )
        manual_candidate = self._create_segment(
            Polygon(((70, 10), (82, 10), (82, 22), (70, 22), (70, 10))),
            label_state="CANDIDATE",
            source_model="manual",
        )
        confirmed_other_source = self._create_segment(
            Polygon(((100, 10), (112, 10), (112, 22), (100, 22), (100, 10))),
            label_state="CONFIRMED",
            source_model="omniem:mito",
        )

        state = rebuild_overlay_full(
            self.segmentation, source_model="quantem:mito", desired_revision=0
        )
        labels0 = _open_labels_level0(state)

        # The selected model and the hand-drawn object are painted. The other
        # model's two objects never receive labels in this bundle.
        self.assertEqual(
            _label_value_at(labels0, 15, 15),
            _label_for_object(state, active_candidate.id),
        )
        self.assertEqual(_label_value_at(labels0, 15, 45), 0)
        self.assertEqual(
            _label_value_at(labels0, 15, 75),
            _label_for_object(state, manual_candidate.id),
        )
        self.assertEqual(_label_value_at(labels0, 15, 105), 0)
        for omitted in (other_candidate, confirmed_other_source):
            self.assertFalse(
                SegmentationOverlayLabel.objects.filter(
                    overlay_state=state, object_uuid=omitted.id
                ).exists()
            )

    def test_touching_segments_keep_visible_border_channel(self):
        left = self._create_segment(Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10))))
        right = self._create_segment(Polygon(((22, 10), (34, 10), (34, 22), (22, 22), (22, 10))))

        state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        labels0 = _open_labels_level0(state)
        border0 = _open_border_level0(state)

        # Each touching object keeps its own distinct dense label...
        left_label = _label_for_object(state, left.id)
        right_label = _label_for_object(state, right.id)
        self.assertNotEqual(left_label, right_label)
        self.assertEqual(_label_value_at(labels0, 15, 15), left_label)
        self.assertEqual(_label_value_at(labels0, 15, 28), right_label)
        # ...and the shared seam is baked into the border mask.
        self.assertGreater(_border_max(border0, 15, 16, 21, 24), 0)

        bundle_root = get_overlay_active_bundle_path(state)
        self.assertTrue((bundle_root / ".zattrs").exists())
        self.assertTrue((bundle_root / ".zgroup").exists())

    def test_extract_polygons_ignores_non_polygon_iterables(self):
        geometry = shapely_wkt.loads(
            "GEOMETRYCOLLECTION(POLYGON((0 0, 4 0, 4 4, 0 4, 0 0)),LINESTRING(4 0, 8 0))"
        )

        polygons = extract_polygons(geometry)

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].geom_type, "Polygon")

    def test_polygon_holes_render_empty_fill_and_interior_border(self):
        outer = Polygon(((10, 10), (40, 10), (40, 40), (10, 40), (10, 10)))
        inner = Polygon(((20, 20), (30, 20), (30, 30), (20, 30), (20, 20)))
        polygon = outer.difference(inner)
        obj = self._create_segment(polygon)

        state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        labels0 = _open_labels_level0(state)
        border0 = _open_border_level0(state)

        label = _label_for_object(state, obj.id)
        # Interior of the hole is background (no label painted there)...
        self.assertEqual(_label_value_at(labels0, 25, 25), 0)
        # ...the ring wall carries the object's label...
        self.assertEqual(_label_value_at(labels0, 15, 25), label)
        # ...and the hole boundary is baked into the border mask.
        self.assertGreater(_border_max(border0, 19, 22, 24, 27), 0)


class SegmentationOverlayDownsampleTests(TestCase):
    def test_labels_use_mode_pooling_while_border_uses_max_pooling(self):
        # Labels never average: a 2x2 block mode-pools to the most frequent
        # non-zero id (here, three 5s beat one background).
        label_block = np.array([[5, 5], [5, 0]], dtype=np.uint32)
        pooled_labels = overlay_ngff.render.mode_downsample_2x2(label_block)
        self.assertEqual(pooled_labels.shape, (1, 1))
        self.assertEqual(int(pooled_labels[0, 0]), 5)

        # The border mask max-pools: a block is border if any child is.
        border_block = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        pooled_border = overlay_ngff.render.max_downsample_2x2(border_block)
        self.assertEqual(pooled_border.shape, (1, 1))
        self.assertEqual(int(pooled_border[0, 0]), 1)

    def test_mode_pooling_ties_resolve_to_smaller_id(self):
        # Two distinct ids each appear twice; the smaller id wins the tie.
        tie_block = np.array([[7, 7], [3, 3]], dtype=np.uint32)
        pooled = overlay_ngff.render.mode_downsample_2x2(tie_block)
        self.assertEqual(int(pooled[0, 0]), 3)

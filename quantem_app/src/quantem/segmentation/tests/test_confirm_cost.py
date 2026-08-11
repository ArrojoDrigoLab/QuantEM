"""What one answer costs: the confirm/reject request must not touch the raster.

Proofreading is a keypress rhythm. Every kept/removed answer used to rasterise a
region of the overlay **inside the HTTP request** -- open the zarr store, paint
the dirty tile, rewrite every pyramid level above it, close the store -- so the
click that should have cost a database write cost a disk round-trip, and eight
reviewers' worth of clicks arriving at once cost an HTTP 500 apiece when two
writes to the same store collided on Windows.

These are the acceptance tests for that: the request defers the raster, the
deferred raster still happens, and concurrent answers do not fail.
"""

from __future__ import annotations

import threading
import time
from unittest import skipIf
from unittest.mock import patch

import numpy as np
from django.db import connection, connections
from django.test import TestCase, TransactionTestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationOverlayState,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff import mutations
from quantem.segmentation.overlay_ngff.dirty import merge_dirty_bboxes
from quantem.segmentation.overlay_ngff.mutations import rebuild_overlay_full
from quantem.segmentation.overlay_ngff.paths import OverlayStoreError
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff

#: UX_PLAN section 6: p95 of one answer, measured at the endpoint.
ONE_ANSWER_P95_BUDGET_SECONDS = 0.080
#: UX_PLAN section 6: 100 objects reviewed must not cost more than this in total.
HUNDRED_ANSWERS_BUDGET_SECONDS = 5.0


def _square(x: int, y: int, size: int = 12) -> Polygon:
    return Polygon(
        (
            (x, y),
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
            (x, y),
        )
    )


def _make_segment(segmentation: ImageSegmentation, polygon: Polygon) -> SegmentObject:
    return SegmentObject.objects.create(
        segmentation=segmentation,
        geometry=polygon,
        centroid=polygon.centroid,
        bbox=polygon.envelope,
        label_state="CANDIDATE",
        confidence_score=0.8,
        features={"sam_score": 0.9},
    )


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


class ConfirmDefersTheRasterTests(TestCase):
    """One answer registers the edit and queues it. It does not paint it."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Confirm Cost Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.segments = [
            _make_segment(self.segmentation, _square(10 + 40 * idx, 10 + 40 * idx))
            for idx in range(4)
        ]
        self.segment = self.segments[0]
        rebuild_overlay_full(self.segmentation, desired_revision=0)

    def _settled_state(self) -> SegmentationOverlayState:
        return SegmentationOverlayState.objects.get(
            segmentation=self.segmentation, candidate_source_model=""
        )

    def test_a_batch_answer_defers_the_raster_instead_of_writing_it(self):
        state = self._settled_state()
        self.assertEqual(state.applied_revision, state.desired_revision)

        with patch.object(
            mutations,
            "apply_partial_overlay_update",
            side_effect=AssertionError(
                "the request rasterised the overlay instead of deferring it"
            ),
        ):
            response = self.client.post(
                "/api/segments/labels/batch/",
                {"labels": [{"id": str(self.segment.id), "label_state": "CONFIRMED"}]},
                format="json",
            )

        self.assertEqual(response.status_code, 200)
        overlay = response.data["overlays"][str(self.segmentation.id)]
        self.assertEqual(overlay["rebuild_mode"], "async_partial")
        self.assertFalse(overlay["sync_applied"])

    def test_the_single_segment_answer_defers_the_raster_too(self):
        with patch.object(
            mutations,
            "apply_partial_overlay_update",
            side_effect=AssertionError(
                "the request rasterised the overlay instead of deferring it"
            ),
        ):
            response = self.client.post(
                f"/api/segments/{self.segment.id}/label/",
                {"label_state": "CONFIRMED"},
                format="json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["overlay"]["rebuild_mode"], "async_partial")
        self.assertFalse(response.data["overlay"]["sync_applied"])

    def test_clearing_manual_labels_defers_the_raster_too(self):
        self.client.post(
            "/api/segments/labels/batch/",
            {"labels": [{"id": str(self.segment.id), "label_state": "CONFIRMED"}]},
            format="json",
        )
        mutations.run_overlay_rebuild_job(self.segmentation, mode="partial")

        with patch.object(
            mutations,
            "apply_partial_overlay_update",
            side_effect=AssertionError(
                "the request rasterised the overlay instead of deferring it"
            ),
        ):
            response = self.client.post(
                f"/api/segmentations/{self.segmentation.id}/labels/clear",
                {},
                format="json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["overlay"]["rebuild_mode"], "async_partial")

    def test_the_deferred_edit_is_queued_and_then_applied(self):
        """Deferring is not dropping. The queued rebuild must settle the bundle."""
        response = self.client.post(
            "/api/segments/labels/batch/",
            {"labels": [{"id": str(self.segment.id), "label_state": "CONFIRMED"}]},
            format="json",
        )
        overlay = response.data["overlays"][str(self.segmentation.id)]
        desired = overlay["desired_revision"]

        state = self._settled_state()
        self.assertEqual(state.status, SegmentationOverlayState.STATUS_DIRTY)
        self.assertLess(state.applied_revision, desired)
        self.assertTrue(state.dirty_chunk_runs)

        mutations.run_overlay_rebuild_job(self.segmentation, mode="partial")

        state = self._settled_state()
        self.assertEqual(state.status, SegmentationOverlayState.STATUS_READY)
        self.assertEqual(state.applied_revision, desired)
        self.assertEqual(state.dirty_chunk_runs, [])

    def test_a_burst_of_answers_leaves_nothing_stranded(self):
        """Every answer of a burst reaches the raster, however they coalesced.

        Deferring makes the queue do the work, and the queue deduplicates: an
        answer arriving while a rebuild is already queued adds its dirty region
        to the pending set rather than a second job. What must not happen is an
        answer whose region is recorded and never painted, so this drives the
        loop the running system drives -- run whatever is queued, ask the
        manifest, repeat -- and requires it to settle.
        """
        def _answer(segment) -> None:
            response = self.client.post(
                "/api/segments/labels/batch/",
                {"labels": [{"id": str(segment.id), "label_state": "CONFIRMED"}]},
                format="json",
            )
            self.assertEqual(response.status_code, 200)

        _answer(self.segments[0])
        _answer(self.segments[1])

        # The tail case, forced rather than hoped for: an answer given while the
        # rebuild is already RUNNING finds an active job for its bundle, so it
        # adds a dirty region and enqueues nothing. When that job finishes it
        # has no knowledge of the region that arrived after it started, and the
        # last answer of a burst is exactly the one that lands there.
        job = mutations.overlay_jobs_for_bundle(str(self.segmentation.id)).get()
        job.status = "RUNNING"
        job.save(update_fields=["status"])

        real_apply = mutations.apply_partial_overlay_update

        def _answer_after_the_job_read_its_work(*args, **kwargs):
            # The handler has already taken its snapshot of the dirty regions
            # and the revision to apply; this answer is not in it.
            _answer(self.segments[2])
            return real_apply(*args, **kwargs)

        with patch.object(
            mutations,
            "apply_partial_overlay_update",
            side_effect=_answer_after_the_job_read_its_work,
        ):
            mutations.run_overlay_rebuild_job(self.segmentation, mode="partial")
        job.status = "COMPLETED"
        job.save(update_fields=["status"])

        stranded = self._settled_state()
        self.assertEqual(stranded.status, SegmentationOverlayState.STATUS_DIRTY)
        self.assertFalse(
            mutations.overlay_jobs_for_bundle(str(self.segmentation.id))
            .filter(status__in=["PENDING", "RUNNING", "RETRY"])
            .exists()
        )

        for _ in range(10):
            state = self._settled_state()
            if (
                state.status == SegmentationOverlayState.STATUS_READY
                and state.applied_revision == state.desired_revision
                and not state.dirty_chunk_runs
            ):
                break
            job = (
                mutations.overlay_jobs_for_bundle(str(self.segmentation.id))
                .filter(status__in=["PENDING", "RUNNING", "RETRY"])
                .first()
            )
            if job is None:
                # Nothing queued but work outstanding: the manifest endpoint is
                # the requeue path, and the viewer asks it on every poll.
                self.client.get(
                    f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
                )
                continue
            job.status = "RUNNING"
            job.save(update_fields=["status"])
            mutations.run_overlay_rebuild_job(
                self.segmentation, mode=job.payload_json.get("mode", "partial")
            )
            job.status = "COMPLETED"
            job.save(update_fields=["status"])

        state = self._settled_state()
        self.assertEqual(state.status, SegmentationOverlayState.STATUS_READY)
        self.assertEqual(state.applied_revision, state.desired_revision)
        self.assertEqual(state.dirty_chunk_runs, [])

    def test_a_geometry_edit_still_takes_the_synchronous_path(self):
        """The deferral is scoped to answers, not to every overlay mutation.

        A drawn or reshaped outline changes pixels the user is looking at right
        now, and it is a single deliberate gesture rather than a rhythm, so it
        keeps the synchronous partial write and its instantly-correct overlay.
        """
        result = mutations.register_overlay_mutation(
            self.segmentation,
            dirty_bbox=merge_dirty_bboxes(self.segmentation, [self.segment.geometry]),
        )
        self.assertEqual(result["rebuild_mode"], "sync_partial")
        self.assertTrue(result["sync_applied"])


class ConfirmCostBudgetTests(TestCase):
    """The measured budgets from UX_PLAN section 6, as a regression test."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Confirm Budget Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.segments = [
            _make_segment(self.segmentation, _square(16 * (idx % 60), 16 * (idx // 60)))
            for idx in range(100)
        ]
        rebuild_overlay_full(self.segmentation, desired_revision=0)

    def test_one_answer_costs_under_eighty_milliseconds_and_a_hundred_under_five_seconds(
        self,
    ):
        durations: list[float] = []
        for segment in self.segments:
            started = time.perf_counter()
            response = self.client.post(
                "/api/segments/labels/batch/",
                {"labels": [{"id": str(segment.id), "label_state": "CONFIRMED"}]},
                format="json",
            )
            durations.append(time.perf_counter() - started)
            self.assertEqual(response.status_code, 200)

        p95 = _percentile(durations, 0.95)
        total = sum(durations)
        # Printed so a run with -s reports the headroom rather than only
        # pass/fail: these are the two numbers UX_PLAN section 6 tracks, and a
        # regression that halves the margin is worth seeing before it fails.
        print(
            f"\n[confirm cost] p95 {p95 * 1000:.1f} ms, "
            f"median {_percentile(durations, 0.5) * 1000:.1f} ms, "
            f"{len(durations)} answers in {total:.2f} s"
        )
        self.assertLessEqual(
            p95,
            ONE_ANSWER_P95_BUDGET_SECONDS,
            f"p95 of one answer was {p95 * 1000:.1f} ms "
            f"(budget {ONE_ANSWER_P95_BUDGET_SECONDS * 1000:.0f} ms); "
            f"total for {len(durations)} answers {total:.2f} s",
        )
        self.assertLessEqual(
            total,
            HUNDRED_ANSWERS_BUDGET_SECONDS,
            f"{len(durations)} answers cost {total:.2f} s of server work "
            f"(budget {HUNDRED_ANSWERS_BUDGET_SECONDS:.0f} s)",
        )


class BundleWriteLockTests(TestCase):
    """Two writers of one bundle's live store must not interleave.

    This is the mechanism behind the 29 % HTTP 500 rate the deferral removes:
    an in-place partial update is a read-modify-write of the active zarr store,
    and two of them at once interleave chunk writes until Windows refuses a
    colliding rename inside zarr's atomic write. Deferring answers means far
    fewer writers, but not *one*: a background rebuild and a drawn geometry edit
    can still meet, so the store write itself has to be exclusive.
    """

    CONCURRENCY = 8

    def setUp(self):
        self.image = create_image_from_test_tiff("Bundle Write Lock Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def test_only_one_writer_of_a_bundle_is_inside_the_lock_at_a_time(self):
        """Mutual exclusion, asserted directly rather than inferred from luck.

        A race test that only ever asserts "nothing raised" passes for the wrong
        reason on a machine that happens not to interleave. This counts.
        """
        inside = 0
        peak = 0
        counter_guard = threading.Lock()
        barrier = threading.Barrier(self.CONCURRENCY)
        errors: list[BaseException] = []

        def _hold_the_lock() -> None:
            nonlocal inside, peak
            try:
                barrier.wait(timeout=30)
                with mutations.bundle_write_lock(str(self.segmentation.id)):
                    with counter_guard:
                        inside += 1
                        peak = max(peak, inside)
                    time.sleep(0.01)
                    with counter_guard:
                        inside -= 1
            except BaseException as exc:  # noqa: BLE001 - recorded, then asserted
                errors.append(exc)

        threads = [
            threading.Thread(target=_hold_the_lock) for _ in range(self.CONCURRENCY)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertEqual(inside, 0)
        self.assertEqual(peak, 1, f"{peak} writers were inside the bundle lock at once")

    def test_the_lock_is_the_operating_system_s_and_not_only_this_process_s(self):
        """The job worker is a *spawned process*; a thread lock cannot see it.

        Two independent handles on the same file is the cheapest honest test of
        that: both ``msvcrt.locking`` and ``flock`` conflict between separate
        opens, in one process or across several, so this fails the moment the
        OS lock is dropped in favour of the in-process one.
        """
        lock_path = (
            mutations.get_overlay_root(str(self.segmentation.id))
            / mutations.BUNDLE_WRITE_LOCK_FILENAME
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as first, open(lock_path, "a+b") as second:
            self.assertTrue(mutations._try_lock_file(first))
            self.assertFalse(mutations._try_lock_file(second))
            mutations._unlock_file(first)
            self.assertTrue(mutations._try_lock_file(second))
            mutations._unlock_file(second)

    def test_the_aggregate_and_a_per_source_bundle_do_not_block_each_other(self):
        """Separate stores, separate locks. Serialising them would be pure waiting."""
        with mutations.bundle_write_lock(str(self.segmentation.id)):
            with mutations.bundle_write_lock(str(self.segmentation.id), "quantem_mito"):
                pass

    def test_a_bundle_held_too_long_is_reported_not_waited_on_forever(self):
        holder_ready = threading.Event()
        release_holder = threading.Event()
        errors: list[BaseException] = []

        def _hold() -> None:
            try:
                with mutations.bundle_write_lock(str(self.segmentation.id)):
                    holder_ready.set()
                    release_holder.wait(timeout=30)
            except BaseException as exc:  # noqa: BLE001 - recorded, then asserted
                errors.append(exc)

        holder = threading.Thread(target=_hold)
        holder.start()
        try:
            self.assertTrue(holder_ready.wait(timeout=30))
            with self.assertRaises(OverlayStoreError) as caught:
                with mutations.bundle_write_lock(str(self.segmentation.id), timeout=0.1):
                    pass
        finally:
            release_holder.set()
            holder.join(timeout=30)
        self.assertEqual(errors, [])
        message = str(caught.exception)
        self.assertIn("still being written", message)
        # A sentence for a person: no path, no identifier, nothing internal.
        self.assertNotIn(str(self.segmentation.id), message)


class TileWriteRetryTests(TestCase):
    """A transient Windows sharing violation must not fail the write."""

    def test_a_locked_chunk_file_is_retried_rather_than_raised(self):
        attempts = {"labels": 0}

        class _FlakyArray:
            def __init__(self, fail_times: int):
                self.fail_times = fail_times
                self.writes: list[tuple] = []

            def __setitem__(self, key, value):
                attempts["labels"] += 1
                if self.fail_times > 0:
                    self.fail_times -= 1
                    raise PermissionError(5, "Access is denied")
                self.writes.append((key, value.shape))

        labels_array = _FlakyArray(fail_times=2)
        border_array = _FlakyArray(fail_times=0)
        arrays = {"labels": [labels_array], "border": [border_array]}
        crop = np.zeros((4, 4), dtype=np.uint32)
        border = np.zeros((4, 4), dtype=np.uint8)

        mutations._write_tile_result(arrays, (8, 12, crop, border))

        self.assertEqual(len(labels_array.writes), 1)
        self.assertEqual(len(border_array.writes), 1)
        self.assertEqual(labels_array.writes[0][0], (slice(12, 16), slice(8, 12)))

    def test_a_chunk_that_never_unlocks_still_fails(self):
        class _AlwaysLockedArray:
            def __setitem__(self, key, value):
                raise PermissionError(5, "Access is denied")

        arrays = {"labels": [_AlwaysLockedArray()], "border": [_AlwaysLockedArray()]}
        with self.assertRaises(PermissionError):
            mutations._write_tile_result(
                arrays,
                (0, 0, np.zeros((2, 2), dtype=np.uint32), np.zeros((2, 2), dtype=np.uint8)),
            )


@skipIf(
    connection.settings_dict.get("TEST", {}).get("NAME") is None,
    "needs a file-backed test database: Django's default SQLite test database "
    "is an in-memory one with a shared cache, whose table-level SQLITE_LOCKED "
    "is not what the shipped WAL file does and is not covered by busy_timeout. "
    'Run with a plugin that sets DATABASES["default"]["TEST"]["NAME"].',
)
class ConcurrentConfirmTests(TransactionTestCase):
    """Eight answers arriving at once against a settled overlay: no 500s.

    The acceptance criterion of this package, at the endpoint. Measured at 7
    failures in 24 attempts (29 %) before it: every answer took the synchronous
    partial write, so eight of them opened and rewrote the same zarr store at
    the same time and Windows refused the colliding rename.
    """

    TRIALS = 5
    CONCURRENCY = 8

    def _post_one(self, segment_id: str, barrier: threading.Barrier, results: list):
        try:
            barrier.wait(timeout=30)
            client = APIClient(raise_request_exception=False)
            response = client.post(
                "/api/segments/labels/batch/",
                {"labels": [{"id": segment_id, "label_state": "CONFIRMED"}]},
                format="json",
            )
            results.append(response.status_code)
        except Exception as exc:  # noqa: BLE001 - recorded, then asserted on
            results.append(exc)
        finally:
            connections.close_all()

    def test_eight_simultaneous_answers_produce_no_server_errors(self):
        failures: list[object] = []
        for trial in range(self.TRIALS):
            image = create_image_from_test_tiff(f"Concurrent Confirm Image {trial}")
            segmentation = ImageSegmentation.objects.create(
                asset=image.asset,
                segmentation_type=get_or_create_mitochondria_type(),
            )
            segments = [
                _make_segment(segmentation, _square(40 * idx, 40 * idx))
                for idx in range(self.CONCURRENCY)
            ]
            rebuild_overlay_full(segmentation, desired_revision=0)

            barrier = threading.Barrier(self.CONCURRENCY)
            results: list = []
            threads = [
                threading.Thread(
                    target=self._post_one,
                    args=(str(segment.id), barrier, results),
                )
                for segment in segments
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)

            self.assertEqual(len(results), self.CONCURRENCY)
            failures.extend(
                outcome
                for outcome in results
                if not (isinstance(outcome, int) and outcome == 200)
            )

        self.assertEqual(
            failures,
            [],
            f"{len(failures)} of {self.TRIALS * self.CONCURRENCY} simultaneous "
            f"answers did not return 200: {failures}",
        )

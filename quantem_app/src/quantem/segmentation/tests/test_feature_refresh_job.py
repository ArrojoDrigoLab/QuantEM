"""``recompute_features`` was a documented invalidation that nothing performed.

``services/confirm_batch/feature_refresh.py`` documented the flag as marking
edits that "invalidate segmentation-level aggregates ... even when no individual
segment ids are supplied". ``jobs/handlers.py`` never read it, and both
label-change call sites enqueue ``segment_ids=[]``, so with the trigger on every
label flip queued a job whose payload looped zero times and then reported
*"segment feature refresh complete"* at 100%.

What a label flip actually invalidates is not any one outline -- the geometry
did not move -- but *which* objects the analysis aggregates over. An object that
was never measured contributes blank columns to ``objects.csv`` the moment it
joins that population, so that is what the flag now performs: a sweep for
unmeasured objects. Normally it finds none, and the job says so instead of
claiming a refresh it did not do.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.jobs.constants import JOB_TYPE_REFRESH_SEGMENT_FEATURES
from quantem.jobs.handlers import handle_refresh_segment_features
from quantem.jobs.models import Job
from quantem.segmentation.features.measure import MEASUREMENT_KEYS
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256


class _UnitScope:
    """Records one countable phase, as :class:`UnitProgressScope` would write it.

    Faithful to the real scope's *semantics* rather than its storage: opening it
    records ``0 of total`` (the real one writes the denominator immediately,
    which is what puts "0 of 412 objects" on screen before the first
    measurement), :meth:`set` clamps to the total and holds rather than moving
    backwards, :meth:`advance` is ``set(done + count)``, and :meth:`finish` is
    idempotent and closes the scope against later writes.

    The one deliberate difference is the real scope's wall-clock write floor
    (``UNIT_WRITE_MIN_INTERVAL_SECONDS``, one second). A sweep over a handful of
    small objects here finishes in milliseconds, so a double that copied the
    throttle would retain only the opening and closing samples and could not
    tell a counter that advanced per object from one that jumped straight to the
    end. ``samples`` is therefore every count the handler *reported*, not every
    row the real scope would have written -- the throttle is a database-cost
    policy, tested in ``quantem.jobs.tests.test_progress_units``, not part of
    the contract a handler is holding up.
    """

    def __init__(
        self,
        *,
        total: int,
        label: str,
        stage: str | None = None,
        detail: dict | None = None,
    ) -> None:
        self.label = str(label)
        self.stage = stage
        self.detail = dict(detail or {})
        self.total = max(int(total), 0)
        self.done = 0
        self.closed = False
        self.finished = False
        self.samples: list[tuple[int, int]] = [(self.done, self.total)]

    def set(self, done: int, *, total: int | None = None) -> None:
        if self.closed:
            return
        if total is not None:
            self.total = max(int(total), 0)
        done = max(0, int(done))
        if self.total:
            done = min(done, self.total)
        if done < self.done:
            # The real scope logs and holds the larger count rather than failing
            # the run over a stale report, so the double must not record it
            # either -- otherwise a handler regression would look like progress.
            return
        self.done = done
        self.samples.append((self.done, self.total))

    def advance(self, count: int = 1) -> None:
        self.set(self.done + max(int(count), 0))

    def finish(self) -> None:
        if self.closed:
            return
        self.samples.append((self.done, self.total))
        self.closed = True
        self.finished = True

    def __enter__(self) -> _UnitScope:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # A failed run must not be left claiming its last successful object was
        # the last one there was, so only success writes the final count.
        if exc_type is None:
            self.finish()
        else:
            self.closed = True


class _Reporter:
    """Records what a job would have shown the user.

    Carries the queue reporter's whole surface, not only ``update``: v0.1.6 gave
    the sweep a per-object unit scope, and a double that quietly lacked
    ``unit_scope`` turned that into an ``AttributeError`` at the first measured
    object -- i.e. this stand-in was the only thing standing between the release
    and a handler that raises on its own progress reporting.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.progress: float = 0.0
        self.stages: list[str] = []
        self.scopes: list[_UnitScope] = []

    def update(
        self,
        progress: float | None = None,
        message: str | None = None,
        *,
        stage: str | None = None,
        detail: dict | None = None,
        **_ignored,
    ) -> None:
        if progress is not None:
            self.progress = progress
        if message:
            self.messages.append(message)
        if stage is not None:
            self.stages.append(stage)

    def unit_scope(
        self,
        *,
        total: int,
        label: str,
        stage: str | None = None,
        detail: dict | None = None,
        **_ignored,
    ) -> _UnitScope:
        scope = _UnitScope(total=total, label=label, stage=stage, detail=detail)
        self.scopes.append(scope)
        if stage is not None:
            self.stages.append(stage)
        return scope

    def log(self, level: str, message: str) -> None:
        self.messages.append(message)


class _Cancel:
    def check_cancelled(self) -> None:
        return None


class FeatureRefreshSweepTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Feature refresh", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _segment(self, *, features: dict) -> SegmentObject:
        polygon = Polygon(((40, 40), (120, 40), (120, 120), (40, 120), (40, 40)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=None,
            features=features,
        )

    def _unmeasured_boxes(self, count: int) -> list[SegmentObject]:
        """``count`` never-measured objects, side by side and non-overlapping."""
        segments = []
        for index in range(count):
            left = 140 + 12 * index
            polygon = Polygon(((left, 20), (left + 8, 20), (left + 8, 40), (left, 40), (left, 20)))
            segments.append(
                SegmentObject.objects.create(
                    segmentation=self.segmentation,
                    geometry=polygon,
                    centroid=polygon.centroid,
                    bbox=polygon.envelope,
                    label_state="CONFIRMED",
                    features={},
                )
            )
        return segments

    def _run(self, payload: dict) -> tuple[dict, _Reporter]:
        reporter = _Reporter()
        result = handle_refresh_segment_features(payload, reporter, _Cancel())
        return result, reporter

    def test_the_flag_measures_an_object_that_was_never_measured(self):
        unmeasured = self._segment(features={"mito_generated": True})

        result, _reporter = self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [],
                "recompute_features": True,
            }
        )

        self.assertEqual(result["segment_count"], 1)
        self.assertTrue(result["swept_segmentation"])
        unmeasured.refresh_from_db()
        for key in MEASUREMENT_KEYS:
            self.assertIn(key, unmeasured.features)
        # And provenance is still there afterwards.
        self.assertIs(unmeasured.features["mito_generated"], True)

    def test_the_sweep_counts_objects_one_at_a_time_rather_than_in_one_jump(self):
        """A user watching a sweep is owed "3 of 4 objects", not a frozen bar.

        A sweep after a deferred Preview covers every object in the
        segmentation, so the only honest thing to show is a countable
        denominator up front and a count that moves as objects are measured.
        That is what the unit scope is for, and what a percentage read backwards
        out of ``message`` could never be.
        """
        objects = self._unmeasured_boxes(4)

        result, reporter = self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [],
                "recompute_features": True,
            }
        )

        self.assertEqual(result["segment_count"], len(objects))
        self.assertEqual(len(reporter.scopes), 1)
        scope = reporter.scopes[0]
        self.assertEqual(scope.label, "object")
        self.assertEqual(scope.stage, "measurement")
        # The denominator is the objects being measured, and it is on the row
        # from the moment the scope opens -- including its very first sample,
        # before anything has been measured.
        self.assertEqual(scope.total, len(objects))
        self.assertEqual(scope.samples[0], (0, len(objects)))
        self.assertTrue(all(total == len(objects) for _done, total in scope.samples))

        counts = [done for done, _total in scope.samples]
        self.assertEqual(counts, sorted(counts), f"unit progress went backwards: {counts}")
        # Every object individually, not one jump from 0 to 4: each count in
        # between was reported, which is exactly what batching the SQL was not
        # allowed to cost.
        self.assertEqual(sorted(set(counts)), list(range(len(objects) + 1)))
        self.assertEqual(scope.done, len(objects))
        self.assertTrue(scope.finished, "the scope was left open, so the count never settled")

    def test_a_sweep_that_finds_nothing_says_so_rather_than_claiming_a_refresh(self):
        self._segment(features={"area": 6400.0})

        result, reporter = self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [],
                "recompute_features": True,
            }
        )

        self.assertEqual(result["segment_count"], 0)
        self.assertTrue(result["swept_segmentation"])
        self.assertIn("already measured", reporter.messages[-1])
        self.assertNotIn("segment feature refresh complete", reporter.messages)

    def test_a_sweep_leaves_measured_objects_alone(self):
        measured = self._segment(features={"area": 1.0, "mean_prob": 0.82})

        self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [],
                "recompute_features": True,
            }
        )

        measured.refresh_from_db()
        self.assertEqual(measured.features["area"], 1.0)

    def test_explicit_ids_still_win_over_the_sweep(self):
        named = self._segment(features={"area": 1.0})

        result, _reporter = self._run(
            {
                "segmentation_id": str(self.segmentation.id),
                "segment_ids": [str(named.id)],
                "recompute_features": True,
            }
        )

        self.assertEqual(result["segment_count"], 1)
        self.assertFalse(result["swept_segmentation"])
        named.refresh_from_db()
        self.assertAlmostEqual(named.features["area"], 80 * 80, delta=400)

    @patch.dict(os.environ, {"QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS": "1"})
    def test_a_label_flip_queues_a_job_that_has_something_to_do(self):
        unmeasured = self._segment(features={})
        response = self.client.post(
            f"/api/segments/{unmeasured.id}/label/",
            {"label_state": "EXCLUDED"},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)

        job = Job.objects.filter(type=JOB_TYPE_REFRESH_SEGMENT_FEATURES).get()
        self.assertTrue(job.payload_json["recompute_features"])

        result, _reporter = self._run(job.payload_json)
        self.assertEqual(result["segment_count"], 1)

    @patch.dict(os.environ, {"QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS": "1"})
    def test_a_proofreading_session_does_not_pile_up_identical_sweeps(self):
        """Hundreds of label flips are hundreds of requests for one sweep."""
        segments = []
        for index in range(4):
            offset = 10 * index
            polygon = Polygon(
                (
                    (140 + offset, 20),
                    (160 + offset, 20),
                    (160 + offset, 40),
                    (140 + offset, 40),
                    (140 + offset, 20),
                )
            )
            segments.append(
                SegmentObject.objects.create(
                    segmentation=self.segmentation,
                    geometry=polygon,
                    centroid=polygon.centroid,
                    bbox=polygon.envelope,
                    label_state="CANDIDATE",
                    features={"area": 400.0},
                )
            )

        for segment in segments:
            response = self.client.post(
                f"/api/segments/{segment.id}/label/",
                {"label_state": "CONFIRMED"},
                format="json",
            )
            self.assertEqual(response.status_code, 200, response.data)

        self.assertEqual(Job.objects.filter(type=JOB_TYPE_REFRESH_SEGMENT_FEATURES).count(), 1)

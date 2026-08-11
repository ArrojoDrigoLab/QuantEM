"""A queued run knows how big it is, and the wave counts it.

The defect this file pins was measured on a real image on 2026-08-10. Three
whole-image runs were started on one 2892x2508 montage at 8 nm/px -- mitochondria
(56 tiles), nucleus (6), endoplasmic reticulum (56), **118 tiles of requested
work**. The wire said::

    units_done 19, units_reachable 19, percent 100.0, runs_total 1

and the Tasks drawer said **"Everything on montage16real  100%  25 of 25
tiles · 1 of 2 did not finish"** -- while the third run had not started and
would fail.

Two things were wrong and both are tested here.

1. ``progress_units_total`` was first written *by the run itself*, and
   :func:`~quantem.jobs.serializers.aggregate_batch_progress` counted only jobs
   that had it. A queued run therefore contributed nothing: not to the
   denominator, not to ``runs_total``, not to the failed/cancelled counts. It
   is now written at enqueue by
   :func:`quantem.jobs.tile_plan.planned_units_for`.
2. A failed or cancelled run's unwalked tiles were *removed* from the
   denominator, so the bar reached 100 % on a wave that ran a fifth of itself.
   They stay in it now.
"""

from __future__ import annotations

from uuid import uuid4

from django.test import TestCase

from quantem.assets.models import Asset, ImageROI, Rendition
from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    QUEUE_P2_UPLOAD,
    QUEUE_P3_ROI,
    QUEUE_P4_FULL,
)
from quantem.jobs.models import STAGE_QUEUED, UNIT_TILE, Job
from quantem.jobs.serializers import batch_progress_for
from quantem.jobs.tile_plan import planned_units_for
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_mitochondria_type,
    get_or_create_nucleus_type,
)

#: The montage the finding was measured on: sixteen real 723x627 TEM crops at
#: 8.0 nm/px, tiled 4x4.
MONTAGE_WIDTH = 2892
MONTAGE_HEIGHT = 2508
MONTAGE_NM = 8.0

#: The tiling plans those three models lay out over that montage, computed
#: independently in ``w0c_verify_report.md`` from the published window geometry
#: (stride = round(tile x (1 - overlap)), starts range(0, L - tile + 1, stride)
#: plus a flush window, region padded to a whole number of patches).
MITO_TILES = 56
NUCLEUS_TILES = 6
ER_TILES = 56
WAVE_TILES = MITO_TILES + NUCLEUS_TILES + ER_TILES  # 118


def _montage_asset() -> Asset:
    asset = Asset.objects.create(
        display_name="montage16real",
        original_filename="montage16real.png",
        logical_width=MONTAGE_WIDTH,
        logical_height=MONTAGE_HEIGHT,
        channels=1,
        bit_depth=8,
        pixel_size_nm=MONTAGE_NM,
        preprocess_stage="DONE",
        preprocess_progress=100.0,
    )
    Rendition.objects.create(
        asset=asset,
        type=Rendition.TYPE_FULL,
        storage_root="DATA_DIR",
        stored_path=f"images/montage_{asset.id}.png",
        path_exists=False,
        is_directory=False,
        stored_width=MONTAGE_WIDTH,
        stored_height=MONTAGE_HEIGHT,
        stored_channels=1,
        stored_bit_depth=8,
    )
    return asset


def _segmentation(asset: Asset, type_factory) -> ImageSegmentation:
    return ImageSegmentation.objects.create(
        asset=asset, segmentation_type=type_factory()
    )


def _queue_full_run(segmentation: ImageSegmentation, source_model: str) -> Job:
    """Exactly what the run button enqueues."""
    return Job.enqueue(
        job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
        payload={
            "segmentation_id": str(segmentation.id),
            "segmentation_type": segmentation.segmentation_type.internal_name,
            "asset_id": str(segmentation.asset_id),
            "source_model": source_model,
        },
        priority="default",
        resource_class="gpu",
        queue_name=QUEUE_P4_FULL,
    )


class ThePlanIsOnTheRowBeforeTheRunStarts(TestCase):
    def setUp(self):
        self.asset = _montage_asset()

    def test_a_queued_run_carries_its_tile_count(self):
        segmentation = _segmentation(self.asset, get_or_create_mitochondria_type)

        job = _queue_full_run(segmentation, "quantem:mito")

        assert job.status == "PENDING"
        assert job.progress_units_total == MITO_TILES
        assert job.progress_units_done == 0
        assert job.progress_unit_label == UNIT_TILE
        assert job.progress_stage == STAGE_QUEUED

    def test_the_plan_is_the_number_the_run_will_count_to(self):
        """Not a second opinion: the same function, the same segmenter.

        A denominator that changes when the run starts is worse than no
        denominator, so the enqueue-time estimate is the run's own estimator
        called with the run's own segmenter over the run's own region shape.
        """
        from quantem.seg_core.db.inference import _estimate_model_tile_count
        from quantem.seg_core.registry import get_segmenter

        segmentation = _segmentation(self.asset, get_or_create_nucleus_type)
        job = _queue_full_run(segmentation, "quantem:nucleus")

        segmenter = get_segmenter(
            "dino_nucleus",
            source_model="quantem:nucleus",
            pixel_size_nm=MONTAGE_NM,
        )
        at_run_time = _estimate_model_tile_count(
            segmenter, (MONTAGE_HEIGHT, MONTAGE_WIDTH)
        )

        assert job.progress_units_total == at_run_time == NUCLEUS_TILES

    def test_an_roi_run_plans_the_crop_and_not_the_whole_image(self):
        segmentation = _segmentation(self.asset, get_or_create_mitochondria_type)
        roi = ImageROI.objects.create(
            asset=self.asset, x=0, y=0, width=1024, height=1024
        )

        job = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            payload={
                "segmentation_id": str(segmentation.id),
                "segmentation_type": segmentation.segmentation_type.internal_name,
                "roi_id": str(roi.id),
                "asset_id": str(self.asset.id),
            },
            resource_class="gpu",
            queue_name=QUEUE_P3_ROI,
        )

        assert job.progress_units_total == 9  # 3 x 3 windows over 1024 square
        assert job.progress_units_total < MITO_TILES

    def test_a_job_that_walks_no_tiles_is_left_saying_nothing(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            payload={"asset_id": str(self.asset.id)},
            queue_name=QUEUE_P2_UPLOAD,
        )

        assert job.progress_units_total is None
        assert job.progress_unit_label == ""
        assert planned_units_for(JOB_TYPE_UPLOAD_IMAGE_PIPELINE, {}) is None

    def test_an_image_with_nothing_to_open_yet_is_not_guessed_at(self):
        """The plan is measured from the rendition inference will read.

        Before one exists there is no run to plan for -- inference would refuse
        the image -- and a denominator taken from a different number than the
        one the run will use is worse than none.
        """
        bare = Asset.objects.create(
            display_name="not imported yet",
            original_filename="pending.png",
            logical_width=MONTAGE_WIDTH,
            logical_height=MONTAGE_HEIGHT,
            channels=1,
            bit_depth=8,
            pixel_size_nm=MONTAGE_NM,
        )
        segmentation = _segmentation(bare, get_or_create_mitochondria_type)

        job = _queue_full_run(segmentation, "quantem:mito")

        assert job.progress_units_total is None

    def test_a_run_whose_plan_cannot_be_worked_out_still_enqueues(self):
        """Progress reporting must never be the thing that refuses the work."""
        job = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload={"segmentation_id": str(uuid4()), "asset_id": str(uuid4())},
            resource_class="gpu",
            queue_name=QUEUE_P4_FULL,
        )

        assert job.pk is not None
        assert job.status == "PENDING"
        assert job.progress_units_total is None

    def test_an_uncalibrated_image_is_planned_at_native_scale(self):
        """The plan has to make the same assumption the run makes.

        With no pixel size a model that declares a canonical nm/px runs at the
        image's own scale, which is a different number of windows. Planning at
        25 nm and running at native would move the denominator under the user
        the moment the run started.
        """
        from quantem.seg_core.db.inference import _estimate_model_tile_count
        from quantem.seg_core.registry import get_segmenter

        Asset.objects.filter(id=self.asset.id).update(pixel_size_nm=None)
        self.asset.refresh_from_db()
        segmentation = _segmentation(self.asset, get_or_create_nucleus_type)

        job = _queue_full_run(segmentation, "quantem:nucleus")

        native = _estimate_model_tile_count(
            get_segmenter("dino_nucleus", source_model="quantem:nucleus"),
            (MONTAGE_HEIGHT, MONTAGE_WIDTH),
        )
        assert job.progress_units_total == native
        assert native != NUCLEUS_TILES


class TheWaveCountsWorkTheUserAskedFor(TestCase):
    """The reproduction from the finding, on the rollup the API serialises."""

    def setUp(self):
        self.asset = _montage_asset()
        self.mito = _queue_full_run(
            _segmentation(self.asset, get_or_create_mitochondria_type), "quantem:mito"
        )
        self.nucleus = _queue_full_run(
            _segmentation(self.asset, get_or_create_nucleus_type), "quantem:nucleus"
        )
        self.er = _queue_full_run(
            _segmentation(self.asset, get_or_create_er_type), "quantem:er"
        )

    def _rollup(self) -> dict:
        self.mito.refresh_from_db()
        return batch_progress_for(self.mito)

    def _set(self, job: Job, done: int, status: str) -> None:
        Job.objects.filter(id=job.id).update(
            progress_units_done=done, status=status
        )

    def test_the_whole_wave_is_in_the_denominator_the_moment_it_is_queued(self):
        rollup = self._rollup()

        assert self.mito.batch_id == self.nucleus.batch_id == self.er.batch_id
        assert [job.progress_units_total for job in (self.mito, self.nucleus, self.er)] == [
            MITO_TILES,
            NUCLEUS_TILES,
            ER_TILES,
        ]
        assert rollup["units_total"] == WAVE_TILES == 118
        assert rollup["units_done"] == 0
        assert rollup["percent"] == 0.0
        assert rollup["runs_total"] == 3
        assert rollup["runs_pending"] == 3

    def test_the_measured_reproduction_now_tells_the_truth(self):
        """mito CANCELLED 19/56, nucleus SUCCESS 6/6, ER FAILED 0/56.

        118 tiles asked for, 25 walked. The old rollup said ``units_done 19,
        units_reachable 19, percent 100.0, runs_total 1``.
        """
        self._set(self.mito, 19, "CANCELLED")
        self._set(self.nucleus, NUCLEUS_TILES, "SUCCESS")
        self._set(self.er, 0, "FAILED")

        rollup = self._rollup()

        assert rollup["units_done"] == 25
        assert rollup["units_total"] == 118
        assert rollup["units_abandoned"] == 37 + 56
        assert rollup["percent"] == 21.2
        assert rollup["runs_total"] == 3
        assert rollup["runs_cancelled"] == 1
        assert rollup["runs_failed"] == 1
        assert rollup["runs_succeeded"] == 1
        assert rollup["complete"] is True

    def test_the_percentage_is_monotone_and_never_passes_a_hundred(self):
        """Sampled through the whole reproduction, in order."""
        seen: list[float] = [self._rollup()["percent"]]

        self._set(self.mito, 6, "RUNNING")
        seen.append(self._rollup()["percent"])
        self._set(self.mito, 19, "RUNNING")
        seen.append(self._rollup()["percent"])
        self._set(self.mito, 19, "CANCELLED")
        seen.append(self._rollup()["percent"])
        self._set(self.nucleus, 3, "RUNNING")
        seen.append(self._rollup()["percent"])
        self._set(self.nucleus, NUCLEUS_TILES, "SUCCESS")
        seen.append(self._rollup()["percent"])
        self._set(self.er, 0, "FAILED")
        seen.append(self._rollup()["percent"])

        assert seen == sorted(seen), seen
        assert max(seen) <= 100.0
        assert seen[-1] == 21.2

    def test_a_run_that_cannot_be_planned_withholds_the_bar_and_is_still_counted(self):
        """A fraction of an unknown is not a percentage.

        The run still appears in ``runs_total`` -- the user started it -- but
        the wave says it cannot draw a bar rather than drawing one over a
        denominator that is missing a run.
        """
        Job.objects.filter(id=self.er.id).update(
            progress_units_total=None, progress_unit_label=""
        )
        self._set(self.mito, 19, "RUNNING")

        rollup = self._rollup()

        assert rollup["percent"] is None
        assert rollup["runs_unplanned"] == 1
        assert rollup["runs_total"] == 3
        assert rollup["units_total"] == MITO_TILES + NUCLEUS_TILES

    def test_the_endpoint_still_carries_the_rollup_after_the_wave_ends(self):
        """The summary of what happened has to survive the last run ending.

        The rollup used to be computed for open waves only, so the honest
        final line -- "25 of 118 tiles · 2 of 3 did not finish" -- vanished at
        the instant it became final and the user never read it.
        """
        from rest_framework.test import APIClient

        self._set(self.mito, 19, "CANCELLED")
        self._set(self.nucleus, NUCLEUS_TILES, "SUCCESS")
        self._set(self.er, 0, "FAILED")

        body = APIClient().get("/api/jobs/queue-status/").json()

        assert body["running"] == [] and body["queues"] == []
        rollups = [
            job["batch_progress"]
            for job in body["failed"] + body["completed"]
            if job["batch_progress"]
        ]
        assert rollups, body
        assert rollups[0]["units_done"] == 25
        assert rollups[0]["units_total"] == 118
        assert rollups[0]["percent"] == 21.2
        assert rollups[0]["runs_failed"] + rollups[0]["runs_cancelled"] == 2
        assert rollups[0]["complete"] is True

    def test_a_claimed_attempt_starts_its_tile_walk_from_zero(self):
        """A retry re-walks the tiles; the row must not still claim the old ones.

        Otherwise the wave counts 19 tiles that are about to be done again, and
        then watches the number fall when the new attempt's first write lands.
        """
        from quantem.jobs.scheduler import JobScheduler

        self._set(self.mito, 19, "RETRY")
        Job.objects.filter(id=self.nucleus.id).delete()
        Job.objects.filter(id=self.er.id).delete()

        scheduler = JobScheduler()
        claimed = scheduler._claim_next_ready_job()

        assert claimed is not None and str(claimed.id) == str(self.mito.id)
        assert claimed.status == "RUNNING"
        assert claimed.progress_units_done == 0
        assert claimed.progress_units_total == MITO_TILES

    def test_retrying_a_run_by_hand_puts_its_tiles_back_to_zero(self):
        from rest_framework.test import APIClient

        self._set(self.mito, 19, "FAILED")

        response = APIClient().post(f"/api/jobs/{self.mito.id}/retry/", {}, format="json")

        assert response.status_code == 200
        self.mito.refresh_from_db()
        assert self.mito.status == "PENDING"
        assert self.mito.progress_units_done == 0
        assert self.mito.progress_units_total == MITO_TILES
        assert self.mito.progress_stage == STAGE_QUEUED

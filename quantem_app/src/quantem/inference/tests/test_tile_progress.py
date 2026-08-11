"""The tile numbers a user is shown must be the tile numbers the loop runs.

Two separate promises are tested here:

1. the count quoted **before** the run (``engine.estimate_tiles``) equals the
   count the run will reach (``TilePlan.n_tiles``), for every shipped pack and
   a sweep of region shapes; and
2. the tiling loop reports every window exactly once, in order, with the plan's
   own total -- so a bar built from it is monotone and lands on 100 %.
"""

from __future__ import annotations

import numpy as np
import pytest
from django.test import TestCase

from quantem.inference import engine, resample, tiling
from quantem.inference import segmenter as inference_segmenter
from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.models import STAGE_INFERENCE, STAGE_LOADING_MODEL, UNIT_TILE, Job
from quantem.jobs.reporter import JobReporter, active_reporter

OVERLAP = tiling.DEFAULT_OVERLAP


def _plan_the_run_will_build(spec, native_shape, pixel_size_nm):
    """Exactly what ``predict_region`` does, on an empty array of that shape."""
    context = resample.plan_resample(native_shape, pixel_size_nm, spec.canonical_nm)
    scaled = np.zeros(context.model_shape, dtype=np.uint8)
    padded, _pads = tiling.pad_for_tiling(scaled, spec.tile_size, spec.patch_size)
    return tiling.plan_tiles(padded.shape[:2], spec.tile_size, OVERLAP)


@pytest.mark.parametrize("pack_id", sorted(MODEL_SPECS))
def test_quoted_tile_count_is_the_count_the_run_reaches(pack_id):
    """No pack, shape or pixel size may make the quote and the plan disagree.

    Before this was fixed the quote skipped the patch-multiple padding a run
    applies before laying windows out, so a region whose model shape lands just
    under a stride boundary was quoted 4 tiles and ran 6 -- a bar that reads
    150 % or freezes at 67 %, depending which number the UI trusted.
    """
    spec = MODEL_SPECS[pack_id]
    for pixel_size_nm in (None, 4.0, 8.0):
        for height in range(600, 1400, 37):
            for width in (900, 1301, 2048):
                native = (height, width)
                quoted = engine.estimate_tiles(
                    spec, native, pixel_size_nm=pixel_size_nm, overlap=OVERLAP
                )
                planned = _plan_the_run_will_build(spec, native, pixel_size_nm).n_tiles
                assert quoted == planned, (
                    f"{pack_id} {native} at {pixel_size_nm} nm/px: "
                    f"quoted {quoted}, plan runs {planned}"
                )


def test_padded_shape_agrees_with_pad_for_tiling():
    """The shape helper and the padder are one fact, not two."""
    for shape in [(10, 10), (500, 4000), (517, 519), (1024, 1024), (4096, 3230)]:
        for tile, patch in [(512, 16), (518, 14)]:
            image = np.zeros(shape, dtype=np.uint8)
            padded, _ = tiling.pad_for_tiling(image, tile, patch)
            assert padded.shape[:2] == tiling.padded_shape(shape, tile, patch)


def test_every_tile_is_reported_once_in_order_with_the_plans_total():
    plan = tiling.plan_tiles((1024, 1536), tile=512, overlap=OVERLAP)
    seen: list[tuple[int, int]] = []

    tiling.blend_region(
        plan,
        lambda tile: np.full((plan.tile, plan.tile), 0.5, dtype=np.float32),
        on_tile=lambda done, total: seen.append((done, total)),
    )

    assert [done for done, _ in seen] == list(range(1, plan.n_tiles + 1))
    assert {total for _, total in seen} == {plan.n_tiles}


def test_predict_region_reports_tiles_with_a_stand_in_forward():
    """``on_tile`` survives the whole engine path, not just the tiling module."""
    spec = MODEL_SPECS["quantem:mito"]
    model = engine.LoadedModel(
        spec=spec,
        device="cpu",
        module=None,
        forward=lambda tile: np.full(tile.shape, 0.25, dtype=np.float32),
    )
    image = np.zeros((900, 1300), dtype=np.uint8)
    seen: list[tuple[int, int]] = []

    prediction = engine.predict_region(
        model,
        image,
        pixel_size_nm=8.0,
        on_tile=lambda done, total: seen.append((done, total)),
    )

    assert seen[-1] == (prediction.plan.n_tiles, prediction.plan.n_tiles)
    assert len(seen) == prediction.plan.n_tiles
    assert engine.estimate_tiles(spec, image.shape[:2], pixel_size_nm=8.0) == (
        prediction.plan.n_tiles
    )


class SegmenterReportsOntoTheJobRow(TestCase):
    """The whole path, with a stand-in forward: segmenter -> job row.

    No reporter is threaded through ``BaseSegmenter.predict`` or
    ``seg_core.db.inference``; the tile counts arrive because the running job's
    reporter owns the thread. If that wiring breaks, this is the test that says
    so, and it says so without needing model weights.
    """

    def tearDown(self):
        reporter = active_reporter()
        if reporter is not None:
            reporter.deactivate()
        super().tearDown()

    def _segmenter(self, pack_id="quantem:mito", pixel_size_nm=8.0):
        seg = inference_segmenter.DinoMitoSegmenter(
            source_model=pack_id, pixel_size_nm=pixel_size_nm
        )
        seg._model = engine.LoadedModel(
            spec=MODEL_SPECS[pack_id],
            device="cpu",
            module=None,
            forward=lambda tile: np.full(tile.shape, 0.1, dtype=np.float32),
        )
        return seg

    def test_a_run_lands_on_its_exact_tile_count(self):
        job = Job.objects.create(type="run_segmentation_full_task", payload_json={})
        JobReporter(str(job.id), min_interval_seconds=0.0)
        segmenter = self._segmenter()
        image = np.zeros((900, 1300), dtype=np.uint8)
        expected = engine.estimate_tiles(
            MODEL_SPECS["quantem:mito"], image.shape[:2], pixel_size_nm=8.0
        )

        segmenter.run_dl_inference(image, {"DINO": None})

        job.refresh_from_db()
        assert job.progress_units_total == expected
        assert job.progress_units_done == expected
        assert job.progress_unit_label == UNIT_TILE
        assert job.progress_stage == STAGE_INFERENCE
        assert job.progress_detail_json["model"] == "quantem:mito"
        assert job.progress_detail_json["organelle"] == "mito"
        # The other indicator stays empty: this run downloaded nothing.
        assert job.progress_current_bytes is None
        assert job.progress_total_bytes is None

    def test_loading_the_model_is_its_own_stage_before_any_tile(self):
        job = Job.objects.create(type="run_segmentation_full_task", payload_json={})
        JobReporter(str(job.id), min_interval_seconds=0.0)
        segmenter = inference_segmenter.DinoMitoSegmenter(source_model="quantem:mito")
        loaded = engine.LoadedModel(
            spec=MODEL_SPECS["quantem:mito"], device="cpu", module=None,
            forward=lambda tile: np.zeros(tile.shape, dtype=np.float32),
        )
        self.addCleanup(setattr, engine, "load_model", engine.load_model)
        engine.load_model = lambda pack_id, device=None: loaded

        segmenter.load_models()

        job.refresh_from_db()
        assert job.progress_stage == STAGE_LOADING_MODEL
        assert job.progress_detail_json == {"model": "quantem:mito"}

    def test_inference_outside_a_job_writes_nothing_and_still_runs(self):
        reporter = active_reporter()
        if reporter is not None:
            reporter.deactivate()
        segmenter = self._segmenter()
        maps = segmenter.run_dl_inference(np.zeros((600, 600), np.uint8), {"DINO": None})
        assert maps["DINO"].shape == (600, 600)

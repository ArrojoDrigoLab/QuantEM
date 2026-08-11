"""End-to-end guided fine-tuning, on the rung that always works.

``threshold_only`` needs no torch, no GPU and no downloaded weights — only the
probability maps the app already computed and the user's own annotations. That
is the point of it, so it is tested end to end against a real database rather
than in pieces.
"""

from __future__ import annotations

import pytest
from django.test import TestCase

from quantem.finetune.calibrate import DEFAULT_THRESHOLD
from quantem.finetune.job import adapter_job
from quantem.finetune.tests.fixtures import FakeCancel, FakeReporter, annotated_segmentation
from quantem.segmentation.services.adapt import CompletedRoiRequired


def _run(segmentation, **overrides):
    payload = {
        "segmentation_id": str(segmentation.id),
        "base_model": "quantem:mito",
        "mode": "threshold_only",
        "name": "mito @ test",
    }
    payload.update(overrides)
    reporter = FakeReporter()
    result = adapter_job(payload, reporter, FakeCancel())
    return result, reporter


class ThresholdOnlyAdaptTests(TestCase):
    def test_two_images_produce_a_calibrated_threshold_held_out_image_disjointly(self):
        first = annotated_segmentation("Adapt image one")
        annotated_segmentation("Adapt image two")

        result, reporter = _run(first)

        assert result["status"] == "SUCCESS"
        assert result["mode"] == "threshold_only"
        assert result["split_mode"] == "image-disjoint"

        sweep = result["sweep"]
        # The fixture's ring sits at 0.55, so any threshold at or below it
        # over-segments: calibration has to climb past the default.
        assert sweep["calibrated_threshold"] > DEFAULT_THRESHOLD
        assert sweep["train_dice_at_calibrated"] == pytest.approx(1.0, abs=0.02)
        assert sweep["train_dice_at_default"] < sweep["train_dice_at_calibrated"]

        # Held out at that threshold, never used to choose it.
        assert sweep["heldout_dice_at_calibrated"] is not None
        assert sweep["improvement"] > 0
        assert len(sweep["thresholds"]) == 19

        # Honesty rule 2: the crops the threshold was fit on are named.
        assert sweep["train_crop_names"] and sweep["heldout_crop_names"]
        assert set(sweep["train_crop_names"]).isdisjoint(sweep["heldout_crop_names"])
        assert set(sweep["per_crop"]) == set(
            sweep["train_crop_names"] + sweep["heldout_crop_names"]
        )

        # Honesty rule 3: the oracle is reported, and it is a ceiling.
        assert sweep["heldout_oracle"] >= sweep["heldout_dice_at_calibrated"]

        assert any("training crops only" in c for c in result["caveats"])
        assert reporter.updates[-1][0] == 100.0

    def test_one_image_is_reported_as_within_image_not_generalisation(self):
        segmentation = annotated_segmentation("Adapt single image")
        # A second completed area on the *same* image: enough to hold something
        # out, not enough to call it image-disjoint.
        from quantem.finetune.tests.fixtures import square
        from quantem.segmentation.models import CompletedROI, SegmentObject

        CompletedROI.objects.create(
            segmentation=segmentation, geometry=square(190, 20, 250, 90)
        )
        polygon = square(200, 30, 230, 60)
        SegmentObject.objects.create(
            segmentation=segmentation,
            label_state="CONFIRMED",
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )

        result, _ = _run(segmentation)
        assert result["split_mode"] == "within-image"
        assert any("within-image" in c for c in result["caveats"])

    def test_a_single_region_says_there_is_no_heldout_score(self):
        segmentation = annotated_segmentation("Adapt one region")
        result, _ = _run(segmentation)

        assert result["split_mode"] == "no-heldout"
        assert result["sweep"]["heldout_dice_at_calibrated"] is None
        assert any("no held-out score" in c for c in result["caveats"])

    def test_without_a_completed_roi_the_job_refuses(self):
        segmentation = annotated_segmentation("Adapt no ROI", with_roi=False, organelle="er")
        with pytest.raises(CompletedRoiRequired) as caught:
            _run(segmentation)
        assert "marked as finished" in str(caught.value)

    def test_without_a_probability_map_the_job_says_to_run_the_model(self):
        segmentation = annotated_segmentation("Adapt no prob", with_prob=False, organelle="er")
        with pytest.raises(CompletedRoiRequired) as caught:
            _run(segmentation)
        assert "probability map" in str(caught.value)

    def test_unknown_mode_is_refused(self):
        segmentation = annotated_segmentation("Adapt bad mode")
        with pytest.raises(ValueError, match="mode must be one of"):
            _run(segmentation, mode="full_finetune")

    def test_missing_base_model_is_refused(self):
        segmentation = annotated_segmentation("Adapt no base")
        with pytest.raises(ValueError, match="base_model"):
            _run(segmentation, base_model="")


class JobEntryPointTests(TestCase):
    """The name ``quantem.jobs.handlers`` lazily imports must exist and work."""

    def test_handler_import_path_resolves_to_the_job(self):
        from quantem.finetune.adapter_job import train_organelle_adapter_job

        assert train_organelle_adapter_job is adapter_job

    def test_handler_runs_the_adaptation(self):
        from quantem.jobs.handlers import handle_train_organelle_adapter

        segmentation = annotated_segmentation("Adapt via handler")
        result = handle_train_organelle_adapter(
            {
                "segmentation_id": str(segmentation.id),
                "base_model": "quantem:mito",
                "mode": "threshold_only",
            },
            FakeReporter(),
            FakeCancel(),
        )
        assert result["sweep"]["calibrated_threshold"] > 0

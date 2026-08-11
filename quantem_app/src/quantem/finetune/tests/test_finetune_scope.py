"""What a fine-tune is over, what that is worth, and how it will be split.

The owner's own example is the shape of the first test: *a 10-image dataset
where two images have 3 annotations each and a third has 1 shows **7***. That
number is the one the dialog leads with, so it is pinned here against real rows
rather than against a mock.
"""

from __future__ import annotations

from django.test import TestCase

from quantem.finetune.models import (
    TRAINING_MODE_HOLDOUT_1,
    TRAINING_MODE_USE_ALL,
)
from quantem.finetune.scope import (
    default_training_mode,
    plan_folds,
    planned_round_count,
    planned_training_units,
    resolve_scope,
)
from quantem.finetune.tests.fixtures import annotated_segmentation, done_roi, square
from quantem.library.models import Dataset, Experiment
from quantem.segmentation.models import CompletedROI, SegmentationType
from quantem.segmentation.services.adapt import (
    SOURCE_CONFIRMED_AREA,
    SOURCE_DONE_ROI,
    collect_crops_for_scope,
)
from quantem.segmentation.services.adapt.extract_crops import count_annotations
from quantem.segmentation.type_service import get_or_create_mitochondria_type

#: Three areas that do not touch, all comfortably over the 16 px floor.
THREE_AREAS = ((20, 20, 90, 90), (110, 20, 180, 90), (20, 110, 90, 180))


def _mito_type() -> SegmentationType:
    return get_or_create_mitochondria_type()


class OwnersCountTests(TestCase):
    """The 10-image dataset that shows 7."""

    def setUp(self):
        self.experiment = Experiment.objects.create(name="Fasted cohort")
        self.dataset = Dataset.objects.create(
            experiment=self.experiment, name="Liver 24h"
        )
        self.segmentations = []
        for index in range(10):
            segmentation = annotated_segmentation(
                f"liver_{index:02d}.tif", with_roi=False, with_object=True
            )
            asset = segmentation.asset
            asset.experiment = self.experiment
            asset.save(update_fields=["experiment"])
            asset.datasets.add(self.dataset)
            self.segmentations.append(segmentation)

        # Two images with three annotated areas each...
        for segmentation in self.segmentations[:2]:
            for area in THREE_AREAS:
                CompletedROI.objects.create(
                    segmentation=segmentation, geometry=square(*area)
                )
        # ...and a third with one.
        CompletedROI.objects.create(
            segmentation=self.segmentations[2], geometry=square(*THREE_AREAS[0])
        )

    def test_the_dataset_shows_seven(self):
        asset_ids = [str(s.asset_id) for s in self.segmentations]
        counts = count_annotations(str(_mito_type().id), asset_ids)
        assert sum(entry["annotation_count"] for entry in counts.values()) == 7
        # Three images contributed; the other seven are in the dataset and have
        # nothing to teach yet, which is not an error.
        assert len(counts) == 3

    def test_the_crop_set_agrees_with_the_count(self):
        """The number on the dialog and the areas trained on are one set.

        Counting rows and building crops are separate code paths and they used
        to be able to disagree; both now apply the same clipping, minimum-size
        and de-overlap rules, so this equality is the thing that keeps them
        honest.
        """
        asset_ids = [str(s.asset_id) for s in self.segmentations]
        crop_set = collect_crops_for_scope(str(_mito_type().id), asset_ids)
        assert crop_set.annotation_count == 7
        assert crop_set.confirmed_areas == 7
        assert crop_set.done_rois == 0

    def test_the_scope_endpoint_reports_the_same_seven(self):
        response = self.client.get(
            "/api/finetune/scope/", {"segmentation_type": str(_mito_type().id)}
        )
        assert response.status_code == 200
        body = response.json()
        dataset = body["experiments"][0]["datasets"][0]
        assert dataset["name"] == "Liver 24h"
        assert dataset["image_count"] == 10
        assert dataset["annotated_image_count"] == 3
        assert dataset["annotation_count"] == 7
        assert body["unassigned_images"] == []

    def test_a_dataset_resolves_to_its_images(self):
        scope = resolve_scope(dataset_ids=[str(self.dataset.id)])
        assert scope.eligible
        assert len(scope.asset_ids) == 10
        assert scope.experiment_id == str(self.experiment.id)


class SameExperimentTests(TestCase):
    """One experiment at a time. A hard refusal, not a warning (owner R13)."""

    def setUp(self):
        self.first = Experiment.objects.create(name="Fasted")
        self.second = Experiment.objects.create(name="Fed")
        self.a = annotated_segmentation("fasted_01.tif")
        self.b = annotated_segmentation("fed_01.tif")
        self.c = annotated_segmentation("loose_01.tif")
        for segmentation, experiment in ((self.a, self.first), (self.b, self.second)):
            segmentation.asset.experiment = experiment
            segmentation.asset.save(update_fields=["experiment"])

    def test_two_experiments_are_refused(self):
        scope = resolve_scope(
            asset_ids=[str(self.a.asset_id), str(self.b.asset_id)]
        )
        assert not scope.eligible
        assert "more than one experiment" in scope.blockers[0]

    def test_mixing_an_experiment_with_an_unassigned_image_is_refused(self):
        scope = resolve_scope(
            asset_ids=[str(self.a.asset_id), str(self.c.asset_id)]
        )
        assert not scope.eligible

    def test_unassigned_images_together_are_one_group(self):
        other = annotated_segmentation("loose_02.tif")
        scope = resolve_scope(
            asset_ids=[str(self.c.asset_id), str(other.asset_id)]
        )
        assert scope.eligible
        assert scope.experiment_id is None

    def test_starting_a_run_across_experiments_is_a_400(self):
        response = self.client.post(
            "/api/finetune/runs/",
            {
                "name": "Across the aisle",
                "segmentation_type": str(_mito_type().id),
                "asset_ids": [str(self.a.asset_id), str(self.b.asset_id)],
            },
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "more than one experiment" in response.json()["detail"]

    def test_preview_says_why_rather_than_failing(self):
        response = self.client.post(
            "/api/finetune/preview/",
            {
                "segmentation_type": str(_mito_type().id),
                "asset_ids": [str(self.a.asset_id), str(self.b.asset_id)],
            },
            content_type="application/json",
        )
        assert response.status_code == 200
        body = response.json()
        assert body["eligible"] is False
        assert body["blockers"]


class DoneRoiSourceTests(TestCase):
    """A ROI ticked as done is training data, and is not double-counted."""

    def setUp(self):
        self.segmentation = annotated_segmentation("done_roi.tif", with_roi=False)
        self.asset_ids = [str(self.segmentation.asset_id)]

    def _crops(self):
        return collect_crops_for_scope(str(_mito_type().id), self.asset_ids)

    def test_a_done_roi_alone_contributes_a_crop(self):
        done_roi(self.segmentation, (20, 20, 160, 160))
        crop_set = self._crops()
        assert crop_set.annotation_count == 1
        assert crop_set.done_rois == 1
        assert crop_set.confirmed_areas == 0
        crop = crop_set.crops[0]
        assert crop.source == SOURCE_DONE_ROI
        # Dense inside, so the rectangle is entirely valid and the confirmed
        # object inside it is foreground.
        assert crop.valid.all()
        assert crop.foreground_px > 0

    def test_a_done_roi_and_a_completed_area_are_two_annotations(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=square(20, 20, 90, 90)
        )
        done_roi(self.segmentation, (110, 20, 80, 80))
        crop_set = self._crops()
        assert crop_set.annotation_count == 2
        assert crop_set.confirmed_areas == 1
        assert crop_set.done_rois == 1
        assert {c.source for c in crop_set.crops} == {
            SOURCE_CONFIRMED_AREA,
            SOURCE_DONE_ROI,
        }

    def test_a_done_roi_inside_a_completed_area_is_not_counted_twice(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=square(20, 20, 200, 200)
        )
        done_roi(self.segmentation, (40, 40, 100, 100))
        crop_set = self._crops()
        assert crop_set.annotation_count == 1
        assert crop_set.done_rois == 0

    def test_where_they_overlap_the_area_is_supervised_once(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=square(20, 20, 120, 120)
        )
        done_roi(self.segmentation, (60, 60, 120, 120))
        crop_set = self._crops()
        assert crop_set.annotation_count == 2
        done = next(c for c in crop_set.crops if c.source == SOURCE_DONE_ROI)
        confirmed = next(
            c for c in crop_set.crops if c.source == SOURCE_CONFIRMED_AREA
        )
        # The done ROI keeps only what the polygon does not already claim.
        assert done.annotated_px < done.width * done.height
        overlap_w = min(confirmed.x + confirmed.width, done.x + done.width) - done.x
        overlap_h = min(confirmed.y + confirmed.height, done.y + done.height) - done.y
        assert done.valid[:overlap_h, :overlap_w].sum() == 0

    def test_the_count_helper_and_the_crop_set_agree_about_overlap(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=square(20, 20, 200, 200)
        )
        done_roi(self.segmentation, (40, 40, 100, 100))
        counts = count_annotations(str(_mito_type().id), self.asset_ids)
        assert counts[self.asset_ids[0]]["annotation_count"] == 1


class FoldPlanTests(TestCase):
    """How the hold-out is chosen, and how many rounds that means."""

    def setUp(self):
        self.one = annotated_segmentation("fold_a.tif", with_roi=False)
        self.two = annotated_segmentation("fold_b.tif", with_roi=False)
        for segmentation in (self.one, self.two):
            for area in THREE_AREAS[:2]:
                CompletedROI.objects.create(
                    segmentation=segmentation, geometry=square(*area)
                )
        self.asset_ids = [str(self.one.asset_id), str(self.two.asset_id)]
        self.crops = list(
            collect_crops_for_scope(str(_mito_type().id), self.asset_ids).crops
        )

    def test_use_all_is_one_round_over_everything(self):
        folds, split_mode = plan_folds(
            self.crops, training_mode=TRAINING_MODE_USE_ALL
        )
        assert len(folds) == 1
        assert split_mode == "no-heldout"
        assert len(folds[0].train) == len(self.crops)
        assert folds[0].heldout == []

    def test_holding_out_uses_a_whole_image_when_there_are_two(self):
        folds, split_mode = plan_folds(
            self.crops, training_mode=TRAINING_MODE_HOLDOUT_1
        )
        assert split_mode == "image-disjoint"
        assert len(folds) == 1
        held_images = {c.image_key for c in folds[0].heldout}
        train_images = {c.image_key for c in folds[0].train}
        assert held_images.isdisjoint(train_images)
        assert folds[0].held_out_asset_id in self.asset_ids

    def test_cross_validation_holds_every_image_out_once(self):
        folds, split_mode = plan_folds(
            self.crops, training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True
        )
        assert split_mode == "image-disjoint"
        assert len(folds) == 2
        assert sorted(f.held_out_asset_id for f in folds) == sorted(self.asset_ids)

    def test_one_annotated_image_holds_out_by_tile_and_says_so(self):
        crops = [c for c in self.crops if c.image_key == self.asset_ids[0]]
        folds, split_mode = plan_folds(
            crops, training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True
        )
        assert split_mode == "within-image"
        assert len(folds) == len(crops)
        # A within-image split names no image: the held-out area's image is in
        # the training set too, so attributing the score to it would overclaim.
        assert all(f.held_out_asset_id is None for f in folds)

    def test_a_single_area_cannot_be_split_and_says_so(self):
        folds, split_mode = plan_folds(
            self.crops[:1], training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True
        )
        assert split_mode == "no-heldout"
        assert len(folds) == 1
        assert folds[0].heldout == []


class ProgressDenominatorTests(TestCase):
    """``steps x rounds``, for each of the three ways a run can be set up."""

    def setUp(self):
        self.one = annotated_segmentation("den_a.tif", with_roi=False)
        self.two = annotated_segmentation("den_b.tif", with_roi=False)
        self.three = annotated_segmentation("den_c.tif", with_roi=False)
        for segmentation in (self.one, self.two, self.three):
            CompletedROI.objects.create(
                segmentation=segmentation, geometry=square(*THREE_AREAS[0])
            )
        self.asset_ids = [
            str(self.one.asset_id),
            str(self.two.asset_id),
            str(self.three.asset_id),
        ]

    def _payload(self, **overrides):
        payload = {
            "segmentation_type_id": str(_mito_type().id),
            "asset_ids": self.asset_ids,
            "steps": 300,
        }
        payload.update(overrides)
        return payload

    def test_use_all_is_one_round(self):
        payload = self._payload(training_mode=TRAINING_MODE_USE_ALL)
        assert planned_round_count(payload) == 1
        assert planned_training_units(payload) == 300

    def test_a_plain_hold_out_is_also_one_round(self):
        payload = self._payload(training_mode=TRAINING_MODE_HOLDOUT_1)
        assert planned_round_count(payload) == 1
        assert planned_training_units(payload) == 300

    def test_cross_validation_is_one_round_per_held_out_image(self):
        payload = self._payload(
            training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True
        )
        assert planned_round_count(payload) == 3
        assert planned_training_units(payload) == 900

    def test_a_recorded_plan_is_believed_over_a_recomputed_one(self):
        """The dialog worked it out; the queue must not quietly disagree.

        Recomputing at enqueue would read the library a second time, and an
        annotation added between the click and the enqueue would move the
        denominator under a bar that had already been drawn.
        """
        payload = self._payload(
            training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True, planned_rounds=2
        )
        assert planned_round_count(payload) == 2
        assert planned_training_units(payload) == 600

    def test_the_queue_asks_for_steps_and_gets_steps(self):
        from quantem.jobs.constants import JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
        from quantem.jobs.tile_plan import planned_units_for

        planned = planned_units_for(
            JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            self._payload(training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True),
        )
        assert planned == (900, "step")

    def test_the_older_single_image_payload_still_counts_no_units(self):
        """It counts none, and saying it counts 300 would freeze its bar at zero.

        Only the scoped run writes steps. ``overall_percent`` prefers the unit
        percentage over the coarse one, so a denominator nothing counts against
        is worse than no denominator: the Improve panel would read 0 % from the
        first second to the last.
        """
        from quantem.jobs.constants import JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
        from quantem.jobs.tile_plan import planned_units_for

        payload = {
            "segmentation_id": str(self.one.id),
            "base_model": "quantem:mito",
            "mode": "head",
            "steps": 300,
        }
        assert planned_training_units(payload) is None
        assert planned_units_for(JOB_TYPE_TRAIN_ORGANELLE_ADAPTER, payload) is None

    def test_a_fine_tune_does_not_join_an_image_run_wave(self):
        """Steps and tiles are never added up, so they never share a wave.

        A fine-tune launched from a labeling view carries that view's
        segmentation, which is exactly the key the wave rollup groups on. Left
        in, it would have blanked the image's run progress for the length of
        every fine-tune.
        """
        from quantem.jobs.constants import JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
        from quantem.jobs.models import Job

        batch_id, _seq = Job.resolve_batch(
            JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            {"segmentation_id": str(self.one.id)},
        )
        assert batch_id == ""


class DefaultModeTests(TestCase):
    def test_three_tiles_or_fewer_uses_everything(self):
        assert default_training_mode(0) == TRAINING_MODE_USE_ALL
        assert default_training_mode(3) == TRAINING_MODE_USE_ALL

    def test_four_tiles_holds_one_back(self):
        """Owner R13 left four unstated; the round-3 contract resolves it as
        hold-out, and the reasoning lives at the constant."""
        assert default_training_mode(4) == TRAINING_MODE_HOLDOUT_1
        assert default_training_mode(40) == TRAINING_MODE_HOLDOUT_1

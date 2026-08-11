"""The one rule, and what happens when an image is moved across it.

An image's datasets must all belong to its experiment. Everything here is a
consequence of that sentence, including the two cases the models' docstring
calls out and the case the contract left to be decided: what a move does to the
memberships the new experiment cannot hold.
"""

from __future__ import annotations

from django.core.exceptions import ValidationError
from django.test import TestCase

from quantem.assets.models import Asset
from quantem.library.grouping import (
    UNSET,
    apply_grouping,
    resolve_dataset,
    resolve_experiment,
)
from quantem.library.models import Dataset, Experiment, validate_asset_grouping


def _asset(name: str = "Scan") -> Asset:
    return Asset.objects.create(display_name=name, original_filename=f"{name}.tif")


class ValidateAssetGroupingTests(TestCase):
    """The predicate the whole layer rests on."""

    def setUp(self):
        self.fasted = Experiment.objects.create(name="Fasted cohort")
        self.fed = Experiment.objects.create(name="Fed cohort")
        self.liver = Dataset.objects.create(experiment=self.fasted, name="Liver 24h")

    def test_an_unorganised_image_is_valid(self):
        """The state every library that exists today is in."""
        validate_asset_grouping(_asset())

    def test_an_image_in_its_own_experiments_dataset_is_valid(self):
        asset = _asset()
        asset.experiment = self.fasted
        asset.save()
        asset.datasets.add(self.liver)

        validate_asset_grouping(asset)

    def test_a_dataset_from_another_experiment_is_refused(self):
        asset = _asset()
        asset.experiment = self.fed
        asset.save()
        asset.datasets.add(self.liver)

        with self.assertRaises(ValidationError):
            validate_asset_grouping(asset)

    def test_a_dataset_with_no_experiment_at_all_is_refused(self):
        asset = _asset()
        asset.datasets.add(self.liver)

        with self.assertRaises(ValidationError):
            validate_asset_grouping(asset)


class MovingBetweenExperimentsTests(TestCase):
    """The case the contract left open, pinned.

    A dataset belongs to exactly one experiment, so an image cannot both move
    to another experiment and stay in a dataset of the old one. Refusing the
    move would make the user empty the datasets by hand first for no gain. So
    the move drops those memberships, counts them, and reports the count.
    """

    def setUp(self):
        self.fasted = Experiment.objects.create(name="Fasted cohort")
        self.fed = Experiment.objects.create(name="Fed cohort")
        self.liver = Dataset.objects.create(experiment=self.fasted, name="Liver 24h")
        self.kidney = Dataset.objects.create(experiment=self.fed, name="Kidney 24h")
        self.asset = _asset()
        apply_grouping([self.asset], experiment=self.fasted, datasets=[self.liver])

    def test_the_move_drops_the_datasets_the_new_experiment_cannot_hold(self):
        outcome = apply_grouping([self.asset], experiment=self.fed)

        self.asset.refresh_from_db()
        self.assertEqual(self.asset.experiment_id, self.fed.id)
        self.assertEqual(list(self.asset.datasets.all()), [])
        self.assertEqual(outcome.dataset_links_dropped, 1)
        self.assertEqual(outcome.assets_moved_out_of_datasets, 1)
        self.assertEqual(outcome.datasets_left, ["Liver 24h"])

    def test_the_move_can_name_the_new_dataset_in_the_same_breath(self):
        apply_grouping([self.asset], experiment=self.fed, datasets=[self.kidney])

        self.asset.refresh_from_db()
        self.assertEqual([dataset.name for dataset in self.asset.datasets.all()], ["Kidney 24h"])

    def test_clearing_the_experiment_clears_the_datasets_with_it(self):
        """An image with no experiment cannot be in any dataset."""
        outcome = apply_grouping([self.asset], experiment=None)

        self.asset.refresh_from_db()
        self.assertIsNone(self.asset.experiment_id)
        self.assertEqual(list(self.asset.datasets.all()), [])
        self.assertEqual(outcome.dataset_links_dropped, 1)

    def test_leaving_the_experiment_unmentioned_leaves_the_datasets_alone(self):
        """The tri-state. This is what makes a notes-only edit harmless."""
        outcome = apply_grouping([self.asset], experiment=UNSET, datasets=UNSET)

        self.asset.refresh_from_db()
        self.assertEqual(self.asset.experiment_id, self.fasted.id)
        self.assertEqual([dataset.name for dataset in self.asset.datasets.all()], ["Liver 24h"])
        self.assertEqual(outcome.assets_changed, 0)

    def test_a_dataset_from_the_wrong_experiment_is_refused_and_rolled_back(self):
        with self.assertRaises(ValidationError):
            apply_grouping([self.asset], datasets=[self.kidney])

        self.asset.refresh_from_db()
        self.assertEqual(self.asset.experiment_id, self.fasted.id)
        self.assertEqual([dataset.name for dataset in self.asset.datasets.all()], ["Liver 24h"])

    def test_a_failed_bulk_assignment_lands_nothing_at_all(self):
        """Half a bulk write is worse than none: nothing says which half."""
        other = _asset("Scan 2")

        with self.assertRaises(ValidationError):
            apply_grouping([other, self.asset], datasets=[self.kidney])

        other.refresh_from_db()
        self.assertIsNone(other.experiment_id)
        self.assertEqual(list(other.datasets.all()), [])

    def test_add_mode_keeps_what_is_already_there(self):
        second = Dataset.objects.create(experiment=self.fasted, name="Liver 48h")

        apply_grouping([self.asset], datasets=[second], datasets_mode="add")

        self.assertEqual(
            sorted(dataset.name for dataset in self.asset.datasets.all()),
            ["Liver 24h", "Liver 48h"],
        )


class ResolvingNamesTests(TestCase):
    """ "Pick an existing one or type a new name", server side."""

    def test_a_typed_name_creates_the_experiment(self):
        experiment = resolve_experiment(experiment_name="  Fasted cohort  ")

        self.assertEqual(experiment.name, "Fasted cohort")

    def test_typing_the_same_name_again_reuses_it(self):
        first = resolve_experiment(experiment_name="Fasted cohort")
        second = resolve_experiment(experiment_name="fasted COHORT")

        self.assertEqual(first.id, second.id)
        self.assertEqual(Experiment.objects.count(), 1)

    def test_naming_nothing_is_not_an_error(self):
        self.assertIsNone(resolve_experiment())
        self.assertIsNone(resolve_dataset(experiment=None))

    def test_an_id_that_is_gone_is_a_validation_error_not_a_crash(self):
        with self.assertRaises(ValidationError):
            resolve_experiment(experiment_id="00000000-0000-0000-0000-000000000001")

    def test_a_dataset_needs_an_experiment(self):
        with self.assertRaises(ValidationError):
            resolve_dataset(experiment=None, dataset_name="Liver 24h")

    def test_a_dataset_from_another_experiment_is_refused_by_name(self):
        fasted = Experiment.objects.create(name="Fasted cohort")
        fed = Experiment.objects.create(name="Fed cohort")
        liver = Dataset.objects.create(experiment=fasted, name="Liver 24h")

        with self.assertRaises(ValidationError):
            resolve_dataset(experiment=fed, dataset_id=str(liver.id))

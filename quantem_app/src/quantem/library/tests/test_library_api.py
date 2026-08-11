"""The HTTP surface: experiments, datasets, assignment, and filtering.

The through-line of every test here is that **nothing is required**. An
unorganised library answers every one of these routes, the import door works
exactly as it did before the fields existed, and "no experiment" is a bucket the
filter can name rather than a hole it falls into.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.assets.models import Asset
from quantem.library.models import Dataset, Experiment
from quantem.testing import build_test_upload_file


def _asset(name: str = "Scan") -> Asset:
    return Asset.objects.create(display_name=name, original_filename=f"{name}.tif")


class ExperimentApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_an_unorganised_library_answers_with_an_empty_list(self):
        """Not a 404, not a prompt. There is simply nothing filed yet."""
        response = self.client.get("/api/experiments/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, [])

    def test_creating_one_returns_it_with_its_counts(self):
        response = self.client.post("/api/experiments/", {"name": "Fasted cohort"}, format="json")

        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["name"], "Fasted cohort")
        self.assertEqual(response.data["datasets"], [])
        self.assertEqual(response.data["asset_count"], 0)

    def test_a_duplicate_name_is_refused_in_words_a_person_can_act_on(self):
        Experiment.objects.create(name="Fasted cohort")

        response = self.client.post("/api/experiments/", {"name": "fasted cohort"}, format="json")

        self.assertEqual(response.status_code, 400)
        self.assertIn("already an experiment", response.data["detail"])

    def test_a_blank_name_is_refused(self):
        response = self.client.post("/api/experiments/", {"name": "  "}, format="json")

        self.assertEqual(response.status_code, 400)

    def test_renaming_keeps_the_images(self):
        experiment = Experiment.objects.create(name="Fasted cohort")
        asset = _asset()
        asset.experiment = experiment
        asset.save()

        response = self.client.patch(
            f"/api/experiments/{experiment.id}/", {"name": "Fasted"}, format="json"
        )

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["name"], "Fasted")
        self.assertEqual(response.data["asset_count"], 1)

    def test_deleting_an_experiment_keeps_every_image(self):
        """``SET_NULL``, deliberately. Deleting a label is not a way to lose data."""
        experiment = Experiment.objects.create(name="Fasted cohort")
        dataset = Dataset.objects.create(experiment=experiment, name="Liver 24h")
        asset = _asset()
        asset.experiment = experiment
        asset.save()
        asset.datasets.add(dataset)

        response = self.client.delete(f"/api/experiments/{experiment.id}/")

        self.assertEqual(response.status_code, 204)
        asset.refresh_from_db()
        self.assertTrue(Asset.objects.filter(id=asset.id).exists())
        self.assertIsNone(asset.experiment_id)
        self.assertEqual(list(asset.datasets.all()), [])
        self.assertFalse(Dataset.objects.filter(id=dataset.id).exists())

    def test_one_that_is_gone_answers_with_a_sentence_not_a_stack_trace(self):
        response = self.client.get("/api/experiments/00000000-0000-0000-0000-000000000001/")

        self.assertEqual(response.status_code, 404)
        self.assertIn("no longer in the library", response.data["detail"])

    def test_the_counts_ignore_deleted_images(self):
        experiment = Experiment.objects.create(name="Fasted cohort")
        kept, gone = _asset("Kept"), _asset("Gone")
        for asset in (kept, gone):
            asset.experiment = experiment
            asset.save()
        gone.lifecycle_status = Asset.LIFECYCLE_DELETED
        gone.save()

        response = self.client.get(f"/api/experiments/{experiment.id}/")

        self.assertEqual(response.data["asset_count"], 1)

    def test_an_experiment_reports_its_images_that_are_in_no_dataset(self):
        experiment = Experiment.objects.create(name="Fasted cohort")
        dataset = Dataset.objects.create(experiment=experiment, name="Liver 24h")
        filed, loose = _asset("Filed"), _asset("Loose")
        for asset in (filed, loose):
            asset.experiment = experiment
            asset.save()
        filed.datasets.add(dataset)

        response = self.client.get(f"/api/experiments/{experiment.id}/")

        self.assertEqual(response.data["asset_count"], 2)
        self.assertEqual(response.data["ungrouped_asset_count"], 1)
        self.assertEqual(response.data["datasets"][0]["asset_count"], 1)


class DatasetApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.experiment = Experiment.objects.create(name="Fasted cohort")

    def test_creating_one_inside_an_experiment(self):
        response = self.client.post(
            "/api/datasets/",
            {"experiment": str(self.experiment.id), "name": "Liver 24h"},
            format="json",
        )

        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["experiment"], str(self.experiment.id))

    def test_a_dataset_with_no_experiment_is_refused(self):
        response = self.client.post("/api/datasets/", {"name": "Liver 24h"}, format="json")

        self.assertEqual(response.status_code, 400)
        self.assertIn("lives inside an experiment", response.data["detail"])

    def test_the_same_name_twice_in_one_experiment_is_refused(self):
        Dataset.objects.create(experiment=self.experiment, name="Liver 24h")

        response = self.client.post(
            "/api/datasets/",
            {"experiment": str(self.experiment.id), "name": "liver 24h"},
            format="json",
        )

        self.assertEqual(response.status_code, 400)

    def test_the_same_name_in_two_experiments_is_fine(self):
        other = Experiment.objects.create(name="Fed cohort")
        Dataset.objects.create(experiment=self.experiment, name="Liver 24h")

        response = self.client.post(
            "/api/datasets/",
            {"experiment": str(other.id), "name": "Liver 24h"},
            format="json",
        )

        self.assertEqual(response.status_code, 201, response.data)

    def test_the_list_can_be_narrowed_to_one_experiment(self):
        other = Experiment.objects.create(name="Fed cohort")
        Dataset.objects.create(experiment=self.experiment, name="Liver 24h")
        Dataset.objects.create(experiment=other, name="Kidney 24h")

        response = self.client.get("/api/datasets/", {"experiment": str(other.id)})

        self.assertEqual([row["name"] for row in response.data], ["Kidney 24h"])

    def test_deleting_a_dataset_keeps_its_images_in_the_experiment(self):
        dataset = Dataset.objects.create(experiment=self.experiment, name="Liver 24h")
        asset = _asset()
        asset.experiment = self.experiment
        asset.save()
        asset.datasets.add(dataset)

        response = self.client.delete(f"/api/datasets/{dataset.id}/")

        self.assertEqual(response.status_code, 204)
        asset.refresh_from_db()
        self.assertEqual(asset.experiment_id, self.experiment.id)


class AssignmentApiTests(TestCase):
    """Most images already exist. This is the route that reaches them."""

    def setUp(self):
        self.client = APIClient()
        self.fasted = Experiment.objects.create(name="Fasted cohort")
        self.fed = Experiment.objects.create(name="Fed cohort")
        self.liver = Dataset.objects.create(experiment=self.fasted, name="Liver 24h")
        self.first = _asset("Scan 1")
        self.second = _asset("Scan 2")

    def _assign(self, payload, expect=200):
        response = self.client.post("/api/assets/grouping/", payload, format="json")
        self.assertEqual(response.status_code, expect, response.data)
        return response.data

    def test_a_selection_can_be_filed_in_one_call(self):
        body = self._assign(
            {
                "asset_ids": [str(self.first.id), str(self.second.id)],
                "experiment": str(self.fasted.id),
                "datasets": [str(self.liver.id)],
            }
        )

        self.assertEqual(body["assets_changed"], 2)
        for asset in (self.first, self.second):
            asset.refresh_from_db()
            self.assertEqual(asset.experiment_id, self.fasted.id)
            self.assertEqual([d.name for d in asset.datasets.all()], ["Liver 24h"])

    def test_a_typed_name_creates_the_experiment_and_the_dataset(self):
        body = self._assign(
            {
                "asset_ids": [str(self.first.id)],
                "experiment_name": "Starved cohort",
                "dataset_name": "Liver 6h",
            }
        )

        self.assertEqual(body["experiment"]["name"], "Starved cohort")
        self.first.refresh_from_db()
        self.assertEqual(self.first.experiment.name, "Starved cohort")
        self.assertEqual([d.name for d in self.first.datasets.all()], ["Liver 6h"])

    def test_clearing_puts_an_image_back_in_the_unassigned_bucket(self):
        self._assign(
            {
                "asset_ids": [str(self.first.id)],
                "experiment": str(self.fasted.id),
                "datasets": [str(self.liver.id)],
            }
        )

        self._assign({"asset_ids": [str(self.first.id)], "experiment": None})

        self.first.refresh_from_db()
        self.assertIsNone(self.first.experiment_id)
        self.assertEqual(list(self.first.datasets.all()), [])

    def test_a_move_reports_the_datasets_it_cost(self):
        self._assign(
            {
                "asset_ids": [str(self.first.id)],
                "experiment": str(self.fasted.id),
                "datasets": [str(self.liver.id)],
            }
        )

        body = self._assign({"asset_ids": [str(self.first.id)], "experiment": str(self.fed.id)})

        self.assertEqual(body["dataset_links_dropped"], 1)
        self.assertEqual(body["datasets_left"], ["Liver 24h"])

    def test_a_dataset_from_the_wrong_experiment_is_a_400_not_a_500(self):
        body = self.client.post(
            "/api/assets/grouping/",
            {
                "asset_ids": [str(self.first.id)],
                "experiment": str(self.fed.id),
                "datasets": [str(self.liver.id)],
            },
            format="json",
        )

        self.assertEqual(body.status_code, 400)
        self.assertIn("different", body.data["detail"])

    def test_a_dataset_with_no_experiment_at_all_is_a_400(self):
        response = self.client.post(
            "/api/assets/grouping/",
            {"asset_ids": [str(self.first.id)], "datasets": [str(self.liver.id)]},
            format="json",
        )

        self.assertEqual(response.status_code, 400)

    def test_an_empty_selection_is_refused_in_words(self):
        response = self.client.post("/api/assets/grouping/", {"asset_ids": []}, format="json")

        self.assertEqual(response.status_code, 400)
        self.assertIn("at least one image", response.data["detail"])

    def test_an_experiment_that_is_gone_is_a_400_not_a_500(self):
        response = self.client.post(
            "/api/assets/grouping/",
            {
                "asset_ids": [str(self.first.id)],
                "experiment": "00000000-0000-0000-0000-000000000001",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)


class LibraryFilterTests(TestCase):
    """Grouping is only usable if the library can be narrowed to it."""

    def setUp(self):
        self.client = APIClient()
        self.fasted = Experiment.objects.create(name="Fasted cohort")
        self.liver = Dataset.objects.create(experiment=self.fasted, name="Liver 24h")
        self.filed = _asset("Filed")
        self.filed.experiment = self.fasted
        self.filed.save()
        self.filed.datasets.add(self.liver)
        self.loose = _asset("Loose")

    def _names(self, params):
        response = self.client.get("/api/assets/", params)
        self.assertEqual(response.status_code, 200)
        return sorted(entry["display_name"] for entry in response.data)

    def test_the_list_entry_says_where_each_image_sits(self):
        response = self.client.get("/api/assets/")
        entries = {row["display_name"]: row for row in response.data}

        self.assertEqual(entries["Filed"]["experiment_name"], "Fasted cohort")
        self.assertEqual(entries["Filed"]["dataset_names"], ["Liver 24h"])
        self.assertIsNone(entries["Loose"]["experiment_id"])
        self.assertEqual(entries["Loose"]["dataset_ids"], [])

    def test_filtering_to_one_experiment(self):
        self.assertEqual(self._names({"experiment": str(self.fasted.id)}), ["Filed"])

    def test_filtering_to_one_dataset(self):
        self.assertEqual(self._names({"dataset": str(self.liver.id)}), ["Filed"])

    def test_unassigned_is_a_bucket_the_filter_can_name(self):
        self.assertEqual(self._names({"experiment": "none"}), ["Loose"])

    def test_an_experiment_and_the_unassigned_bucket_together(self):
        self.assertEqual(
            self._names({"experiment": [str(self.fasted.id), "none"]}),
            ["Filed", "Loose"],
        )

    def test_no_filter_still_returns_the_whole_library(self):
        self.assertEqual(self._names({}), ["Filed", "Loose"])

    def test_an_image_in_two_of_the_chosen_datasets_appears_once(self):
        second = Dataset.objects.create(experiment=self.fasted, name="Liver 48h")
        self.filed.datasets.add(second)

        self.assertEqual(self._names({"dataset": [str(self.liver.id), str(second.id)]}), ["Filed"])


class ImportAssignmentTests(TestCase):
    """The import form's two optional fields, at the door."""

    def setUp(self):
        self.client = APIClient()

    def _upload(self, **extra):
        payload = {"file": build_test_upload_file(), "display_name": "Scan 1"}
        payload.update(extra)
        return self.client.post("/api/assets/upload/", payload, format="multipart")

    def test_an_import_that_names_nothing_behaves_exactly_as_before(self):
        response = self._upload()

        self.assertEqual(response.status_code, 201, response.data)
        self.assertIsNone(response.data["experiment_id"])
        self.assertEqual(response.data["dataset_ids"], [])
        self.assertEqual(Experiment.objects.count(), 0)

    def test_a_typed_experiment_and_dataset_are_created_and_attached(self):
        response = self._upload(experiment_name="Fasted cohort", dataset_name="Liver 24h")

        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["experiment_name"], "Fasted cohort")
        self.assertEqual(response.data["dataset_names"], ["Liver 24h"])

    def test_a_second_import_reuses_the_experiment_rather_than_doubling_it(self):
        self._upload(experiment_name="Fasted cohort", dataset_name="Liver 24h")
        self._upload(experiment_name="Fasted cohort", dataset_name="Liver 24h")

        self.assertEqual(Experiment.objects.count(), 1)
        self.assertEqual(Dataset.objects.count(), 1)

    def test_an_existing_experiment_can_be_picked_by_id(self):
        experiment = Experiment.objects.create(name="Fasted cohort")

        response = self._upload(experiment_id=str(experiment.id))

        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["experiment_id"], str(experiment.id))

    def test_a_dataset_with_no_experiment_is_refused_before_anything_is_imported(self):
        response = self._upload(dataset_name="Liver 24h")

        self.assertEqual(response.status_code, 400)
        self.assertIn("lives inside an experiment", response.data["error"])
        self.assertEqual(Asset.objects.count(), 0)

    def test_an_experiment_id_that_is_gone_refuses_the_import_in_words(self):
        response = self._upload(experiment_id="00000000-0000-0000-0000-000000000001")

        self.assertEqual(response.status_code, 400, response.data)
        self.assertIn("no longer in the library", response.data["error"])

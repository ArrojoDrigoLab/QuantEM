from django.test import TestCase
from rest_framework.test import APIClient

from quantem.library.models import Experiment
from quantem.testing import create_small_test_image


class WorkspaceTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.asset = create_small_test_image("Frequent experiment", width=32, height=32).asset
        self.experiment = self.asset.experiment
        self.url = f"/api/experiments/{self.experiment.id}/"

    def test_visit_does_not_rewrite_scientific_update_time(self):
        updated = self.experiment.updated_at
        route = f"/assets/{self.asset.id}/analysis?seg=test"
        self.assertEqual(
            self.client.post(self.url + "navigation/", {"route": route}, format="json").status_code,
            200,
        )
        self.experiment.refresh_from_db()
        self.assertEqual(self.experiment.last_route, route)
        self.assertIsNotNone(self.experiment.last_opened_at)
        self.assertEqual(self.experiment.updated_at, updated)

    def test_navigation_rejects_another_experiment_and_external_routes(self):
        other = Experiment.objects.create(name="Other")
        for route in ("https://example.com", f"/experiments/{other.id}"):
            self.assertEqual(
                self.client.post(
                    self.url + "navigation/", {"route": route}, format="json"
                ).status_code,
                400,
            )

    def test_sample_metadata_round_trips_without_guessing_from_dataset(self):
        response = self.client.get(self.url + "workspace/")
        self.assertEqual(response.data["images"][0]["sample_id"], self.asset.display_name)
        self.assertFalse(response.data["images"][0]["sample_is_explicit"])
        response = self.client.patch(
            self.url + "workspace/",
            {"asset_id": str(self.asset.id), "sample_id": "Donor 02", "group": "T1D"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.client.get(self.url + "workspace/").data["images"][0]["sample_id"], "Donor 02"
        )
        response = self.client.patch(
            self.url + "workspace/",
            {"asset_id": str(self.asset.id), "sample_id": "   "},
            format="json",
        )
        self.assertEqual(response.data["sample_id"], self.asset.display_name)
        self.assertFalse(response.data["sample_is_explicit"])
        self.assertEqual(response.data["group"], "T1D")

    def test_blank_image_renames_are_rejected_without_changing_other_fields(self):
        for suffix in ("", "library-edit/"):
            for name in ("", " \t ", None):
                response = self.client.patch(
                    f"/api/assets/{self.asset.id}/{suffix}",
                    {"display_name": name, "notes": "Should not be saved"},
                    format="json",
                )
                self.assertEqual(response.status_code, 400, response.data)
                self.asset.refresh_from_db()
                self.assertEqual(self.asset.display_name, "Frequent experiment")
                self.assertEqual(self.asset.notes, "")

    def test_navigation_listing_avoids_loading_measurements(self):
        with self.assertNumQueries(2):
            response = self.client.get(self.url + "workspace/?navigation=true")
        self.assertEqual(
            response.data["images"], [{"id": str(self.asset.id), "name": self.asset.display_name}]
        )

    def test_malformed_image_ids_return_bad_requests(self):
        response = self.client.patch(
            self.url + "workspace/", {"asset_id": "invalid", "sample_id": "S1"}, format="json"
        )
        self.assertEqual(response.status_code, 400)
        response = self.client.post(
            self.url + "navigation/", {"route": "/assets/" + "-" * 36}, format="json"
        )
        self.assertEqual(response.status_code, 400)

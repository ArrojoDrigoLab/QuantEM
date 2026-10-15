from types import SimpleNamespace
from unittest import mock

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from quantem.core.config import MODELS_DIR
from quantem.finetune.job import _training_start
from quantem.finetune.models import Adapter
from quantem.finetune.tests.fixtures import annotated_segmentation
from quantem.finetune.tests.test_finetune_runs import installed_pack
from quantem.jobs.models import Job


class ModelLifecycleTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.seg = annotated_segmentation("Parent training image")
        self.parent = Adapter.objects.create(
            segmentation=self.seg,
            segmentation_type=self.seg.segmentation_type,
            experiment=self.seg.asset.experiment,
            base_model="quantem:mito",
            name="Parent",
            status="SUCCESS",
            mode="head",
            head_path="",
        )
        self.parent.scope_assets.add(self.seg.asset)
        path = MODELS_DIR / "adapters" / str(self.parent.id) / "head.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test learned weights")
        self.parent.head_path = str(path.relative_to(MODELS_DIR))
        self.parent.save()
        self.url = f"/api/finetune/models/{self.parent.id}/"

    def test_continue_creates_child_and_keeps_parent_unchanged(self):
        with installed_pack():
            response = self.client.post(
                "/api/finetune/runs/",
                {
                    "name": "Child",
                    "segmentation_type": str(self.seg.segmentation_type_id),
                    "asset_ids": [str(self.seg.asset_id)],
                    "parent_adapter_id": str(self.parent.id),
                    "include_previous_annotations": True,
                },
                format="json",
            )
        self.assertEqual(response.status_code, 202, response.data)
        child = Adapter.objects.get(id=response.data["adapter_id"])
        self.assertEqual(child.parent_id, self.parent.id)
        self.assertIn(str(self.seg.asset_id), child.training_exposure)
        payload = Job.objects.get(id=response.data["job_id"]).payload_json
        self.assertEqual(payload["parent_adapter_id"], str(self.parent.id))
        self.parent.refresh_from_db()
        self.assertEqual(self.parent.status, "SUCCESS")
        self.assertEqual(self.parent.head_file.read_bytes(), b"test learned weights")

    def test_continuation_cannot_overwrite_an_existing_version(self):
        response = self.client.post(
            "/api/finetune/runs/",
            {
                "name": "Child",
                "segmentation_type": str(self.seg.segmentation_type_id),
                "asset_ids": [str(self.seg.asset_id)],
                "parent_adapter_id": str(self.parent.id),
                "overwrite_adapter_id": str(self.parent.id),
            },
            format="json",
        )
        self.assertEqual(response.status_code, 400, response.data)
        self.assertIn("new version", response.data["detail"])
        self.parent.refresh_from_db()
        self.assertEqual(self.parent.status, "SUCCESS")
        self.assertEqual(self.parent.head_file.read_bytes(), b"test learned weights")

    def test_training_initializes_every_round_from_parent_weights(self):
        engine = mock.Mock()
        engine.load_model.side_effect = [
            SimpleNamespace(module=object()),
            SimpleNamespace(module=object()),
        ]
        with mock.patch("quantem.finetune.job.load_head") as load:
            first = _training_start(engine, "quantem:mito", str(self.parent.id))
            second = _training_start(engine, "quantem:mito", str(self.parent.id))
        self.assertIsNot(first.module, second.module)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(load.call_args.args[1], self.parent.head_file)

    def test_delete_keeps_identity_and_existing_segmentation(self):
        path = self.parent.head_file
        response = self.client.delete(self.url)
        self.assertEqual(response.status_code, 204)
        self.assertFalse(path.exists())
        self.parent.refresh_from_db()
        self.assertIsNotNone(self.parent.deleted_at)
        self.assertTrue(type(self.seg).objects.filter(id=self.seg.id).exists())
        self.assertEqual(self.client.get("/api/finetune/models/").data, [])
        self.assertEqual(
            self.client.post(
                f"/api/finetune/runs/{self.parent.id}/apply/", {}, format="json"
            ).status_code,
            409,
        )

    def test_deleted_name_stays_reserved_with_an_actionable_message(self):
        self.assertEqual(self.client.delete(self.url).status_code, 204)
        with installed_pack():
            response = self.client.post(
                "/api/finetune/runs/",
                {
                    "name": self.parent.name,
                    "segmentation_type": str(self.seg.segmentation_type_id),
                    "asset_ids": [str(self.seg.asset_id)],
                },
                format="json",
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "model_name_reserved")
        self.assertEqual(
            response.data["detail"],
            "This name was used by a previous model and cannot be re-used. Make a new name.",
        )
        other = Adapter.objects.create(
            name="Other",
            segmentation_type=self.seg.segmentation_type,
            base_model="quantem:mito",
            status="SUCCESS",
        )
        response = self.client.patch(
            f"/api/finetune/models/{other.id}/", {"name": self.parent.name}, format="json"
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "model_name_reserved")

    def test_delete_refuses_active_inference_and_descendants(self):
        job = Job.objects.create(
            type="run_segmentation_full_task",
            payload_json={"adapter_id": str(self.parent.id)},
            status="PENDING",
        )
        self.assertEqual(self.client.delete(self.url).status_code, 409)
        job.status = "SUCCESS"
        job.save()
        Adapter.objects.create(
            parent=self.parent,
            name="Descendant",
            base_model=self.parent.base_model,
            status="SUCCESS",
        )
        response = self.client.delete(self.url)
        self.assertEqual(response.status_code, 409)
        self.assertIn("used to train another saved model", response.data["detail"])
        self.assertTrue(self.parent.head_file.exists())

    def test_archive_hides_new_selection_but_is_reversible(self):
        self.assertEqual(
            self.client.patch(self.url, {"archived": True}, format="json").status_code, 200
        )
        self.assertEqual(len(self.client.get("/api/finetune/models/").data), 0)
        self.assertEqual(len(self.client.get("/api/finetune/models/?archived=true").data), 1)
        self.assertEqual(
            self.client.patch(
                self.url, {"archived": False, "name": "Renamed"}, format="json"
            ).status_code,
            200,
        )
        self.assertEqual(self.client.get("/api/finetune/models/").data[0]["name"], "Renamed")

    def test_implicit_applied_model_is_protected_during_inference(self):
        self.parent.applied_at = timezone.now()
        self.parent.save()
        Job.objects.create(
            type="run_segmentation_full_task",
            payload_json={"segmentation_id": str(self.seg.id)},
            status="RUNNING",
        )
        self.assertEqual(self.client.delete(self.url).status_code, 409)
        self.assertTrue(self.parent.head_file.exists())

    def test_parent_exposure_is_always_disclosed(self):
        child = Adapter.objects.create(
            parent=self.parent, name="Child", base_model=self.parent.base_model, status="SUCCESS"
        )
        body = self.client.get(f"/api/finetune/runs/{child.id}/").data
        self.assertIn("not independent", body["evaluation_note"])

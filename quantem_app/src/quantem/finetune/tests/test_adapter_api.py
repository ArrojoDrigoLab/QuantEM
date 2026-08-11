"""The guided fine-tuning endpoints, against the contract.

These tests install ``quantem.finetune`` themselves (see
:mod:`quantem.finetune.tests.app_support`) because ``core/settings.py`` and
``core/urls.py`` are owned elsewhere and have not been wired up yet. When they
are, the helper becomes a no-op and these tests keep passing unchanged.
"""

from __future__ import annotations

from django.urls import reverse
from rest_framework.test import APIClient

from quantem.finetune.tests.app_support import FinetuneAppTestCase
from quantem.finetune.tests.fixtures import annotated_segmentation
from quantem.jobs.constants import JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
from quantem.jobs.models import Job


class AdaptApiTests(FinetuneAppTestCase):
    def setUp(self):
        self.client = APIClient()
        self.segmentation = annotated_segmentation("API image one")

    # -- GET crops ----------------------------------------------------------

    def test_crops_endpoint_reports_readiness_and_the_split(self):
        annotated_segmentation("API image two")
        response = self.client.get(reverse("adapt-crops", args=[self.segmentation.id]))
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["blockers"] == []
        assert body["split_mode"] == "image-disjoint"
        assert body["n_images"] == 2
        assert len(body["crops"]) == 2
        crop = body["crops"][0]
        assert {"id", "name", "image_key", "width", "height", "n_objects", "annotated_px"} <= set(
            crop
        )
        # threshold_only is offered whatever the hardware.
        assert "threshold_only" in body["modes"]

    def test_crops_endpoint_names_the_blocker_when_nothing_is_marked_complete(self):
        # A different organelle, so the annotated mito image in setUp is not a
        # sibling of it and cannot supply crops.
        segmentation = annotated_segmentation("API no ROI", with_roi=False, organelle="er")
        response = self.client.get(reverse("adapt-crops", args=[segmentation.id]))
        body = response.json()
        assert body["ready"] is False
        assert "marked as finished" in body["blockers"][0]

    # -- POST adapt ---------------------------------------------------------

    def test_start_enqueues_a_job_and_records_an_adapter(self):
        response = self.client.post(
            reverse("adapt-start", args=[self.segmentation.id]),
            {"base_model": "quantem:mito", "mode": "threshold_only", "name": "mito @ liver"},
            format="json",
        )
        assert response.status_code == 202
        body = response.json()

        adapter = self.Adapter.objects.get(id=body["adapter_id"])
        assert adapter.base_model == "quantem:mito"
        assert adapter.status == "PENDING"
        assert adapter.split_mode == "no-heldout"  # one image, one region

        job = Job.objects.get(id=body["job_id"])
        assert job.type == JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
        assert job.payload_json["adapter_id"] == str(adapter.id)
        assert job.payload_json["segmentation_id"] == str(self.segmentation.id)

    def test_start_refuses_an_unknown_base_model(self):
        response = self.client.post(
            reverse("adapt-start", args=[self.segmentation.id]),
            {"base_model": "quantem:golgi"},
            format="json",
        )
        assert response.status_code == 400
        assert "Unknown model" in response.json()["error"]

    def test_start_refuses_when_nothing_is_marked_complete(self):
        segmentation = annotated_segmentation("API start no ROI", with_roi=False, organelle="er")
        response = self.client.post(
            reverse("adapt-start", args=[segmentation.id]),
            {"base_model": "quantem:mito"},
            format="json",
        )
        assert response.status_code == 400
        assert "marked as finished" in response.json()["error"]

    # -- GET adapter / POST apply -------------------------------------------

    def _finished_adapter(self):
        return self.Adapter.objects.create(
            segmentation=self.segmentation,
            base_model="quantem:mito",
            name="mito @ liver",
            mode="threshold_only",
            status="SUCCESS",
            params={"steps": 0, "lr": 1e-4, "seed": 0},
            calibrated_threshold=0.6,
            split_mode="within-image",
            sweep={
                "thresholds": [0.5, 0.6],
                "train_dice": [0.8, 0.9],
                "calibrated_threshold": 0.6,
                "train_dice_at_calibrated": 0.9,
                "train_dice_at_default": 0.8,
                "heldout_dice_at_calibrated": 0.87,
                "heldout_dice_at_default": 0.81,
                "heldout_oracle": 0.91,
                "improvement": 0.06,
                "per_crop": {"a_0": 0.9, "b_0": 0.87},
                "train_crop_names": ["a_0"],
                "heldout_crop_names": ["b_0"],
            },
        )

    def test_adapter_detail_carries_the_split_mode_and_the_caveats(self):
        adapter = self._finished_adapter()
        response = self.client.get(reverse("adapter-detail", args=[adapter.id]))
        assert response.status_code == 200
        body = response.json()

        assert body["split_mode"] == "within-image"
        assert body["heldout_dice"] == 0.87
        assert body["train_crop_names"] == ["a_0"]
        assert body["heldout_crop_names"] == ["b_0"]
        assert body["sweep"]["heldout_oracle"] == 0.91
        # Honesty rules 1 and 3 reach the response, not just the docs.
        assert any("within-image" in c for c in body["caveats"])
        assert any("ceiling" in c for c in body["caveats"])

    def test_apply_marks_the_adapter_active(self):
        from quantem.finetune.models import active_adapter_for

        adapter = self._finished_adapter()
        response = self.client.post(reverse("adapter-apply", args=[adapter.id]))
        assert response.status_code == 200

        adapter.refresh_from_db()
        assert adapter.applied_at is not None
        assert active_adapter_for(self.segmentation) == adapter

    def test_an_unfinished_adapter_cannot_be_applied(self):
        adapter = self.Adapter.objects.create(
            segmentation=self.segmentation,
            base_model="quantem:mito",
            mode="threshold_only",
        )
        response = self.client.post(reverse("adapter-apply", args=[adapter.id]))
        assert response.status_code == 409
        adapter.refresh_from_db()
        assert adapter.applied_at is None

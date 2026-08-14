"""Starting, overwriting, watching and applying a named fine-tune.

The API half needs no torch: nothing here runs a training loop except the two
classes at the bottom, which are marked ``slow`` and use the same convolutional
stand-in as ``test_adapt_job_head.py``.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest
from django.test import TestCase
from django.utils import timezone

from quantem.finetune.models import (
    TRAINING_MODE_HOLDOUT_1,
    TRAINING_MODE_USE_ALL,
    Adapter,
    active_adapter_for,
)
from quantem.finetune.tests.fixtures import (
    FakeCancel,
    FakeReporter,
    annotated_segmentation,
    square,
)
from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)
from quantem.jobs.models import Job
from quantem.library.models import Dataset, Experiment
from quantem.segmentation.models import CompletedROI
from quantem.segmentation.type_service import get_or_create_mitochondria_type

AREAS = ((20, 20, 90, 90), (110, 20, 180, 90), (20, 110, 90, 180))


def _mito():
    return get_or_create_mitochondria_type()


def _post(client, url, body):
    return client.post(url, json.dumps(body), content_type="application/json")


def installed_pack():
    """Pretend the released pack is installed and loadable on this machine.

    Whether ``quantem:mito`` is downloaded is a property of the machine, not of
    the routing, the naming rules or the scope arithmetic these tests are about.
    Without this the whole class goes red on a clean checkout for a reason that
    has nothing to do with what broke -- and "no runnable base model" has its own
    test below, which asserts it fires rather than asserting it does not.
    """
    return mock.patch("quantem.finetune.run_views._runnable_reason", return_value=None)


class NamedRunTests(TestCase):
    def setUp(self):
        self._pack = installed_pack()
        self._pack.start()
        self.addCleanup(self._pack.stop)
        self.experiment = Experiment.objects.create(name="Fasted cohort")
        self.segmentations = []
        for index in range(2):
            segmentation = annotated_segmentation(f"named_{index}.tif", with_roi=False)
            segmentation.asset.experiment = self.experiment
            segmentation.asset.save(update_fields=["experiment"])
            for area in AREAS[:2]:
                CompletedROI.objects.create(segmentation=segmentation, geometry=square(*area))
            self.segmentations.append(segmentation)
        self.asset_ids = [str(s.asset_id) for s in self.segmentations]

    def _start(self, name="Fasted mitochondria", **extra):
        body = {
            "name": name,
            "segmentation_type": str(_mito().id),
            "asset_ids": self.asset_ids,
        }
        body.update(extra)
        return _post(self.client, "/api/finetune/runs/", body)

    # -- starting -----------------------------------------------------------

    def test_a_run_records_its_name_scope_and_experiment(self):
        response = self._start()
        assert response.status_code == 202, response.content
        adapter = Adapter.objects.get(id=response.json()["adapter_id"])
        assert adapter.name == "Fasted mitochondria"
        assert adapter.segmentation_type_id == _mito().id
        assert str(adapter.experiment_id) == str(self.experiment.id)
        assert sorted(str(a.id) for a in adapter.scope_assets.all()) == sorted(self.asset_ids)

    def test_a_run_opened_from_a_labeling_view_remembers_where_it_came_from(self):
        """``active_adapter_for`` has a second, older way in, and it still works.

        The old Improve panel matches on the ``segmentation`` foreign key, and
        the dialog can be opened from that same view. Recording it costs nothing
        and keeps one lookup where there would otherwise be two.
        """
        response = self._start(segmentation_id=str(self.segmentations[0].id))
        adapter = Adapter.objects.get(id=response.json()["adapter_id"])
        assert adapter.segmentation_id == self.segmentations[0].id

    def test_a_segmentation_outside_the_scope_is_ignored_not_obeyed(self):
        outsider = annotated_segmentation("elsewhere.tif")
        response = self._start(segmentation_id=str(outsider.id))
        adapter = Adapter.objects.get(id=response.json()["adapter_id"])
        assert adapter.segmentation_id is None

    def test_a_run_with_no_name_is_refused(self):
        response = self._start(name="")
        assert response.status_code == 400
        assert "name" in response.json()["detail"]

    def test_the_queued_job_carries_the_scope_and_the_round_plan(self):
        response = self._start(mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True)
        job = Job.objects.get(id=response.json()["job_id"])
        assert job.type == JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
        assert sorted(job.payload_json["asset_ids"]) == sorted(self.asset_ids)
        # Two annotated images, so cross-validation is two rounds.
        assert job.payload_json["planned_rounds"] == 2
        assert job.payload_json["planned_steps"] == [300, 300]
        assert job.payload_json["fixed_steps"] is None
        assert job.progress_units_total == sum(job.payload_json["planned_steps"])
        assert job.progress_unit_label == "step"

    def test_an_explicit_step_override_remains_fixed(self):
        response = self._start(steps=420)
        job = Job.objects.get(id=response.json()["job_id"])
        assert job.payload_json["fixed_steps"] == 420
        assert job.payload_json["planned_steps"] == [420]
        assert job.progress_units_total == 420

    def test_an_explicit_step_override_cannot_bypass_the_bounds(self):
        too_few = self._start(name="Too few", steps=299)
        too_many = self._start(name="Too many", steps=601)
        assert too_few.status_code == 400
        assert too_many.status_code == 400
        assert Job.objects.count() == 0

    def test_the_queued_scoped_job_is_accepted_by_the_worker(self):
        """The named dialog has no single segmentation to put in its payload."""
        from quantem.jobs.handlers import handle_train_organelle_adapter

        response = self._start()
        job = Job.objects.get(id=response.json()["job_id"])
        assert "segmentation_id" not in job.payload_json

        with mock.patch(
            "quantem.finetune.adapter_job.train_organelle_adapter_job",
            return_value={"status": "SUCCESS"},
        ) as train:
            result = handle_train_organelle_adapter(job.payload_json, FakeReporter(), FakeCancel())

        assert result == {"status": "SUCCESS"}
        train.assert_called_once()

    def test_the_default_mode_follows_the_tile_count(self):
        response = _post(
            self.client,
            "/api/finetune/preview/",
            {"segmentation_type": str(_mito().id), "asset_ids": self.asset_ids},
        )
        body = response.json()
        assert body["annotation_count"] == 4
        assert body["default_mode"] in (TRAINING_MODE_USE_ALL, TRAINING_MODE_HOLDOUT_1)
        assert body["experiment"]["name"] == "Fasted cohort"
        assert len(body["per_image"]) == 2

    def test_one_annotation_defaults_to_use_all_and_cannot_be_held_out(self):
        segmentation = annotated_segmentation("one_annotation.tif", with_roi=False)
        segmentation.asset.experiment = self.experiment
        segmentation.asset.save(update_fields=["experiment"])
        CompletedROI.objects.create(
            segmentation=segmentation,
            geometry=square(*AREAS[0]),
        )
        asset_ids = [str(segmentation.asset_id)]

        preview = _post(
            self.client,
            "/api/finetune/preview/",
            {"segmentation_type": str(_mito().id), "asset_ids": asset_ids},
        ).json()
        response = self._start(
            name="Too little to hold out",
            asset_ids=asset_ids,
            mode=TRAINING_MODE_HOLDOUT_1,
        )

        assert preview["annotation_count"] == 1
        assert preview["default_mode"] == TRAINING_MODE_USE_ALL
        assert response.status_code == 400
        assert "at least 2 annotations" in response.json()["detail"]

    def test_cross_validation_requires_three_annotations(self):
        two_annotation_asset = [str(self.segmentations[0].asset_id)]

        holdout = self._start(
            name="Two annotation holdout",
            asset_ids=two_annotation_asset,
            mode=TRAINING_MODE_HOLDOUT_1,
        )
        cross_validation = self._start(
            name="Two annotation CV",
            asset_ids=two_annotation_asset,
            mode=TRAINING_MODE_HOLDOUT_1,
            cv_benchmark=True,
        )

        assert holdout.status_code == 202, holdout.content
        assert cross_validation.status_code == 400
        assert "at least 3 annotations" in cross_validation.json()["detail"]

    def test_a_model_that_cannot_run_here_is_a_blocker_not_a_surprise(self):
        """The refusal happens at the door, with the registry's own sentence.

        The alternative is a queued job that dies minutes later with a message
        the user never sees, which is the defect the pre-flight exists to close.
        """
        self._pack.stop()
        try:
            with mock.patch(
                "quantem.finetune.run_views._runnable_reason",
                return_value="Not installed yet.",
            ):
                preview = _post(
                    self.client,
                    "/api/finetune/preview/",
                    {
                        "segmentation_type": str(_mito().id),
                        "asset_ids": self.asset_ids,
                    },
                ).json()
                assert preview["eligible"] is False
                assert "Not installed yet." in preview["blockers"]
                assert self._start().status_code == 400
        finally:
            self._pack.start()

    def test_preview_checks_the_pack_the_user_selected(self):
        def runnable_reason(pack_id):
            return "OmniEM is not installed." if pack_id == "omniem:mito" else None

        with mock.patch(
            "quantem.finetune.run_views._runnable_reason",
            side_effect=runnable_reason,
        ) as check:
            response = _post(
                self.client,
                "/api/finetune/preview/",
                {
                    "segmentation_type": str(_mito().id),
                    "asset_ids": self.asset_ids,
                    "base_model": "omniem:mito",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["base_model"] == "omniem:mito"
        assert body["eligible"] is False
        assert "OmniEM is not installed." in body["blockers"]
        check.assert_called_once_with("omniem:mito")

    def test_a_pack_for_another_organelle_is_refused(self):
        payload = {
            "segmentation_type": str(_mito().id),
            "asset_ids": self.asset_ids,
            "base_model": "quantem:er",
        }

        preview_response = _post(self.client, "/api/finetune/preview/", payload)
        run_response = self._start(base_model="quantem:er")

        assert preview_response.status_code == 200
        assert preview_response.json()["eligible"] is False
        assert "different organelle" in preview_response.json()["blockers"][0]
        assert run_response.status_code == 400
        assert "different organelle" in run_response.json()["detail"]

    # -- name collisions ----------------------------------------------------

    def test_a_second_run_with_the_same_name_is_a_conflict(self):
        assert self._start().status_code == 202
        clash = self._start()
        assert clash.status_code == 409
        assert "already called" in clash.json()["detail"]

    def test_the_same_name_for_another_organelle_is_a_different_fine_tune(self):
        from quantem.segmentation.type_service import get_or_create_er_type

        er = annotated_segmentation("er_01.tif", organelle="er", with_roi=False)
        er.asset.experiment = self.experiment
        er.asset.save(update_fields=["experiment"])
        CompletedROI.objects.create(segmentation=er, geometry=square(*AREAS[0]))

        assert self._start(name="Shared name").status_code == 202
        response = _post(
            self.client,
            "/api/finetune/runs/",
            {
                "name": "Shared name",
                "segmentation_type": str(get_or_create_er_type().id),
                "asset_ids": [str(er.asset_id)],
            },
        )
        assert response.status_code == 202, response.content

    def test_naming_the_run_to_overwrite_is_accepted(self):
        first = self._start().json()["adapter_id"]
        again = self._start(overwrite_adapter_id=first)
        assert again.status_code == 202
        assert again.json()["adapter_id"] == first
        assert Adapter.objects.filter(name="Fasted mitochondria").count() == 1

    def test_overwriting_a_fine_tune_for_another_organelle_is_refused(self):
        from quantem.segmentation.type_service import get_or_create_er_type

        other = Adapter.objects.create(
            base_model="quantem:er",
            name="Elsewhere",
            segmentation_type=get_or_create_er_type(),
        )
        response = self._start(overwrite_adapter_id=str(other.id))
        assert response.status_code == 400
        assert "different organelle" in response.json()["detail"]

    # -- overwrite semantics ------------------------------------------------

    def test_an_overwrite_resets_the_results_but_keeps_the_weights(self):
        """An unused old head stays on disk until its replacement is safe."""
        adapter_id = self._start().json()["adapter_id"]
        Adapter.objects.filter(id=adapter_id).update(
            status="SUCCESS",
            head_path="adapters/keep/head.pt",
            sweep={"calibrated_threshold": 0.4},
            calibrated_threshold=0.4,
            cv_results={"folds": [{"fold": 0, "dice": 0.8, "n_tiles": 2}]},
        )
        self._start(overwrite_adapter_id=adapter_id)
        adapter = Adapter.objects.get(id=adapter_id)
        assert adapter.status == "PENDING"
        assert adapter.sweep == {}
        assert adapter.cv_results == {}
        assert adapter.calibrated_threshold is None
        assert adapter.head_path == "adapters/keep/head.pt"

    def test_an_overwrite_keeps_the_live_version_and_its_targets_routable(self):
        adapter_id = self._start().json()["adapter_id"]
        adapter = Adapter.objects.get(id=adapter_id)
        adapter.status = "SUCCESS"
        adapter.applied_at = timezone.now()
        adapter.head_path = "adapters/keep/head.pt"
        adapter.calibrated_threshold = 0.37
        adapter.save(
            update_fields=[
                "status",
                "applied_at",
                "head_path",
                "calibrated_threshold",
                "updated_at",
            ]
        )
        adapter.applied_assets.add(*adapter.scope_assets.all())

        response = self._start(overwrite_adapter_id=adapter_id)

        assert response.status_code == 202, response.content
        adapter.refresh_from_db()
        assert adapter.status == "PENDING"
        assert adapter.preserves_live_version is True
        assert adapter.head_path == "adapters/keep/head.pt"
        assert adapter.calibrated_threshold == 0.37
        assert adapter.applied_assets.count() == 2
        for segmentation in self.segmentations:
            assert active_adapter_for(segmentation) == adapter

    def test_an_adapter_cannot_be_replaced_while_its_apply_batch_is_open(self):
        adapter_id = self._start().json()["adapter_id"]
        Adapter.objects.filter(id=adapter_id).update(status="SUCCESS")
        applying = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            status="RUNNING",
            batch_id=f"finetune-apply:{adapter_id}:open",
            payload_json={"segmentation_id": str(self.segmentations[0].id)},
        )

        response = self._start(overwrite_adapter_id=adapter_id)

        assert response.status_code == 409
        assert str(applying.id) in response.json()["detail"]
        assert Adapter.objects.get(id=adapter_id).status == "SUCCESS"

    def test_a_failed_overwrite_says_the_old_one_is_still_there(self):
        from quantem.finetune.job import adapter_job
        from quantem.finetune.tests.fixtures import FakeCancel, FakeReporter

        adapter_id = self._start().json()["adapter_id"]
        adapter = Adapter.objects.get(id=adapter_id)
        adapter.status = "SUCCESS"
        adapter.applied_at = timezone.now()
        adapter.head_path = "adapters/keep/head.pt"
        adapter.calibrated_threshold = 0.37
        adapter.save()
        adapter.applied_assets.add(*adapter.scope_assets.all())
        response = self._start(overwrite_adapter_id=adapter_id)
        payload = Job.objects.get(id=response.json()["job_id"]).payload_json
        payload["base_model"] = "quantem:nope"

        with pytest.raises(ValueError):
            adapter_job(payload, FakeReporter(), FakeCancel())
        adapter = Adapter.objects.get(id=adapter_id)
        assert adapter.status == "FAILED"
        assert adapter.preserves_live_version is True
        assert "untouched and still in use" in adapter.error
        assert adapter.head_path == "adapters/keep/head.pt"
        assert adapter.calibrated_threshold == 0.37
        for segmentation in self.segmentations:
            assert active_adapter_for(segmentation) == adapter

    # -- the overwrite dropdown --------------------------------------------

    def test_existing_fine_tunes_are_listed_for_this_organelle_only(self):
        from quantem.segmentation.type_service import get_or_create_er_type

        self._start(name="Mito one")
        Adapter.objects.create(
            base_model="quantem:er",
            name="ER one",
            segmentation_type=get_or_create_er_type(),
        )
        response = self.client.get(
            "/api/finetune/adapters/", {"segmentation_type": str(_mito().id)}
        )
        assert response.status_code == 200
        names = [row["name"] for row in response.json()]
        assert names == ["Mito one"]
        assert response.json()[0]["asset_count"] == 2
        assert sorted(response.json()[0]["asset_ids"]) == sorted(self.asset_ids)


class ProgressEndpointTests(TestCase):
    def setUp(self):
        self.adapter = Adapter.objects.create(
            base_model="quantem:mito",
            name="Watched",
            segmentation_type=_mito(),
            training_mode=TRAINING_MODE_HOLDOUT_1,
            cv_benchmark=True,
            status="RUNNING",
        )
        self.job = Job.objects.create(
            type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            payload_json={"adapter_id": str(self.adapter.id)},
            status="RUNNING",
            progress_units_total=600,
            progress_units_done=240,
            progress_unit_label="step",
            progress_stage="training",
            progress_detail_json={"round": 2, "total_rounds": 2},
            message="Round 2 of 2",
        )

    def _body(self):
        response = self.client.get(f"/api/finetune/runs/{self.adapter.id}/progress/")
        assert response.status_code == 200
        return response.json()

    def test_the_wire_shape_is_the_contract(self):
        body = self._body()
        assert set(body) == {
            "status",
            "stage",
            "step",
            "total_steps",
            "round",
            "total_rounds",
            "percent",
            "eta_seconds",
            "message",
            "error",
        }
        assert body["status"] == "RUNNING"
        assert body["stage"] == "training"
        assert (body["step"], body["total_steps"]) == (240, 600)
        assert (body["round"], body["total_rounds"]) == (2, 2)

    def test_the_percentage_is_computed_from_the_steps(self):
        """One number behind the bar and the text, so they cannot disagree."""
        assert self._body()["percent"] == 40.0

    def test_no_eta_is_offered_from_a_standing_start(self):
        from django.utils import timezone

        self.job.started_at = timezone.now()
        self.job.progress_units_done = 3
        self.job.progress_detail_json = {"round": 1, "total_rounds": 2}
        self.job.save()
        assert self._body()["eta_seconds"] is None

    def test_an_eta_appears_once_enough_has_happened(self):
        from datetime import timedelta

        from django.utils import timezone

        self.job.started_at = timezone.now() - timedelta(seconds=120)
        self.job.save(update_fields=["started_at"])
        eta = self._body()["eta_seconds"]
        # 240 of 600 steps in 120 s leaves 360 steps, so about 180 s.
        assert eta is not None
        assert 170 <= eta <= 190

    def test_a_finished_run_reads_as_finished_whatever_the_queue_says(self):
        Adapter.objects.filter(id=self.adapter.id).update(status="SUCCESS")
        body = self._body()
        assert body["percent"] == 100.0
        assert body["eta_seconds"] is None


class ApplyTests(TestCase):
    def setUp(self):
        self.experiment = Experiment.objects.create(name="Applied cohort")
        self.segmentations = []
        for index in range(2):
            segmentation = annotated_segmentation(f"apply_{index}.tif")
            segmentation.asset.experiment = self.experiment
            segmentation.asset.save(update_fields=["experiment"])
            self.segmentations.append(segmentation)
        self.outsider = annotated_segmentation("outsider.tif")
        self.dataset_only = annotated_segmentation(
            "dataset_only.tif", with_roi=False, with_object=False, with_prob=False
        )
        self.dataset_only.asset.experiment = self.experiment
        self.dataset_only.asset.save(update_fields=["experiment"])
        self.dataset = Dataset.objects.create(
            experiment=self.experiment, name="Every field of view"
        )
        for segmentation in [*self.segmentations, self.dataset_only]:
            segmentation.asset.datasets.add(self.dataset)
        self.adapter = Adapter.objects.create(
            base_model="quantem:mito",
            name="Applied",
            segmentation_type=_mito(),
            experiment=self.experiment,
            status="SUCCESS",
            calibrated_threshold=0.45,
        )
        self.adapter.scope_assets.set([s.asset_id for s in self.segmentations])

    def test_nothing_is_queued_until_it_is_asked_for(self):
        assert Job.objects.count() == 0
        assert self.adapter.applied_at is None

    def test_applying_queues_one_run_per_chosen_image(self):
        response = _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"asset_ids": [str(self.segmentations[0].asset_id)]},
        )
        assert response.status_code == 202, response.content
        queued = response.json()["queued"]
        assert len(queued) == 1
        assert queued[0]["asset_id"] == str(self.segmentations[0].asset_id)
        assert Job.objects.filter(id=queued[0]["job_id"]).exists()

    def test_apply_refuses_a_completed_segmentation_without_partial_scheduling(self):
        self.segmentations[0].status_stage = "COMPLETED"
        self.segmentations[0].save(update_fields=["status_stage", "updated_at"])

        response = _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"dataset_ids": [str(self.dataset.id)]},
        )

        assert response.status_code == 409
        assert "marked complete" in response.json()["detail"]
        self.adapter.refresh_from_db()
        assert self.adapter.applied_at is None
        assert self.adapter.applied_assets.count() == 0
        assert Job.objects.count() == 0

    def test_apply_refuses_an_image_with_an_active_segmentation_job(self):
        busy = self.segmentations[0]
        existing = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload_json={"segmentation_id": str(busy.id)},
        )

        response = _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"asset_ids": [str(busy.asset_id)]},
        )

        assert response.status_code == 409
        assert str(existing.id) in response.json()["detail"]
        self.adapter.refresh_from_db()
        assert self.adapter.applied_at is None
        assert self.adapter.applied_assets.count() == 0
        assert Job.objects.count() == 1

    def test_an_image_outside_the_scope_is_refused(self):
        response = _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"asset_ids": [str(self.outsider.asset_id)]},
        )
        assert response.status_code == 400
        assert "fine-tune's experiment" in response.json()["detail"]

    def test_an_unassigned_fine_tune_cannot_target_an_assigned_image(self):
        adapter = Adapter.objects.create(
            base_model="quantem:mito",
            name="Unassigned",
            segmentation_type=_mito(),
            experiment=None,
            status="SUCCESS",
        )

        response = _post(
            self.client,
            f"/api/finetune/runs/{adapter.id}/apply/",
            {"asset_ids": [str(self.segmentations[0].asset_id)]},
        )

        assert response.status_code == 400
        assert "fine-tune's experiment" in response.json()["detail"]

    def test_apply_rejects_non_list_and_invalid_identifiers(self):
        url = f"/api/finetune/runs/{self.adapter.id}/apply/"
        non_list = _post(self.client, url, {"asset_ids": "not-a-list"})
        invalid = _post(self.client, url, {"dataset_ids": ["not-a-uuid"]})

        assert non_list.status_code == 400
        assert invalid.status_code == 400

    def test_applying_a_dataset_queues_every_image_in_one_reportable_batch(self):
        response = _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"dataset_ids": [str(self.dataset.id)]},
        )
        assert response.status_code == 202, response.content
        body = response.json()
        assert len(body["queued"]) == 3
        assert body["batch_id"].startswith(f"finetune-apply:{self.adapter.id}:")
        jobs = list(Job.objects.filter(batch_id=body["batch_id"]).order_by("batch_seq"))
        assert len(jobs) == 3
        assert {job.payload_json["legs"][0]["adapter_id"] for job in jobs} == {str(self.adapter.id)}
        assert {job.payload_json["asset_id"] for job in jobs} == {
            str(segmentation.asset_id) for segmentation in [*self.segmentations, self.dataset_only]
        }
        self.adapter.refresh_from_db()
        assert {
            str(asset_id) for asset_id in self.adapter.applied_assets.values_list("id", flat=True)
        } == {
            str(segmentation.asset_id) for segmentation in [*self.segmentations, self.dataset_only]
        }
        assert active_adapter_for(self.dataset_only) == self.adapter

    def test_apply_progress_and_failure_are_reported_per_image(self):
        response = _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"dataset_ids": [str(self.dataset.id)]},
        )
        batch_id = response.json()["batch_id"]
        jobs = list(Job.objects.filter(batch_id=batch_id).order_by("batch_seq"))
        Job.objects.filter(id=jobs[0].id).update(
            status="RUNNING",
            progress=37.5,
            progress_stage="inference",
            progress_units_done=3,
            progress_units_total=8,
            message="segmenting",
        )
        Job.objects.filter(id=jobs[1].id).update(
            status="FAILED",
            progress=60.0,
            message="model could not be loaded",
        )
        progress = self.client.get(
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"batch_id": batch_id},
        )
        assert progress.status_code == 200
        payload = progress.json()
        assert payload["total"] == 3
        assert payload["complete"] == 1
        assert payload["failed"] == 1
        by_status = {item["status"]: item for item in payload["images"]}
        assert by_status["RUNNING"]["progress"] == 37.5
        assert (by_status["RUNNING"]["units_done"], by_status["RUNNING"]["units_total"]) == (
            3,
            8,
        )
        assert by_status["FAILED"]["failure"] == "model could not be loaded"
        assert by_status["FAILED"]["adapter_id"] == str(self.adapter.id)

    def test_a_queue_failure_rolls_back_the_entire_apply_batch(self):
        from quantem.finetune.run_views import FineTuneRunApplyView

        original = FineTuneRunApplyView._queue_run
        calls = 0

        def fail_after_first(asset, segmentation, adapter, *, batch_id):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("synthetic queue failure")
            return original(asset, segmentation, adapter, batch_id=batch_id)

        with mock.patch.object(
            FineTuneRunApplyView,
            "_queue_run",
            side_effect=fail_after_first,
        ):
            with pytest.raises(RuntimeError, match="synthetic queue failure"):
                _post(
                    self.client,
                    f"/api/finetune/runs/{self.adapter.id}/apply/",
                    {"dataset_ids": [str(self.dataset.id)]},
                )

        self.adapter.refresh_from_db()
        assert self.adapter.applied_at is None
        assert self.adapter.applied_assets.count() == 0
        assert Job.objects.count() == 0

    def test_an_unfinished_fine_tune_cannot_be_applied(self):
        Adapter.objects.filter(id=self.adapter.id).update(status="RUNNING")
        response = _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"asset_ids": [str(self.segmentations[0].asset_id)]},
        )
        assert response.status_code == 409

    def test_a_scoped_fine_tune_becomes_the_active_one_for_every_image_in_it(self):
        """``active_adapter_for`` is what the run path reads, and a scoped
        fine-tune covers many images with one row -- so the lookup cannot be on
        the single ``segmentation`` foreign key alone."""
        _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"asset_ids": [str(s.asset_id) for s in self.segmentations]},
        )
        for segmentation in self.segmentations:
            assert active_adapter_for(segmentation) == self.adapter
        assert active_adapter_for(self.outsider) is None

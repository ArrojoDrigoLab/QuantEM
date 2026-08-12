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
from quantem.jobs.constants import JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
from quantem.jobs.models import Job
from quantem.library.models import Experiment
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
        assert job.progress_units_total == 2 * job.payload_json["steps"]
        assert job.progress_unit_label == "step"

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
        """A failed overwrite must lose nothing, so the reset cannot touch the
        head path: the old file stays where it is and stays in use until a new
        one has been written and moved over it."""
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

    def test_a_failed_overwrite_says_the_old_one_is_still_there(self):
        from quantem.finetune.job import adapter_job
        from quantem.finetune.tests.fixtures import FakeCancel, FakeReporter

        adapter_id = self._start().json()["adapter_id"]
        Adapter.objects.filter(id=adapter_id).update(head_path="adapters/keep/head.pt")
        payload = {
            "adapter_id": adapter_id,
            "segmentation_type_id": str(_mito().id),
            "asset_ids": self.asset_ids,
            "base_model": "quantem:nope",
            "overwrite": True,
        }
        with pytest.raises(ValueError):
            adapter_job(payload, FakeReporter(), FakeCancel())
        adapter = Adapter.objects.get(id=adapter_id)
        assert adapter.status == "FAILED"
        assert "untouched and still in use" in adapter.error
        assert adapter.head_path == "adapters/keep/head.pt"

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

    def test_an_image_outside_the_scope_is_refused(self):
        response = _post(
            self.client,
            f"/api/finetune/runs/{self.adapter.id}/apply/",
            {"asset_ids": [str(self.outsider.asset_id)]},
        )
        assert response.status_code == 400
        assert "not part of this fine-tune" in response.json()["detail"]

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

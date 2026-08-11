"""One run per image, across every organelle the user ticked (package P4).

Four things are pinned here, and each of them was broken before this package:

1. **The plan is exact and it is free.** ``GET /api/assets/<id>/runs/`` quotes
   the tile count :func:`quantem.inference.engine.estimate_tiles` will produce,
   for a pack that is not installed as much as for one that is, and it queues
   nothing.
2. **One job carries the sum.** Ticking three organelles produces one row whose
   ``progress_units_total`` is their three tile counts added up -- at enqueue,
   before any model loads.
3. **The count never runs backwards.** The second organelle's tiles are offset
   by the first's; a leg that fails hands on only the tiles it actually walked,
   so the bar cannot fill over work that was skipped.
4. **One organelle's failure does not throw away another's objects.** The
   segmentation that finished keeps its own stage and its own objects, and the
   job's summary error names both halves.
"""

from unittest import mock

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.inference import engine
from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.constants import (
    JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    QUEUE_P4_FULL,
)
from quantem.jobs.models import Job
from quantem.jobs.serializers import run_legs
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.type_definitions import MITOCHONDRIA, NUCLEUS
from quantem.testing import create_image_from_test_tiff

RUNS_URL = "/api/assets/{asset_id}/runs/"


class RunPlanTests(TestCase):
    """The costed plan, before anything is queued."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("One run plan")
        self.asset = self.image.asset

    def test_plan_quotes_engine_estimate_tiles_per_organelle(self):
        response = self.client.get(
            RUNS_URL.format(asset_id=self.asset.id),
            {"organelles": "mito,nucleus"},
        )
        self.assertEqual(response.status_code, 200)
        entries = {item["organelle"]: item for item in response.data["organelles"]}
        # Every organelle this image can be run for is offered, ticked or not:
        # a checklist that listed only the ticked ones could never be added to.
        self.assertEqual(set(entries), {"mito", "er", "nucleus", "ld"})
        self.assertEqual(response.data["selected"], ["mito", "nucleus"])

        shape = (self.image.height, self.image.width)
        pixel_size_nm = self.asset.pixel_size_nm or None
        for organelle, entry in entries.items():
            spec = MODEL_SPECS[entry["pack_id"]]
            expected = engine.estimate_tiles(spec, shape, pixel_size_nm=pixel_size_nm)
            self.assertEqual(
                entry["tiles"],
                expected,
                f"{organelle} was costed at {entry['tiles']}, not {expected}",
            )

    def test_plan_total_covers_the_ticked_organelles_only(self):
        response = self.client.get(
            RUNS_URL.format(asset_id=self.asset.id),
            {"organelles": "mito,nucleus"},
        )
        chosen = [
            item for item in response.data["organelles"] if item["organelle"] in {"mito", "nucleus"}
        ]
        self.assertEqual(response.data["tiles_total"], sum(item["tiles"] for item in chosen))
        # Costing the run is free: nothing is created and nothing is queued
        # while the user is still deciding.
        self.assertEqual(Job.objects.count(), 0)
        self.assertEqual(ImageSegmentation.objects.count(), 0)

    def test_unticking_everything_leaves_no_total_to_quote(self):
        response = self.client.get(RUNS_URL.format(asset_id=self.asset.id), {"organelles": ""})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.data["tiles_total"])
        self.assertEqual(response.data["download_bytes_total"], 0)

    def test_plan_refuses_an_organelle_it_cannot_run(self):
        response = self.client.get(
            RUNS_URL.format(asset_id=self.asset.id),
            {"organelles": "mito,tissue"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["invalid"], ["tissue"])

    def test_download_figure_is_deduped_across_one_family(self):
        """Three QuantEM packs share one encoder; the figure must count it once."""
        from quantem.registry.catalogue import deduped_download_bytes, download_bytes

        pack_ids = ["quantem:mito", "quantem:nucleus", "quantem:ld"]
        naive = sum(download_bytes(MODEL_SPECS[pack_id]) for pack_id in pack_ids)
        deduped = deduped_download_bytes(pack_ids)
        one = deduped_download_bytes(["quantem:mito"])
        head_only = deduped_download_bytes(pack_ids) - deduped_download_bytes(
            ["quantem:nucleus", "quantem:ld"]
        )
        encoder = one - head_only
        self.assertGreater(encoder, 0, "the family shares no encoder to dedupe")
        # Adding the three figures up counts the shared encoder three times;
        # the deduped figure counts it once.
        self.assertEqual(deduped, naive - 2 * encoder)


class RunEnqueueTests(TestCase):
    """One job, one denominator, from the moment the user presses go."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("One run enqueue")
        self.asset = self.image.asset

    def _start(self, organelles):
        return self.client.post(
            RUNS_URL.format(asset_id=self.asset.id),
            {"organelles": organelles},
            format="json",
        )

    def test_three_organelles_produce_one_job(self):
        response = self._start(["mito", "nucleus", "ld"])
        self.assertEqual(response.status_code, 202)
        jobs = list(Job.objects.all())
        self.assertEqual([job.type for job in jobs], [JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE])
        self.assertEqual(jobs[0].queue_name, QUEUE_P4_FULL)
        self.assertEqual(len(jobs[0].payload_json["legs"]), 3)
        # Not one full-image job per organelle, which is what this replaces.
        self.assertFalse(Job.objects.filter(type=JOB_TYPE_RUN_SEGMENTATION_FULL).exists())

    def test_the_single_job_carries_the_summed_tile_plan_while_still_queued(self):
        response = self._start(["mito", "nucleus"])
        job = Job.objects.get()
        self.assertEqual(job.status, "PENDING")
        expected = response.data["plan"]["tiles_total"]
        self.assertIsNotNone(expected)
        self.assertEqual(job.progress_units_total, expected)
        self.assertEqual(job.progress_units_done, 0)
        self.assertEqual(job.progress_unit_label, "tile")

    def test_the_queued_run_is_the_whole_denominator_of_its_wave(self):
        """The HIGH finding: a queued run contributes its full plan at once."""
        from quantem.jobs.serializers import batch_progress_for

        self._start(["mito", "nucleus"])
        job = Job.objects.get()
        batch = batch_progress_for(job)
        self.assertIsNotNone(batch)
        self.assertEqual(batch["units_total"], job.progress_units_total)
        self.assertEqual(batch["units_done"], 0)
        self.assertEqual(batch["percent"], 0.0)

    def test_starting_with_nothing_ticked_is_refused_in_plain_words(self):
        response = self.client.post(
            RUNS_URL.format(asset_id=self.asset.id), {"organelles": []}, format="json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least one", response.data["detail"])

    def test_a_second_run_over_a_held_organelle_is_refused_before_it_is_queued(self):
        self._start(["mito"])
        self.assertEqual(Job.objects.count(), 1)
        response = self._start(["mito", "nucleus"])
        self.assertEqual(response.status_code, 409)
        # And nothing was queued behind the refusal.
        self.assertEqual(Job.objects.count(), 1)


class RunLegProgressTests(TestCase):
    """The per-organelle lines that share one job row."""

    def setUp(self):
        self.image = create_image_from_test_tiff("One run legs")
        self.asset = self.image.asset

    def _job_with_legs(self, legs):
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
            payload_json={"asset_id": str(self.asset.id)},
            progress_detail_json={"legs": legs},
        )
        return job

    def test_legs_are_published_with_their_own_share_of_the_tiles(self):
        job = self._job_with_legs(
            [
                {
                    "segmentation_id": "a",
                    "name": "Mitochondria",
                    "status": "SUCCESS",
                    "units_done": 858,
                    "units_total": 858,
                    "unit_label": "tile",
                },
                {
                    "segmentation_id": "b",
                    "name": "Nucleus",
                    "status": "RUNNING",
                    "units_done": 19,
                    "units_total": 88,
                    "unit_label": "tile",
                },
            ]
        )
        rows = run_legs(job)
        self.assertEqual([row["name"] for row in rows], ["Mitochondria", "Nucleus"])
        self.assertEqual(rows[0]["percent"], 100.0)
        self.assertEqual(rows[1]["percent"], 21.6)

    def test_a_job_with_no_legs_reports_none(self):
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload_json={},
            progress_detail_json={},
        )
        self.assertIsNone(run_legs(job))

    def test_a_leg_cannot_report_more_tiles_than_it_planned(self):
        job = self._job_with_legs(
            [
                {
                    "segmentation_id": "a",
                    "name": "Mitochondria",
                    "status": "RUNNING",
                    "units_done": 900,
                    "units_total": 858,
                    "unit_label": "tile",
                }
            ]
        )
        self.assertEqual(run_legs(job)[0]["units_done"], 858)


class SharedRowTests(TestCase):
    """Two writers, one row, and the keys neither of them owns."""

    def setUp(self):
        self.image = create_image_from_test_tiff("Shared row")
        self.job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
            payload_json={"asset_id": str(self.image.asset.id)},
            progress_detail_json={
                "legs": [
                    {
                        "segmentation_id": "a",
                        "name": "Mitochondria",
                        "status": "RUNNING",
                        "units_done": 0,
                        "units_total": 858,
                        "unit_label": "tile",
                    }
                ]
            },
        )

    def test_a_tile_write_does_not_erase_the_per_organelle_lines(self):
        """`progress_detail_json` is written whole; the legs must survive it."""
        from quantem.jobs.reporter import JobReporter, unit_window

        reporter = JobReporter(str(self.job.id), min_interval_seconds=0.0)
        with unit_window(0, 946):
            with reporter.unit_scope(total=858, label="tile", stage="inference") as scope:
                scope.set(12)
        self.job.refresh_from_db()
        detail = self.job.progress_detail_json or {}
        self.assertIn("legs", detail, "the per-organelle lines were wiped")
        # The wave's denominator, not this organelle's 858.
        self.assertEqual(self.job.progress_units_total, 946)
        self.assertEqual(self.job.progress_units_done, 12)

    def test_the_tiling_loops_own_scope_reports_into_the_wave(self):
        from quantem.jobs.reporter import JobReporter, unit_window

        reporter = JobReporter(str(self.job.id), min_interval_seconds=0.0)
        with unit_window(858, 946):
            scope = reporter.unit_scope(total=88, label="tile", stage="inference")
            scope.set(19)
            scope._write(force=True)
        self.job.refresh_from_db()
        # 858 already walked plus 19 of this organelle's 88, against the wave.
        self.assertEqual(self.job.progress_units_done, 877)
        self.assertEqual(self.job.progress_units_total, 946)


class TileWindowTests(TestCase):
    """The arithmetic that keeps one row's count monotone across organelles."""

    def test_the_second_organelle_continues_the_first_ones_count(self):
        from quantem.seg_core.db.inference import TileWindow

        window = TileWindow(base=0, total=946)
        window.note_walked(858)
        window.advance()
        self.assertEqual(window.base, 858)
        window.note_walked(19)
        self.assertEqual(window.base + window.walked, 877)

    def test_a_leg_that_stopped_early_hands_on_only_what_it_walked(self):
        from quantem.seg_core.db.inference import TileWindow

        window = TileWindow(base=0, total=946)
        window.note_walked(18)  # failed at tile 18 of 858
        window.advance()
        self.assertEqual(window.base, 18)

    def test_a_late_smaller_sample_cannot_pull_the_count_back(self):
        from quantem.seg_core.db.inference import TileWindow

        window = TileWindow(base=100, total=946)
        window.note_walked(40)
        window.note_walked(12)
        self.assertEqual(window.walked, 40)


class ImageRunDriverTests(TestCase):
    """The driver itself, with inference stubbed out."""

    def setUp(self):
        self.image = create_image_from_test_tiff("One run driver")
        self.asset = self.image.asset
        self.client = APIClient()
        self.client.post(
            RUNS_URL.format(asset_id=self.asset.id),
            {"organelles": ["mito", "nucleus"]},
            format="json",
        )
        self.job = Job.objects.get()
        self.legs = self.job.payload_json["legs"]

    def _segmentation(self, definition):
        return ImageSegmentation.objects.get(
            asset=self.asset, segmentation_type__internal_name=definition.internal_name
        )

    def test_every_organelle_runs_and_each_gets_its_own_result(self):
        from quantem.segmentation import organelle_tasks

        calls = []

        def fake_run(**kwargs):
            calls.append(kwargs["segmentation_id"])
            return 7

        with mock.patch.object(organelle_tasks, "_run_segmentation", fake_run):
            outcome = organelle_tasks.run_segmentation_for_image_task(
                asset_id=str(self.asset.id),
                legs=[{"segmentation_id": leg["segmentation_id"]} for leg in self.legs],
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome["segment_count"], 0)
        self.assertEqual(outcome["stored_output_count"], 14)
        self.assertTrue(outcome["threshold_ready"])
        self.assertEqual({item["status"] for item in outcome["organelles"]}, {"SUCCESS"})

    def test_the_image_is_decoded_once_for_the_whole_run(self):
        from quantem.segmentation import organelle_tasks

        seen = []

        def fake_run(**kwargs):
            seen.append(id(kwargs["image_array"]))
            return 1

        with mock.patch.object(organelle_tasks, "_run_segmentation", fake_run):
            organelle_tasks.run_segmentation_for_image_task(
                asset_id=str(self.asset.id),
                legs=[{"segmentation_id": leg["segmentation_id"]} for leg in self.legs],
            )
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(set(seen)), 1, "each organelle re-decoded the image")

    def test_every_organelle_shares_one_tile_window(self):
        from quantem.segmentation import organelle_tasks

        windows = []

        def fake_run(**kwargs):
            window = kwargs["tile_window"]
            windows.append((window.base, window.total))
            window.note_walked(5)
            return 1

        with mock.patch.object(organelle_tasks, "_run_segmentation", fake_run):
            organelle_tasks.run_segmentation_for_image_task(
                asset_id=str(self.asset.id),
                legs=[{"segmentation_id": leg["segmentation_id"]} for leg in self.legs],
            )
        totals = {total for _base, total in windows}
        self.assertEqual(len(totals), 1, "the denominator changed between organelles")
        bases = [base for base, _total in windows]
        self.assertEqual(bases, sorted(bases), "the count ran backwards")
        self.assertGreater(bases[1], bases[0])

    def test_one_organelle_failing_does_not_undo_the_other(self):
        from quantem.segmentation import organelle_tasks

        first = self.legs[0]["segmentation_id"]

        def fake_run(**kwargs):
            if kwargs["segmentation_id"] == first:
                return 11
            raise RuntimeError("that model is not installed")

        with mock.patch.object(organelle_tasks, "_run_segmentation", fake_run):
            with self.assertRaises(RuntimeError) as caught:
                organelle_tasks.run_segmentation_for_image_task(
                    asset_id=str(self.asset.id),
                    legs=[{"segmentation_id": leg["segmentation_id"]} for leg in self.legs],
                )
        message = str(caught.exception)
        # Both halves are named: what did not finish, and what did.
        self.assertIn("did not finish", message)
        self.assertIn("unaffected", message)

    def test_a_finished_organelle_is_not_marked_failed_by_another_ones_error(self):
        """The reconciler must not write one leg's failure over another's success."""
        from quantem.jobs.failure_reconcile import (
            reconcile_domain_objects_for_failed_job,
        )

        done = self._segmentation(MITOCHONDRIA)
        done.status_stage = "CANDIDATES_READY"
        done.save(update_fields=["status_stage"])
        broken = self._segmentation(NUCLEUS)

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
            self.job.payload_json,
            "Nucleus did not finish.",
            supersede_stale_failure=True,
        )
        done.refresh_from_db()
        broken.refresh_from_db()
        self.assertEqual(done.status_stage, "CANDIDATES_READY")
        self.assertEqual(broken.status_stage, "FAILED")

    def test_a_crashed_run_does_not_strand_its_organelles_at_running(self):
        from quantem.jobs.failure_reconcile import (
            reconcile_domain_objects_for_failed_job,
        )

        for definition in (MITOCHONDRIA, NUCLEUS):
            segmentation = self._segmentation(definition)
            segmentation.status_stage = "RUNNING_INFERENCE"
            segmentation.save(update_fields=["status_stage"])

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
            self.job.payload_json,
            "The run stopped unexpectedly.",
            supersede_stale_failure=True,
        )
        for definition in (MITOCHONDRIA, NUCLEUS):
            self.assertEqual(
                self._segmentation(definition).status_stage,
                "FAILED",
                f"{definition.long_name} was left mid-run",
            )

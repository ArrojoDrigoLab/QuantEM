"""The three refusals and the one report the Improve panel is built on.

Each test here names the defect it is the fix for, because every one of them
was reachable through the shipping UI before this package:

* a threshold run with no probability map reached the queue and died there;
* a head run on a region too small to cut a window from did the same, minutes
  later, with a message about a "20 % coverage rule";
* the panel had no server-side way to find a run it had started, so a reload
  lost it and a *second* run could not be started at all;
* nothing told the user, at the moment they pressed the button, that their own
  kept, removed and hand-drawn objects survive a re-run.
"""

from __future__ import annotations

import math
import time

from django.urls import reverse
from rest_framework.test import APIClient

from quantem.finetune.job import adapter_job
from quantem.finetune.preflight import check_head_size, required_model_px
from quantem.finetune.tests.app_support import FinetuneAppTestCase
from quantem.finetune.tests.fixtures import (
    FakeCancel,
    FakeReporter,
    annotated_segmentation,
)
from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.models import Job
from quantem.segmentation.models import SegmentObject

#: A checked area 220 px across on the 5 nm test image: 1.1 µm.
SMALL = 320
SMALL_ROI = (20, 20, 240, 240)
SMALL_OBJECT = (60, 60, 160, 160)

#: Big enough for a 518 px window at 8 nm: 8 nm / 5 nm is a 1.6x downsample, so
#: 2 400 native px is 1 500 model px.
BIG = 2560
BIG_ROI = (40, 40, 2440, 2440)
BIG_OBJECT = (400, 400, 1200, 1200)


class HeadSizeRuleTests(FinetuneAppTestCase):
    """The geometry rule itself, against the trainer it is derived from."""

    def test_required_span_matches_the_trainers_own_coverage_rule(self):
        # 20 % of one tile's area, expressed as the side of a square. If
        # `build_patches` ever changes its rule this is the test that says so.
        for pack_id in ("quantem:mito", "omniem:mito"):
            spec = MODEL_SPECS[pack_id]
            expected = spec.tile_size * math.sqrt(0.2)
            assert required_model_px(spec) == expected
        # The numbers the sentences quote, pinned.
        assert round(required_model_px(MODEL_SPECS["quantem:mito"])) == 229
        assert round(required_model_px(MODEL_SPECS["omniem:mito"])) == 232

    def test_a_small_checked_area_is_refused_with_both_spans_in_micrometres(self):
        segmentation = annotated_segmentation(
            "Preflight small", size=SMALL, roi=SMALL_ROI, obj=SMALL_OBJECT
        )
        from quantem.segmentation.services.adapt import collect_crops

        crops = collect_crops(segmentation).crops
        assert len(crops) == 1

        verdict = check_head_size(crops, "omniem:mito")
        assert verdict is not None
        assert verdict.ok is False
        # The sentence UX_PLAN §2.3 asks for, verbatim.
        assert verdict.reason == ("Your checked area is 1.1 µm across; this needs about 1.9 µm.")

    def test_a_large_checked_area_passes(self):
        segmentation = annotated_segmentation(
            "Preflight big", size=BIG, roi=BIG_ROI, obj=BIG_OBJECT
        )
        from quantem.segmentation.services.adapt import collect_crops

        verdict = check_head_size(collect_crops(segmentation).crops, "quantem:mito")
        assert verdict is not None
        assert verdict.ok is True
        assert verdict.reason is None

    def test_an_unknown_pack_is_not_a_size_refusal(self):
        segmentation = annotated_segmentation(
            "Preflight unknown pack", size=SMALL, roi=SMALL_ROI, obj=SMALL_OBJECT
        )
        from quantem.segmentation.services.adapt import collect_crops

        crops = collect_crops(segmentation).crops
        assert check_head_size(crops, "quantem:golgi") is None
        assert check_head_size([], "quantem:mito") is None


class CropsEndpointTests(FinetuneAppTestCase):
    def setUp(self):
        self.client = APIClient()

    def test_head_size_reason_reaches_mode_blockers_when_a_pack_is_named(self):
        segmentation = annotated_segmentation(
            "Crops small", size=SMALL, roi=SMALL_ROI, obj=SMALL_OBJECT
        )
        url = reverse("adapt-crops", args=[segmentation.id])

        # Without a pack there is no pack-specific rule to apply, so head is
        # only blocked by what blocks everything.
        body = self.client.get(url).json()
        assert body["head_size"] is None
        assert body["mode_blockers"]["head"] == []

        body = self.client.get(url, {"base_model": "omniem:mito"}).json()
        assert body["head_size"]["ok"] is False
        assert body["mode_blockers"]["head"] == [
            "Your checked area is 1.1 µm across; this needs about 1.9 µm."
        ]
        # The always-available rung is untouched by a geometry problem.
        assert body["mode_blockers"]["threshold_only"] == []

    def test_every_crop_carries_its_image_name_and_never_a_bare_identifier(self):
        first = annotated_segmentation("Grid2 Cell11")
        annotated_segmentation("Grid2 Cell12")
        body = self.client.get(reverse("adapt-crops", args=[first.id])).json()

        names = {crop["image_name"] for crop in body["crops"]}
        assert names == {"Grid2 Cell11", "Grid2 Cell12"}
        mine = [crop for crop in body["crops"] if crop["is_this_image"]]
        assert len(mine) == 1
        assert mine[0]["image_name"] == "Grid2 Cell11"

    def test_no_probability_map_blocks_only_threshold_calibration(self):
        segmentation = annotated_segmentation("Crops no prob", with_prob=False)
        body = self.client.get(reverse("adapt-crops", args=[segmentation.id])).json()
        assert body["has_probability"] is False
        assert body["mode_blockers"]["threshold_only"]
        assert body["mode_blockers"]["head"] == []


class StartRefusalTests(FinetuneAppTestCase):
    """Refused *before* it is queued -- the whole point of the pre-flight."""

    def setUp(self):
        self.client = APIClient()

    def _start(self, segmentation, **payload):
        return self.client.post(
            reverse("adapt-start", args=[segmentation.id]),
            {"base_model": "quantem:mito", **payload},
            format="json",
        )

    def test_threshold_run_with_no_probability_map_is_refused_and_nothing_queued(self):
        segmentation = annotated_segmentation("Start no prob", with_prob=False)
        before = Job.objects.count()

        response = self._start(segmentation, mode="threshold_only")

        assert response.status_code == 400
        assert "probability" in response.json()["error"].lower()
        assert Job.objects.count() == before
        assert not self.Adapter.objects.filter(segmentation=segmentation).exists()

    def test_head_run_on_a_small_checked_area_is_refused_with_the_two_spans(self):
        segmentation = annotated_segmentation(
            "Start small", size=SMALL, roi=SMALL_ROI, obj=SMALL_OBJECT
        )
        before = Job.objects.count()

        response = self._start(segmentation, base_model="omniem:mito", mode="head")

        assert response.status_code == 400
        assert response.json()["error"] == (
            "Your checked area is 1.1 µm across; this needs about 1.9 µm."
        )
        assert Job.objects.count() == before
        assert not self.Adapter.objects.filter(segmentation=segmentation).exists()

    def test_a_runnable_threshold_run_still_starts(self):
        segmentation = annotated_segmentation("Start fine")
        response = self._start(segmentation, mode="threshold_only")
        assert response.status_code == 202
        job = Job.objects.get(id=response.json()["job_id"])
        assert job.payload_json["apply_and_rerun"] is False


class LatestRunTests(FinetuneAppTestCase):
    """Reattachment comes from the server, so a second run is always possible."""

    def setUp(self):
        self.client = APIClient()
        self.segmentation = annotated_segmentation("Latest")

    def test_nothing_run_yet(self):
        body = self.client.get(reverse("adapt-latest", args=[self.segmentation.id])).json()
        assert body == {"adapter": None, "job_id": None}

    def test_the_most_recent_run_and_its_job_are_found_without_browser_storage(self):
        started = self.client.post(
            reverse("adapt-start", args=[self.segmentation.id]),
            {"base_model": "quantem:mito", "mode": "threshold_only"},
            format="json",
        ).json()

        body = self.client.get(reverse("adapt-latest", args=[self.segmentation.id])).json()
        assert body["adapter"]["id"] == started["adapter_id"]
        assert body["job_id"] == started["job_id"]
        assert body["adapter"]["status"] == "PENDING"

    def test_a_second_run_supersedes_the_first(self):
        first = self.client.post(
            reverse("adapt-start", args=[self.segmentation.id]),
            {"base_model": "quantem:mito", "mode": "threshold_only"},
            format="json",
        ).json()
        self.Adapter.objects.filter(id=first["adapter_id"]).update(status="SUCCESS")

        second = self.client.post(
            reverse("adapt-start", args=[self.segmentation.id]),
            {"base_model": "quantem:mito", "mode": "threshold_only"},
            format="json",
        ).json()
        assert second["adapter_id"] != first["adapter_id"]

        body = self.client.get(reverse("adapt-latest", args=[self.segmentation.id])).json()
        assert body["adapter"]["id"] == second["adapter_id"]


class ApplyAndRerunJobTests(FinetuneAppTestCase):
    """One button means calibrate *and* use, without a second hunt for Apply."""

    def setUp(self):
        self.segmentation = annotated_segmentation("Apply and rerun")

    def _payload(self, adapter, **extra):
        return {
            "segmentation_id": str(self.segmentation.id),
            "adapter_id": str(adapter.id),
            "base_model": "quantem:mito",
            "mode": "threshold_only",
            **extra,
        }

    def _adapter(self):
        return self.Adapter.objects.create(
            segmentation=self.segmentation,
            base_model="quantem:mito",
            mode="threshold_only",
        )

    def test_the_result_says_what_a_rerun_would_change_even_when_not_applied(self):
        adapter = self._adapter()
        result = adapter_job(self._payload(adapter), FakeReporter(), FakeCancel())

        block = result["apply_and_rerun"]
        assert block["requested"] is False
        assert block["applied"] is False
        assert block["previous_include_level"] == MODEL_SPECS["quantem:mito"].threshold
        assert block["include_level"] == result["sweep"]["calibrated_threshold"]
        assert block["preserves_manual_work"] is True
        assert "kept, removed or drawn by hand" in block["preservation"]

        adapter.refresh_from_db()
        assert adapter.applied_at is None

    def test_asking_for_it_stamps_the_run_and_reports_the_rerun_as_still_pending(self):
        adapter = self._adapter()
        result = adapter_job(
            self._payload(adapter, apply_and_rerun=True), FakeReporter(), FakeCancel()
        )

        block = result["apply_and_rerun"]
        assert block["requested"] is True
        assert block["applied"] is True
        adapter.refresh_from_db()
        assert adapter.applied_at is not None

        # Applying writes no object. The re-run is a separate, stated-cost act,
        # so the result says it has not happened rather than implying it has.
        assert block["rerun_pending"] is block["changes_objects"]
        assert (
            SegmentObject.objects.filter(
                segmentation=self.segmentation, label_state="CONFIRMED"
            ).count()
            == 1
        )

    def test_calibration_finishes_well_inside_the_two_seconds_the_panel_promises(self):
        # The panel says "usually about a second" before the button is pressed
        # (I-13: a stated cost is the real cost). The bound here is deliberately
        # loose -- it is a regression guard against something turning the sweep
        # into an inference run, not a benchmark.
        adapter = self._adapter()
        started = time.perf_counter()
        adapter_job(self._payload(adapter), FakeReporter(), FakeCancel())
        assert time.perf_counter() - started < 2.0


class ApplyAdviceTests(FinetuneAppTestCase):
    def setUp(self):
        self.client = APIClient()
        self.segmentation = annotated_segmentation("Apply advice")

    def _adapter(self, threshold: float):
        return self.Adapter.objects.create(
            segmentation=self.segmentation,
            base_model="quantem:mito",
            status="SUCCESS",
            mode="threshold_only",
            calibrated_threshold=threshold,
            sweep={"calibrated_threshold": threshold},
        )

    def test_apply_reports_what_a_rerun_would_change_and_what_it_would_not(self):
        default = MODEL_SPECS["quantem:mito"].threshold
        adapter = self._adapter(default + 0.15)

        body = self.client.post(
            reverse("adapter-apply", args=[adapter.id]), {}, format="json"
        ).json()

        advice = body["rerun_advice"]
        assert advice["previous_include_level"] == default
        assert advice["changes_objects"] is True
        assert advice["preserves_manual_work"] is True
        assert "kept, removed or drawn by hand" in advice["preservation"]
        # The include level a reader would have to write in a methods section.
        assert body["default_threshold"] == default

    def test_an_unchanged_include_level_says_so_rather_than_inviting_a_pointless_run(
        self,
    ):
        default = MODEL_SPECS["quantem:mito"].threshold
        adapter = self._adapter(default)
        body = self.client.post(
            reverse("adapter-apply", args=[adapter.id]), {}, format="json"
        ).json()
        assert body["rerun_advice"]["changes_objects"] is False
        assert "same objects" in body["rerun_advice"]["summary"]

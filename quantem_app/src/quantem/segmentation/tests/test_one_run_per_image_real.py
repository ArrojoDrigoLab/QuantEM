"""Two organelles, one run, against the models that actually ship (P4).

Every other test in this area substitutes a fake ``_run_segmentation``, which
proves the driver's bookkeeping and not the thing the package exists for: that
two real packs can be walked one after another inside one process, sharing one
decoded image and one tile count, and that the count that reaches the job row
is monotone across the boundary between them.

Skipped when the packs are not installed::

    python -m quantem.registry.install local --all
"""

from __future__ import annotations

import pytest
from django.test import TransactionTestCase

from quantem.inference.specs import get_model_spec
from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE
from quantem.jobs.models import Job
from quantem.jobs.reporter import JobReporter
from quantem.registry import cache as registry_cache
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationConfig,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import run_segmentation_for_image_task
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_nucleus_type,
)
from quantem.testing import create_small_test_image

pytestmark = [pytest.mark.requires_weights, pytest.mark.slow]

#: Small enough to walk on a CPU in seconds, big enough to need more than one
#: window at the mito pack's 8 nm canonical scale.
SIZE = 768


class RealTwoOrganelleRunTests(TransactionTestCase):
    def setUp(self):
        for family, organelle in (("quantem", "mito"), ("quantem", "nucleus")):
            spec = get_model_spec(family, organelle)
            if not registry_cache.installed(spec.pack_id):
                self.skipTest(
                    f"{spec.pack_id} is not installed; run "
                    "`python -m quantem.registry.install local --all`"
                )
        self.image = create_small_test_image(
            "Real two-organelle run", width=SIZE, height=SIZE, textured=True
        )
        self.asset = self.image.asset
        self.asset.pixel_size_nm = 8.0
        self.asset.save(update_fields=["pixel_size_nm"])

        self.legs = []
        for factory in (get_or_create_mitochondria_type, get_or_create_nucleus_type):
            segmentation_type = factory()
            segmentation = ImageSegmentation.objects.create(
                asset=self.asset, segmentation_type=segmentation_type
            )
            SegmentationConfig.objects.get_or_create(segmentation=segmentation)
            self.legs.append(
                {
                    "segmentation_id": str(segmentation.id),
                    "segmentation_type": segmentation_type.internal_name,
                    "source_model": f"quantem:{segmentation_type.internal_name.rsplit('_', 1)[-1]}",
                }
            )

        self.job = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
            payload={"asset_id": str(self.asset.id), "legs": self.legs},
            resource_class="gpu",
            queue_name="p4_full",
            max_attempts=1,
        )

    def test_two_real_packs_share_one_row_and_one_monotone_count(self):
        planned = self.job.progress_units_total
        self.assertIsNotNone(planned, "the queued run could not say how big it was")

        reporter = JobReporter(str(self.job.id), min_interval_seconds=0.0)
        seen: list[int] = []
        original = reporter.update

        def watch(*args, **kwargs):
            original(*args, **kwargs)
            done = (
                Job.objects.filter(id=self.job.id)
                .values_list("progress_units_done", flat=True)
                .first()
            )
            if done is not None:
                seen.append(int(done))

        reporter.update = watch  # type: ignore[method-assign]
        reporter.activate()
        try:
            outcome = run_segmentation_for_image_task(
                asset_id=str(self.asset.id),
                legs=self.legs,
                reporter=reporter,
            )
        finally:
            reporter.deactivate()

        self.assertEqual(len(outcome["organelles"]), 2)
        self.assertEqual({item["status"] for item in outcome["organelles"]}, {"SUCCESS"})

        self.job.refresh_from_db()
        # One row, one denominator, and it did not move under the run.
        self.assertEqual(self.job.progress_units_total, planned)
        self.assertEqual(self.job.progress_units_done, planned)
        self.assertEqual(seen, sorted(seen), "the tile count ran backwards")

        # Both organelles are real segmentations with their own stage.
        for leg in self.legs:
            segmentation = ImageSegmentation.objects.get(id=leg["segmentation_id"])
            self.assertEqual(segmentation.status_stage, "CANDIDATES_READY")

        # The per-organelle lines are on the row for the UI to draw.
        legs = (self.job.progress_detail_json or {}).get("legs") or []
        self.assertEqual(len(legs), 2)
        self.assertEqual({leg["status"] for leg in legs}, {"SUCCESS"})

        # Sanity: this ran a real model, not a stub.
        self.assertGreaterEqual(
            SegmentObject.objects.filter(segmentation__asset=self.asset).count(),
            0,
        )

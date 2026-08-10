"""The persistence fix, against the segmenter that actually ships.

Every other test in this area substitutes a fake segmenter, which proves the
plumbing but not the thing that was broken: ``DinoOrganelleSegmenter`` declares
``persist_probability_maps = False``, and the whole bug was that no map ever
reached disk for *that* class. So this one runs the released mito pack on a
synthetic micrograph and then asks the crop reader whether guided fine-tuning is
reachable — the exact question the user's UI asks.

Skipped when the pack is not installed::

    python -m quantem.registry.install local --all
"""

from __future__ import annotations

import pytest
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.inference.specs import get_model_spec
from quantem.registry import cache as registry_cache
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    ProbabilityMap,
    SegmentationConfig,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import run_segmentation_full_task
from quantem.segmentation.prob_maps.io import resolve_probability_map_path
from quantem.segmentation.prob_maps.persistence import SCOPE_FULL
from quantem.segmentation.services.adapt import collect_crops
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

pytestmark = [pytest.mark.requires_weights, pytest.mark.slow]

#: Two 512 px windows across at 8 nm/px, small enough to run on a CPU in seconds.
SIZE = 768
MITO_INTERNAL_NAME = "quantem_internal_mito"


def _square(x0, y0, x1, y1) -> Polygon:
    return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))


class RealModelProbabilityMapTests(TestCase):
    def setUp(self):
        spec = get_model_spec("quantem", "mito")
        if not registry_cache.installed(spec.pack_id):
            self.skipTest(
                f"{spec.pack_id} is not installed; run "
                "`python -m quantem.registry.install local --all`"
            )
        self.image = create_small_test_image(
            "Real mito run", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)

    def test_a_real_run_leaves_a_map_that_makes_calibration_reachable(self):
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(64, 64, 448, 448)
        )
        polygon = _square(100, 100, 200, 200)
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            label_state="CONFIRMED",
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )

        blocked = collect_crops(self.segmentation, require_probability=True)
        assert not blocked.ready
        assert "probability map" in blocked.blockers[0]

        run_segmentation_full_task(
            segmentation_id=str(self.segmentation.id),
            segmentation_type=MITO_INTERNAL_NAME,
            source_model="quantem:mito",
        )

        (stored,) = ProbabilityMap.objects.filter(segmentation=self.segmentation)
        path = resolve_probability_map_path(stored)
        assert path.exists()
        assert stored.metadata["run_scope"] == SCOPE_FULL
        # Provenance the segmenter reports, so a stored map can be traced to the
        # pack and threshold that produced it.
        assert stored.metadata["pack_id"] == "quantem:mito"
        assert stored.metadata["organelle"] == "mito"

        unblocked = collect_crops(
            self.segmentation, require_probability=True, load_prob=True
        )
        assert unblocked.ready, unblocked.blockers
        (crop,) = unblocked.crops
        assert crop.prob is not None
        assert crop.prob.shape == (384, 384)
        assert 0.0 <= float(crop.prob.min()) <= float(crop.prob.max()) <= 1.0

"""A run must leave behind a probability map guided fine-tuning can read.

The bug these cover: ``persist_probability_maps`` is ``False`` for the shipped
organelle segmenters, so nothing ever wrote a map, while ``collect_crops``
treated a missing map as a hard blocker for *every* adaptation mode. The user
was told to "run the model on this image first", ran it, and nothing changed —
there was no sequence of actions that reached guided fine-tuning at all.

So these tests are written as the round trip: run the segmenter, then ask the
crop reader whether adaptation can proceed.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from django.test import TestCase
from shapely.geometry import Polygon
from skimage.measure import label, regionprops

from quantem.assets.utils import create_roi_image_from_image
from quantem.seg_core.base_segmenter import BaseSegmenter
from quantem.seg_core.extraction import build_segment_from_region
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    ProbabilityMap,
    SegmentationConfig,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import (
    run_segmentation_full_task,
    run_segmentation_roi_task,
)
from quantem.segmentation.prob_maps.io import resolve_probability_map_path
from quantem.segmentation.prob_maps.persistence import (
    MAX_MEGAPIXELS_ENV,
    SCOPE_FULL,
    SCOPE_ROI,
    ProbabilityMapPersistenceError,
    persist_run_probability_maps,
)
from quantem.segmentation.services.adapt import collect_crops
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image, set_env

SIZE = 256
MITO_INTERNAL_NAME = "quantem_internal_mito"


def _square(x0, y0, x1, y1) -> Polygon:
    return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))


class FakeOrganelleSegmenter(BaseSegmenter):
    """A mito segmenter with the shipped models' persistence policy.

    ``persist_probability_maps`` is ``False`` here for the same reason it is
    ``False`` on ``DinoOrganelleSegmenter``: a stored uint8 map must never be
    replayed in place of running the model. That is exactly the configuration
    under which nothing was being written, so it is the one worth testing.
    """

    def __init__(self, prob: np.ndarray) -> None:
        self._prob = prob

    @property
    def name(self) -> str:
        return "mito"

    @property
    def generated_flag(self) -> str:
        return "mito_generated"

    @property
    def prob_map_prefix(self) -> str:
        return "mito"

    @property
    def source_model(self) -> str:
        return "quantem:mito"

    @property
    def persist_probability_maps(self) -> bool:
        return False

    def load_models(self) -> None:
        return None

    def get_dl_model_names(self) -> list[str]:
        return ["DINO"]

    def run_dl_inference(self, image, cached_prob_maps, on_progress=None, **kwargs):
        _ = (cached_prob_maps, on_progress, kwargs)
        return {"DINO": self._prob[: image.shape[0], : image.shape[1]].copy()}

    def combine_prob_maps(self, prob_maps):
        return prob_maps["DINO"]

    def extract_instances(
        self,
        prob,
        image,
        prob_maps,
        *,
        min_area=0,
        coordinate_offset=None,
        on_progress=None,
    ):
        _ = on_progress
        dx, dy = coordinate_offset or (0.0, 0.0)
        labels = label(prob > 0.5)
        segments = []
        for region in regionprops(labels, intensity_image=image):
            if region.area < max(int(min_area), 1):
                continue
            segment = build_segment_from_region(
                region,
                labels,
                prob_maps,
                prob,
                self.generated_flag,
                float(dx),
                float(dy),
                image,
            )
            if segment is not None:
                segments.append(segment)
        return segments

    def get_probability_map_metadata(self, model_name: str) -> dict[str, object]:
        return {"model_name": model_name, "pack_id": "fake:mito", "threshold": 0.5}


class _Reporter:
    """Records what a job would have shown the user."""

    def __init__(self) -> None:
        self.updates: list[tuple[float | None, str | None]] = []
        self.logs: list[tuple[str, str]] = []

    def update(self, progress=None, message=None) -> None:
        self.updates.append((progress, message))

    def log(self, level: str, message: str) -> None:
        self.logs.append((level, message))


def _blob_prob(size: int = SIZE) -> np.ndarray:
    prob = np.full((size, size), 0.02, dtype=np.float32)
    prob[40:100, 40:100] = 0.95
    return prob


class ProbabilityMapPersistedByRunTests(TestCase):
    """The round trip: run the model, then ask whether adaptation can proceed."""

    def setUp(self):
        self.image = create_small_test_image("Prob map run", width=SIZE, height=SIZE, textured=True)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)

    def _annotate(self):
        """A completed ROI with a confirmed object inside it: the other half of
        what calibration needs, so the only thing left to test is the map."""
        CompletedROI.objects.create(
            segmentation=self.segmentation, geometry=_square(20, 20, 140, 140)
        )
        polygon = _square(40, 40, 100, 100)
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            label_state="CONFIRMED",
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )

    def _run_full(self, prob: np.ndarray | None = None) -> int:
        segmenter = FakeOrganelleSegmenter(prob if prob is not None else _blob_prob())
        with patch("quantem.segmentation.organelle_tasks.get_segmenter", return_value=segmenter):
            return run_segmentation_full_task(
                segmentation_id=str(self.segmentation.id),
                segmentation_type=MITO_INTERNAL_NAME,
                source_model="quantem:mito",
            )

    def test_a_full_run_stores_a_map_that_unblocks_threshold_calibration(self):
        self._annotate()
        before = collect_crops(self.segmentation, require_probability=True)
        assert not before.ready
        assert "probability map" in before.blockers[0]

        self._run_full()

        (stored,) = ProbabilityMap.objects.filter(segmentation=self.segmentation)
        assert stored.name == "MITO_DINO"
        assert resolve_probability_map_path(stored).exists()
        assert stored.metadata["run_scope"] == SCOPE_FULL
        assert stored.metadata["pack_id"] == "fake:mito"

        after = collect_crops(self.segmentation, require_probability=True, load_prob=True)
        assert after.ready, after.blockers
        assert after.has_probability
        (crop,) = after.crops
        # The map really is the one the run produced: the blob is inside the
        # confirmed object and the rest of the ROI is background.
        assert crop.prob is not None
        assert crop.prob[40, 40] > 0.9  # image (60, 60), inside the blob
        assert crop.prob[110, 110] < 0.1  # image (130, 130), outside it

    def test_rerunning_replaces_the_map_instead_of_accumulating_rows(self):
        self._run_full()
        self._run_full()
        self._run_full()

        maps = list(ProbabilityMap.objects.filter(segmentation=self.segmentation))
        assert len(maps) == 1
        assert resolve_probability_map_path(maps[0]).exists()

    def test_an_roi_run_records_the_window_so_it_is_read_at_the_right_offset(self):
        self._annotate()
        roi = create_roi_image_from_image(
            self.image, x=20, y=20, width=120, height=120, source="AUTO", is_active=True
        )
        prob = np.full((SIZE, SIZE), 0.02, dtype=np.float32)
        prob[40:100, 40:100] = 0.95
        segmenter = FakeOrganelleSegmenter(prob[20:140, 20:140])

        with patch("quantem.segmentation.organelle_tasks.get_segmenter", return_value=segmenter):
            run_segmentation_roi_task(
                segmentation_id=str(self.segmentation.id),
                segmentation_type=MITO_INTERNAL_NAME,
                roi_id=str(roi.id),
                source_model="quantem:mito",
            )

        window_map = ProbabilityMap.objects.get(
            segmentation=self.segmentation, metadata__run_scope=SCOPE_ROI
        )
        assert window_map.metadata["roi"] == {
            "x": 20,
            "y": 20,
            "width": 120,
            "height": 120,
        }

        crop_set = collect_crops(self.segmentation, require_probability=True, load_prob=True)
        assert crop_set.ready, crop_set.blockers
        (crop,) = crop_set.crops
        # Without the recorded window the map would be read as if it started at
        # (0, 0) and the model would be scored against the wrong pixels.
        assert crop.prob is not None
        assert crop.prob[20, 20] > 0.9  # image (40, 40)
        assert crop.prob[110, 110] < 0.1  # image (130, 130)

    def test_a_configured_full_image_ceiling_refuses_with_its_real_reason(self):
        details: list[str] = []
        segmenter = FakeOrganelleSegmenter(_blob_prob())
        # SIZE*SIZE = 65536 px = 0.065 MP, so a 0.01 MP ceiling refuses it.
        with set_env({MAX_MEGAPIXELS_ENV: "0.01"}):
            with self.assertRaises(ProbabilityMapPersistenceError) as caught:
                persist_run_probability_maps(
                    segmentation=self.segmentation,
                    segmenter=segmenter,
                    prob_maps={"DINO": _blob_prob()},
                    on_detail=details.append,
                )

        assert ProbabilityMap.objects.filter(segmentation=self.segmentation).count() == 0
        assert "configured not to keep" in str(caught.exception)
        assert any("Set that value to 0" in message for message in details)

    def test_a_configured_ceiling_does_not_masquerade_as_a_disk_space_error(self):
        reporter = _Reporter()
        segmenter = FakeOrganelleSegmenter(_blob_prob())
        with (
            set_env({MAX_MEGAPIXELS_ENV: "0.01"}),
            patch("quantem.segmentation.organelle_tasks.get_segmenter", return_value=segmenter),
        ):
            with self.assertRaises(ProbabilityMapPersistenceError) as caught:
                run_segmentation_full_task(
                    segmentation_id=str(self.segmentation.id),
                    segmentation_type=MITO_INTERNAL_NAME,
                    source_model="quantem:mito",
                    reporter=reporter,
                )

        self.segmentation.refresh_from_db()
        assert "configured not to keep" in str(caught.exception)
        assert self.segmentation.status_stage == "FAILED"
        assert "configured not to keep" in self.segmentation.status_error
        assert "disk space" not in self.segmentation.status_error.lower()

    def test_a_segmenter_that_persists_its_own_maps_is_not_written_twice(self):
        segmenter = FakeOrganelleSegmenter(_blob_prob())
        with patch.object(
            FakeOrganelleSegmenter, "persist_probability_maps", property(lambda _self: True)
        ):
            written = persist_run_probability_maps(
                segmentation=self.segmentation,
                segmenter=segmenter,
                prob_maps={"DINO": _blob_prob()},
            )

        assert written == []
        assert ProbabilityMap.objects.filter(segmentation=self.segmentation).count() == 0


class EmptyRunReportingTests(TestCase):
    """A run that finds nothing must not look like a run that succeeded."""

    def setUp(self):
        self.image = create_small_test_image("Empty run", width=SIZE, height=SIZE, textured=True)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)

    def _run(self, prob: np.ndarray) -> tuple[int, _Reporter]:
        reporter = _Reporter()
        segmenter = FakeOrganelleSegmenter(prob)
        with patch("quantem.segmentation.organelle_tasks.get_segmenter", return_value=segmenter):
            count = run_segmentation_full_task(
                segmentation_id=str(self.segmentation.id),
                segmentation_type=MITO_INTERNAL_NAME,
                source_model="quantem:mito",
                reporter=reporter,
            )
        return count, reporter

    def test_a_run_saves_the_threshold_result_without_creating_candidates(self):
        stored_count, reporter = self._run(np.zeros((SIZE, SIZE), dtype=np.float32))

        assert stored_count == 1
        assert SegmentObject.objects.filter(segmentation=self.segmentation).count() == 0
        assert reporter.logs == []
        self.segmentation.refresh_from_db()
        assert self.segmentation.status_stage == "THRESHOLD_READY"
        assert self.segmentation.status_error == ""

    def test_a_nonempty_model_result_still_waits_for_apply(self):
        stored_count, reporter = self._run(_blob_prob())

        assert stored_count == 1
        assert SegmentObject.objects.filter(segmentation=self.segmentation).count() == 0
        assert [message for level, message in reporter.logs if level == "warning"] == []


class SegmentationJobHandlerOutcomeTests(TestCase):
    """The job's terminal message and result are what the queue UI reads."""

    def setUp(self):
        self.image = create_small_test_image(
            "Handler outcome", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)

    def test_zero_objects_is_reported_on_the_job_result(self):
        from quantem.jobs.handlers import _segmentation_run_outcome

        message, outcome = _segmentation_run_outcome(0)
        assert outcome["found_objects"] is False
        assert outcome["segment_count"] == 0
        assert outcome["next_steps"]
        assert "no objects found" in message

        message, outcome = _segmentation_run_outcome(7)
        assert outcome == {"segment_count": 7, "found_objects": True}
        assert "7 objects found" in message

"""Every model-produced object says which run, and which settings, made it.

The gap these close: an object recorded only ``source_model: "quantem:mito"``.
A run through a user-fitted adapter at threshold 0.31 and a released-pack run at
0.50 produced objects that were byte-for-byte indistinguishable, so an export
manifest could name the model and could not say which settings produced which
numbers.

The shape written here is a fixed contract shared with :mod:`quantem.analysis`;
see :mod:`quantem.segmentation.run_identity`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.segmentation.models import SegmentObject
from quantem.segmentation.run_identity import (
    RUN_FEATURE_KEY,
    RUN_IDENTITY_KEYS,
    build_run_identity,
    read_run_identity,
    resolve_ran_at_nm,
    run_identity_from_segmenter,
    utc_timestamp,
)
from quantem.testing import create_mitochondria_segmentation, create_small_test_image

pytestmark = pytest.mark.django_db


class RunIdentityPayloadTests(TestCase):
    def test_payload_carries_every_contract_key(self):
        payload = build_run_identity(
            run_id="job-1",
            pack_id="quantem:mito",
            threshold=0.45,
            adapter_id=None,
            ran_at_nm=8.0,
            native_pixel_size_nm=5.0,
            min_area=60,
        )
        self.assertEqual(tuple(payload), RUN_IDENTITY_KEYS)

    def test_nullable_fields_stay_null_rather_than_being_dropped(self):
        # A reader must be able to tell "ran at native scale" from "the writer
        # did not know about ran_at_nm", so the key is present and null.
        payload = build_run_identity(
            run_id="job-1",
            pack_id="quantem:er",
            threshold=0.5,
            adapter_id=None,
            ran_at_nm=None,
            native_pixel_size_nm=None,
            min_area=100,
        )
        self.assertIn("ran_at_nm", payload)
        self.assertIsNone(payload["ran_at_nm"])
        self.assertIsNone(payload["adapter_id"])
        self.assertIsNone(payload["native_pixel_size_nm"])

    def test_values_are_normalised_to_json_scalars(self):
        # A UUID or a numpy float reaching ``features`` makes the row
        # unserialisable at write time, deep inside the save loop.
        import uuid as uuid_module

        run_uuid = uuid_module.uuid4()
        payload = build_run_identity(
            run_id=run_uuid,
            pack_id="quantem:mito",
            threshold="0.45",
            adapter_id=uuid_module.UUID(int=3),
            ran_at_nm="8",
            native_pixel_size_nm="5",
            min_area=60.0,
        )
        self.assertEqual(payload["id"], str(run_uuid))
        self.assertEqual(payload["adapter_id"], str(uuid_module.UUID(int=3)))
        self.assertIsInstance(payload["threshold"], float)
        self.assertIsInstance(payload["min_area"], int)
        import json

        json.dumps(payload)  # must survive the JSONField round trip

    def test_finished_at_is_iso_utc(self):
        moment = datetime(2026, 8, 7, 9, 15, 2, 481000, tzinfo=UTC)
        payload = build_run_identity(
            run_id="job-1",
            pack_id="quantem:mito",
            threshold=0.5,
            adapter_id=None,
            ran_at_nm=None,
            native_pixel_size_nm=None,
            min_area=60,
            finished_at=moment,
        )
        self.assertEqual(payload["finished_at"], "2026-08-07T09:15:02.481Z")

    def test_a_non_utc_stamp_is_converted_not_relabelled(self):
        moment = datetime(
            2026, 8, 7, 11, 15, 2, 481000, tzinfo=timezone(timedelta(hours=2))
        )
        self.assertEqual(utc_timestamp(moment), "2026-08-07T09:15:02.481Z")

    def test_ran_at_nm_is_null_when_the_asset_is_uncalibrated(self):
        # No pixel size means nothing could resample; the run saw native pixels
        # and saying "8 nm" would document a run that did not happen.
        self.assertIsNone(resolve_ran_at_nm(canonical_nm=8.0, native_pixel_size_nm=None))
        self.assertIsNone(resolve_ran_at_nm(canonical_nm=8.0, native_pixel_size_nm=0))

    def test_ran_at_nm_is_null_for_a_native_resolution_pack(self):
        # ER declares canonical_nm=None: it runs at whatever the asset is.
        self.assertIsNone(resolve_ran_at_nm(canonical_nm=None, native_pixel_size_nm=5.0))

    def test_ran_at_nm_is_the_canonical_size_when_both_are_known(self):
        self.assertEqual(
            resolve_ran_at_nm(canonical_nm=8.0, native_pixel_size_nm=5.0), 8.0
        )


class ReadRunIdentityTests(TestCase):
    def test_absent_run_reads_as_none(self):
        # Absence means "no model produced this", which is a different fact from
        # "produced with unknown settings". No defaults are substituted.
        self.assertIsNone(read_run_identity({}))
        self.assertIsNone(read_run_identity(None))
        self.assertIsNone(read_run_identity({"sam_score": 0.5}))

    def test_a_run_without_an_id_is_not_a_run(self):
        self.assertIsNone(read_run_identity({RUN_FEATURE_KEY: {"pack_id": "x"}}))
        self.assertIsNone(read_run_identity({RUN_FEATURE_KEY: "quantem:mito"}))

    def test_a_stored_run_round_trips(self):
        payload = build_run_identity(
            run_id="job-1",
            pack_id="quantem:mito",
            threshold=0.45,
            adapter_id="adapter-3",
            ran_at_nm=8.0,
            native_pixel_size_nm=5.0,
            min_area=60,
        )
        self.assertEqual(read_run_identity({RUN_FEATURE_KEY: payload}), payload)


class RunIdentityFromSegmenterTests(TestCase):
    def test_it_reads_the_threshold_the_run_will_actually_use(self):
        from quantem.inference.segmenter import DinoMitoSegmenter

        segmenter = DinoMitoSegmenter(source_model="quantem:mito", pixel_size_nm=5.0)
        segmenter.apply_adapter(
            adapter_id="adapter-3",
            base_model="quantem:mito",
            calibrated_threshold=0.31,
        )
        payload = run_identity_from_segmenter(
            segmenter,
            run_id="job-1",
            pack_id_fallback="quantem:mito",
            native_pixel_size_nm=5.0,
            min_area=60,
        )
        # 0.31, not the pack's published 0.5: recording the default would
        # describe a run that never happened.
        self.assertEqual(payload["threshold"], 0.31)
        self.assertEqual(payload["adapter_id"], "adapter-3")
        self.assertEqual(payload["pack_id"], "quantem:mito")
        self.assertEqual(payload["ran_at_nm"], 8.0)
        self.assertEqual(payload["native_pixel_size_nm"], 5.0)
        self.assertEqual(payload["min_area"], 60)

    def test_a_segmenter_with_no_spec_falls_back_to_its_source_model(self):
        payload = run_identity_from_segmenter(
            SimpleNamespace(),
            run_id="job-1",
            pack_id_fallback="quantem:mito",
            native_pixel_size_nm=None,
            min_area=None,
        )
        self.assertEqual(payload["pack_id"], "quantem:mito")
        self.assertIsNone(payload["threshold"])
        self.assertIsNone(payload["min_area"])


class MinAreaResolutionTests(TestCase):
    """The area floor a run applies is the segmenter's, not a generic 100."""

    def test_each_organelle_keeps_its_own_floor(self):
        from quantem.inference.segmenter import (
            DinoMitoSegmenter,
            DinoNucleusSegmenter,
        )
        from quantem.seg_core.db.extraction import resolve_min_area

        self.assertEqual(resolve_min_area(DinoMitoSegmenter(), None), 60)
        self.assertEqual(resolve_min_area(DinoNucleusSegmenter(), None), 8000)

    def test_an_explicit_caller_value_still_wins(self):
        from quantem.inference.segmenter import DinoMitoSegmenter
        from quantem.seg_core.db.extraction import resolve_min_area

        self.assertEqual(resolve_min_area(DinoMitoSegmenter(), 500), 500)

    def test_a_segmenter_with_no_opinion_gets_the_fallback(self):
        from quantem.seg_core.db.extraction import FALLBACK_MIN_AREA, resolve_min_area

        self.assertEqual(
            resolve_min_area(SimpleNamespace(), None), FALLBACK_MIN_AREA
        )


class _StubReporter:
    """Just enough JobReporter for the driver: a job id and the two callbacks."""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def update(self, progress=None, message=None) -> None:
        return None

    def log(self, level: str, message: str) -> None:
        return None


class _StubCancel:
    def check_cancelled(self) -> None:
        return None


class RunIdentityOnRealObjectsTests(TestCase):
    """The whole path: a run, then the objects it wrote.

    No weights -- the model's forward is the stand-in seam
    :func:`quantem.inference.engine.predict_region` documents -- but everything
    from the segmenter down through
    :func:`quantem.seg_core.db.extraction.extract_and_save_segments` is real.
    """

    OBJECT_PROB = 0.4
    CALIBRATED = 0.31

    def setUp(self):
        import numpy as np

        from quantem.inference import engine
        from quantem.inference.specs import MODEL_SPECS

        image = create_small_test_image("run identity", width=256, height=256)
        self.segmentation = create_mitochondria_segmentation(image)

        def forward(tile: np.ndarray) -> np.ndarray:
            prob = np.full(tile.shape[:2], 0.05, dtype=np.float32)
            prob[50:150, 50:150] = self.OBJECT_PROB
            return prob

        engine.clear_model_cache()
        engine.cache_model(
            engine.LoadedModel(
                spec=MODEL_SPECS["quantem:mito"],
                # Match the auto-selected device so a CUDA test environment
                # does not miss this stand-in cache and reach for real weights.
                device=engine.select_device(None),
                module=None,
                forward=forward,
                encoder_tier="stand-in",
            )
        )
        self.addCleanup(engine.clear_model_cache)

    def _run(self, reporter=None) -> int:
        from quantem.jobs.handlers.rethreshold import (
            handle_reextract_at_include_level,
        )
        from quantem.segmentation.organelle_tasks import run_segmentation_full_task

        stored_count = run_segmentation_full_task(
            segmentation_id=str(self.segmentation.id),
            segmentation_type=self.segmentation.segmentation_type.internal_name,
            source_model="quantem:mito",
            reporter=reporter,
        )
        self.assertEqual(stored_count, 1)
        apply_reporter = reporter or _StubReporter("include-level")
        outcome = handle_reextract_at_include_level(
            {
                "segmentation_id": str(self.segmentation.id),
                "source_model": "quantem:mito",
                "include_level": self.CALIBRATED,
            },
            apply_reporter,
            _StubCancel(),
        )
        return int(outcome["segment_count"])

    def _adapter(self, **overrides):
        from django.utils import timezone

        from quantem.finetune.models import STATUS_SUCCESS, Adapter

        fields = {
            "segmentation": self.segmentation,
            "base_model": "quantem:mito",
            "name": "mito @ liver",
            "status": STATUS_SUCCESS,
            "mode": "threshold_only",
            "calibrated_threshold": self.CALIBRATED,
            "split_mode": "image-disjoint",
            "applied_at": timezone.now(),
        }
        fields.update(overrides)
        return Adapter.objects.create(**fields)

    def _runs(self) -> list[dict]:
        return [
            segment.features.get(RUN_FEATURE_KEY)
            for segment in SegmentObject.objects.filter(segmentation=self.segmentation)
        ]

    def test_a_released_run_stamps_its_settings_on_every_object(self):
        self._adapter()  # applied, so the object clears the 0.4 probability
        self.assertEqual(self._run(reporter=_StubReporter("job-77")), 1)

        runs = self._runs()
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run["id"], "job-77")
        self.assertEqual(run["pack_id"], "quantem:mito")
        self.assertEqual(run["threshold"], self.CALIBRATED)
        self.assertEqual(run["ran_at_nm"], 8.0)
        self.assertEqual(run["native_pixel_size_nm"], 5.0)
        self.assertEqual(run["min_area"], 60)
        self.assertTrue(run["finished_at"].endswith("Z"))

    def test_the_applied_adapter_is_named_on_the_objects_it_produced(self):
        adapter = self._adapter()
        self.assertEqual(self._run(), 1)
        self.assertEqual(self._runs()[0]["adapter_id"], str(adapter.id))

    def test_two_runs_of_the_same_pack_are_distinguishable(self):
        # The point of the record: same pack, same image, different threshold.
        # Without a run id and a threshold these are one indistinguishable
        # population in the export manifest.
        adapter = self._adapter()
        self._run(reporter=_StubReporter("job-A"))
        first = self._runs()[0]

        adapter.applied_at = None
        adapter.save(update_fields=["applied_at", "updated_at"])
        self._run(reporter=_StubReporter("job-B"))

        self.assertNotEqual(first["id"], "job-B")
        self.assertEqual(first["threshold"], self.CALIBRATED)
        self.assertIsNotNone(first["adapter_id"])

    def test_every_object_of_one_run_shares_one_id(self):
        self._adapter()
        self._run(reporter=_StubReporter("job-77"))
        ids = {run["id"] for run in self._runs()}
        self.assertEqual(ids, {"job-77"})

    def test_a_run_outside_the_queue_still_gets_an_id(self):
        self._adapter()
        self.assertEqual(self._run(reporter=None), 1)
        self.assertTrue(self._runs()[0]["id"])

    def test_a_hand_drawn_object_carries_no_run_key_at_all(self):
        # Absence is the statement "no model produced this". A run dict with
        # null fields would say "a model produced it, settings unknown".
        polygon = Polygon(((10, 10), (30, 10), (30, 30), (10, 30), (10, 10)))
        segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            label_state="CONFIRMED",
            source_model="manual",
            confidence_score=None,
            features={},
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )
        self.assertNotIn(RUN_FEATURE_KEY, segment.features)
        self.assertIsNone(read_run_identity(segment.features))

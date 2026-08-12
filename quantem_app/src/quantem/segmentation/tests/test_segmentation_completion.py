"""``POST /api/segmentations/<id>/complete`` must not be able to destroy work quietly.

The report behind these: a user clicked "Mark Image Done" on a segmentation
holding 32 model-detected mitochondria and zero confirmations, expecting the
confirmed-area dialog. There was no dialog, the counts went to all-zero, and
"Unlock segmentation" restored nothing -- recovery was a fresh inference pass.

Their words: *"creating a segmentation, which is cheap and reversible, gets a
well-written confirmation dialog; destroying a run's output, which is neither,
is the single most prominent green button on the screen and fires on the first
click."*

So the endpoint is safe by construction rather than by asking nicely: the
destructive path is unreachable without an explicit flag **and** a count that
matches, and what it destroys is archived so unlock can put it back.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from django.test import TestCase
from shapely.geometry import Polygon

from quantem.assets.utils import create_roi_image_from_image
from quantem.seg_core.db.prob_maps import get_prob_map_file_path, save_probability_map
from quantem.segmentation.completion import completion_preview
from quantem.segmentation.models import (
    ImageSegmentation,
    ProbabilityMap,
    SegmentationCompletionArchive,
    SegmentObject,
)
from quantem.segmentation.run_identity import build_run_identity
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


class _CompletionTestBase(TestCase):
    def setUp(self):
        self.image = create_small_test_image("completion")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )
        self.url = f"/api/segmentations/{self.segmentation.id}/complete"

    def _segment(self, index: int, *, label_state: str, **fields) -> SegmentObject:
        x = 10 + (index * 12)
        polygon = Polygon(((x, 10), (x + 8, 10), (x + 8, 18), (x, 18), (x, 10)))
        defaults = {
            "segmentation": self.segmentation,
            "geometry": polygon,
            "centroid": polygon.centroid,
            "bbox": polygon.envelope,
            "label_state": label_state,
            "source_model": "quantem:mito",
            "confidence_score": 0.8,
            "features": {"mito_generated": True},
        }
        defaults.update(fields)
        return SegmentObject.objects.create(**defaults)

    def _candidates(self, count: int) -> list[SegmentObject]:
        return [self._segment(index, label_state="CANDIDATE") for index in range(count)]

    def _post(self, **body):
        return self.client.post(self.url, data=body, content_type="application/json")

    def _live_ids(self) -> set:
        return set(
            SegmentObject.objects.filter(segmentation=self.segmentation).values_list(
                "id", flat=True
            )
        )


class CompletionPreviewTests(_CompletionTestBase):
    """A dialog cannot name a number it has no way to ask for."""

    def test_get_reports_what_a_discard_would_destroy(self):
        self._candidates(32)
        self._segment(90, label_state="CONFIRMED")

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["discard_count"], 32)
        self.assertEqual(response.data["confirmed_count"], 1)
        self.assertEqual(response.data["discard_by_label_state"]["CANDIDATE"], 32)
        self.assertEqual(response.data["discard_by_source_model"]["quantem:mito"], 32)
        self.assertTrue(response.data["restorable"])

    def test_get_changes_nothing(self):
        self._candidates(3)

        self.client.get(self.url)

        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "CANDIDATES_READY")
        self.assertEqual(len(self._live_ids()), 3)

    def test_preview_counts_inferred_and_excluded_as_discardable(self):
        self._segment(0, label_state="CANDIDATE")
        self._segment(1, label_state="INFERRED")
        self._segment(2, label_state="EXCLUDED")
        self._segment(3, label_state="CONFIRMED")

        preview = completion_preview(self.segmentation)

        self.assertEqual(preview["discard_count"], 3)
        self.assertEqual(preview["discard_by_label_state"]["INFERRED"], 1)
        self.assertEqual(preview["discard_by_label_state"]["EXCLUDED"], 1)
        self.assertNotIn("CONFIRMED", preview["discard_by_label_state"])


class CompletionRequiresAcknowledgementTests(_CompletionTestBase):
    def test_a_bare_post_locks_without_deleting_anything(self):
        candidates = self._candidates(32)

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 200)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "COMPLETED")
        self.assertEqual(self._live_ids(), {candidate.id for candidate in candidates})

    def test_discard_without_a_count_is_refused_and_deletes_nothing(self):
        self._candidates(32)

        response = self._post(discard_unconfirmed=True)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self._live_ids()), 32)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "CANDIDATES_READY")
        # The refusal carries the number the client should have sent, so it
        # does not need a second round trip to recover.
        self.assertEqual(response.data["preview"]["discard_count"], 32)

    def test_a_stale_count_is_refused_with_the_fresh_one(self):
        # The dialog said 32; a run finished while it was open and there are now
        # 40. Deleting eight objects the user was never shown is not an option.
        self._candidates(40)

        response = self._post(discard_unconfirmed=True, acknowledged_discard_count=32)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(self._live_ids()), 40)
        self.assertEqual(response.data["preview"]["discard_count"], 40)
        self.assertIn("40", response.data["detail"])

    def test_a_non_boolean_discard_flag_is_refused(self):
        self._candidates(3)

        response = self._post(discard_unconfirmed="maybe")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self._live_ids()), 3)

    def test_an_explicit_false_keeps_everything(self):
        self._candidates(3)

        response = self._post(discard_unconfirmed=False)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self._live_ids()), 3)

    def test_a_matching_count_discards(self):
        confirmed = self._segment(90, label_state="CONFIRMED")
        self._candidates(32)

        response = self._post(discard_unconfirmed=True, acknowledged_discard_count=32)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._live_ids(), {confirmed.id})
        self.assertEqual(response.data["completion"]["discarded_count"], 32)

    def test_discarding_nothing_is_allowed_and_reported_as_nothing(self):
        self._segment(90, label_state="CONFIRMED")

        response = self._post(discard_unconfirmed=True, acknowledged_discard_count=0)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["completion"]["discarded_count"], 0)
        self.assertFalse(response.data["completion"]["restorable"])

    def test_marking_the_segmentation_done_reclaims_all_probability_maps(self):
        full_map = save_probability_map(
            segmentation=self.segmentation,
            model_name="DINO",
            prob_data=np.zeros((64, 64), dtype=np.uint8),
            prefix="mito",
            generated_flag="mito_generated",
        )
        roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=64,
            height=64,
            source="MANUAL",
        )
        roi_map = save_probability_map(
            segmentation=self.segmentation,
            model_name="OmniDINO",
            prob_data=np.zeros((64, 64), dtype=np.uint8),
            prefix="mito",
            generated_flag="mito_generated",
            roi_id=str(roi.id),
        )
        full_path = get_prob_map_file_path(self.segmentation, "DINO", "mito")
        roi_path = get_prob_map_file_path(
            self.segmentation,
            "OmniDINO",
            "mito",
            roi_id=str(roi.id),
        )
        self.assertTrue(full_path.exists())
        self.assertTrue(roi_path.exists())
        self.assertEqual(
            ProbabilityMap.objects.filter(segmentation=self.segmentation).count(),
            3,
        )

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(ProbabilityMap.objects.filter(id=full_map.id).exists())
        self.assertFalse(ProbabilityMap.objects.filter(id=roi_map.id).exists())
        self.assertFalse(ProbabilityMap.objects.filter(segmentation=self.segmentation).exists())
        self.assertFalse(full_path.exists())
        self.assertFalse(roi_path.exists())


class CompletionIsUndoableTests(_CompletionTestBase):
    def test_unlock_restores_the_objects_the_discard_destroyed(self):
        candidates = self._candidates(32)
        original_ids = {candidate.id for candidate in candidates}

        self._post(discard_unconfirmed=True, acknowledged_discard_count=32)
        self.assertEqual(self._live_ids(), set())

        response = self.client.delete(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["restored"]["restored_count"], 32)
        # Restored under their original ids, so anything still holding one --
        # a base_segment link, an open client -- keeps working.
        self.assertEqual(self._live_ids(), original_ids)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "CANDIDATES_READY")

    def test_a_restored_object_keeps_its_geometry_labels_and_provenance(self):
        run = build_run_identity(
            run_id="job-77",
            pack_id="quantem:mito",
            threshold=0.31,
            adapter_id="adapter-3",
            ran_at_nm=8.0,
            native_pixel_size_nm=5.0,
            min_area=60,
        )
        original = self._segment(
            0,
            label_state="CANDIDATE",
            confidence_score=0.42,
            features={"mito_generated": True, "run": run},
        )

        self._post(discard_unconfirmed=True, acknowledged_discard_count=1)
        self.client.delete(self.url)

        restored = SegmentObject.objects.get(id=original.id)
        self.assertEqual(restored.label_state, "CANDIDATE")
        self.assertEqual(restored.source_model, "quantem:mito")
        self.assertAlmostEqual(restored.confidence_score, 0.42)
        self.assertEqual(restored.features["run"], run)
        self.assertTrue(restored.geometry.equals(original.geometry))
        self.assertAlmostEqual(restored.bbox_minx, original.bbox_minx)
        self.assertAlmostEqual(restored.centroid_y, original.centroid_y)

    def test_a_restored_family_keeps_its_base_segment_link(self):
        base = self._segment(0, label_state="CANDIDATE")
        child = self._segment(1, label_state="CANDIDATE", base_segment=base)

        self._post(discard_unconfirmed=True, acknowledged_discard_count=2)
        self.client.delete(self.url)

        self.assertEqual(SegmentObject.objects.get(id=child.id).base_segment_id, base.id)

    def test_unlocking_a_segmentation_that_kept_everything_restores_nothing(self):
        candidates = self._candidates(3)
        self.client.post(self.url)

        response = self.client.delete(self.url)

        self.assertEqual(response.data["restored"]["restored_count"], 0)
        self.assertEqual(self._live_ids(), {c.id for c in candidates})

    def test_only_one_archive_is_kept_per_segmentation(self):
        # Otherwise every completion of a big segmentation leaves a multi-MB row
        # behind forever.
        self._candidates(2)
        self._post(discard_unconfirmed=True, acknowledged_discard_count=2)
        self.client.delete(self.url)  # restores the 2
        self._candidates(3)
        self._post(discard_unconfirmed=True, acknowledged_discard_count=5)

        self.assertEqual(
            SegmentationCompletionArchive.objects.filter(segmentation=self.segmentation).count(),
            1,
        )

    def test_a_successful_restore_consumes_its_archive(self):
        self._candidates(2)
        self._post(discard_unconfirmed=True, acknowledged_discard_count=2)

        self.client.delete(self.url)

        self.assertFalse(SegmentationCompletionArchive.objects.exists())

    def test_restore_does_not_overwrite_a_live_object_that_reclaimed_an_id(self):
        candidate = self._segment(0, label_state="CANDIDATE")
        self._post(discard_unconfirmed=True, acknowledged_discard_count=1)
        # A later run happens to write an object under the same id.
        reused = self._segment(5, label_state="CONFIRMED", id=candidate.id, confidence_score=0.99)

        self.client.delete(self.url)

        survivor = SegmentObject.objects.get(id=reused.id)
        self.assertEqual(survivor.label_state, "CONFIRMED")
        self.assertAlmostEqual(survivor.confidence_score, 0.99)


class CompletionArchiveCeilingTests(_CompletionTestBase):
    """Past a point the undo is not worth the row it is written in -- say so."""

    def test_an_oversized_discard_is_reported_as_not_restorable(self):
        self._candidates(4)

        with patch("quantem.segmentation.completion.archive_max_objects", return_value=2):
            response = self._post(discard_unconfirmed=True, acknowledged_discard_count=4)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["completion"]["discarded_count"], 4)
        self.assertFalse(response.data["completion"]["restorable"])
        self.assertEqual(self._live_ids(), set())

    def test_unlock_after_an_oversized_discard_admits_it_cannot_restore(self):
        self._candidates(4)
        with patch("quantem.segmentation.completion.archive_max_objects", return_value=2):
            self._post(discard_unconfirmed=True, acknowledged_discard_count=4)

        response = self.client.delete(self.url)

        self.assertEqual(response.data["restored"]["restored_count"], 0)
        self.assertEqual(response.data["restored"]["archived_count"], 4)
        self.assertFalse(response.data["restored"]["restorable"])

    def test_a_byte_ceiling_also_refuses_to_archive(self):
        self._candidates(3)

        with patch("quantem.segmentation.completion.archive_max_bytes", return_value=10):
            response = self._post(discard_unconfirmed=True, acknowledged_discard_count=3)

        self.assertFalse(response.data["completion"]["restorable"])

    def test_the_preview_predicts_that_a_discard_is_not_restorable(self):
        self._candidates(4)

        with patch("quantem.segmentation.completion.archive_max_objects", return_value=2):
            preview = completion_preview(self.segmentation)

        self.assertFalse(preview["restorable"])


class CompletionDeletesOnlyWhatItArchivedTests(_CompletionTestBase):
    def test_an_object_written_after_the_snapshot_survives(self):
        """A worker finishing mid-completion must not lose its output silently.

        The discard deletes by id, not by predicate. Deleting by predicate would
        also take whatever landed between the snapshot and the delete -- objects
        that would then be gone with nothing in the archive to bring them back.
        """
        from quantem.segmentation import completion as completion_module

        self._candidates(2)
        real_build = completion_module.build_snapshot

        def build_then_race(segmentation):
            rows, complete = real_build(segmentation)
            self._segment(50, label_state="CANDIDATE")
            return rows, complete

        with patch.object(completion_module, "build_snapshot", side_effect=build_then_race):
            response = self._post(discard_unconfirmed=True, acknowledged_discard_count=2)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["completion"]["discarded_count"], 2)
        self.assertEqual(len(self._live_ids()), 1)


class CompletionFailureIsAtomicTests(_CompletionTestBase):
    def test_a_failed_archive_write_leaves_every_object_alone(self):
        # The one outcome that must never happen: work destroyed and no record
        # of it.
        self._candidates(5)

        with patch(
            "quantem.segmentation.completion.SegmentationCompletionArchive.objects.create",
            side_effect=RuntimeError("disk full"),
        ):
            with self.assertRaises(RuntimeError):
                self._post(discard_unconfirmed=True, acknowledged_discard_count=5)

        self.assertEqual(len(self._live_ids()), 5)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "CANDIDATES_READY")

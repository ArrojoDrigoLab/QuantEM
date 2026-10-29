"""Regression coverage for immutable pixels and independent display revisions."""

from unittest.mock import patch

import numpy as np
import zarr
from django.test import TestCase
from django.utils import timezone
from numcodecs import Blosc
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, SegmentationOverlayLabel, SegmentObject
from quantem.segmentation.overlay_ngff import (
    DirtyBBox,
    apply_partial_overlay_update,
    build_label_lut_json,
    ensure_overlay_manifest,
    get_or_create_overlay_state,
    get_overlay_active_bundle_path,
    rebuild_overlay_full,
    register_label_state_mutation,
)
from quantem.segmentation.overlay_ngff.dirty import dirty_runs_to_chunk_coords
from quantem.segmentation.overlay_ngff.mutations import (
    queue_overlay_followup,
    run_overlay_rebuild_job,
)
from quantem.segmentation.overlay_ngff.snapshots import read_snapshot
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff


class OverlaySnapshotTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        image = create_image_from_test_tiff("Snapshot regression", width=1025, height=769)
        self.seg = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def object(self, geometry=None, **kwargs):
        geometry = geometry if geometry is not None else box(10, 10, 30, 30)
        return SegmentObject.objects.create(
            segmentation=self.seg,
            geometry=geometry,
            centroid=geometry.centroid,
            bbox=geometry.envelope,
            label_state="CANDIDATE",
            source_model="manual",
            **kwargs,
        )

    def token(self, state):
        return f"{state.bundle_version}-{state.applied_revision}"

    def labels(self, state):
        return zarr.open_array(
            str(get_overlay_active_bundle_path(state) / "labels" / "0"), mode="r"
        )

    def test_partial_preserves_old_pixels_and_matching_identity_map(self):
        first = self.object()
        old = rebuild_overlay_full(self.seg, desired_revision=1)
        old_root = get_overlay_active_bundle_path(old)
        old_chunk = (old_root / "labels" / "0" / "0.0").read_bytes()
        first.delete()
        second = self.object(box(45, 45, 65, 65))
        new = apply_partial_overlay_update(
            self.seg, dirty_bbox=DirtyBBox(0, 0, 100, 100), desired_revision=2
        )
        self.assertEqual(new.bundle_version, old.bundle_version + 1)
        self.assertEqual((old_root / "labels" / "0" / "0.0").read_bytes(), old_chunk)
        self.assertEqual(int(self.labels(new)[20, 20]), 0)
        self.assertGreater(int(self.labels(new)[50, 50]), 0)
        _, old_snapshot = read_snapshot(str(self.seg.id), None, self.token(old))
        self.assertNotIn(str(second.id), [row[1] for row in old_snapshot["labels"]])
        response = self.client.get(
            f"/api/segmentations/{self.seg.id}/overlay-lut/",
            {
                "format": "json",
                "rev": self.token(old),
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["bundle_version"], old.bundle_version)
        self.assertNotIn(str(second.id), [obj["uuid"] for obj in response.data["objects"]])
        url = f"/segmentation-overlays/{self.seg.id}.zarr/labels/0/0.0?rev={self.token(old)}"
        response = self.client.get(url)
        self.assertEqual(b"".join(response.streaming_content), old_chunk)

    def test_unknown_revision_never_substitutes_current_pixels(self):
        rebuild_overlay_full(self.seg, desired_revision=1)
        response = self.client.get(
            f"/segmentation-overlays/{self.seg.id}.zarr/labels/0/0.0?rev=1-999"
        )
        self.assertEqual(response.status_code, 404)
        response = self.client.get(f"/api/segmentations/{self.seg.id}/overlay-lut/?rev=1-999")
        self.assertEqual(response.status_code, 409)

    def test_edge_sparse_chunk_uses_full_encoded_chunk_shape(self):
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        response = self.client.get(
            f"/segmentation-overlays/{self.seg.id}.zarr/labels/0/3.4?rev={self.token(state)}"
        )
        self.assertEqual(response.status_code, 200)
        decoded = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE).decode(response.content)
        self.assertEqual(len(decoded), 256 * 256 * 4)
        self.assertFalse(np.frombuffer(decoded, dtype=np.uint32).any())

    def test_superseded_object_is_hidden_immediately_and_removed_by_rebuild(self):
        obj = self.object()
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        obj.superseded_at = timezone.now()
        obj.save(update_fields=["superseded_at"])
        self.assertEqual(build_label_lut_json(state)["objects"], [])
        new = rebuild_overlay_full(self.seg, desired_revision=2)
        self.assertEqual(int(self.labels(new)[20, 20]), 0)
        self.assertFalse(SegmentationOverlayLabel.objects.filter(overlay_state=new).exists())

    def test_uncontested_answer_only_bumps_palette(self):
        obj = self.object()
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        obj.label_state = "CONFIRMED"
        obj.save(update_fields=["label_state"])
        response = register_label_state_mutation(self.seg, [obj])
        self.assertEqual(response["rebuild_mode"], "metadata_only")
        self.assertEqual(response["bundle_version"], state.bundle_version)
        self.assertEqual(response["desired_revision"], state.desired_revision)
        self.assertGreater(response["confirmed_display_lut_revision"], state.lut_revision)
        self.assertFalse(Job.objects.exists())

    def test_contested_answer_queues_repaint_without_waiting(self):
        obj = self.object()
        self.object(box(20, 20, 40, 40))
        rebuild_overlay_full(self.seg, desired_revision=1)
        obj.label_state = "CONFIRMED"
        obj.save(update_fields=["label_state"])
        response = register_label_state_mutation(self.seg, [obj])
        self.assertEqual(response["rebuild_mode"], "async_partial")
        self.assertFalse(response["sync_applied"])
        self.assertEqual(Job.objects.count(), 1)

    def test_fallback_exposes_requested_build_lifecycle(self):
        aggregate = rebuild_overlay_full(self.seg, desired_revision=1)
        manifest = ensure_overlay_manifest(self.seg, "quantem:mito")
        self.assertTrue(manifest["is_fallback"])
        self.assertEqual(manifest["bundle_version"], aggregate.bundle_version)
        self.assertEqual(manifest["status"], "BUILDING")
        self.assertEqual(manifest["update_job"]["status"], "PENDING")
        Job.objects.all().update(status="FAILED")
        requested = get_or_create_overlay_state(self.seg, "quantem:mito")
        requested.status, requested.last_error = "FAILED", "test failure"
        requested.save()
        manifest = ensure_overlay_manifest(self.seg, "quantem:mito")
        self.assertEqual(manifest["status"], "FAILED")
        self.assertEqual(manifest["last_error"], "test failure")

    def test_coalesced_runs_keep_disjoint_chunks_and_bound_raster_memory(self):
        self.object()
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        runs = [
            {"bbox": b.as_dict(), "revision": 2}
            for b in [
                DirtyBBox(10, 10, 31, 31),
                DirtyBBox(800, 600, 820, 620),
            ]
        ]
        coords = dirty_runs_to_chunk_coords(self.seg, runs)
        self.assertEqual(coords, {(0, 0), (3, 2)})
        from quantem.segmentation.overlay_ngff import render

        with patch.object(render, "rasterize_region", wraps=render.rasterize_region) as raster:
            apply_partial_overlay_update(
                self.seg, dirty_bbox=None, dirty_chunks=coords, desired_revision=2
            )
        self.assertEqual(raster.call_count, 2)
        for call in raster.call_args_list:
            self.assertLessEqual(call.kwargs["x1"] - call.kwargs["x0"], 260)
            self.assertLessEqual(call.kwargs["y1"] - call.kwargs["y0"], 260)
        self.assertEqual(int(self.labels(state)[20, 20]), 1)

    def test_execution_rechecks_coalesced_full_build_threshold(self):
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        state.desired_revision = 2
        state.dirty_chunk_runs = [{"revision": 2, "bbox": DirtyBBox(0, 0, 1025, 769).as_dict()}]
        state.save()
        with patch(
            "quantem.segmentation.overlay_ngff.mutations.rebuild_overlay_full", return_value=state
        ) as full:
            run_overlay_rebuild_job(self.seg, mode="partial")
        full.assert_called_once()

    def test_followup_is_queued_without_a_viewer_poll(self):
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        state.desired_revision = 2
        state.pending_full_rebuild = True
        state.save()
        queue_overlay_followup({"segmentation_id": str(self.seg.id)})
        self.assertEqual(Job.objects.get().payload_json["mode"], "full")

    def test_older_queued_target_does_not_create_an_unresolvable_token(self):
        rebuild_overlay_full(self.seg, desired_revision=3)
        state = rebuild_overlay_full(self.seg, desired_revision=2)
        _, snapshot = read_snapshot(str(self.seg.id), None, self.token(state))
        self.assertEqual(snapshot["applied_revision"], 3)

    def test_revision_without_dirty_runs_still_publishes_a_matching_snapshot(self):
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        state.desired_revision = 2
        state.save()
        current = run_overlay_rebuild_job(self.seg, mode="partial")
        _, snapshot = read_snapshot(str(self.seg.id), None, self.token(current))
        self.assertEqual(snapshot["applied_revision"], 2)

    def test_partial_matches_full_labels_and_borders_across_a_chunk_seam(self):
        from quantem.segmentation.overlay_ngff.dirty import merge_dirty_bboxes

        old_geometry = box(250, 10, 260, 30)
        obj = self.object(old_geometry)
        rebuild_overlay_full(self.seg, desired_revision=1)
        obj.geometry = box(253, 10, 270, 30)
        obj.centroid, obj.bbox = obj.geometry.centroid, obj.geometry.envelope
        obj.save()
        partial = apply_partial_overlay_update(
            self.seg,
            dirty_bbox=merge_dirty_bboxes(self.seg, [old_geometry, obj.geometry]),
            desired_revision=2,
        )
        full = rebuild_overlay_full(self.seg, desired_revision=2)
        for kind in ("labels", "border"):
            first = zarr.open_group(str(get_overlay_active_bundle_path(partial) / kind), mode="r")
            second = zarr.open_group(str(get_overlay_active_bundle_path(full) / kind), mode="r")
            for level in first.array_keys():
                np.testing.assert_array_equal(first[level][:], second[level][:])

    def test_polygon_hole_preserves_underlying_object(self):
        inner = self.object(box(15, 15, 25, 25))
        ring = box(0, 0, 40, 40).difference(box(10, 10, 30, 30))
        self.object(ring, refined="MANUAL")
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        label = SegmentationOverlayLabel.objects.get(
            overlay_state=state, object_uuid=inner.id
        ).label
        self.assertEqual(int(self.labels(state)[20, 20]), label)

    def test_completed_roi_changes_invalidate_palette_without_pixels(self):
        self.object()
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        url = f"/api/segmentations/{self.seg.id}/completed-rois/"
        response = self.client.post(
            url, {"polygon_coords": list(box(0, 0, 50, 50).exterior.coords)}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        current = get_or_create_overlay_state(self.seg)
        self.assertEqual(current.bundle_version, state.bundle_version)
        self.assertGreater(current.lut_revision, state.lut_revision)
        self.assertTrue(build_label_lut_json(current)["objects"][0]["hidden_by_completed_roi"])

    def test_roi_initialization_registers_a_real_dirty_box(self):
        from types import SimpleNamespace

        from quantem.segmentation.roi_initialization import initialize_roi

        obj = self.object()
        obj.source_model = "quantem:mito"
        obj.save(update_fields=["source_model"])
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        initialize_roi(self.seg, SimpleNamespace(x=0, y=0, width=50, height=50), prepopulate=True)
        current = get_or_create_overlay_state(self.seg)
        self.assertGreater(current.desired_revision, state.desired_revision)
        self.assertTrue(Job.objects.exists())

    def test_analysis_uses_confirmed_union_when_other_objects_occlude_it(self):
        from quantem.analysis.loaders import _settled_overlay_mask, segmentation_mask

        confirmed = self.object(box(10, 10, 40, 40))
        confirmed.label_state = "CONFIRMED"
        confirmed.save(update_fields=["label_state"])
        self.object(box(20, 20, 30, 30), refined="MANUAL")
        rebuild_overlay_full(self.seg, desired_revision=1)
        shape = (769, 1025)
        self.assertIsNone(_settled_overlay_mask(self.seg, shape))
        mask = segmentation_mask(self.seg, shape)
        self.assertTrue(mask[25, 25])
        self.assertEqual(int(mask.sum()), 900)

    def test_stale_manifest_status_update_does_not_overwrite_new_work(self):
        from quantem.segmentation.overlay_ngff.manifest import _save_manifest_state

        state = rebuild_overlay_full(self.seg, desired_revision=1)
        current = get_or_create_overlay_state(self.seg)
        current.status, current.desired_revision = "DIRTY", 2
        current.save()
        state.status = "READY"
        _save_manifest_state(state, ["status", "updated_at"])
        self.assertEqual(state.status, "DIRTY")
        self.assertEqual(state.desired_revision, 2)

    def test_changed_chunks_are_private_even_for_an_in_place_storage_writer(self):
        from quantem.segmentation.overlay_ngff.paths import get_overlay_stage_bundle_path
        from quantem.segmentation.overlay_ngff.snapshots import clone_store, detach_chunk

        self.object()
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        source = get_overlay_active_bundle_path(state)
        stage = get_overlay_stage_bundle_path(str(self.seg.id), 99)
        original = (source / "labels" / "0" / "0.0").read_bytes()
        clone_store(source, stage)
        detach_chunk(stage, "labels", 0, 0, 0)
        (stage / "labels" / "0" / "0.0").write_bytes(b"replacement")
        self.assertEqual((source / "labels" / "0" / "0.0").read_bytes(), original)

    def test_retention_keeps_recent_readers_and_the_newest_three_versions(self):
        import os
        import time

        from quantem.segmentation.overlay_ngff.snapshots import prune_snapshots

        states = [rebuild_overlay_full(self.seg, desired_revision=i) for i in range(1, 5)]
        first = get_overlay_active_bundle_path(states[0]).parent
        self.assertTrue(first.exists())
        old = time.time() - 3700
        os.utime(first, (old, old))
        prune_snapshots(str(self.seg.id), None, states[-1].bundle_version)
        self.assertFalse(first.exists())
        self.assertTrue(all(get_overlay_active_bundle_path(state).exists() for state in states[1:]))

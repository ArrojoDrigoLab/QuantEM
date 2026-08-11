from __future__ import annotations

from unittest.mock import patch

import numpy as np
import zarr
from django.test import TestCase
from numcodecs import Blosc
from rest_framework.test import APIClient
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon

from quantem.jobs.constants import JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY
from quantem.jobs.models import Job
from quantem.segmentation import overlay_ngff
from quantem.segmentation.geometry import extract_polygons
from quantem.segmentation.models import (
    ImageSegmentation,
    SegmentationOverlayLabel,
    SegmentationOverlayState,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff import (
    DirtyBBox,
    apply_partial_overlay_update,
    build_label_lut_binary,
    get_overlay_active_bundle_path,
    rebuild_overlay_full,
)
from quantem.segmentation.overlay_ngff.constants import (
    COLOR_CONFIRMED,
    COLOR_EXCLUDED,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _open_labels_level0(state: SegmentationOverlayState):
    """Open the level-0 ``labels`` array (uint32, 2D) of a built bundle."""
    root = get_overlay_active_bundle_path(state)
    return zarr.open_array(str(root / "labels" / "0"), mode="r")


def _open_border_level0(state: SegmentationOverlayState):
    """Open the level-0 ``border`` array (uint8, 2D) of a built bundle."""
    root = get_overlay_active_bundle_path(state)
    return zarr.open_array(str(root / "border" / "0"), mode="r")


def _label_value_at(arr, y: int, x: int) -> int:
    return int(np.asarray(arr[y, x]))


def _border_max(arr, y0: int, y1: int, x0: int, x1: int) -> int:
    return int(np.asarray(arr[y0:y1, x0:x1]).max())


def _label_for_object(state: SegmentationOverlayState, object_uuid) -> int:
    """Look up the dense label assigned to an object's uuid for this bundle."""
    row = SegmentationOverlayLabel.objects.get(overlay_state=state, object_uuid=object_uuid)
    return int(row.label)


def _lut_rgba_for_label(state: SegmentationOverlayState, label: int) -> tuple[int, int, int, int]:
    rgba_bytes, max_label = build_label_lut_binary(state)
    assert label <= max_label, f"label {label} exceeds max_label {max_label}"
    buffer = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape((max_label + 1, 4))
    return tuple(int(channel) for channel in buffer[label])


class SegmentationOverlayManifestTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Manifest Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def test_manifest_endpoint_queues_initial_build(self):
        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/overlay-manifest/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "BUILDING")
        self.assertIsNone(response.data["ngff_url"])
        # The ID-map overlay exposes two integer arrays + a render-time LUT,
        # replacing the legacy pre-colored channel-index map.
        self.assertEqual(response.data["arrays"], ["labels", "border"])
        self.assertEqual(response.data["label_dtype"], "uint32")
        self.assertIsNotNone(response.data["lut_url"])
        self.assertNotIn("channel_indices", response.data)
        self.assertTrue(Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY).exists())

    def test_manifest_endpoint_requeues_dirty_valid_overlay(self):
        state = rebuild_overlay_full(self.segmentation, desired_revision=1)
        state.status = SegmentationOverlayState.STATUS_DIRTY
        state.applied_revision = 1
        state.desired_revision = 2
        state.pending_full_rebuild = True
        state.save(
            update_fields=[
                "status",
                "applied_revision",
                "desired_revision",
                "pending_full_rebuild",
                "updated_at",
            ]
        )

        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/overlay-manifest/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "BUILDING")
        queued_job = Job.objects.get(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY)
        self.assertEqual(
            queued_job.payload_json,
            {"segmentation_id": str(self.segmentation.id), "mode": "full"},
        )

    def test_one_bundle_is_queued_once_however_often_it_is_asked_for(self):
        url = f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
        for _ in range(3):
            self.assertEqual(self.client.get(url).status_code, 200)
        for _ in range(3):
            self.assertEqual(
                self.client.get(url, {"source_model": "quantem:nucleus"}).status_code,
                200,
            )
        # Differently cased: normalised, so it must not start a third build.
        self.assertEqual(self.client.get(url, {"source_model": "QuantEM:Nucleus"}).status_code, 200)

        jobs = list(Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY))
        self.assertEqual(
            len(jobs),
            2,
            "expected exactly one rebuild per bundle: the aggregate and quantem:nucleus",
        )
        self.assertEqual(
            sorted(job.payload_json.get("source_model", "") for job in jobs),
            ["", "quantem:nucleus"],
        )

    def test_the_aggregate_and_a_per_source_bundle_are_separate_builds(self):
        """Reported as duplicate work; it is not. Do not collapse these.

        A segmentation carries an aggregate overlay and one overlay per model
        that produced objects in it, each its own zarr store with its own
        revisions. Opening a nucleus segmentation asks for both, so two jobs is
        the right number -- merging them would leave one bundle stale for good.

        What made the pair look like duplicate work is that both queue rows
        render as ``"Rebuild segmentation overlay"``
        (``jobs.constants.JOB_TYPE_LABELS``) with nothing to tell them apart.
        The distinguishing value is on the job, in the payload and in the tag,
        and this pins it there so the queue view has something to show.
        """
        url = f"/api/segmentations/{self.segmentation.id}/overlay-manifest/"
        self.client.get(url)
        self.client.get(url, {"source_model": "quantem:nucleus"})

        jobs = {
            job.payload_json.get("source_model", ""): job
            for job in Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY)
        }
        self.assertEqual(set(jobs), {"", "quantem:nucleus"})
        self.assertIn("source_model:quantem:nucleus", jobs["quantem:nucleus"].tags)
        self.assertNotIn(
            "source_model:quantem:nucleus",
            jobs[""].tags,
            "the aggregate build must not be tagged with one model's name",
        )


class SegmentationOverlayRebuildStateTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Overlay Rebuild State Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10)))
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=None,
            features={"sam_score": 0.9},
        )

    def test_full_rebuild_preserves_newer_revision_bumped_mid_build(self):
        initial_state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        self.assertEqual(initial_state.applied_revision, 0)

        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        state.status = SegmentationOverlayState.STATUS_BUILDING
        state.applied_revision = 0
        state.desired_revision = 1
        state.pending_full_rebuild = False
        state.dirty_chunk_runs = []
        state.save(
            update_fields=[
                "status",
                "applied_revision",
                "desired_revision",
                "pending_full_rebuild",
                "dirty_chunk_runs",
                "updated_at",
            ]
        )

        original_rasterize_tile_worker = overlay_ngff.render.rasterize_tile_worker
        mutated = False

        def rasterize_with_revision_bump(*args, **kwargs):
            nonlocal mutated
            if not mutated:
                mutated = True
                current_state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
                current_state.status = SegmentationOverlayState.STATUS_DIRTY
                current_state.desired_revision = 2
                current_state.pending_full_rebuild = True
                current_state.save(
                    update_fields=[
                        "status",
                        "desired_revision",
                        "pending_full_rebuild",
                        "updated_at",
                    ]
                )
            return original_rasterize_tile_worker(*args, **kwargs)

        with patch(
            "quantem.segmentation.overlay_ngff.render.rasterize_tile_worker",
            side_effect=rasterize_with_revision_bump,
        ):
            rebuilt_state = rebuild_overlay_full(self.segmentation, desired_revision=1)

        rebuilt_state.refresh_from_db()
        self.assertEqual(rebuilt_state.applied_revision, 1)
        self.assertEqual(rebuilt_state.desired_revision, 2)
        self.assertTrue(rebuilt_state.pending_full_rebuild)
        self.assertEqual(rebuilt_state.status, SegmentationOverlayState.STATUS_DIRTY)

    def test_partial_rebuild_preserves_later_dirty_runs_and_applied_revision(self):
        rebuild_overlay_full(self.segmentation, desired_revision=0)
        state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
        state.status = SegmentationOverlayState.STATUS_BUILDING
        state.applied_revision = 0
        state.desired_revision = 1
        state.pending_full_rebuild = False
        state.dirty_chunk_runs = []
        state.save(
            update_fields=[
                "status",
                "applied_revision",
                "desired_revision",
                "pending_full_rebuild",
                "dirty_chunk_runs",
                "updated_at",
            ]
        )

        original_rasterize_tile_worker = overlay_ngff.render.rasterize_tile_worker
        mutated = False
        later_dirty_run = {
            "revision": 2,
            "bbox": {
                "x_min": 0,
                "y_min": 0,
                "x_max": 64,
                "y_max": 64,
            },
            "chunk_x_min": 0,
            "chunk_x_max": 0,
            "chunk_y_min": 0,
            "chunk_y_max": 0,
        }

        def rasterize_with_dirty_run_bump(*args, **kwargs):
            nonlocal mutated
            if not mutated:
                mutated = True
                current_state = SegmentationOverlayState.objects.get(segmentation=self.segmentation)
                current_state.status = SegmentationOverlayState.STATUS_DIRTY
                current_state.desired_revision = 2
                current_state.dirty_chunk_runs = [later_dirty_run]
                current_state.save(
                    update_fields=[
                        "status",
                        "desired_revision",
                        "dirty_chunk_runs",
                        "updated_at",
                    ]
                )
            return original_rasterize_tile_worker(*args, **kwargs)

        with patch(
            "quantem.segmentation.overlay_ngff.render.rasterize_tile_worker",
            side_effect=rasterize_with_dirty_run_bump,
        ):
            updated_state = apply_partial_overlay_update(
                self.segmentation,
                dirty_bbox=DirtyBBox(x_min=0, y_min=0, x_max=64, y_max=64),
                desired_revision=1,
            )

        updated_state.refresh_from_db()
        self.assertEqual(updated_state.applied_revision, 1)
        self.assertEqual(updated_state.desired_revision, 2)
        self.assertFalse(updated_state.pending_full_rebuild)
        self.assertEqual(updated_state.dirty_chunk_runs, [later_dirty_run])
        self.assertEqual(updated_state.status, SegmentationOverlayState.STATUS_DIRTY)


class SegmentationOverlayQueryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Query Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.confirmed = self._create_segment(10, 10, 20, 20, "CONFIRMED")
        self.candidate = self._create_segment(30, 30, 42, 42, "CANDIDATE")

    def _create_segment(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label_state: str,
    ) -> SegmentObject:
        polygon = Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            confidence_score=0.8 if label_state != "CONFIRMED" else None,
            features={"sam_score": 0.9},
        )

    def test_at_point_respects_states_filter(self):
        response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/segments/at-point",
            {"x": 15, "y": 15, "states": "CONFIRMED"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["id"], str(self.confirmed.id))

    def test_query_region_returns_exact_state_filtered_hits(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/query-region",
            {
                "bbox": {"x0": 0, "y0": 0, "x1": 50, "y1": 50},
                "states": ["CANDIDATE"],
                "include_geometry": True,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["segments"]), 1)
        self.assertEqual(response.data["segments"][0]["id"], str(self.candidate.id))
        self.assertGreaterEqual(len(response.data["segments"][0]["geometry_coords"]), 4)


class SegmentationOverlaySyncPartialTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Sync Partial Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.segment = self._create_segment("CANDIDATE")

    def _create_segment(self, label_state: str) -> SegmentObject:
        polygon = Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10)))
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            confidence_score=0.8,
            features={"sam_score": 0.9},
        )

    def test_label_update_defers_the_raster_and_recolours_immediately(self):
        rebuild_overlay_full(self.segmentation, desired_revision=0)

        response = self.client.post(
            f"/api/segments/{self.segment.id}/label/",
            {"label_state": "CONFIRMED"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        # An answer registers the edit and leaves the raster to the queue (see
        # api_views.segments.labels): the response is not waiting on a disk
        # write. What the reviewer sees is still correct at once, because the
        # colour comes from the LUT rather than the raster -- so this asserts
        # both that the dense label survived and that the LUT already resolves
        # the confirmed colour, with `applied_revision` still behind.
        self.assertEqual(response.data["overlay"]["rebuild_mode"], "async_partial")
        self.assertFalse(response.data["overlay"]["sync_applied"])
        self.assertGreater(
            response.data["overlay"]["desired_revision"],
            response.data["overlay"]["applied_revision"],
        )

        state = SegmentationOverlayState.objects.get(
            segmentation=self.segmentation, candidate_source_model=""
        )
        labels0 = _open_labels_level0(state)
        label = _label_for_object(state, self.segment.id)
        self.assertEqual(_label_value_at(labels0, 15, 15), label)

        rgba = _lut_rgba_for_label(state, label)
        self.assertEqual(rgba[:3], _hex_to_rgb(COLOR_CONFIRMED))
        # CONFIRMED is visible by default.
        self.assertEqual(rgba[3], 255)

    def test_batch_reject_moves_candidate_out_of_candidate_channels(self):
        rebuild_overlay_full(self.segmentation, desired_revision=0)

        response = self.client.post(
            "/api/segments/labels/batch/",
            {"labels": [{"id": str(self.segment.id), "label_state": "EXCLUDED"}]},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(str(self.segmentation.id), response.data["overlays"])

        state = SegmentationOverlayState.objects.get(
            segmentation=self.segmentation, candidate_source_model=""
        )
        labels0 = _open_labels_level0(state)
        label = _label_for_object(state, self.segment.id)
        # The raster keeps the object's dense label; only its LUT colour/alpha
        # changes to the excluded state.
        self.assertEqual(_label_value_at(labels0, 15, 15), label)

        rgba = _lut_rgba_for_label(state, label)
        self.assertEqual(rgba[:3], _hex_to_rgb(COLOR_EXCLUDED))
        # EXCLUDED is hidden by default, so the LUT alpha is zeroed.
        self.assertEqual(rgba[3], 0)


class SegmentationOverlaySparseChunkTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Overlay Sparse Chunk Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10)))
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=None,
            features={"sam_score": 0.9},
        )

    def test_missing_sparse_chunk_returns_zero_filled_bytes(self):
        state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        store_root = get_overlay_active_bundle_path(state)
        # The only object is at (10..22, 10..22); chunk (cy=2, cx=0) covers
        # pixels y=512..768 and is guaranteed never written.
        missing_chunk_path = store_root / "labels" / "0" / "2.0"
        self.assertFalse(missing_chunk_path.exists())

        response = self.client.get(
            f"/segmentation-overlays/{self.segmentation.id}.zarr/labels/0/2.0?rev=0"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/octet-stream")
        self.assertEqual(
            response["Cache-Control"],
            "public, max-age=31536000, immutable",
        )

        labels0 = zarr.open_array(str(store_root / "labels" / "0"), mode="r")
        chunk_h = min(256, int(labels0.shape[0]))
        chunk_w = min(256, int(labels0.shape[1]))
        # Decode using the same codec the labels array was created with
        # (Blosc zstd, level 5, byte-shuffle) and assert all-background zeros.
        decoded = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE).decode(response.content)
        chunk = np.frombuffer(decoded, dtype=np.uint32).reshape((chunk_h, chunk_w))
        self.assertEqual(int(chunk.max()), 0)

    def test_missing_sparse_chunk_matches_encode_zero_chunk(self):
        state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        store_root = get_overlay_active_bundle_path(state)
        labels0 = zarr.open_array(str(store_root / "labels" / "0"), mode="r")
        chunk_h = min(256, int(labels0.shape[0]))
        chunk_w = min(256, int(labels0.shape[1]))

        response = self.client.get(
            f"/segmentation-overlays/{self.segmentation.id}.zarr/labels/0/2.0?rev=0"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.content,
            overlay_ngff.encode_zero_chunk("labels", (chunk_h, chunk_w)),
        )


class SegmentationOverlayRasterizationTests(TestCase):
    def setUp(self):
        self.image = create_image_from_test_tiff("Overlay Rasterization Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _create_segment(
        self,
        polygon: Polygon,
        *,
        label_state: str = "CONFIRMED",
        source_model: str = "manual",
    ) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            source_model=source_model,
            confidence_score=0.8 if label_state != "CONFIRMED" else None,
            features={"sam_score": 0.9},
        )

    def test_source_overlay_renders_active_candidates_manual_and_all_confirmed(self):
        # Membership rule for a per-source bundle: CONFIRMED (any source) OR
        # manual OR this exact source model. The cross-source candidate must be
        # excluded from the raster entirely.
        active_candidate = self._create_segment(
            Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10))),
            label_state="CANDIDATE",
            source_model="quantem:mito",
        )
        other_candidate = self._create_segment(
            Polygon(((40, 10), (52, 10), (52, 22), (40, 22), (40, 10))),
            label_state="CANDIDATE",
            source_model="omniem:mito",
        )
        manual_candidate = self._create_segment(
            Polygon(((70, 10), (82, 10), (82, 22), (70, 22), (70, 10))),
            label_state="CANDIDATE",
            source_model="manual",
        )
        confirmed_other_source = self._create_segment(
            Polygon(((100, 10), (112, 10), (112, 22), (100, 22), (100, 10))),
            label_state="CONFIRMED",
            source_model="omniem:mito",
        )

        state = rebuild_overlay_full(
            self.segmentation, source_model="quantem:mito", desired_revision=0
        )
        labels0 = _open_labels_level0(state)

        # Active-source candidate, manual candidate, and confirmed (any source)
        # all painted; the cross-source candidate is background.
        self.assertEqual(
            _label_value_at(labels0, 15, 15),
            _label_for_object(state, active_candidate.id),
        )
        self.assertEqual(_label_value_at(labels0, 15, 45), 0)
        self.assertEqual(
            _label_value_at(labels0, 15, 75),
            _label_for_object(state, manual_candidate.id),
        )
        self.assertEqual(
            _label_value_at(labels0, 15, 105),
            _label_for_object(state, confirmed_other_source.id),
        )
        # The cross-source candidate never received a label for this bundle.
        self.assertFalse(
            SegmentationOverlayLabel.objects.filter(
                overlay_state=state, object_uuid=other_candidate.id
            ).exists()
        )

    def test_touching_segments_keep_visible_border_channel(self):
        left = self._create_segment(Polygon(((10, 10), (22, 10), (22, 22), (10, 22), (10, 10))))
        right = self._create_segment(Polygon(((22, 10), (34, 10), (34, 22), (22, 22), (22, 10))))

        state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        labels0 = _open_labels_level0(state)
        border0 = _open_border_level0(state)

        # Each touching object keeps its own distinct dense label...
        left_label = _label_for_object(state, left.id)
        right_label = _label_for_object(state, right.id)
        self.assertNotEqual(left_label, right_label)
        self.assertEqual(_label_value_at(labels0, 15, 15), left_label)
        self.assertEqual(_label_value_at(labels0, 15, 28), right_label)
        # ...and the shared seam is baked into the border mask.
        self.assertGreater(_border_max(border0, 15, 16, 21, 24), 0)

        bundle_root = get_overlay_active_bundle_path(state)
        self.assertTrue((bundle_root / ".zattrs").exists())
        self.assertTrue((bundle_root / ".zgroup").exists())

    def test_extract_polygons_ignores_non_polygon_iterables(self):
        geometry = shapely_wkt.loads(
            "GEOMETRYCOLLECTION(POLYGON((0 0, 4 0, 4 4, 0 4, 0 0)),LINESTRING(4 0, 8 0))"
        )

        polygons = extract_polygons(geometry)

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].geom_type, "Polygon")

    def test_polygon_holes_render_empty_fill_and_interior_border(self):
        outer = Polygon(((10, 10), (40, 10), (40, 40), (10, 40), (10, 10)))
        inner = Polygon(((20, 20), (30, 20), (30, 30), (20, 30), (20, 20)))
        polygon = outer.difference(inner)
        obj = self._create_segment(polygon)

        state = rebuild_overlay_full(self.segmentation, desired_revision=0)
        labels0 = _open_labels_level0(state)
        border0 = _open_border_level0(state)

        label = _label_for_object(state, obj.id)
        # Interior of the hole is background (no label painted there)...
        self.assertEqual(_label_value_at(labels0, 25, 25), 0)
        # ...the ring wall carries the object's label...
        self.assertEqual(_label_value_at(labels0, 15, 25), label)
        # ...and the hole boundary is baked into the border mask.
        self.assertGreater(_border_max(border0, 19, 22, 24, 27), 0)


class SegmentationOverlayDownsampleTests(TestCase):
    def test_labels_use_mode_pooling_while_border_uses_max_pooling(self):
        # Labels never average: a 2x2 block mode-pools to the most frequent
        # non-zero id (here, three 5s beat one background).
        label_block = np.array([[5, 5], [5, 0]], dtype=np.uint32)
        pooled_labels = overlay_ngff.render.mode_downsample_2x2(label_block)
        self.assertEqual(pooled_labels.shape, (1, 1))
        self.assertEqual(int(pooled_labels[0, 0]), 5)

        # The border mask max-pools: a block is border if any child is.
        border_block = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        pooled_border = overlay_ngff.render.max_downsample_2x2(border_block)
        self.assertEqual(pooled_border.shape, (1, 1))
        self.assertEqual(int(pooled_border[0, 0]), 1)

    def test_mode_pooling_ties_resolve_to_smaller_id(self):
        # Two distinct ids each appear twice; the smaller id wins the tie.
        tie_block = np.array([[7, 7], [3, 3]], dtype=np.uint32)
        pooled = overlay_ngff.render.mode_downsample_2x2(tie_block)
        self.assertEqual(int(pooled[0, 0]), 3)

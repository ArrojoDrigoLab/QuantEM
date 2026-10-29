"""Cost bounds must not change the published pixels or object identities."""

from unittest.mock import patch

import numpy as np
from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from shapely.geometry import box

from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff import DirtyBBox, rebuild_overlay_full
from quantem.segmentation.overlay_ngff.labels_lut import existing_label_map
from quantem.segmentation.overlay_ngff.mutations import (
    _bbox_intersects,
    _macro_tile_payloads,
    apply_partial_overlay_update,
    overlay_rebuild_policy,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_image_from_test_tiff


class OverlayPerformanceTests(TestCase):
    def setUp(self):
        image = create_image_from_test_tiff("Performance regression", width=513, height=513)
        self.seg = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        geometry = box(10, 10, 40, 40)
        self.obj = SegmentObject.objects.create(
            segmentation=self.seg,
            geometry=geometry,
            centroid=geometry.centroid,
            bbox=geometry.envelope,
            source_model="manual",
        )

    def test_live_membership_does_not_depend_on_join_statistics(self):
        state = rebuild_overlay_full(self.seg, desired_revision=1)
        # Warm the FK relation so only the two membership scans are counted.
        _ = state.segmentation
        with CaptureQueriesContext(connection) as queries:
            labels = existing_label_map(state, current_only=True)
        self.assertEqual(set(labels), {self.obj.pk})
        self.assertEqual(len(queries), 2)
        self.assertTrue(
            all(" JOIN " not in q["sql"] and " IN (SELECT " not in q["sql"] for q in queries)
        )

    def test_large_store_queues_a_tiny_edit(self):
        rebuild_overlay_full(self.seg, desired_revision=1)
        with (
            patch(
                "quantem.segmentation.overlay_ngff.mutations.segmentation_dimensions",
                return_value=(16384, 16384),
            ),
            patch(
                "quantem.segmentation.overlay_ngff.mutations._is_valid_label_store",
                return_value=True,
            ),
        ):
            self.assertEqual(
                overlay_rebuild_policy(self.seg, dirty_bbox=DirtyBBox(10, 10, 20, 20)),
                "async_partial",
            )
        self.assertEqual(
            overlay_rebuild_policy(self.seg, dirty_bbox=DirtyBBox(10, 10, 20, 20)), "sync_partial"
        )

    def test_binary_cache_hit_does_not_rebuild_an_oversized_json_palette(self):
        from rest_framework.test import APIClient

        from quantem.segmentation.overlay_ngff.cache import palette_cache

        state = rebuild_overlay_full(self.seg, desired_revision=1)
        url = f"/api/segmentations/{self.seg.id}/overlay-lut/?rev={state.bundle_version}-{state.applied_revision}"
        client = APIClient()
        original_put = palette_cache.put

        def binary_only(key, value, size=0):
            if not isinstance(value, dict):
                original_put(key, value, size)

        with patch.object(palette_cache, "put", side_effect=binary_only):
            first = client.get(url)
        with patch(
            "quantem.segmentation.api_views.overlay.build_label_lut_json",
            side_effect=AssertionError("JSON rebuilt"),
        ):
            second = client.get(url)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.content, first.content)

    def test_ancestor_chunk_retries_preserve_old_snapshot(self):
        import zarr

        from quantem.segmentation.overlay_ngff import get_overlay_active_bundle_path, render

        old = rebuild_overlay_full(self.seg, desired_revision=1)
        path = get_overlay_active_bundle_path(old)
        before = zarr.open_array(str(path / "labels" / "1"), mode="r")[:]
        original = render.write_parent_chunk
        calls = 0

        def transient(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise PermissionError("temporary sharing lock")
            return original(*args, **kwargs)

        with patch.object(render, "write_parent_chunk", side_effect=transient):
            apply_partial_overlay_update(
                self.seg, dirty_bbox=DirtyBBox(0, 0, 60, 60), desired_revision=2
            )
        self.assertGreater(calls, 1)
        np.testing.assert_array_equal(
            before, zarr.open_array(str(path / "labels" / "1"), mode="r")[:]
        )


def test_spatial_buckets_match_scan_including_halos_and_order():
    rng = np.random.default_rng(42)
    ops = []
    for index in range(2000):
        x, y = rng.integers(-100, 18000, size=2)
        w, h = rng.integers(1, 6000, size=2)
        ops.append({"bbox": (x, y, x + w, y + h), "label": index})
    # Nonzero region origins and shapes straddling a border are intentional.
    for payload in _macro_tile_payloads(
        ops, width=16384, height=16384, region_x0=111, region_y0=333
    ):
        expected = [op for op in ops if _bbox_intersects(op["bbox"], *payload["region"])]
        assert payload["draw_ops"] == expected

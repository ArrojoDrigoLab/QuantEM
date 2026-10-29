import json
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from django.test import TestCase
from skimage.morphology import closing, disk

from quantem.core.local_storage import storage_path
from quantem.inference.resample import NativeProbabilityMap, binarize_quantized
from quantem.segmentation.global_masks import (
    load_global_mask,
    postprocess_global_prediction,
    save_global_mask,
)
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.overlay_ngff import snapshots
from quantem.segmentation.type_service import get_or_create_er_type
from quantem.testing import create_small_test_image


def test_snapshot_count_bytes_and_reader_grace(tmp_path):
    now = time.time()
    for version in range(1, 13):
        folder = tmp_path / str(version)
        folder.mkdir()
        (folder / "snapshot.json").write_text(json.dumps({"logical_bytes": 100}))
        os.utime(folder, (now, now))
    # A reader of version 1 outlives the ordinary retention limit.
    (tmp_path / "1" / ".reader").touch()
    with (
        patch.object(
            snapshots, "get_overlay_version_dir", side_effect=lambda s, v, m: tmp_path / str(v)
        ),
        patch.object(snapshots, "MAX_RETAINED_LOGICAL_BYTES", 500),
    ):
        snapshots.prune_snapshots("seg", None, 12)
        assert {p.name for p in tmp_path.iterdir()} == {"1", "8", "9", "10", "11", "12"}
        os.utime(tmp_path / "1" / ".reader", (now - 120, now - 120))
        snapshots.prune_snapshots("seg", None, 12)
        assert {p.name for p in tmp_path.iterdir()} == {"8", "9", "10", "11", "12"}


def test_clone_fallback_and_detachment_preserve_published_bytes(tmp_path):
    old = tmp_path / "old" / "store"
    chunk = old / "labels" / "0" / "0.0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"previous")
    new = tmp_path / "new" / "store"
    with patch.object(snapshots.os, "link", side_effect=OSError("unsupported")):
        snapshots.clone_store(old, new)
    snapshots.detach_chunk(new, "labels", 0, 0, 0)
    (new / "labels" / "0" / "0.0").write_bytes(b"replacement")
    assert chunk.read_bytes() == b"previous"


def test_clone_omits_chunks_that_will_be_fully_repainted(tmp_path):
    old = tmp_path / "old" / "store"
    folder = old / "labels" / "0"
    folder.mkdir(parents=True)
    for name in (".zarray", "0.0", "0.1"):
        (folder / name).write_bytes(name.encode())
    new = tmp_path / "new" / "store"
    snapshots.clone_store(old, new, replaced_chunks={"labels/0/0.0"})
    assert not (new / "labels/0/0.0").exists()
    assert (new / "labels/0/0.1").read_bytes() == b"0.1"
    assert (new / "labels/0/.zarray").read_bytes() == b".zarray"
    assert (folder / "0.0").read_bytes() == b"0.0"


def test_striped_global_closing_preserves_seams_and_holes():
    # Force several stripes without allocating a genuinely huge fixture.
    data = np.random.default_rng(5).integers(0, 256, (63, 71), dtype=np.uint8)
    segmenter = SimpleNamespace(
        native_probability_map=NativeProbabilityMap(
            data=data, interpolation="identity", back_factor=1
        ),
        _organelle=SimpleNamespace(close_radius=3),
    )
    with (
        patch("quantem.segmentation.global_masks.MAX_HEAP_ARRAY_BYTES", 1),
        patch("quantem.segmentation.global_masks.MASK_STRIPE_BYTES", 71 * 7),
    ):
        actual, radius = postprocess_global_prediction(segmenter, data, 0.62)
    np.testing.assert_array_equal(actual, closing(binarize_quantized(data, 0.62), disk(3)))
    assert radius == 3


class LargeMaskStorageTests(TestCase):
    def test_tiff_mask_and_revision_round_trip_keep_all_pixels(self):
        from quantem.segmentation.revisions import restore_revision, save_revision

        image = create_small_test_image(width=71, height=63)
        seg = ImageSegmentation.objects.create(
            asset=image.asset, segmentation_type=get_or_create_er_type()
        )
        expected = np.random.default_rng(7).random((63, 71)) > 0.7
        with patch("quantem.segmentation.global_masks.MAX_HEAP_ARRAY_BYTES", 1):
            record = save_global_mask(seg, expected, source="test")
            assert storage_path(record.file_path).suffix == ".tif"
            np.testing.assert_array_equal(load_global_mask(seg), expected)
            revision = save_revision(seg, "large mask")
            save_global_mask(seg, ~expected, source="test")
            restore_revision(revision)
            np.testing.assert_array_equal(load_global_mask(seg), expected)

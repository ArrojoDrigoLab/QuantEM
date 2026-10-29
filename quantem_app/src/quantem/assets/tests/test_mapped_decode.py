import gc

import numpy as np
import tifffile

from quantem.assets.canonical_decode import decode_canonical_plane
from quantem.core import mapped_arrays


def test_mapped_decode_preserves_uint16_conversion_and_lifetime(tmp_path, monkeypatch):
    image = np.random.default_rng(71).integers(0, 65536, (193, 257), dtype=np.uint16)
    path = tmp_path / "compressed.tif"
    tifffile.imwrite(path, image, compression="deflate")
    reference = decode_canonical_plane(path)
    monkeypatch.setattr(mapped_arrays, "MAX_HEAP_ARRAY_BYTES", 1)
    mapped = decode_canonical_plane(path)
    gc.collect()
    np.testing.assert_array_equal(mapped.array, reference.array)
    assert mapped.provenance == reference.provenance
    assert mapped.conversion == reference.conversion


def test_mapped_uint8_view_outlives_decoder_handles(tmp_path, monkeypatch):
    image = np.arange(4096, dtype=np.uint8).reshape(64, 64)
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, image)
    monkeypatch.setattr(mapped_arrays, "MAX_HEAP_ARRAY_BYTES", 1)
    array = decode_canonical_plane(path).array
    gc.collect()
    np.testing.assert_array_equal(array[9:29], image[9:29])

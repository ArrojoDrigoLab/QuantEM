from __future__ import annotations

import numpy as np
from PIL import Image

from quantem.segmentation.prob_maps.preview import (
    ensure_probability_preview,
    probability_preview_path,
    save_probability_preview,
)


def test_new_maps_get_a_browser_sized_sidecar_without_changing_the_source(tmp_path):
    source = tmp_path / "prob.png"
    probability = np.arange(48, dtype=np.uint8).reshape(6, 8)
    Image.fromarray(probability, mode="L").save(source)

    preview = save_probability_preview(source, probability, max_dimension=4)

    assert preview == probability_preview_path(source, max_dimension=4)
    assert Image.open(source).size == (8, 6)
    assert Image.open(preview).size == (4, 3)
    np.testing.assert_array_equal(np.asarray(Image.open(preview)), probability[::2, ::2])


def test_an_existing_large_map_gets_one_cached_lazy_preview(tmp_path):
    source = tmp_path / "prob.png"
    Image.fromarray(np.arange(48, dtype=np.uint8).reshape(6, 8), mode="L").save(source)

    preview = ensure_probability_preview(source, max_dimension=4)
    first_mtime = preview.stat().st_mtime_ns

    assert Image.open(preview).size == (4, 3)
    assert ensure_probability_preview(source, max_dimension=4) == preview
    assert preview.stat().st_mtime_ns == first_mtime

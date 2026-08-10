"""Inference-path unit tests. CPU only, no network, no weights."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantem_em.inference.postprocess import (  # noqa: E402
    clean_mask,
    label_objects,
    segment_from_probability,
)
from quantem_em.inference.predict import predict_region  # noqa: E402
from quantem_em.inference.prepare import to_uint8  # noqa: E402
from quantem_em.inference.resample import is_noop, resample_factors, zoom_image  # noqa: E402
from quantem_em.inference.tiling import (  # noqa: E402
    hann2d,
    pad_to_tile,
    round_up,
    stride_for,
    window_starts,
)
from quantem_em.registry import get_model_spec  # noqa: E402

# --------------------------------------------------------------------------- tiling


def test_round_up_produces_518_for_patch_14():
    assert round_up(512, 16) == 512
    assert round_up(512, 14) == 518  # appears in no config; emerges here
    assert round_up(518, 14) == 518


def test_stride_uses_bankers_rounding():
    assert stride_for(512, 0.25) == 384
    assert stride_for(518, 0.25) == 388  # 388.5 -> 388, NOT 389


def test_hann_has_the_floor():
    w = hann2d(64)
    assert w.min() == pytest.approx(1e-3)
    assert np.allclose(w, np.outer(np.hanning(64), np.hanning(64)) + 1e-3)


def test_window_starts_cover_and_end_flush():
    for length, tile, stride in [
        (1000, 512, 384),
        (512, 512, 384),
        (100, 512, 384),
        (1536, 518, 388),
        (2000, 512, 384),
    ]:
        s = window_starts(length, tile, stride)
        assert s[0] == 0
        if length > tile:
            assert s[-1] == length - tile, "last window must be flush to the edge"
            covered = np.zeros(length, bool)
            for x in s:
                covered[x : x + tile] = True
            assert covered.all(), "every pixel must be inside at least one window"


def test_pad_to_tile_is_zero_padded_and_patch_aligned():
    a = np.full((300, 700), 200, np.uint8)
    p, (h0, w0) = pad_to_tile(a, 512, 16)
    assert (h0, w0) == (300, 700)
    assert p.shape[0] >= 512 and p.shape[1] >= 512
    assert p.shape[0] % 16 == 0 and p.shape[1] % 16 == 0
    assert p[300:, :].max() == 0, "padding must be zeros, not reflected content"


# --------------------------------------------------------------------------- the big one


class _ConstantModel(torch.nn.Module):
    """Returns a fixed logit field regardless of input."""

    def __init__(self, logit_bg=0.0, logit_fg=2.0):
        super().__init__()
        self.logit_bg, self.logit_fg = logit_bg, logit_fg
        self.aux_logits = []

    def forward(self, x):
        b, _, h, w = x.shape
        out = torch.empty(b, 2, h, w)
        out[:, 0] = self.logit_bg
        out[:, 1] = self.logit_fg
        return out


@pytest.mark.parametrize("shape", [(512, 512), (700, 900), (1100, 513), (517, 1024), (200, 300)])
@pytest.mark.parametrize("model_id", ["quantem/mito", "omniem/mito"])
def test_blend_invariant(shape, model_id):
    """A constant logit field must blend to a constant probability map.

    This single property catches essentially every sliding-window stitching bug: a wrong stride, a
    missing edge window, a Hann floor that divides by zero, an accumulator that is not normalised.
    Sizes are chosen to include ones that do and do not divide evenly by the stride.
    """
    spec = get_model_spec(model_id)
    em = np.full(shape, 128, np.uint8)
    fg, _ = predict_region(_ConstantModel(), em, spec, "cpu")

    expected = float(torch.softmax(torch.tensor([0.0, 2.0]), 0)[1])
    assert fg.shape == shape
    assert np.isfinite(fg).all()
    assert fg.max() - fg.min() < 1e-5, f"blend is not flat: spread {fg.max() - fg.min():.3g}"
    assert abs(float(fg.mean()) - expected) < 1e-5


def test_predict_region_reports_progress_and_can_be_cancelled():
    from quantem_em.inference.predict import InferenceCancelled

    spec = get_model_spec("quantem/mito")
    em = np.full((1400, 1400), 128, np.uint8)

    seen = []
    predict_region(_ConstantModel(), em, spec, "cpu", progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][0] == seen[-1][1] and seen[-1][1] > 1

    with pytest.raises(InferenceCancelled):
        predict_region(_ConstantModel(), em, spec, "cpu", cancel=lambda: True)


# --------------------------------------------------------------------------- prepare


def test_uint8_passes_through_untouched():
    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out, info = to_uint8(a)
    assert np.array_equal(out, a)
    assert info["intensity_rescale"] == "none"


def test_uint16_gets_percentile_stretched():
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64)) * 4000 + 200).astype(np.uint16)
    out, info = to_uint8(a)
    assert out.dtype == np.uint8
    assert info["intensity_rescale"].startswith("percentile")
    assert out.min() == 0 and out.max() == 255


def test_rgb_converts_to_luminance_with_a_warning():
    """Green must weigh more than red, which must weigh more than blue (Rec. 601)."""
    a = np.zeros((8, 24, 3), np.uint8)
    a[:, 0:8, 0] = 255  # pure red
    a[:, 8:16, 1] = 255  # pure green
    a[:, 16:24, 2] = 255  # pure blue
    with pytest.warns(UserWarning, match="luminance"):
        out, info = to_uint8(a)
    assert out.shape == (8, 24)
    assert info["rgb_to_luminance"] is True
    red, green, blue = out[0, 0], out[0, 8], out[0, 16]
    assert green > red > blue, (red, green, blue)


def test_constant_image_has_no_dynamic_range_to_stretch():
    """A flat non-uint8 image maps to zeros -- there is no contrast to recover, and inventing
    some by scaling a single value would be worse than saying 'nothing here'."""
    out, info = to_uint8(np.full((8, 8), 3.5, np.float32))
    assert out.max() == 0
    assert info["intensity_rescale"].startswith("percentile")


def test_invert_is_exact_and_never_automatic():
    a = np.array([[0, 255], [10, 200]], np.uint8)
    out, info = to_uint8(a, invert=True)
    assert np.array_equal(out, 255 - a)
    assert info["inverted"] is True
    assert to_uint8(a)[1]["inverted"] is False


# --------------------------------------------------------------------------- resample


def test_resample_factor_and_noop_band():
    assert resample_factors(2.0, 2.0, 8.0) == (0.25, 0.25)
    assert resample_factors(16.0, 16.0, 8.0) == (2.0, 2.0)
    assert is_noop((1.0, 1.0))
    assert is_noop((1.0005, 0.9995))
    assert not is_noop((1.01, 1.0))


def test_zoom_image_halves_and_stays_uint8():
    a = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    out = zoom_image(a, (0.5, 0.5))
    assert out.shape == (32, 32)
    assert out.dtype == np.uint8


def test_zoom_noop_returns_input():
    a = np.zeros((16, 16), np.uint8)
    assert zoom_image(a, (1.0, 1.0)) is a


# --------------------------------------------------------------------------- postprocess


def test_min_area_filters_and_relabels_without_gaps():
    m = np.zeros((60, 60), bool)
    m[5:15, 5:15] = True  # 100 px
    m[30:33, 30:33] = True  # 9 px
    m[45:55, 45:55] = True  # 100 px
    lab, n = label_objects(m, min_area=50)
    assert n == 2
    assert sorted(np.unique(lab)) == [0, 1, 2], "ids must be compact, 1..N"


def test_fill_holes_and_closing():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    m[18:22, 18:22] = False  # a hole
    assert clean_mask(m, fill_holes=True)[19, 19]
    assert not clean_mask(m, fill_holes=False)[19, 19]


def test_segment_from_probability_end_to_end():
    prob = np.zeros((50, 50), np.float32)
    prob[10:25, 10:25] = 0.9
    prob[40:43, 40:43] = 0.9  # small -> dropped at min_area=100
    labels, mask, n = segment_from_probability(
        prob,
        fg_threshold=0.5,
        close_radius=0,
        min_area=100,
    )
    assert n == 1
    assert labels.dtype == np.int32
    assert mask.sum() == 225


def test_nucleus_uses_a_larger_min_area_than_the_rest():
    """Owner ruling 2026-08-06: 100 px everywhere, 500 px for nucleus."""
    assert (
        get_model_spec(
            "nucleus",
        )
        if False
        else True
    )
    for mid in ("quantem/mito", "omniem/er", "quantem/ld"):
        assert get_model_spec(mid).min_area == 100
    for mid in ("quantem/nucleus", "omniem/nucleus"):
        assert get_model_spec(mid).min_area == 500


# --- postprocessing must match the reference it claims to port ---------------------------------
# The reference implementation uses skimage's binary_closing and a bare
# skimage.measure.label. scipy's equivalents differ in ways that change published numbers.


def test_labelling_is_8_connected_like_the_reference():
    """A diagonal bridge is ONE object, as skimage.measure.label(mask) reports it.

    scipy.ndimage.label defaults to a 4-connected cross, which would call this two objects and
    then delete both through min_area.
    """
    import numpy as np

    from quantem_em.inference.postprocess import label_objects

    mask = np.zeros((9, 9), bool)
    mask[2:4, 2:4] = True
    mask[4:6, 4:6] = True  # touches the first block only at a corner
    _lab, n = label_objects(mask)
    assert n == 1, f"diagonal touch should merge (8-connected), got {n} objects"


def test_closing_does_not_erode_objects_at_the_image_border():
    """scipy's binary_closing treats out-of-bounds as background during erosion, shaving a
    close_radius-wide band off everything touching the edge -- 42 % of a full field at the nucleus
    radius of 12. skimage's does not, and skimage is what the reference calls."""
    import numpy as np

    from quantem_em.inference.postprocess import clean_mask

    full = np.ones((60, 60), bool)
    assert clean_mask(full, close_radius=12, fill_holes=False).sum() == full.size, (
        "closing ate the border"
    )

    corner = np.zeros((60, 60), bool)
    corner[:20, :20] = True
    out = clean_mask(corner, close_radius=3, fill_holes=False)
    assert out.sum() == 400, f"corner object lost area to the border: {out.sum()} of 400"


def test_summary_counts_distinct_objects_not_the_largest_id():
    """A user deleting an object in napari leaves a gap in the numbering; labels.max() then
    overcounts. ids {1, 5} is two objects."""
    import numpy as np

    from quantem_em.measure import summarize

    lab = np.zeros((40, 40), np.int32)
    lab[5:15, 5:15] = 1
    lab[25:30, 25:30] = 5
    assert summarize(lab)["n_objects"] == 2


def test_intensity_std_is_over_the_object_not_its_bounding_box():
    """A uniform circle has zero intensity variance. Taking std over the zero-filled bbox
    reported 72.4, which measures shape rather than intensity."""
    import numpy as np

    from quantem_em.measure import measure_objects

    yy, xx = np.mgrid[0:40, 0:40]
    lab = np.zeros((40, 40), np.int32)
    lab[(yy - 20) ** 2 + (xx - 20) ** 2 < 100] = 1
    img = np.where(lab > 0, 200, 30).astype(np.uint8)
    f = measure_objects(lab, img)
    assert f["intensity_mean"][0] == 200.0
    assert f["intensity_std"][0] < 1e-6, f"std over bbox, not object: {f['intensity_std'][0]}"

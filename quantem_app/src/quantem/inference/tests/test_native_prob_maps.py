"""The probability map reaches native coordinates before anything is decided.

The ordering these cover (owner ruling R11):

1. the model predicts on whatever grid it wants;
2. the probability *field* is carried back to the image's own pixel grid;
3. it is quantised to uint8 and that array is stored;
4. **that stored array** is thresholded, in native coordinates;
5. every object-level filter runs there too.

The ordering it replaces thresholded on the model's grid and brought the
resulting binary mask back with nearest-neighbour. The two are not the same
computation, and the tests below pin the three places the difference is
observable: which interpolator carries the field back, which array the threshold
reads, and what a stored map replays to.

Each test names, in its own body or docstring, what it caught -- because every
one of them passes trivially under the old ordering *or* fails loudly under it,
and which of the two is the point.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from quantem.inference import postprocess, resample
from quantem.inference.segmenter import DL_MODEL_NAME, DinoMitoSegmenter
from quantem.inference.specs import get_model_spec

# A native grid twice as fine as the mito head's 8 nm canonical scale: the
# common case in real data (a 4 nm image), and the one where nearest-neighbour
# produces 2x2 staircase blocks.
NATIVE = (256, 320)
NATIVE_NM = 4.0


def _context(native=NATIVE, pixel_size_nm=NATIVE_NM, canonical_nm=8.0):
    return resample.plan_resample(native, pixel_size_nm, canonical_nm)


def _model_field(shape: tuple[int, int]) -> np.ndarray:
    """A smooth, deterministic probability field with a real boundary in it.

    Smooth matters: a field that is only 0 and 1 quantises and interpolates the
    same way under every convention, which would make these tests agree for the
    wrong reason.
    """
    rows = np.linspace(-1.6, 1.6, shape[0], dtype=np.float32)[:, None]
    cols = np.linspace(-1.6, 1.6, shape[1], dtype=np.float32)[None, :]
    blob = np.exp(-((rows - 0.35) ** 2 + (cols + 0.3) ** 2) * 4.0)
    ridge = np.exp(-((rows * 0.9 + cols * 0.4) ** 2) * 9.0) * 0.85
    field = np.clip(blob + 0.55 * ridge, 0.0, 1.0)
    return field.astype(np.float32)


# --- Step 2: the interpolator that carries the field back --------------------


def test_the_resample_back_is_linear_up_area_down_and_never_nearest():
    """One function decides this, and it decides it from the two shapes.

    ``probability_to_native`` used to hard-code ``INTER_LINEAR`` in both
    directions, which is right for the dominant upsample-back and wrong for a
    16 nm image on the 8 nm head.
    """
    up = _context()                                     # 4 nm image, 8 nm head
    down = resample.plan_resample((256, 320), 16.0, 8.0)  # 16 nm image
    identity = resample.plan_resample((256, 320), None, None)  # ER

    assert up.back_upsamples and up.back_factor == pytest.approx(2.0)
    assert not down.back_upsamples and down.back_factor == pytest.approx(0.5)

    assert resample.probability_interpolation(up) == cv2.INTER_LINEAR
    assert resample.probability_interpolation(down) == cv2.INTER_AREA
    assert resample.probability_interpolation(identity) is None

    # The names are what land in provenance, so they are part of the contract.
    assert resample.interpolation_name(cv2.INTER_LINEAR) == "INTER_LINEAR"
    assert resample.interpolation_name(cv2.INTER_AREA) == "INTER_AREA"
    assert resample.interpolation_name(None) == resample.NO_RESAMPLE


def test_inter_area_upsampling_is_nearest_neighbour_in_disguise():
    """The trap the measurement flagged, pinned so nobody "simplifies" into it.

    "Area averaging for a continuous field" is the obvious-sounding choice, and
    OpenCV silently answers it with nearest-neighbour whenever the output is
    larger than the input. Choosing it would reinstate the exact staircase this
    ordering removes while the code and the provenance both claimed otherwise.
    """
    context = _context()
    field = _model_field(context.model_shape)
    height, width = context.native_shape

    area = cv2.resize(field, (width, height), interpolation=cv2.INTER_AREA)
    nearest = cv2.resize(field, (width, height), interpolation=cv2.INTER_NEAREST)
    linear = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)

    assert np.array_equal(area, nearest), (
        "cv2.INTER_AREA no longer degenerates to nearest when upsampling on "
        "this OpenCV build. Re-measure before treating 'area' as a continuous "
        "interpolator anywhere."
    )
    assert not np.array_equal(linear, nearest)
    assert resample.probability_interpolation(context) == cv2.INTER_LINEAR


def test_the_old_ordering_is_the_new_one_with_a_nearest_interpolator():
    """Ordering A == ordering B run with NEAREST, exactly, at an integer factor.

    This is the whole content of the change stated as an identity: nothing about
    thresholds or quantisation differs between the two orderings, only which
    interpolator carries the field. It is also what makes the *next* assertion
    meaningful -- the difference below is the interpolator and nothing else.
    """
    context = _context()
    field = _model_field(context.model_shape)
    height, width = context.native_shape

    old_ordering = resample.mask_to_native(postprocess.binarize(field, 0.5), context)
    nearest_back = cv2.resize(
        field, (width, height), interpolation=cv2.INTER_NEAREST
    )
    new_ordering_with_nearest = resample.binarize_quantized(
        resample.quantize_probability(nearest_back), 0.5
    )
    assert np.array_equal(old_ordering, new_ordering_with_nearest)

    shipped = resample.NativeProbabilityMap.from_model_grid(field, context)
    differing = int((old_ordering != shipped.foreground(0.5)).sum())
    assert differing > 0, (
        "the shipped interpolator produced the nearest-neighbour boundary; "
        "probability_to_native has regressed to NEAREST or to INTER_AREA"
    )


def test_the_interpolated_boundary_is_smoother_than_the_staircase():
    """Perimeter falls, which is the measured effect on published numbers.

    Not a tolerance check on a magic number -- the direction is the claim. The
    measurement put it at -9.6 % median for mitochondria at a 2x back-factor
    before the closing, and -2.6 to -5.2 % after it.
    """
    from skimage.measure import label, regionprops

    context = _context()
    field = _model_field(context.model_shape)

    old_mask = resample.mask_to_native(postprocess.binarize(field, 0.5), context)
    new_mask = resample.NativeProbabilityMap.from_model_grid(
        field, context
    ).foreground(0.5)

    def perimeter(mask):
        regions = regionprops(label(mask))
        assert regions, "the fixture must produce at least one object"
        return max(region.perimeter_crofton for region in regions)

    assert perimeter(new_mask) < perimeter(old_mask)


# --- Step 3: quantisation ----------------------------------------------------


def test_every_stored_level_survives_the_float_round_trip():
    """uint8 -> float -> uint8 is the identity, for all 256 levels.

    Load-bearing twice over: the cached-map path re-adopts a stored map through
    this round trip, and so does the replay reader
    (``prob_maps.persistence.load_stored_native_map``), which recovers the
    stored bytes by requantising rather than opening the PNG a second time.
    """
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    recovered = resample.quantize_probability(resample.dequantize_probability(levels))
    assert np.array_equal(recovered, levels)


def test_the_quantiser_rounds_to_nearest_not_down():
    """A truncating cast is a biased quantiser, and the bias moves pixels.

    ``(p * 255).astype(uint8)`` -- what the store used to do -- always rounds
    down, so the stored value is up to 1/255 below the truth. Measured on real
    maps that flips up to 13 956 pixels and loses an object at the default
    threshold. These are the smallest cases that show it.
    """
    field = np.array([[0.0, 0.5, 0.9999, 1.0], [0.002, 0.4999, 0.5001, 0.998]],
                     dtype=np.float32)
    stored = resample.quantize_probability(field)
    truncated = (np.clip(field, 0, 1) * 255).astype(np.uint8)

    assert stored.tolist() == [[0, 128, 255, 255], [1, 127, 128, 254]]
    assert truncated.tolist() == [[0, 127, 254, 255], [0, 127, 127, 254]]
    # At the product default the round convention cuts at exactly 0.5; the
    # truncating one cuts a level low and calls 0.4999 foreground.
    assert resample.binarize_quantized(stored, 0.5).tolist() == [
        [False, True, True, True],
        [False, False, True, True],
    ]
    assert (truncated >= 128).tolist() == [
        [False, False, True, True],
        [False, False, False, True],
    ]


def test_a_threshold_becomes_a_level_and_the_level_is_recorded():
    """The dial has 255 stops; say which one was used and what it cuts at."""
    assert resample.threshold_level(0.5) == 128
    assert resample.realised_threshold(128) == pytest.approx(0.5)
    for exact in (0.1, 0.3, 0.5, 0.7, 0.9):
        level = resample.threshold_level(exact)
        assert resample.realised_threshold(level) == pytest.approx(exact, abs=1e-12)
    # Where 255t + 0.5 is not an integer the map cannot cut exactly where asked,
    # and the honest record is the cut that happened.
    level = resample.threshold_level(0.4)
    assert resample.realised_threshold(level) == pytest.approx(0.4 + 1 / 510, abs=1e-9)
    assert abs(resample.realised_threshold(level) - 0.4) <= 1 / 510 + 1e-12
    # The ends stay meaningful rather than wrapping.
    assert resample.threshold_level(0.0) == 0
    assert resample.binarize_quantized(np.zeros((2, 2), np.uint8), 0.0).all()
    assert not resample.binarize_quantized(
        np.full((2, 2), 255, np.uint8), 1.0
    ).any()


def test_quantising_is_exact_regardless_of_array_size():
    """The chunked float64 path must not depend on where a chunk boundary falls."""
    rng = np.random.default_rng(17)
    field = rng.random((1024, 1024)).astype(np.float32)
    reference = np.floor(field.astype(np.float64) * 255 + 0.5).astype(np.uint8)
    assert np.array_equal(resample.quantize_probability(field), reference)

    original = field.astype(np.float64)
    untouched = original.copy()
    resample.quantize_probability(original)
    assert np.array_equal(original, untouched), "quantising mutated its input"


# --- Step 4: the threshold reads the stored bytes ----------------------------


def _fake_engine(prob_by_shape, spec):
    """Stand in for ``engine`` so the segmenter runs with no weights."""

    def predict_region(_model, image, *, pixel_size_nm=None, **_kwargs):
        context = resample.plan_resample(
            image.shape[:2], pixel_size_nm, spec.canonical_nm
        )
        return SimpleNamespace(
            prob=prob_by_shape(context.model_shape),
            context=context,
            plan=None,
        )

    return SimpleNamespace(
        load_model=lambda *_a, **_k: SimpleNamespace(device="cpu"),
        load_adapted_model=lambda *_a, **_k: SimpleNamespace(device="cpu"),
        predict_region=predict_region,
        estimate_tiles=lambda *_a, **_k: 1,
        LoadedModel=object,
        RegionPrediction=object,
    )


def _run(segmenter, image, *, pixel_size_nm=NATIVE_NM, field=_model_field):
    spec = get_model_spec("quantem", "mito")
    # No job is watching. Said explicitly rather than left to chance:
    # ``JobReporter.__init__`` activates itself on the constructing thread and
    # nothing deactivates it, so any earlier test in the session that built one
    # leaves this thread owning a reporter -- and these tests, which need no
    # database, would then try to write progress rows to one.
    with (
        patch("quantem.inference.segmenter.engine", _fake_engine(field, spec)),
        patch("quantem.inference.segmenter._job_progress", return_value=None),
    ):
        return segmenter.run_dl_inference(
            image, {}, None, pixel_size_nm=pixel_size_nm
        )


def test_a_fresh_run_thresholds_the_stored_map_not_the_float_it_came_from():
    """The requirement R11 states in one line, and the reason it is stated.

    The discriminator is a threshold the 255-level map cannot express exactly:
    at t = 0.4 the stored map cuts at 0.401961, so any pixel whose probability
    lands in [0.4, 0.401961) is foreground if the *float* was thresholded and
    background if the *stored map* was. A run that quietly thresholded the float
    would still look right at 0.5 and disagree with its own dial everywhere
    else, which is precisely the drift this ordering removes.
    """
    image = np.zeros((64, 64), dtype=np.uint8)
    boundary = np.full((32, 32), 0.401, dtype=np.float32)   # inside the gap
    boundary[:4, :4] = 0.95                                  # something certain

    segmenter = DinoMitoSegmenter(source_model="quantem:mito", fg_threshold=0.4)
    maps = _run(segmenter, image, field=lambda _shape: boundary)

    native = segmenter.native_probability_map
    assert native is not None and native.data.dtype == np.uint8
    assert native.shape == (64, 64)

    float_decision = maps[DL_MODEL_NAME] >= 0.4
    stored_decision = segmenter._foreground_mask(maps[DL_MODEL_NAME])
    assert float_decision.sum() > stored_decision.sum(), (
        "the fixture no longer separates the two decisions"
    )
    assert np.array_equal(
        stored_decision, resample.binarize_quantized(native.data, 0.4)
    )


def test_the_returned_map_is_the_stored_map_read_back_as_floats():
    """``InferenceResult.prob_maps`` is a view of the stored bytes, not a peer.

    Two arrays that disagree by a fraction of a level is how a per-object
    ``mean_prob`` comes to describe a boundary the run did not draw.
    """
    image = np.zeros(NATIVE, dtype=np.uint8)
    segmenter = DinoMitoSegmenter(source_model="quantem:mito")
    maps = _run(segmenter, image)

    native = segmenter.native_probability_map
    assert np.array_equal(
        resample.quantize_probability(maps[DL_MODEL_NAME]), native.data
    )
    assert maps[DL_MODEL_NAME].shape == NATIVE


def test_the_no_resample_case_is_exactly_the_previous_pipeline():
    """ER runs at native scale, so there is no ordering to reverse.

    With nothing resampled the only thing that could move is the uint8 step, and
    at a threshold the 255 levels express exactly it does not: the two orderings
    agree pixel for pixel. If this ever fails, the change has leaked into the
    path it was never meant to touch.
    """
    context = resample.plan_resample(NATIVE, 5.0, None)   # ER: canonical_nm None
    assert context.is_identity

    field = _model_field(NATIVE)
    stored = resample.NativeProbabilityMap.from_model_grid(field, context)
    assert stored.interpolation == resample.NO_RESAMPLE
    assert stored.back_factor == pytest.approx(1.0)

    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        previous = resample.mask_to_native(
            postprocess.binarize(field, threshold), context
        )
        assert np.array_equal(previous, stored.foreground(threshold)), (
            f"identity path differs at t={threshold}"
        )


def test_object_level_filters_all_run_on_native_pixels():
    """Area floor and closing radius are native-pixel constants; keep them there.

    A 60 px mito floor applied on a 2x-coarser model grid is a 240 px floor in
    the image, and nothing in the object record would say so.
    """
    image = np.zeros(NATIVE, dtype=np.uint8)
    segmenter = DinoMitoSegmenter(source_model="quantem:mito", fg_threshold=0.5)
    maps = _run(segmenter, image)
    prob = segmenter.combine_prob_maps(maps)

    mask = segmenter._foreground_mask(prob)
    assert mask.shape == NATIVE

    labels = postprocess.postprocess_mask(mask, close_radius=3, min_area=60)
    assert labels.shape == NATIVE
    survivors = np.bincount(labels.ravel())
    survivors[0] = 0
    assert (survivors[survivors > 0] >= 60).all()


# --- Provenance --------------------------------------------------------------


def test_provenance_records_the_interpolator_the_quantiser_and_the_cut():
    """A reader with only the pack id and the requested threshold cannot
    reconstruct the boundary. These are the missing degrees of freedom."""
    image = np.zeros(NATIVE, dtype=np.uint8)
    segmenter = DinoMitoSegmenter(source_model="quantem:mito", fg_threshold=0.4)
    _run(segmenter, image)

    metadata = segmenter.get_probability_map_metadata(DL_MODEL_NAME)
    assert metadata["native_coordinates"] is True
    assert metadata["resample_interpolation"] == "INTER_LINEAR"
    assert metadata["resample_back_factor"] == pytest.approx(2.0)
    assert metadata["quantization"] == resample.QUANTIZATION_ID
    assert metadata["quantization_levels"] == 255
    assert metadata["threshold"] == pytest.approx(0.4)
    assert metadata["threshold_level"] == resample.threshold_level(0.4)
    assert metadata["realised_threshold"] == pytest.approx(0.4 + 1 / 510, abs=1e-9)
    assert metadata["thresholded_on"] == "stored_native_uint8"


def test_adopting_a_stored_map_refuses_anything_but_uint8():
    """The stored map is bytes. A float "stored map" is a second decision."""
    segmenter = DinoMitoSegmenter(source_model="quantem:mito")
    with pytest.raises(TypeError):
        resample.NativeProbabilityMap.from_stored(np.zeros((4, 4), np.float32))
    with pytest.raises(TypeError):
        segmenter.adopt_native_probability_map(
            resample.NativeProbabilityMap(data=np.zeros((4, 4), np.float32))
        )


def test_the_threshold_can_be_moved_without_touching_the_stored_map():
    """The dial's backend contract: same bytes, new level, new objects."""
    image = np.zeros(NATIVE, dtype=np.uint8)
    segmenter = DinoMitoSegmenter(source_model="quantem:mito", fg_threshold=0.5)
    _run(segmenter, image)
    stored = segmenter.native_probability_map
    before = stored.data.copy()

    counts = {}
    for threshold in (0.2, 0.5, 0.8):
        segmenter.set_fg_threshold(threshold)
        counts[threshold] = int(stored.foreground(threshold).sum())
    assert np.array_equal(stored.data, before), "moving the dial rewrote the map"
    assert counts[0.2] > counts[0.5] > counts[0.8]

    with pytest.raises(ValueError):
        segmenter.set_fg_threshold(1.4)

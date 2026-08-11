"""Replaying a stored probability map is the same computation as a fresh run.

The acceptance test for owner rulings R9 and R11, written as the round trip a
user performs when they move the accuracy dial:

* run the model at threshold T, which stores a probability map in the image's
  own pixel coordinates;
* later, *without running the model*, re-threshold that stored map at T;
* the objects must be **identical** -- same count, same polygon, same centroid,
  same bounding box, same measured area and perimeter -- not merely similar.

"Identical" is the whole requirement. A dial that produced objects close to
what a re-run would produce is a dial that quietly changes a scientist's
candidate set every time they touch it, and nothing on screen would say so.

Exactness here is structural, not lucky: the run thresholds the stored uint8
array, and the replay thresholds the same array with the same function
(:func:`quantem.inference.resample.binarize_quantized`), after which both go
through one shared extraction path. These tests exist to keep it that way, and
to fail if anything reintroduces a second decision procedure -- a threshold
applied to the float, a re-quantisation on write, an interpolation on read.

The run below is deliberately a **resampled** one: a 5 nm image on the 8 nm mito
head, so the probability map is upsampled 1.6x on its way back to native pixels.
Under the previous ordering there was no native-coordinate probability map to
replay at all, so this test could not have been written.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from django.test import TestCase

from quantem.inference import resample
from quantem.inference.segmenter import DL_MODEL_NAME, DinoMitoSegmenter
from quantem.inference.specs import get_model_spec
from quantem.seg_core.db.extraction import extract_and_save_segments, resolve_min_area
from quantem.seg_core.db.inference import (
    StoredMapUnavailable,
    replay_stored_probability_map,
)
from quantem.seg_core.db.prob_maps import load_prob_map_from_path
from quantem.segmentation.models import (
    ImageSegmentation,
    ProbabilityMap,
    SegmentationConfig,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import run_segmentation_full_task
from quantem.segmentation.prob_maps.persistence import load_stored_native_map
from quantem.segmentation.run_identity import run_identity_from_segmenter
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256
PIXEL_SIZE_NM = 5.0          # the mito head is 8 nm: a 1.6x upsample back
MITO_INTERNAL_NAME = "quantem_internal_mito"
SOURCE_MODEL = "quantem:mito"

#: Four thresholds spanning the useful range, plus the product default.
THRESHOLDS = (0.25, 0.4, 0.5, 0.65, 0.8)


def _model_field(shape: tuple[int, int]) -> np.ndarray:
    """Five blobs of different confidence, on the model's own grid.

    Different peaks on purpose: the object count has to *move* across the dial,
    or a test that compares two identical answers proves nothing.
    """
    height, width = shape
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    field = np.zeros(shape, dtype=np.float32)
    blobs = (
        (0.30, 0.30, 16.0, 0.97),
        (0.30, 0.70, 13.0, 0.83),
        (0.70, 0.30, 12.0, 0.72),
        (0.70, 0.70, 11.0, 0.58),
        (0.50, 0.50, 9.0, 0.45),
    )
    for row_frac, col_frac, sigma, peak in blobs:
        centre_row = row_frac * height
        centre_col = col_frac * width
        squared = ((rows - centre_row) ** 2 + (cols - centre_col) ** 2) / (
            2.0 * sigma * sigma
        )
        field = np.maximum(field, peak * np.exp(-squared))
    # A little smooth background so quantisation and interpolation have
    # something to disagree about if they are going to.
    field += 0.04 * np.sin(rows / 9.0) * np.cos(cols / 11.0)
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def _fake_engine():
    """The ``engine`` module surface the segmenter uses, with no weights.

    Everything real about the path under test is kept: the resample plan, the
    crossing back to native pixels, the quantisation, the threshold, the
    morphology and the extraction. Only the forward pass is replaced, because a
    released pack is a 1 GB download and this is not a model test.
    """
    spec = get_model_spec("quantem", "mito")

    def predict_region(_model, image, *, pixel_size_nm=None, **_kwargs):
        context = resample.plan_resample(
            image.shape[:2], pixel_size_nm, spec.canonical_nm
        )
        return SimpleNamespace(
            prob=_model_field(context.model_shape), context=context, plan=None
        )

    return SimpleNamespace(
        load_model=lambda *_a, **_k: SimpleNamespace(device="cpu"),
        load_adapted_model=lambda *_a, **_k: SimpleNamespace(device="cpu"),
        predict_region=predict_region,
        estimate_tiles=lambda *_a, **_k: 1,
    )


def _segmenter(threshold: float) -> DinoMitoSegmenter:
    return DinoMitoSegmenter(
        source_model=SOURCE_MODEL,
        fg_threshold=threshold,
        pixel_size_nm=PIXEL_SIZE_NM,
    )


def _fingerprint(segmentation: ImageSegmentation) -> list[tuple]:
    """Everything about a candidate that a later measurement can read.

    Not just the count: a dial that produced the same number of objects with
    different boundaries would pass a count check and change every perimeter in
    the paper.
    """
    rows = SegmentObject.objects.filter(
        segmentation=segmentation, label_state="CANDIDATE"
    ).order_by("centroid_x", "centroid_y")
    out = []
    for row in rows:
        features = row.features or {}
        out.append(
            (
                round(float(row.centroid_x), 9),
                round(float(row.centroid_y), 9),
                round(float(row.bbox_minx), 9),
                round(float(row.bbox_miny), 9),
                round(float(row.bbox_maxx), 9),
                round(float(row.bbox_maxy), 9),
                features.get("area"),
                features.get("perimeter"),
                features.get("mean_prob"),
                bytes(row.geometry_wkb),
            )
        )
    return out


class ThresholdReplayTests(TestCase):
    """Fresh run vs replay, at five thresholds, on one resampled image."""

    def setUp(self):
        self.image = create_small_test_image(
            "Replay", width=SIZE, height=SIZE, textured=True
        )
        asset = self.image.asset
        asset.pixel_size_nm = PIXEL_SIZE_NM
        asset.save(update_fields=["pixel_size_nm"])
        self.segmentation = ImageSegmentation.objects.create(
            asset=asset, segmentation_type=get_or_create_mitochondria_type()
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)

    # --- helpers ---

    def _fresh_run(self, threshold: float) -> tuple[int, DinoMitoSegmenter]:
        segmenter = _segmenter(threshold)
        with (
            patch("quantem.inference.segmenter.engine", _fake_engine()),
            patch(
                "quantem.segmentation.organelle_tasks.get_segmenter",
                return_value=segmenter,
            ),
        ):
            count = run_segmentation_full_task(
                segmentation_id=str(self.segmentation.id),
                segmentation_type=MITO_INTERNAL_NAME,
                source_model=SOURCE_MODEL,
            )
        return count, segmenter

    def _replay(self, threshold: float) -> int:
        """The dial: no model, no engine patch -- nothing may try to load one."""
        segmenter = _segmenter(threshold)
        result, image_array = replay_stored_probability_map(
            segmenter, self.segmentation, threshold=threshold
        )
        area_floor = resolve_min_area(segmenter, None)
        return extract_and_save_segments(
            segmenter,
            self.segmentation,
            result,
            image_array,
            None,
            min_area=area_floor,
            run_identity=run_identity_from_segmenter(
                segmenter,
                run_id="replay",
                pack_id_fallback=SOURCE_MODEL,
                native_pixel_size_nm=PIXEL_SIZE_NM,
                min_area=area_floor,
            ),
        )

    # --- the acceptance test ---

    def test_replay_at_t_is_identical_to_a_fresh_run_at_t(self):
        """The non-negotiable one. Five thresholds, exact equality each time."""
        moved = set()
        for threshold in THRESHOLDS:
            with self.subTest(threshold=threshold):
                fresh_count, _ = self._fresh_run(threshold)
                fresh = _fingerprint(self.segmentation)
                assert fresh_count == len(fresh) > 0, (
                    f"the fixture found no objects at t={threshold}"
                )

                replay_count = self._replay(threshold)
                replayed = _fingerprint(self.segmentation)

                assert replay_count == fresh_count
                assert replayed == fresh, (
                    f"replay at t={threshold} produced different objects than a "
                    "fresh run at the same threshold"
                )
                moved.add((fresh_count, tuple(row[6] for row in fresh)))

        assert len(moved) > 1, (
            "every threshold produced the same objects, so this test would pass "
            "on a dial that ignored its input"
        )

    def test_the_stored_file_holds_exactly_the_bytes_the_run_thresholded(self):
        """"Threshold the stored map" is only true if the store is lossless.

        The run thresholds an array in memory; the dial thresholds what came
        back off disk. If PNG storage moved a single level the two would part
        company, and the acceptance test above would be measuring a coincidence.
        """
        _, segmenter = self._fresh_run(0.5)
        in_memory = segmenter.native_probability_map
        assert in_memory is not None

        from_disk = load_prob_map_from_path(
            self.segmentation, DL_MODEL_NAME, segmenter.prob_map_prefix, None
        )
        assert from_disk is not None
        assert np.array_equal(
            resample.quantize_probability(from_disk), in_memory.data
        )

        stored = load_stored_native_map(
            segmentation=self.segmentation,
            segmenter=segmenter,
            model_name=DL_MODEL_NAME,
        )
        assert stored is not None
        assert stored.native.data.dtype == np.uint8
        assert np.array_equal(stored.native.data, in_memory.data)

    def test_the_map_is_in_the_images_own_pixel_coordinates(self):
        """1.6x upsampled back from the model grid, and it lands on the image."""
        _, segmenter = self._fresh_run(0.5)
        native = segmenter.native_probability_map

        assert native.shape == (SIZE, SIZE)
        assert native.interpolation == "INTER_LINEAR"
        assert native.back_factor == pytest.approx(SIZE / round(SIZE * 0.625), rel=1e-3)

        record = ProbabilityMap.objects.get(segmentation=self.segmentation)
        assert record.metadata["native_coordinates"] is True
        assert record.metadata["resample_interpolation"] == "INTER_LINEAR"
        assert record.metadata["quantization"] == resample.QUANTIZATION_ID
        assert record.metadata["thresholded_on"] == "stored_native_uint8"
        assert record.metadata["threshold_level"] == resample.threshold_level(0.5)
        assert record.metadata["realised_threshold"] == pytest.approx(0.5)

    def test_the_replay_provenance_records_the_new_cut_not_the_old_one(self):
        """A replayed candidate set was made at the dial's threshold."""
        self._fresh_run(0.5)
        segmenter = _segmenter(0.3)
        replay_stored_probability_map(
            segmenter, self.segmentation, threshold=0.3
        )
        metadata = segmenter.get_probability_map_metadata(DL_MODEL_NAME)
        assert metadata["threshold"] == pytest.approx(0.3)
        assert metadata["threshold_level"] == resample.threshold_level(0.3)
        assert metadata["realised_threshold"] == pytest.approx(0.3)
        assert metadata["thresholded_on"] == "stored_native_uint8"
        # The map itself is untouched by the dial.
        assert metadata["resample_interpolation"] == "INTER_LINEAR"

    def test_replay_never_loads_a_model(self):
        """The point of the dial is that it is not a run.

        ``engine`` is left unpatched here on purpose: any attempt to resolve a
        pack raises, so this fails loudly rather than silently taking 30 s.
        """
        self._fresh_run(0.5)
        segmenter = _segmenter(0.6)
        with patch.object(
            DinoMitoSegmenter, "load_models", side_effect=AssertionError("loaded")
        ):
            result, _image = replay_stored_probability_map(
                segmenter, self.segmentation, threshold=0.6
            )
        assert result.prob.shape == (SIZE, SIZE)

    def test_no_stored_map_says_so_instead_of_inventing_one(self):
        """Nothing to replay is a different answer from "found nothing"."""
        assert not ProbabilityMap.objects.filter(
            segmentation=self.segmentation
        ).exists()
        with pytest.raises(StoredMapUnavailable) as caught:
            replay_stored_probability_map(
                _segmenter(0.5), self.segmentation, threshold=0.5
            )
        assert "running the model again" in str(caught.value)

    def test_a_map_that_does_not_fit_the_image_is_refused(self):
        """An image replaced under a segmentation invalidates its map."""
        _, segmenter = self._fresh_run(0.5)
        record = ProbabilityMap.objects.get(segmentation=self.segmentation)
        stored = load_stored_native_map(
            segmentation=self.segmentation,
            segmenter=segmenter,
            model_name=DL_MODEL_NAME,
        )
        assert stored is not None

        # The image now reads back at a different size than the map covers.
        smaller = np.zeros((SIZE // 2, SIZE // 2), dtype=np.uint8)
        with (
            patch(
                "quantem.seg_core.db.inference.load_image_array",
                return_value=(smaller, None),
            ),
            pytest.raises(StoredMapUnavailable) as caught,
        ):
            replay_stored_probability_map(
                _segmenter(0.5), self.segmentation, threshold=0.5
            )
        assert "run again" in str(caught.value)
        assert record.id  # the row itself is left alone for a human to look at


class StoredMapQuantisationTests(TestCase):
    """What lands in the PNG is the level the probability rounds to.

    Separated from the replay tests because it is a property of the store, not
    of the dial: a float caller (an uploaded map, a segmenter that does not keep
    its own) must get the same convention as the inference path, or two maps in
    the same database mean different things.
    """

    def setUp(self):
        self.image = create_small_test_image("Store", width=32, height=32)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _store(self, array) -> np.ndarray:
        from quantem.seg_core.db.prob_maps import save_probability_map

        save_probability_map(
            self.segmentation, DL_MODEL_NAME, array, "mito", "mito_generated"
        )
        read_back = load_prob_map_from_path(
            self.segmentation, DL_MODEL_NAME, "mito", None
        )
        return resample.quantize_probability(read_back)

    def test_a_float_map_is_rounded_to_nearest_not_truncated(self):
        """``(p * 255).astype(uint8)`` puts 0.5 at level 127, which is 0.498.

        The bias is one-sided, so it does not average out: every stored value
        sits up to 1/255 below the probability it represents, and a threshold
        that should have cut at 0.5 cuts at 0.49804.
        """
        field = np.full((32, 32), 0.5, dtype=np.float32)
        field[0, :4] = [0.0, 0.002, 0.9999, 1.0]
        stored = self._store(field)

        assert stored[1, 0] == 128, "0.5 must store as 128, not 127"
        assert stored[0, :4].tolist() == [0, 1, 255, 255]
        truncating = (np.clip(field, 0, 1) * 255).astype(np.uint8)
        assert truncating[1, 0] == 127

    def test_an_already_quantised_map_is_stored_byte_for_byte(self):
        """The run's own array must not be re-derived on the way to disk."""
        levels = (np.arange(1024) % 256).astype(np.uint8).reshape(32, 32)
        assert np.array_equal(self._store(levels), levels)


class ReplayEquivalenceWithoutResamplingTests(TestCase):
    """The ER case: no resampling, so there is no ordering to reverse.

    R11 requires that this path be *exactly* what it was before the change.
    Nothing here is upsampled, nothing is interpolated, and the only step that
    could move a pixel is the uint8 quantisation -- which at a threshold the 255
    levels express exactly does not.
    """

    def setUp(self):
        self.image = create_small_test_image(
            "Replay ER", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)

    def test_a_native_scale_run_thresholds_the_same_pixels_as_before(self):
        from quantem.inference import postprocess

        context = resample.plan_resample((SIZE, SIZE), PIXEL_SIZE_NM, None)
        assert context.is_identity

        field = _model_field((SIZE, SIZE))
        native = resample.NativeProbabilityMap.from_model_grid(field, context)
        assert native.interpolation == resample.NO_RESAMPLE

        for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
            previous = resample.mask_to_native(
                postprocess.binarize(field, threshold), context
            )
            assert np.array_equal(previous, native.foreground(threshold)), (
                f"the native-scale path changed at t={threshold}"
            )

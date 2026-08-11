"""A probability map from an older build is refused, not reinterpreted.

The dial's exactness (owner rulings R9 and R11) rests on one fact: the bytes on
disk are the bytes the run thresholded, quantised by a known rule, in the
image's own pixel coordinates. A map written before that ordering existed
satisfies none of those and records none of them, and it is byte-compatible with
one that does -- both are an 8-bit PNG the size of the image.

So the reader had no way to tell them apart and did not try: it handed any PNG
to :meth:`~quantem.inference.resample.NativeProbabilityMap.from_stored`, whose
defaults then filled the silence with the *current* conventions. The result was
worse than a wrong number. The replay re-decided pixels under rules the file was
never written under, and then described itself in provenance as the pipeline
that had not touched it, so nothing downstream -- not the object rows, not the
export manifest -- could say the two candidate sets were different computations.

The two quantisation rules that have existed here differ on roughly a fifth to
two fifths of stored bytes by one level, which moves object *counts* by around a
percent (measured across five thresholds in the R11 report). A scientist's count
is the output of this application. It may not change because a file was old.

These tests fix the answer: refuse, with a sentence that says the model has to
run again, and keep that distinct from "there is no map here", which is a
different sentence and a different situation.
"""

from __future__ import annotations

import numpy as np
from django.test import TestCase
from PIL import Image

from quantem.inference import resample
from quantem.inference.segmenter import DL_MODEL_NAME, DinoMitoSegmenter
from quantem.inference.specs import get_model_spec
from quantem.seg_core.db.inference import (
    StoredMapUnavailable,
    replay_stored_probability_map,
)
from quantem.seg_core.db.prob_maps import get_prob_map_file_path
from quantem.segmentation.models import (
    ImageSegmentation,
    ProbabilityMap,
    SegmentationConfig,
    SegmentObject,
)
from quantem.segmentation.prob_maps.persistence import (
    LEGACY_MAP_MESSAGE,
    THRESHOLDED_ON_STORED_MAP,
    load_stored_native_map,
    persist_run_probability_maps,
    replay_provenance_problem,
    stored_map_is_replayable,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 128
#: The mito head is 8 nm, so a 5 nm image is a 1.6x upsample on the way back:
#: the case where an interpolator and a back factor are real facts about the
#: file rather than both being the no-op the reader used to assume.
PIXEL_SIZE_NM = 5.0
SOURCE_MODEL = "quantem:mito"
THRESHOLD = 0.5

#: Every marker the run's writer records together. Each one alone is enough to
#: refuse, because a file carrying four of the five did not come from that
#: writer either.
MARKERS = (
    "native_coordinates",
    "thresholded_on",
    "quantization",
    "resample_interpolation",
    "resample_back_factor",
)


def _model_field(shape: tuple[int, int]) -> np.ndarray:
    """Four blobs on the model's grid, with a background that is not flat."""
    height, width = shape
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    field = np.zeros(shape, dtype=np.float32)
    for row_frac, col_frac, sigma, peak in (
        (0.30, 0.30, 9.0, 0.97),
        (0.30, 0.70, 8.0, 0.81),
        (0.70, 0.30, 7.0, 0.70),
        (0.70, 0.70, 6.0, 0.57),
    ):
        squared = ((rows - row_frac * height) ** 2 + (cols - col_frac * width) ** 2) / (
            2.0 * sigma * sigma
        )
        field = np.maximum(field, peak * np.exp(-squared))
    field += 0.04 * np.sin(rows / 7.0) * np.cos(cols / 9.0)
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def _legacy_quantise(prob: np.ndarray) -> np.ndarray:
    """What the previous build wrote: ``(p * 255).astype(uint8)``, truncating.

    One-sided, so it does not average out: every level sits at or below the one
    round-to-nearest would have stored.
    """
    return (np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )


class LegacyStoredMapTests(TestCase):
    """One stored map, made by the real writer, then aged by hand."""

    def setUp(self):
        image = create_small_test_image(
            "Legacy map", width=SIZE, height=SIZE, textured=True
        )
        asset = image.asset
        asset.pixel_size_nm = PIXEL_SIZE_NM
        asset.save(update_fields=["pixel_size_nm"])
        self.segmentation = ImageSegmentation.objects.create(
            asset=asset, segmentation_type=get_or_create_mitochondria_type()
        )
        SegmentationConfig.objects.get_or_create(segmentation=self.segmentation)
        self.segmenter = self._store_a_run()

    # --- fixture helpers ---

    def _segmenter(self, threshold: float = THRESHOLD) -> DinoMitoSegmenter:
        return DinoMitoSegmenter(
            source_model=SOURCE_MODEL,
            fg_threshold=threshold,
            pixel_size_nm=PIXEL_SIZE_NM,
        )

    def _store_a_run(self) -> DinoMitoSegmenter:
        """Write a map the way a run writes one: real resample, real writer.

        No model runs -- only the forward pass is stood in for -- so the
        provenance under test is the provenance the shipped path records, not a
        dictionary this test made up.
        """
        segmenter = self._segmenter()
        spec = get_model_spec("quantem", "mito")
        context = resample.plan_resample(
            (SIZE, SIZE), PIXEL_SIZE_NM, spec.canonical_nm
        )
        assert not context.is_identity, "the fixture must exercise a resample"
        native = resample.NativeProbabilityMap.from_model_grid(
            _model_field(context.model_shape), context
        )
        segmenter.adopt_native_probability_map(native)
        written = persist_run_probability_maps(
            segmentation=self.segmentation,
            segmenter=segmenter,
            prob_maps={DL_MODEL_NAME: native.as_float()},
        )
        assert len(written) == 1
        return segmenter

    @property
    def _record(self) -> ProbabilityMap:
        return ProbabilityMap.objects.get(segmentation=self.segmentation)

    def _map_path(self):
        return get_prob_map_file_path(
            self.segmentation, DL_MODEL_NAME, self.segmenter.prob_map_prefix, None
        )

    def _stored_bytes(self) -> np.ndarray:
        with Image.open(self._map_path()) as handle:
            return np.array(handle.convert("L"), dtype=np.uint8)

    def _age_the_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Turn the stored map into one the previous build would have left.

        Both halves of it: the bytes are re-quantised by the truncating rule
        from the same underlying probabilities, and the row is stripped back to
        the metadata that build recorded. Returns ``(today, legacy)`` bytes.
        """
        today = self._stored_bytes()
        # today = floor(255p + 0.5), so p is within 1/510 of today/255; the
        # legacy file for the same field holds floor(255p).
        probabilities = np.clip(today.astype(np.float64) / 255.0 - 0.5 / 255.0, 0.0, 1.0)
        legacy = _legacy_quantise(probabilities)
        Image.fromarray(legacy, mode="L").save(self._map_path())

        record = self._record
        record.metadata = {
            "model_type": DL_MODEL_NAME,
            "mito_generated": True,
            "threshold": THRESHOLD,
            "pack_id": SOURCE_MODEL,
        }
        record.save(update_fields=["metadata"])
        return today, legacy

    def _load(self):
        return load_stored_native_map(
            segmentation=self.segmentation,
            segmenter=self.segmenter,
            model_name=DL_MODEL_NAME,
        )

    # --- the map this build writes is still read ---

    def test_a_map_this_build_wrote_is_read_with_its_own_provenance(self):
        """The refusal must not be a blanket one, and must not default.

        Both halves matter: an interpolator of "none" and a back factor of 1.0
        are what the reader used to invent, and they are wrong here by 1.6x.
        """
        stored = self._load()
        assert stored is not None
        assert np.array_equal(
            stored.native.data, self.segmenter.native_probability_map.data
        )
        assert stored.native.interpolation == "INTER_LINEAR"
        assert stored.native.back_factor > 1.5
        assert stored.native.quantization == resample.QUANTIZATION_ID
        assert stored_map_is_replayable(self._record.metadata)
        assert replay_provenance_problem(dict(self._record.metadata)) is None

    def test_nothing_stored_is_still_nothing_to_replay(self):
        """"No map here" stays a different answer from "this map is refused"."""
        untouched = create_small_test_image("No map", width=SIZE, height=SIZE)
        other = ImageSegmentation.objects.create(
            asset=untouched.asset,
            segmentation_type=self.segmentation.segmentation_type,
        )
        assert (
            load_stored_native_map(
                segmentation=other,
                segmenter=self.segmenter,
                model_name=DL_MODEL_NAME,
            )
            is None
        )

    # --- the map an older build wrote is refused ---

    def test_the_two_conventions_disagree_on_the_pixels_they_call_foreground(self):
        """Why the refusal is worth having, measured on this fixture.

        If the two byte arrays agreed there would be nothing to protect and this
        whole file would be ceremony.
        """
        today, legacy = self._age_the_map()
        differing = int(np.count_nonzero(today != legacy))
        assert differing > 0
        today_fg = int(resample.binarize_quantized(today, THRESHOLD).sum())
        legacy_fg = int(resample.binarize_quantized(legacy, THRESHOLD).sum())
        assert today_fg != legacy_fg, (
            f"the aged fixture threshold-matches the current one "
            f"({today_fg} px both ways), so it cannot demonstrate the drift"
        )

    def test_a_map_from_the_previous_build_is_refused_by_the_reader(self):
        self._age_the_map()
        assert not stored_map_is_replayable(self._record.metadata)
        with self.assertRaises(StoredMapUnavailable) as caught:
            self._load()
        message = str(caught.exception)
        assert message == LEGACY_MAP_MESSAGE
        assert "run on this image again" in message

    def test_the_dial_refuses_it_too_and_writes_no_objects(self):
        """The caller's contract, end to end: refuse, do not re-decide."""
        self._age_the_map()
        assert not SegmentObject.objects.filter(
            segmentation=self.segmentation
        ).exists()
        with self.assertRaises(StoredMapUnavailable) as caught:
            replay_stored_probability_map(
                self._segmenter(0.4), self.segmentation, threshold=0.4
            )
        assert str(caught.exception) == LEGACY_MAP_MESSAGE
        assert not SegmentObject.objects.filter(
            segmentation=self.segmentation
        ).exists()
        # The bytes and the row survive for a human to look at; refusing to
        # read a file is not a reason to destroy it.
        assert self._map_path().exists()
        assert ProbabilityMap.objects.filter(segmentation=self.segmentation).exists()

    def test_each_marker_is_load_bearing_on_its_own(self):
        """A file carrying four of the five did not come from the writer either."""
        original = dict(self._record.metadata)
        for marker in MARKERS:
            with self.subTest(missing=marker):
                record = self._record
                metadata = dict(original)
                metadata.pop(marker)
                record.metadata = metadata
                record.save(update_fields=["metadata"])
                assert replay_provenance_problem(metadata) is not None
                with self.assertRaises(StoredMapUnavailable):
                    self._load()

    def test_a_map_on_the_models_own_grid_is_refused(self):
        """The pre-R11 ordering, recorded honestly, is still not replayable."""
        record = self._record
        metadata = dict(record.metadata)
        metadata["native_coordinates"] = False
        record.metadata = metadata
        record.save(update_fields=["metadata"])
        with self.assertRaises(StoredMapUnavailable):
            self._load()

    def test_an_unknown_quantisation_rule_is_refused_not_assumed(self):
        record = self._record
        metadata = dict(record.metadata)
        metadata["quantization"] = "uint8"  # the column default: rule unrecorded
        record.metadata = metadata
        record.save(update_fields=["metadata"])
        problem = replay_provenance_problem(metadata)
        assert problem is not None and "uint8" in problem
        with self.assertRaises(StoredMapUnavailable):
            self._load()

    def test_a_decision_taken_somewhere_else_is_refused(self):
        record = self._record
        metadata = dict(record.metadata)
        assert metadata["thresholded_on"] == THRESHOLDED_ON_STORED_MAP
        metadata["thresholded_on"] = "model_grid_float"
        record.metadata = metadata
        record.save(update_fields=["metadata"])
        with self.assertRaises(StoredMapUnavailable):
            self._load()

    def test_a_file_with_no_row_behind_it_is_refused(self):
        """Bytes whose conventions nobody wrote down are the same problem."""
        ProbabilityMap.objects.filter(segmentation=self.segmentation).delete()
        assert self._map_path().exists()
        with self.assertRaises(StoredMapUnavailable):
            self._load()

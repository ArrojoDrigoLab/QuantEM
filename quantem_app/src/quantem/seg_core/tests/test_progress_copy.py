"""What a run says about itself while it runs.

Every string this module checks is one a user reads: they land in
``Job.message``, which the Tasks drawer renders, and in the segmentation's
status. Two rules they have to obey.

**No internal model codename.** A segmenter calls its deep-learning output
``"DINO"`` -- the foundation encoder's name -- and that name used to be
concatenated straight into the message, so ``DINO: 57% (Tile 32/56)`` was on
screen in the Tasks drawer during every run.

**One divisor.** The percentage and the count in a progress sentence are the
same fraction of the same tiling plan. The run's coarse ``progress`` covers the
phases either side of the tiles as well and therefore divides by more; it is
not the number that belongs beside "32 of 56 tiles", and the test below pins
the arithmetic that keeps them from being confused.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import SimpleTestCase

from quantem.seg_core.db.inference import run_inference_for_segmentation
from quantem.seg_core.types import InferenceResult

TILES = 56
#: The name the segmenter uses for its own output, and the one the product
#: must never repeat. Written as a literal here on purpose: this test is the
#: thing that fails if somebody pipes ``get_dl_model_names()`` back into copy.
DL_MODEL_NAME = "DINO"


def _segmentation() -> SimpleNamespace:
    return SimpleNamespace(id="seg-1", asset_id="asset-1", asset=object())


def _openable(height: int = 2508, width: int = 2892) -> SimpleNamespace:
    return SimpleNamespace(height=height, width=width)


class _TiledSegmenter:
    """A segmenter that walks ``TILES`` windows and reports each one.

    ``on_progress`` fires with ``done / total`` exactly as
    :func:`quantem.inference.tiling.run_windows` does, so the integers this test
    sees are the integers a real run produces.
    """

    name = "mito"
    generated_flag = "mito_generated"
    prob_map_prefix = "mito"
    persist_probability_maps = False

    def __init__(self, tiles: int = TILES):
        self.tiles = tiles
        self.load_models_called = False

    def load_models(self) -> None:
        self.load_models_called = True

    def get_dl_model_names(self) -> list[str]:
        return [DL_MODEL_NAME]

    def estimate_dl_tile_count(self, image_shape) -> int:
        _ = image_shape
        return self.tiles

    def predict(self, image, cached_prob_maps=None, on_progress=None, **kwargs):
        _ = (cached_prob_maps, kwargs)
        if on_progress is not None:
            for done in range(1, self.tiles + 1):
                on_progress(DL_MODEL_NAME, done / self.tiles)
            on_progress("combine", 1.0)
        shape = image.shape[:2] if getattr(image, "ndim", 0) == 2 else (16, 16)
        prob = np.full(shape, 0.75, dtype=np.float32)
        return InferenceResult(prob_maps={DL_MODEL_NAME: prob}, prob=prob)

    def get_probability_map_metadata(self, model_name: str) -> dict[str, object]:
        return {"family": "quantem", "model_name": model_name}


class _Recorder:
    def __init__(self):
        self.status: list[tuple[str, float, str | None]] = []
        self.detail: list[str] = []

    def on_status(self, stage, progress, message=None):
        self.status.append((stage, progress, message))

    def on_detail(self, message):
        self.detail.append(message)

    @property
    def messages(self) -> list[str]:
        return [m for _s, _p, m in self.status if m] + list(self.detail)


class ProgressCopyTests(SimpleTestCase):
    def _run(self, *, roi=None, tiles: int = TILES) -> _Recorder:
        recorder = _Recorder()
        with (
            patch(
                "quantem.seg_core.db.inference.get_asset_openable",
                return_value=_openable(),
            ),
            patch(
                "quantem.seg_core.db.inference.load_image_array",
                return_value=(np.zeros((64, 64), dtype=np.uint8), 0.0),
            ),
            patch(
                "quantem.seg_core.db.inference.load_image_roi_array",
                return_value=np.zeros((64, 64), dtype=np.uint8),
            ),
        ):
            run_inference_for_segmentation(
                _TiledSegmenter(tiles),
                _segmentation(),
                MagicMock(),
                roi,
                on_status=recorder.on_status,
                on_detail=recorder.on_detail,
            )
        return recorder

    def test_no_message_names_the_model_architecture(self):
        recorder = self._run()
        self.assertTrue(recorder.messages, "the run reported nothing at all")
        offenders = [m for m in recorder.messages if DL_MODEL_NAME.lower() in m.lower()]
        self.assertEqual(
            offenders,
            [],
            f"an internal model codename reached a user-visible string: {offenders}",
        )

    def test_the_tile_count_reads_as_a_sentence_not_a_ratio(self):
        recorder = self._run()
        self.assertTrue(
            any("of 56 tiles" in m for m in recorder.messages),
            f"no tile sentence in {recorder.messages}",
        )
        self.assertFalse(
            any("Tile " in m and "/" in m for m in recorder.messages),
            "the machine-log tile ratio is still being written",
        )

    def test_the_percentage_and_the_count_share_the_tiling_plans_divisor(self):
        """57% next to "32 of 56 tiles", never 57% next to 32 of something else."""
        recorder = self._run()
        checked = 0
        for message in recorder.messages:
            if " of 56 tiles" not in message:
                continue
            percent_text, rest = message.split("%", 1)
            percent = int(percent_text.rsplit(" ", 1)[-1])
            done = int(rest.split("(", 1)[1].split(" of ", 1)[0])
            self.assertEqual(
                percent,
                round(100.0 * done / 56),
                f"the percentage and the count disagree in {message!r}",
            )
            checked += 1
        self.assertGreater(checked, 0, "no tile sentence was produced")

    def test_a_patch_scoped_run_reports_its_tiles_too(self):
        """The tile text used to be gated on the run being whole-image.

        A patch run walks real windows and takes real minutes; there was no
        reason for it to be the silent one.
        """
        roi = SimpleNamespace(id="roi-1", x=10, y=20, width=1024, height=1024)
        recorder = self._run(roi=roi)
        self.assertTrue(
            any("of 56 tiles" in m for m in recorder.messages),
            f"a patch run said nothing about tiles: {recorder.messages}",
        )
        self.assertTrue(
            any("patch" in m for m in recorder.detail),
            "the planning line should name the patch it is covering",
        )

    def test_one_tile_is_a_tile_and_not_a_tiles(self):
        recorder = self._run(tiles=1)
        self.assertTrue(
            any("1 of 1 tile" in m for m in recorder.messages),
            f"bad pluralisation in {recorder.messages}",
        )
        self.assertFalse(any("1 of 1 tiles" in m for m in recorder.messages))

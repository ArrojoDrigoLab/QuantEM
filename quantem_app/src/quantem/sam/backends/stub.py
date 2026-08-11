"""A backend with no weights, no torch and no GPU.

Turned on with ``QUANTEM_SAM_STUB=1``. The whole endpoint -- crop planning,
the embedding cache, the coordinate round trip, object creation -- can then be
exercised offline in CI, which is the only way that plumbing gets tested at
all: a suite that needs 375 MB of weights and a GPU is a suite that does not
run.

It is not a mock. It encodes and decodes for real, just cheaply: the
"embedding" is a downsampled copy of the crop, and the "mask" is the
brighter-than-local-median pixels inside the box. So a test can assert on a
mask whose shape it controls by choosing the pixels.
"""

from __future__ import annotations

import numpy as np

from quantem.sam.backends.base import Embedding, MaskCandidate

#: Matches the real backend's stride, so cached-entry sizes are representative.
_STRIDE = 16


class StubBackend:
    """Deterministic, dependency-free stand-in for a SAM runtime."""

    identity = "stub:threshold"
    device = "cpu"

    def __init__(self) -> None:
        #: Counts encodes so a test can prove the cache prevented one.
        self.encode_calls = 0

    def encode(self, image_rgb: np.ndarray) -> Embedding:
        self.encode_calls += 1
        grey = np.asarray(image_rgb, dtype=np.float32)
        if grey.ndim == 3:
            grey = grey.mean(axis=-1)
        height, width = grey.shape
        # A real embedding is a coarse grid over the image; so is this.
        coarse = grey[::_STRIDE, ::_STRIDE].astype(np.float32)
        return Embedding(
            features=coarse[None, None, :, :],
            original_size=(int(height), int(width)),
            input_size=(int(height), int(width)),
        )

    def predict(
        self,
        embedding: Embedding,
        box_xyxy: tuple[float, float, float, float],
    ) -> list[MaskCandidate]:
        height, width = embedding.original_size
        coarse = embedding.features[0, 0]
        # Back to crop resolution, nearest-neighbour, then cropped to size --
        # ``repeat`` overshoots when the crop is not a multiple of the stride.
        grey = np.repeat(np.repeat(coarse, _STRIDE, axis=0), _STRIDE, axis=1)
        grey = grey[:height, :width]
        if grey.shape != (height, width):
            padded = np.zeros((height, width), dtype=np.float32)
            padded[: grey.shape[0], : grey.shape[1]] = grey
            grey = padded

        x0, y0, x1, y1 = (int(round(value)) for value in box_xyxy)
        x0 = max(0, min(x0, width))
        y0 = max(0, min(y0, height))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(y0 + 1, min(y1, height))

        inside = grey[y0:y1, x0:x1]
        threshold = float(np.median(inside)) if inside.size else 0.0

        best = np.zeros((height, width), dtype=bool)
        best[y0:y1, x0:x1] = inside > threshold

        # A slightly tighter and a slightly looser alternative, so the response's
        # "other candidates" block is exercised too.
        tight = np.zeros_like(best)
        tight[y0:y1, x0:x1] = inside > (threshold + 0.5 * (float(inside.max()) - threshold))
        loose = np.zeros_like(best)
        loose[y0:y1, x0:x1] = True

        return [
            MaskCandidate(mask=best, score=0.90),
            MaskCandidate(mask=tight, score=0.60),
            MaskCandidate(mask=loose, score=0.30),
        ]

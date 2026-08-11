"""What a SAM runtime has to provide.

The split into ``encode`` and ``predict`` is not incidental -- it is the reason
this feature is fast enough to be synchronous. ``encode`` is the image encoder
and costs hundreds of milliseconds; ``predict`` is the mask decoder and costs
tens. Caching sits between them, so an :class:`Embedding` has to be a plain
value a cache can hold and hand back later, not a handle into a predictor's
mutable internal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Embedding:
    """One encoded crop.

    ``features`` is kept as a numpy array rather than a torch tensor so the
    cache holds no device memory: an entry that pinned VRAM would make the cap
    in :data:`~quantem.sam.config.EMBEDDING_CACHE_ENTRIES` a VRAM budget as well
    as a RAM one, and eviction would then have to be timely rather than merely
    eventual.
    """

    features: np.ndarray
    #: ``(height, width)`` of the crop that was encoded.
    original_size: tuple[int, int]
    #: ``(height, width)`` of the resized tensor the encoder actually ran on.
    input_size: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return int(self.features.nbytes)


@dataclass(frozen=True)
class MaskCandidate:
    """One mask the decoder proposed for a box, with its own predicted quality."""

    mask: np.ndarray
    score: float


@runtime_checkable
class SamBackend(Protocol):
    """A runtime that can encode a crop and decode a box against it."""

    #: Mixed into the embedding cache key, so changing weights cannot serve an
    #: embedding the previous weights produced.
    identity: str

    #: ``"cuda"``, ``"mps"`` or ``"cpu"`` -- reported to the client so a slow
    #: first prompt is explainable.
    device: str

    def encode(self, image_rgb: np.ndarray) -> Embedding:
        """Run the image encoder over an ``(H, W, 3)`` uint8 crop."""
        ...

    def predict(
        self,
        embedding: Embedding,
        box_xyxy: tuple[float, float, float, float],
    ) -> list[MaskCandidate]:
        """Decode ``box_xyxy`` against a cached embedding, best score first.

        The box is crop-relative and in ``xyxy`` order.
        """
        ...

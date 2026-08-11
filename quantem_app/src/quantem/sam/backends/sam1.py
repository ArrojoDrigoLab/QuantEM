"""The shipped runtime: Meta's Segment Anything carrying micro-SAM's EM weights.

See :mod:`quantem.sam.config` for why this combination rather than the
``micro_sam`` package (321 MB of wheels, including napari and PyQt6) or stock
Meta weights (not EM-trained).

The Segment Anything code is vendored at
:mod:`quantem.sam._vendor.segment_anything`, so nothing new is installed: it
needs only torch, numpy and torchvision, which the app already has.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from quantem.sam.backends.base import Embedding, MaskCandidate
from quantem.sam.checkpoint import CheckpointMissing, checkpoint_path
from quantem.sam.config import CHECKPOINT

logger = logging.getLogger(__name__)

#: Inference runs one at a time, process-wide.
#:
#: Django's threaded server will happily call this from several request threads
#: at once, and a single predictor holding one set of ``_features`` is not
#: reentrant -- concurrent prompts interleave and return each other's masks.
#: A GPU also does not go faster for being asked twice at once. The lock is held
#: across the encode, which is the slow part, so a second prompt during a first
#: encode waits rather than starting a competing one.
_INFERENCE_LOCK = threading.Lock()


def _load_state_dict(path: Any) -> OrderedDict[str, Any]:
    """The SAM state dict inside a checkpoint file, whatever wrapper it wears.

    micro-SAM's weights are saved by torch-em, which nests the model under
    ``model_state`` and prefixes every key with ``sam.``. Stock Meta checkpoints
    are the bare state dict. Both are accepted so switching ``CHECKPOINT``
    between them needs no code change.
    """
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        inner = state["model_state"]
        return OrderedDict(
            (key[len("sam.") :] if key.startswith("sam.") else key, value)
            for key, value in inner.items()
        )
    return OrderedDict(state)


class Sam1Backend:
    """A SAM-1 style predictor: ``vit_b`` encoder, box-prompted mask decoder."""

    identity = CHECKPOINT.identity

    def __init__(self) -> None:
        import torch

        from quantem.inference.device import select_device

        try:
            from quantem.sam._vendor.segment_anything import sam_model_registry
            from quantem.sam._vendor.segment_anything.predictor import SamPredictor
        except ImportError as exc:  # pragma: no cover - depends on the install
            # Reachable if torchvision is missing: the vendored transforms
            # import it, and it is the one part of the runtime QuantEM does not
            # carry itself.
            raise CheckpointMissing(
                "Box prompting cannot start because part of the image-processing "
                "toolkit it needs is missing from this installation."
            ) from exc

        path = checkpoint_path()
        if not path.is_file():
            raise CheckpointMissing(
                f"{CHECKPOINT.display_name} has not been downloaded yet."
            )

        self.device = select_device()
        model = sam_model_registry[CHECKPOINT.architecture]()
        missing, unexpected = model.load_state_dict(_load_state_dict(path), strict=False)
        if missing or unexpected:
            # Not fatal -- a partially matching checkpoint still runs -- but it
            # means the weights are not the architecture we think they are, and
            # the masks will be quietly poor rather than absent.
            logger.warning(
                "%s loaded with %d missing and %d unexpected weights",
                CHECKPOINT.identity,
                len(missing),
                len(unexpected),
            )
        model.to(self.device)
        model.eval()
        self._predictor = SamPredictor(model)
        self._torch = torch
        logger.info("SAM backend %s ready on %s", self.identity, self.device)

    def encode(self, image_rgb: np.ndarray) -> Embedding:
        """Encode a crop.

        No resizing, normalisation, padding or contrast stretch happens here on
        purpose. The crop arrives as stored 8-bit greyscale replicated to RGB,
        and SAM's own preprocessing (its 1024-px long-side resize and its
        pixel-mean/std) is left to the library. Anything else added in front of
        it would be a second, undocumented normalisation.
        """
        with _INFERENCE_LOCK:
            self._predictor.set_image(image_rgb)
            features = self._predictor.features.detach().to("cpu").numpy()
            return Embedding(
                features=features,
                original_size=tuple(self._predictor.original_size),
                input_size=tuple(self._predictor.input_size),
            )

    def predict(
        self,
        embedding: Embedding,
        box_xyxy: tuple[float, float, float, float],
    ) -> list[MaskCandidate]:
        """Decode a box against a cached embedding.

        The three assignments after ``reset_image`` are the load-bearing part:
        they put a previously computed embedding back into the predictor so the
        encoder does not run again. ``is_image_set`` is what ``predict`` checks
        before it will do anything.

        ``xyxy``, not ``yxyx``. The Meta predictor and micro-SAM's own wrapper
        disagree about this and we are on the Meta side; getting it wrong
        transposes every mask.
        """
        torch = self._torch
        with _INFERENCE_LOCK:
            predictor = self._predictor
            predictor.reset_image()
            predictor.features = torch.as_tensor(embedding.features, device=self.device)
            predictor.original_size = tuple(embedding.original_size)
            predictor.input_size = tuple(embedding.input_size)
            predictor.is_image_set = True

            box = np.asarray(box_xyxy, dtype=np.float32)[None, :]
            masks, scores, _logits = predictor.predict(
                box=box,
                multimask_output=True,
            )

        candidates = [
            MaskCandidate(mask=np.asarray(mask), score=float(score))
            for mask, score in zip(masks, scores, strict=False)
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates

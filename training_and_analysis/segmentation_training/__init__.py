"""Organelle segmentation on a frozen or adapted EM foundation encoder.

A four-level pipeline, each level swappable from the configuration:

    encoder (frozen or adapted) -> neck -> decoder -> loss

Encoders are loaded through the same checkpoint index that pretraining writes, so a run here and
a run in foundation_training see the same weights and the same normalisation.
"""

from __future__ import annotations

# constants is standard-library only, so importing this package stays cheap.
from .constants import BACKGROUND, FOREGROUND, IGNORE_INDEX  # noqa: F401

__all__ = ["BACKGROUND", "FOREGROUND", "IGNORE_INDEX"]

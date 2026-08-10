"""EM-appropriate, single-channel-safe augmentation transforms.

These modules deliberately do not import dinov3, so they are unit-testable on CPU.
"""

from .primitives import (  # noqa: F401
    MaybeApply,
    RandomBrightnessContrast,
    RandomDihedral,
    RandomGamma,
    RandomGaussianBlur,
    RandomGaussianNoise,
    to_float_chw,
)
from .em_dino_augmentations import EMDataAugmentationDINO, EMAugmentationConfig  # noqa: F401

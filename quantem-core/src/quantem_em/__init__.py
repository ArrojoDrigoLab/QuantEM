"""QuantEM — inference and lightweight adaptation for EM organelle segmentation models.

Importing this package must stay cheap: napari imports plugin top-levels during manifest
discovery, so ``torch`` is loaded only when a model is actually built.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from .registry import (  # noqa: F401  (light: dataclasses only, no torch)
    DEFAULT_MODEL_FOR_ORGANELLE,
    ORGANELLE_LABELS,
    get_model_spec,
    list_models,
)

__all__ = [
    "__version__",
    "list_models",
    "get_model_spec",
    "DEFAULT_MODEL_FOR_ORGANELLE",
    "ORGANELLE_LABELS",
]

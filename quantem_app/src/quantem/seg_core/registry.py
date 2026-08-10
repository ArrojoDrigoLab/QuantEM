"""
Segmenter Registry
===================

Simple registry mapping type names to segmenter classes.

Registrations are import-path strings, so a segmenter module (and therefore
torch) is imported only when a segmenter of that type is actually instantiated.
:func:`register_default_segmenters` is called from ``AppConfig.ready()`` and
installs the eight released QuantEM/OmniEM models -- four organelles x two
families, routed to one segmenter class per organelle. The family is chosen per
run from the ``source_model`` kwarg (``"quantem:mito"`` / ``"omniem:mito"``).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_segmenter import BaseSegmenter

SegmenterRegistration = type | str

_registry: dict[str, SegmenterRegistration] = {}

#: The only segmenters QuantEM ships. Keyed by the segmenter internal name that
#: ``quantem.segmentation.source_models.resolve_segmenter_internal_name``
#: returns for a released source model.
DEFAULT_SEGMENTERS: dict[str, str] = {
    "dino_mito": "quantem.inference.segmenter.DinoMitoSegmenter",
    "dino_er": "quantem.inference.segmenter.DinoErSegmenter",
    "dino_ld": "quantem.inference.segmenter.DinoLdSegmenter",
    "dino_nucleus": "quantem.inference.segmenter.DinoNucleusSegmenter",
}

_defaults_installed = False


def register_segmenter(
    type_name: str,
    segmenter_class: SegmenterRegistration,
) -> None:
    """Register a segmenter class or import path for a given type name."""
    _registry[type_name.lower()] = segmenter_class


def register_default_segmenters() -> None:
    """Install the built-in QuantEM/OmniEM registrations. Idempotent.

    Call site: ``SegmentationConfig.ready()``. Also invoked lazily on a registry
    miss so non-Django entry points (CLI, unit tests) resolve segmenters too.
    """
    global _defaults_installed
    for type_name, import_path in DEFAULT_SEGMENTERS.items():
        _registry.setdefault(type_name, import_path)
    _defaults_installed = True


def _resolve_segmenter_class(
    registration: SegmenterRegistration,
) -> type[BaseSegmenter]:
    if isinstance(registration, str):
        module_name, _, attr_name = registration.rpartition(".")
        if not module_name or not attr_name:
            raise ImportError(f"Invalid segmenter import path: {registration!r}")
        module = import_module(module_name)
        resolved = getattr(module, attr_name)
    else:
        resolved = registration

    if not isinstance(resolved, type):
        raise TypeError(f"Segmenter registration is not a class: {resolved!r}")
    return resolved


def get_segmenter(
    segmentation_type_internal_name: str,
    **segmenter_kwargs,
) -> BaseSegmenter:
    """Look up and instantiate a segmenter by canonical internal name.

    Raises:
        ValueError: If no segmenter is registered for the given type name.
    """
    name = segmentation_type_internal_name.strip().lower()
    registration = _registry.get(name)
    if registration is None and not _defaults_installed:
        register_default_segmenters()
        registration = _registry.get(name)
    if registration is None:
        raise ValueError(f"No segmenter registered for type: {name}")
    segmenter_class = _resolve_segmenter_class(registration)
    _registry[name] = segmenter_class
    return segmenter_class(**segmenter_kwargs)


def get_segmenter_or_none(
    segmentation_type_internal_name: str,
    **segmenter_kwargs,
) -> BaseSegmenter | None:
    """Look up segmenter by type name, returning None if not found."""
    try:
        return get_segmenter(
            segmentation_type_internal_name,
            **segmenter_kwargs,
        )
    except ValueError:
        return None

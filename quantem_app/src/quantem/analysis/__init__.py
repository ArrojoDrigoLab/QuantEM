"""Quantitative analysis: morphometrics, composition, compartments, spatial statistics.

The algorithms are ports of the validated Figure-4 pipeline
(``gk_gold_seg/scripts/gold_pipeline/``), generalised away from immunogold: a
point set is any ``(x, y)`` array and a compartment is any binary mask. Two
reproducibility defects in that reference are corrected here and documented at
their call sites -- per-(image, replicate) Monte-Carlo seeding, and a single
distance implementation shared by observed and simulated points.

Everything re-exported below is pure numpy and importable without Django. It is
resolved lazily so Django can register the app without loading SciPy,
scikit-image, or the image-processing pipeline during every normal startup.
The app layer is deliberately *not*
re-exported, so this package stays usable from a notebook: :mod:`.models` (the ``AnalysisRun`` row), :mod:`.loaders` (database ->
masks and point arrays), :mod:`.service` (``run_for_segmentation``, which writes
the export bundle), :mod:`.job` (the queue entry point), :mod:`.serializers` and
:mod:`.urls`. The per-run views are in ``quantem.segmentation.api_views.analysis``
alongside every other ``/api/segmentations/<id>/...`` endpoint; the group rollup,
which is about a set of runs rather than a segmentation, is in :mod:`.views`.
"""

from importlib import import_module

from .rollup import Aggregate, aggregate, rollup, weighted_mean_for_comparison

_LAZY_EXPORTS = {
    "AreaFractions": "compartments",
    "CompartmentSet": "compartments",
    "PointAssignment": "compartments",
    "area_fractions": "compartments",
    "assign_points": "compartments",
    "DEFAULT_BAND_EDGES_NM": "distances",
    "DistanceResult": "distances",
    "band_labels": "distances",
    "boundary_pixels": "distances",
    "contact_fraction": "distances",
    "distance_to_boundary": "distances",
    "nearest_neighbour_nm": "distances",
    "DEFAULT_REPLICATES": "montecarlo",
    "DEFAULT_SEED": "montecarlo",
    "NullResult": "montecarlo",
    "csr_null": "montecarlo",
    "sample_uniform_in_mask": "montecarlo",
    "self_check": "montecarlo",
    "ObjectMetrics": "morphometrics",
    "count_by_source": "morphometrics",
    "density": "morphometrics",
    "derive": "morphometrics",
    "summarize": "morphometrics",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_BAND_EDGES_NM",
    "DEFAULT_REPLICATES",
    "DEFAULT_SEED",
    "Aggregate",
    "AreaFractions",
    "CompartmentSet",
    "DistanceResult",
    "NullResult",
    "ObjectMetrics",
    "PointAssignment",
    "aggregate",
    "area_fractions",
    "assign_points",
    "band_labels",
    "boundary_pixels",
    "contact_fraction",
    "count_by_source",
    "csr_null",
    "density",
    "derive",
    "distance_to_boundary",
    "nearest_neighbour_nm",
    "rollup",
    "sample_uniform_in_mask",
    "self_check",
    "summarize",
    "weighted_mean_for_comparison",
]

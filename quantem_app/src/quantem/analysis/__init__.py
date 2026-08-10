"""Quantitative analysis: morphometrics, composition, compartments, spatial statistics.

The algorithms are ports of the validated Figure-4 pipeline
(``gk_gold_seg/scripts/gold_pipeline/``), generalised away from immunogold: a
point set is any ``(x, y)`` array and a compartment is any binary mask. Two
reproducibility defects in that reference are corrected here and documented at
their call sites -- per-(image, replicate) Monte-Carlo seeding, and a single
distance implementation shared by observed and simulated points.

Everything re-exported below is pure numpy and importable without Django. The
app layer is deliberately *not* re-exported, so this package stays usable from a
notebook: :mod:`.models` (the ``AnalysisRun`` row), :mod:`.loaders` (database ->
masks and point arrays), :mod:`.service` (``run_for_segmentation``, which writes
the export bundle), :mod:`.job` (the queue entry point), :mod:`.serializers` and
:mod:`.urls`. The per-run views are in ``quantem.segmentation.api_views.analysis``
alongside every other ``/api/segmentations/<id>/...`` endpoint; the group rollup,
which is about a set of runs rather than a segmentation, is in :mod:`.views`.
"""

from .compartments import (
    AreaFractions,
    CompartmentSet,
    PointAssignment,
    area_fractions,
    assign_points,
)
from .distances import (
    DEFAULT_BAND_EDGES_NM,
    DistanceResult,
    band_labels,
    boundary_pixels,
    contact_fraction,
    distance_to_boundary,
    nearest_neighbour_nm,
)
from .montecarlo import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    NullResult,
    csr_null,
    sample_uniform_in_mask,
    self_check,
)
from .morphometrics import (
    ObjectMetrics,
    count_by_source,
    density,
    derive,
    summarize,
)
from .rollup import Aggregate, aggregate, rollup, weighted_mean_for_comparison

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

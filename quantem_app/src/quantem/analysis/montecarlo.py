"""Monte-Carlo null: complete spatial randomness within the tissue mask.

For each image, scatter ``N`` = the observed on-tissue point count uniformly at
random inside the tissue mask, ``replicates`` times, and push each draw through
**the identical** assignment and distance code the observed points went through.
The null then answers "what would this enrichment be if position carried no
information, given this tissue's geometry and this many points?"

Reported as a z-score against the replicate distribution, which is the Figure-4D
axis, plus an empirical two-sided p.

Ported from ``gk_gold_seg/scripts/gold_pipeline/mc.py`` with two deliberate
corrections:

1. **Seeding is per (image, replicate), not global.** The reference creates one
   module-level ``np.random.default_rng(SEED)`` at ``mc.py:87`` and draws from it
   in a loop over a dict, so its published numbers depend on dictionary
   iteration order and on how many images were processed before this one. Here
   each draw's seed is derived from ``(seed, image_key, replicate_index)``, so a
   single image reproduces identically whether analysed alone or in a batch.
2. **One distance implementation.** The reference used exact KD-tree distances
   for observed points and a 96 nm binary dilation for the null. Both sides here
   call :func:`quantem.analysis.distances.distance_to_boundary`.

Consequence, and it must be stated wherever these numbers are shown: **this will
not reproduce the published Figure-4D z-scores exactly.** It is the same method,
correctly seeded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .compartments import AreaFractions, CompartmentSet, area_fractions, assign_points

#: Defaults chosen to match the manuscript ("Twenty independent replicates were
#: generated per image from a fixed random seed").
DEFAULT_REPLICATES = 20
DEFAULT_SEED = 12345

#: Expected points in the smallest compartment when running :func:`self_check`.
#: At 2,000 the relative standard error of a compartment count is ~2%.
TARGET_PER_COMPARTMENT = 2_000

#: Floor and ceiling on the self-check draw. The floor keeps a mask made almost
#: entirely of one compartment from being "checked" on a handful of points. The
#: ceiling is a cost bound: the check runs on *every* analysis, and sizing off
#: the smallest compartment alone let a 1 %-area compartment ask for 200,000
#: draws each time. See :func:`self_check` for the third, principled bound.
SELF_CHECK_MIN_POINTS = 2_000
SELF_CHECK_MAX_POINTS = 50_000


def _derive_seed(seed: int, image_key: str, replicate: int) -> int:
    """A stable per-draw seed, independent of processing order."""
    payload = f"{seed}|{image_key}|{replicate}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def sample_uniform_in_mask(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` uniformly random integer ``(x, y)`` positions inside ``mask``.

    Rejection sampling, batched by the mask's fill fraction so the expected
    number of rounds is small even for sparse tissue.
    """
    m = mask.astype(bool)
    h, w = m.shape
    area = int(m.sum())
    if n <= 0:
        return np.empty((0, 2), dtype=int)
    if area == 0:
        raise ValueError("cannot sample inside an empty mask")

    frac = area / (h * w)
    xs = np.empty(0, dtype=int)
    ys = np.empty(0, dtype=int)
    while xs.size < n:
        batch = int((n - xs.size) / max(frac, 0.05) * 1.3) + 16
        cx = rng.integers(0, w, batch)
        cy = rng.integers(0, h, batch)
        keep = m[cy, cx]
        xs = np.concatenate([xs, cx[keep]])
        ys = np.concatenate([ys, cy[keep]])
    return np.column_stack([xs[:n], ys[:n]])


@dataclass(frozen=True)
class NullResult:
    """Observed value against its null distribution, per metric."""

    observed: dict[str, float]
    #: ``None`` where the null produced no draws for that metric at all.
    null_mean: dict[str, float | None]
    #: ``None`` where fewer than two replicates ran, so there is no sample SD.
    #: Zero here is a real measurement -- every replicate landed on the same
    #: value -- and both :attr:`z` and :attr:`p_two_sided` are ``None`` for it.
    null_sd: dict[str, float | None]
    z: dict[str, float | None]
    #: ``None`` wherever :attr:`z` is: an empirical p against a null with no
    #: spread is 1/(R+1) or 1.0 by construction. See :func:`_p_two_sided`.
    p_two_sided: dict[str, float | None]
    replicates: int
    seed: int
    #: Per-replicate values, kept so the UI can draw the null distribution and
    #: so a reviewer can recompute anything from the export.
    null_samples: dict[str, list[float]] = field(default_factory=dict)


def _null_sd(samples: np.ndarray) -> float | None:
    """Sample SD of the null draws, or ``None`` when there is no such thing.

    One replicate has no spread to measure. Reporting ``0.0`` for it is a
    measurement of zero variance, which is a strong claim about the geometry and
    was never made.
    """
    if samples.size < 2:
        return None
    return float(samples.std(ddof=1))


def _z(obs: float, samples: np.ndarray) -> float | None:
    sd = _null_sd(samples)
    if not sd:
        return None
    return (obs - float(samples.mean())) / sd


def _p_two_sided(obs: float, samples: np.ndarray) -> float | None:
    """Empirical two-sided p with the +1 correction, so p is never 0.

    **Guarded exactly as :func:`_z` is, and for the same reason.** When every
    replicate returns the same number the null has no distribution to be
    extreme against, and this formula still returns something: ``1 / (R + 1)``
    if the observed value differs from that number at all, ``1.0`` if it does
    not. At the default twenty replicates the first is **0.0476**, the smallest
    value the method can produce, which is read as significance at 0.05 -- and
    it arrives whatever the data are, including from a single point, which
    cannot exhibit spatial structure of any kind. Measured: one point inside a
    compartment covering 1.5 % of the tissue, twenty simulated singletons that
    all missed it, ``null_sd = 0.0``, ``z = None``, ``p = 0.0476``.

    A null with no spread supports no p-value. The caller says so in words;
    here it is ``None``.
    """
    if samples.size == 0 or not _null_sd(samples):
        return None
    centre = float(samples.mean())
    extreme = int((np.abs(samples - centre) >= abs(obs - centre)).sum())
    return (extreme + 1) / (samples.size + 1)


def csr_null(
    points_xy: np.ndarray,
    comp: CompartmentSet,
    *,
    image_key: str,
    metric: Callable[[np.ndarray], dict[str, float]] | None = None,
    areas: AreaFractions | None = None,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> NullResult:
    """Compare observed point statistics against complete spatial randomness.

    ``metric`` maps a point array to a dict of scalars. It defaults to
    per-compartment enrichment, which is what Figure 4 reports; pass your own to
    test any statistic — a distance-band fraction, a nearest-neighbour median —
    under the same null.

    ``image_key`` makes the draw reproducible; use the asset id.
    """
    areas = areas or area_fractions(comp)

    if metric is None:

        def metric(pts: np.ndarray) -> dict[str, float]:
            a = assign_points(pts, comp, areas=areas)
            return {f"enrichment_{k}": v for k, v in a.enrichment.items() if v is not None}

    observed_assignment = assign_points(points_xy, comp, areas=areas)
    n = observed_assignment.n_on_tissue

    # Both sides of the comparison must see the same population. The null can
    # only scatter points *inside* the tissue, so the observed side is
    # restricted to its on-tissue points and trimmed to (x, y) before ``metric``
    # sees it. The default enrichment metric normalises this away internally, so
    # this changes none of its numbers -- but ``metric`` is a documented
    # extension point, and a custom one was being handed ``n_total`` observed
    # points against ``n_on_tissue`` simulated ones. Any statistic that is not
    # already a ratio (a band fraction, a nearest-neighbour median, a raw count)
    # would then be compared against a null of a different size.
    pts = np.asarray(points_xy, dtype=float)
    observed = metric(pts[observed_assignment.on_tissue][:, :2])

    tissue = comp.tissue_mask()
    samples: dict[str, list[float]] = {k: [] for k in observed}
    for r in range(replicates):
        if cancel_check is not None:
            cancel_check()
        rng = np.random.default_rng(_derive_seed(seed, image_key, r))
        drawn = sample_uniform_in_mask(tissue, n, rng)
        for key, value in metric(drawn.astype(float)).items():
            samples.setdefault(key, []).append(float(value))
        if on_progress is not None:
            on_progress(r + 1, replicates)

    null_mean: dict[str, float | None] = {}
    null_sd: dict[str, float | None] = {}
    zs: dict[str, float | None] = {}
    ps: dict[str, float | None] = {}
    for key, obs in observed.items():
        arr = np.asarray(samples.get(key, []), dtype=float)
        # ``nan`` for a mean with no draws behind it, which is what this was, is
        # not a number the rest of the app can carry: it is not JSON, so it
        # cannot be stored, and every table renders it as a value.
        null_mean[key] = float(arr.mean()) if arr.size else None
        null_sd[key] = _null_sd(arr)
        zs[key] = _z(float(obs), arr)
        ps[key] = _p_two_sided(float(obs), arr)

    return NullResult(
        observed={k: float(v) for k, v in observed.items()},
        null_mean=null_mean,
        null_sd=null_sd,
        z=zs,
        p_two_sided=ps,
        replicates=replicates,
        seed=seed,
        null_samples=samples,
    )


def self_check(
    comp: CompartmentSet,
    *,
    image_key: str = "selfcheck",
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Randomised input must give enrichment ~1.0 in every compartment.

    This is the manuscript's stated internal control ("compartment enrichment on
    randomized data recovered values of ~1.0 in all groups"). It is exposed as a
    callable, not just a test, so the app can run it on the user's own masks and
    show that the normalisation is unbiased for *their* geometry.

    Returns ``skipped_reason`` rather than raising when there is nothing to
    sample: a control that cannot run must not take the analysis down with it.
    """
    if cancel_check is not None:
        cancel_check()
    tissue = comp.tissue_mask()
    tissue_px = int(tissue.sum())
    if tissue_px == 0:
        return {
            "n_points": 0,
            "smallest_compartment_fraction": None,
            "enrichment": {},
            "max_abs_deviation": None,
            "skipped_reason": (
                "The tissue mask is empty, so there is nowhere to scatter a "
                "point and no geometry to check the normalisation against."
            ),
        }

    areas = area_fractions(comp)

    # Draw enough that the *smallest* compartment gets a statistically meaningful
    # count. Sizing off the image instead would let a 2 %-area compartment land
    # ~25 points and fail the check on sampling noise alone, which says nothing
    # about whether the normalisation is biased.
    smallest = min((f for f in areas.fractions.values() if f > 0), default=1.0)
    # ...but never more points than the tissue has pixels. Sampling is uniform
    # over a finite pixel set, so past ``tissue_px`` draws the estimate has
    # already converged on the exact area fraction the check compares against;
    # the extra points buy precision the geometry cannot supply and cost linear
    # time on a control that runs unconditionally.
    n = int(
        min(
            SELF_CHECK_MAX_POINTS,
            tissue_px,
            max(SELF_CHECK_MIN_POINTS, TARGET_PER_COMPARTMENT / smallest),
        )
    )

    rng = np.random.default_rng(_derive_seed(DEFAULT_SEED, image_key, 0))
    pts = sample_uniform_in_mask(tissue, n, rng).astype(float)
    if cancel_check is not None:
        cancel_check()
    a = assign_points(pts, comp)
    return {
        "n_points": n,
        "smallest_compartment_fraction": smallest,
        "enrichment": a.enrichment,
        "max_abs_deviation": max(
            (abs(v - 1.0) for v in a.enrichment.values() if v is not None),
            default=None,
        ),
        "skipped_reason": None,
    }

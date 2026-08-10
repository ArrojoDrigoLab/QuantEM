"""Group-level aggregation — unweighted means over experimental units.

This module is small and its rule is one line, but it is the single easiest
place in the whole analysis suite to produce a wrong published number, so it is
isolated here with a regression test rather than inlined at each call site.

**The rule: roll up as an unweighted mean over units (animals, samples, images),
never weighted by how many points each unit contributed.**

Why it matters, from the reference implementation's own history: the Figure-4
pipeline shipped ``verify_null.py`` for exactly this. Pooling the null by
count-weighting the numerator while area-weighting the denominator produced a
*random* enrichment of **0.73** instead of 1.0 — a 27 % apparent depletion out of
nothing but arithmetic. Per-unit means pin every null at 1.0
(``analyze2.py:78-101`` ``pooled()``; shipped values 1.032 / 1.024 / 0.981).

An image with 4,000 gold particles and an image with 40 are two observations of
the same experiment, not 100:1 evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Aggregate:
    n_units: int
    mean: float | None
    sd: float | None
    sem: float | None
    values: list[float]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_units": self.n_units,
            "mean": self.mean,
            "sd": self.sd,
            "sem": self.sem,
            "values": self.values,
        }


def aggregate(values: Iterable[float | None]) -> Aggregate:
    """Unweighted mean, sample SD (ddof=1) and SEM over the non-null values.

    ``ddof=1`` throughout. The reference was inconsistent — ``mc.py`` used
    ``np.std`` (ddof=0) while ``mc_near100_per_image.py:59`` used ddof=1 — which
    makes its z-scores differ depending on which script produced them.
    """
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return Aggregate(0, None, None, None, [])
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if n == 1:
        return Aggregate(1, mean, None, None, vals)
    sd = float(arr.std(ddof=1))
    return Aggregate(n, mean, sd, sd / np.sqrt(n), vals)


def rollup(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
    metrics: Sequence[str],
) -> dict[str, dict[str, Aggregate]]:
    """Group ``rows`` by ``row[group_key]`` and aggregate each metric.

    Each row is one experimental unit. Do not pass one row per point.
    """
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key, "")), []).append(row)

    out: dict[str, dict[str, Aggregate]] = {}
    for name, members in groups.items():
        out[name] = {
            metric: aggregate(m.get(metric) for m in members)  # type: ignore[arg-type]
            for metric in metrics
        }
    return out


def weighted_mean_for_comparison(
    values: Iterable[float | None], weights: Iterable[float]
) -> float | None:
    """A count-weighted mean — provided **only** so the UI can show the contrast.

    Never use this as the reported group value. It exists so the app can display
    "unweighted 1.00 vs count-weighted 0.73" when a user asks why the two differ,
    which is a more convincing explanation than a footnote.
    """
    vs, ws = [], []
    for v, w in zip(values, weights, strict=False):
        if v is None or not np.isfinite(float(v)):
            continue
        vs.append(float(v))
        ws.append(float(w))
    if not vs or sum(ws) == 0:
        return None
    return float(np.average(vs, weights=ws))

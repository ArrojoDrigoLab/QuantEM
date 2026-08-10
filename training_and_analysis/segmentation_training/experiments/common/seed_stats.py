"""Seed-level statistics and the wash-by-default verdict used to compare one arm against a baseline.

The validation-to-test gap is larger than most of the effects being measured, so a point-estimate
difference readily manufactures a spurious winner. These helpers enforce one rule: a difference must
escape the seed-noise band before it counts as an effect. Inside that band the verdict is 'wash',
the default outcome.
"""

from __future__ import annotations

import math
import statistics as st


def seed_ci(values) -> dict:
    """(mean, sd, ci_halfwidth, n) over seed-level values. CI = t≈2 × SEM for n≥3, else None."""
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0:
        return {"mean": None, "sd": None, "ci": None, "n": 0}
    mean = sum(vals) / n
    sd = st.pstdev(vals) if n >= 2 else None
    ci = (2 * sd / math.sqrt(n)) if (sd is not None and n >= 3) else None
    return {"mean": mean, "sd": sd, "ci": ci, "n": n}


def wash_verdict(delta: float, band: float, *, lower_better: bool = False) -> str:
    """'wash' unless |delta| clears the seed-noise ``band``; then 'help'/'hurt' by sign.

    ``band`` is the seed-noise scale (e.g. the larger of the two arms' across-seed SDs, ×k). 'wash' is
    the default outcome that a delta must escape, not a bucket it falls into by being small."""
    if delta is None or band is None:
        return "unknown"
    if abs(delta) <= band:
        return "wash"
    positive_is_help = not lower_better
    return "help" if (delta > 0) == positive_is_help else "hurt"


def compare_arms(a_vals, b_vals, *, tie_k: float = 1.0, lower_better: bool = False) -> dict:
    """Compare arm A vs baseline B on seed-level values. delta = mean(A) − mean(B); band = tie_k × the
    larger across-seed SD; the verdict is 'wash' unless the delta escapes the band. Returns the full
    record with the underlying numbers, not just the label."""
    A, B = seed_ci(a_vals), seed_ci(b_vals)
    delta = (A["mean"] - B["mean"]) if (A["mean"] is not None and B["mean"] is not None) else None
    sds = [x for x in (A["sd"], B["sd"]) if x is not None]
    band = tie_k * max(sds) if sds else None
    return {"a": A, "b": B, "delta": delta, "band": band,
            "verdict": wash_verdict(delta, band, lower_better=lower_better),
            "min_seeds": min(A["n"], B["n"]), "note": ("fewer than 3 seeds" if min(A["n"], B["n"]) < 3 else "")}

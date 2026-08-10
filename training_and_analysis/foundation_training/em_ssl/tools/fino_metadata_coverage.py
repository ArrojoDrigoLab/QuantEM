"""FINO metadata coverage / missingness report over the EM parent-tile manifest.

Streams the manifest once (reusing :func:`em_ssl.data.manifest.iter_manifest`, the shared
filter args, and the SSL filter) and reports, over the SSL-trainable tile population:

  * per-field present/missing counts + percentages for the FINO objective fields
    (``effective_nm_per_px``, ``modality``, ``organ``) and the diagnostic grouping
    fields (``source_id``, ``dataset_id``);
  * canonicalised ``modality`` / ``organ`` value distributions;
  * ``effective_nm_per_px`` validity (positive + finite) + log-scale mean/std + quantiles;
  * cross-tabs: modality×organ, modality×nm-quantile, organ×nm-quantile,
    dataset×modality, dataset×organ.

Writes ``fino_metadata_coverage.json``, ``fino_metadata_coverage.csv``,
``fino_metadata_missingness.md`` and a machine-readable ``fino_metadata_spec.json`` (vocab
``classes`` + ``normalize_map`` for the discrete factors, log-scale ``standardize`` for scale,
per-factor ``valid_fraction``, and a content fingerprint). It also prints a paste-ready
``metadata_factors:`` YAML block for the metadata-conditioning configs.

    python -m em_ssl.tools.fino_metadata_coverage --manifest <M> --output-root <R>/data_prep \\
        --min-side 512

``source_id`` / ``dataset_id`` are reported as diagnostics only — they are never FINO objectives.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from ..data.filters import SSLTileFilter
from ..data.manifest import iter_manifest
from ..fino.factors import (
    DEFAULT_MODALITY_CLASSES,
    DEFAULT_MODALITY_NORMALIZE,
    DEFAULT_ORGAN_CLASSES,
    UNKNOWN,
    _norm_token,
)
from .common import add_common_data_args, add_filter_args, build_filter_config, resolve_exports_root

# Diagnostic-only grouping fields (counted, never objectives).
DIAGNOSTIC_FIELDS = ("source_id", "dataset_id")

def _present(v) -> bool:
    return v is not None and not (isinstance(v, str) and v.strip() == "")

def _canon_modality(raw) -> str | None:
    if not _present(raw):
        return None
    lut = {_norm_token(c): c for c in DEFAULT_MODALITY_CLASSES}
    lut.update({_norm_token(k): v for k, v in DEFAULT_MODALITY_NORMALIZE.items()})
    return lut.get(_norm_token(raw), UNKNOWN)

def _canon_organ(raw) -> str | None:
    # Organ values are already clean (~9 labels) -> use as-is (data-derived vocab, no fixed map).
    if not _present(raw):
        return None
    return str(raw).strip()

def _quantiles(sorted_vals: list[float], qs: list[float]) -> list[float]:
    if not sorted_vals:
        return [float("nan")] * len(qs)
    n = len(sorted_vals)
    out = []
    for q in qs:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        out.append(sorted_vals[idx])
    return out

def _nm_quantile_bin(v: float, thresholds: list[float]) -> str:
    """Assign a value to a quartile bin label given the 25/50/75 thresholds."""
    if v <= thresholds[0]:
        return "Q1"
    if v <= thresholds[1]:
        return "Q2"
    if v <= thresholds[2]:
        return "Q3"
    return "Q4"

def run(args) -> dict:
    exports_root = resolve_exports_root(args)
    filt = SSLTileFilter(build_filter_config(args, min_side_default=args.min_side))

    total = 0
    kept = 0
    present = collections.Counter()  # field -> # present (kept tiles)
    modality_raw = collections.Counter()
    modality_canon = collections.Counter()
    organ_raw = collections.Counter()
    organ_canon = collections.Counter()
    source_ids: set[str] = set()
    dataset_ids: set[str] = set()
    dataset_counts = collections.Counter()
    nm_valid_log: list[float] = []
    # Per-tile joint rows for cross-tabs (kept tiles): (modality_canon, organ_canon, nm, dataset_id)
    rows: list[tuple] = []

    objective_fields = ("effective_nm_per_px", "modality", "organ")

    for rec in iter_manifest(args.manifest):
        total += 1
        if not filt(rec):
            continue
        kept += 1
        for fld in objective_fields + DIAGNOSTIC_FIELDS:
            if _present(rec.get(fld)):
                present[fld] += 1
        mr = rec.get("modality")
        mc = _canon_modality(mr)
        if _present(mr):
            modality_raw[str(mr)] += 1
            modality_canon[mc] += 1
        tr = rec.get("organ")
        tc = _canon_organ(tr)
        if _present(tr):
            organ_raw[str(tr)] += 1
            organ_canon[tc] += 1
        sid = rec.get("source_id")
        if _present(sid):
            source_ids.add(str(sid))
        did = rec.get("dataset_id")
        if _present(did):
            dataset_ids.add(str(did))
            dataset_counts[str(did)] += 1
        nm = rec.get("effective_nm_per_px")
        nm_val = None
        try:
            v = float(nm)
            if math.isfinite(v) and v > 0:
                nm_val = v
                nm_valid_log.append(math.log(v))
        except (TypeError, ValueError):
            pass
        rows.append((mc, tc, nm_val, str(did) if _present(did) else None))

    # ---- scale stats ----
    n_nm_valid = len(nm_valid_log)
    log_mean = sum(nm_valid_log) / n_nm_valid if n_nm_valid else None
    log_std = (
        math.sqrt(sum((x - log_mean) ** 2 for x in nm_valid_log) / n_nm_valid) if n_nm_valid and log_mean is not None else None
    )
    nm_sorted = sorted(math.exp(x) for x in nm_valid_log)
    nm_q = _quantiles(nm_sorted, [0.25, 0.5, 0.75])

    def _pct(field: str) -> dict:
        p = present[field]
        return {
            "valid": p,
            "missing": kept - p,
            "pct_valid": round(100.0 * p / kept, 3) if kept else 0.0,
            "pct_missing": round(100.0 * (kept - p) / kept, 3) if kept else 0.0,
        }

    coverage = {f: _pct(f) for f in objective_fields + DIAGNOSTIC_FIELDS}
    # effective_nm_per_px "valid" means positive+finite (stricter than merely present)
    coverage["effective_nm_per_px"]["valid_positive_finite"] = n_nm_valid
    coverage["effective_nm_per_px"]["pct_valid_positive_finite"] = (
        round(100.0 * n_nm_valid / kept, 3) if kept else 0.0
    )

    # ---- cross-tabs ----
    def _xtab(key_fn) -> dict:
        c: dict = collections.defaultdict(lambda: collections.Counter())
        for r in rows:
            a, b = key_fn(r)
            if a is None or b is None:
                continue
            c[a][b] += 1
        return {a: dict(bc.most_common()) for a, bc in c.items()}

    def _nm_bin(r):
        nm_val = r[2]
        if nm_val is None or not nm_sorted:
            return None
        return _nm_quantile_bin(nm_val, nm_q)

    cross_tabs = {
        "modality_x_organ": _xtab(lambda r: (r[0], r[1])),
        "modality_x_nm_quantile": _xtab(lambda r: (r[0], _nm_bin(r))),
        "organ_x_nm_quantile": _xtab(lambda r: (r[1], _nm_bin(r))),
        "dataset_id_x_modality": _xtab(lambda r: (r[3], r[0])),
        "dataset_id_x_organ": _xtab(lambda r: (r[3], r[1])),
    }

    # ---- derived factor spec (paste-ready vocab + standardize) ----
    modality_observed = [c for c, _ in modality_canon.most_common() if c != UNKNOWN]
    organ_observed = [c for c, _ in organ_canon.most_common() if c != UNKNOWN]
    valid_fraction = {
        "log_effective_nm_per_px": round(n_nm_valid / kept, 6) if kept else 0.0,
        "modality": round(present["modality"] / kept, 6) if kept else 0.0,
        "organ": round(present["organ"] / kept, 6) if kept else 0.0,
    }
    spec = {
        "kept_tiles": kept,
        "total_records": total,
        "valid_fraction": valid_fraction,
        "factors": {
            "log_effective_nm_per_px": {
                "field": "effective_nm_per_px",
                "type": "continuous",
                "log_transform": True,
                "standardize": {"mean": round(log_mean, 6) if log_mean is not None else None,
                                "std": round(log_std, 6) if log_std is not None else None},
                "effective_nm_per_px_quartiles": [round(x, 4) for x in nm_q],
            },
            "modality": {
                "field": "modality",
                "type": "discrete",
                "classes": modality_observed or list(DEFAULT_MODALITY_CLASSES),
                "normalize_map": _observed_normalize_map(modality_raw, _canon_modality),
            },
            "organ": {
                "field": "organ",
                "type": "discrete",
                "classes": organ_observed or list(DEFAULT_ORGAN_CLASSES),
                "normalize_map": _observed_normalize_map(organ_raw, _canon_organ),
            },
        },
    }
    spec["fingerprint"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    report = {
        "manifest": str(args.manifest),
        "kept_tiles": kept,
        "total_records": total,
        "coverage": coverage,
        "modality_distribution": dict(modality_canon.most_common()),
        "modality_raw_distribution": dict(modality_raw.most_common(50)),
        "organ_distribution": dict(organ_canon.most_common()),
        "organ_raw_distribution": dict(organ_raw.most_common(50)),
        "effective_nm_per_px": {
            "valid_positive_finite": n_nm_valid,
            "log_mean": log_mean,
            "log_std": log_std,
            "quartiles_nm": [round(x, 4) for x in nm_q],
        },
        "num_unique_source_id": len(source_ids),
        "num_unique_dataset_id": len(dataset_ids),
        "dataset_id_top": dataset_counts.most_common(25),
        "cross_tabs": cross_tabs,
        "spec": spec,
        "filter_summary": filt.summary(),
    }

    if args.output_root:
        _write_outputs(Path(args.output_root), report, spec)
        print(f"[fino_metadata_coverage] wrote coverage + spec to {args.output_root}")
    print(_paste_block(spec))
    return report

def _observed_normalize_map(raw_counter: collections.Counter, canon_fn) -> dict:
    """raw value -> canonical for every observed raw variant that differs from its canonical."""
    out: dict[str, str] = {}
    for raw, _ in raw_counter.most_common():
        canon = canon_fn(raw)
        if canon and canon != UNKNOWN and raw != canon:
            out[raw] = canon
    return out

def _write_outputs(out: Path, report: dict, spec: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "fino_metadata_coverage.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "fino_metadata_spec.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")
    # CSV: one row per field with counts/percentages.
    with open(out / "fino_metadata_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field", "role", "valid", "missing", "pct_valid", "pct_missing"])
        roles = {
            "effective_nm_per_px": "objective",
            "modality": "objective",
            "organ": "objective",
            "source_id": "diagnostic",
            "dataset_id": "diagnostic",
        }
        for field, cov in report["coverage"].items():
            w.writerow([field, roles.get(field, ""), cov["valid"], cov["missing"], cov["pct_valid"], cov["pct_missing"]])
    (out / "fino_metadata_missingness.md").write_text(_render_md(report), encoding="utf-8")

def _render_md(r: dict) -> str:
    lines = [
        "# FINO metadata coverage / missingness",
        "",
        f"- Manifest: `{r['manifest']}`",
        f"- SSL-trainable (kept) tiles: **{r['kept_tiles']:,}** / {r['total_records']:,} records",
        "",
        "## Coverage (over kept tiles)",
        "",
        "| field | role | valid | missing | % valid | % missing |",
        "|---|---|---|---|---|---|",
    ]
    roles = {"effective_nm_per_px": "objective", "modality": "objective", "organ": "objective",
             "source_id": "diagnostic", "dataset_id": "diagnostic"}
    for field, cov in r["coverage"].items():
        lines.append(
            f"| `{field}` | {roles.get(field,'')} | {cov['valid']:,} | {cov['missing']:,} | "
            f"{cov['pct_valid']} | {cov['pct_missing']} |"
        )
    nm = r["effective_nm_per_px"]
    lines += [
        "",
        f"- `effective_nm_per_px` positive+finite: **{nm['valid_positive_finite']:,}** "
        f"(log mean {nm['log_mean']}, log std {nm['log_std']}, nm quartiles {nm['quartiles_nm']})",
        f"- unique `source_id`: {r['num_unique_source_id']:,} | unique `dataset_id`: {r['num_unique_dataset_id']:,} "
        "(diagnostics only — never FINO objectives)",
        "",
        "## modality distribution (canonical)",
    ]
    for k, v in r["modality_distribution"].items():
        lines.append(f"- `{k}`: {v:,}")
    lines += ["", "## organ distribution (canonical)"]
    for k, v in r["organ_distribution"].items():
        lines.append(f"- `{k}`: {v:,}")
    lines += ["", "## Cross-tabs"]
    for name, tab in r["cross_tabs"].items():
        lines.append(f"\n### {name}\n")
        lines.append("```json")
        lines.append(json.dumps(tab, indent=2))
        lines.append("```")
    lines += ["", "## Derived factor spec (fingerprint: `%s`)" % r["spec"]["fingerprint"], "", "```yaml", _paste_block(r["spec"]), "```"]
    return "\n".join(lines)

def _paste_block(spec: dict) -> str:
    """Render a paste-ready metadata_factors example (modality M+ shown) from the derived spec."""
    fac = spec["factors"]
    mod = fac["modality"]
    sc = fac["log_effective_nm_per_px"]["standardize"]
    classes = ", ".join(mod["classes"])
    return (
        "# paste into a metadata-conditioning config; set name and guidance per arm:\n"
        "metadata_factors:\n"
        "  - name: modality\n"
        "    field: modality\n"
        "    type: discrete\n"
        "    guidance: positive   # positive (M+) | negative (M-) | disabled\n"
        "    loss_weight: 0.1\n"
        f"    classes: [{classes}]\n"
        "  # scale factor (continuous), for the preserve- and suppress-scale arms:\n"
        "  # - name: log_effective_nm_per_px\n"
        "  #   field: effective_nm_per_px\n"
        "  #   type: continuous\n"
        "  #   guidance: negative\n"
        "  #   loss_weight: 0.05\n"
        "  #   log_transform: true\n"
        f"  #   standardize_mean: {sc['mean']}\n"
        f"  #   standardize_std: {sc['std']}\n"
    )

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="FINO metadata coverage / missingness report.")
    add_common_data_args(p)
    add_filter_args(p)
    args = p.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()

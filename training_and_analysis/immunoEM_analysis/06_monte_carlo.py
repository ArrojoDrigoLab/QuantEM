"""Monte Carlo null: complete spatial randomness of gold within the tissue mask.

Tissue geometry alone will put some gold near mitochondria — mitochondria occupy
a large share of hepatocyte cytoplasm, so proximity is expected even with no
association whatsoever. To separate genuine association from that baseline, the
observed distribution is compared against complete spatial randomness.

For each image, the same number of localizations observed on tissue is placed
uniformly at random inside the same tissue mask and passed through the identical
downstream pipeline. Repeating this gives a null distribution per metric, and the
observed value is expressed as a z-score against it:

    z = (observed − null mean) / null s.d.        0 = chance, |z| < 2 = unremarkable

This controls for tissue morphology, compartment area fractions, organelle shape
and spatial arrangement, and the number of localizations per image. Seeded, so
runs are reproducible.

Group values are the unweighted mean over animals. With count-weighted pooling
the random null drifts away from 1.0 whenever an animal's gold count and its
nuclear area pull in opposite directions (one animal with many particles and a
small nucleus is enough to do it). Per-animal means keep each animal's null
pinned at 1.0 by construction.

Internal control: because each image's random nuclear enrichment is
a_nuc / a_nuc = 1.0 exactly, the group-level null must also be ~1.0. The run
asserts this at the end; a deviation means the normalization is biased.

    python 06_monte_carlo.py --reps 20
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    base_parser, canvases, group_of, groups, load_config, need, nm_per_px,
)

# Reuse the loaders and binning from step 05 so observed and null go through
# genuinely identical code — the whole validity of the null rests on that.
# (importlib, because "05_..." is not a valid identifier for an import statement.)
import importlib  # noqa: E402

_step05 = importlib.import_module("05_compartment_analysis")
BIN_EDGES, BIN_LABELS = _step05.BIN_EDGES, _step05.BIN_LABELS
load_masks, load_marks, boundary_tree = (
    _step05.load_masks, _step05.load_marks, _step05.boundary_tree
)


def tally(px, py, tissue, nucleus, mito, tree, px_nm):
    """Metrics for one set of points. Identical for observed and simulated."""
    H, W = tissue.shape
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)
    on = tissue[py, px]
    px, py = px[on], py[on]
    n_on = len(px)
    if not n_on:
        return None

    is_nuc = nucleus[py, px]
    n_nuc = int(is_nuc.sum())
    n_mito = int(mito[py, px].sum())

    cx, cy = px[~is_nuc], py[~is_nuc]
    inside = mito[cy, cx]
    if tree is not None and len(cx):
        d = tree.query(np.column_stack([cx, cy]))[0] * px_nm
    else:
        d = np.zeros(len(cx))

    return dict(
        n_on=n_on, n_nuc=n_nuc, n_mito=n_mito, n_cyto=int(len(cx)),
        n_cyto_in=int(inside.sum()),
        in_counts=np.histogram(d[inside], bins=BIN_EDGES)[0].tolist(),
        out_counts=np.histogram(d[~inside], bins=BIN_EDGES)[0].tolist(),
    )


def metrics(t, areas):
    """Per-image metrics from a tally. areas = (a_nuc, a_cyto, a_mito)."""
    a_nuc, a_cyto, a_mito = areas
    n = t["n_on"]
    enr = ((t["n_nuc"] / n) / a_nuc if a_nuc else None,
           ((n - t["n_nuc"]) / n) / a_cyto if a_cyto else None,
           (t["n_mito"] / n) / a_mito if a_mito else None)
    pct_in = 100 * t["n_cyto_in"] / t["n_cyto"] if t["n_cyto"] else None

    def frac(counts):
        s = sum(counts)
        return [c / s if s else None for c in counts]

    return dict(enr=enr, pct_in=pct_in,
                in_f=frac(t["in_counts"]), out_f=frac(t["out_counts"]))


def animal_mean(mets):
    """Unweighted mean across animals — each animal contributes once."""
    enr = tuple(float(np.mean([m["enr"][i] for m in mets if m["enr"][i] is not None]))
                for i in range(3))
    pv = [m["pct_in"] for m in mets if m["pct_in"] is not None]

    def bins(key):
        return [float(np.mean([m[key][i] for m in mets if m[key][i] is not None]))
                if any(m[key][i] is not None for m in mets) else None for i in range(4)]

    return dict(enr=enr, pct_in=float(np.mean(pv)) if pv else None,
                in_f=bins("in_f"), out_f=bins("out_f"))


def sample_in_tissue(tissue, n, rng):
    """Rejection-sample n points uniformly inside the tissue mask."""
    H, W = tissue.shape
    frac = max(float(tissue.sum()) / (H * W), 0.05)
    xs = np.empty(0, int)
    ys = np.empty(0, int)
    while len(xs) < n:
        draw = int((n - len(xs)) / frac * 1.3) + 16
        cx = rng.integers(0, W, draw)
        cy = rng.integers(0, H, draw)
        keep = tissue[cy, cx]
        xs = np.concatenate([xs, cx[keep]])
        ys = np.concatenate([ys, cy[keep]])
    return xs[:n], ys[:n]


def main(argv=None) -> int:
    ap = base_parser(__doc__.split("\n")[0])
    ap.add_argument("--canvas", nargs="*", help="canvas name(s); default all in paths.yaml")
    ap.add_argument("--masks", help="directory of tissue and organelle masks")
    ap.add_argument("--results", help="directory holding particle tables; results written here")
    ap.add_argument("--reps", type=int, default=20, help="simulated replicates per image")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)

    cfg = load_config(args.paths)
    masks_dir = need(cfg, "masks_dir", args.masks)
    results_dir = need(cfg, "results_dir", args.results)
    px_nm = nm_per_px(cfg)
    rng = np.random.default_rng(args.seed)

    observed, simulated, per_image = {}, {}, {}

    for canvas in canvases(cfg, args.canvas):
        tissue, nucleus, mito = load_masks(masks_dir, canvas)
        px, py = load_marks(results_dir / f"{canvas}_gold_particles.csv", tissue.shape, canvas)
        tree = boundary_tree(mito)

        area = int(tissue.sum())
        a_nuc = float(nucleus.sum()) / area
        areas = (a_nuc, 1.0 - a_nuc, float(mito.sum()) / area)

        obs = tally(px, py, tissue, nucleus, mito, tree, px_nm)
        if obs is None:
            print(f"{canvas}: no on-tissue localizations — skipped", file=sys.stderr)
            continue

        reps = []
        for _ in range(args.reps):
            sx, sy = sample_in_tissue(tissue, obs["n_on"], rng)
            reps.append(tally(sx, sy, tissue, nucleus, mito, tree, px_nm))

        observed[canvas] = (obs, areas)
        simulated[canvas] = reps

        om = metrics(obs, areas)
        mm = np.array([metrics(r, areas)["enr"] for r in reps], dtype=float)
        per_image[canvas] = dict(
            group=group_of(cfg, canvas), n=obs["n_on"], areas=areas,
            obs_enr=list(om["enr"]),
            null_enr_mean=mm.mean(0).tolist(), null_enr_sd=mm.std(0).tolist(),
            obs_pct_cyto_inside=om["pct_in"],
        )
        print(f"{canvas:18s} N={obs['n_on']:6d}  "
              f"enrNuc obs={om['enr'][0]:.2f} null={mm[:, 0].mean():.2f}±{mm[:, 0].std():.2f}   "
              f"enrMito obs={om['enr'][2]:.2f} null={mm[:, 2].mean():.2f}±{mm[:, 2].std():.2f}")

    print(f"\n=== GROUP: OBSERVED vs NULL "
          f"(per-animal mean; CSR null over {args.reps} replicates) ===")
    per_group = {}
    for g in groups(cfg):
        members = [c for c in observed if group_of(cfg, c) == g]
        if not members:
            continue
        obs_g = animal_mean([metrics(*observed[c]) for c in members])
        null_g = [animal_mean([metrics(simulated[c][r], observed[c][1]) for c in members])
                  for r in range(args.reps)]

        def stat(key, idx=None):
            vals = [(p[key][idx] if idx is not None else p[key]) for p in null_g]
            vals = [v for v in vals if v is not None]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

        enr_null = [stat("enr", i) for i in range(3)]
        pct_null = stat("pct_in")
        z = lambda o, s: (o - s[0]) / s[1] if (o is not None and s[1]) else 0.0  # noqa: E731

        per_group[g] = dict(
            n_animals=len(members), observed=obs_g,
            null_enr=enr_null, null_pct_cyto_inside=pct_null,
            z_enr=[z(obs_g["enr"][i], enr_null[i]) for i in range(3)],
            z_pct_cyto_inside=z(obs_g["pct_in"], pct_null),
            null_out_f=[stat("out_f", i) for i in range(4)],
            obs_out_f=obs_g["out_f"],
        )

        print(f"\n{g}  (n={len(members)})")
        for i, comp in enumerate(("nuclear", "cytoplasmic", "mitochondrial")):
            print(f"  {comp:14s} obs {obs_g['enr'][i]:.2f}   "
                  f"null {enr_null[i][0]:.2f}±{enr_null[i][1]:.2f}   "
                  f"z={per_group[g]['z_enr'][i]:+.1f}")
        print(f"  % cytoplasmic gold inside mitochondria: obs {obs_g['pct_in']:.1f}%   "
              f"null {pct_null[0]:.1f}±{pct_null[1]:.1f}%   z={per_group[g]['z_pct_cyto_inside']:+.1f}")

    # ---- internal control -------------------------------------------------- #
    # Each image's random nuclear enrichment is a_nuc/a_nuc = 1.0 by construction,
    # so the group null must be ~1.0. If it is not, the normalization is biased.
    print("\n=== INTERNAL CONTROL: null enrichment on randomized data ===")
    worst, ok = 0.0, True
    for g, gr in per_group.items():
        for i, comp in enumerate(("nuclear", "cytoplasmic", "mitochondrial")):
            val = gr["null_enr"][i][0]
            if val is None:
                continue
            worst = max(worst, abs(val - 1.0))
            flag = "OK " if abs(val - 1.0) <= 0.05 else "!! "
            ok &= abs(val - 1.0) <= 0.05
            print(f"  {flag}{g:12s} {comp:14s} null enrichment = {val:.3f}")
    print(f"\n  max deviation from 1.0: {worst:.3f} — "
          f"{'normalization is unbiased' if ok else 'normalization may be biased'}")

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "monte_carlo_per_image.json").write_text(
        json.dumps(per_image, indent=2, default=float), encoding="utf8")
    (results_dir / "monte_carlo_per_group.json").write_text(
        json.dumps(per_group, indent=2, default=float), encoding="utf8")
    print(f"\nWrote monte_carlo_per_image.json and monte_carlo_per_group.json to {results_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Assign gold localizations to compartments and measure mitochondrial proximity.

Segmentation and tissue masks are inputs to this script — see the README for the
expected layout. Organelle masks come from QuantEM; the tissue mask is a manual
delineation of the section.

Compartments, all within the tissue mask:
    nuclear      = nucleus ∩ tissue
    cytoplasmic  = tissue ∖ nucleus
    mitochondrial = mitochondria ∩ tissue   (a subset of cytoplasm)

Enrichment normalizes for how much of the section each compartment occupies:

    enrichment = (fraction of on-tissue gold in the compartment)
               / (fraction of tissue AREA the compartment occupies)

so 1.0 is exactly what area alone predicts and > 1 is over-representation.
Localizations outside the tissue mask are excluded.

For cytoplasmic gold, distance to the nearest mitochondrial boundary is measured
and binned, separately for gold inside a mitochondrion (how deep) and outside
one (how close).

Group values are the unweighted mean over animals — each animal counts once,
regardless of how many particles it contributed.

    python 05_compartment_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    base_parser, canvases, group_of, groups, load_config, load_mask, need, nm_per_px,
)

BIN_EDGES = [0, 50, 100, 200, np.inf]
BIN_LABELS = ["<50nm", "50-100nm", "100-200nm", ">200nm"]


def load_masks(masks_dir: Path, canvas: str):
    """tissue, nucleus, mitochondria — organelles restricted to the tissue mask.

    Masks are full-canvas overlays at the same pixel grid as the EM and the
    registered mosaics; mismatched dimensions are an error rather than something
    to resample around.
    """
    loaded = {}
    for kind in ("tissue", "nucleus", "mitochondria"):
        hits = sorted((masks_dir / canvas).glob(f"{kind}.*"))
        if not hits:
            raise SystemExit(f"{canvas}: no '{kind}' mask under {masks_dir / canvas}")
        loaded[kind] = load_mask(hits[0])

    shapes = {k: v.shape for k, v in loaded.items()}
    if len(set(shapes.values())) != 1:
        detail = ", ".join(f"{k}={v[1]}x{v[0]}" for k, v in shapes.items())
        raise SystemExit(
            f"{canvas}: mask dimensions differ ({detail}). Masks must be full-canvas "
            f"overlays on the same pixel grid."
        )

    tissue = loaded["tissue"]
    return tissue, loaded["nucleus"] & tissue, loaded["mitochondria"] & tissue


def load_marks(csv_path: Path, shape, canvas: str = ""):
    """Particle marks -> integer pixel coordinates on the mask/EM grid."""
    H, W = shape
    xs, ys = [], []
    with Path(csv_path).open(encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["mark_x"]))
            ys.append(float(row["mark_y"]))
    if not xs:
        return np.empty(0, int), np.empty(0, int)
    px = np.round(np.asarray(xs, float)).astype(int)
    py = np.round(np.asarray(ys, float)).astype(int)

    # Localizations are in mosaic pixels; the masks share that grid. Coordinates
    # outside it mean the two are not on the same grid.
    if px.max() >= W or py.max() >= H or px.min() < 0 or py.min() < 0:
        raise SystemExit(
            f"{canvas or csv_path.name}: localizations extend to "
            f"({int(px.max())}, {int(py.max())}) but the masks are {W}x{H}. "
            f"Masks and mosaics must be on the same pixel grid."
        )
    return px, py


def boundary_tree(mito: np.ndarray):
    """KD-tree over mitochondrial boundary pixels (mask minus its erosion)."""
    edge = mito & ~ndi.binary_erosion(mito)
    by, bx = np.where(edge)
    return cKDTree(np.column_stack([bx, by])) if len(bx) else None


def _binned(distances):
    n = len(distances)
    counts, _ = np.histogram(distances, bins=BIN_EDGES)
    frac = {BIN_LABELS[i]: (float(counts[i] / n) if n else None) for i in range(4)}
    median = float(np.median(distances)) if n else None
    return n, frac, median


def analyse(canvas: str, cfg: dict, masks_dir: Path, results_dir: Path, px_nm: float):
    tissue, nucleus, mito = load_masks(masks_dir, canvas)
    px, py = load_marks(results_dir / f"{canvas}_gold_particles.csv", tissue.shape, canvas)

    area = int(tissue.sum())
    a_nuc = float(nucleus.sum()) / area
    a_mito = float(mito.sum()) / area
    a_cyto = 1.0 - a_nuc

    on = tissue[py, px]
    n_on = int(on.sum())
    if not n_on:
        raise SystemExit(f"{canvas}: no localizations fall on the tissue mask")

    g_nuc = int((nucleus[py, px] & on).sum())
    g_mito = int((mito[py, px] & on).sum())
    f_nuc, f_mito = g_nuc / n_on, g_mito / n_on
    f_cyto = 1.0 - f_nuc

    # cytoplasmic gold: on tissue, not nuclear
    cyto = on & ~nucleus[py, px]
    cx, cy = px[cyto], py[cyto]
    inside = mito[cy, cx]

    tree = boundary_tree(mito)
    if tree is not None and len(cx):
        d_nm = tree.query(np.column_stack([cx, cy]))[0] * px_nm
    else:
        d_nm = np.zeros(len(cx))

    n_in, frac_in, med_in = _binned(d_nm[inside])
    n_out, frac_out, med_out = _binned(d_nm[~inside])

    return dict(
        canvas=canvas, group=group_of(cfg, canvas),
        n_total=int(len(px)), n_on_tissue=n_on, n_off_tissue=int((~on).sum()),
        area_nuc=a_nuc, area_cyto=a_cyto, area_mito=a_mito,
        gold_nuc=f_nuc, gold_cyto=f_cyto, gold_mito=f_mito,
        enr_nuc=f_nuc / a_nuc if a_nuc else None,
        enr_cyto=f_cyto / a_cyto if a_cyto else None,
        enr_mito=f_mito / a_mito if a_mito else None,
        n_cyto=int(len(cx)), n_cyto_inside=n_in, n_cyto_outside=n_out,
        pct_cyto_inside=100 * n_in / len(cx) if len(cx) else None,
        dist_inside=frac_in, dist_outside=frac_out,
        median_inside_nm=med_in, median_outside_nm=med_out,
    )


def roll_up(rows):
    """Unweighted mean over animals, with SEM across animals."""
    def mean(key):
        v = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(v)) if v else None

    def sem(key):
        v = [r[key] for r in rows if r[key] is not None]
        return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    def bins(key):
        return {lab: (float(np.mean([r[key][lab] for r in rows if r[key][lab] is not None]))
                      if any(r[key][lab] is not None for r in rows) else None)
                for lab in BIN_LABELS}

    return dict(
        n_animals=len(rows), n_on_tissue=sum(r["n_on_tissue"] for r in rows),
        area_nuc=mean("area_nuc"), area_cyto=mean("area_cyto"), area_mito=mean("area_mito"),
        gold_nuc=mean("gold_nuc"), gold_cyto=mean("gold_cyto"), gold_mito=mean("gold_mito"),
        enr_nuc=mean("enr_nuc"), enr_cyto=mean("enr_cyto"), enr_mito=mean("enr_mito"),
        enr_nuc_sem=sem("enr_nuc"), enr_mito_sem=sem("enr_mito"),
        pct_cyto_inside=mean("pct_cyto_inside"),
        n_cyto_inside=sum(r["n_cyto_inside"] for r in rows),
        n_cyto_outside=sum(r["n_cyto_outside"] for r in rows),
        dist_inside=bins("dist_inside"), dist_outside=bins("dist_outside"),
    )


def main(argv=None) -> int:
    ap = base_parser(__doc__.split("\n")[0])
    ap.add_argument("--canvas", nargs="*", help="canvas name(s); default all in paths.yaml")
    ap.add_argument("--masks", help="directory of tissue and organelle masks")
    ap.add_argument("--results", help="directory holding particle tables; results are written here")
    args = ap.parse_args(argv)

    cfg = load_config(args.paths)
    masks_dir = need(cfg, "masks_dir", args.masks)
    results_dir = need(cfg, "results_dir", args.results)
    px_nm = nm_per_px(cfg)

    rows = [analyse(c, cfg, masks_dir, results_dir, px_nm)
            for c in canvases(cfg, args.canvas)]

    pc = lambda v: "   n/a" if v is None else f"{100 * v:5.1f}"  # noqa: E731
    en = lambda v: " n/a" if v is None else f"{v:4.2f}"          # noqa: E731

    print("\n=== COMPARTMENT AREA %, GOLD %, ENRICHMENT (gold% / area%) ===")
    print(f"{'canvas':18s} {'group':12s} | {'nuc%':>6s} {'cyt%':>6s} {'mit%':>6s} |"
          f" {'gNuc%':>6s} {'gCyt%':>6s} {'gMit%':>6s} | {'eNuc':>4s} {'eCyt':>4s} {'eMit':>4s}")
    for r in rows:
        print(f"{r['canvas']:18s} {r['group']:12s} |"
              f" {pc(r['area_nuc'])} {pc(r['area_cyto'])} {pc(r['area_mito'])} |"
              f" {pc(r['gold_nuc'])} {pc(r['gold_cyto'])} {pc(r['gold_mito'])} |"
              f" {en(r['enr_nuc'])} {en(r['enr_cyto'])} {en(r['enr_mito'])}")

    print("\n=== CYTOPLASMIC GOLD: distance to nearest mitochondrial border ===")
    for tag, key, n_key in (("INSIDE (depth)", "dist_inside", "n_cyto_inside"),
                            ("OUTSIDE (gap)", "dist_outside", "n_cyto_outside")):
        print(f"\n{tag}")
        print(f"{'canvas':18s} | {'n':>6s} | " + " ".join(f"{b:>10s}" for b in BIN_LABELS))
        for r in rows:
            d = r[key]
            print(f"{r['canvas']:18s} | {r[n_key]:6d} | " +
                  " ".join(pc(d[b]) + "%" for b in BIN_LABELS))

    print("\n=== GROUP ROLL-UP (unweighted per-animal mean) ===")
    per_group = {}
    for g in groups(cfg):
        members = [r for r in rows if r["group"] == g]
        if not members:
            continue
        gr = roll_up(members)
        per_group[g] = gr
        print(f"\n{g}  (n={gr['n_animals']})")
        print(f"  area  nuc/cyto/mito = {100*gr['area_nuc']:.1f}/{100*gr['area_cyto']:.1f}"
              f"/{100*gr['area_mito']:.1f}%")
        print(f"  gold  nuc/cyto/mito = {100*gr['gold_nuc']:.1f}/{100*gr['gold_cyto']:.1f}"
              f"/{100*gr['gold_mito']:.1f}%")
        print(f"  enrichment nuc={gr['enr_nuc']:.2f}±{gr['enr_nuc_sem']:.2f}"
              f"  cyto={gr['enr_cyto']:.2f}  mito={gr['enr_mito']:.2f}±{gr['enr_mito_sem']:.2f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "compartments_per_image.json").write_text(
        json.dumps({r["canvas"]: r for r in rows}, indent=2), encoding="utf8")
    (results_dir / "compartments_per_group.json").write_text(
        json.dumps(per_group, indent=2), encoding="utf8")
    print(f"\nWrote compartments_per_image.json and compartments_per_group.json to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

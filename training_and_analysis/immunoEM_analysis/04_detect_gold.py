"""Detect discrete gold localizations in a registered ¹⁹⁷Au mosaic.

The per-pixel ¹⁹⁷Au count distribution is bimodal. Most non-zero pixels carry
only a few counts and form a diffuse background — stray secondary ions,
redeposition, and the tail of the primary-beam point-spread function. Specific
antibody labelling appears as compact, high-amplitude points.

A particle is therefore one 8-connected component of pixels above a count
threshold. The threshold does double duty: it removes the diffuse background,
and it stops the low-count halo around a genuine peak from bridging two
neighbouring labels into one component.

A single primary antibody may be detected by more than one gold-conjugated
secondary, so no component is ever split — one component is one labelling event,
positioned at its intensity-weighted centroid (the centre of gold counts, not of
the thresholded shape).

    python 04_detect_gold.py --canvas 73_6hrfast_M1
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, canvases, load_config, need, read_image  # noqa: E402

EIGHT_CONNECTED = np.ones((3, 3), bool)

COLUMNS = ["component_id", "mark_x", "mark_y", "peak_x", "peak_y", "peak_value",
           "geom_x", "geom_y", "area_px", "total_counts"]


def detect(mosaic: np.ndarray, floor: int):
    """Return one row per labelling event. Coordinates are in mosaic pixels."""
    # Crop to the non-zero extent first — the mosaics are mostly empty canvas,
    # and labelling the full frame is needlessly expensive.
    nz = mosaic > 0
    if not nz.any():
        return []
    rows_nz = np.where(nz.any(axis=1))[0]
    cols_nz = np.where(nz.any(axis=0))[0]
    y0, x0 = int(rows_nz.min()), int(cols_nz.min())
    sub = mosaic[y0:int(rows_nz.max()) + 1, x0:int(cols_nz.max()) + 1]

    binary = sub > floor
    labels, n = ndi.label(binary, structure=EIGHT_CONNECTED)
    if n == 0:
        return []

    idx = np.arange(1, n + 1)
    counts = sub.astype(np.float64)
    peak = ndi.maximum(sub, labels, idx)
    area = ndi.sum(binary, labels, idx).astype(int)
    total = ndi.sum(counts, labels, idx)
    weighted = np.atleast_2d(np.asarray(ndi.center_of_mass(counts, labels, idx)))
    geometric = np.atleast_2d(np.asarray(ndi.center_of_mass(binary, labels, idx)))
    peak_pos = np.atleast_2d(np.asarray(ndi.maximum_position(sub, labels, idx)))

    out = []
    for k in range(n):
        out.append({
            "component_id": k,
            "mark_x": round(float(weighted[k, 1]) + x0, 2),
            "mark_y": round(float(weighted[k, 0]) + y0, 2),
            "peak_x": int(peak_pos[k, 1]) + x0,
            "peak_y": int(peak_pos[k, 0]) + y0,
            "peak_value": int(peak[k]),
            "geom_x": round(float(geometric[k, 1]) + x0, 2),
            "geom_y": round(float(geometric[k, 0]) + y0, 2),
            "area_px": int(area[k]),
            "total_counts": int(total[k]),
        })
    return out


def main(argv=None) -> int:
    ap = base_parser(__doc__.split("\n")[0])
    ap.add_argument("--canvas", nargs="*", help="canvas name(s); default all in paths.yaml")
    ap.add_argument("--mosaics", help="directory of registered mosaics (from step 03)")
    ap.add_argument("--out", help="output directory for particle tables")
    ap.add_argument("--isotope", default="197Au", help="gold isotope raster to read")
    ap.add_argument("--floor", type=int, default=4,
                    help="keep components of pixels strictly above this count")
    args = ap.parse_args(argv)

    cfg = load_config(args.paths)
    mosaic_dir = need(cfg, "mosaic_dir", args.mosaics)
    out_dir = need(cfg, "results_dir", args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in canvases(cfg, args.canvas):
        src = next(iter((mosaic_dir / name).glob(f"{args.isotope}.*")), None)
        if src is None:
            print(f"{name}: no {args.isotope} mosaic — skipped", file=sys.stderr)
            continue

        rows = detect(read_image(src), args.floor)
        dst = out_dir / f"{name}_gold_particles.csv"
        with dst.open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)

        if rows:
            areas = np.array([r["area_px"] for r in rows])
            peaks = np.array([r["peak_value"] for r in rows])
            print(f"{name:16s} {len(rows):6d} particles  "
                  f"median area={int(np.median(areas))}px  max area={int(areas.max())}px  "
                  f"median peak={int(np.median(peaks))}")
        else:
            print(f"{name:16s}      0 particles above {args.floor}")

    print(f"\nParticle tables written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

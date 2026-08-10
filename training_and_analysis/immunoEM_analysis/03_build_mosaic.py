"""Assemble registered MIMS fields into one canvas-sized mosaic per isotope.

Each field was warped into EM space by step 02 and carries the bounding box it
occupies on the canvas. This script places them, without blending: where two
acquisitions overlap, the earlier one wins. Fields are named with a numeric
suffix in acquisition order, so sorting on that suffix is sorting by acquisition.

Only pixels that are still empty get painted, which is what makes "earlier wins"
hold regardless of how many fields overlap.

    python 03_build_mosaic.py --canvas 73_6hrfast_M1
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    base_parser, canvases, load_config, need, read_image, write_image,
)

# ¹⁹⁷Au is written 8-bit to match the released rasters. Counts above 255 are
# clipped; the gold detector thresholds well below that, so detection is
# unaffected — relevant only where peak values are reused quantitatively.
UINT8_ISOTOPES = {"197Au"}


def acquisition_order(field_dir: Path):
    """Sort key: trailing integer in the field name, then the name itself."""
    m = re.search(r"(\d+)$", field_dir.name)
    return (int(m.group(1)) if m else 0, field_dir.name)


def build(canvas_dir: Path, width: int, height: int, isotope: str, dtype) -> np.ndarray:
    mosaic = np.zeros((height, width), dtype=dtype)
    placed = 0
    for field in sorted((p for p in canvas_dir.iterdir() if p.is_dir()), key=acquisition_order):
        src = field / f"{isotope.replace(' ', '_')}.tif"
        place = field / "placement.json"
        if not (src.exists() and place.exists()):
            continue
        x0, y0, x1, y1 = json.loads(place.read_text(encoding="utf8"))["bbox"]
        patch = read_image(src)
        if dtype == np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)

        h, w = patch.shape
        h = min(h, height - y0)
        w = min(w, width - x0)
        if h <= 0 or w <= 0:
            continue
        patch = patch[:h, :w]
        region = mosaic[y0:y0 + h, x0:x0 + w]

        # Paint only where the canvas is still empty: earlier acquisitions win.
        write_where = (patch > 0) & (region == 0)
        region[write_where] = patch[write_where]
        placed += 1
    return mosaic, placed


def main(argv=None) -> int:
    ap = base_parser(__doc__.split("\n")[0])
    ap.add_argument("--canvas", nargs="*", help="canvas name(s); default all in paths.yaml")
    ap.add_argument("--warped", help="directory of warped fields (from step 02)")
    ap.add_argument("--out", help="output directory for mosaics")
    ap.add_argument("--isotopes", nargs="+", help="isotopes to assemble; default: all present")
    ap.add_argument("--format", choices=["png", "tif"], default="png")
    args = ap.parse_args(argv)

    cfg = load_config(args.paths)
    warped_dir = need(cfg, "warped_dir", args.warped)
    out_dir = need(cfg, "mosaic_dir", args.out)

    for name, meta in canvases(cfg, args.canvas).items():
        canvas_dir = warped_dir / name
        if not canvas_dir.is_dir():
            print(f"{name}: nothing under {canvas_dir} — skipped", file=sys.stderr)
            continue
        width, height = int(meta["width"]), int(meta["height"])

        isotopes = args.isotopes
        if not isotopes:
            isotopes = sorted({p.stem.replace("_", " ")
                               for d in canvas_dir.iterdir() if d.is_dir()
                               for p in d.glob("*.tif")})
        print(f"\n{name}  canvas {width}x{height}  ({len(isotopes)} isotopes)")

        for iso in isotopes:
            dtype = np.uint8 if iso in UINT8_ISOTOPES else np.uint16
            mosaic, placed = build(canvas_dir, width, height, iso, dtype)
            dst = out_dir / name / f"{iso.replace(' ', '_')}.{args.format}"
            write_image(dst, mosaic)
            print(f"  {iso:10s} {placed:3d} fields  max={int(mosaic.max()):5d}  -> {dst.name}")
            del mosaic

    print(f"\nMosaics written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

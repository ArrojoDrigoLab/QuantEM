"""Cameca .im -> one accumulated secondary-ion count image per isotope.

MIMS data arrive as multiplane secondary-ion count images. Each acquisition plane
is a separate raster of the same field, so the planes must be aligned to each
other before they can be summed: the stage drifts over the tens of minutes an
acquisition takes.

For each isotope:
  1. read the plane stack from the .im file
  2. correct inter-plane drift by frame-to-frame affine registration
  3. sum along z into a single accumulated count image

Counts are preserved as integers; no contrast scaling is applied here.

    python 01_mims_ingest.py --isotopes 197Au 12C 13C "14N 12C" 32S
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, load_config, need, write_image  # noqa: E402


def accumulate(im_path: Path, isotope: str) -> np.ndarray:
    """Drift-correct and z-sum one isotope's plane stack from a Cameca .im file."""
    import sims
    from pystackreg import StackReg
    from pystackreg.util import to_uint16

    m = sims.SIMS(str(im_path))
    if m is None or m.data is None or m.data.species is None:
        raise ValueError(f"Not a readable .im file: {im_path}")
    if isotope not in m.data.species.values:
        return None  # this field did not acquire this isotope

    planes = m.data.loc[isotope].to_numpy()

    # Frame-to-frame affine registration against the previous plane, so drift
    # accumulated over the acquisition is removed rather than referenced to a
    # single (possibly atypical) first frame.
    sr = StackReg(StackReg.AFFINE)
    sr.register_stack(planes, reference="previous")
    aligned = to_uint16(sr.transform_stack(planes))

    return aligned.sum(axis=0).astype(np.uint32)


def main(argv=None) -> int:
    ap = base_parser(__doc__.split("\n")[0])
    ap.add_argument("--im-dir", help="directory of raw Cameca .im files")
    ap.add_argument("--out", help="output directory for accumulated count images")
    ap.add_argument("--isotopes", nargs="+", required=True,
                    help='isotope names as recorded in the .im file, e.g. 197Au "14N 12C"')
    ap.add_argument("--format", choices=["tif", "png"], default="tif",
                    help="output format; TIFF preserves counts above 65535")
    args = ap.parse_args(argv)

    cfg = load_config(args.paths)
    im_dir = need(cfg, "mims_im_dir", args.im_dir)
    out_dir = need(cfg, "accumulated_dir", args.out)

    im_files = sorted(im_dir.glob("*.im"))
    if not im_files:
        raise SystemExit(f"No .im files under {im_dir}")

    for im_path in im_files:
        field = im_path.stem
        for iso in args.isotopes:
            dst = out_dir / field / f"{iso.replace(' ', '_')}.{args.format}"
            if dst.exists():
                print(f"  skip {field}/{iso} (exists)")
                continue
            try:
                acc = accumulate(im_path, iso)
            except Exception as exc:  # a corrupt field should not stop the batch
                print(f"  FAIL {field}/{iso}: {exc}", file=sys.stderr)
                continue
            if acc is None:
                continue
            # uint16 covers the accumulated count range of the released isotopes; wider ranges are kept
            # at full width.
            arr = acc.astype(np.uint16) if acc.max() <= 65535 else acc
            write_image(dst, arr)
            print(f"  {field}/{iso}: {arr.shape} max={int(arr.max())} -> {dst.name}")

    print(f"\nAccumulated count images written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

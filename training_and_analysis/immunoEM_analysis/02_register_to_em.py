"""Register each accumulated MIMS field to its EM canvas, using saved landmarks.

Landmark correspondences are an input to this script. They were selected
externally in standard image tools (Fiji, napari, or equivalent) and saved as
JSON; see the format note in the README.

Per field:
  1. reduce landmark polygons to their centroids, and stack them with the point
     landmarks to give matched MIMS/EM correspondence lists
  2. estimate a similarity transform; separately estimate one with the MIMS
     landmarks mirrored in x, and keep whichever gives the smaller residual
     (sections are mounted either face up or face down)
  3. remove residual translational bias using the median landmark offset, which
     is robust to a few badly placed landmarks in a way least-squares is not
  4. fit a thin-plate spline to what the similarity transform could not explain,
     correcting local non-linear distortion
  5. warp each isotope into EM space and record its bounding box on the canvas

Writes one warped image per isotope plus a `placement.json` giving the bounding
box, which `03_build_mosaic.py` consumes.

    python 02_register_to_em.py --canvas 73_6hrfast_M1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from skimage.transform import (
    SimilarityTransform,
    ThinPlateSplineTransform,
    estimate_transform,
    warp,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    base_parser, canvases, load_config, need, read_image, write_image,
)


# --------------------------------------------------------------------------- #
# landmarks
# --------------------------------------------------------------------------- #
def _centroid(poly: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(poly, float)[:, :2], axis=0)


def load_landmarks(path: Path):
    """Return matched (mims_xy, em_xy) landmark arrays.

    The file holds two optional pairs of index-aligned lists: traced polygons
    (`mims_shapes` / `em_shapes`, reduced here to their centroids) and discrete
    points (`mims_points` / `em_points`). Points are stored row-major, so they
    are swapped to (x, y) on load.
    """
    d = json.loads(Path(path).read_text(encoding="utf8"))

    def shapes(key):
        out = []
        for s in d.get(key, []):
            a = np.asarray(s, float)
            if a.ndim == 2 and a.shape[0] >= 3 and a.shape[1] >= 2:
                out.append(a)
        return out

    def points(key):
        return [np.asarray([p[1], p[0]], float) for p in d.get(key, [])]

    em_s, mims_s = shapes("em_shapes"), shapes("mims_shapes")
    em_p, mims_p = points("em_points"), points("mims_points")

    if len(em_s) != len(mims_s) or len(em_p) != len(mims_p):
        raise ValueError(f"{path.name}: MIMS and EM landmark lists differ in length")

    mims = [_centroid(s) for s in mims_s] + mims_p
    em = [_centroid(s) for s in em_s] + em_p
    if len(mims) < 3:
        raise ValueError(f"{path.name}: need at least 3 landmark pairs, got {len(mims)}")
    return np.asarray(mims, float), np.asarray(em, float)


def _mirror_x(a: np.ndarray, max_x: float) -> np.ndarray:
    a = np.asarray(a, float).copy()
    if a.ndim == 1:
        a[0] = max_x - a[0]
    else:
        a[:, 0] = max_x - a[:, 0]
    return a


def _residual(tf, src, dst) -> float:
    return float(np.mean(np.linalg.norm(tf(src) - dst, axis=1)))


def similarity_with_mirror_test(mims_xy, em_xy, width):
    """Estimate a similarity transform, testing whether an x-mirror fits better."""
    max_x = width - 1
    plain = estimate_transform("similarity", mims_xy, em_xy)
    mirrored_src = _mirror_x(mims_xy, max_x)
    mirrored = estimate_transform("similarity", mirrored_src, em_xy)
    if _residual(mirrored, mirrored_src, em_xy) < _residual(plain, mims_xy, em_xy):
        return mirrored, True, max_x
    return plain, False, max_x


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def registration_geometry(mims_xy, em_xy, mims_shape, em_shape):
    """Build the forward warp and the EM bounding box for one field."""
    h_mims, w_mims = mims_shape
    base, flip, max_x = similarity_with_mirror_test(mims_xy, em_xy, w_mims)
    if flip:
        mims_xy = _mirror_x(mims_xy, max_x)

    # 3. median-offset debias — robust to a minority of poorly placed landmarks
    offset = np.median(base(mims_xy) - em_xy, axis=0)

    mims_tf = SimilarityTransform(scale=1, rotation=base.rotation, translation=[0, 0])
    em_tf = SimilarityTransform(
        scale=1 / base.scale,
        translation=-(base.translation - offset) / base.scale,
    )

    # Keep the warped field in positive canvas coordinates.
    corners = np.array(
        [[0, 0], [w_mims - 1, 0], [w_mims - 1, h_mims - 1], [0, h_mims - 1]], float
    )
    if flip:
        corners = _mirror_x(corners, max_x)
    shift = -np.minimum(mims_tf(corners).min(axis=0), 0)
    if (shift > 0).any():
        mims_tf = SimilarityTransform(scale=1, rotation=base.rotation, translation=shift)
        em_tf = SimilarityTransform(
            scale=1 / base.scale,
            translation=-(base.translation - offset) / base.scale + shift,
        )
    canvas_corners = mims_tf(corners)
    out_shape = [
        int(np.ptp(canvas_corners[:, 1])),
        int(np.ptp(canvas_corners[:, 0])),
    ]

    # 4. thin-plate spline on the residual. estimate(dst, src) yields the inverse
    #    map, which is what skimage.transform.warp expects.
    tps = ThinPlateSplineTransform()
    tps.estimate(em_tf(em_xy), mims_tf(mims_xy))

    # 5. bounding box of this field on the EM canvas
    em_corners = em_tf.inverse(canvas_corners)
    x0, y0 = np.floor(em_corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(em_corners.max(axis=0)).astype(int)
    H, W = em_shape
    precrop = [int(x0), int(y0), int(x1), int(y1)]
    x0, x1 = np.clip([x0, x1], 0, W)
    y0, y1 = np.clip([y0, y1], 0, H)

    return dict(flip=flip, max_x=max_x, mims_tf=mims_tf, tps=tps,
                bbox=[int(x0), int(y0), int(x1), int(y1)],
                precrop=precrop, out_shape=out_shape)


def warp_isotope(src: np.ndarray, geom: dict) -> np.ndarray:
    """Warp one accumulated isotope image into EM space and crop to its bbox.

    Nearest-neighbour throughout: these are integer ion counts, and the gold
    detection downstream thresholds on raw count values. Interpolating would
    manufacture fractional counts and blur the threshold.
    """
    import cv2

    if geom["flip"]:
        src = src[:, ::-1]

    kw = dict(output_shape=geom["out_shape"], preserve_range=True, order=0)
    img = warp(src, geom["mims_tf"].inverse, **kw)
    img = warp(img, geom["tps"], **kw)

    # Warp a field of ones the same way, to know which pixels came from real data.
    ones = np.ones_like(src, np.uint8)
    mask = warp(ones, geom["mims_tf"].inverse, cval=0, **kw)
    mask = warp(mask, geom["tps"], cval=0, **kw)

    x0, y0, x1, y1 = geom["bbox"]
    px0, py0, px1, py1 = geom["precrop"]
    img = cv2.resize(img, (px1 - px0, py1 - py0), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (px1 - px0, py1 - py0), interpolation=cv2.INTER_NEAREST)

    xs, ys = max(0, x0 - px0), max(0, y0 - py0)
    img = img[ys:ys + (y1 - y0), xs:xs + (x1 - x0)]
    mask = mask[ys:ys + (y1 - y0), xs:xs + (x1 - x0)]

    out = np.clip(img, 0, 65535).astype(np.uint16)
    out[mask == 0] = 0
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = base_parser(__doc__.split("\n")[0])
    ap.add_argument("--canvas", nargs="*", help="canvas name(s); default all in paths.yaml")
    ap.add_argument("--accumulated", help="directory of accumulated count images (from step 01)")
    ap.add_argument("--landmarks", help="directory of landmark JSON files")
    ap.add_argument("--em", help="directory of full-resolution EM canvas images")
    ap.add_argument("--out", help="output directory for warped fields")
    args = ap.parse_args(argv)

    cfg = load_config(args.paths)
    acc_dir = need(cfg, "accumulated_dir", args.accumulated)
    lm_dir = need(cfg, "landmarks_dir", args.landmarks)
    em_dir = need(cfg, "em_canvas_dir", args.em)
    out_dir = need(cfg, "warped_dir", args.out)

    for name, meta in canvases(cfg, args.canvas).items():
        em_path = next(iter(em_dir.glob(f"{meta.get('em_image', name)}*")), None)
        if em_path is None:
            print(f"{name}: no EM image in {em_dir} — skipped", file=sys.stderr)
            continue
        em_shape = read_image(em_path).shape[:2]
        print(f"\n{name}  EM {em_shape[1]}x{em_shape[0]}")

        # Fields are associated with a canvas by the landmark directory layout:
        #   <landmarks_dir>/<canvas>/<field>.json
        canvas_lm = lm_dir / name
        if not canvas_lm.is_dir():
            print(f"{name}: no landmark directory {canvas_lm} — skipped", file=sys.stderr)
            continue

        for lm_path in sorted(canvas_lm.glob("*.json")):
            field_dir = acc_dir / lm_path.stem
            if not field_dir.is_dir():
                print(f"  no accumulated images for {lm_path.stem}", file=sys.stderr)
                continue
            isos = sorted(field_dir.glob("*.tif")) + sorted(field_dir.glob("*.png"))
            if not isos:
                continue
            try:
                mims_xy, em_xy = load_landmarks(lm_path)
                ref_shape = read_image(isos[0]).shape[:2]
                geom = registration_geometry(mims_xy, em_xy, ref_shape, em_shape)
            except Exception as exc:
                print(f"  FAIL {field_dir.name}: {exc}", file=sys.stderr)
                continue

            dst_dir = out_dir / name / field_dir.name
            for iso_path in isos:
                write_image(dst_dir / f"{iso_path.stem}.tif",
                            warp_isotope(read_image(iso_path), geom))
            (dst_dir / "placement.json").write_text(
                json.dumps({"bbox": geom["bbox"], "mirrored": bool(geom["flip"]),
                            "n_landmarks": int(len(mims_xy))}, indent=2),
                encoding="utf8",
            )
            print(f"  {field_dir.name}: {len(isos)} isotopes, bbox={geom['bbox']}, "
                  f"{len(mims_xy)} landmarks{', mirrored' if geom['flip'] else ''}")

    print(f"\nWarped fields written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

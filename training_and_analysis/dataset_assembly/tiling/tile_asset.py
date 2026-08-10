"""Tile one EM asset into the pretraining tile layout.

    python tile_asset.py --asset path/to/image.tif --out tiles/
    python tile_asset.py --asset-list assets.csv --out tiles/

Accepts 2D images (PNG, TIFF) and 3D stacks (multi-page or ome-TIFF). Writes one
uint8 PNG per accepted tile plus a JSON sidecar carrying the tile record, under
    <out>/tiles/source_id=<source_id>/

`build_manifest.py` reads that layout directly, so no shared index is written here.

Two passes over the asset. Pass one scores every candidate tile on every selected
plane and applies the per-source cap; pass two re-reads the planes and writes only
the tiles that survived. Planes are cropped to their non-zero content identically
in both passes so coordinates address the same pixels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tile_export.config import TileExportConfig, stable_json  # noqa: E402
from tile_export.filtering import (  # noqa: E402
    build_plane_scorer,
    crop_to_content,
    score_to_json,
    tile_status,
)
from tile_export.identity import make_tile_id, seeded_digest  # noqa: E402
from tile_export.normalization import (  # noqa: E402
    effective_normalization_scope,
    estimate_from_record,
    estimate_percentile_normalization,
    normalize_window_to_uint8,
    tile_uint8_stats,
)
from tile_export.selection import apply_source_cap  # noqa: E402
from tile_export.tiling import (  # noqa: E402
    NominalTile,
    evenly_spaced_indices,
    iter_candidate_tiles,
    sliding_window_starts,
)

Image.MAX_IMAGE_PIXELS = None  # gigapixel source images are normal here

# A chosen tile may be shifted off its nominal grid position to catch more tissue.
# This caps how much two kept tiles may then overlap, tighter than the primitives'
# own default, so shifting cannot collapse neighbours onto each other.
OVERLAP_CAP = 0.40

# Minimum physical z-spacing between planes selected from a 3D volume. Adjacent
# planes in a dense volume are near-duplicates and add little view diversity.
MIN_Z_SPACING_NM = 200.0


# --------------------------------------------------------------------------- io
def to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in (1, 2, 3, 4) else (
            arr[0] if arr.shape[0] in (1, 2, 3, 4) else arr[..., 0]
        )
    elif arr.ndim > 3:
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr[..., 0]
    return arr


class PlaneReader:
    """Reads planes from a 2D image or a 3D stack. Caches the single 2D plane."""

    def __init__(self, path: Path, is_3d: bool):
        self.path = Path(path)
        self.is_3d = is_3d
        self._tif = None
        self._zarr = None
        self._plane_2d = None

    def __enter__(self):
        if self.is_3d:
            import zarr

            self._tif = tifffile.TiffFile(str(self.path))
            self._zarr = zarr.open(self._tif.series[0].aszarr(), mode="r")
        return self

    def __exit__(self, *exc):
        if self._tif is not None:
            self._tif.close()
        return False

    @property
    def depth(self) -> int:
        if not self.is_3d:
            return 1
        return int(self._zarr.shape[0])

    def read(self, index: int) -> np.ndarray:
        if self.is_3d:
            return to_2d(np.asarray(self._zarr[index]))
        if self._plane_2d is None:
            if self.path.suffix.lower() == ".png":
                im = Image.open(self.path)
                if im.mode in ("RGB", "RGBA", "P", "LA"):
                    im = im.convert("L")
                self._plane_2d = to_2d(np.asarray(im))
            else:
                self._plane_2d = to_2d(tifffile.imread(str(self.path)))
        return self._plane_2d


def looks_3d(path: Path) -> bool:
    if path.suffix.lower() not in (".tif", ".tiff"):
        return False
    try:
        with tifffile.TiffFile(str(path)) as tf:
            return len(tf.series[0].shape) >= 3 and tf.series[0].shape[0] > 1
    except Exception:
        return False


# ------------------------------------------------------------------- geometry
def make_thumb(arr: np.ndarray, max_size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    step = max(1, math.ceil(max(h, w) / float(max_size)))
    return arr[::step, ::step]


def crop_plane(arr: np.ndarray):
    """Crop to non-zero content. Deterministic, so both passes crop identically."""
    cropped, x_off, y_off, orig_w, orig_h = crop_to_content(arr)
    if x_off == 0 and y_off == 0 and cropped.shape[1] == orig_w and cropped.shape[0] == orig_h:
        return cropped, None
    return cropped, {
        "x_off": int(x_off), "y_off": int(y_off),
        "orig_width": int(orig_w), "orig_height": int(orig_h),
        "cropped_width": int(cropped.shape[1]), "cropped_height": int(cropped.shape[0]),
    }


def nominal_tiles(width: int, height: int, cfg: TileExportConfig):
    xs = sliding_window_starts(width, cfg.tile_size, cfg.stride, OVERLAP_CAP)
    ys = sliding_window_starts(height, cfg.tile_size, cfg.stride, OVERLAP_CAP)
    tiles, idx = [], 0
    for y in ys:
        for x in xs:
            idx += 1
            tiles.append(NominalTile(
                index=idx, nominal_x=int(x), nominal_y=int(y),
                width=min(cfg.tile_size, max(1, int(width) - int(x))),
                height=min(cfg.tile_size, max(1, int(height) - int(y))),
            ))
    return tiles, len(xs) * len(ys)


def _overlaps_chosen(cand, chosen, z) -> bool:
    for (cz, cx, cy, cw, ch) in chosen:
        if cz != z:
            continue
        ox = max(0, min(cand.x + cand.width, cx + cw) - max(cand.x, cx))
        oy = max(0, min(cand.y + cand.height, cy + ch) - max(cand.y, cy))
        smaller = min(cand.width * cand.height, cw * ch)
        if smaller > 0 and (ox * oy) / smaller > OVERLAP_CAP:
            return True
    return False


def choose_candidate(nominal, scorer, *, z, source_id, chosen, cfg):
    """Pick the best-scoring shift of one nominal tile that does not over-overlap a kept tile."""
    candidates = iter_candidate_tiles(
        nominal, image_width=scorer.image_width, image_height=scorer.image_height,
        tile_size=cfg.tile_size, max_shift=cfg.max_shift_px,
    )
    scored = [(c, scorer.score_window(x=c.x, y=c.y, width=c.width, height=c.height))
              for c in candidates]
    scored.sort(
        key=lambda it: (
            it[1].tissue_score,
            it[1].non_background_fraction,
            -abs(it[0].shift_x) - abs(it[0].shift_y),
            seeded_digest(f"{source_id}:{z}:{it[0].x}:{it[0].y}", seed=cfg.seed),
        ),
        reverse=True,
    )
    for cand, score in scored:
        key = (z, cand.x, cand.y, cand.width, cand.height)
        if key in chosen or _overlaps_chosen(cand, chosen, z):
            continue
        chosen.add(key)
        return cand, score
    return None


def _normalization_rejections(normalization, cfg) -> list[str]:
    reasons = []
    warnings = set(str(normalization.normalization_warning or "").split(";"))
    if "insufficient_valid_support" in warnings:
        reasons.append("insufficient_valid_support")
    if normalization.low_dynamic_range and not cfg.allow_low_dynamic_range:
        reasons.append("low_dynamic_range")
    return reasons


# --------------------------------------------------------------------- scoring
def score_plane(plane, *, z, source_index, source_id, content_crop, cfg):
    arr = to_2d(plane)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    thumb = make_thumb(arr, cfg.thumbnail_max_size)
    scope = effective_normalization_scope(config=cfg, is_3d=z is not None)
    normalization = estimate_percentile_normalization(
        thumb, config=cfg, raw_dtype=arr.dtype, scope=scope, inverted=False,
        estimation_method_prefix="thumbnail",
    )
    scorer = build_plane_scorer(thumb, image_width=width, image_height=height, config=cfg)

    records, chosen = [], set()
    for nominal in nominal_tiles(width, height, cfg)[0]:
        picked = choose_candidate(nominal, scorer, z=z, source_id=source_id,
                                  chosen=chosen, cfg=cfg)
        if picked is None:
            continue
        candidate, score = picked
        status = tile_status(score, config=cfg)
        reasons = list(score.reasons)
        for reason in _normalization_rejections(normalization, cfg):
            status = "rejected"
            if reason not in reasons:
                reasons.append(reason)
        records.append({
            "tile_id": make_tile_id(
                source_id=source_id, asset_id=source_id, z=z,
                x=candidate.x, y=candidate.y, tile_size=cfg.tile_size,
                effective_nm_per_px=None, normalization=normalization.normalization_hash,
            ),
            "source_id": source_id,
            "z": z,
            "z_source_index": source_index,
            "nominal_index": nominal.index,
            "nominal_x": nominal.nominal_x,
            "nominal_y": nominal.nominal_y,
            "x": candidate.x, "y": candidate.y,
            "width": candidate.width, "height": candidate.height,
            "tile_size": cfg.tile_size,
            "stride": cfg.stride,
            "overlap_fraction": cfg.overlap_fraction,
            "overlap_cap": OVERLAP_CAP,
            "shift_x": candidate.shift_x, "shift_y": candidate.shift_y,
            "content_crop": content_crop,
            "status": status,
            "rejection_reason": "" if status == "accepted" else ",".join(reasons),
            "normalization": normalization.sidecar_payload(),
            "normalization_hash": normalization.normalization_hash,
            "scoring": cfg.scoring,
            "tile_path": "",
            **normalization.flat_fields(),
            **score_to_json(score),
        })
    return records


def select_planes(reader: PlaneReader, cfg: TileExportConfig, z_nm: float | None):
    """Which planes of a 3D volume to tile.

    With a known z-spacing, walk the stack keeping planes at least MIN_Z_SPACING_NM
    apart. Without one, spread the plane budget evenly, where the budget is the
    per-source tile cap divided by the tiles one plane yields.
    """
    depth = reader.depth
    if not reader.is_3d:
        return [0]
    if z_nm and z_nm > 0:
        keep, last = [0], 0.0
        for i in range(1, depth):
            if i * z_nm - last >= MIN_Z_SPACING_NM:
                keep.append(i)
                last = i * z_nm
        return keep
    first = to_2d(reader.read(0))
    per_plane = max(1, nominal_tiles(first.shape[1], first.shape[0], cfg)[1])
    n = min(depth, max(1, cfg.max_tiles_per_source // per_plane))
    return list(evenly_spaced_indices(count=depth, selected_count=n))


# ----------------------------------------------------------------------- driver
def tile_one(asset: Path, out_root: Path, *, source_id: str, cfg: TileExportConfig,
             z_nm: float | None = None, dry_run: bool = False) -> dict:
    asset = Path(asset)
    is_3d = looks_3d(asset)
    result = {"source_id": source_id, "path": str(asset), "is_3d": is_3d,
              "accepted": 0, "reason": ""}

    with PlaneReader(asset, is_3d) as reader:
        planes = select_planes(reader, cfg, z_nm)
        result["planes_selected"] = len(planes)

        records, max_w, max_h = [], 1, 1
        for index in planes:
            arr, content_crop = crop_plane(to_2d(reader.read(index)))
            max_w, max_h = max(max_w, arr.shape[1]), max(max_h, arr.shape[0])
            records.extend(score_plane(
                arr, z=(index if is_3d else None), source_index=index,
                source_id=source_id, content_crop=content_crop, cfg=cfg,
            ))

        result["candidates"] = len(records)
        result["capped"] = apply_source_cap(
            records, image_width=max_w, image_height=max_h, config=cfg)
        accepted = [r for r in records if r["status"] == "accepted"]
        result["accepted"] = len(accepted)
        if not accepted:
            result["reason"] = _zero_reason(records, cfg)
            return result
        if dry_run:
            return result

        tile_dir = Path(out_root) / "tiles" / f"source_id={source_id}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        by_plane: dict[int, list] = {}
        for record in accepted:
            by_plane.setdefault(record["z_source_index"], []).append(record)

        for index, group in by_plane.items():
            plane, _ = crop_plane(to_2d(reader.read(index)))
            for record in group:
                tile_path = tile_dir / tile_filename(record)
                window = plane[record["y"]:record["y"] + record["height"],
                               record["x"]:record["x"] + record["width"]]
                tile = normalize_window_to_uint8(
                    window, normalization=estimate_from_record(record))
                Image.fromarray(tile, mode="L").save(
                    tile_path, compress_level=0, optimize=False)
                record.update(tile_uint8_stats(tile))
                record["tile_path"] = f"tiles/source_id={source_id}/{tile_path.name}"
                tile_path.with_suffix(".json").write_text(
                    json.dumps(record, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return result


def _zero_reason(records, cfg) -> str:
    if not records:
        return "no_candidate_tiles_too_small"
    reasons = {piece for r in records
               for piece in str(r.get("rejection_reason") or "").split(",") if piece}
    if "low_dynamic_range" in reasons:
        return "low_dynamic_range"
    return f"no_tiles>={cfg.min_tissue_fraction}_tissue"


def safe_name(value: str) -> str:
    return "".join(c if (c.isalnum() or c in {"-", "_"}) else "_" for c in str(value))


def tile_filename(record) -> str:
    return (f"{safe_name(record['source_id'])[:32]}"
            f"__z{int(record['z'] or 0):06d}"
            f"__y{int(record['y']):06d}__x{int(record['x']):06d}.png")


def load_asset_list(path: Path) -> list[tuple[Path, str, float | None]]:
    """CSV with a `path` column, plus optional `source_id` and `z_nm`."""
    out = []
    with Path(path).open(encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            asset = Path(row["path"]).expanduser()
            z_nm = row.get("z_nm") or ""
            out.append((asset,
                        row.get("source_id") or asset.stem,
                        float(z_nm) if z_nm.strip() else None))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--asset", help="a single image file to tile")
    src.add_argument("--asset-list", help="CSV of assets: path[,source_id][,z_nm]")
    ap.add_argument("--out", required=True, help="output root; tiles land under <out>/tiles/")
    ap.add_argument("--source-id", help="identifier for a single asset (default: filename stem)")
    ap.add_argument("--z-nm", type=float, help="z-spacing in nm for a 3D stack, if known")
    ap.add_argument("--dry-run", action="store_true", help="score and cap without writing tiles")
    args = ap.parse_args(argv)

    cfg = TileExportConfig()
    cfg.validate()
    out_root = Path(args.out).expanduser()

    if args.asset:
        asset = Path(args.asset).expanduser()
        assets = [(asset, args.source_id or asset.stem, args.z_nm)]
    else:
        assets = load_asset_list(Path(args.asset_list).expanduser())

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "tiler_config.json").write_text(
        json.dumps(cfg.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")

    total = 0
    for asset, source_id, z_nm in assets:
        if not asset.exists():
            print(f"{source_id}: missing {asset}", file=sys.stderr)
            continue
        try:
            res = tile_one(asset, out_root, source_id=source_id, cfg=cfg,
                           z_nm=z_nm, dry_run=args.dry_run)
        except Exception as exc:  # one bad asset should not stop a batch
            print(f"{source_id}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        total += res["accepted"]
        note = f"  ({res['reason']})" if res["reason"] else ""
        capped = f", {res['capped']} over cap" if res.get("capped") else ""
        print(f"{source_id:40s} {'3D' if res['is_3d'] else '2D'} "
              f"{res.get('planes_selected', 1):4d} plane(s)  "
              f"{res['candidates']:5d} candidates -> {res['accepted']:4d} accepted{capped}{note}")

    print(f"\n{total} tiles under {out_root / 'tiles'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

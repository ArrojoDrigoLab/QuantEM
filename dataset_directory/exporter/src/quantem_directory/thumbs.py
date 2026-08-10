"""Render one thumbnail per asset.

An electron micrograph does not survive being shrunk whole. A 20,000-pixel
montage reduced to 256 pixels is a uniform grey rectangle — technically the
image, visually nothing. So a thumbnail here is a *representative crop*: the
single tile of that asset with the highest tissue content, taken from the tile
corpus that was already cut and scored for pretraining. At 256 px a 2048 px
tile still resolves membranes and organelles, which is what makes a grid of
these worth looking at.

Sources, in the order tried:

1. The highest-scoring canonical tile. Covers almost the whole corpus.
2. A previously rendered preview PNG, matched on asset id.
3. The largest raster the asset has on disk, downsampled. Slow, and only worth
   running for the handful of assets the first two miss.

Reading is over a network share and dominates the runtime, so the work is
threaded rather than forked — the bottleneck is bytes, not CPU.
"""
from __future__ import annotations

import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Reject sources too small to say anything, or too flat to be tissue. Both
#: guard against scraped placeholder images that reached the preview folders —
#: licence badges, spacer GIFs, and 10x12 px stubs.
MIN_SOURCE_SIDE = 96
MIN_SOURCE_STDDEV = 3.0

#: File types worth opening when falling back to an asset's own raster.
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

_print_lock = threading.Lock()


#: How many runner-up tiles to keep per asset. The tile index is a point-in-time
#: build and some of the files it lists have since been moved or removed, so the
#: single best tile is not always still on disk. Assets typically have hundreds
#: of tiles, so a few alternates recover essentially all of them.
CANDIDATES_PER_ASSET = 6


def _pick_best_tiles(index_csv: Path, wanted: set) -> Dict[str, List[Tuple[str, str]]]:
    """Rank each asset's tiles by tissue content, then contrast, best first."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    ranked: Dict[str, List[Tuple[float, float, str, str]]] = {}
    with index_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            asset_id = row["canonical_asset_id"]
            if asset_id not in wanted or row.get("png_exists") != "True":
                continue
            try:
                score = float(row.get("tissue_score") or 0.0)
                contrast = float(row.get("tile_std_uint8") or 0.0)
            except ValueError:
                continue
            bucket = ranked.setdefault(asset_id, [])
            bucket.append((score, contrast, row["run"], row["rel_path"]))
            # Keep the bucket small as we stream, rather than sorting 364k rows.
            if len(bucket) > CANDIDATES_PER_ASSET * 4:
                bucket.sort(key=lambda c: c[:2], reverse=True)
                del bucket[CANDIDATES_PER_ASSET:]
    out: Dict[str, List[Tuple[str, str]]] = {}
    for asset_id, bucket in ranked.items():
        bucket.sort(key=lambda c: c[:2], reverse=True)
        out[asset_id] = [(run, rel) for _, _, run, rel in bucket[:CANDIDATES_PER_ASSET]]
    return out


def _stretch(image, low_pct: float = 1.0, high_pct: float = 99.0):
    """Rescale intensities so the middle of the histogram spans the full range.

    Raw microscope rasters are frequently stored over a narrow slice of the
    available range — genuine images whose pixel values run from, say, 17 to 40.
    Displayed as-is they are a flat grey rectangle. Clipping at percentiles
    rather than at the extrema keeps a few hot or dead pixels from undoing the
    stretch.

    Returns ``(image, span)``; a span near zero means the source really is blank.
    """
    histogram = image.histogram()
    total = sum(histogram)
    if not total:
        return image, 0
    low_target = total * low_pct / 100.0
    high_target = total * high_pct / 100.0
    running = 0
    low = 0
    high = 255
    for value, count in enumerate(histogram):
        running += count
        if running <= low_target:
            low = value
        if running <= high_target:
            high = value
    span = high - low
    if span < 2:
        return image, span
    scale = 255.0 / span
    return image.point(lambda v: max(0, min(255, int((v - low) * scale)))), span


def _render(source: Path, target: Path, *, px: int, quality: int, normalize: bool) -> Optional[str]:
    """Downscale ``source`` into ``target``. Returns a rejection reason, or None.

    ``normalize`` stretches the histogram. Canonical tiles are left alone: the
    tiling pipeline already normalized them using percentiles estimated per
    *source*, so tiles of one asset are consistent with each other, and
    re-stretching each tile independently would undo that. Fallback rasters have
    had no such treatment and need it.
    """
    from PIL import Image, ImageStat

    # Whole-asset rasters legitimately exceed the decompression-bomb threshold;
    # the file-size guard in the caller is what actually bounds this.
    Image.MAX_IMAGE_PIXELS = None

    try:
        with Image.open(source) as image:
            # Cheap for JPEG: decode straight to a smaller size.
            if image.format == "JPEG":
                image.draft("L", (px * 2, px * 2))
            if max(image.size) < MIN_SOURCE_SIDE:
                return f"source too small ({image.size[0]}x{image.size[1]})"
            if image.mode in ("I", "I;16", "I;16B", "F"):
                # Higher bit depths need a real rescale; PIL's plain convert
                # truncates and leaves a near-black image.
                extrema = image.getextrema()
                spread = (extrema[1] - extrema[0]) or 1
                image = image.point(lambda v: (v - extrema[0]) * 255.0 / spread).convert("L")
            else:
                image = image.convert("L")

            image.thumbnail((px, px), Image.LANCZOS)

            if normalize:
                image, span = _stretch(image)
                if span < 2:
                    return "source is blank"
            elif ImageStat.Stat(image).stddev[0] < MIN_SOURCE_STDDEV:
                return "source has no contrast"

            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, "WEBP", quality=quality, method=6)
    except Exception as error:  # noqa: BLE001 — one bad file must not stop the run
        return f"{type(error).__name__}: {error}"
    return None


def pack(args) -> int:
    """Bundle the rendered thumbnails into one release archive.

    Thumbnails never enter git history — every regeneration would add hundreds
    of megabytes of permanent, incompressible blobs to every future clone.
    They ship as a release asset instead, and the deploy job unpacks them into
    the published site. The checksum recorded here is what that job verifies.
    """
    import hashlib
    import tarfile

    out_dir = Path(args.out)
    index_path = out_dir / "index.json"
    if not index_path.exists():
        print(f"no thumbnail index at {index_path} — run 'thumbs' first", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text(encoding="utf-8"))

    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(out_dir.glob("*/*.webp"))
    print(f"packing {len(files)} thumbnails into {archive.name}")
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        # WebP is already compressed; gzip here is for a single-file transfer,
        # not for size, so the cheapest level is the right one.
        for path in files:
            tar.add(path, arcname=f"{path.parent.name}/{path.name}")

    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{checksum}  {archive.name}\n", encoding="utf-8"
    )

    index["meta"].update(
        {
            "archive": archive.name,
            "sha256": checksum,
            "release_tag": args.release_tag,
            "archive_bytes": archive.stat().st_size,
        }
    )
    index["meta"].pop("interim", None)
    index_path.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")

    print(
        f"{archive} — {archive.stat().st_size / 1e6:.1f} MB\n"
        f"sha256 {checksum}\n\n"
        f"Next: attach it to release '{args.release_tag}', then re-run 'build' so the\n"
        f"manifest records the tag and checksum the deploy job verifies."
    )
    return 0


def run(args) -> int:
    """Render the thumbnail set described by ``args`` and write ``index.json``."""
    from . import extract as extract_module

    out_dir = Path(args.out)
    tile_root = Path(args.tile_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = extract_module.load(Path(args.extract))
    assets = {a.id: a for a in corpus.assets}
    print(f"corpus: {len(assets)} assets")

    chosen = _pick_best_tiles(Path(args.tile_index), set(assets))
    print(f"canonical tiles available for {len(chosen)} assets")

    preview_dirs = [Path(p) for p in (args.fallback_preview or [])]
    images_root = Path(args.fallback_images) if args.fallback_images else None
    max_bytes = args.max_source_mb * 1_000_000

    def sources_for(asset_id: str) -> List[Tuple[Path, bool]]:
        """Every source worth trying, best first, paired with "is a canonical tile"."""
        candidates = []
        # The index stores the run separately from the path relative to it.
        for run_name, rel_path in chosen.get(asset_id, ()):
            candidates.append((tile_root / run_name / rel_path, True))
        hex_id = asset_id.replace("-", "")
        for directory in preview_dirs:
            candidates.append((directory / f"{asset_id}.png", False))
            candidates.append((directory / f"{hex_id}.png", False))
        # Last tier: the asset's own raster. Slow — these run to hundreds of
        # megabytes — so it is only reached by assets the tile corpus missed.
        if images_root:
            for name in (asset_id, hex_id):
                folder = images_root / name
                if not folder.is_dir():
                    continue
                for path in sorted(folder.iterdir()):
                    if path.suffix.lower() in RASTER_SUFFIXES and path.stat().st_size <= max_bytes:
                        candidates.append((path, False))
                        break
        return candidates

    def rescan_tile_directories(asset_id: str) -> List[Tuple[Path, bool]]:
        """Last resort: take any tile the asset still has on disk.

        Some assets were re-tiled after the index was built, so the index names
        tiles at coordinates that no longer exist while the directory holds a
        perfectly good set at different ones. Any tile of the asset is still a
        representative crop, so prefer one of those over no thumbnail at all.
        """
        found: List[Tuple[Path, bool]] = []
        seen = set()
        for run_name, rel_path in chosen.get(asset_id, ()):
            directory = (tile_root / run_name / rel_path).parent
            if directory in seen or not directory.is_dir():
                continue
            seen.add(directory)
            found.extend((path, True) for path in sorted(directory.glob("*.png"))[:2])
        return found

    # A targeted re-render list, used to repair thumbnails an audit flagged as
    # too flat to be worth looking at.
    only_ids = None
    if getattr(args, "only_ids", None):
        text = Path(args.only_ids).read_text(encoding="utf-8")
        only_ids = {line.strip().replace("-", "") for line in text.split() if line.strip()}
        print(f"restricted to {len(only_ids)} assets")

    todo: List[str] = []
    for asset_id in assets:
        hex_id = asset_id.replace("-", "")
        if only_ids is not None and hex_id not in only_ids:
            continue
        target = out_dir / hex_id[:2] / f"{hex_id}.webp"
        if args.only_missing and only_ids is None and target.exists():
            continue
        todo.append(asset_id)
    if args.limit:
        todo = todo[: args.limit]
    print(f"to render: {len(todo)}")

    done = {"ok": 0, "skipped": 0, "failed": 0}
    reasons: Dict[str, str] = {}

    def record(outcome: str, hex_id: str, reason: Optional[str] = None) -> None:
        with _print_lock:
            done[outcome] += 1
            if reason:
                reasons[hex_id] = reason
            total = done["ok"] + done["skipped"] + done["failed"]
            if total % 500 == 0 or total == len(todo):
                print(
                    f"  {total}/{len(todo)}  ok={done['ok']} skip={done['skipped']} fail={done['failed']}",
                    flush=True,
                )

    def work(asset_id: str) -> None:
        hex_id = asset_id.replace("-", "")
        target = out_dir / hex_id[:2] / f"{hex_id}.webp"
        candidates = sources_for(asset_id)
        if not candidates:
            record("skipped", hex_id, "no local pixel source")
            return
        problem = None
        for source, is_tile in candidates + rescan_tile_directories(asset_id):
            if not source.exists():
                problem = f"missing: {source.name}"
                continue
            problem = _render(
                source,
                target,
                px=args.px,
                quality=args.quality,
                normalize=(not is_tile) or getattr(args, "normalize_tiles", False),
            )
            if problem is None:
                record("ok", hex_id)
                return
        record("failed", hex_id, problem or "no candidate could be read")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))

    rendered = sorted(
        path.stem for path in out_dir.glob("*/*.webp") if path.stem in {a.replace("-", "") for a in assets}
    )
    total_bytes = sum(path.stat().st_size for path in out_dir.glob("*/*.webp"))
    index = {
        "meta": {
            "px": args.px,
            "format": "webp",
            "quality": args.quality,
            "count": len(rendered),
            "bytes": total_bytes,
        },
        "assets": rendered,
    }
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    if reasons:
        (out_dir / "unrendered.json").write_text(json.dumps(reasons, indent=2), encoding="utf-8")

    print(
        f"\nrendered {done['ok']}, skipped {done['skipped']}, failed {done['failed']}\n"
        f"{len(rendered)} thumbnails on disk, {total_bytes / 1e6:.1f} MB"
    )
    return 0

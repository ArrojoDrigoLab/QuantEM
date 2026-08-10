"""Render thumbnails for assets whose source had to be downloaded by hand.

Almost every asset's thumbnail comes from the tile corpus or from the source
repository directly. A few repositories cannot be fetched by script — Dryad, for
one, sits behind an interactive anti-bot check that should be respected rather
than worked around — so their files have to be downloaded through a browser.

Point this at the folder you downloaded them into. Files are matched to assets
by name, so nothing is guessed: an asset only gets a thumbnail if a file
plausibly *is* that asset.

    python tools/thumbs_from_folder.py \\
        --folder /path/to/downloads \\
        --extract /path/to/corpus-extract \\
        --out ../thumbs \\
        --dataset "Data from: Hepatic steatosis induced by nicotine"

Add --dry-run first to see what would match what.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantem_directory import extract as extract_module  # noqa: E402
from quantem_directory.thumbs import _render  # noqa: E402

SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True, help="directory of downloaded source images")
    parser.add_argument("--extract", required=True, help="corpus extract directory")
    parser.add_argument("--out", required=True, help="thumbnail directory")
    parser.add_argument("--dataset", help="only consider assets of this dataset (prefix match)")
    parser.add_argument("--px", type=int, default=256)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    corpus = extract_module.load(Path(args.extract))
    datasets = {d.id: d.name for d in corpus.datasets}
    assets = [
        a for a in corpus.assets
        if not args.dataset or datasets.get(a.dataset_id, "").startswith(args.dataset)
    ]
    print(f"{len(assets)} assets in scope")

    # Loose image files, plus anything inside a ZIP sitting in the folder —
    # several repositories hand you one archive and nothing else.
    index: dict = {}
    zips: dict = {}
    files = [p for p in Path(args.folder).rglob("*") if p.suffix.lower() in SUFFIXES]
    for path in files:
        index.setdefault(normalise(path.name), path)
        index.setdefault(normalise(path.stem), path)

    for archive_path in Path(args.folder).rglob("*.zip"):
        try:
            archive = zipfile.ZipFile(archive_path)
        except zipfile.BadZipFile:
            print(f"  skipping unreadable archive {archive_path.name}")
            continue
        for name in archive.namelist():
            leaf = name.rsplit("/", 1)[-1]
            if not leaf or leaf.startswith("._") or "__MACOSX" in name:
                continue
            if Path(leaf).suffix.lower() not in SUFFIXES:
                continue
            for form in (normalise(leaf), normalise(Path(leaf).stem)):
                zips.setdefault(form, (archive, name))
    print(f"{len(files)} loose images, {len(zips)} images inside archives")

    out = Path(args.out)
    rendered = unmatched = 0
    for asset in assets:
        hex_id = asset.id.replace("-", "")
        target = out / hex_id[:2] / f"{hex_id}.webp"
        if target.exists():
            continue
        keys = (normalise(asset.name), normalise(asset.name.rsplit(".", 1)[0]))
        source = next((index[k] for k in keys if k in index), None)
        member = next((zips[k] for k in keys if k in zips), None) if source is None else None
        if source is None and member is None:
            unmatched += 1
            print(f"  no file for {asset.name!r}")
            continue

        label = source.name if source else member[1].rsplit("/", 1)[-1]
        if args.dry_run:
            print(f"  {asset.name[:44]:44s} <- {label}")
            rendered += 1
            continue

        # Archive members are extracted to a temporary file so the renderer
        # sees a path, then discarded.
        scratch = None
        if source is None:
            archive, name = member
            scratch = Path(tempfile.mkstemp(suffix=Path(name).suffix, dir=args.folder)[1])
            scratch.write_bytes(archive.read(name))
            source = scratch
        try:
            # Raw repository files, never pre-normalised tiles, so they always
            # get the histogram stretch.
            problem = _render(source, target, px=args.px, quality=args.quality, normalize=True)
        finally:
            if scratch is not None:
                scratch.unlink(missing_ok=True)
        print(f"  {asset.name[:44]:44s} <- {label[:34]:34s} {problem or 'ok'}")
        if not problem:
            rendered += 1

    print(f"\n{'would render' if args.dry_run else 'rendered'} {rendered}, unmatched {unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

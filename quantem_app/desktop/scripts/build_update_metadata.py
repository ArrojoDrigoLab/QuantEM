"""Create Tauri's signed static-update metadata from release build manifests."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _read_manifests(assets_dir: Path, version: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for path in sorted(assets_dir.glob("*.update-manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        platform = str(payload.get("platform") or "")
        filename = str(payload.get("update_filename") or "")
        signature_filename = str(payload.get("signature_filename") or "")
        if not platform or not filename or not signature_filename:
            raise ValueError(f"invalid updater manifest: {path.name}")
        if platform in entries:
            raise ValueError(f"duplicate updater platform {platform!r}")
        if f"_{version}_" not in filename and f"-{version}-" not in filename:
            raise ValueError(f"update asset {filename!r} does not identify version {version}")
        signature_path = assets_dir / signature_filename
        if not signature_path.is_file():
            raise ValueError(f"missing updater signature {signature_filename!r}")
        signature = signature_path.read_text(encoding="utf-8").strip()
        if not signature:
            raise ValueError(f"empty updater signature {signature_filename!r}")
        entries[platform] = {"filename": filename, "signature": signature}
    if set(entries) != {"windows-x86_64", "darwin-x86_64", "darwin-aarch64"}:
        raise ValueError(
            "release must include signed updater assets for Windows x64, macOS Intel, and macOS Apple Silicon"
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    platforms = _read_manifests(args.assets_dir, args.version)
    tag = args.tag.strip()
    if tag != f"v{args.version}":
        raise SystemExit(f"tag {tag!r} does not match version {args.version!r}")

    payload = {
        "version": args.version,
        "notes": args.notes_file.read_text(encoding="utf-8").strip(),
        "pub_date": datetime.now(UTC).isoformat(),
        "platforms": {
            platform: {
                "url": (
                    "https://github.com/ArrojoDrigoLab/QuantEM/releases/download/"
                    f"{tag}/{entry['filename']}"
                ),
                "signature": entry["signature"],
            }
            for platform, entry in sorted(platforms.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

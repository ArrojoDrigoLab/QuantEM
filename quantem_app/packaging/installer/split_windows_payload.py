"""Split an oversized runtime payload below GitHub's 2 GiB asset limit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DEFAULT_PART_BYTES = 1_900_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_payload(
    *,
    archive: Path,
    manifest_path: Path,
    max_part_bytes: int = DEFAULT_PART_BYTES,
    remove_archive: bool = False,
) -> list[dict[str, object]]:
    if max_part_bytes <= 0 or max_part_bytes >= 2 * 1024**3:
        raise ValueError("part size must be positive and strictly below GitHub's 2 GiB limit")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if archive.name != payload.get("filename"):
        raise ValueError("archive filename does not match its payload manifest")
    if archive.stat().st_size != payload.get("size") or _sha256(archive) != payload.get("sha256"):
        raise ValueError("archive does not match its payload manifest")

    if archive.stat().st_size <= max_part_bytes:
        parts = [{"filename": archive.name, "sha256": payload["sha256"], "size": payload["size"]}]
    else:
        part_count = math.ceil(archive.stat().st_size / max_part_bytes)
        if part_count > 4:
            raise ValueError("payload requires more than four release assets; raise the part size")
        parts = []
        with archive.open("rb") as source:
            for index in range(1, part_count + 1):
                part_path = archive.with_name(f"{archive.name}.part{index:02d}")
                digest = hashlib.sha256()
                written = 0
                with part_path.open("wb") as target:
                    while written < max_part_bytes:
                        chunk = source.read(min(8 * 1024 * 1024, max_part_bytes - written))
                        if not chunk:
                            break
                        target.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                parts.append(
                    {"filename": part_path.name, "sha256": digest.hexdigest(), "size": written}
                )
        if remove_archive:
            archive.unlink()

    payload["parts"] = parts
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-part-bytes", type=int, default=DEFAULT_PART_BYTES)
    parser.add_argument("--remove-archive", action="store_true")
    args = parser.parse_args()
    split_payload(
        archive=args.archive,
        manifest_path=args.manifest,
        max_part_bytes=args.max_part_bytes,
        remove_archive=args.remove_archive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

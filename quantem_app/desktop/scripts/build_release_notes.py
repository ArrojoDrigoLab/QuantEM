"""Put the user-facing installers ahead of GitHub's flat release asset list."""

from __future__ import annotations

import argparse
from pathlib import Path

REPOSITORY = "ArrojoDrigoLab/QuantEM"


def build_release_notes(*, version: str, tag: str, generated_notes: str) -> str:
    version = version.strip()
    tag = tag.strip()
    if tag != f"v{version}":
        raise ValueError(f"tag {tag!r} does not match version {version!r}")

    base = f"https://github.com/{REPOSITORY}/releases/download/{tag}"
    windows = f"QuantEM_{version}_x64-setup.exe"
    apple_silicon = f"QuantEM_{version}_darwin-aarch64.dmg"
    intel = f"QuantEM_{version}_darwin-x86_64.dmg"
    changes = generated_notes.strip() or "No additional release notes."

    return f"""## Install QuantEM

[Download for Windows 10/11 (64-bit) (.exe)]({base}/{windows})

The installer detects compatible NVIDIA hardware and selects the CUDA runtime when appropriate; CPU mode remains available.

[Download for macOS (Apple silicon) (.dmg)]({base}/{apple_silicon})

Use this for Macs with an Apple chip.

[Download for macOS (Intel) (.dmg)]({base}/{intel})

Use this for Intel-based Macs.

## Changes

{changes}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--generated-notes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    notes = build_release_notes(
        version=args.version,
        tag=args.tag,
        generated_notes=args.generated_notes.read_text(encoding="utf-8"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(notes, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

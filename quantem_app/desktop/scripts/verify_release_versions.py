"""Fail a desktop release unless every public version source agrees."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]


def _match(path: Path, pattern: str, label: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"could not read {label} from {path}")
    return match.group(1)


def versions() -> dict[str, str]:
    project = tomllib.loads((APP_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((APP_ROOT / "desktop/src-tauri/Cargo.toml").read_text(encoding="utf-8"))
    tauri = json.loads((APP_ROOT / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    return {
        "pyproject": str(project["project"]["version"]),
        "cargo": str(cargo["package"]["version"]),
        "tauri": str(tauri["version"]),
        "frozen_fallback": _match(
            APP_ROOT / "src/quantem/_version.py",
            r'FALLBACK_VERSION\s*=\s*"([^"]+)"',
            "frozen fallback version",
        ),
        "conda": _match(
            APP_ROOT / "packaging/conda/meta.yaml",
            r'set version\s*=\s*"([^"]+)"',
            "conda version",
        ),
    }


def verify(tag: str | None = None) -> str:
    found = versions()
    expected = found["pyproject"]
    mismatches = {name: value for name, value in found.items() if value != expected}
    if mismatches:
        detail = ", ".join(f"{name}={value}" for name, value in found.items())
        raise ValueError(f"release versions disagree: {detail}")
    if tag is not None and tag != f"v{expected}":
        raise ValueError(f"release tag {tag!r} does not match v{expected}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()
    print(verify(args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

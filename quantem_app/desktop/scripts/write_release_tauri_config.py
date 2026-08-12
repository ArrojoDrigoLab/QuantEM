"""Materialize the Tauri release overlay without committing signing material.

The updater public key is not secret, but it is intentionally provisioned by
the protected release environment.  That lets the repository be bootstrapped
without accidentally pinning a throwaway development key that could never sign
a real release.  The private signing key is read directly by Tauri from the
CI environment and is never passed on this script's command line.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PLACEHOLDER = "__TAURI_UPDATER_PUBLIC_KEY__"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public_key = os.environ.get("TAURI_UPDATER_PUBKEY", "").strip()
    if not public_key:
        raise SystemExit("TAURI_UPDATER_PUBKEY must be set by the protected release environment.")

    template = args.template.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise SystemExit("release Tauri configuration has an invalid public-key placeholder.")

    rendered = template.replace(PLACEHOLDER, public_key)
    # Parse before writing so an invalid public key can never leave a malformed
    # config that Tauri reports later as an unrelated bundling failure.
    json.loads(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

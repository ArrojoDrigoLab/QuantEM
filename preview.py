"""Browse the published site locally, before pushing.

The Pages workflow assembles one folder out of several that live apart in the
working tree:

    /                            pages/
    /dataset_directory/          dataset_directory/site/
    /dataset_directory/data/     dataset_directory/data/
    /dataset_directory/thumbs/   dataset_directory/thumbs/   (unpacked, ignored by git)

This serves that layout by mapping URLs onto those directories, rather than
copying 300 MB into a dist/ every time you want to look at the page.

    python preview.py                    # http://localhost:45176/
    python preview.py --base /QuantEM    # http://localhost:45176/QuantEM/

``--base`` mirrors the path prefix a project site is published under
(``arrojodrigolab.github.io/QuantEM/``). Every link on the site is relative, so
it should make no difference — which is exactly why it is worth checking.

Two things the deployed site does that this does not: it compresses JSON on the
fly, so assets.json feels heavier here than in production, and it is fronted by
a CDN. Neither affects what the page looks like or whether its links resolve.

Development only.
"""
from __future__ import annotations

import argparse
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: URL prefix (below --base) -> directory on disk. Longest prefix wins.
MOUNTS = (
    ("/dataset_directory/data", ROOT / "dataset_directory" / "data"),
    ("/dataset_directory/thumbs", ROOT / "dataset_directory" / "thumbs"),
    ("/dataset_directory", ROOT / "dataset_directory" / "site"),
    ("", ROOT / "pages"),
)

#: Stripped from the artifact by CI, so they must 404 here too — otherwise a
#: link into them looks fine locally and breaks once deployed.
UNPUBLISHED = ("/dataset_directory/tests", "/dataset_directory/tests.html")


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".webp": "image/webp",
        ".json": "application/json",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    base = ""

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if self.base and clean.startswith(self.base):
            clean = clean[len(self.base) :] or "/"
        for prefix, directory in MOUNTS:
            if not prefix:
                continue
            if clean == prefix or clean.startswith(prefix + "/"):
                return _join(directory, clean[len(prefix) :])
        return _join(MOUNTS[-1][1], clean)

    def send_head(self):
        clean = self.path.split("?", 1)[0].split("#", 1)[0]
        if self.base:
            # Outside the base prefix nothing exists, the way it would not on the
            # real host either.
            if clean == "/" or clean == self.base.rstrip("/"):
                self.send_response(301)
                self.send_header("Location", self.base + "/")
                self.end_headers()
                return None
            if not clean.startswith(self.base):
                # send_error puts `message` in the status line, which is latin-1
                # only. Anything worth saying goes in `explain`.
                self.send_error(404, "Not Found", "outside the site base path")
                return None
            clean = clean[len(self.base) :] or "/"
        for blocked in UNPUBLISHED:
            if clean == blocked or clean.startswith(blocked + "/"):
                self.send_error(
                    404, "Not Found", "not published: CI strips this from the site"
                )
                return None
        return super().send_head()

    def end_headers(self) -> None:
        # No caching, so a rebuilt export shows up on reload.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def _join(directory: Path, relative: str) -> str:
    parts = [p for p in relative.split("/") if p and p != "." and p != ".."]
    return str(directory.joinpath(*parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Not 8000: that port is the first thing every other dev server on a
    # machine claims, and losing a preview to a port fight is pure noise.
    parser.add_argument("--port", type=int, default=45176)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--base",
        default="",
        help="path prefix to publish under, e.g. /QuantEM (default: site root)",
    )
    args = parser.parse_args()

    base = "/" + args.base.strip("/") if args.base.strip("/") else ""

    missing = []
    for prefix, directory in MOUNTS:
        ok = directory.is_dir()
        if not ok:
            missing.append(directory)
        print(f"  {base + prefix or '/':28s} -> {directory}  [{'ok' if ok else 'MISSING'}]")

    thumbs = ROOT / "dataset_directory" / "thumbs"
    if thumbs in missing:
        print(
            "\nThumbnails are not in git. Unpack them before previewing the cards:\n"
            "  mkdir dataset_directory/thumbs\n"
            "  tar -xzf dataset_directory/thumbs-256-v1.tar.gz -C dataset_directory/thumbs"
        )
    if any(d for d in missing if d != thumbs):
        print("\nA directory the published site needs is missing.", file=sys.stderr)
        return 1

    print(f"\nhttp://{args.host}:{args.port}{base}/\n")

    Handler.base = base
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

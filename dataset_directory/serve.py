"""Serve the directory locally, assembled the way the deployed site is.

The published site is one folder: the page, its data, and its thumbnails. In the
working tree those live apart — ``data/`` is committed, ``thumbs/`` is built and
ignored — so this maps them into the layout the page expects rather than making
you copy 300 MB around.

    python serve.py            # then open http://localhost:8000/

Development only. The real site is static files behind a CDN; nothing here runs
in production.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: URL prefix -> directory on disk. Longest prefix wins.
MOUNTS = {
    "/data": ROOT / "data",
    "/thumbs": ROOT / "thumbs",
    "": ROOT / "site",
}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".webp": "image/webp",
        ".json": "application/json",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, directory in MOUNTS.items():
            if prefix and (clean == prefix or clean.startswith(prefix + "/")):
                relative = clean[len(prefix) :].lstrip("/")
                return str(directory.joinpath(*relative.split("/")))
        return str(MOUNTS[""].joinpath(*clean.lstrip("/").split("/")))

    def end_headers(self) -> None:
        # No caching in development, so a rebuilt export shows up on reload.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    for prefix, directory in MOUNTS.items():
        state = "ok" if directory.is_dir() else "MISSING"
        print(f"  {prefix or '/':10s} -> {directory}  [{state}]")
    print(f"\nhttp://{args.host}:{args.port}/\n")

    HTTPServer((args.host, args.port), partial(Handler)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Path configuration and small shared loaders for the scripts in this folder.

Private helper, not a pipeline step. Every path comes from `paths.yaml` (copied
from `paths.example.yaml`) and can be overridden on the command line.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def load_config(paths_file: str | None = None) -> dict:
    p = Path(paths_file) if paths_file else HERE / "paths.yaml"
    if not p.exists():
        raise SystemExit(
            f"Config not found: {p}\n"
            f"Copy paths.example.yaml to paths.yaml and edit it for the local machine."
        )
    return yaml.safe_load(p.read_text(encoding="utf8")) or {}


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--paths", help="paths.yaml to use (default: alongside these scripts)")
    return ap


def need(cfg: dict, key: str, override=None) -> Path:
    """Resolve a path from CLI override, else paths.yaml. Fail loudly if absent."""
    v = override or cfg.get("paths", {}).get(key)
    if not v:
        raise SystemExit(
            f"'{key}' is not set. Add it to paths.yaml or pass --{key.replace('_', '-')}."
        )
    return Path(v).expanduser()


def canvases(cfg: dict, only: list[str] | None = None) -> dict:
    """The canvases (one per sample) declared in paths.yaml."""
    cv = cfg.get("canvases") or {}
    if not cv:
        raise SystemExit("No 'canvases' declared in paths.yaml.")
    if only:
        missing = [c for c in only if c not in cv]
        if missing:
            raise SystemExit(f"Unknown canvas(es): {', '.join(missing)}")
        return {k: cv[k] for k in only}
    return cv


def read_image(path: Path) -> np.ndarray:
    """Read a single-channel image. Handles the gigapixel mosaics."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(str(path))
    else:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None  # mosaics exceed the decompression-bomb guard
        a = np.array(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def write_image(path: Path, arr: np.ndarray) -> None:
    """Write uint8/uint16 single-channel. PNG via OpenCV, TIFF via tifffile (BigTIFF)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(str(path), arr, bigtiff=arr.nbytes > 2**31)
    else:
        import cv2

        if not cv2.imwrite(str(path), arr, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise IOError(f"Failed to write {path}")


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask from .npy or an image file. Anything > 0 is foreground."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Mask not found: {path}")
    a = np.load(path) if path.suffix == ".npy" else read_image(path)
    return a > 0


def nm_per_px(cfg: dict) -> float:
    return float(cfg.get("scales", {}).get("nm_per_px", 1.0))


def group_of(cfg: dict, canvas: str) -> str:
    return canvases(cfg)[canvas].get("group", "ungrouped")


def groups(cfg: dict) -> list[str]:
    cv = canvases(cfg)
    seen = []
    for c in cv.values():
        g = c.get("group", "ungrouped")
        if g not in seen:
            seen.append(g)
    return seen

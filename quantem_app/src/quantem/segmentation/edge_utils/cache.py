from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from skimage.filters import gaussian, scharr_h, scharr_v
from skimage.transform import resize

from quantem.core.config import CACHE_DIR

_scharr_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_scharr_cache_lock = threading.Lock()


def _scharr_cache_path(cache_key: str, sigma: float, scale: float) -> Path:
    cache_dir = CACHE_DIR / "roi_edges"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sigma_tag = f"{sigma:.2f}".replace(".", "_")
    scale_tag = f"{scale:.2f}".replace(".", "_")
    return cache_dir / f"{cache_key}_scharr_s{sigma_tag}_sc{scale_tag}.npz"


def get_cached_scharr(
    cache_key: str, roi_image: np.ndarray, sigma: float, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    memory_key = f"{cache_key}:scharr:{sigma:.3f}:{scale:.3f}"
    with _scharr_cache_lock:
        cached = _scharr_cache.get(memory_key)
    if cached is not None:
        return cached

    cache_path = _scharr_cache_path(cache_key, sigma, scale)
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            gmag = data["gmag"]
            gx = data["gx"]
            gy = data["gy"]
            with _scharr_cache_lock:
                _scharr_cache[memory_key] = (gmag, gx, gy)
            return gmag, gx, gy
        except Exception:
            pass

    img = roi_image.astype(np.float32)
    if scale != 1.0:
        img = resize(
            img,
            (int(round(img.shape[0] * scale)), int(round(img.shape[1] * scale))),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
    smoothed = gaussian(img, sigma=sigma, preserve_range=True).astype(np.float32)
    gx = scharr_h(smoothed).astype(np.float32)
    gy = scharr_v(smoothed).astype(np.float32)
    gmag = np.hypot(gx, gy).astype(np.float32)

    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, gmag=gmag, gx=gx, gy=gy)
    os.replace(tmp_path, cache_path)

    with _scharr_cache_lock:
        _scharr_cache[memory_key] = (gmag, gx, gy)
    return gmag, gx, gy

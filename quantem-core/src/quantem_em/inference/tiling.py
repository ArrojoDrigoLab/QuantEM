"""Sliding-window geometry and the Hann blend.

Ported verbatim from ``segmentation_training/harness/evaluate.py`` lines 33-52. Every constant here
is load-bearing for reproducing the published numbers:

* ``round_up`` gives the working tile a whole number of encoder patches — 512 stays 512 at patch 16
  and becomes **518** at patch 14. That number appears in no config file; it emerges here.
* ``hann2d`` adds a **1e-3 floor**. Without it, a pixel covered by exactly one window whose Hann
  weight is 0 at the border gets zero total weight and the blend divides by ~0.
* ``window_starts`` walks by ``stride`` and then forces the final window flush to the edge, so the
  right/bottom margins are covered without a partial tile.
* the stride is ``int(round(t * (1 - overlap)))``. At patch 14, ``518 * 0.75 = 388.5`` exactly, and
  Python rounds halves to **even** -> 388. (An earlier planning document said 389; it was wrong, and
  an off-by-one shifts every window.)
"""

from __future__ import annotations

import numpy as np


def round_up(n: int, m: int) -> int:
    """Smallest multiple of ``m`` that is >= ``n``."""
    return ((int(n) + int(m) - 1) // int(m)) * int(m)


def hann2d(t: int) -> np.ndarray:
    """Separable 2-D Hann window with the 1e-3 floor. Shape ``[t, t]``, float32."""
    w = np.hanning(t).astype(np.float32)
    return np.outer(w, w) + 1e-3


def window_starts(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering ``[0, length)``, with the last window flush to the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def stride_for(tile: int, overlap: float) -> int:
    return max(1, int(round(tile * (1.0 - float(overlap)))))


def pad_to_tile(em: np.ndarray, tile: int, patch: int) -> tuple[np.ndarray, tuple[int, int]]:
    """0-pad so the array is at least one tile and a whole number of patches in each axis.

    Returns ``(padded, (H0, W0))``. Padding is ``mode="constant"`` — the "honest border" choice, not
    reflect: a reflected border invents plausible tissue that the model then segments.
    """
    h0, w0 = em.shape
    ph = max(tile - h0, 0)
    pw = max(tile - w0, 0)
    ht, wt = h0 + ph, w0 + pw
    ph += round_up(ht, patch) - ht
    pw += round_up(wt, patch) - wt
    if ph or pw:
        em = np.pad(em, ((0, ph), (0, pw)), mode="constant")
    return np.ascontiguousarray(em), (h0, w0)


def window_count(shape: tuple[int, int], tile: int, stride: int) -> int:
    """How many forward passes a region of this size will take. Used for progress and estimates."""
    h, w = shape
    return len(window_starts(h, tile, stride)) * len(window_starts(w, tile, stride))

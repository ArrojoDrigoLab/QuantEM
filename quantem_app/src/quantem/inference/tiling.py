"""Sliding-window tiling with Hann blending.

This is the piece the manuscript commits to: *"Inference ... as sliding tiles
with 25% overlap and Hann blending."* The reference implementation is
``fig3/harness/evaluate.py::predict_region`` in the research tree; the geometry
here is deliberately identical to it so an app result reproduces a published
number:

* ``stride = max(1, round(tile * (1 - overlap)))``
* window starts walk ``range(0, length - tile + 1, stride)`` with the **last
  window flush to the edge**, so the right/bottom margins are never dropped
* the blend weight is a separable 2-D Hann window, ``outer(hann(t), hann(t))``,
  plus a small floor so a pixel covered by exactly one window (a region corner)
  does not get a zero weight and vanish in the normalisation
* per-window probabilities are accumulated weighted, then divided by the summed
  weight -- never averaged unweighted, never overwritten

What this module adds over the reference
----------------------------------------
The reference allocates ``[K, H, W]`` accumulators for the whole region, which
is fine for a 2k test crop and impossible for a gigapixel asset. :class:`BandBlender` keeps only
``tile`` rows of accumulator alive. Tiles are fed in row-major order; once the
next tile row starts, every row above it is final, so it is normalised and
handed to a sink (a memmap, a PNG encoder, a downstream labeller) and its memory
is reused. Peak cost is ``2 x tile x width x 4`` bytes regardless of image
height.

Pure numpy. No torch, no Django, no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: Manuscript-committed overlap fraction between adjacent windows.
DEFAULT_OVERLAP = 0.25

#: Added to the Hann window so single-window pixels keep a non-zero weight.
#: Same value as the reference implementation.
HANN_FLOOR = 1e-3

#: Guard for the weight-sum divide.
_WEIGHT_EPS = 1e-6

BandSink = Callable[[int, np.ndarray], None]
TilePredictor = Callable[["Tile"], np.ndarray]


@lru_cache(maxsize=8)
def hann2d(tile: int, floor: float = HANN_FLOOR) -> np.ndarray:
    """2-D separable Hann window of edge ``tile``, with a small floor.

    Cached: the same window is reused for every tile of a run. The returned
    array is read-only so a caller cannot corrupt the cache.
    """
    if tile < 1:
        raise ValueError(f"tile must be >= 1, got {tile}")
    w = np.hanning(tile).astype(np.float32)
    window = np.outer(w, w).astype(np.float32) + np.float32(floor)
    window.setflags(write=False)
    return window


def stride_for(tile: int, overlap: float = DEFAULT_OVERLAP) -> int:
    """Window step for a tile edge and overlap fraction."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    return max(1, int(round(tile * (1.0 - overlap))))


def window_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile start offsets covering ``[0, length)``, last window flush to the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def round_up(n: int, m: int) -> int:
    """Smallest multiple of ``m`` that is >= ``n``."""
    return ((n + m - 1) // m) * m


@dataclass(frozen=True)
class Tile:
    """One square window in padded-image coordinates."""

    y: int
    x: int
    size: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return (slice(self.y, self.y + self.size), slice(self.x, self.x + self.size))


@dataclass(frozen=True)
class TilePlan:
    """The complete window layout for one region."""

    height: int
    width: int
    tile: int
    overlap: float
    stride: int
    ys: tuple[int, ...]
    xs: tuple[int, ...]

    @property
    def n_tiles(self) -> int:
        return len(self.ys) * len(self.xs)

    @property
    def n_rows(self) -> int:
        return len(self.ys)

    def tiles(self) -> Iterator[Tile]:
        """Every window, in row-major order (the order :class:`BandBlender` needs)."""
        for y in self.ys:
            for x in self.xs:
                yield Tile(y=y, x=x, size=self.tile)

    def row_tiles(self, y: int) -> Iterator[Tile]:
        for x in self.xs:
            yield Tile(y=y, x=x, size=self.tile)


def plan_tiles(
    shape: tuple[int, int],
    tile: int,
    overlap: float = DEFAULT_OVERLAP,
) -> TilePlan:
    """Lay out sliding windows over a region that is already at least one tile.

    Args:
        shape: ``(height, width)`` of the region to tile. Both must be >= tile;
            call :func:`pad_for_tiling` first if they are not.
        tile: window edge in pixels (a whole number of encoder patches).
        overlap: fraction of a window shared with its neighbour.

    Raises:
        ValueError: if the region is smaller than one tile in either axis.
    """
    height, width = int(shape[0]), int(shape[1])
    if height < tile or width < tile:
        raise ValueError(
            f"region {height}x{width} is smaller than tile {tile}; "
            "pad it with pad_for_tiling() first"
        )
    stride = stride_for(tile, overlap)
    return TilePlan(
        height=height,
        width=width,
        tile=tile,
        overlap=float(overlap),
        stride=stride,
        ys=tuple(window_starts(height, tile, stride)),
        xs=tuple(window_starts(width, tile, stride)),
    )


def estimate_tile_count(
    shape: tuple[int, int],
    tile: int,
    overlap: float = DEFAULT_OVERLAP,
) -> int:
    """Number of windows a region will need, for progress reporting.

    Unlike :func:`plan_tiles` this tolerates regions smaller than a tile (they
    become one window after padding).
    """
    stride = stride_for(tile, overlap)
    rows = len(window_starts(max(int(shape[0]), tile), tile, stride))
    cols = len(window_starts(max(int(shape[1]), tile), tile, stride))
    return max(rows * cols, 1)


def pad_for_tiling(
    image: np.ndarray,
    tile: int,
    patch: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Zero-pad so the region is >= one tile and a whole number of patches.

    Matches the reference implementation, including its choice of constant
    (zero) padding rather than reflection -- an honest border beats an invented
    one, because a reflected organelle is a plausible-looking hallucination.

    Returns:
        ``(padded, (pad_bottom, pad_right))``. Crop the result of inference back
        with ``prob[:h0, :w0]``.
    """
    h0, w0 = image.shape[:2]
    ph = max(tile - h0, 0)
    pw = max(tile - w0, 0)
    ht, wt = h0 + ph, w0 + pw
    ph += round_up(ht, patch) - ht
    pw += round_up(wt, patch) - wt
    if ph == 0 and pw == 0:
        return image, (0, 0)
    padded = np.pad(image, ((0, ph), (0, pw)), mode="constant")
    return padded, (ph, pw)


class BandBlender:
    """Bounded-memory Hann accumulation over row-major tiles.

    Feed windows with :meth:`add` in row-major order (``plan.tiles()`` yields
    them correctly), then call :meth:`finish`. Each time a tile row completes,
    the rows above the next row's start are normalised and passed to ``on_band``
    as ``(y0, band)`` where ``band`` is float32 ``[d, width]`` and ``y0`` is its
    absolute top row in the padded region.

    Memory is ``2 x tile x width x 4`` bytes: for a 32768-px-wide asset with a
    512 tile that is 134 MB, independent of image height.
    """

    def __init__(
        self,
        plan: TilePlan,
        on_band: BandSink,
        *,
        floor: float = HANN_FLOOR,
    ) -> None:
        self._plan = plan
        self._on_band = on_band
        self._window = hann2d(plan.tile, floor)
        self._acc = np.zeros((plan.tile, plan.width), dtype=np.float32)
        self._wsum = np.zeros((plan.tile, plan.width), dtype=np.float32)
        self._row = 0                  # index into plan.ys
        self._buf_y0 = plan.ys[0]      # absolute row of buffer line 0
        self._finished = False

    @property
    def buffer_bytes(self) -> int:
        return self._acc.nbytes + self._wsum.nbytes

    def add(self, tile: Tile, prob: np.ndarray) -> None:
        """Accumulate one window's probability map.

        Args:
            tile: the window, as produced by the plan.
            prob: float32 ``[tile, tile]`` foreground probabilities.
        """
        if self._finished:
            raise RuntimeError("BandBlender.add() after finish()")
        if prob.shape != (self._plan.tile, self._plan.tile):
            raise ValueError(
                f"tile prob shape {prob.shape} != ({self._plan.tile}, {self._plan.tile})"
            )
        last_row = len(self._plan.ys) - 1
        while self._row < last_row and tile.y > self._plan.ys[self._row]:
            self._advance()
        if tile.y != self._plan.ys[self._row]:
            raise ValueError(
                f"tiles must arrive in row-major order; got y={tile.y} "
                f"while at row y={self._plan.ys[self._row]}"
            )

        offset = tile.y - self._buf_y0
        rows = slice(offset, offset + tile.size)
        cols = slice(tile.x, tile.x + tile.size)
        self._acc[rows, cols] += prob.astype(np.float32, copy=False) * self._window
        self._wsum[rows, cols] += self._window

    def finish(self) -> None:
        """Flush every remaining row. Idempotent."""
        if self._finished:
            return
        while self._row < len(self._plan.ys) - 1:
            self._advance()
        height = min(self._plan.tile, self._plan.height - self._buf_y0)
        if height > 0:
            self._emit(height)
        self._finished = True

    # --- internals ---

    def _advance(self) -> None:
        """Retire the rows above the next tile row and shift the buffer up."""
        next_y = self._plan.ys[self._row + 1]
        step = next_y - self._buf_y0
        if step <= 0:
            self._row += 1
            return
        self._emit(step)
        keep = self._plan.tile - step
        if keep > 0:
            self._acc[:keep] = self._acc[step:]
            self._wsum[:keep] = self._wsum[step:]
        self._acc[max(keep, 0):] = 0.0
        self._wsum[max(keep, 0):] = 0.0
        self._buf_y0 = next_y
        self._row += 1

    def _emit(self, height: int) -> None:
        band = self._acc[:height] / np.maximum(self._wsum[:height], _WEIGHT_EPS)
        self._on_band(self._buf_y0, band.astype(np.float32, copy=False))


def blend_region_streaming(
    plan: TilePlan,
    predict_tile: TilePredictor,
    on_band: BandSink,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Run every window and stream normalised row-bands to ``on_band``.

    ``predict_tile(tile)`` returns the float32 ``[tile, tile]`` foreground
    probability for one window. This is the bounded-memory entry point: nothing
    the size of the full region is ever allocated here.
    """
    blender = BandBlender(plan, on_band)
    total = plan.n_tiles
    for done, tile in enumerate(plan.tiles(), start=1):
        blender.add(tile, predict_tile(tile))
        if on_progress is not None:
            on_progress(done / total)
    blender.finish()
    if on_progress is not None:
        on_progress(1.0)


def blend_region(
    plan: TilePlan,
    predict_tile: TilePredictor,
    *,
    on_progress: Callable[[float], None] | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Materialise the whole blended probability map.

    Convenience wrapper around :func:`blend_region_streaming` for regions that
    do fit in memory. Pass ``out`` (e.g. a ``np.memmap``) to stream into
    backing storage instead of RAM.
    """
    target = (
        np.zeros((plan.height, plan.width), dtype=np.float32) if out is None else out
    )
    if target.shape != (plan.height, plan.width):
        raise ValueError(
            f"out shape {target.shape} != plan ({plan.height}, {plan.width})"
        )

    def sink(y0: int, band: np.ndarray) -> None:
        target[y0:y0 + band.shape[0]] = band

    blend_region_streaming(plan, predict_tile, sink, on_progress=on_progress)
    return target


def memmap_band_sink(
    path,
    shape: tuple[int, int],
    dtype=np.float32,
) -> tuple[BandSink, np.memmap]:
    """Open a memmap and return ``(sink, memmap)`` for use with streaming blends.

    The caller owns the memmap: flush and delete it when the run is done.
    """
    store = np.memmap(str(path), dtype=dtype, mode="w+", shape=shape)

    def sink(y0: int, band: np.ndarray) -> None:
        store[y0:y0 + band.shape[0]] = band.astype(dtype, copy=False)

    return sink, store

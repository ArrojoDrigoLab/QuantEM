"""Cross-platform IterableDataset over EM tile shards (single-channel).

The shards are standard WebDataset-style tars (``<tile_id>.png`` + ``<tile_id>.json``), so
on Linux they can be consumed by the ``webdataset`` library directly. This module reads them with
stdlib ``tarfile`` rather than webdataset's URL/``gopen`` layer, which mis-parses Windows drive
letters and URL-encodes spaces in paths, so those paths never resolve. The native reader is simple,
fast (PNG decode + augmentation dominate, not tar iteration), and behaves identically on Windows
and Linux.

Yields per sample:
  * (transformed, target) when ``transform`` is given (EM DINO aug -> multi-crop dict; the target is
    ``target_transform(metadata)`` when one is set, else ``empty_target``, which defaults to the empty
    tuple DINOv3's target_transform produces), or
  * (PIL 'L' image, metadata dict) when ``transform`` is None.

Single-channel 'L' decode always (never RGB). ``min_side`` drops tiles whose stored
width/height are below the experiment crop (so 768/1024 runs share a 512-built shard set
without upsampling). ``resampled=True`` gives an infinite stream (matches DINOv3's INFINITE
sampler) with per-worker/per-rank shard sharding and reshuffling.
"""

from __future__ import annotations

import glob
import io
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from torch.utils.data import IterableDataset, get_worker_info

def list_shard_urls(shard_dir: str | Path, prefix: str = "em_tiles_v0") -> list[str]:
    """Sorted list of shard tar paths for a prefix (deterministic)."""
    shard_dir = Path(shard_dir)
    urls = sorted(glob.glob(str(shard_dir / f"{prefix}-*.tar")))
    if not urls:
        urls = sorted(glob.glob(str(shard_dir / "*.tar")))
    if not urls:  # recursive fallback (shards/<prefix>/<prefix>-*.tar)
        urls = sorted(glob.glob(str(shard_dir / "**" / f"{prefix}-*.tar"), recursive=True))
    if not urls:
        urls = sorted(glob.glob(str(shard_dir / "**" / "*.tar"), recursive=True))
    return urls

def _group_shard(path: str) -> Iterator[dict[str, Any]]:
    """Stream samples from one tar, grouping consecutive members by key (basename)."""
    current_key: str | None = None
    parts: dict[str, bytes] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            # key = basename before first dot; ext = after.
            base = os.path.basename(name)
            if "." not in base:
                key, ext = base, ""
            else:
                key, ext = base.split(".", 1)
            if current_key is not None and key != current_key:
                if parts:
                    yield {"__key__": current_key, **parts}
                parts = {}
            current_key = key
            f = tar.extractfile(member)
            parts[ext] = f.read() if f is not None else b""
        if current_key is not None and parts:
            yield {"__key__": current_key, **parts}

def _decode_grayscale(sample: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from PIL import Image

    png = sample.get("png") or sample.get("PNG")
    meta_raw = sample.get("json") or b"{}"
    meta = json.loads(meta_raw) if isinstance(meta_raw, (bytes, str, bytearray)) else dict(meta_raw)
    img = Image.open(io.BytesIO(png)).convert("L")
    meta["__key__"] = sample.get("__key__")
    return img, meta

def _shuffle_buffer(it: Iterator, bufsize: int, rng: random.Random) -> Iterator:
    if bufsize <= 1:
        yield from it
        return
    buf: list = []
    for x in it:
        if len(buf) < bufsize:
            buf.append(x)
            continue
        idx = rng.randrange(len(buf))
        yield buf[idx]
        buf[idx] = x
    rng.shuffle(buf)
    yield from buf

def _dist_rank_world() -> tuple[int, int]:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1

class EMShardDataset(IterableDataset):
    def __init__(
        self,
        urls: Sequence[str] | str,
        transform: Callable | None = None,
        min_side: int = 0,
        resampled: bool = True,
        shuffle_buffer: int = 1000,
        shard_shuffle: bool = True,
        seed: int = 0,
        target_transform: Callable | None = None,
        empty_target=(),
        split_by_node: bool = True,
        split_by_worker: bool = True,
        max_samples: int | None = None,
    ):
        super().__init__()
        if isinstance(urls, (str, os.PathLike)):
            urls = [str(urls)]
        self.urls = list(urls)
        if not self.urls:
            raise FileNotFoundError("No shard URLs provided to EMShardDataset.")
        self.transform = transform
        self.target_transform = target_transform
        self.empty_target = empty_target
        self.min_side = int(min_side)
        self.resampled = resampled
        self.shuffle_buffer = int(shuffle_buffer)
        self.shard_shuffle = shard_shuffle
        self.seed = int(seed)
        self.split_by_node = split_by_node
        self.split_by_worker = split_by_worker
        self.max_samples = max_samples

    # -- assignment of shards to this (rank, worker) ------------------------
    def _assigned_shards(self) -> tuple[list[str], int]:
        urls = list(self.urls)
        rank, world = _dist_rank_world()
        wi = get_worker_info()
        worker_id, num_workers = (wi.id, wi.num_workers) if wi is not None else (0, 1)

        if not self.resampled:
            # Disjoint split so each sample is seen once per pass.
            if self.split_by_node and world > 1:
                urls = urls[rank::world]
            if self.split_by_worker and num_workers > 1:
                urls = urls[worker_id::num_workers]
        # unique stream id for seeding (resampled streams stay statistically independent)
        stream_id = (rank * 100003 + worker_id) if (self.split_by_node or self.split_by_worker) else 0
        return urls, stream_id

    def _sample_stream(self) -> Iterator[tuple[Any, dict[str, Any]]]:
        urls, stream_id = self._assigned_shards()
        if not urls:
            return
        epoch = 0
        rng = random.Random(self.seed + stream_id)
        while True:
            shard_order = list(urls)
            if self.shard_shuffle:
                random.Random(self.seed + stream_id + epoch * 7919).shuffle(shard_order)
            for shard in shard_order:
                try:
                    for raw in _group_shard(shard):
                        yield _decode_grayscale(raw)
                except (tarfile.TarError, OSError) as exc:  # pragma: no cover - corrupt shard guard
                    import warnings

                    warnings.warn(f"Skipping unreadable shard {shard}: {exc!r}")
                    continue
            epoch += 1
            if not self.resampled:
                break

    def __iter__(self):
        urls, stream_id = self._assigned_shards()
        rng = random.Random(self.seed + stream_id + 104729)
        stream = self._sample_stream()
        if self.shuffle_buffer > 1:
            stream = _shuffle_buffer(stream, self.shuffle_buffer, rng)
        count = 0
        for img, meta in stream:
            if self.min_side > 0:
                w, h = meta.get("width"), meta.get("height")
                if w and h and min(int(w), int(h)) < self.min_side:
                    continue
            if self.transform is None:
                out = (img, meta)
            else:
                t = self.transform(img)
                # Bridge the realized native_fov global downsample, emitted by the EM DINO augmentation as
                # `global_downsample`, into the metadata stream so crop-scale-aware FINO factors can correct nm/px
                # to the crop's true resolution. Always popped, so the model collate only ever sees crop tensors.
                m_g = t.pop("global_downsample", None) if isinstance(t, dict) else None
                if self.target_transform is not None:
                    if m_g is not None:
                        meta["global_downsample"] = m_g
                    tgt = self.target_transform(meta)
                else:
                    tgt = self.empty_target
                out = (t, tgt)
            yield out
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                return

    # Compatibility alias for callers that expect a .pipeline() returning an iterable dataset.
    def pipeline(self):
        return self

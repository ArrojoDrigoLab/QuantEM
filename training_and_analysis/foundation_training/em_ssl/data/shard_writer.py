"""Deterministic, reproducible WebDataset-style shard writer for EM parent tiles.

Each shard is a POSIX tar containing, per tile, two consecutive members that share a key:
    <tile_id>.png    # PNG bytes (verbatim, or re-encoded at ``compress_level``)
    <tile_id>.json   # compact downstream metadata (source_id, dims, stats, ...)
WebDataset groups members by the basename before the first dot, so tile_ids (hex hashes,
no dots) form clean sample keys.

Reproducibility:
  * tar member metadata is normalized (mtime=0, uid/gid=0, fixed mode) so byte-identical
    inputs + ordering produce byte-identical shards and stable sha256 fingerprints.
  * ordering is fully determined by --seed; each shard is a fixed contiguous slice of that
    global order, so the output is independent of worker count.

Performance: the default packer parallelizes across shards — the deterministic global order is
computed once, sliced into per-shard chunks, and the chunks are packed as one task per shard over a
process pool. ``sequential_read`` selects the alternative packer, which reads every tile once in
manifest (on-disk) order through a reader -> encode -> writer pipeline and routes each tile into its
pre-assigned shard, trading per-shard parallelism for sequential reads. With ``compress_level`` set,
each PNG is decoded and re-encoded (lossless) at that zlib level to shrink shards (EM tiles ~1.8x at
level 6); the encode CPU then hides behind the file-read I/O across workers, so compression costs
little wall-clock vs the verbatim copy. ``compress_level=None`` (default) copies the PNG bytes
verbatim.

Balancing: with ``balance_by_source`` each source's tiles are spread proportionally and
evenly across the global order (fractional-rank interleave), so any contiguous shard sees
a source mix close to the corpus distribution — no shard is dominated by one source when
avoidable. The tiler already capped per-source tile counts, so counts are not re-balanced here,
only their placement. Without balancing, a plain seeded shuffle is used.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .manifest import ResolvedTile

_FIXED_MTIME = 0

def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "big")

def order_tiles(tiles: list[ResolvedTile], seed: int, balance_by_source: bool) -> list[ResolvedTile]:
    """Return a deterministic global ordering of tiles.

    balance_by_source=True  -> proportional fractional-rank interleave (mixes sources).
    balance_by_source=False -> plain seeded shuffle.
    """
    if not balance_by_source:
        # Seeded shuffle via stable per-tile hash keyed by seed.
        return sorted(tiles, key=lambda t: _stable_hash(f"{seed}:{t.tile_id}"))

    by_source: dict[str, list[ResolvedTile]] = defaultdict(list)
    for t in tiles:
        by_source[t.source_id].append(t)

    keyed: list[tuple[float, int, str, ResolvedTile]] = []
    for source_id, group in by_source.items():
        # Deterministic intra-source shuffle.
        group_sorted = sorted(group, key=lambda t: _stable_hash(f"{seed}:{source_id}:{t.tile_id}"))
        n = len(group_sorted)
        src_tiebreak = _stable_hash(f"{seed}:src:{source_id}")
        for i, t in enumerate(group_sorted):
            frac = (i + 0.5) / n  # even spread of this source across [0,1)
            keyed.append((frac, src_tiebreak, t.tile_id, t))
    keyed.sort(key=lambda x: (x[0], x[1], x[2]))
    return [k[3] for k in keyed]

@dataclass
class ShardInfo:
    name: str
    path: str
    num_samples: int
    sha256: str
    num_bytes: int
    source_counts: dict[str, int] = field(default_factory=dict)

@dataclass
class ShardBuildResult:
    shard_prefix: str
    output_dir: str
    num_shards: int
    num_tiles: int
    samples_per_shard: int
    seed: int
    balance_by_source: bool
    shards: list[ShardInfo] = field(default_factory=list)
    global_source_counts: dict[str, int] = field(default_factory=dict)
    skipped_missing: int = 0
    skipped_duplicate_ids: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = _FIXED_MTIME
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))

class _HashingStream:
    """Write-through file wrapper that sha256-hashes bytes as they're written.

    Lets the streaming packer compute each shard's hash + byte size inline, so it never has
    to read the finished tar back off disk (which on a spinning disk would be a second pass).
    Used with ``tarfile`` stream mode (``w|``), which only writes — never seeks.
    """

    def __init__(self, fileobj):
        self._f = fileobj
        self.hasher = hashlib.sha256()
        self.nbytes = 0

    def write(self, data):
        self.hasher.update(data)
        self.nbytes += len(data)
        return self._f.write(data)

    def flush(self):
        self._f.flush()

    def tell(self):
        return self.nbytes

    def close(self):
        self._f.close()

def _maybe_recode(raw: bytes, compress_level: int | None, verify_png: bool) -> bytes | None:
    """Return the PNG bytes to store (verbatim or re-encoded), or None if the tile is unreadable.

    compress_level=None -> verbatim copy (optionally decode-verified). int -> decode + re-encode
    PNG at that zlib level (lossless), single-channel 'L'.
    """
    if compress_level is None:
        if verify_png:
            try:
                from PIL import Image

                Image.open(io.BytesIO(raw)).verify()
            except Exception:
                return None
        return raw
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        if img.mode != "L":
            img = img.convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=int(compress_level), optimize=False)
        return buf.getvalue()
    except Exception:
        return None

def _pack_one_shard(job: tuple) -> tuple[ShardInfo, int] | None:
    """Pack one shard (a fixed slice of the global order) into a tar. Worker entrypoint.

    Returns (ShardInfo, skipped_missing) for a non-empty shard, or None if every tile was
    missing/unreadable (the empty tar is removed). Top-level + picklable for ProcessPoolExecutor.
    """
    idx, tiles, output_dir, shard_prefix, compress_level, verify_png = job
    name = f"{shard_prefix}-{idx:06d}.tar"
    path = Path(output_dir) / name
    counts: Counter[str] = Counter()
    written = 0
    skipped_missing = 0
    with tarfile.open(path, "w") as tar:  # uncompressed tar (fast random read); PNGs carry the compression
        for t in tiles:
            try:
                raw = Path(t.path).read_bytes()
            except (FileNotFoundError, OSError):
                skipped_missing += 1
                continue
            png_bytes = _maybe_recode(raw, compress_level, verify_png)
            if png_bytes is None:
                skipped_missing += 1
                continue
            meta = dict(t.metadata)
            meta.setdefault("tile_id", t.tile_id)
            meta.setdefault("source_id", t.source_id)
            json_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
            _add_bytes(tar, f"{t.tile_id}.png", png_bytes)
            _add_bytes(tar, f"{t.tile_id}.json", json_bytes)
            written += 1
            counts[t.source_id] += 1
    if written == 0:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    data = path.read_bytes()
    return (
        ShardInfo(
            name=name,
            path=str(path),
            num_samples=written,
            sha256=hashlib.sha256(data).hexdigest(),
            num_bytes=len(data),
            source_counts=dict(counts.most_common()),
        ),
        skipped_missing,
    )

def _read_bytes(path: str) -> bytes | None:
    """Read a tile's bytes, or None if missing/unreadable. Run on the reader pool."""
    try:
        return Path(path).read_bytes()
    except (FileNotFoundError, OSError):
        return None

def _build_streaming(
    deduped: list[ResolvedTile],
    shard_of: dict[str, int],
    num_shards: int,
    output_dir: Path,
    shard_prefix: str,
    compress_level: int | None,
    verify_png: bool,
    num_workers: int,
    progress: bool,
    read_workers: int = 8,
) -> tuple[list[ShardInfo], int]:
    """Pipelined single-pass packer: concurrent readers -> encode pool -> one buffered writer.

    Tiles are consumed in ``deduped`` (manifest ≈ on-disk) order, so a spinning disk reads
    sequentially instead of seeking per balanced shard. Three decoupled stages overlap:
      * ``read_workers`` reader threads open+read tiles concurrently (overlapping the per-file
        open latency that dominates on an HDD) — submitted and awaited in manifest order;
      * an ``num_workers``-thread encode pool re-encodes PNGs in parallel (PIL frees the GIL);
      * a single writer thread drains results in order and appends each tile to its pre-assigned
        balanced shard tar, with large per-shard write buffers so the round-robin appends across
        all open shards flush as big sequential chunks instead of tiny seeks.
    Because the writer consumes in manifest order, the output is byte-identical to the
    single-reader path (deterministic shards / stable sha256); shards hash inline (no read-back).
    """
    import queue
    import threading
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    # Per-shard write buffer, capped so the total stays ~<= 8 GB regardless of shard count.
    buf_bytes = min(16 << 20, max(1 << 20, (8 << 30) // max(1, num_shards)))
    paths = [output_dir / f"{shard_prefix}-{idx:06d}.tar" for idx in range(num_shards)]
    streams = [_HashingStream(open(p, "wb", buffering=buf_bytes)) for p in paths]
    tars = [tarfile.open(fileobj=s, mode="w|") for s in streams]
    counts = [Counter() for _ in range(num_shards)]
    written = [0] * num_shards
    skipped = 0  # writer-thread only

    pbar = None
    if progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(deduped), desc=f"packing tiles (seq-read r{read_workers}/x{num_workers})", unit="tile")
        except Exception:
            pbar = None

    read_pool = ThreadPoolExecutor(max_workers=max(1, read_workers))
    enc_pool = ThreadPoolExecutor(max_workers=max(1, num_workers))
    write_q: "queue.Queue" = queue.Queue(maxsize=max(16, num_workers * 4))

    def _writer():
        nonlocal skipped
        while True:
            item = write_q.get()
            if item is None:
                break
            t0, sidx, enc_future = item
            png_bytes = None if enc_future is None else enc_future.result()
            if png_bytes is None:
                skipped += 1
            else:
                meta = dict(t0.metadata)
                meta.setdefault("tile_id", t0.tile_id)
                meta.setdefault("source_id", t0.source_id)
                json_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
                _add_bytes(tars[sidx], f"{t0.tile_id}.png", png_bytes)
                _add_bytes(tars[sidx], f"{t0.tile_id}.json", json_bytes)
                written[sidx] += 1
                counts[sidx][t0.source_id] += 1
            if pbar is not None:
                pbar.update(1)

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    read_window: deque = deque()
    readahead = max(8, read_workers * 4)
    tiles_iter = iter(deduped)
    exhausted = False
    try:
        while not exhausted or read_window:
            while len(read_window) < readahead and not exhausted:
                try:
                    t = next(tiles_iter)
                except StopIteration:
                    exhausted = True
                    break
                read_window.append((t, read_pool.submit(_read_bytes, str(t.path))))
            if read_window:
                t0, read_future = read_window.popleft()
                raw = read_future.result()  # await oldest read => preserves manifest order
                if raw is None:
                    write_q.put((t0, None, None))
                else:
                    enc_future = enc_pool.submit(_maybe_recode, raw, compress_level, verify_png)
                    write_q.put((t0, shard_of[t0.tile_id], enc_future))
    finally:
        write_q.put(None)
        writer_thread.join()
        read_pool.shutdown(wait=True)
        enc_pool.shutdown(wait=True)
        if pbar is not None:
            pbar.close()

    infos: list[ShardInfo] = []
    for idx in range(num_shards):
        tars[idx].close()
        streams[idx].close()
        if written[idx] == 0:
            try:
                paths[idx].unlink()
            except OSError:
                pass
            continue
        infos.append(
            ShardInfo(
                name=paths[idx].name,
                path=str(paths[idx]),
                num_samples=written[idx],
                sha256=streams[idx].hasher.hexdigest(),
                num_bytes=streams[idx].nbytes,
                source_counts=dict(counts[idx].most_common()),
            )
        )
    return infos, skipped

def build_shards(
    tiles: Iterable[ResolvedTile],
    output_dir: str | Path,
    shard_prefix: str = "em_tiles_v0",
    samples_per_shard: int = 1000,
    seed: int = 1337,
    balance_by_source: bool = True,
    verify_png: bool = False,
    progress: bool = True,
    compress_level: int | None = None,
    num_workers: int | None = None,
    sequential_read: bool = False,
    read_workers: int | None = None,
) -> ShardBuildResult:
    """Pack resolved tiles into reproducible WebDataset tar shards.

    ``tiles`` are materialized into a list (needed for deterministic global ordering); each
    ResolvedTile is lightweight (~hundreds of bytes) so the full ~306k-tile corpus fits
    comfortably in memory. By default the global order is sliced into ``samples_per_shard`` chunks
    and the chunks are packed concurrently across ``num_workers`` processes (default: CPU count);
    output is byte-identical regardless of worker count. ``sequential_read`` instead packs through
    the pipelined single-pass reader (``_build_streaming``), where ``num_workers`` is the
    encode-thread count and ``read_workers`` the number of concurrent readers. ``compress_level``
    (0-9) re-encodes each PNG losslessly at that zlib level (None = verbatim copy).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate by tile_id (content hash should already be unique; guard anyway).
    seen: set[str] = set()
    deduped: list[ResolvedTile] = []
    dup = 0
    for t in tiles:
        if t.tile_id in seen:
            dup += 1
            continue
        seen.add(t.tile_id)
        deduped.append(t)

    ordered = order_tiles(deduped, seed=seed, balance_by_source=balance_by_source)

    result = ShardBuildResult(
        shard_prefix=shard_prefix,
        output_dir=str(output_dir),
        num_shards=0,
        num_tiles=0,
        samples_per_shard=samples_per_shard,
        seed=seed,
        balance_by_source=balance_by_source,
        skipped_duplicate_ids=dup,
    )

    workers = num_workers if num_workers is not None else (os.cpu_count() or 4)
    shard_infos: list[ShardInfo] = []
    global_counts: Counter[str] = Counter()
    skipped_missing_total = 0

    if sequential_read:
        # Read every tile once in manifest (≈ on-disk) order so the spinning-disk reads are
        # sequential, while a thread pool re-encodes in parallel and routes each tile into its
        # pre-assigned balanced shard. ``num_workers`` is the encode-thread count here, not a
        # count of concurrent disk readers, so it does not drive random-read thrash.
        num_shards = (len(ordered) + samples_per_shard - 1) // samples_per_shard if ordered else 0
        shard_of = {t.tile_id: (p // samples_per_shard) for p, t in enumerate(ordered)}
        rw = read_workers if read_workers is not None else 8
        shard_infos, skipped_missing_total = _build_streaming(
            deduped, shard_of, num_shards, output_dir, shard_prefix,
            compress_level, verify_png, max(1, int(workers)), progress, max(1, int(rw)),
        )
        for info in shard_infos:
            for k, v in info.source_counts.items():
                global_counts[k] += v
    else:
        # Slice the deterministic global order into fixed per-shard chunks (independent of workers).
        chunks = [ordered[i : i + samples_per_shard] for i in range(0, len(ordered), samples_per_shard)]
        jobs = [(idx, chunk, str(output_dir), shard_prefix, compress_level, verify_png) for idx, chunk in enumerate(chunks)]
        workers = max(1, min(int(workers), len(jobs))) if jobs else 1

        def _progress(iterable, total):
            if not progress:
                return iterable
            try:
                from tqdm import tqdm

                desc = "packing shards" + (f" (x{workers})" if workers > 1 else "")
                return tqdm(iterable, total=total, desc=desc, unit="shard")
            except Exception:
                return iterable

        def _collect(res):
            nonlocal skipped_missing_total
            if res is None:
                return
            info, missing = res
            shard_infos.append(info)
            skipped_missing_total += missing
            for k, v in info.source_counts.items():
                global_counts[k] += v

        if workers <= 1:
            for res in _progress((_pack_one_shard(j) for j in jobs), total=len(jobs)):
                _collect(res)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_pack_one_shard, j) for j in jobs]
                for fut in _progress(as_completed(futures), total=len(futures)):
                    _collect(fut.result())

    # Sort shards by name so the index/fingerprints are deterministic regardless of completion order.
    shard_infos.sort(key=lambda s: s.name)

    result.shards = shard_infos
    result.num_shards = len(shard_infos)
    result.num_tiles = sum(s.num_samples for s in shard_infos)
    result.skipped_missing = skipped_missing_total
    result.global_source_counts = dict(global_counts.most_common())
    return result

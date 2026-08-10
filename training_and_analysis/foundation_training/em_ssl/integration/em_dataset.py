"""EM dataset bridge for DINOv3's data pipeline.

DINOv3 builds its training data by calling `make_dataset(dataset_str, transform, target_transform)`
and wrapping the result in `make_data_loader`. ``dinov3_patch`` dispatches an ``EMShards`` dataset
string here, and also makes `_make_sampler` return None for this streaming dataset so
`make_data_loader` builds a plain iterable DataLoader. A config can then say:

    train.dataset_path = "EMShards:root=/data/shards/em_tiles_v0:prefix=em_tiles_v0:min_side=512:shuffle=2000:seed=0"

and DINOv3 will stream single-channel tiles from the WebDataset shards, applying the
EM multi-crop augmentation (passed in as ``transform``) and DINOv3's
``target_transform=lambda _: ()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..data.shard_dataset import EMShardDataset, list_shard_urls

def parse_em_dataset_str(dataset_str: str) -> dict[str, Any]:
    """Parse 'EMShards:root=...:prefix=...:min_side=...:shuffle=...:seed=...'.

    Robust to ``:`` inside values (e.g. a Windows drive letter in ``root=``): tokens after
    the name that lack ``=`` are treated as continuations of the previous value and
    rejoined with ``:``. Harmless on POSIX paths (no drive colon).
    """
    tokens = dataset_str.split(":")
    name = tokens[0]
    kw: dict[str, Any] = {}
    last_key: str | None = None
    for tok in tokens[1:]:
        if tok == "":
            # empty token (a trailing colon or '::') — attach as ':' to the value
            if last_key is not None:
                kw[last_key] = f"{kw[last_key]}:"
            continue
        if "=" in tok:
            k, _, v = tok.partition("=")
            kw[k] = v
            last_key = k
        elif last_key is not None:
            kw[last_key] = f"{kw[last_key]}:{tok}"
    return {"name": name, **kw}

def make_em_dataset(
    dataset_str: str,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
):
    """Build an EMShardDataset (IterableDataset) from a DINOv3 dataset string."""
    cfg = parse_em_dataset_str(dataset_str)
    root = cfg.get("root")
    if not root:
        raise ValueError(f"EMShards dataset string missing root=: {dataset_str!r}")
    prefix = cfg.get("prefix", "em_tiles_v0")
    urls_arg = cfg.get("urls")
    if urls_arg:
        urls = urls_arg.split(",")
    else:
        urls = list_shard_urls(root, prefix)
    if not urls:
        raise FileNotFoundError(f"No shards found under {root} with prefix {prefix}")

    # FINO mode: emit per-sample metadata (image, ((), EMTileMetadata)) so the upstream FINO
    # collate can batch it for the guide heads. This overrides DINOv3's target_transform (which
    # drops the target). Read the live ACTIVE_FINO module global so the value set by the runner
    # is seen here even when this module was imported earlier.
    from . import dinov3_patch

    runtime = getattr(dinov3_patch, "ACTIVE_FINO", None)
    if runtime is not None and getattr(runtime, "enabled_factors", None):
        target_transform = runtime.target_transform()

    return EMShardDataset(
        urls=urls,
        transform=transform,
        target_transform=target_transform,
        min_side=int(cfg.get("min_side", 0) or 0),
        resampled=str(cfg.get("resampled", "1")) not in ("0", "false", "False"),
        shuffle_buffer=int(cfg.get("shuffle", 2000) or 0),
        seed=int(cfg.get("seed", 0) or 0),
    )

def is_em_dataset_str(dataset_str: str | None) -> bool:
    return bool(dataset_str) and str(dataset_str).split(":")[0] in ("EMShards", "EMShardDataset", "EM")

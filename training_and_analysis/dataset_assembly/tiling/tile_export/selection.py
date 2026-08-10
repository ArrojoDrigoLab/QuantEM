"""Per-source tile cap.

An asset that yields more accepted tiles than the cap has the surplus rejected.
Selection is spatially stratified rather than score-ranked: accepted tiles are
binned into a 10x10 grid over the plane (separately per z), each bin is ordered
by descending tissue score, and bins are then visited round-robin taking the best
remaining tile from each until the cap is reached. Taking the top-N by score alone
would concentrate the kept tiles in whichever region of the asset happened to
score highest.

Bin visiting order and score ties are both broken by a seeded digest, so the
selection is deterministic for a given config seed.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, MutableMapping

from .config import TileExportConfig, stable_json
from .identity import seeded_digest

BINS_PER_AXIS = 10


def select_capped_tile_ids(
    accepted: Iterable[MutableMapping[str, Any]],
    *,
    cap: int,
    image_width: int,
    image_height: int,
    config: TileExportConfig,
) -> set[str]:
    """Return the tile_ids to keep, at most ``cap`` of them."""
    x_bin = max(1, math.ceil(max(int(image_width), 1) / BINS_PER_AXIS))
    y_bin = max(1, math.ceil(max(int(image_height), 1) / BINS_PER_AXIS))

    bins: dict[tuple[int, int, int], list[MutableMapping[str, Any]]] = {}
    for record in accepted:
        key = (
            int(record.get("z") or 0),
            int(record["x"]) // x_bin,
            int(record["y"]) // y_bin,
        )
        bins.setdefault(key, []).append(record)

    for bucket in bins.values():
        bucket.sort(
            key=lambda r: (
                -float(r["tissue_score"]),
                seeded_digest(str(r["tile_id"]), seed=config.seed),
            )
        )

    keep: set[str] = set()
    order = sorted(bins, key=lambda k: seeded_digest(stable_json(k), seed=config.seed))
    while len(keep) < cap and order:
        remaining = []
        for key in order:
            bucket = bins[key]
            if not bucket:
                continue
            keep.add(str(bucket.pop(0)["tile_id"]))
            if len(keep) >= cap:
                break
            if bucket:
                remaining.append(key)
        order = remaining
    return keep


def apply_source_cap(
    records: list[MutableMapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
    config: TileExportConfig,
) -> int:
    """Reject accepted tiles above the per-source cap, in place. Returns the number rejected."""
    accepted = [r for r in records if r["status"] == "accepted"]
    cap = int(config.max_tiles_per_source)
    if len(accepted) <= cap:
        return 0

    keep = select_capped_tile_ids(
        accepted,
        cap=cap,
        image_width=image_width,
        image_height=image_height,
        config=config,
    )
    rejected = 0
    for record in accepted:
        if str(record["tile_id"]) in keep:
            continue
        record["status"] = "rejected"
        record["rejection_reason"] = "source_tile_cap"
        rejected += 1
    return rejected

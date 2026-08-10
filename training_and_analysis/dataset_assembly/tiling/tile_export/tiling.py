from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NominalTile:
    index: int
    nominal_x: int
    nominal_y: int
    width: int
    height: int


@dataclass(frozen=True)
class CandidateTile:
    nominal: NominalTile
    x: int
    y: int
    width: int
    height: int
    shift_x: int
    shift_y: int


# An edge tile may never overlap its neighbour by more than this fraction of a tile;
# if reaching the image edge would exceed it, the edge is undertiled (the sliver is left uncovered)
# rather than stamp a near-duplicate tile. This is the fallback for callers that do not supply
# their own cap; the tiling driver passes tile_asset.OVERLAP_CAP instead.
MAX_OVERLAP_FRACTION = 0.65


def sliding_window_starts(
    length: int, tile_size: int, stride: int, max_overlap_fraction: float = MAX_OVERLAP_FRACTION
) -> list[int]:
    length = int(length)
    tile_size = int(tile_size)
    stride = max(1, int(stride))
    if length <= 0:
        return [0]
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last_start = max(length - tile_size, 0)
    if not starts:
        return [last_start]
    if starts[-1] != last_start:
        # Add the edge tile only if it advances enough that its overlap with the previous
        # tile stays <= max_overlap_fraction; otherwise undertile this edge.
        min_advance = max(1, int(round(tile_size * (1.0 - max_overlap_fraction))))
        if last_start - starts[-1] >= min_advance:
            starts.append(last_start)
    return starts


def candidate_shift_values(max_shift: int) -> list[int]:
    max_shift = max(0, int(max_shift))
    if max_shift == 0:
        return [0]
    half_shift = max(1, int(round(max_shift / 2.0)))
    values = [-max_shift, -half_shift, 0, half_shift, max_shift]
    return sorted(set(values))


def iter_candidate_tiles(
    nominal: NominalTile,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    max_shift: int,
) -> list[CandidateTile]:
    tile_width = min(int(tile_size), max(1, int(image_width)))
    tile_height = min(int(tile_size), max(1, int(image_height)))
    max_x = max(0, int(image_width) - tile_width)
    max_y = max(0, int(image_height) - tile_height)
    shifts = candidate_shift_values(max_shift)
    candidates: list[CandidateTile] = []
    seen: set[tuple[int, int]] = set()
    for shift_y in shifts:
        for shift_x in shifts:
            x = min(max(int(nominal.nominal_x) + int(shift_x), 0), max_x)
            y = min(max(int(nominal.nominal_y) + int(shift_y), 0), max_y)
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CandidateTile(
                    nominal=nominal,
                    x=x,
                    y=y,
                    width=tile_width,
                    height=tile_height,
                    shift_x=x - int(nominal.nominal_x),
                    shift_y=y - int(nominal.nominal_y),
                )
            )
    return candidates


def evenly_spaced_indices(
    *,
    count: int,
    selected_count: int,
    minimum_spacing: int = 1,
    include_middle: bool = True,
) -> list[int]:
    count = int(count)
    selected_count = int(selected_count)
    minimum_spacing = max(1, int(minimum_spacing))
    if count <= 0 or selected_count <= 0:
        return []
    if count == 1:
        return [0]

    selected_count = min(selected_count, count)
    if minimum_spacing > 1:
        max_by_spacing = int(math.floor((count - 1) / minimum_spacing)) + 1
        selected_count = max(1, min(selected_count, max_by_spacing))

    if selected_count == 1:
        return [count // 2 if include_middle else 0]
    values = {
        int(round(position))
        for position in [
            index * ((count - 1) / float(selected_count - 1))
            for index in range(selected_count)
        ]
    }
    values.add(0)
    values.add(count - 1)
    if include_middle and len(values) < selected_count:
        values.add(count // 2)

    ordered = sorted(values)
    while len(ordered) > selected_count:
        removable = [value for value in ordered if value not in {0, count - 1}]
        if not removable:
            break
        target = count / 2.0
        remove_value = max(removable, key=lambda value: abs(value - target))
        ordered.remove(remove_value)

    while len(ordered) < selected_count:
        candidates = [value for value in range(count) if value not in set(ordered)]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda value: (
                min(abs(value - existing) for existing in ordered),
                -abs(value - (count / 2.0)),
                -value,
            ),
        )
        ordered.append(best)
        ordered.sort()

    if minimum_spacing <= 1:
        return ordered
    spaced: list[int] = []
    for value in ordered:
        if not spaced or value - spaced[-1] >= minimum_spacing:
            spaced.append(value)
    if ordered[-1] not in spaced:
        spaced[-1:] = [ordered[-1]]
    return spaced[:selected_count]

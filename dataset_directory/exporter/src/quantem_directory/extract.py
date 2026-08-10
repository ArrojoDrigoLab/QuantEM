"""Read the corpus extract into memory.

The extract is a set of CSVs produced by a read-only pass over the corpus
database. This module is the only place that knows their filenames and column
names, so a future re-export only has to satisfy this shape.

Nothing here filters or interprets — see :mod:`derive` for the rules and
:mod:`build` for what becomes public.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Some extract columns hold long free-text values; the stdlib default field
# limit is smaller than the widest of them.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass
class Asset:
    id: str
    name: str
    dataset_id: str
    width: int | None
    height: int | None
    depth: int | None
    resolution_field: str
    inplane_nm: float | None = None
    tiles: int = 0
    # tag group -> values, restricted to published groups
    tags: dict[str, list[str]] = field(default_factory=dict)
    # Inputs to the dimensionality rule, and the extract's own answer to
    # cross-check it against. Neither is published.
    dimensionality_tags: list[str] = field(default_factory=list)
    extract_dim: str = ""


@dataclass
class Dataset:
    id: str
    name: str
    experiment_name: str
    url: str | None
    doi: str


@dataclass
class Extract:
    assets: list[Asset]
    datasets: list[Dataset]
    tag_groups_seen: set

    def by_dataset(self) -> dict[str, list[Asset]]:
        out: dict[str, list[Asset]] = {d.id: [] for d in self.datasets}
        for asset in self.assets:
            out.setdefault(asset.dataset_id, []).append(asset)
        return out


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def load(
    extract_dir: Path,
    *,
    urls_csv: Path | None = None,
    exclude_datasets: set | None = None,
    vocabulary_overrides: dict[str, dict[str, str | None]] | None = None,
    link_overrides: dict[str, str] | None = None,
) -> Extract:
    """Load the extract rooted at ``extract_dir``.

    ``urls_csv`` supplies the three separate DOI/URL columns that the extract's
    ``datasets.csv`` collapses; without it, only the dataset's own DOI is
    available and datasets whose link comes from their experiment lose it.

    ``exclude_datasets`` drops datasets by name. It exists for corpus entries
    that are deliberately not part of the published inventory.

    ``vocabulary_overrides`` rewrites or drops individual facet values, mapping
    ``{group: {original: replacement_or_None}}``. Dropping affects only that
    value on that asset.

    ``link_overrides`` maps a dataset name to a public URL, and wins over the
    link derived from the extract. It exists so a dataset can be given its
    repository link as its deposition completes, without waiting for a fresh
    corpus export.

    Names arrive already fit to publish: the extract is produced from the corpus
    under the names each dataset is deposited and catalogued under, so nothing
    here rewrites them.
    """
    from . import allowlist, derive

    extract_dir = Path(extract_dir)
    exclude = {n.strip() for n in (exclude_datasets or set())}

    # ---- datasets -------------------------------------------------------
    url_rows = {r["dataset_id"]: r for r in _rows(Path(urls_csv))} if urls_csv else {}

    datasets: list[Dataset] = []
    dropped_dataset_ids = set()
    for row in _rows(extract_dir / "datasets.csv"):
        name = (row["name"] or "").strip()
        if name in exclude:
            dropped_dataset_ids.add(row["dataset_id"])
            continue
        link = url_rows.get(row["dataset_id"], {})
        override = (link_overrides or {}).get(name)
        datasets.append(
            Dataset(
                id=row["dataset_id"],
                name=name,
                experiment_name=(row.get("experiment_name") or "").strip(),
                url=override
                or derive.reference_url(
                    dataset_doi=link.get("dataset_doi") or row.get("doi") or "",
                    source_url=link.get("source_url") or "",
                    experiment_doi=link.get("experiment_doi") or "",
                ),
                doi=(link.get("dataset_doi") or row.get("doi") or "").strip(),
            )
        )
    known_datasets = {d.id for d in datasets}

    # ---- per-asset side tables -----------------------------------------
    inplane: dict[str, float] = {}
    for row in _rows(extract_dir / "derived" / "asset_inplane_resolution_nm.csv"):
        # The id column is unnamed in the extract.
        asset_id = row.get("") or row.get("asset_id") or ""
        value = derive.parse_float(row.get("inplane_nm"))
        if asset_id and value is not None:
            inplane[asset_id] = value

    tiles: dict[str, int] = {}
    for row in _rows(extract_dir / "asset_tiles.csv"):
        tiles[row["asset_id"]] = derive.parse_int(row.get("accepted_tiles")) or 0

    extract_dim: dict[str, str] = {}
    for row in _rows(extract_dir / "derived" / "asset_tidy.csv"):
        extract_dim[row["asset_id"]] = (row.get("dim") or "").strip()

    # ---- tags -----------------------------------------------------------
    # Every group is recorded so the allow-list can assert it has a ruling for
    # each, but only published groups are carried forward.
    overrides = {
        group: values
        for group, values in (vocabulary_overrides or {}).items()
        if not group.startswith("_")
    }
    groups_seen = set()
    tags: dict[str, dict[str, list[str]]] = {}
    dimensionality: dict[str, list[str]] = {}
    for row in _rows(extract_dir / "asset_tag_long.csv"):
        group = (row["group"] or "").strip()
        groups_seen.add(group)
        if group == "dimensionality":
            dimensionality.setdefault(row["asset_id"], []).append(row["name"])
        if group not in allowlist.PUBLISHED_TAG_GROUPS:
            continue
        value = row["name"]
        if group in overrides and value in overrides[group]:
            value = overrides[group][value]
            if value is None:
                continue
        if not allowlist.is_publishable_vocabulary_value(value):
            # Curated facet labels are held to a higher bar than dataset names.
            # Dropping by shape rather than by listing the value keeps the rule
            # working as the corpus grows, and keeps the value out of this repo.
            continue
        tags.setdefault(row["asset_id"], {}).setdefault(group, []).append(value)

    allowlist.check_tag_groups(groups_seen)

    # ---- assets ---------------------------------------------------------
    assets: list[Asset] = []
    for row in _rows(extract_dir / "asset_meta.csv"):
        dataset_id = row["dataset_id"]
        if dataset_id in dropped_dataset_ids or dataset_id not in known_datasets:
            continue
        asset_id = row["asset_id"]
        assets.append(
            Asset(
                id=asset_id,
                name=(row.get("display_name") or "").strip(),
                dataset_id=dataset_id,
                width=derive.parse_int(row.get("width")),
                height=derive.parse_int(row.get("height")),
                depth=derive.parse_int(row.get("depth")),
                resolution_field=(row.get("resolution_field") or "").strip(),
                inplane_nm=inplane.get(asset_id),
                tiles=tiles.get(asset_id, 0),
                tags={g: sorted(set(v)) for g, v in tags.get(asset_id, {}).items()},
                dimensionality_tags=dimensionality.get(asset_id, []),
                extract_dim=extract_dim.get(asset_id, ""),
            )
        )

    return Extract(assets=assets, datasets=datasets, tag_groups_seen=groups_seen)

"""Turn the corpus extract into the static files the site fetches.

Three shapes, chosen for what reads them:

* ``datasets.json`` stays row-oriented. Six hundred rows is small enough that
  being readable in a diff is worth more than the bytes.
* ``assets.json`` is columnar and dictionary-encoded. The site loads it into
  typed arrays and filters the whole corpus on every keystroke, so the wire
  format matches the in-memory format.
* Multi-valued facets are flattened to ``offsets``/``values`` pairs rather than
  arrays of arrays. An asset may carry several species or several organs, and
  this is both smaller on the wire and directly usable as typed arrays.

``facets.json`` carries the dictionaries the columns index into, so they are
fetched once and shared.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from . import SCHEMA_VERSION, derive
from .extract import Extract

# Facet key -> (source tag group, display label). Order is display order.
TREE_FACETS = (
    ("taxonomy", ("kingdom", "species"), "Kingdom and species"),
    ("anatomy", ("organ", "Tissue Region"), "Organ and tissue context"),
)

#: Shown when an asset carries no value for a facet at all. Kept selectable so
#: those assets stay reachable.
NO_VALUE_LABEL = "Not recorded"


def _csr(rows: Sequence[Sequence[int]]) -> dict[str, list[int]]:
    """Flatten variable-length integer rows into offsets and values."""
    offsets = [0]
    values: list[int] = []
    for row in rows:
        values.extend(row)
        offsets.append(len(values))
    return {"offsets": offsets, "values": values}


def _dictionary(counts: Counter) -> list[str]:
    """Order a facet vocabulary by frequency, then alphabetically."""
    return [label for label, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].casefold()))]


def _write_json(path: Path, payload: object, *, compact: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def build(
    extract: Extract,
    out_dir: Path,
    *,
    thumb_ids: Iterable[str] | None = None,
    thumb_meta: dict | None = None,
    source_snapshot: str = "",
) -> dict:
    """Write every published artifact into ``out_dir`` and return a report."""
    out_dir = Path(out_dir)
    thumbs = set(thumb_ids or ())

    datasets = sorted(extract.datasets, key=lambda d: d.name.casefold())
    dataset_index = {d.id: i for i, d in enumerate(datasets)}
    assets = [a for a in extract.assets if a.dataset_id in dataset_index]
    assets.sort(key=lambda a: (dataset_index[a.dataset_id], a.name.casefold()))

    repository_of_dataset = {d.id: derive.repository_of(d.url) for d in datasets}

    # ---- vocabularies ---------------------------------------------------
    counts: dict[str, Counter] = {
        "kingdom": Counter(),
        "species": Counter(),
        "organ": Counter(),
        "Tissue Region": Counter(),
        "modality": Counter(),
        "repository": Counter(),
    }
    for asset in assets:
        for group in ("kingdom", "species", "organ", "Tissue Region"):
            for value in asset.tags.get(group, ()):
                counts[group][value] += 1
        modality = asset.tags.get("modality") or []
        if modality:
            counts["modality"][modality[0]] += 1
        counts["repository"][repository_of_dataset[asset.dataset_id]] += 1

    vocab = {group: _dictionary(counter) for group, counter in counts.items()}
    lookup = {group: {v: i for i, v in enumerate(values)} for group, values in vocab.items()}
    resolution_labels = derive.resolution_band_labels()
    resolution_lookup = {label: i for i, label in enumerate(resolution_labels)}

    # ---- asset columns --------------------------------------------------
    columns: dict[str, list] = {k: [] for k in ("id", "name", "dataset", "w", "h", "z", "nm")}
    single: dict[str, list] = {k: [] for k in ("modality", "resolution", "dim", "repository", "thumb")}
    multi_rows: dict[str, list[list[int]]] = {
        "kingdom": [],
        "species": [],
        "organ": [],
        "Tissue Region": [],
    }

    per_dataset = defaultdict(lambda: {"n2d": 0, "n3d": 0, "n": 0})
    resolution_counts: Counter = Counter()
    dimension_counts = Counter()
    modality_missing = 0

    for asset in assets:
        three_d = derive.is_three_dimensional(
            dimensionality_tags=asset.dimensionality_tags,
            resolution_field=asset.resolution_field,
            depth=asset.depth,
        )
        band = derive.resolution_band(asset.inplane_nm)

        columns["id"].append(asset.id.replace("-", ""))
        columns["name"].append(asset.name)
        columns["dataset"].append(dataset_index[asset.dataset_id])
        columns["w"].append(asset.width)
        columns["h"].append(asset.height)
        columns["z"].append(asset.depth if (asset.depth or 0) > 1 else None)
        columns["nm"].append(asset.inplane_nm)

        modality = asset.tags.get("modality") or []
        if modality:
            single["modality"].append(lookup["modality"][modality[0]])
        else:
            single["modality"].append(None)
            modality_missing += 1
        single["resolution"].append(resolution_lookup[band])
        single["dim"].append(1 if three_d else 0)
        single["repository"].append(lookup["repository"][repository_of_dataset[asset.dataset_id]])
        single["thumb"].append(1 if asset.id.replace("-", "") in thumbs else 0)

        for group in multi_rows:
            multi_rows[group].append(
                sorted(lookup[group][v] for v in asset.tags.get(group, ()))
            )

        bucket = per_dataset[asset.dataset_id]
        bucket["n3d" if three_d else "n2d"] += 1
        bucket["n"] += 1
        resolution_counts[band] += 1
        dimension_counts["3D" if three_d else "2D"] += 1

    # ---- dataset rows ---------------------------------------------------
    hero_pool: dict[str, list[tuple]] = defaultdict(list)
    for asset in assets:
        if asset.id.replace("-", "") in thumbs:
            hero_pool[asset.dataset_id].append((-asset.tiles, asset.name, asset.id.replace("-", "")))

    dataset_rows = []
    for dataset in datasets:
        aggregate = per_dataset.get(dataset.id, {"n2d": 0, "n3d": 0, "n": 0})
        heroes = [hex_id for _, _, hex_id in sorted(hero_pool.get(dataset.id, []))[:4]]
        dataset_rows.append(
            {
                "id": dataset.id,
                "name": dataset.name,
                "experiment": dataset.experiment_name,
                "url": dataset.url,
                "doi": dataset.doi or None,
                "repository": lookup["repository"][repository_of_dataset[dataset.id]],
                "n2d": aggregate["n2d"],
                "n3d": aggregate["n3d"],
                "n": aggregate["n"],
                "hero": heroes,
            }
        )

    # ---- facets ---------------------------------------------------------
    facets = []
    for key, (parent_group, child_group), label in TREE_FACETS:
        pair_counts: dict[int, Counter] = defaultdict(Counter)
        parent_counts: Counter = Counter()
        for asset in assets:
            parents = asset.tags.get(parent_group, ())
            children = asset.tags.get(child_group, ())
            for parent in parents:
                parent_counts[lookup[parent_group][parent]] += 1
                for child in children:
                    pair_counts[lookup[parent_group][parent]][lookup[child_group][child]] += 1
        roots = []
        for parent_id, total in sorted(
            parent_counts.items(), key=lambda kv: (-kv[1], vocab[parent_group][kv[0]].casefold())
        ):
            children = [
                {"id": child_id, "label": vocab[child_group][child_id], "n": n}
                for child_id, n in sorted(
                    pair_counts[parent_id].items(),
                    key=lambda kv: vocab[child_group][kv[0]].casefold(),
                )
            ]
            roots.append(
                {
                    "id": parent_id,
                    "label": vocab[parent_group][parent_id],
                    "n": total,
                    "children": children,
                }
            )
        facets.append(
            {
                "key": key,
                "label": label,
                "kind": "tree",
                "ranks": [parent_group, "tissue context" if child_group == "Tissue Region" else child_group],
                "parentDictionary": parent_group,
                "childDictionary": child_group,
                "roots": roots,
            }
        )

    def flat_facet(key: str, label: str, dictionary: str, tally: Counter, missing: int = 0) -> dict:
        return {
            "key": key,
            "label": label,
            "kind": "flat",
            "dictionary": dictionary,
            "values": [
                {"id": i, "label": name, "n": tally[name]}
                for i, name in enumerate(vocab.get(dictionary, resolution_labels))
                if tally[name]
            ],
            "noValue": {"label": NO_VALUE_LABEL, "n": missing} if missing else None,
        }

    facets.append(flat_facet("modality", "Imaging modality", "modality", counts["modality"], modality_missing))
    facets.append(
        {
            "key": "resolution",
            "label": "In-plane resolution",
            "kind": "flat",
            "dictionary": "resolution",
            "values": [
                {"id": i, "label": name, "n": resolution_counts[name]}
                for i, name in enumerate(resolution_labels)
                if resolution_counts[name]
            ],
            "noValue": None,
        }
    )
    facets.append(
        {
            "key": "dim",
            "label": "Dimensionality",
            "kind": "flat",
            "dictionary": "dimensionality",
            "values": [
                {"id": 0, "label": "2D image", "n": dimension_counts["2D"]},
                {"id": 1, "label": "3D acquisition", "n": dimension_counts["3D"]},
            ],
            "noValue": None,
        }
    )
    facets.append(flat_facet("repository", "Source repository", "repository", counts["repository"]))

    dictionaries = dict(vocab)
    dictionaries["resolution"] = resolution_labels
    dictionaries["dimensionality"] = ["2D image", "3D acquisition"]

    # ---- write ----------------------------------------------------------
    sizes = {}
    sizes["facets.json"] = _write_json(
        out_dir / "facets.json",
        {"schema_version": SCHEMA_VERSION, "dictionaries": dictionaries, "facets": facets},
        compact=False,
    )
    sizes["datasets.json"] = _write_json(
        out_dir / "datasets.json",
        {"schema_version": SCHEMA_VERSION, "rows": dataset_rows},
        compact=False,
    )
    sizes["assets.json"] = _write_json(
        out_dir / "assets.json",
        {
            "schema_version": SCHEMA_VERSION,
            "n": len(assets),
            "columns": columns,
            "single": single,
            "multi": {group: _csr(rows) for group, rows in multi_rows.items()},
        },
        compact=True,
    )

    csv_path = out_dir / "datasets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Dataset", "Experiment", "Repository", "URL", "2D images", "3D acquisitions", "Assets"]
        )
        for row in dataset_rows:
            writer.writerow(
                [
                    row["name"],
                    row["experiment"],
                    dictionaries["repository"][row["repository"]],
                    row["url"] or "",
                    row["n2d"],
                    row["n3d"],
                    row["n"],
                ]
            )
    sizes["datasets.csv"] = csv_path.stat().st_size

    corpus_counts = {
        "datasets": len(datasets),
        "assets": len(assets),
        "images_2d": dimension_counts["2D"],
        "volumes_3d": dimension_counts["3D"],
        "kingdoms": len(vocab["kingdom"]),
        "species": len(vocab["species"]),
        "organs": len(vocab["organ"]),
        "tissue_contexts": len(vocab["Tissue Region"]),
        "modalities": len(vocab["modality"]),
        "repositories": len(vocab["repository"]),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot": source_snapshot,
        "export_id": hashlib.sha256(
            json.dumps(corpus_counts, sort_keys=True).encode() + source_snapshot.encode()
        ).hexdigest()[:12],
        "counts": corpus_counts,
        "thumbnails": thumb_meta,
    }
    sizes["manifest.json"] = _write_json(out_dir / "manifest.json", manifest, compact=False)

    return {"counts": corpus_counts, "sizes": sizes, "manifest": manifest}

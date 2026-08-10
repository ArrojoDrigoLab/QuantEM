# The published data contract

## `manifest.json`

Identifies the export and points at its thumbnails.

```jsonc
{
  "schema_version": "1.0.0",
  "source_snapshot": "2026-08-07",   // the corpus snapshot this was built from
  "export_id": "9f378c957162",       // changes when any published count changes
  "counts": { "datasets": 0, "assets": 0, "images_2d": 0, "volumes_3d": 0,
              "kingdoms": 0, "species": 0, "organs": 0, "tissue_contexts": 0,
              "modalities": 0, "repositories": 0 },   // actual totals, whatever the corpus holds
  "thumbnails": { "px": 256, "format": "webp", "quality": 75, "count": 0,
                  "bytes": 0, "archive": "thumbs-256-v1.tar.gz",
                  "sha256": "…", "release_tag": "directory-thumbs-v1",
                  "archive_bytes": 0 }
}
```

## `facets.json`

The vocabularies the asset columns index into, and the facets built from them.

```jsonc
{
  "dictionaries": {
    "kingdom": ["Animalia", "Plantae", …],     // index -> label, ordered by frequency
    "species": […], "organ": […], "Tissue Region": […],
    "modality": […], "repository": […],
    "resolution": ["< 1 nm/px", …, "Unknown"], // display order; Unknown last
    "dimensionality": ["2D image", "3D acquisition"]
  },
  "facets": [
    { "key": "taxonomy", "kind": "tree", "ranks": ["kingdom", "species"],
      "parentDictionary": "kingdom", "childDictionary": "species",
      "roots": [ { "id": 0, "label": "Animalia", "n": 15964,
                   "children": [ { "id": 4, "label": "Mus musculus", "n": 4102 } ] } ] },
    { "key": "modality", "kind": "flat", "dictionary": "modality",
      "values": [ { "id": 0, "label": "TEM", "n": 15711 } ],
      "noValue": { "label": "Not recorded", "n": 119 } }
  ]
}
```

`roots[].children[].n` is the count of assets carrying *both* that
parent and that child. A child may appear under several parents with a different count under each,
because an image can contain multiple species/kingdoms and a tissue context potentially under more \
than one organ as presented here. Summing children does not give the parent's `n`.

`noValue` is present only when some assets have no value for that facet at all. It is selectable. 

## `datasets.json`

One row per dataset, ordered by name.

```jsonc
{ "rows": [ {
    "id": "003739fc-f71b-49e6-a55d-eb2a12dacd6f",  // stable across exports; used in permalinks
    "name": "Mouse Lateral Geniculate Nucleus SBEM",
    "experiment": "Mouse Lateral Geniculate Nucleus SBF-SEM",
    "url": "https://doi.org/10.60533/BOSS-2023-3TJJ", // null when not yet deposited
    "doi": "10.60533/BOSS-2023-3TJJ",                 // null when there is none
    "repository": 3,                                  // index into dictionaries.repository
    "n2d": 0, "n3d": 2, "n": 2,                       // n2d + n3d == n, always
    "hero": ["e6ef312c…", …]                          // up to 4 asset ids that have a thumbnail
} ] }
```

## `assets.json`

Every asset, columnar and dictionary-encoded — the same shape the site keeps in memory, so
loading is a copy rather than a transform. 

```jsonc
{
  "n": 0,                             // asset count, whatever the corpus holds
  "columns": {
    "id":      ["e6ef312cc2e84317bc8c819b62d0f77a", …],  // 32 hex chars, no dashes
    "name":    ["liver volume 01", …],
    "dataset": [117, …],        // index into datasets.json rows
    "w": [2048, …], "h": [2119, …],
    "z": [310, …],              // null for 2D
    "nm": [1.59, …]             // null when no in-plane resolution could be parsed
  },
  "single": {
    "modality":   [0, …],       // dictionary index, or -1 for "not recorded"
    "resolution": [2, …],       // band index; never -1, Unknown is a real band
    "dim":        [0, …],       // 0 = 2D image, 1 = 3D acquisition
    "repository": [3, …],
    "thumb":      [1, …]        // 1 if thumbs/<id[0:2]>/<id>.webp exists
  },
  "multi": {
    "kingdom": { "offsets": [0, 1, 2, …], "values": [0, 0, 1, …] },
    "species": {…}, "organ": {…}, "Tissue Region": {…}
  }
}
```

Multi-valued facets are flattened: asset `i` holds `values[offsets[i] … offsets[i+1]]`. `offsets`
has `n + 1` entries and its last entry equals `values.length`. 

Thumbnail path: `thumbs/<first two characters of id>/<id>.webp`. Check `single.thumb` first; the
directory does not contain a file for every asset.

## `datasets.csv`

The dataset inventory as a spreadsheet:
`Dataset, Experiment, Repository, URL, 2D images, 3D acquisitions, Assets`.

---

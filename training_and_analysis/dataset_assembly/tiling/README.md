# `tiling/`

Cutting EM assets into the tiles used for foundation-model pretraining.

---

## Usage

```bash
# one asset
python tile_asset.py --asset /data/em/liver_fibsem.tif --out ./tiles

# a batch: CSV with a `path` column, plus optional `source_id` and `z_nm`
python tile_asset.py --asset-list assets.csv --out ./tiles

# collect the sidecars into the pretraining manifest
python build_manifest.py --tiles ./tiles --out manifest.jsonl

# and the corpus composition, with facets from a per-asset metadata CSV
python build_manifest.py --tiles ./tiles --out manifest.jsonl \
    --summary composition.json --asset-meta asset_meta.csv

# attach per-asset catalogue facts (licence, modality, organ, pixel size) to every tile
python join_asset_metadata.py --manifest manifest.jsonl \
    --asset-meta asset_meta.csv --out manifest_enriched.jsonl
```

`--out` from `tile_asset.py` is the same directory `build_manifest.py` takes as `--tiles`.

## Files

| File | What it is |
|---|---|
| `tile_asset.py` | Driver: one asset (2D image or 3D stack) → accepted tiles + JSON sidecars |
| `build_manifest.py` | Sidecars → pretraining manifest, and optionally the corpus composition |
| `join_asset_metadata.py` | Attaches per-asset catalogue facts to every tile of that asset |
| `tile_export/config.py` | `TileExportConfig`: every tiling parameter, and the identity digest |
| `tile_export/filtering.py` | Tissue-content scoring, content cropping, accept/borderline/reject |
| `tile_export/tiling.py` | Nominal grid, candidate shifts, even plane spacing |
| `tile_export/normalization.py` | Source-percentile normalization to uint8 |
| `tile_export/selection.py` | The per-source tile cap |
| `tile_export/identity.py` | Tile IDs and the seeded digest used for deterministic tie-breaks |
| `RULES.md` | The parameters as used |

## Inputs

**2D images as PNG or TIFF; 3D stacks as multi-page or OME-TIFF.** Assets are read from disk —
nothing here queries a database or a catalog.

Repository-native formats — zarr, N5, precomputed volumes, MRC, DM3/DM4, IMS — must be converted to
PNG or TIFF first. The corpus was tiled from already-converted assets.

`z_nm` matters for 3D volumes: with it, planes are selected at a fixed physical spacing; without it,
they are spread evenly under the tile budget, which selects a different set of planes. It is
available in the source repository's own metadata for most volumes.

The batch CSV takes one asset per row:

```csv
path,source_id,z_nm
/data/em/liver_fibsem.tif,liver_fibsem,8
/data/em/islet_tem.png,islet_tem,
```

`source_id` defaults to the filename stem.

## Outputs

```
<out>/tiler_config.json                          the parameters this run used
<out>/tiles/source_id=<source_id>/*.png          accepted tiles, uint8
<out>/tiles/source_id=<source_id>/*.json         one sidecar per tile
```

Each sidecar carries the tile's position, dimensions, tissue score and component fractions, status
and any rejection reason, the normalization applied, and the tile ID. `build_manifest.py` reads them
directly, so no shared index is written during tiling.

`--summary` reports tile and asset counts by dimensionality, plus one breakdown per extra column in
`--asset-meta` (a CSV keyed by `source_id`) — modality, organ, species, or whatever else is supplied.

## Per-asset metadata

Tiles carry only what is measurable from the pixels. Facts about the asset they came from live in the
dataset catalogue, and `join_asset_metadata.py` copies them onto every tile of that asset:

```csv
source_id,license,modality,organ,species,effective_nm_per_px,source_kind
liver_fibsem,CC BY 4.0,FIB-SEM,Liver,Mus musculus,8,public
islet_tem,CC0 1.0,TEM,Pancreas,Homo sapiens,2.2,internal
```

Any column other than `source_id` is joined through. Pretraining filters the corpus on `license` and
`source_kind`, and metadata conditioning reads `effective_nm_per_px`, `modality` and `organ` — none
of which can be derived from a tile alone. Unmatched assets pass through unchanged unless
`--require-match` is given.

## Notes

- Two passes: the first scores every candidate tile and applies the cap, the second writes only the
  survivors. Planes are cropped to their non-zero content identically in both, so coordinates address
  the same pixels.
- Tiles are written uint8, percentile-normalized per source (2D) or per plane (3D). The further
  standardization to zero mean and unit variance, the random rescale, the random crop and the
  augmentations all belong to the training pipeline — see [`RULES.md`](RULES.md#normalization).
- A failed asset is reported and skipped; a batch continues.
- Re-running writes tiles again; remove the output directory for a clean run.

## Reproducing the published tile counts

The published dataset inventory gives per-dataset tile counts. To check against them: obtain the
assets from the repository URLs in that inventory, convert to PNG/TIFF, run `tile_asset.py` over
them with `z_nm` supplied for volumes, then `build_manifest.py`, and sum the per-asset counts within
each dataset. Selection is deterministic under the fixed seed, so the same input pixels give the
same tiles.

Counts are reported per dataset while tiling runs per asset, and a dataset generally holds many
assets.

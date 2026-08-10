# Tiling rules

The parameters applied when cutting EM assets into pretraining tiles. All values are the in-code
defaults of `TileExportConfig` in [`tile_export/config.py`](tile_export/config.py) and the driver
constants in [`tile_asset.py`](tile_asset.py); the code is authoritative.

## Grid, filtering, caps, normalization

| Parameter | Value | Where |
|---|---|---|
| Tile size | 2048 px square, written uint8 PNG | `TileExportConfig.tile_size` |
| Overlap fraction | 0.15 → stride **1741 px**, overlap **307 px** | `overlap_fraction`; derived `stride`, `overlap_px` |
| Min tissue fraction (accept) | 0.50 | `min_tissue_fraction` |
| Borderline tissue fraction | 0.25 — flagged, not accepted | `borderline_tissue_fraction` |
| Max tiles per source | 400 | `max_tiles_per_source` |
| Max tiles per 3D volume | 400 | `max_tiles_per_3d_volume` |
| Max planes per 3D volume | 80 | `max_planes_per_3d_volume` |
| Normalization | source-percentile → uint8, low/high 0.1 / 99.9 pct | `normalization`, `low_percentile`, `high_percentile` |
| Normalization scope | **per source for 2D, per plane for 3D** | `normalization_scope`, resolved by `effective_normalization_scope()` |
| Seed | 1337 | `seed` |
| Max edge overlap before undertiling | 0.65 | `tile_export/tiling.py`, `MAX_OVERLAP_FRACTION` |
| Edge overlap cap applied by the driver | 0.40 | `tile_asset.py`, `OVERLAP_CAP` |
| Min z-spacing for 3D plane selection | 200 nm | `tile_asset.py`, `MIN_Z_SPACING_NM` |

Stride is `round(2048 × (1 − 0.15)) = round(1740.8) = 1741`, leaving 307 px of overlap.

## Normalization

Normalization happens twice, in two different places.

**Here, at tile creation.** Intensity percentiles (0.1 / 99.9) are estimated on a downsampled
analysis surface and used to linearly rescale each tile to uint8:
`(raw − low) × 255 / (high − low)`, clipped. The percentiles are estimated **once per source for 2D
assets and once per plane for 3D volumes**, never per tile, so intensity relationships within an
asset are preserved rather than each tile being independently stretched. A source whose dynamic
range is degenerate is rejected (`low_dynamic_range`) rather than amplified into noise. Contrast
inversion is detected and recorded but not applied (`invert_policy = auto_report_only`). The
resulting uint8 tiles are what is written to disk, and the per-tile mean, standard deviation and
1st/99th percentiles are recorded in each sidecar.

**In the training pipeline, not here.** Tiles are further standardized to zero mean and unit
variance using the corpus statistics at load time, alongside the random rescale, random crop and
augmentation described in [`../../foundation_training/`](../../foundation_training/). 

## Tissue-content score

Computed per candidate tile in [`tile_export/filtering.py`](tile_export/filtering.py), on a
downsampled analysis surface of the plane rather than the full-resolution tile:

```
tissue_score = 0.45·non_background + 0.35·texture + 0.25·gradient − 0.30·artifact
```

clipped to [0, 1]. Per-component rejection reasons are recorded when
`background ≥ 0.50`, `texture < 0.20`, `gradient < 0.15`, or `artifact ≥ 0.20`.

Status follows from the score alone: `accepted` at ≥ 0.50, `borderline` at ≥ 0.25, otherwise
`rejected` (`tile_status()`).

## Tile placement

Nominal tile positions come from a fixed stride grid. Each nominal position is then allowed to shift
by up to `overlap_px` to catch more tissue, and the best-scoring shift is kept, subject to not
overlapping an already-kept tile by more than the driver's edge-overlap cap. Ties are broken by a
seeded digest, so placement is deterministic for a given seed.

Planes are cropped to their non-zero content extent before tiling, so a padded or partially blank
plane does not generate tiles over its padding.

## Per-source cap

A source yielding more than 400 accepted tiles has the surplus rejected with reason
`source_tile_cap`. Selection is spatially stratified, not score-ranked: accepted tiles are binned into
a 10×10 grid over the plane (separately per z), each bin is ordered by descending tissue score, and
bins are visited round-robin taking the best remaining tile from each until the cap is reached. Bin
order and score ties are broken by a seeded digest.

Taking the top 400 by score alone would concentrate the kept tiles in whichever region of the asset
happened to score highest; stratifying keeps them spread across the asset. Implemented in
[`tile_export/selection.py`](tile_export/selection.py).

## 3D plane selection

When the z-voxel size is known, planes are stepped so consecutive selected planes are at least 200 nm
apart in physical depth. When it is unknown, planes are spread evenly across the stack under the
per-source tile budget — the cap divided by the number of tiles one plane yields
(`evenly_spaced_indices`). The plane count is bounded by `max_planes_per_3d_volume`, though the
z-spacing rule usually binds first for real EM volumes.

## Downstream use

These tiles are the pretraining corpus. During pretraining each tile is randomly kept at native
resolution or downscaled, then randomly cropped to the model's context size before augmentation; see
[`../../foundation_training/`](../../foundation_training/).

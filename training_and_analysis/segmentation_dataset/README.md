# `segmentation_dataset/`

Assembly of the ground-truth segmentation corpus — the annotated public datasets turned into
crops, metadata, and the split definitions that the encoder comparison, segmentation training,
and benchmark consume. Distinct from [`dataset_assembly/`](../dataset_assembly/), which covers
the unlabeled pretraining corpus.

---

## Usage

```bash
conda env create -f environment.yml
conda activate quantem-segmentation-dataset
export SEG_CORPUS_ROOT=<corpus root>

# 1. one driver per source dataset -> <corpus root>/<dataset>/{crops/, manifest.json}
python crop_build/run_empiar_10982.py --all          # each driver documents its own flags
python crop_build/openorganelle/oo_gt_batch.py run-all   # then oo_download, oo_tiles_both,
                                                         # oo_finalize

# 2. estimated resolutions for the sources that record none, then the metadata index
python crop_build/patch_estimated_res.py --apply
python splits/build_crops_metadata.py                # -> <corpus root>/crops_metadata.csv

# 3. split definitions -> <corpus root>/splits/
python splits/make_group_splits.py                   # group1_* (encoder), group2_* (training)
python splits/make_benchmark_splits.py               # held-out benchmark test membership
python splits/make_final_splits.py                   # final_* train/val/test

# 4. benchmark tiles (--org both = ER + mitochondria; nucleus and ld run per --org)
python benchmark_tiles/build_benchmark_tiles.py build   --out <tiles root>
python benchmark_tiles/build_benchmark_tiles.py regrid  --out <tiles root> --write
python benchmark_tiles/build_benchmark_tiles.py add-cem --out <tiles root> \
    --cem-zip <cem_mitolab.zip> --cem-xlsx <cem_mitolab_metadata.xlsx>
```

Every path is anchored at `SEG_CORPUS_ROOT`. Most drivers download their source themselves;
`run_empiar_10791`, `run_empiar_13156`, `run_deeppi`, `run_guay`, `run_orgsegnet` and
`run_deepcontact` read a manual download placed under `<corpus root>/_work/` (each driver's
docstring names the layout), and `add-cem` takes the EMPIAR-11037 archive.

## Sources

The source datasets are those listed in Supplementary Table 3, which carries each source's
accession, organelles and licence; every `run_*` driver names its accession and source in its
dataset-metadata block. 

## Contents

| Item | What it is |
|---|---|
| `crop_build/seg_crop.py` | The crop contract: canvas, tiling, z-sampling, manifest records |
| `crop_build/readers.py` | Dependency-free NIfTI and uint8 readers used by the drivers |
| `crop_build/run_<source>.py` | One ingest driver per source dataset |
| `crop_build/openorganelle/` | The OpenOrganelle chain: GT scan, crop download, XY/XZ tiles, manifest |
| `crop_build/patch_estimated_res.py` | Estimated nm/px for sources that record none, flagged as estimated |
| `splits/build_crops_metadata.py` | Per-crop metadata index across both collections |
| `splits/make_group_splits.py` | `group1_{mito,er}` (encoder comparison) and `group2_<organelle>` (segmentation training) |
| `splits/make_benchmark_splits.py` | Benchmark split CSVs (`benchmark_*`); its test membership is reused by `make_final_splits.py` |
| `splits/make_final_splits.py` | `final_<organelle>` train/val/test; test = the benchmark membership |
| `benchmark_tiles/build_benchmark_tiles.py` | Standardized benchmark tiles, the 512 px train regrid, and the CEM train pool |

## Crop rules

- 4096 px canvas; 3D volumes sampled at z-planes ≥ 400 nm apart; non-overlapping grid with one
  final tile flush to the far edge; a tile is kept iff it contains at least one in-scope
  organelle pixel; sources smaller than the canvas are centred and zero-padded; labels keep
  their native encoding and are never interpolated.
- `valid_region` is the non-zero-EM extent inside the canvas — everything outside it is padding.
  Coverage tiers: `full` ≥ 0.999 of the canvas, `partial` ≥ 0.25, else `sparse`.
- OpenOrganelle: window of min(4096, dim) per axis centred on the annotation, no padding; planes
  ≥ 200 nm apart; two orientations per crop (`raw_xy`, `raw_xz`); no resampling.

## Split rules

- **group1 / group2** match the bare organelle tokens `mito|er|nucleus|ld`; the benchmark and
  final splits use the family token sets in `make_benchmark_splits.py`.
- **group1** was used by the encoder comparison. 
- **group2** was used by the segmentation training architecture comparisons. 
- **final** (benchmark): test = the per-organelle `TESTSETS` membership in
  `make_benchmark_splits.py`; train/val = every other same-organelle crop, split ~80/20 by crop
  count. 

## Benchmark tiles

- `build` resamples each crop to the organelle target nm/px (ER 4, mitochondria 8, LD 8,
  nucleus 25) and writes `0/1/255` (background / organelle / ignore) labels; `--native` keeps
  source resolution. The reported benchmarks use the 8 nm mitochondria build and the
  native-resolution ER build.
- `regrid` expands the mitochondria train split on a zero-overlap 512 px grid, keeping a cell
  only if ≥ 50% of it is annotated (non-ignore) and it holds ≥ 256 foreground pixels, under
  per-source caps (each MitoEM sub-volume 600, every other source 1,000, plant set to 15% of
  the total). Val and test are frozen.
- `add-cem` appends CEM-MitoLab tiles as an optional train-only pool (`split_role =
  train_cem_optional`). Source volumes whose names overlap the benchmark test/val pools, the
  corpus, or previously used sources (`wei2020_mitoem`, `kasthuri`, `guay`,
  `openorganelle`/`jrc_*`) and the flagged `bleck`/`cremi`/`perez` volumes are excluded, so not
  all of CEM is used. 

## Scope of the reported runs

The encoder-comparison and segmentation-training experiments were run on subsets of the final
corpus as it stood at the time of each run, and without CEM-MitoLab mitochondria in training:
the training tiles required more surrounding context than most CEM tiles carry. CEM enters
only through the benchmark train pool above. Rebuilding from the current sources
reproduces an approximate version of each tile set. 

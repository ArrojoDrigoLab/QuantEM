# Internal-model benchmark runs

Fine-tuning and evaluation of the QuantEM and OmniEM segmentation models on the held-out
benchmark. Every run trains on the benchmark train split and is evaluated on the **full**
held-out test split (mito 518, ER 359, nucleus 738, LD 446 test crops); all configs set
`max_eval_crops: 0` and `max_region_px: 0`, so nothing is subsampled.

All commands are launched from `training_and_analysis/` so that `segmentation_training` is
importable.

Placeholders used below:

| placeholder | meaning |
|---|---|
| `<tile-root>` | benchmark tile build (8 nm mito/LD, 25 nm nucleus) from `segmentation_dataset/benchmark_tiles/build_benchmark_tiles.py` |
| `<native-tile-root>` | the `--native` tile build from the same script (ER at source resolution) |
| `<seg-data-root>` | destination root for the transcoded harness data groups |
| `<QuantEM ViT-B run dir>` | pretraining run directory of the QuantEM ViT-B encoder |
| `<OmniEM ViT-L run dir>` | pretraining run directory of the OmniEM ViT-L encoder |
| `<output-dir>` | root directory for the eight run outputs |

## 1. Transcode the benchmark tiles into harness data groups

```
python benchmarking/internal_models/tiles_to_harness.py --src-root <native-tile-root> --org er --dst-root <seg-data-root>
python benchmarking/internal_models/tiles_to_harness.py --src-root <tile-root> --org ld --dst-root <seg-data-root> --min-fg-train-px 1
python benchmarking/internal_models/tiles_to_harness.py --src-root <tile-root> --org nucleus --dst-root <seg-data-root>
python benchmarking/internal_models/build_mito_cem_group.py --tile-root <tile-root> --out <seg-data-root>/mito_cem
```

`--min-fg-train-px 1` drops empty train/val tiles for the sparse LD organelle; test splits
are always kept in full. The mito group is built by `build_mito_cem_group.py` from
`manifest_mito_regrid_cem.csv`, i.e. the 8 nm standard tiles plus the regridded whole-cell
and CEM-MitoLab train pools.

## 2. Launch the eight runs

Mitochondria and ER use the released-model configs in
`segmentation_training/configs/released_models/`; nucleus and LD use the configs in
`benchmarking/internal_models/configs/`. When pointing the released mito/ER configs at the
benchmark data groups, set `data.group` to `benchmark_mito_cem` / `benchmark_er` (as
released they default to the full training-corpus group). OmniEM rows load the latest
checkpoint of `<OmniEM ViT-L run dir>`; QuantEM rows load `<QuantEM ViT-B run dir>` at
`--step 674999`.

```
# mitochondria (group benchmark_mito_cem)
python -u -m segmentation_training.harness.run_seg --config segmentation_training/configs/released_models/mitochondria_omniem.yaml --data-root <seg-data-root>/mito_cem --run-dir "<OmniEM ViT-L run dir>" --output-dir <output-dir>/mito_omniem
python -u -m segmentation_training.harness.run_seg --config segmentation_training/configs/released_models/mitochondria_quantem.yaml --data-root <seg-data-root>/mito_cem --run-dir "<QuantEM ViT-B run dir>" --step 674999 --output-dir <output-dir>/mito_quantem

# ER (group benchmark_er, native resolution)
python -u -m segmentation_training.harness.run_seg --config segmentation_training/configs/released_models/er_omniem.yaml --data-root <seg-data-root>/er --run-dir "<OmniEM ViT-L run dir>" --output-dir <output-dir>/er_omniem
python -u -m segmentation_training.harness.run_seg --config segmentation_training/configs/released_models/er_quantem.yaml --data-root <seg-data-root>/er --run-dir "<QuantEM ViT-B run dir>" --step 674999 --output-dir <output-dir>/er_quantem

# lipid droplets (group benchmark_ld)
python -u -m segmentation_training.harness.run_seg --config benchmarking/internal_models/configs/ld_omniem.yaml --data-root <seg-data-root>/ld --run-dir "<OmniEM ViT-L run dir>" --output-dir <output-dir>/ld_omniem
python -u -m segmentation_training.harness.run_seg --config benchmarking/internal_models/configs/ld_quantem.yaml --data-root <seg-data-root>/ld --run-dir "<QuantEM ViT-B run dir>" --step 674999 --output-dir <output-dir>/ld_quantem

# nucleus (group benchmark_nucleus)
python -u -m segmentation_training.harness.run_seg --config benchmarking/internal_models/configs/nucleus_omniem.yaml --data-root <seg-data-root>/nucleus --run-dir "<OmniEM ViT-L run dir>" --output-dir <output-dir>/nucleus_omniem
python -u -m segmentation_training.harness.run_seg --config benchmarking/internal_models/configs/nucleus_quantem.yaml --data-root <seg-data-root>/nucleus --run-dir "<QuantEM ViT-B run dir>" --step 674999 --output-dir <output-dir>/nucleus_quantem
```

Each run writes the trained head and the test metrics (per-subgroup macro Dice with
bootstrap confidence intervals) under its `--output-dir`.

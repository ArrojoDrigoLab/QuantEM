# `training_and_analysis/`

Code used to produce the models and results for the QuantEM manuscript, aside from the quantEM application itself. Each subfolder contains its own README and, where it ships code, its own environment file and path configuration.

| Subfolder | What it covers |
|---|---|
| [`dataset_assembly/`](dataset_assembly/) | Assembling the EM image corpus: repository scraping, LLM triage prompts, the outreach literature screen, and the tiling rules that cut assets into pretraining tiles |
| [`segmentation_dataset/`](segmentation_dataset/) | Assembly of the annotated segmentation corpus: source ingest into crops and metadata, the split definitions the encoder comparison, segmentation training and benchmark consume, and the benchmark tiles |
| [`foundation_training/`](foundation_training/) | Self-supervised pretraining of the QuantEM encoder, and the comparison of candidate encoders |
| [`segmentation_training/`](segmentation_training/) | Organelle segmentation heads: decoders, necks, losses, backbone adaptation, input scaling, style conditioning, and test-time support |
| [`benchmarking/`](benchmarking/) | Evaluation against published segmentation models, and label-efficiency fine-tuning |
| [`immunoEM_analysis/`](immunoEM_analysis/) | MIMS-EM registration, gold detection, compartment assignment, and spatial-randomness nulls |

---

Each subfolder README covers its own inputs and instructions.

## Environments

Each subfolder that ships code carries its own `environment.yml` and is otherwise independent, with one exception:
segmentation training loads encoders through the checkpoint index that pretraining writes, so
[`foundation_training/`](foundation_training/) must be importable when running it.

```bash
conda env create -f <subfolder>/environment.yml
```

Both training environments take torch from PyTorch's own wheel index rather than from conda, because
the CUDA builds the reported runs used are not available through conda — pinning them there resolves
to a different CUDA build. Each `environment.yml` records which build its runs used.

## Tests

`dataset_assembly` and `segmentation_training` contain test suites; `foundation_training` contains the
pretraining tests.

```bash
python -m pytest dataset_assembly/catalog/tests dataset_assembly/literature_screen/tests
python -m pytest segmentation_training/tests     # run from this directory
python -m pytest foundation_training/tests
```


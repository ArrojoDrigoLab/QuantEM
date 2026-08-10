# External model weights

Obtain each model's published
weights from its original source below and pass them on the command line — weights are
not redistributed with this repository. Per-model conda environments live in
[`envs/`](envs/); each adapter's docstring names the environment it runs in.

## empanada (MitoNet / NucleoNet / DropNet) — `run_empanada.py`, `run_empanada_scaled.py`

| File | Source |
|---|---|
| `MitoNet_v1.pth` | Zenodo record [6861565](https://zenodo.org/records/6861565) |
| `NucleoNet_v2.pth` | Zenodo record [18142651](https://zenodo.org/records/18142651) |
| `DropNet_v1.pth` | Zenodo record [15298854](https://zenodo.org/records/15298854) |

Place all three files in one directory and pass it as `--weights`.

## micro-sam — `run_microsam.py`

No `--weights` argument. `micro_sam` fetches the model named by `--model` (default
`vit_l_em_organelles`, the BioImage.IO "humorous-crab" export: `vit_l.pt` +
`vit_l_decoder.pt`) into its own model cache on first use.

## incasem — `run_incasem.py`

| File | Argument |
|---|---|
| `model_checkpoint_1847_mito_CF.pt` | `--ckpt-mito` |
| `model_checkpoint_1841_er_CF.pt` | `--ckpt-er` |

Checkpoints are distributed from the ASEM project bucket (`s3://asem-project`); the
upstream repository <https://github.com/kirchhausenlab/incasem> documents the download.

## DeepContact — `run_deepcontact_mito.py`, `run_deepcontact_er.py`

Weights: figshare <https://doi.org/10.6084/m9.figshare.19845940.v1> — six files,
`{tem,sem,cell}_mito.h5` (Mask R-CNN) and `{tem,sem,cell}_er.pth` (smp). Place them in
one directory and pass it as `--weights` to either runner.

Code (mito runner only): clone the upstream repository
<https://github.com/LX-doctorAI1/DeepContact> and pass the clone as `--deepcontact-repo`
(evaluated at commit `c9c10da55c5592b959969b15b983ff80533606bc`). The ER runner needs no
clone; the architecture is rebuilt directly.

## OrgSegNet — `run_orgsegnet.py`

Checkpoint: `OrgSegNet_iter_Version1.pth`, published at Zenodo DOI
[10.5281/zenodo.8419877](https://doi.org/10.5281/zenodo.8419877) (linked from the
upstream README; the paper's associated datasets are on Science Data Bank). Pass the
checkpoint file as `--weights`.

Code: clone the upstream repository <https://github.com/yzy0102/OrgSegNet> and pass the
clone as `--orgsegnet-repo` (evaluated at commit
`245602f6f13f5021f62d8e336e7c22c1ca76c652`). The clone provides both the `mmseg` package
and the `configs/OrgSegNet/OrgSeg_PlantCell_768x512.py` inference config.

# QuantEM

**An optimized platform of vision transformer-based models for segmentation and analysis of
electron microscopy data.**

QuantEM is an open-source platform for segmentation and quantitative analysis of electron
microscopy (EM) data across imaging modalities, tissues, and species. It provides pretrained
models for **mitochondria, endoplasmic reticulum, nuclei, and lipid droplets**, together with
interactive proofreading, guided fine-tuning, and downstream quantitative analysis.

Browse the intracellular EM data directory [here](https://arrojodrigolab.github.io/QuantEM/dataset_directory/). 

To run QuantEM, either use the [`napari plugin`](napari-quantem/) or [`standalone app`](quantem_app/) for Windows and Mac. 

This repository is the code and data resource accompanying the QuantEM manuscript, available at
https://www.biorxiv.org/content/10.64898/2026.08.06.743293v1

---

## What is here

| Directory | Contents |
|---|---|
| [`training_and_analysis/`](training_and_analysis/) | All code behind the manuscript: dataset assembly, tiling, foundation-model pretraining, decoder training, benchmarking, and the immuno-EM analysis. |
| [`dataset_directory/`](dataset_directory/) | A browsable, filterable directory of the assembled EM image corpus, with a thumbnail for every image and volume. Served as a static site. |
| [`quantem_app/`](quantem_app/) | The QuantEM application: segmentation, manual annotation and interactive proofreading, guided fine-tuning, and built-in quantitative analyses, running offline on one machine. Installable as a Python package or as a Windows desktop installer. Under active construction. |
| [`quantem-core/`](quantem-core/) | The inference and lightweight-adaptation engine for the eight released models — the shared core the napari plugin builds on. |
| [`napari-quantem/`](napari-quantem/) | The QuantEM and OmniEM segmentation models packaged as a [napari](https://napari.org) plugin, with widgets for segmenting, fine-tuning, proofreading, measuring and batch runs. |
| [`pages/`](pages/) | The landing page for the project's GitHub Pages site, which also hosts the dataset directory. |

## Data availability

Source EM and segmentation data is available via public repositories, reference the Supplementary Tables with the manuscript for a full list of sources. 

Model weights are available at https://huggingface.co/ArrojoeDrigoLab/quantem

## Citation

If you use QuantEM — the software, the released model weights, or the dataset directory — please
cite:

> Acree C, Krystofiak E, Coate K, DelGiorno KE, Winn NCE, Novak SW, Zaganjor E, Magnuson MA,
> Arrojo e Drigo R. **QuantEM: An optimized platform of vision transformer-based models for
> segmentation and analysis of electron microscopy data.** *bioRxiv* 2026.08.06.743293 (2026).
> doi: [10.64898/2026.08.06.743293](https://doi.org/10.64898/2026.08.06.743293)

```bibtex
@article{acree2026quantem,
  title   = {QuantEM: An optimized platform of vision transformer-based models for
             segmentation and analysis of electron microscopy data},
  author  = {Acree, Christopher and Krystofiak, Evan and Coate, Kathryn and
             DelGiorno, Kathleen E. and Winn, Nathan C. E. and Novak, Sammy Weiser and
             Zaganjor, Elma and Magnuson, Mark A. and {Arrojo e Drigo}, Rafael},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.06.743293},
  url     = {https://www.biorxiv.org/content/10.64898/2026.08.06.743293v1}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).


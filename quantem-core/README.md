# quantem-core

Inference and lightweight adaptation for the QuantEM and OmniEM electron-microscopy organelle
segmentation models. This is the core engine under
both [`napari-quantem`](../napari-quantem). 

Contains eight released models — {mitochondria, endoplasmic reticulum, nucleus, lipid droplets} ×
{QuantEM ViT-B, OmniEM ViT-L} — with the inference procedure: resample to the model's training resolution, EM-corpus normalisation, 512 px sliding
windows at 25 % overlap with Hann blending, threshold, connected components.

## Development

```bash
pip install -e ".[test]"
pytest
```

Tests that need the original pretraining checkpoint skip unless `QUANTEM_REF_CKPT` points at it.

## Licence

BSD-3-Clause — see [`LICENSE`](LICENSE).

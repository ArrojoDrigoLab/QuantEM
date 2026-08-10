# Sample data — `islet_mito_5nm.png`

Mouse pancreatic islet, scanning electron microscopy. A 1024 × 1024 crop at **5 nm/px**.

The dark round bodies with pale haloes are insulin granules; the elongated structures with visible
cristae are mitochondria. That contrast is why this crop was chosen — telling the two apart is the
actual task, and the result is legible at a glance to anyone who knows islet ultrastructure.

## Credit

**© Arrojo e Drigo Lab, Vanderbilt University.**
Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

> Mouse pancreatic islet SEM, Arrojo e Drigo Lab, Vanderbilt University. CC BY 4.0.

This crop is distributed with the plugin. It is a single 1024x1024 tile, included so the plugin
can be tried on real data before any model is downloaded.


    Accession: (pending)
    DOI:       (pending)

## Provenance

| | |
|---|---|
| Source collection | `segmentations/lab_islet_mito` — Drigo Lab manual mitochondria annotation |
| Source montage | `C57_August_CD2_Islet1_5nm` (`653b3562-e697-40c1-9387-a1fc566be74a`) |
| Crop id | `C57_August_CD2_Islet1_5nm_00000` |
| Region in the montage | x 13997–18093, y 9897–13993 (the 4096 px annotated canvas) |
| This file | rows 1662–2686, cols 1384–2408 of that canvas |
| Pixel size | 5.0 nm/px |
| Bit depth | 8-bit greyscale, unmodified source pixels (verified byte-identical) |

Only the crop is redistributed here, not the montage. Nothing was contrast-adjusted, resampled or
retouched; the PNG is a lossless encoding of the source pixels.

Attribution is also embedded in the PNG's text chunks, so it survives the file being copied out of
the package.

## What entering the pixel size actually changes here

The mitochondria models are trained at 8 nm/px, so entering **5** exercises the resampling path
(a 0.63× downsample) rather than the "no pixel size given" fallback — the sample therefore
demonstrates the normal, recommended way to use the plugin.

Measured on this image with `quantem/mito`:

| pixel size | working size | objects | area fraction |
|---|---|--:|--:|
| 5 nm/px (correct) | 640 × 640 | 17 | 17.1 % |
| left blank / 8 nm/px | 1024 × 1024 | 9 | 15.5 % |

The **segmentation barely moves** — Dice between the two masks is **0.927**, and the area fraction
differs by under two points. What changes is the **object count**: at native scale adjacent
mitochondria merge into single components, halving the count. So for area-based measurements the
pixel size matters little on this image, and for anything that counts or measures objects
individually it matters a lot.

(Leaving the field blank and entering 8 give identical results, as they must: the resample factor is
1.0 and the code treats that as a no-op.)

## Citation

Acree C, *et al.* *QuantEM: An optimized platform of vision transformer-based models for segmentation and analysis of electron microscopy data.* bioRxiv 2026.
[https://www.biorxiv.org/content/10.64898/2026.08.06.743293v1](https://www.biorxiv.org/content/10.64898/2026.08.06.743293v1)

# `immunoEM_analysis/`

Correlative immuno-electron microscopy and multi-isotope imaging mass spectrometry (MIMS-EM)
analysis: registering MIMS fields to their EM canvas, detecting gold labelling events, assigning
them to intracellular compartments, and testing the result against a spatial-randomness null.

---

## Pipeline

Six steps, run in order. Each is a standalone script; `--help` lists its options.

| Step | Script | In → out |
|---|---|---|
| 1 | `01_mims_ingest.py` | Cameca `.im` → one drift-corrected, z-summed count image per isotope |
| 2 | `02_register_to_em.py` | accumulated images + landmarks + EM canvas → warped fields + bounding boxes |
| 3 | `03_build_mosaic.py` | warped fields → canvas-sized mosaic per isotope |
| 4 | `04_detect_gold.py` | ¹⁹⁷Au mosaic → labelling events with intensity-weighted centroids |
| 5 | `05_compartment_analysis.py` | events + masks → compartment assignment, enrichment, distance bands |
| 6 | `06_monte_carlo.py` | events + masks → spatial-randomness null, z-scores, internal control |

```bash
python 01_mims_ingest.py --isotopes 197Au 12C 13C "14N 12C" 32S
python 02_register_to_em.py
python 03_build_mosaic.py
python 04_detect_gold.py
python 05_compartment_analysis.py
python 06_monte_carlo.py
```

`_common.py` holds path configuration and image loading helpers. 

## Inputs

**Landmark correspondences** — one JSON per MIMS field, under `<landmarks_dir>/<canvas>/<field>.json`,
where `<field>` matches the `.im` stem. Selected externally in standard image tools (Fiji, napari, or
equivalent). Two optional pairs of index-aligned lists:

```json
{
  "em_shapes":   [[[x, y], ...], ...],   "mims_shapes": [[[x, y], ...], ...],
  "em_points":   [[row, col], ...],      "mims_points": [[row, col], ...]
}
```

Polygons are reduced to their centroids; points are stored row-major and swapped to (x, y) on load.
At least three pairs are required. Shapes and points may be mixed. 

**Tissue and organelle masks** — assumed inputs, laid out as
`<masks_dir>/<canvas>/{tissue,nucleus,mitochondria}.{npy,tif,png}`; anything above zero is
foreground. Organelle masks come from QuantEM segmentation; the tissue mask is a manual delineation
of the section.

## Outputs

| Path | Contents |
|---|---|
| `<accumulated_dir>/<field>/<isotope>.tif` | accumulated secondary-ion counts, one per isotope |
| `<warped_dir>/<canvas>/<field>/` | warped isotopes plus `placement.json` giving the canvas bounding box |
| `<mosaic_dir>/<canvas>/<isotope>.png` | canvas-sized registered mosaic |
| `<results_dir>/<canvas>_gold_particles.csv` | one row per labelling event |
| `<results_dir>/compartments_per_{image,group}.json` | areas, gold fractions, enrichment, distance bands |
| `<results_dir>/monte_carlo_per_{image,group}.json` | null distributions and z-scores |

Particle columns: `component_id`, `mark_x`/`mark_y` (intensity-weighted centroid, mosaic pixels),
`peak_x`/`peak_y`/`peak_value`, `geom_x`/`geom_y` (geometric centroid), `area_px`, `total_counts`.

## Definitions used in the outputs

- **Labelling event** — one 8-connected component of ¹⁹⁷Au pixels above the count threshold
  (`--floor`, default 4). Components are not split. Position is the intensity-weighted centroid.
- **Compartments**, all within the tissue mask — nuclear = nucleus ∩ tissue; cytoplasmic =
  tissue ∖ nucleus; mitochondrial = mitochondria ∩ tissue. Localizations off tissue are excluded.
- **Enrichment** — fraction of on-tissue gold in a compartment ÷ fraction of tissue area it
  occupies. 1.0 is what area alone predicts.
- **Distance bands** — distance from cytoplasmic gold to the nearest mitochondrial boundary,
  binned separately for gold inside a mitochondrion and outside one.
- **z-score** — observed value against the mean and s.d. of the spatial-randomness null.
- **Group values** — unweighted mean over animals; each animal contributes once regardless of
  particle count.
- **Internal control** — the group-level null enrichment must be ~1.0 by construction. 

## Notes

- Overlap precedence at mosaic assembly is by acquisition order.
- ¹⁹⁷Au mosaics are written 8-bit to match the released rasters; counts above 255 are clipped. The
  gold threshold sits well below that, so detection is unaffected — relevant only where `peak_value`
  is reused quantitatively.
- Mosaics are canvas-sized rasters, 16-bit for every isotope other than ¹⁹⁷Au, so assembling one can
  require a machine with substantial memory. 

---

## Setup

See environment file and the path configuration.

```bash
conda env create -f environment.yml
cp paths.example.yaml paths.yaml   # then edit paths.yaml for the local machine
```

**Roots required:** `mims_im_dir`, `landmarks_dir`, `em_canvas_dir`, `masks_dir` (inputs);
`accumulated_dir`, `warped_dir`, `mosaic_dir`, `results_dir` (outputs).

`paths.example.yaml` also declares per-sample canvas dimensions, pixel size, and experimental group.

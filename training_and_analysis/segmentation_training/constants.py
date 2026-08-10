"""Shared constants for the segmentation harness: label encoding, organelle synonyms,
canonical resolutions, split wiring, paths.

Kept dependency-free (stdlib only) so both data-prep and harness import it cheaply and it imports on
a CPU-only machine (no numpy-BLAS / sklearn / skimage at import time).

Three properties of this configuration:
  * The harness uses the group2 external-comparison split CSVs (the held-out-source design), not
    group1 (the encoder-comparison benchmark).
  * The harness resamples every source asset to a per-organelle canonical nm/px
    (``CANONICAL_NM``); the encoder comparison keeps native pixel resolution.
  * The on-disk split value ``train_pool`` maps to the derived-dir name ``train``
    (``SPLIT_VALUE_TO_DIR``); ``val`` is a token 8-row set — benchmark scoring is on ``test``.
"""

from __future__ import annotations

# --- canonical derived-mask encoding -------------------------------------------------------------
# A derived mask pixel is one of: background, organelle-foreground, or ignore (unlabelled / unknown /
# outside the densely-annotated region). 255 doubles as the loss/metric ignore_index.
BACKGROUND = 0
FOREGROUND = 1
IGNORE_INDEX = 255

# --- organelle target definitions ----------------------------------------------------------------
# Canonicalisation is by name (the per-crop organelles_present[name].value), never by a fixed
# integer, because several datasets renumber organelles per volume. ``include`` names map to
# foreground; ``exclude`` names are not foreground even though their name contains the stem.
# The targets defined here are mito and er.
ORGANELLES: dict[str, dict[str, object]] = {
    "mito": {
        "include": ("mitochondria", "mito", "mitos"),
        "include_prefix": ("mito",),  # robust to per-volume renaming
        "exclude": (),
    },
    "er": {
        "include": (
            "endoplasmic_reticulum",
            "er",
            "er_sheets",
            "er_tubules",
            "peripheral_er",
        ),
        "include_prefix": ("er",),  # catches per-volume ER variants not enumerated above
        "exclude": ("er_other_cells",),  # belongs to a neighbouring cell (empiar_13420)
    },
}

# Organelles the released experiments run on; the derived-dataset build defaults to this set and
# ignores anything outside it.
VALID_ORGANELLES = ("mito", "er")

# --- per-organelle canonical resample target (nm/px); the input-scale experiment sweeps this -----
# ER -> 2 nm; mitochondria -> 8 nm. The nucleus and ld entries are documented defaults for the
# shared-model configuration (nucleus 25 nm; lipid droplets 5-10 nm, 8 nm used here);
# ``VALID_ORGANELLES`` scopes which organelles the experiments run on.
CANONICAL_NM: dict[str, float] = {
    "er": 2.0,
    "mito": 8.0,
    "nucleus": 25.0,
    "ld": 8.0,
}

# --- split wiring (held-out-source splits) -------------------------------------------------------
SPLIT_CSV = {
    "mito": "group2_mito.csv",
    "er": "group2_er.csv",
    "nucleus": "group2_nucleus.csv",
    "ld": "group2_ld.csv",
}

# On-disk split value -> derived-dataset directory name. ``train_pool`` is the adaptation pool; the
# token ``val`` (8 rows) is for tuning, ``test`` is the held-out-source benchmark report set.
SPLIT_VALUE_TO_DIR = {
    "train_pool": "train",
    "train": "train",
    "val": "val",
    "test": "test",
}

# --- null / unknown-scale policy  ---------------------------------------------
# 160 corpus crops (all 2D-TEM) have voxel_x_nm=null & scale_band=unknown; they cannot be resampled
# to a canonical nm/px without an assumed scale. They are excluded from the canonical benchmark by
# default; the branch stays configurable so an assumed or recovered scale can be supplied.
#   "drop"          -> skip the crop entirely (default).
#   "native_bucket" -> emit unresampled into <root>/<group>/native_unscaled/ tagged scale_band=unknown.
#   "estimate"      -> impute nm/px from ASSUMED_SCALE_BAND_NM[scale_band] and resample (opt-in only).
NULL_SCALE_POLICY = "drop"
# Coarse per-band nm/px priors for the opt-in "estimate" policy (band midpoints; documented, unused
# by default). scale_band values on disk: 0.5-2, 2-6, 6-15, 15-40, 40+, unknown.
ASSUMED_SCALE_BAND_NM: dict[str, float] = {
    "0.5-2": 1.25,
    "2-6": 4.0,
    "6-15": 10.0,
    "15-40": 27.0,
    "40+": 60.0,
}

# --- corpus + output default paths (overridable via CLI / config) --------------------------------
# No default corpus root: --corpus-root is required.
DEFAULT_CORPUS_ROOT = None
# No default derived root: --data-root is required.
DEFAULT_DERIVED_ROOT = None

# Split CSV column order (authoritative; the same schema as the group1 splits).
SPLIT_COLUMNS = (
    "collection",
    "dataset",
    "crop_id",
    "image_path",
    "split",
    "subgroup",
    "modality",
    "scale_band",
    "tissue_context",
    "species_group",
)

# crops_metadata.csv column order (the master crop index; supplies voxel_x_nm / scale_band / etc).
CROPS_METADATA_COLUMNS = (
    "collection",
    "dataset",
    "crop_id",
    "image_path",
    "modality",
    "dimensionality",
    "voxel_x_nm",
    "scale_band",
    "tissue_context",
    "species_group",
    "in_situ_status",
    "external_annotation",
    "organelles",
    "coverage_tier",
    "official_split",
    "n_tiles",
)

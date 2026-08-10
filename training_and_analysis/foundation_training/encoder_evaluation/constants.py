"""Shared constants for the encoder-comparison probe: label encoding, organelle synonyms, EM stats, paths.

Kept dependency-free (stdlib only) so both data-prep and harness import it cheaply.
"""

from __future__ import annotations

# --- canonical derived-mask encoding -------------------------------------------------------------
# A derived mask pixel is one of: background, organelle-foreground, or ignore (unlabelled / unknown /
# outside the densely-annotated region). 255 doubles as the loss/metric ignore_index.
BACKGROUND = 0
FOREGROUND = 1
IGNORE_INDEX = 255

# --- EM corpus normalisation (single channel; never ImageNet) ------------------------------------
# Defaults; the real per-checkpoint values are read from checkpoint_index.json (manifest.image_mean
# / image_std). These are em_ssl's EM_DEFAULT_MEAN/STD rounded to three decimals.
EM_DEFAULT_MEAN = 0.583
EM_DEFAULT_STD = 0.244

# --- organelle target definitions ----------------------------------------------------------------
# Canonicalisation is by name (the per-crop organelles_present[name].value), never by a fixed integer,
# because several datasets renumber organelles per volume. ``include`` names map to foreground;
# ``exclude`` names stay background even though their name contains the stem.
ORGANELLES: dict[str, dict[str, object]] = {
    "mito": {
        "include": ("mitochondria", "mito", "mitos"),
        # robust to per-volume renaming: any "mito*" organelle name is mitochondria here.
        "include_prefix": ("mito",),
        "exclude": (),
    },
    "er": {
        # Endoplasmic reticulum and its sub-class label values (sheets, tubules, peripheral, proximal),
        # all of which are ER for a binary target. `include_prefix` catches per-volume ER variants that
        # are not enumerated here, so genuine ER is never dropped to background; names without an "er"
        # prefix stay in the explicit include set.
        "include": (
            "endoplasmic_reticulum",
            "er",
            "er_sheets",
            "er_tubules",
            "peripheral_er",
        ),
        "include_prefix": ("er",),
        # er_other_cells belongs to a neighbouring cell in empiar_13420 -> not part of the annotated
        # cell's ER.
        "exclude": ("er_other_cells",),
    },
}

VALID_ORGANELLES = tuple(ORGANELLES.keys())

# Map an organelle target -> the evaluation-corpus split CSV filename.
SPLIT_CSV = {
    "mito": "group1_mito.csv",
    "er": "group1_er.csv",
}

# --- corpus + output default paths (overridable via CLI / config) --------------------------------
# No default corpus root: --corpus-root is required.
DEFAULT_CORPUS_ROOT = None
# No default derived root: --derived-root is required.
DEFAULT_DERIVED_ROOT = None

# Column schema of the split CSVs named in SPLIT_CSV, in file order. ``dataprep.splits.load_split_rows``
# reads the same columns by name into ``CropRow``.
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

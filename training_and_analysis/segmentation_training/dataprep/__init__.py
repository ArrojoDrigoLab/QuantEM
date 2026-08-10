"""Segmentation data-prep: native source assets -> crops resampled to the organelle's canonical
nm/px (``constants.CANONICAL_NM``: ER 2 nm, mito 8 nm). Crop extent follows the annotation plus its
context margin, so crops vary in size unless ``--pad-even-to`` sets a floor."""

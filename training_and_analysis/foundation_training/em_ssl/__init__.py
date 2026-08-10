"""em_ssl — the EM self-supervised foundation-model training harness.

A thin, EM-appropriate layer over the official DINOv3 SSL framework, operating on true
single-channel (in_chans=1) electron-microscopy tiles. See ``foundation_training/README.md``.
"""

__version__ = "0.1.0"

# EM corpus intensity normalization: single channel, images scaled to [0, 1]. These are defaults,
# used only when a run config does not point at a computed tile_intensity_stats.json. The values
# for any run are recomputed by `em_ssl.tools.compute_tile_stats` over the filtered corpus and
# frozen into the run's resolved config. ImageNet statistics do not apply to EM.
EM_DEFAULT_MEAN = 0.583175
EM_DEFAULT_STD = 0.244468

__all__ = ["__version__", "EM_DEFAULT_MEAN", "EM_DEFAULT_STD"]

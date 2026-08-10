"""Image-style conditioning and per-image adaptation.

The building blocks that let the decoder condition on image appearance — the confirmed cause of the
held-out-source gap that encoder adaptation leaves open. Motivation: in manual EM segmentation one
first reads the image's overall and organelle-specific contrast/patterns and uses that as a guide;
image-style conditioning gives the model the same appearance-calibration signal.

Modules
-------
* ``grl``               — gradient-reversal layer + DANN lambda schedule (non-spurious-style option).
* ``style_encoder``     — the style encoder (raw tile + cheap low-level stats -> style code s), the
                          confident-region appearance code pooled from encoder features, and the
                          gradient-reversed source adversary.
* ``film``              — FiLM / conditional-GroupNorm re-injected at every neck+decoder norm.
* ``mixstyle``          — MixStyle + DSU training-time feature-statistic mixing.
* ``pooling``           — tile / source / dataset style-scope pooling + multi-prototype.
* ``positional_debias`` — positional-feature debiasing + matchability diagnostic + self-support
                          multi-prototype propagation (used by the feature-matching test-time arm).

Everything here is torch-only and needs no GPU: sklearn, skimage, matplotlib and numpy-BLAS matmul are
not used, so the package imports on a CPU-only machine. The orchestration seam that wires these onto a
``SegModel`` lives in ``segmentation_training/hooks/film_conditioning.py``.
"""

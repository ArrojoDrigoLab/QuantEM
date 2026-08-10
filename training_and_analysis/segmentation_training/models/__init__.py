"""Segmentation model registries: swappable necks, decoders and losses.

Literal module-level ``dict[str, builder]`` mappings plus ``build_*`` factories, rather than a
decorator framework. The optional heavy dependencies — HuggingFace ``transformers`` for the
Mask2Former query decoder, ``affogato`` for mutex-watershed clustering — are lazy-imported inside
the builders that need them, so every other arm builds without them installed.
"""

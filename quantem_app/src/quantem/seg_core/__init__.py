"""
seg_core: Abstract Segmenter Interface
=======================================

Shared infrastructure for organelle segmentation pipelines: the
:class:`~quantem.seg_core.base_segmenter.BaseSegmenter` contract, the segmenter
registry, generic instance-extraction helpers, and the DB layer that drives a
segmenter for an ``ImageSegmentation``.

The one concrete implementation lives in :mod:`quantem.inference`; seg_core
never imports it directly (the registry resolves it lazily by import path).

This is a plain Python package (not a Django app).
"""

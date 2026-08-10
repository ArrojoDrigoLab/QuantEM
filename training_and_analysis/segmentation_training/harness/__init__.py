"""Segmentation harness: frozen-encoder loading, neck+decoder training, sliding-window eval, metrics.

Intentionally light at package import: submodules that pull in torch (encoders, train, evaluate) are
imported explicitly by callers, so ``import segmentation_training.harness`` stays cheap to import.
"""

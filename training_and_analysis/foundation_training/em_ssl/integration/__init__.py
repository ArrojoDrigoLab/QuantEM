"""Thin EM adapter layer over upstream DINOv3 (applied at runtime, never a fork).

`dinov3_patch.apply_em_patches()` installs the minimal monkeypatches that make DINOv3 train
true single-channel EM models. Nothing here imports dinov3
at module import time, so the rest of em_ssl does not require dinov3.
"""

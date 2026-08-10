"""infer_dinov3_build_kwargs: rebuild the DINOv3 ViT block config from a teacher state dict.

Guards the LayerScale/storage-token/untie-norm inference that the encoder-evaluation decoder probe,
the segmentation harness and fino_diagnostics all rely on to load teacher checkpoints without silently
dropping ls1/ls2.gamma.
"""

from __future__ import annotations

import torch

from em_ssl.utils.checkpoint_index import infer_dinov3_build_kwargs

def test_infers_layerscale_storage_and_untie():
    sd = {
        "blocks.0.ls1.gamma": torch.zeros(768),
        "blocks.0.ls2.gamma": torch.zeros(768),
        "storage_tokens": torch.zeros(1, 4, 768),
        "cls_norm.weight": torch.zeros(768),
        "patch_embed.proj.weight": torch.zeros(768, 1, 16, 16),
    }
    kw = infer_dinov3_build_kwargs(sd, {"patch_size": 16, "in_chans": 1})
    assert kw["layerscale_init"] == 1.0e-05
    assert kw["n_storage_tokens"] == 4
    assert kw["untie_cls_and_patch_norms"] is True
    assert kw["patch_size"] == 16 and kw["in_chans"] == 1

def test_plain_backbone_leaves_base_kwargs_untouched():
    sd = {"blocks.0.mlp.fc1.weight": torch.zeros(3072, 768), "norm.weight": torch.zeros(768)}
    assert infer_dinov3_build_kwargs(sd, {"patch_size": 16, "in_chans": 1}) == {"patch_size": 16, "in_chans": 1}

def test_does_not_mutate_base_kwargs():
    base = {"patch_size": 16, "in_chans": 1}
    infer_dinov3_build_kwargs({"blocks.0.ls1.gamma": torch.zeros(1)}, base)
    assert "layerscale_init" not in base  # helper returns a copy

"""ExperimentSpec loading + DINOv3 config translation (no dinov3 import needed)."""

from __future__ import annotations

from pathlib import Path

import yaml

from em_ssl.config import load_experiment
from em_ssl.integration.config_translation import encoder_manifest_fields, translate_stage

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "pretraining" / "quantem_vitb_512.yaml"

def test_load_example_config():
    spec = load_experiment(EXAMPLE)
    assert spec.framework == "dinov3"
    assert spec.model.arch == "vit_base" and spec.model.in_chans == 1
    assert spec.max_global_crop == 512 and spec.effective_min_side == 512
    m, s = spec.resolved_mean_std()
    assert 0.0 < m < 1.0 and 0.0 < s < 1.0  # EM stats, never ImageNet

def test_translate_stage_is_single_channel():
    spec = load_experiment(EXAMPLE)
    cfg = translate_stage(spec, 0, output_dir="runs/_x", seed=0)
    assert cfg["student"]["in_chans"] == 1
    assert cfg["teacher"]["in_chans"] == 1
    assert len(cfg["crops"]["rgb_mean"]) == 1 and len(cfg["crops"]["rgb_std"]) == 1  # never RGB
    assert cfg["crops"]["global_crops_size"] == 512
    assert cfg["train"]["dataset_path"].startswith("EMShards:")
    assert cfg["compute_precision"]["sharding_strategy"] == "SHARD_GRAD_OP"
    # step budget ~ max_steps
    total = cfg["optim"]["epochs"] * cfg["train"]["OFFICIAL_EPOCH_LENGTH"]
    assert abs(total - spec.crops.schedule[0].max_steps) <= cfg["train"]["OFFICIAL_EPOCH_LENGTH"]
    assert cfg["gram"]["use_loss"] is False
    assert "em" in cfg  # EM augmentation block present
    # Only rescale is official (default on); shift/jitter default off (no DINOv3-config precedent) -> absent.
    assert cfg["student"]["pos_embed_rope_rescale_coords"] == 2.0
    assert "pos_embed_rope_shift_coords" not in cfg["student"]
    assert "pos_embed_rope_jitter_coords" not in cfg["student"]
    # drop_path and register tokens are emitted explicitly rather than inherited; the released values
    # are asserted in test_released_recipe below.
    assert "drop_path_rate" in cfg["student"]
    assert "n_storage_tokens" in cfg["student"]

def test_released_recipe():
    """The released encoder's configuration carries the settings it was trained with."""
    spec = load_experiment(EXAMPLE)
    cfg = translate_stage(spec, 0, output_dir="runs/_x", seed=0)
    assert cfg["student"]["n_storage_tokens"] == 4            # register tokens
    assert cfg["student"]["drop_path_rate"] == 0.1            # ViT-B stochastic depth
    assert cfg["student"]["pos_embed_rope_rescale_coords"] == 2.0   # the RoPE coordinate augmentation used upstream
    assert "pos_embed_rope_shift_coords" not in cfg["student"]      # unofficial aug off
    assert "pos_embed_rope_jitter_coords" not in cfg["student"]
    # DINOv3 indefinite-training schedules: constant lr/wd/momentum (peak == end => no decay), warmups only.
    sch = cfg["schedules"]
    assert sch["lr"]["peak"] == sch["lr"]["end"] == 5.0e-4          # constant LR
    assert sch["weight_decay"]["peak"] == sch["weight_decay"]["end"] == 0.04    # constant WD
    assert sch["momentum"]["peak"] == sch["momentum"]["end"] == 0.994           # constant momentum
    assert sch["teacher_temp"]["start"] == 0.04 and sch["teacher_temp"]["end"] == 0.07  # warmup then constant
    assert spec.checkpointing.teacher_export_every_steps == 25000  # probe-able teacher every 25k
    assert spec.checkpointing.keep_every_steps == 25000

def test_encoder_manifest_fields():
    spec = load_experiment(EXAMPLE)
    f = encoder_manifest_fields(spec, 0)
    assert f["arch"] == "vit_base" and f["embedding_dim"] == 768 and f["depth"] == 12
    assert f["input_channels"] == 1 and f["objective"] == "dino+ibot+koleo"
    assert f["intermediate_layers"] == [8, 9, 10, 11]

def test_rope_coord_aug_control(tmp_path):
    spec = load_experiment(EXAMPLE)
    # None disables a single aug -> its key is not emitted (inherits DINOv3's default); others stay on.
    spec.model.rope_shift_coords = None
    cfg = translate_stage(spec, 0, output_dir=str(tmp_path), seed=0)
    assert "pos_embed_rope_shift_coords" not in cfg["student"]
    assert cfg["student"]["pos_embed_rope_rescale_coords"] == 2.0
    # The dinov3: escape hatch still wins (deep-merged last).
    spec.dinov3 = {"student": {"pos_embed_rope_rescale_coords": 3.0}}
    cfg2 = translate_stage(spec, 0, output_dir=str(tmp_path), seed=0)
    assert cfg2["student"]["pos_embed_rope_rescale_coords"] == 3.0
    # Provenance stays clean: rope augs never leak into the encoder manifest / FINO fields.
    f = encoder_manifest_fields(spec, 0)
    assert not any("rope" in k.lower() for k in f)

def test_gram_translation(tmp_path):
    spec = load_experiment(EXAMPLE)
    spec.crops.gram_teacher_crops_size = 512
    cfg = translate_stage(spec, 0, output_dir=str(tmp_path), seed=0)
    assert cfg["gram"]["use_loss"] is True
    assert cfg["crops"]["gram_teacher_crops_size"] == 512

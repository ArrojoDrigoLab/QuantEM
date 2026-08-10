"""Checkpoint index records FINO provenance and round-trips it."""

from __future__ import annotations

from em_ssl.config.schema import ExperimentSpec, _coerce
from em_ssl.integration import config_translation as ct
from em_ssl.utils.checkpoint_index import CheckpointIndex, EncoderManifest

def _fino_spec():
    return _coerce(ExperimentSpec, {
        "name": "meta_scale_plus", "framework": "dinov3",
        "crops": {"schedule": [{"global_crops_size": 512, "max_steps": 100}]},
        "data": {"shard_dir": "/d"},
        "metadata_factors": [
            {"name": "modality", "field": "modality", "type": "discrete", "guidance": "negative", "loss_weight": 0.1}
        ],
    })

def test_encoder_manifest_round_trip_with_fino(tmp_path):
    spec = _fino_spec()
    fields = ct.encoder_manifest_fields(spec, 0)
    man = EncoderManifest(
        run_id="meta_modality_minus_stage0", framework="dinov3", objective=fields["objective"], arch=fields["arch"],
        patch_size=fields["patch_size"], embedding_dim=fields["embedding_dim"], depth=fields["depth"],
        input_channels=fields["input_channels"],
        fino_enabled=fields["fino_enabled"], fino_factors=fields["fino_factors"],
        fino_lambda_schedule=fields["fino_lambda_schedule"], fino_factors_fingerprint=fields["fino_factors_fingerprint"],
        manifest_fingerprint="mf", shard_fingerprint="sf", metadata_coverage_fingerprint="cf",
    )
    idx = CheckpointIndex(tmp_path, man)
    idx.add(step=100, kind="teacher", path=str(tmp_path / "eval/100/teacher_checkpoint.pth"), crop_size=512)

    loaded = CheckpointIndex.load(tmp_path)
    m = loaded.manifest
    assert m.fino_enabled is True
    assert m.fino_factors[0]["name"] == "modality" and m.fino_factors[0]["guidance"] == "negative"
    assert m.fino_factors[0]["grl"] is True and m.fino_factors[0]["n_outputs"] == spec.metadata_factors[0].n_outputs
    assert m.fino_factors_fingerprint and m.manifest_fingerprint == "mf"
    assert m.metadata_coverage_fingerprint == "cf"
    assert loaded.latest("teacher").step == 100

def test_baseline_manifest_defaults_are_fino_off(tmp_path):
    man = EncoderManifest(run_id="base_512", framework="dinov3", objective="dino+ibot+koleo", arch="vit_base",
                          patch_size=16, embedding_dim=768, depth=12)
    idx = CheckpointIndex(tmp_path, man)
    idx.save()
    m = CheckpointIndex.load(tmp_path).manifest
    assert m.fino_enabled is False and m.fino_factors == [] and m.fino_factors_fingerprint is None

"""A DINOv3-indexed QuantEM pack builds through timm, and is the same model.

The four QuantEM packs ran on the processor even on a machine with a working
CUDA card. The mechanism was not the encoder: a complete, GPU-capable timm path
for this ViT-B already existed and was reachable only from an index written by
the Hugging Face installer. A pack installed from a release bundle or from a
local research directory carries the *research* index instead --
``framework: dinov3`` -- and that one word sent it down a tier needing a package
this project does not ship, so the accelerator ladder fell through to the CPU.

MEASURED on a Quadro RTX 8000 (Turing sm_75, 48 GB, torch 2.13.0+cu126) with the
released ``quantem:mito``, 512 px windows, fp32:

===============================================  ==============  ==========
                                                 per window      cold load
===============================================  ==============  ==========
before, CPU (the shipped TorchScript artifact)   0.666 s         3.5 s
after, CUDA (eager timm, first run)              0.143 s         25.0 s
after, CUDA (the artifact the first run wrote)   0.139 s         12.7 s
===============================================  ==============  ==========

4.8x per window. The numerical half of that claim is the first test below and
the parity test at the end.

The unit tests here use tiny synthetic state dicts: the naming is the whole
subject, and a rename is exactly the class of bug that a 525 MB fixture would
hide behind an out-of-memory error. The two ``requires_weights`` tests at the
end are the ones that run the real ViT-B.
"""

from __future__ import annotations

import pytest
import torch

from quantem.inference import encoders
from quantem.inference.dinov3_hint import DINOV3_PATH_ENV_VAR, hint_provides_dinov3
from quantem.registry import cache as registry_cache

# --- The rename --------------------------------------------------------------


class TestRemap:
    def test_the_fused_qkv_bias_splits_and_keeps_its_k_third(self):
        # T10: timm registers k_bias as a non-persistent zeroed buffer because
        # Meta's distilled checkpoints trained with mask_k_bias=True. Ours did
        # not. Dropping the k third would zero 9 216 trained values and raise
        # nothing at all.
        bias = torch.arange(12, dtype=torch.float32)
        out = encoders.remap_dinov3_state_dict({"blocks.0.attn.qkv.bias": bias})
        assert set(out) == {
            "blocks.0.attn.q_bias",
            "blocks.0.attn.k_bias",
            "blocks.0.attn.v_bias",
        }
        assert torch.equal(out["blocks.0.attn.k_bias"], bias[4:8])

    def test_layerscale_and_storage_tokens_take_their_timm_names(self):
        out = encoders.remap_dinov3_state_dict(
            {
                "blocks.3.ls1.gamma": torch.ones(2),
                "blocks.3.ls2.gamma": torch.ones(2),
                "storage_tokens": torch.ones(1, 4, 8),
            }
        )
        assert set(out) == {"blocks.3.gamma_1", "blocks.3.gamma_2", "reg_token"}

    def test_the_rope_periods_are_set_aside_not_loaded(self):
        # timm registers rope.periods non-persistently, so it cannot travel in
        # a state dict; the builder installs it by hand, in bfloat16 (T11).
        out = encoders.remap_dinov3_state_dict({"rope_embed.periods": torch.ones(4)})
        assert list(out) == ["rope.periods"]

    def test_the_pretraining_mask_token_is_dropped(self):
        assert encoders.remap_dinov3_state_dict({"mask_token": torch.ones(3)}) == {}

    def test_an_ordinary_key_is_left_alone(self):
        w = torch.ones(2, 2)
        assert encoders.remap_dinov3_state_dict({"patch_embed.proj.weight": w}) == {
            "patch_embed.proj.weight": w
        }


class TestOverlayRemap:
    """The head's ``encoder_trainable`` block makes the same journey."""

    def test_backbone_tensors_are_remapped_under_their_prefix(self):
        out = encoders.remap_dinov3_overlay({"backbone.blocks.8.ls1.gamma": torch.ones(2)})
        assert list(out) == ["backbone.blocks.8.gamma_1"]

    def test_lora_modules_pass_through_untouched(self):
        w = torch.ones(2, 2)
        out = encoders.remap_dinov3_overlay({"_conv_lora.0.down.weight": w})
        assert out == {"_conv_lora.0.down.weight": w}

    def test_a_rope_buffer_in_a_head_is_not_placed_as_a_parameter(self):
        out = encoders.remap_dinov3_overlay({"backbone.rope_embed.periods": torch.ones(4)})
        assert out == {}


# --- The synthesised entry point ---------------------------------------------


def _manifest(**over) -> encoders.EncoderManifest:
    base = {
        "arch": "vit_base",
        "depth": 12,
        "embedding_dim": 768,
        "patch_size": 16,
        "framework": "dinov3",
        "image_mean": 0.583175,
        "image_std": 0.244468,
        "input_channels": 1,
        "entry_point": {"loader": "dinov3_teacher", "checkpoint_key": "teacher"},
        "run_id": "M1_dinov3_vitb_512",
    }
    base.update(over)
    return encoders.EncoderManifest(**base)


class TestSynthesisedEntryPoint:
    def test_it_matches_what_the_hugging_face_installer_writes(self):
        # The point of the bridge is that a bundle-installed pack and an
        # HF-installed pack build the *same* module. Every value the installer
        # fixes has to be fixed the same way here, so this reads them from that
        # module rather than restating them.
        from quantem.registry import hf_install

        fe = encoders._timm_entry_point_for_dinov3(_manifest(), {})
        assert fe["variant"] == hf_install.QUANTEM_TIMM_VARIANT
        assert fe["timm_model"] == "vit_base_patch16_dinov3_qkvb"
        assert fe["norm_eps"] == 1e-06
        assert fe["rope_periods_bf16"] is True
        assert fe["in_chans"] == 1
        assert fe["img_size_build"] == 512

    def test_the_timm_model_it_names_exists_in_the_installed_timm(self):
        import timm

        fe = encoders._timm_entry_point_for_dinov3(_manifest(), {})
        assert fe["timm_model"] in timm.list_models("*dinov3*")

    def test_the_prefix_token_count_is_read_from_the_checkpoint(self):
        # One class token plus however many storage tokens this encoder was
        # trained with. Getting it wrong splits features silently.
        fe = encoders._timm_entry_point_for_dinov3(
            _manifest(), {"storage_tokens": torch.zeros(1, 4, 768)}
        )
        assert fe["n_prefix_tokens"] == 5

    def test_the_trunk_leaves_nothing_for_the_head_to_provide(self):
        # Unlike the HF trunk, the research checkpoint carries all twelve
        # blocks; the head overwrites the fine-tuned ones rather than filling
        # a hole.
        assert encoders._timm_entry_point_for_dinov3(_manifest(), {})["overlay_blocks"] == []

    def test_an_arch_with_no_timm_equivalent_is_refused_not_guessed(self):
        with pytest.raises(encoders.EncoderUnavailable, match="no timm equivalent"):
            encoders._timm_entry_point_for_dinov3(_manifest(arch="vit_gigantic"), {})


# --- The environment hint, which two modules have to agree about -------------


class TestDinov3Hint:
    def test_a_checkout_root_is_recognised(self, tmp_path, monkeypatch):
        (tmp_path / "dinov3").mkdir()
        (tmp_path / "dinov3" / "__init__.py").write_text("", encoding="utf-8")
        monkeypatch.setenv(DINOV3_PATH_ENV_VAR, str(tmp_path))
        assert hint_provides_dinov3() is True

    def test_a_directory_holding_a_dinov3_module_is_recognised_too(self, tmp_path, monkeypatch):
        # This is the case the two callers disagreed about: the importer put the
        # hint on sys.path and let `import dinov3` decide, while the Models
        # screen's probe insisted on a dinov3/ *directory* and called the same
        # machine unrunnable.
        (tmp_path / "dinov3.py").write_text("", encoding="utf-8")
        monkeypatch.setenv(DINOV3_PATH_ENV_VAR, str(tmp_path))
        assert hint_provides_dinov3() is True

    def test_a_directory_with_no_dinov3_in_it_is_not(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DINOV3_PATH_ENV_VAR, str(tmp_path))
        assert hint_provides_dinov3() is False

    def test_an_unset_or_missing_hint_is_not(self, tmp_path, monkeypatch):
        monkeypatch.delenv(DINOV3_PATH_ENV_VAR, raising=False)
        assert hint_provides_dinov3() is False
        monkeypatch.setenv(DINOV3_PATH_ENV_VAR, str(tmp_path / "nowhere"))
        assert hint_provides_dinov3() is False

    def test_the_probe_and_the_importer_read_the_same_variable(self):
        from quantem.registry import catalogue

        assert catalogue.DINOV3_PATH_ENV_VAR == encoders.DINOV3_PATH_ENV_VAR


# --- The real weights ---------------------------------------------------------

QUANTEM_PACKS = ("quantem:mito", "quantem:er", "quantem:ld", "quantem:nucleus")


def _require(pack_id: str) -> None:
    if not registry_cache.installed(pack_id):
        pytest.skip(
            f"{pack_id} is not installed here; run with QUANTEM_DATA_DIR pointing "
            "at a data directory that has it"
        )


@pytest.mark.requires_weights
@pytest.mark.slow
@pytest.mark.parametrize("pack_id", QUANTEM_PACKS)
def test_the_timm_build_is_the_shipped_model_to_the_bit(pack_id: str) -> None:
    """The renamed build and the shipped artifact agree exactly, on the CPU.

    The gate on the whole change. Preferring timm is only safe if it is the
    same model, and "same" here is not a tolerance: MEASURED max abs difference
    0.0 in output probability for all four packs, against both the released
    TorchScript artifact and (where a checkout is pointed at) Meta's own
    package. Device differences are a separate question, measured in the test
    below; this one holds the device fixed so a regression here can only be the
    builder.
    """
    import numpy as np

    from quantem.inference._fig3.load_head import build_and_load_head
    from quantem.inference._fig3.schema import load_head_config
    from quantem.inference.engine import normalize_tile, resolve_model_files
    from quantem.inference.specs import MODEL_SPECS

    _require(pack_id)
    spec = MODEL_SPECS[pack_id]
    files = resolve_model_files(pack_id)
    if files.export_path is None:
        pytest.skip(f"{pack_id} has no exported artifact to compare against")
    cfg = load_head_config(files.config_path)
    manifest = encoders.EncoderManifest.from_index(files.index_path)
    if manifest.framework != "dinov3":
        pytest.skip(f"{pack_id} here is not a DINOv3-indexed install")

    skeleton = None
    if spec.embeds_encoder and files.encoder_path is None:
        skeleton = torch.load(str(files.head_path), map_location="cpu", weights_only=False).get(
            "encoder_trainable"
        )

    rng = np.random.default_rng(11)
    tile = rng.integers(0, 256, (spec.tile_size, spec.tile_size)).astype(np.uint8)
    x = normalize_tile(tile, spec.input_mean, spec.input_std)
    xt = torch.from_numpy(np.ascontiguousarray(x))[None, None]

    def probability(encoder) -> torch.Tensor:
        model, _ = build_and_load_head(cfg, encoder, files.head_path, device="cpu")
        with torch.no_grad():
            return torch.softmax(model(xt)[0].float(), dim=0)[1]

    shipped = probability(
        encoders.build_encoder(
            manifest=manifest,
            encoder_path=files.encoder_path,
            export_path=files.export_path,
            apply_encoder_norm=cfg.encoder.apply_encoder_norm,
            device="cpu",
        ).module
    )
    renamed = probability(
        encoders.build_quantem_timm_encoder_from_dinov3(
            files.encoder_path,
            manifest,
            cfg.encoder.apply_encoder_norm,
            skeleton_state=skeleton,
        )
    )
    assert float((shipped - renamed).abs().max()) == 0.0


@pytest.mark.requires_weights
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_a_quantem_pack_reaches_the_accelerator() -> None:
    """The pack that used to fall to the processor now loads on the card.

    And produces the same masks: MEASURED on a Quadro RTX 8000, max abs
    difference 2.3e-03 in probability over three 512 px windows and **zero**
    pixels changing side of the 0.5 threshold. That is the ordinary fp32
    device difference -- the same order as the 9.5e-03 measured over 60.7 MP
    for the OmniEM family -- not a different model; the test above pins the
    builder at 0.0.
    """
    import numpy as np

    from quantem.inference import engine
    from quantem.inference.device import cuda_available
    from quantem.inference.specs import MODEL_SPECS

    pack_id = "quantem:mito"
    _require(pack_id)
    if not cuda_available():
        pytest.skip("no CUDA device here")

    spec = MODEL_SPECS[pack_id]
    rng = np.random.default_rng(5)
    tile = rng.integers(0, 256, (spec.tile_size, spec.tile_size)).astype(np.uint8)

    try:
        on_cpu = engine.load_model(pack_id, device="cpu").forward_tile(tile)
        model = engine.load_model(pack_id, device="cuda")
        assert model.device == "cuda", model.load_notices
        on_gpu = model.forward_tile(tile)
    finally:
        engine.clear_model_cache()

    difference = float(np.abs(on_cpu.astype(np.float64) - on_gpu.astype(np.float64)).max())
    assert difference < 0.02, difference
    assert int(((on_cpu >= 0.5) != (on_gpu >= 0.5)).sum()) == 0

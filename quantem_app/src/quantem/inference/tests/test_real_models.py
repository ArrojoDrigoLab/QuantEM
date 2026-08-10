"""End-to-end tests against the real released weights.

Everything else in this package is tested with synthetic tensors and injected
forwards, which proves the tiling, blending and resampling arithmetic but says
nothing about whether a released ``head.pt`` actually loads into the vendored
architecture and produces a segmentation. That is what this file is for, and it
is the only test here that needs gigabytes on disk.

Marked ``requires_weights``. The suite writes to a repo-local data directory
(see :mod:`quantem._pytest_env`), which has no packs, so these skip by default.
To run them, point the suite at a data directory that *does*::

    QUANTEM_DATA_DIR=<a directory you installed packs into> pytest -m requires_weights

An explicit ``QUANTEM_DATA_DIR`` beats the plugin, so that is all it takes.

The assertions are deliberately strict about *wellformedness* and weak about
*content*. A synthetic EM image is not a micrograph, so how much foreground a
model should find in it is not knowable, and pinning that number would test this
fixture rather than the model. What is knowable is what actually breaks when a
checkpoint loads into a wrong graph: the map must have the shape the resample
plan says, lie in ``[0, 1]``, and vary -- a head loaded into a mismatched
architecture reliably saturates to a constant. The stronger claim, that the
output depends on the image at all, gets its own test.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from quantem.core import config
from quantem.inference import engine, resample, tiling
from quantem.inference.specs import MODEL_SPECS
from quantem.registry import cache as registry_cache
from quantem.testing import make_em_like_array

pytestmark = pytest.mark.requires_weights

#: Big enough to need four windows at both tile sizes, small enough to run the
#: whole eight-pack sweep on a CPU in a couple of minutes.
IMAGE_SIZE = 640

#: The fixture's nominal scale. Feeding a real number here matters: the mito and
#: LD packs declare 8 nm/px and the nucleus packs 25 nm/px, so this exercises the
#: resampling path instead of silently running everything at native scale.
PIXEL_SIZE_NM = 8.0

ALL_PACKS = sorted(MODEL_SPECS)

NECK_CLASSES = {"naive_1x1": "Naive1x1Neck", "resnet34_detail": "ResNet34DetailNeck"}
DECODER_CLASSES = {"affinity_mws": "AffinityMWS", "upernet": "UPerNet", "dpt": "DPT"}


def _require(pack_id: str) -> None:
    if not registry_cache.installed(pack_id):
        pytest.skip(
            f"{pack_id} is not installed in {config.STORAGE_DIR}. Install a "
            "release bundle, then re-run with "
            "QUANTEM_DATA_DIR=<that data directory> pytest -m requires_weights"
        )


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return make_em_like_array(IMAGE_SIZE, IMAGE_SIZE, seed=7)


@pytest.fixture(autouse=True)
def _drop_models() -> Iterator[None]:
    """Two packs stay cached and a ViT-L is 1.2 GB; do not accumulate across tests."""
    yield
    engine.clear_model_cache()


@pytest.mark.parametrize("pack_id", ALL_PACKS)
def test_pack_loads_and_segments(pack_id: str, image: np.ndarray) -> None:
    """Every released pack loads its real weights and segments a real array."""
    _require(pack_id)
    spec = MODEL_SPECS[pack_id]

    model = engine.load_model(pack_id, device="cpu")
    assert model.pack_id == pack_id
    assert model.encoder_tier in {"exported", "timm", "dinov3"}
    # load_model caches; a second call must reuse, not rebuild.
    assert engine.load_model(pack_id, device="cpu") is model

    windows: list[tuple[int, ...]] = []

    def spy(tile: np.ndarray) -> np.ndarray:
        # The engine must hand the model a uint8 square at the pack's own tile
        # size; padding, not a short window, is how edges are handled.
        assert tile.shape == (spec.tile_size, spec.tile_size)
        assert tile.dtype == np.uint8
        windows.append(tile.shape)
        return model.forward_tile(tile)

    prediction = engine.predict_region(
        model, image, pixel_size_nm=PIXEL_SIZE_NM, forward=spy
    )
    prob = prediction.prob

    # --- the published sliding-window geometry ---
    plan = prediction.plan
    assert plan.tile == spec.tile_size
    assert plan.overlap == tiling.DEFAULT_OVERLAP
    assert plan.stride == tiling.stride_for(spec.tile_size, tiling.DEFAULT_OVERLAP)
    assert len(windows) == plan.n_tiles
    assert engine.estimate_tiles(spec, image.shape[:2], pixel_size_nm=PIXEL_SIZE_NM) == plan.n_tiles

    # --- the probability map ---
    # Shape is the resample plan's, not the image's: a pack with a canonical
    # scale predicts on a resampled grid and is thresholded there.
    expected = resample.plan_resample(
        image.shape[:2], PIXEL_SIZE_NM, spec.canonical_nm
    ).model_shape
    assert prob.shape == expected
    assert prob.dtype == np.float32
    assert np.isfinite(prob).all(), f"{pack_id} produced non-finite probabilities"
    assert prob.min() >= 0.0
    assert prob.max() <= 1.0

    # A head loaded into the wrong graph does not raise, it saturates. Requiring
    # genuine variation is what separates "ran" from "ran correctly".
    assert prob.min() < prob.max(), f"{pack_id} produced a constant probability map"

    # The map must come back to native pixels intact, or nothing downstream --
    # post-processing, per-object confidence, export -- lines up with the image.
    native = resample.probability_to_native(prob, prediction.context)
    assert native.shape == image.shape[:2]
    if spec.canonical_nm is None:
        assert prediction.context.is_identity


@pytest.mark.parametrize("pack_id", ALL_PACKS)
def test_rebuilt_architecture_matches_the_manifest(pack_id: str) -> None:
    """The graph the head loaded into is the one the manifest declares.

    Every assertion here covers something that produces a plausible-looking
    segmentation when wrong rather than an exception.
    """
    _require(pack_id)
    spec = MODEL_SPECS[pack_id]
    model = engine.load_model(pack_id, device="cpu")

    assert type(model.module.neck).__name__ == NECK_CLASSES[spec.neck]
    assert type(model.module.decoder).__name__ == DECODER_CLASSES[spec.decoder]

    # `last4`: the four deepest blocks, ascending. The neck concatenates them on
    # the channel axis into a trained 1x1 conv, so the order is part of the model.
    encoder = model.module.encoder
    assert model.module.layers == [
        encoder.depth - 4, encoder.depth - 3, encoder.depth - 2, encoder.depth - 1
    ]

    # The OmniEM family takes a raw [0, 1] tile and normalises inside the
    # encoder; QuantEM takes an already-standardised one. Swapping them yields a
    # segmentation that is not the published model.
    assert encoder.input_mean == pytest.approx(spec.input_mean)
    assert encoder.input_std == pytest.approx(spec.input_std)
    assert encoder.patch_size == spec.patch_size
    assert spec.tile_size % encoder.patch_size == 0


@pytest.mark.parametrize("pack_id", ["quantem:mito", "omniem:mito"])
def test_output_depends_on_the_image(pack_id: str, image: np.ndarray) -> None:
    """The model responds to image content, not merely to being called.

    "Not constant" alone would pass for a model whose encoder weights failed to
    load and whose output is driven by padding and blend artefacts. Running the
    same pack over a flat field and over textured EM must differ materially.
    """
    _require(pack_id)
    model = engine.load_model(pack_id, device="cpu")
    flat = np.full_like(image, 128)

    on_texture = engine.predict_region(model, image, pixel_size_nm=PIXEL_SIZE_NM).prob
    on_flat = engine.predict_region(model, flat, pixel_size_nm=PIXEL_SIZE_NM).prob

    assert on_texture.shape == on_flat.shape
    # A generous floor: this asserts "the encoder sees the image", not a
    # particular sensitivity, which would be a claim about the model itself.
    assert np.abs(on_texture - on_flat).max() > 1e-3, (
        f"{pack_id} gave near-identical output on textured and flat input; "
        "its encoder weights are probably not loaded"
    )


def test_affinity_packs_expose_their_instance_aux(image: np.ndarray) -> None:
    """``affinity_mws`` packs still emit the affinities a watershed would need.

    Post-processing is connected components today, so these go unused -- but
    they are the documented route to splitting two touching organelles, and a
    refactor that dropped them would otherwise pass every other test here.
    """
    pack_id = "quantem:mito"
    _require(pack_id)
    spec = MODEL_SPECS[pack_id]
    model = engine.load_model(pack_id, device="cpu")
    model.forward_tile(image[: spec.tile_size, : spec.tile_size])

    aux = model.module.aux_logits
    assert len(aux) == 1
    affinities = aux[0]
    assert affinities.shape[1] == len(model.module.decoder.offsets) == 10
    assert tuple(affinities.shape[-2:]) == (spec.tile_size, spec.tile_size)
    assert float(affinities.min()) >= 0.0
    assert float(affinities.max()) <= 1.0


def test_exported_encoder_reproduces_the_eager_one() -> None:
    """Tier (a) and tier (b) must be the same model, not merely both plausible.

    ``quantem.inference.export`` verifies this once at export time. Checking it
    again here keeps it true across changes to the vendored stack: the whole
    argument for shipping an artifact instead of rebuilding the graph rests on
    the two being interchangeable, and nothing else would notice if they drifted.

    The OmniEM family is used because its eager tier is ``timm``, a real
    dependency -- so unlike the QuantEM family this test can run anywhere.
    """
    pack_id = "omniem:mito"
    _require(pack_id)
    torch = pytest.importorskip("torch")

    from quantem.inference._fig3.schema import load_head_config
    from quantem.inference.encoders import EncoderManifest, build_encoder

    files = engine.resolve_model_files(pack_id)
    if files.export_path is None:
        pytest.skip(
            f"{pack_id} has no exported encoder; rebuild the release bundle "
            "with `python -m quantem.registry.release build`"
        )
    if files.encoder_path is None:
        # There is nothing to compare against. A release bundle ships only the
        # exported encoder and deliberately omits the raw foundation weights
        # (quantem.registry.release), so on a bundle install -- the only shape
        # anyone outside the lab gets -- the eager tier cannot be built at all.
        # This is the maintainer's check, run against a local-path install
        # before the bundle is published; on a bundle it has no question to ask.
        pytest.skip(
            f"{pack_id} was installed from a release bundle, which ships no raw "
            "encoder to compare the exported one against. Run this against a "
            "local-path install, before publishing."
        )
    assert files.index_path is not None, "an installed pack always records its index"
    cfg = load_head_config(files.config_path)
    manifest = EncoderManifest.from_index(files.index_path)

    exported = build_encoder(
        manifest=manifest,
        encoder_path=files.encoder_path,
        export_path=files.export_path,
        apply_encoder_norm=cfg.encoder.apply_encoder_norm,
        device="cpu",
    )
    eager = build_encoder(
        manifest=manifest,
        encoder_path=files.encoder_path,
        export_path=None,
        apply_encoder_norm=cfg.encoder.apply_encoder_norm,
        device="cpu",
    )
    assert exported.contract.tier == "exported"
    assert eager.contract.tier == "timm"

    # The eager encoder is bare; give it the pack's LoRA adapters, which is what
    # the exported graph already has baked in.
    from quantem.inference._fig3.load_head import build_and_load_head

    eager_encoder: Any = eager.module
    exported_encoder: Any = exported.module
    build_and_load_head(cfg, eager_encoder, files.head_path, device="cpu")
    layers = cfg.encoder.resolved_layers(manifest.depth)

    spec = MODEL_SPECS[pack_id]
    x = torch.randn(1, 1, spec.tile_size, spec.tile_size)
    with torch.no_grad():
        want = eager_encoder.features(x, layers)
        got = exported_encoder.features(x, layers)

    assert len(want) == len(got) == len(layers)
    for a, b in zip(want, got, strict=True):
        assert a.shape == b.shape
        assert float((a - b).abs().max()) < 1e-4


def test_uninstalled_pack_says_how_to_install_it() -> None:
    """The failure a user actually hits first must be actionable."""
    with pytest.raises(engine.ModelWeightsNotInstalled) as excinfo:
        engine.resolve_model_files("omniem:not-a-real-organelle")
    assert "install" in str(excinfo.value).lower()

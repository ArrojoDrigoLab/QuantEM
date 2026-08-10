"""Write a ``checkpoint_index.json`` for each external foundation-model baseline weight.

The encoder-evaluation and segmentation harnesses load any frozen encoder through ``em_ssl.utils.CheckpointIndex`` (a
per-run ``checkpoint_index.json`` = an ``EncoderManifest`` + one ``CheckpointRecord`` per saved artifact).
Pretraining runs write theirs during training; the external baselines (EMCF-MAE, Meta-DINOv3, DINOv2
and OmniEM/EM-DINO) do not, so this script synthesises one per encoder pointing at the local weight
file. Paths are written as absolute paths, so the script is run once per host, after the weights root
has been transferred to it.

Usage:
    python -m foundation_baselines.register_external_encoders \
        --weights-root <encoder weights root> \
        [--only emcf_mae_vitb dinov3_meta_vitl omniem_emdino_vitl dinov2_l_base]

Writes ``<weights-root>/<name>/checkpoint_index.json`` for each registered encoder whose weight file
is present. ``--run-dir`` (the encoder-comparison run_probe) or ``encoder.run_dir`` (segmentation_training config) points at that folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from em_ssl.utils.checkpoint_index import CheckpointIndex, EncoderManifest
from foundation_baselines.external_vit import REGISTRY, ExternalEncoderSpec

def _feature_entry_point(spec: ExternalEncoderSpec) -> dict:
    """Everything the ``timm_vit`` loader needs to rebuild + load this encoder (kept in one place so
    the manifest is self-describing and the harness loader stays declarative)."""
    return {
        "loader": "timm_external",
        "timm_model": spec.timm_model,
        "img_size_build": spec.img_size_build,
        "dynamic_img_size": spec.dynamic_img_size,
        "strip_prefix": spec.strip_prefix,
        "drop_key_prefixes": list(spec.drop_key_prefixes),
        "allow_unexpected": list(spec.allow_unexpected),
        "in_chans": spec.in_chans,
        "tile_size": spec.tile_size(),   # the /patch-rounded crop these encoders must be fed
        "base_tile": spec.base_tile,
        "context_sweepable": spec.context_sweepable,  # RoPE (Meta-DINOv3) -> honors --context-tile
        "forward": "forward_intermediates",
    }

def register_one(spec: ExternalEncoderSpec, weights_root: Path) -> Path | None:
    enc_dir = weights_root / spec.name
    weight_path = enc_dir / spec.weight_file
    if not weight_path.exists():
        print(f"[skip] {spec.name}: weight file not found ({weight_path})")
        return None
    manifest = EncoderManifest(
        run_id=spec.name,
        framework="timm_vit",
        objective=spec.objective,
        arch=spec.arch,
        patch_size=spec.patch_size,
        embedding_dim=spec.embed_dim,
        depth=spec.depth,
        input_channels=spec.in_chans,
        # image_mean/std here are the encoder's native per-channel input stats (ImageNet for EMCF-MAE,
        # Meta-DINOv3 and DINOv2; EM stats for OmniEM). The dataset is told mean=0/std=1 for these (so
        # it hands the encoder a raw [0,1] tile); the real normalization is applied inside feature
        # extraction.
        image_mean=list(spec.norm_mean),
        image_std=list(spec.norm_std),
        feature_entry_point=_feature_entry_point(spec),
        notes=f"External public baseline ({spec.objective}) loaded via timm '{spec.timm_model}'. "
              f"Frozen; decoder-only training.",
    )
    index = CheckpointIndex(enc_dir, manifest)
    index.add(step=0, kind="encoder", path=str(weight_path.resolve()))
    print(f"[ok]   {spec.name}: {index.path}  ->  {weight_path.name} "
          f"(tile {spec.tile_size()}, {spec.arch}/{spec.patch_size}, dim {spec.embed_dim})")
    return index.path

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Register external foundation-model baseline encoders.")
    p.add_argument("--weights-root", default="foundation_weights",
                   help="Dir holding <name>/<weight_file> per encoder (default: repo-local foundation_weights).")
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to these encoder names (default: all registered).")
    args = p.parse_args(argv)

    weights_root = Path(args.weights_root)
    names = args.only or list(REGISTRY)
    n = 0
    for name in names:
        spec = REGISTRY.get(name)
        if spec is None:
            print(f"[skip] unknown encoder {name!r}; known: {sorted(REGISTRY)}")
            continue
        if register_one(spec, weights_root) is not None:
            n += 1
    print(f"Registered {n}/{len(names)} external encoders under {weights_root.resolve()}")

if __name__ == "__main__":
    main()

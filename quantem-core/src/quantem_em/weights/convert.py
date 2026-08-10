"""Packaging-time conversion of the reference artifacts into published safetensors.

Run once, by us, on a machine that can see the read-only source trees. The output is what users
download; nothing here runs at inference time.

Why this exists at all: the staged ``head.pt`` files are pickles loaded with
``weights_only=False``. A plugin that downloads a pickle from the internet and unpickles it is a
remote-code-execution vector, so the published artifacts are safetensors.

Artifact layout (flat, prefixed keys)::

    encoder.<timm name>     base or fine-tuned encoder tensors
    neck.<name>             the neck
    decoder.<name>          the decoder
    adapters.<name>         LoRA adapter modules (OmniEM only)

Encoder tensors are split so nothing is downloaded twice:

* ``quantem-vitb-trunk``  blocks 0-7 + embeddings + final norm + rope periods
* ``quantem-<organelle>``  blocks 8-11 (fine-tuned, per organelle) + neck + decoder
* ``quantem-er``           the WHOLE encoder + neck + decoder -- adapt=full, so it needs no trunk
* ``omniem-vitl``          the full ViT-L (LoRA never touches it)
* ``omniem-<organelle>``   adapters + neck + decoder
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ..registry import REGISTRY
from ..spec import ModelSpec

_HEAD_SUBDIR = {"quantem": "quantem", "omniem": "omniem"}
#: Blocks kept in the QuantEM trunk; 8-11 ship with each organelle because last_n fine-tunes them.
QUANTEM_TRUNK_BLOCKS = tuple(range(0, 8))


def _prefix(d: dict, p: str) -> dict:
    return {f"{p}{k}": v.contiguous() for k, v in d.items()}


def _is_block(key: str, blocks) -> bool:
    if not key.startswith("blocks."):
        return False
    return int(key.split(".")[1]) in blocks


def build_quantem_trunk(encoder_state: dict) -> dict:
    """Everything the three last_n heads share: blocks 0-7 + embeddings + norm + rope periods."""
    out = {}
    for k, v in encoder_state.items():
        if k == "__rope_periods__":
            out["rope.periods"] = v
        elif k.startswith("blocks."):
            if _is_block(k, QUANTEM_TRUNK_BLOCKS):
                out[k] = v
        else:
            out[k] = v
    return _prefix(out, "encoder.")


def build_omniem_trunk(encoder_state: dict) -> dict:
    return _prefix({k: v for k, v in encoder_state.items() if k != "__rope_periods__"}, "encoder.")


def build_model_artifact(spec: ModelSpec, head_state: dict, encoder_state: dict | None) -> dict:
    """The per-organelle artifact: whatever encoder tensors this model owns, plus neck + decoder."""
    from ..models.build import _remap_encoder_trainable

    out: dict = {}
    out.update(_prefix(head_state["neck"], "neck."))
    out.update(_prefix(head_state["decoder"], "decoder."))

    enc_tr = head_state.get("encoder_trainable")
    if enc_tr:
        remapped = _remap_encoder_trainable(enc_tr, spec)
        enc = {}
        for k, v in remapped.items():
            if k.startswith("backbone."):
                enc[k[len("backbone."):]] = v
            elif k.startswith("_conv_lora."):
                out[f"adapters.{k[len('_conv_lora.'):]}"] = v.contiguous()
        if spec.adapt == "full":
            # Self-contained: carry the rope periods too, since no trunk ships with it.
            if encoder_state is not None and "__rope_periods__" in encoder_state:
                enc["rope.periods"] = encoder_state["__rope_periods__"]
        out.update(_prefix(enc, "encoder."))
    elif head_state.get("adapters"):
        out.update(_prefix(head_state["adapters"], "adapters."))
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_artifact(tensors: dict, path: Path, metadata: dict | None = None) -> dict:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path), metadata=meta)
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path)}


def convert_all(
    out_dir,
    *,
    heads_root,
    quantem_ckpt,
    omniem_ckpt,
    registry_path=None,
) -> dict:
    """Produce every published artifact and update ``registry.json`` in place."""
    from ..models.build import load_reference_head
    from ..models.encoders import omniem_vit, quantem_vit

    out_dir = Path(out_dir)
    heads_root = Path(heads_root)
    results: dict[str, dict] = {}

    qem_enc = quantem_vit.load_reference_checkpoint(quantem_ckpt, REGISTRY["quantem/mito"].encoder)
    omni_enc = omniem_vit.load_reference_checkpoint(omniem_ckpt, REGISTRY["omniem/mito"].encoder)

    results["quantem-vitb-trunk"] = write_artifact(
        build_quantem_trunk(qem_enc), out_dir / "quantem-vitb-trunk.safetensors",
        {"kind": "encoder_trunk", "family": "quantem", "blocks": "0-7",
         "source": "m1_teacher_674999"},
    )
    results["omniem-vitl"] = write_artifact(
        build_omniem_trunk(omni_enc), out_dir / "omniem-vitl.safetensors",
        {"kind": "encoder_trunk", "family": "omniem", "source": "backbone_emdino_v1"},
    )

    for mid, spec in REGISTRY.items():
        head = load_reference_head(heads_root / f"{spec.organelle}_{_HEAD_SUBDIR[spec.family]}" / "head.pt")
        base = qem_enc if spec.family == "quantem" else omni_enc
        tensors = build_model_artifact(spec, head, base)
        results[spec.model_artifact] = write_artifact(
            tensors, out_dir / f"{spec.model_artifact}.safetensors",
            {"kind": "model", "model_id": mid, "arm_name": spec.arm_name,
             "organelle": spec.organelle, "adapt": spec.adapt,
             "canonical_nm": spec.canonical_nm, "task": spec.task},
        )
        # model card next to the artifact
        (out_dir / f"{spec.model_artifact}.json").write_text(
            json.dumps(_model_card(spec, results[spec.model_artifact]), indent=2), encoding="utf-8"
        )

    reg_path = Path(registry_path) if registry_path else Path(__file__).with_name("registry.json")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    for name, info in results.items():
        reg["artifacts"].setdefault(name, {})
        reg["artifacts"][name].update(
            {"filename": info["filename"], "bytes": info["bytes"], "sha256": info["sha256"]}
        )
    reg_path.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    return results


def _model_card(spec: ModelSpec, info: dict) -> dict:
    return {
        "model_id": spec.model_id,
        "arm_name": spec.arm_name,
        "organelle": spec.organelle,
        "family": spec.family,
        "architecture": {
            "encoder": spec.encoder.timm_model,
            "patch_size": spec.encoder.patch_size,
            "embed_dim": spec.encoder.embed_dim,
            "depth": spec.encoder.depth,
            "neck": spec.neck,
            "decoder": spec.decoder,
            "adaptation": spec.adapt,
            "feature_layers": spec.feature_layers,
        },
        "inference": {
            "tile_size": spec.effective_tile(),
            "overlap": spec.overlap,
            "stride": spec.stride(),
            "fg_threshold": spec.fg_threshold,
            "canonical_nm": spec.canonical_nm,
            "normalization": {"mean": spec.encoder.dataset_mean, "std": spec.encoder.dataset_std},
            "task": spec.task,
        },
        "artifact": info,
        "requires": ([spec.trunk_artifact] if spec.trunk_artifact else []) + [spec.model_artifact],
        # Single source of truth: the card must not be able to drift from what the plugin shows
        # the user in the download-consent dialog.
        "license": _license_of(spec.model_artifact),
    }


def _license_of(artifact: str) -> str:
    from .fetch import load_registry

    return load_registry()["artifacts"].get(artifact, {}).get("license", "")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Convert reference checkpoints to published artifacts.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heads", required=True, help="dino_organelle root with <org>_<family>/head.pt")
    ap.add_argument("--quantem-ckpt", required=True)
    ap.add_argument("--omniem-ckpt", required=True)
    a = ap.parse_args(argv)
    res = convert_all(a.out, heads_root=a.heads, quantem_ckpt=a.quantem_ckpt,
                      omniem_ckpt=a.omniem_ckpt)
    total = sum(v["bytes"] for v in res.values())
    for k, v in res.items():
        print(f"{k:24s} {v['bytes']/1e6:9.1f} MB  {v['sha256'][:16]}...")
    print(f"{'TOTAL':24s} {total/1e6:9.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

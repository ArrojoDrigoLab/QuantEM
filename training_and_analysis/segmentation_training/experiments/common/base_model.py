"""Load an adapted base — a trained encoder plus its trained segmentation head — for the experiments
that measure a change on top of an already-optimised model.

Decoder, loss and test-time choices are measured on the adapted backbone rather than on the frozen
probe, because rankings obtained with a frozen encoder do not transfer to the adapted one. Every
experiment under `experiments/` therefore starts from a trained head reloaded onto its encoder through
``harness.load_adapted.build_and_load_head``.

Encoder-agnostic: ``load_adapted_base`` takes the encoder run directory, the head weights and the
configuration the head was trained with as explicit arguments, so the same code runs against any
encoder run directory carrying a ``checkpoint_index.json``. The reference recipe per organelle is
recorded in ``BASE_RECIPE`` for documentation; the head's own resolved configuration is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The adapted-base recipe each experiment builds on, per organelle. Documentation only — the resolved
# configuration saved alongside a trained head is what actually gets loaded.
BASE_RECIPE = {
    "mito": {"neck": "resnet34_detail", "decoder": "affinity_mws", "loss": "dice_bce", "task": "instance"},
    "er": {"neck": "resnet34_detail", "decoder": "dpt", "loss": "dice_bce", "task": "semantic"},
}

_ADAPT_LOADS_ADAPTERS = ("lora", "conv_lora", "lora_ln", "cond_lora", "last_n", "full")


@dataclass
class LoadedBase:
    """A ready-to-evaluate adapted base plus the context each experiment needs to drive evaluation."""

    model: object          # SegModel (encoder + neck + decoder + adapters), on device, in eval mode
    cfg: object            # SegConfig the head was trained with, run_dir pointed at the local encoder
    encoder: object        # FrozenEncoder (base frozen; adapters trainable)
    vocab: object          # MetaVocab or None
    info: dict             # load bookkeeping (encoder_loaded / skipped ...)
    organelle: str
    device: str

    @property
    def mean(self) -> float:
        return float(self.encoder.image_mean)

    @property
    def std(self) -> float:
        return float(self.encoder.image_std)


def load_adapted_base(organelle: str, *, head: str | Path, config: str | Path,
                      run_dir: str | Path, device: str = "cpu",
                      adapt: str | None = None) -> LoadedBase:
    """Rebuild the adapted base for ``organelle`` and load a trained head onto it.

    Args:
        organelle: ``"mito"``, ``"er"``, ``"ld"`` or ``"nucleus"``.
        head:      ``head.pt`` written by the training run.
        config:    ``resolved_config.yaml`` from the same run.
        run_dir:   encoder run directory (the one holding ``checkpoint_index.json``).
        device:    ``"cpu"``, ``"cuda"`` or ``"cuda:N"``; falls back to cpu when CUDA is absent.
        adapt:     override ``cfg.encoder.adapt``. Leave unset to keep the head's own setting, which
                   must reinstall the same adapters the head was trained with or the saved adapter
                   tensors will not match by name and shape.

    The run directory recorded inside a resolved configuration is the one in use when the head was
    trained, so it is always replaced here by the ``run_dir`` argument.
    """
    from ...config.schema import load_seg_config
    from ...harness.load_adapted import build_and_load_head
    from ...harness.run_seg import resolve_device, resolve_encoder

    head, config, run_dir = Path(head), Path(config), Path(run_dir)
    if not head.exists():
        raise FileNotFoundError(f"trained head not found: {head}")
    if not config.exists():
        raise FileNotFoundError(f"resolved config not found: {config}")

    cfg = load_seg_config(str(config))
    cfg.encoder.run_dir = str(run_dir)
    if adapt is not None:
        cfg.encoder.adapt = adapt
    if str(cfg.encoder.adapt) not in _ADAPT_LOADS_ADAPTERS:
        raise ValueError(f"cfg.encoder.adapt={cfg.encoder.adapt!r} does not reinstall adapters, so the "
                         f"head's adapter tensors would be silently skipped; expected one of "
                         f"{_ADAPT_LOADS_ADAPTERS}.")

    device = resolve_device(device)
    enc, _rec = resolve_encoder(cfg, device)
    enc.to(device)
    model, vocab, info = build_and_load_head(cfg, enc, head, device=device)
    return LoadedBase(model=model, cfg=cfg, encoder=enc, vocab=vocab, info=info,
                      organelle=organelle, device=device)


def build_mock_base(tmp_dir: str | Path, organelle: str = "er", *, decoder: str = "dpt",
                    neck: str = "naive_1x1", adapt: str = "lora", tile_size: int = 64,
                    device: str = "cpu"):
    """A randomly initialised encoder plus a freshly built, untrained head with adapters installed.

    Exercises the same assembly and adaptation path as the real base without a trained checkpoint or a
    GPU, and returns the same ``LoadedBase`` shape. Used by the tests to check an arm's training and
    evaluation plumbing; the outputs are meaningless.
    """
    from em_ssl.utils.checkpoint_index import CheckpointIndex

    from ..._synthetic import write_mock_checkpoint
    from ...config.schema import SegConfig
    from ...harness.encoders import FrozenEncoder, select_checkpoints
    from ...harness.run_seg import resolve_device
    from ...harness.train import build_segmodel

    tmp_dir = Path(tmp_dir)
    run_dir = write_mock_checkpoint(tmp_dir / "mock_encoder", "dinov3", arch="vit_small")
    idx = CheckpointIndex.load(run_dir)
    rec = select_checkpoints(idx, n=1)[-1]
    enc = FrozenEncoder.from_manifest(rec.path, idx.manifest, tile_size=tile_size)
    device = resolve_device(device)
    enc.to(device)

    cfg = SegConfig()
    cfg.data.organelle = organelle
    cfg.data.task = BASE_RECIPE.get(organelle, {}).get("task", "semantic")
    cfg.encoder.tile_size = tile_size
    cfg.encoder.adapt = adapt
    cfg.encoder.adapt_params = {"rank": 8, "conv": True}
    cfg.neck.type = neck
    cfg.decoder.type = decoder
    model = build_segmodel(cfg, enc).to(device).eval()
    return LoadedBase(model=model, cfg=cfg, encoder=enc, vocab=None, info={"mock": True},
                      organelle=organelle, device=device)


if __name__ == "__main__":  # structural check on a trained head; no encoder needed
    import argparse
    import json

    from ...harness.load_adapted import inspect_head

    p = argparse.ArgumentParser(description="Report the structure of a trained segmentation head.")
    p.add_argument("--head", required=True)
    a = p.parse_args()
    print(json.dumps(inspect_head(Path(a.head)), indent=2, default=str))

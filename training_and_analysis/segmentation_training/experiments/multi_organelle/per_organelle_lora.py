"""The per-organelle-adapter arm, and the single-organelle specialist baseline.

The multi-task comparison has three arms:

  * ``shared-dodnet``      — one shared base and one DoDNet head, conditioned on an organelle code.
  * ``per-organelle-lora`` — one shared frozen base, but each organelle gets its own adapter set and
    its own neck and decoder. Each organelle is a specialist by construction, differing from
    ``shared-dodnet`` only in whether capacity is shared or separate. This module trains and evaluates
    one organelle's head.
  * ``specialist``         — an already-trained single-organelle head, loaded through
    ``common.base_model.load_adapted_base``. That head is already a per-organelle specialist, so this
    arm needs no new training; ``run_specialist_eval`` scores it through the same evaluation path for a
    comparable row.

So ``per-organelle-lora`` trained here is a re-derivation of the specialist under the same matched
template, and ``specialist`` is the trained head itself. Both are the no-sharing reference that the
shared-DoDNet arm is measured against.

Training the ``per-organelle-lora`` arm is exactly the baseline setup for one organelle, so the model
is assembled with ``harness.train.build_segmodel`` and trained with ``harness.train.train_segmodel``,
which keeps it matched to the baseline. This module is the wrapper that names the arm and keeps the
evaluation and reporting uniform.
"""

from __future__ import annotations

import copy


def build_per_organelle_config(cfg, organelle: str):
    """Deep-copy ``cfg`` into a single-organelle baseline recipe for ``organelle`` (its own LoRA + neck + decoder).

    Uses the per-organelle baseline recipe from ``config_templates.ORG_RECIPE`` (resnet34_detail neck; dpt for ER /
    affinity_mws for mito; instance-vs-semantic task). The encoder adaptation is kept as configured, and a frozen
    encoder is switched to rank-8 convolutional LoRA so the arm always has its own adapter set.
    """
    from ..common.config_templates import ORG_RECIPE

    c = copy.deepcopy(cfg)
    o = ORG_RECIPE[organelle]
    c.data.organelle = organelle
    c.data.task = o["task"]
    c.data.num_classes = int(o["num_classes"])
    c.neck.type = o["neck"]
    c.decoder.type = o["decoder"]
    if (getattr(c.encoder, "adapt", "frozen") or "frozen") == "frozen":
        c.encoder.adapt = "lora"
        c.encoder.adapt_params = {"rank": 8, "conv": True}
    return c


def train_per_organelle_lora(cfg, encoder, records, data_root, device: str = "cpu", *, organelle: str,
                             logger=None):
    """Train one organelle's own-LoRA head (the adapted-baseline specialist re-derivation). Reuses the standard
    trainer so it is step/seed/adapt-matched to the adapted baseline. Returns the trained ``SegModel``.

    A separate Conv-LoRA adapter set is installed on ``encoder`` for each organelle, so training several
    organelles in one process takes a distinct encoder instance per organelle (or a reset of
    ``encoder._conv_lora``); their adapters then hold independent weights, which is what makes each head a
    specialist by construction.
    """
    from ...harness.train import train_segmodel

    c = build_per_organelle_config(cfg, organelle)
    return train_segmodel(c, encoder, records, data_root, device, logger=logger,
                          tag=f"per_org_lora_{organelle}")


def evaluate_per_organelle(model, records, cfg, data_root, device, mean: float, std: float, *,
                           organelle: str) -> dict:
    """Standard sliding-window eval of a per-organelle-LoRA model (both metrics). ``{summary, per_crop}``."""
    from ...harness.evaluate import evaluate_head

    c = build_per_organelle_config(cfg, organelle)
    return evaluate_head(model, records, c, data_root, device, mean=mean, std=std)


def run_specialist_eval(organelle: str, *, data_root: str, device: str = "cuda", split: str = "test",
                        run_dir=None, head=None, config=None) -> dict:
    """Evaluate an already-trained baseline specialist head for ``organelle`` (the specialist baseline arm).

    Loads the trained head via ``common.base_model.load_adapted_base``, which is encoder-agnostic:
    the head's own resolved configuration names the encoder and the adapters to reinstall, and
    ``run_dir`` points at that encoder's checkpoint index. The head is then scored through the
    standard eval. This is the "no sharing, best per-organelle head" reference. Returns
    ``{split: {summary, per_crop}}`` for ``eval_report.assemble_report``.
    """
    from ..common.base_model import load_adapted_base
    from ...harness.dataset import load_manifest
    from ...harness.evaluate import evaluate_head

    base = load_adapted_base(organelle, head=head, config=config, run_dir=run_dir, device=device)
    group = base.cfg.data.resolved_group()
    bucket = getattr(base.cfg.data, "bucket", "canonical")
    recs = load_manifest(data_root, group, split, bucket=bucket)
    if len(recs) > 300:   # cap evaluation cost (mutex watershed is slow); stratified and seeded
        from ...harness.dataset import subset_fraction
        recs = subset_fraction(recs, 300 / len(recs), seed=int(getattr(base.cfg.optim, "seed", 0) or 0))
    out = evaluate_head(base.model, recs, base.cfg, data_root, base.device, mean=base.mean, std=base.std)
    return {split: out}

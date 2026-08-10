"""hook — image-style conditioning.

Wires the ``segmentation_training/models/conditioning`` building blocks onto a ``SegModel``: a
style-code producer (the inferred ``StyleEncoder``, or ``ConfidentFeatureStyle`` when
``cfg.cond.style_source`` selects the confident-feature code), FiLM / conditional-GroupNorm re-injected
at the neck+decoder norms ``cfg.cond.film_scope`` selects, optional MixStyle/DSU feature-statistic
mixing, and an optional gradient-reversed source adversary. Disabled (``cfg.cond.enabled=False``) ->
the SegModel is byte-identical to the base arm (no conditioning). Called from
``segmentation_training.harness.train.build_segmodel``.

Contract with the trainer / evaluator:
  * ``SegModel.forward`` calls ``conditioner.before_forward(image, feats)`` to compute + install the
    active style code, then runs neck+decoder (the FiLM hooks apply it); the adversary logits (if any) are
    stashed on ``conditioner.last_adv_logits`` for the trainer's GRL loss.
  * Per-forward context (metadata, source ids, DANN alpha, an optional preset dataset-scope code, an
    optional foreground mask) is set via ``set_context`` before the forward — the trainer sets it from
    the batch, the evaluator per record.

Torch-only (no GPU needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..models.conditioning.film import FiLMConditioner
from ..models.conditioning.mixstyle import MixerHooks, build_mixer
from ..models.conditioning.pooling import PrototypeGate, grid_encode, pool_by_source
from ..models.conditioning.style_encoder import (ConfidentFeatureStyle, DomainAdversary,
                                                  StyleEncoder)


def _select_mix_points(neck, decoder, which: str) -> list[nn.Module]:
    """Early neck (and optionally decoder) modules whose outputs get MixStyle/DSU (canonical: early only)."""
    pts: list[nn.Module] = []
    detail = getattr(neck, "detail", None)
    if detail is not None:  # resnet34_detail stem — the canonical layer1..3 insertion
        for nm in ("layer1", "layer2", "layer3"):
            m = getattr(detail, nm, None)
            if m is not None:
                pts.append(m)
    else:
        f = getattr(neck, "fuse", None) or getattr(neck, "tap_fuse", None)
        if f is not None:
            pts.append(f)
    if which == "neck_decoder" and decoder is not None:
        trunk = getattr(decoder, "sem_trunk", None) or getattr(decoder, "reassemble", None)
        if isinstance(trunk, nn.Module):
            pts.append(trunk)
    return pts


class Conditioner(nn.Module):
    """Owns the image-style conditioning machinery for one SegModel (registered as a submodule so its params train)."""

    def __init__(self, cfg, neck: nn.Module, decoder: nn.Module, field_sizes: dict[str, int] | None = None,
                 embed_dim: int = 0):
        super().__init__()
        c = cfg.cond
        field_sizes = field_sizes or {}
        self.vocab = None  # set post-build (harness/train) for record-level eval encoding
        self.style_dim = int(c.style_dim)
        self.style_source = str(c.style_source)
        self.style_scope = str(c.style_scope)
        self.n_prototypes = int(c.n_prototypes)
        self.metadata_dropout = float(c.metadata_dropout)
        self.grad_reversal = float(c.grad_reversal)
        self.adv_targets = list(c.adv_targets)

        # --- code producer -------------------------------------------------
        # Only a code consumer (FiLM or the adversary) needs a producer. MixStyle-only (film=False,
        # no adversary) builds no style encoder — it would otherwise be dead trainable params.
        self.style_encoder = None
        self.proto_gate = None
        self.confident_style = None
        needs_code = bool(c.film) or self.grad_reversal > 0
        if needs_code and self.style_source == "confident_feature":
            self.confident_style = ConfidentFeatureStyle(int(embed_dim), style_dim=self.style_dim,
                                                         k_proto=self.n_prototypes)
        elif needs_code:  # inferred appearance style
            # style_from_features feeds the coarsest encoder tap (embed_dim), globally pooled.
            self.style_encoder = StyleEncoder(
                style_dim=self.style_dim, hidden=int(c.style_hidden), use_stats=bool(c.style_stats),
                feat_dim=(int(embed_dim) if bool(c.style_from_features) else 0))
            if self.n_prototypes > 1:
                self.proto_gate = PrototypeGate(self.style_dim)

        # --- FiLM -------------------------------------------------------
        self.film = None
        if bool(c.film):
            targets: dict[str, nn.Module] = {}
            if bool(c.condition_neck):
                targets["neck"] = neck
            if bool(c.condition_decoder):
                targets["decoder"] = decoder
            self.film = FiLMConditioner(self.style_dim, targets, scope=str(c.film_scope))

        # --- MixStyle / DSU ---------------------------------------------
        self.mixer = None
        if str(c.mixstyle) != "off":
            mixer = build_mixer(str(c.mixstyle), float(c.mixstyle_p), float(c.mixstyle_alpha),
                                str(c.mixstyle_mix))
            pts = _select_mix_points(neck, decoder, str(c.mixstyle_points))
            if pts:
                self.mixer = MixerHooks(mixer, pts)

        # --- gradient-reversed adversary ---------------------------------
        self.adversary = None
        if self.grad_reversal > 0 and self.style_encoder is not None:
            adv_classes = {t: int(field_sizes.get(t, 2)) for t in self.adv_targets}
            self.adversary = DomainAdversary(self.style_dim, self.adv_targets, adv_classes,
                                             hidden=int(c.adv_hidden))

        # per-forward context
        self._meta: dict | None = None
        self._source_ids: torch.Tensor | None = None
        self._alpha: float = 0.0
        self._preset_code: torch.Tensor | None = None
        self._fg_mask: torch.Tensor | None = None  # pooled-global FiLM: confident/GT organelle mask for the appearance code
        self.last_adv_logits: dict | None = None
        # a retrieval arm: an externally-supplied {source: code} bank the evaluator uses instead of its own
        # dataset-scope precompute (so the retrieval-snapped / pooled code is not overwritten).
        self.source_style_override: dict | None = None

    # -- per-forward context ------------------------------------------------
    def set_context(self, meta: dict | None = None, source_ids=None, alpha: float | None = None,
                    preset_code: torch.Tensor | None = None, fg_mask: torch.Tensor | None = None) -> None:
        if meta is not None:
            self._meta = meta
        if source_ids is not None:
            self._source_ids = source_ids
        if alpha is not None:
            self._alpha = float(alpha)
        self._preset_code = preset_code
        self._fg_mask = fg_mask

    def clear_context(self) -> None:
        self._meta, self._source_ids, self._preset_code = None, None, None
        if self.film is not None:
            self.film.set_code(None)

    # -- code computation ---------------------------------------------------
    def compute_code(self, image: torch.Tensor, feats) -> torch.Tensor:
        b, device = image.shape[0], image.device
        if self._preset_code is not None:  # dataset-scope preset (test-time deployment code)
            code = self._preset_code.to(device)
            if code.dim() == 1:
                code = code.unsqueeze(0)
            if code.shape[0] == 1 and b > 1:
                code = code.expand(b, -1)
            return code
        if self.confident_style is not None:  # pooled-global FiLM: appearance code from confident organelle features
            code = self.confident_style(feats, self._fg_mask)
        elif self.n_prototypes > 1 and self.proto_gate is not None:
            grid = int(round(self.n_prototypes ** 0.5))
            protos = grid_encode(self.style_encoder, image, max(1, grid))
            code = self.proto_gate(protos)
        else:
            code = self.style_encoder(image, feats)
        if self.style_scope in ("source", "dataset"):
            code = pool_by_source(code, self._source_ids)
        if self.training and self.metadata_dropout > 0:
            keep = (torch.rand(b, 1, device=device) >= self.metadata_dropout).to(code.dtype)
            code = code * keep
        return code

    # -- record-level eval encoding (uses the saved vocab) ------------------
    def set_record_context(self, record: dict, device) -> None:
        """Set per-record metadata + source-id context for a single eval region (batch size 1)."""
        if self.vocab is None:
            return
        enc = self.vocab.encode(record)
        meta = {k: torch.tensor([v], dtype=torch.long, device=device) for k, v in enc.items()}
        self.set_context(meta=meta, source_ids=meta.get("dataset"))

    # -- driven by SegModel.forward ----------------------------------------
    def before_forward(self, image: torch.Tensor, feats) -> None:
        # Compute the style code only if something consumes it (FiLM or the adversary); mixer-only
        # skips it entirely.
        need_code = (self.film is not None) or (self.adversary is not None and self.training)
        if need_code:
            code = self.compute_code(image, feats)
            if self.film is not None:
                self.film.set_code(code)
            self.last_adv_logits = (self.adversary(code, self._alpha)
                                    if (self.adversary is not None and self.training) else None)
        else:
            self.last_adv_logits = None
        if self.mixer is not None:
            self.mixer.set_source_ids(self._source_ids)


def build_conditioner(cfg, neck: nn.Module, decoder: nn.Module,
                      field_sizes: dict[str, int] | None = None,
                      embed_dim: int = 0) -> Conditioner | None:
    """Build a ``Conditioner`` from ``cfg.cond`` (or ``None`` when conditioning is disabled)."""
    if not getattr(cfg, "cond", None) or not getattr(cfg.cond, "enabled", False):
        return None
    return Conditioner(cfg, neck, decoder, field_sizes=field_sizes, embed_dim=embed_dim)

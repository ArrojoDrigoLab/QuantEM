"""FiLM / conditional-GroupNorm conditioning — the mechanism that consumes the style code.

FiLM (Perez et al., AAAI 2018): ``y = gamma(s) * x + beta(s)``, per-channel, broadcast over space. The
segmentation neck/decoder is built throughout from ``ConvGNAct`` (Conv->GroupNorm->GELU) + bare
``GroupNorm``, so the canonical FiLM placement ("after norm, before activation") is realised by a forward
hook on each ``GroupNorm``/``InstanceNorm2d`` — the same non-invasive hook mechanism ConvLoRA uses on the
encoder. This gives per-block re-injection for free (single-point conditioning fades with depth): hook
every norm (``film_scope='per_block'``) or just the first (``'once'``, the ablation).

Identity at init: the FiLM head is zero-initialised and ``gamma = raw + 1``, so ``gamma≈1, beta≈0`` and the
conditioned model starts byte-identical to the unconditioned base, departing from it only as training makes
the code useful. This is load-bearing: a conditioned arm starts from exactly the corresponding
unconditioned base.

Reference: ethanjperez/film (vr/models/{filmed_net,film_gen}.py); the +1 gamma_baseline lives in the
generator, and the FiLM layer itself is literally ``gamma*x + beta`` (no extra +1). The mixture-of-experts
head follows DoDNet/MoE-adapter practice (K experts + a style-predicted gate).

Torch-only (no GPU needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _sanitize(name: str) -> str:
    """ModuleDict keys cannot contain '.'; map a module path to a safe key."""
    return name.replace(".", "__") or "root"


class FiLMHead(nn.Module):
    """Style code ``s -> (gamma, beta)`` for one injection point (C channels).

    Zero-initialised so ``gamma≈1`` (via +1 baseline) and ``beta≈0`` at init. With ``n_experts>1`` this is a
    style-conditioned mixture-of-experts head: K expert (gamma,beta) predictions combined by a
    softmax gate predicted from the same code.
    """

    def __init__(self, style_dim: int, channels: int, n_experts: int = 1):
        super().__init__()
        self.channels = int(channels)
        self.n_experts = max(1, int(n_experts))
        self.proj = nn.Linear(style_dim, self.n_experts * 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate = nn.Linear(style_dim, self.n_experts) if self.n_experts > 1 else None

    def forward(self, code: torch.Tensor):
        raw = self.proj(code)  # [B, n_experts * 2C]
        if self.n_experts > 1:
            raw = raw.view(-1, self.n_experts, 2 * self.channels)
            w = torch.softmax(self.gate(code), dim=1).unsqueeze(-1)  # [B, K, 1]
            raw = (raw * w).sum(dim=1)  # [B, 2C]
        gamma, beta = raw[:, :self.channels], raw[:, self.channels:]
        return gamma + 1.0, beta  # identity-at-init: effective gamma = 1 + raw


class FiLMConditioner(nn.Module):
    """Installs FiLM heads + forward hooks on the norm layers of the given modules.

    ``set_code(s)`` stashes the current ``[B, style_dim]`` code; each hook then modulates its norm's output
    with that point's ``(gamma, beta)``. ``set_code(None)`` (default) makes every hook a pass-through, so a
    forward without a code is byte-identical to the base model.
    """

    def __init__(self, style_dim: int, targets: dict[str, nn.Module], *, scope: str = "per_block",
                 n_experts: int = 1):
        super().__init__()
        self.style_dim = int(style_dim)
        self.scope = str(scope)
        self.heads = nn.ModuleDict()
        self._handles: list = []
        self._code: torch.Tensor | None = None
        self._points: list[str] = []
        self._attach(targets, n_experts)

    def _attach(self, targets: dict[str, nn.Module], n_experts: int) -> None:
        # Collect every conditionable norm across all targets first, then (for scope='once') keep only the
        # first — otherwise the second target (decoder) would add its own injection point.
        candidates: list[tuple[str, nn.Module, int]] = []
        for tag, module in targets.items():
            if module is None:
                continue
            for name, m in module.named_modules():
                if isinstance(m, nn.GroupNorm):
                    candidates.append((_sanitize(f"{tag}.{name}"), m, m.num_channels))
                elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                    candidates.append((_sanitize(f"{tag}.{name}"), m, m.num_features))
        if self.scope == "once":
            candidates = candidates[:1]
        for key, m, ch in candidates:
            self.heads[key] = FiLMHead(self.style_dim, ch, n_experts=n_experts)
            self._points.append(key)
            self._handles.append(m.register_forward_hook(self._mk_hook(key)))

    def _mk_hook(self, key: str):
        def hook(_module, _inp, out):
            code = self._code
            if code is None:
                return out
            gamma, beta = self.heads[key](code)  # [B, C]
            g = gamma.to(out.dtype).view(gamma.shape[0], gamma.shape[1], 1, 1)
            b = beta.to(out.dtype).view(beta.shape[0], beta.shape[1], 1, 1)
            return g * out + b
        return hook

    def set_code(self, code: torch.Tensor | None) -> None:
        self._code = code

    @property
    def points(self) -> list[str]:
        return list(self._points)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

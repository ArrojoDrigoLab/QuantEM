"""Single-channel-safe augmentation primitives operating on float tensors in [0, 1].

Every transform here is a plain callable taking and returning a CHW float tensor (any
channel count, including C=1). They avoid the RGB-only torchvision ops (saturation/hue
color jitter, RandomGrayscale, solarization) that break or are undefined on grayscale EM
images. Photometric ops are intentionally mild so they do not destroy membranes — see
em_dino_augmentations for the assembled pipelines.

All randomness uses the global torch RNG so DataLoader worker seeding controls it; pass an
explicit ``generator`` for deterministic tests.
"""

from __future__ import annotations

from typing import Callable

import torch
import torchvision.transforms.v2.functional as F

Tensor = torch.Tensor

def to_float_chw(img) -> Tensor:
    """Coerce a PIL image / numpy array / tensor to a float32 CHW tensor in [0, 1].

    Accepts: PIL.Image ('L', 'F', 'I', 'RGB'), HxW or CxHxW / HxWxC arrays/tensors,
    uint8 or float. A bare HxW input becomes 1xHxW (single channel).
    """
    if isinstance(img, torch.Tensor):
        t = img
    else:
        # PIL or numpy
        import numpy as np

        if hasattr(img, "mode"):  # PIL.Image
            arr = np.asarray(img)
        else:
            arr = np.asarray(img)
        t = torch.from_numpy(arr.copy())
    if t.ndim == 2:
        t = t.unsqueeze(0)  # HxW -> 1xHxW
    elif t.ndim == 3:
        # Heuristic: channels-last if last dim is small (1/3/4) and first dim is large.
        if t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
            t = t.permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image ndim={t.ndim} (shape {tuple(t.shape)})")
    t = t.to(torch.float32)
    # Scale uint8-range data to [0, 1]; data already in [0,1] is left alone.
    if t.numel() and float(t.max()) > 1.5:
        t = t / 255.0
    return t.clamp_(0.0, 1.0)

def _rand(generator: torch.Generator | None = None) -> float:
    return float(torch.rand((), generator=generator))

def _uniform(lo: float, hi: float, generator: torch.Generator | None = None) -> float:
    return lo + (hi - lo) * _rand(generator)

class MaybeApply:
    """Apply ``transform`` with probability ``p`` (else identity)."""

    def __init__(self, p: float, transform: Callable[[Tensor], Tensor]):
        self.p = float(p)
        self.transform = transform

    def __call__(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.p >= 1.0 or _rand(generator) < self.p:
            return self.transform(x)
        return x

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MaybeApply(p={self.p}, {self.transform!r})"

class RandomDihedral:
    """Random element of the dihedral group of the square (order 8): rot90^k ∘ optional flip.

    Encompasses horizontal flip, vertical flip, 90/180/270 rotation, and transpose — the
    full set of orientation symmetries that are meaningful and label-preserving for EM
    micrographs. Channel-agnostic. ``flips=False`` restricts it to the cyclic rotation
    subgroup, ``rotations=False`` to the flips alone, and ``enabled=False`` to the identity.
    """

    def __init__(self, enabled: bool = True, flips: bool = True, rotations: bool = True):
        self.enabled = enabled
        self.flips = flips
        self.rotations = rotations

    def __call__(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if not self.enabled:
            return x
        if self.rotations:
            k = int(torch.randint(0, 4, (), generator=generator))
            if k:
                x = torch.rot90(x, k, dims=(-2, -1))
        if self.flips:
            if _rand(generator) < 0.5:
                x = torch.flip(x, dims=(-1,))  # horizontal
            if _rand(generator) < 0.5:
                x = torch.flip(x, dims=(-2,))  # vertical
        return x

class RandomBrightnessContrast:
    """Mild, grayscale-safe brightness & contrast jitter on a [0,1] tensor.

    contrast scales around the per-channel spatial mean; brightness is an additive shift.
    Both factors are sampled uniformly and the result is clamped to [0,1]. Defaults are
    gentle to preserve membrane edges.
    """

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.contrast > 0:
            c = _uniform(1.0 - self.contrast, 1.0 + self.contrast, generator)
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - mean) * c + mean
        if self.brightness > 0:
            b = _uniform(-self.brightness, self.brightness, generator)
            x = x + b
        return x.clamp_(0.0, 1.0)

class RandomGamma:
    """Random gamma correction: x -> x**gamma with gamma in [1/(1+g), 1+g] (log-uniform).

    A grayscale-friendly, contrast-inversion-tolerant alternative to solarization. Operates
    in [0,1]; a small epsilon avoids 0**gamma gradient issues.
    """

    def __init__(self, gamma: float = 0.2, eps: float = 1e-6):
        self.gamma = float(gamma)
        self.eps = float(eps)

    def __call__(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.gamma <= 0:
            return x
        # Log-uniform around 1.0 so brightening/darkening are symmetric in log space.
        lo, hi = 1.0 / (1.0 + self.gamma), 1.0 + self.gamma
        import math

        g = math.exp(_uniform(math.log(lo), math.log(hi), generator))
        return x.clamp(self.eps, 1.0).pow(g).clamp_(0.0, 1.0)

class RandomGaussianNoise:
    """Add zero-mean Gaussian noise with std ~ U[0, sigma_max] (in [0,1] intensity units)."""

    def __init__(self, sigma_max: float = 0.04):
        self.sigma_max = float(sigma_max)

    def __call__(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.sigma_max <= 0:
            return x
        sigma = _uniform(0.0, self.sigma_max, generator)
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device) * sigma
        return (x + noise).clamp_(0.0, 1.0)

class RandomGaussianBlur:
    """Light Gaussian blur with random sigma in [sigma_min, sigma_max].

    Kernel size is fixed at construction (forced odd) and applies to every draw. Sigma range
    is kept mild by default so fine EM structure survives.
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 1.5, kernel_size: int = 9):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        # Kernel must be odd.
        self.kernel_size = int(kernel_size) | 1

    def __call__(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        sigma = _uniform(self.sigma_min, self.sigma_max, generator)
        return F.gaussian_blur(x, kernel_size=[self.kernel_size, self.kernel_size], sigma=[sigma, sigma])

class Compose:
    """Minimal sequential composition of the transforms above.

    Each is called with the shared ``generator``; one that does not accept the keyword is
    called with the tensor alone, so plain single-argument callables compose here too.
    """

    def __init__(self, transforms: list[Callable[[Tensor], Tensor]]):
        self.transforms = transforms

    def __call__(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        for t in self.transforms:
            try:
                x = t(x, generator=generator)
            except TypeError:
                x = t(x)
        return x

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        inner = ",\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {inner}\n])"

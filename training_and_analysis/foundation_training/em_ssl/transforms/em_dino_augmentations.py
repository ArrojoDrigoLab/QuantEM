"""EM-appropriate multi-crop DINO augmentation (single-channel).

Produces the per-sample dict that DINOv3's collate_data_and_cast consumes:

    {
        "weak_flag": True,
        "global_crops": [g1, g2],            # 2 tensors [1, G, G]  (forward_backward hardcodes 2)
        "global_crops_teacher": [g1t, g2t],  # == global_crops, or color-jitter-free if teacher_no_color_jitter
        "local_crops": [l1, ..., lN],        # N tensors [1, L, L]
        "offsets": (),                       # local-subset-of-global offsets (unused here)
        "gram_teacher_crops": [...],         # only if gram_teacher_crops_size is set
        "global_downsample": float,          # only under native_fov; geometric metadata, not a crop
                                             # tensor, and popped by the shard dataset before collate
    }

vs. upstream DataAugmentationDINO this:
  * keeps brightness/contrast jitter only (drops saturation/hue ColorJitter + RandomGrayscale),
  * replaces solarization with gentle gamma (contrast-inversion tolerant),
  * adds vertical flips + 90-degree rotations / dihedral symmetry (valid for EM),
  * adds mild Gaussian noise,
  * normalizes with single-channel EM mean/std,
  * never expands to 3 channels.

It depends only on torch/torchvision, so it is unit-testable on CPU and is wrapped (not
forked) into DINOv3 by em_ssl.integration.dinov3_patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from .primitives import (
    Compose,
    MaybeApply,
    RandomBrightnessContrast,
    RandomDihedral,
    RandomGamma,
    RandomGaussianBlur,
    RandomGaussianNoise,
    to_float_chw,
)

@dataclass
class EMAugmentationConfig:
    """EM-specific augmentation knobs (the upstream-incompatible parts).

    Defaults are deliberately mild — strong augmentation destroys membranes and fine EM
    texture. Every field is logged into the run's resolved config.
    """

    brightness: float = 0.2
    contrast: float = 0.2
    color_jitter_p: float = 0.8
    gamma: float = 0.2
    gamma_p: float = 0.2
    noise_sigma_max: float = 0.04
    noise_p: float = 0.2
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.5
    blur_kernel_size: int = 9
    # Per-crop blur probabilities (DINO asymmetry: global1 always blurred, global2 rarely).
    global1_blur_p: float = 1.0
    global2_blur_p: float = 0.1
    local_blur_p: float = 0.5
    dihedral: bool = True
    horizontal_flips: bool = True
    vertical_flips: bool = True
    rotations: bool = True

    # --- native-resolution field of view: match training magnification to inference ---
    # On a 2048px tile a plain random-resized crop downsamples every crop 2.3-4x, so an encoder trained
    # that way meets native-resolution input at inference that it never saw in training. native_fov
    # instead draws a per-crop downsample factor directly,
    #     M = 1 + (native_downsample_max - 1) * u**native_bias,   u ~ U(0,1)
    # where M=1 is a 1:1 native crop. native_bias above 1 concentrates mass near M=1; the tail up to
    # native_downsample_max supplies scale robustness. The two global crops share one window, sized
    # region * native_overlap_room, so they still overlap as DINO's global-global term requires. Locals
    # draw their own M up to native_local_downsample_max and are placed at random inside that same window.
    native_fov: bool = False
    native_downsample_max: float = 4.0
    native_bias: float = 3.0
    native_overlap_room: float = 1.5
    native_local_downsample_max: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

def native_magnification(downsample_max: float, bias: float, u: float) -> float:
    """Pure: per-crop downsample factor in [1, downsample_max]. ``u``∈[0,1]; ``bias``>1 biases toward
    1.0 (native). ``M = 1 + (downsample_max - 1)·u**bias`` (M=1 ⇒ a true 1:1 crop, no interpolation)."""
    return 1.0 + (max(1.0, downsample_max) - 1.0) * (u ** max(1e-6, bias))

def rand_crop_resize(src: torch.Tensor, region: int, out_size: int) -> torch.Tensor:
    """Random ``region×region`` crop of a CHW tensor, resized to ``out_size`` (bicubic, antialiased).
    Returns the patch unresized when ``region == out_size`` — a true 1:1 native crop. ``region`` is
    clamped to the tensor's real H/W, so a tile smaller than the request caps at its native size
    (never upscaled)."""
    _, h, w = src.shape
    region = max(1, min(int(region), h, w))
    top = int(torch.randint(0, h - region + 1, (1,)).item())
    left = int(torch.randint(0, w - region + 1, (1,)).item())
    patch = src[:, top : top + region, left : left + region]
    if region == out_size:
        return patch
    return v2.functional.resize(
        patch, [out_size, out_size], interpolation=InterpolationMode.BICUBIC, antialias=True
    )

class _Normalize:
    """Single-channel-safe normalization (x - mean) / std for a CHW float tensor."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"Normalize channel mismatch: tensor C={x.shape[0]} but mean/std C={self.mean.shape[0]}. "
                "For 1-channel EM, mean/std must be length-1 (never expand to 3)."
            )
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

class _RRC:
    """RandomResizedCrop on a CHW float tensor (bicubic, antialiased)."""

    def __init__(self, size: int, scale: tuple[float, float], ratio=(3 / 4, 4 / 3)):
        self.t = v2.RandomResizedCrop(
            size=size,
            scale=tuple(scale),
            ratio=tuple(ratio),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.t(x)

class _Resize:
    def __init__(self, size: int):
        self.t = v2.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.t(x)

class EMDataAugmentationDINO:
    """Single-channel EM multi-crop augmentation with the DINOv3 output-dict contract."""

    def __init__(
        self,
        global_crops_scale: tuple[float, float] = (0.32, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.32),
        local_crops_number: int = 8,
        global_crops_size: int = 224,
        local_crops_size: int = 96,
        gram_teacher_crops_size: int | None = None,
        gram_teacher_no_distortions: bool = False,
        teacher_no_color_jitter: bool = False,
        mean: Sequence[float] = (0.583,),
        std: Sequence[float] = (0.244,),
        em: EMAugmentationConfig | None = None,
        expected_channels: int = 1,
    ):
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.local_crops_number = local_crops_number
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.expected_channels = expected_channels
        self.em = em or EMAugmentationConfig()
        e = self.em

        self.normalize = _Normalize(mean, std)

        # --- geometric (channel-agnostic) ---
        dihedral = RandomDihedral(
            enabled=e.dihedral,
            flips=(e.horizontal_flips or e.vertical_flips),
            rotations=e.rotations,
        )
        self.geo_global = Compose([_RRC(global_crops_size, global_crops_scale), dihedral])
        self.geo_local = Compose([_RRC(local_crops_size, local_crops_scale), dihedral])
        self._dihedral = dihedral  # reused by the native_fov path (per-sample crops)

        # --- photometric (grayscale-safe) shared building blocks ---
        color = MaybeApply(e.color_jitter_p, RandomBrightnessContrast(e.brightness, e.contrast))
        noise = MaybeApply(e.noise_p, RandomGaussianNoise(e.noise_sigma_max))
        gamma = MaybeApply(e.gamma_p, RandomGamma(e.gamma))

        def blur(p):
            return MaybeApply(p, RandomGaussianBlur(e.blur_sigma_min, e.blur_sigma_max, e.blur_kernel_size))

        # global crop 1: always-ish blurred (DINO asymmetry); global crop 2: rarely blurred + gamma.
        self.photo_global1 = Compose([color, blur(e.global1_blur_p), noise, self.normalize])
        self.photo_global2 = Compose([color, blur(e.global2_blur_p), gamma, noise, self.normalize])
        self.photo_local = Compose([color, blur(e.local_blur_p), noise, self.normalize])

    # ------------------------------------------------------------------
    def _prep(self, image) -> torch.Tensor:
        x = to_float_chw(image)
        if x.shape[0] != self.expected_channels:
            if x.shape[0] == 3 and self.expected_channels == 1:
                # Defensive: collapse an accidental RGB to luminance rather than silently
                # training a 3-channel model. (Should not happen with the grayscale decoder.)
                x = x.mean(dim=0, keepdim=True)
            else:
                raise ValueError(
                    f"EMDataAugmentationDINO expected C={self.expected_channels}, got C={x.shape[0]}"
                )
        return x

    def _rand_crop(self, src: torch.Tensor, region: int, out_size: int) -> torch.Tensor:
        return rand_crop_resize(src, region, out_size)

    def _native_crops(self, x: torch.Tensor):
        """Magnification-controlled global+local crop bases (see EMAugmentationConfig.native_fov).

        Draws one global downsample ``M_g`` per sample (both globals share it and a window, so they
        overlap); each local draws its own ``M_l``. Returns ``(base1, base2, [local_bases], m_g)``
        where ``m_g`` is the *realized* global downsample (region actually used / crop size), shared
        by both globals — realized, so a small-tile clamp lowers it below the drawn magnification.
        Geometric only; photometric is applied by the caller."""
        _, h, w = x.shape
        g, l = self.global_crops_size, self.local_crops_size
        mg = native_magnification(self.em.native_downsample_max, self.em.native_bias, float(torch.rand(())))
        g_region = round(mg * g)
        win = min(round(g_region * self.em.native_overlap_room), h, w)
        window = self._rand_crop(x, win, win)  # one shared field of view (no resize)
        base1 = self._dihedral(self._rand_crop(window, g_region, g))
        base2 = self._dihedral(self._rand_crop(window, g_region, g))
        # Realized downsample seen by both globals: _rand_crop clamps the region to the window (itself
        # clamped to the tile), so on small tiles the true factor is below the drawn `mg`.
        m_g = max(1, min(g_region, win)) / g
        local_bases = []
        for _ in range(self.local_crops_number):
            ml = native_magnification(self.em.native_local_downsample_max, self.em.native_bias, float(torch.rand(())))
            local_bases.append(self._dihedral(self._rand_crop(window, round(ml * l), l)))
        return base1, base2, local_bases, m_g

    def __call__(self, image, target=None):
        x = self._prep(image)
        if self.em.native_fov:
            base1, base2, local_bases, global_downsample = self._native_crops(x)
        else:
            base1 = self.geo_global(x)
            base2 = self.geo_global(x)
            local_bases = [self.geo_local(x) for _ in range(self.local_crops_number)]
            # Standard RRC draws the two globals independently (no single shared scale to report).
            global_downsample = None

        g1 = self.photo_global1(base1)
        g2 = self.photo_global2(base2)
        global_crops = [g1, g2]

        if self.teacher_no_color_jitter:
            global_crops_teacher = [self.normalize(base1), self.normalize(base2)]
        else:
            global_crops_teacher = [g1, g2]

        local_crops = [self.photo_local(lb) for lb in local_bases]

        output: dict[str, Any] = {
            "weak_flag": True,
            "global_crops": global_crops,
            "global_crops_teacher": global_crops_teacher,
            "local_crops": local_crops,
            "offsets": (),
        }
        if global_downsample is not None:
            # Realized native_fov downsample of the global crops (both share it). The dataset bridges
            # this into FINO metadata so a crop-scale-aware scale factor can correct nm/px -> the
            # crop's true (downsampled) resolution, then pops it before the model collate. Not a crop
            # tensor — geometric metadata only.
            output["global_downsample"] = float(global_downsample)

        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                resize = _Resize(self.gram_teacher_crops_size)
                gram = [self.normalize(resize(base1)), self.normalize(resize(base2))]
            else:
                gram_rrc = _RRC(self.gram_teacher_crops_size, (0.32, 1.0))
                gram = [self.photo_global1(gram_rrc(x)), self.photo_global2(gram_rrc(x))]
            output["gram_teacher_crops"] = gram

        return output

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EMDataAugmentationDINO(global={self.global_crops_size}, local={self.local_crops_size}x"
            f"{self.local_crops_number}, gram={self.gram_teacher_crops_size}, em={self.em.to_dict()})"
        )

"""Head-only adaptation: freeze the encoder, train the neck + decoder.

A direct port of ``gk_gold_seg/scripts/finetune_cv/train_qem_k2_deploy.py``, the
152-line script behind the manuscript's guided fine-tuning result (base held-out
Dice 0.817 -> 0.870 in 17.8 s on a GPU, from two annotated crops). Every number
in the recipe is the reference's:

* freeze everything, then ``requires_grad_(True)`` on ``model.neck`` and
  ``model.decoder`` — 5.775 M trainable parameters for the QuantEM ViT-B
* AdamW, lr 1e-4, weight decay 1e-4, batch of one patch, 300 steps
* tile ``t = round_up(tile_size, patch)`` — 512 for patch-16, 518 for patch-14
* windows on a ``t // 2`` stride, kept only when at least 20 % of the window is
  inside a completed ROI; the target is :data:`IGNORE` wherever it is not
* random horizontal flip, vertical flip, and ``rot90`` with ``k`` in 0..3
* loss = ``cross_entropy(ignore_index=255)`` + soft Dice over valid pixels only

Why the encoder stays frozen: it is the published foundation model, and the
whole claim being reproduced is that a *user's own* handful of crops is enough
to move the decoder. Training the backbone on two crops overfits them, and it
also turns a laptop-scale job into a GPU-scale one — the forward-only backbone
is what keeps this viable without CUDA.

Torch is imported lazily inside the functions that need it, so importing this
module (and therefore the whole package, and therefore Django) works on an
install with no torch at all. That is not a nicety: ``threshold_only``
adaptation must stay available on such a machine.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from quantem.inference import resample
from quantem.inference.engine import normalize_tile
from quantem.inference.tiling import round_up

logger = logging.getLogger(__name__)

#: Target value for "not annotated"; matches ``cross_entropy(ignore_index=...)``
#: and :data:`quantem.segmentation.services.adapt.IGNORE`.
IGNORE = 255

#: Submodules trained by head-only adaptation. Named here rather than inline so
#: the failure message below can list exactly what a module must expose.
HEAD_MODULES: tuple[str, ...] = ("neck", "decoder")

#: On-disk format tag written into every saved head.
HEAD_FORMAT = "quantem-adapted-head/1"


class HeadAdaptationUnavailable(RuntimeError):
    """Head training cannot run here. Carries a user-facing explanation."""


class CropArrays(Protocol):
    """What the trainer needs from a crop: the pixels and where they count."""

    name: str
    em: np.ndarray | None
    gt: np.ndarray
    valid: np.ndarray


def torch_available() -> bool:
    """Whether this install can run head adaptation at all."""
    try:
        import torch  # noqa: F401, PLC0415 -- probe only

        return True
    except Exception:
        return False


def _torch():
    import torch  # noqa: PLC0415 -- deliberately lazy; see the module docstring

    return torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptConfig:
    """The reference recipe. Defaults are the published values."""

    steps: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 0
    #: Keep a training window only when this fraction of it is inside a
    #: completed ROI. Below it the window is nearly all ``ignore`` and the step
    #: is noise.
    min_valid_fraction: float = 0.2
    ignore_index: int = IGNORE

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "min_valid_fraction": self.min_valid_fraction,
        }


@dataclass(frozen=True)
class AdaptProgress:
    """One progress tick, for a job reporter."""

    step: int
    total_steps: int
    loss: float
    elapsed_s: float

    @property
    def fraction(self) -> float:
        return (self.step + 1) / max(1, self.total_steps)

    @property
    def eta_s(self) -> float:
        done = self.step + 1
        if done <= 0 or self.elapsed_s <= 0:
            return 0.0
        return self.elapsed_s * (self.total_steps - done) / done


ProgressCallback = Callable[[AdaptProgress], None]


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def tile_for(tile_size: int, patch_size: int) -> int:
    """Training tile edge: the model's tile rounded up to whole patches."""
    return round_up(int(tile_size), int(patch_size))


def build_patches(
    crops: Sequence[CropArrays],
    tile: int,
    *,
    image_mean: float,
    image_std: float,
    config: AdaptConfig = AdaptConfig(),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut normalised ``(image, target)`` training windows out of the crops.

    Windows are taken on a ``tile // 2`` stride and kept only when at least
    ``min_valid_fraction`` of the window is inside a completed ROI. The target is
    the confirmed-object mask with :data:`IGNORE` everywhere the user did not
    annotate, so the loss never learns "background" from an unlabelled pixel.

    A crop smaller than one tile is padded up to ``tile`` — reflected EM, and
    ``valid = 0`` in the padding so it contributes nothing but shape. The
    reference simply dropped such crops, which is right for its 2048 px research
    crops and wrong for a user who drew a 400 px region; the 20 % rule still
    rejects a region too small to be worth a step.
    """
    stride = max(1, tile // 2)
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    min_valid = config.min_valid_fraction * tile * tile

    for crop in crops:
        if crop.em is None:
            raise ValueError(f"crop {crop.name!r} has no EM pixels loaded")
        em, gt, valid = _pad_to_tile(crop.em, crop.gt, crop.valid, tile)
        normalised = normalize_tile(em, image_mean, image_std)
        height, width = em.shape[:2]
        for y in range(0, max(1, height - tile + 1), stride):
            for x in range(0, max(1, width - tile + 1), stride):
                window_valid = valid[y : y + tile, x : x + tile]
                if window_valid.shape != (tile, tile):
                    continue
                if window_valid.sum() < min_valid:
                    continue
                target = gt[y : y + tile, x : x + tile].astype(np.int64)
                target[window_valid == 0] = config.ignore_index
                patches.append(
                    (normalised[y : y + tile, x : x + tile].copy(), target)
                )
    return patches


def _pad_to_tile(
    em: np.ndarray, gt: np.ndarray, valid: np.ndarray, tile: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a crop up to at least ``tile`` on both edges; padding is not valid."""
    height, width = em.shape[:2]
    pad_y = max(0, tile - height)
    pad_x = max(0, tile - width)
    if not pad_y and not pad_x:
        return em, gt, valid
    pads = ((0, pad_y), (0, pad_x))
    return (
        np.pad(em, pads, mode="reflect"),
        np.pad(gt, pads, mode="constant", constant_values=0),
        np.pad(valid, pads, mode="constant", constant_values=0),
    )


def augment(
    image: np.ndarray, target: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random h-flip, v-flip and ``rot90``. Dihedral only — EM has no canonical
    orientation, but a scale or shear would change the object sizes the decoder
    is being calibrated on."""
    if rng.random() < 0.5:
        image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
    if rng.random() < 0.5:
        image, target = image[::-1].copy(), target[::-1].copy()
    k = int(rng.integers(4))
    return np.rot90(image, k).copy(), np.rot90(target, k).copy()


# ---------------------------------------------------------------------------
# Freezing and the loss
# ---------------------------------------------------------------------------


def freeze_to_head(module: Any) -> tuple[list[Any], int]:
    """Freeze everything, unfreeze neck + decoder. Returns (params, count).

    Raises:
        HeadAdaptationUnavailable: when the loaded module does not expose a neck
            and a decoder to train. That is the state today for a TorchScript
            export, and saying so beats a silent no-op run that reports a Dice
            change of exactly zero.
    """
    missing = [name for name in HEAD_MODULES if getattr(module, name, None) is None]
    if missing:
        raise HeadAdaptationUnavailable(
            "This model cannot be head-adapted: it does not expose "
            f"{' and '.join(missing)}. Head-only training needs a module with a "
            f"separable {' + '.join(HEAD_MODULES)}; threshold calibration works "
            "on any model."
        )

    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for name in HEAD_MODULES:
        for parameter in getattr(module, name).parameters():
            parameter.requires_grad_(True)

    trainable = [p for p in module.parameters() if p.requires_grad]
    return trainable, int(sum(p.numel() for p in trainable))


def head_loss(logits: Any, target: Any, *, ignore_index: int = IGNORE) -> Any:
    """Cross-entropy plus soft Dice, both blind to ignored pixels.

    Cross-entropy alone under-weights a small foreground; Dice alone is unstable
    when a window contains no object at all. The reference used the sum, and the
    Dice term is computed only over valid pixels so an unannotated region can
    neither reward nor punish the model.
    """
    torch = _torch()
    functional = torch.nn.functional

    ce = functional.cross_entropy(logits, target[None], ignore_index=ignore_index)
    prob = torch.softmax(logits, 1)[:, 1]
    valid = (target != ignore_index).float()[None]
    foreground = (target == 1).float()[None]
    intersection = (prob * foreground * valid).sum()
    denominator = ((prob + foreground) * valid).sum()
    return ce + (1 - (2 * intersection + 1) / (denominator + 1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class HeadTrainingResult:
    """What one adaptation run did, for the record and for the UI."""

    steps: int
    trainable_params: int
    n_patches: int
    tile: int
    seconds: float
    losses: list[float] = field(default_factory=list)

    @property
    def final_loss(self) -> float | None:
        return self.losses[-1] if self.losses else None

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "trainable_params": self.trainable_params,
            "n_patches": self.n_patches,
            "tile": self.tile,
            "seconds": round(self.seconds, 2),
            "final_loss": self.final_loss,
        }


def train_head(
    module: Any,
    patches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    device: str = "cpu",
    config: AdaptConfig = AdaptConfig(),
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> HeadTrainingResult:
    """Run the head-only loop over ``patches``.

    ``should_cancel`` is polled every step so a user can stop a run that is
    taking longer than they expected; the partially trained head is returned
    rather than discarded, and the caller decides what to do with it.
    """
    if not patches:
        raise HeadAdaptationUnavailable(
            "No training window survived: every completed area is smaller than "
            f"the model's {config.min_valid_fraction:.0%} coverage rule for one "
            "tile. Annotate a larger region, or use threshold calibration."
        )

    torch = _torch()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    trainable, n_trainable = freeze_to_head(module)
    optimizer = torch.optim.AdamW(
        trainable, lr=config.lr, weight_decay=config.weight_decay
    )
    tile = int(patches[0][0].shape[0])
    logger.info(
        "Head-only adaptation: %d patches, %.3f M trainable params, %d steps on %s",
        len(patches),
        n_trainable / 1e6,
        config.steps,
        device,
    )

    module.train()
    losses: list[float] = []
    started = time.time()
    for step in range(config.steps):
        if should_cancel is not None and should_cancel():
            logger.info("Head adaptation cancelled at step %d", step)
            break
        image, target = augment(*patches[int(rng.integers(len(patches)))], rng)
        x = torch.from_numpy(image)[None, None].float().to(device)
        y = torch.from_numpy(target).to(device)
        loss = head_loss(module(x), y, ignore_index=config.ignore_index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if on_progress is not None:
            on_progress(
                AdaptProgress(
                    step=step,
                    total_steps=config.steps,
                    loss=losses[-1],
                    elapsed_s=time.time() - started,
                )
            )

    module.eval()
    return HeadTrainingResult(
        steps=len(losses),
        trainable_params=n_trainable,
        n_patches=len(patches),
        tile=tile,
        seconds=time.time() - started,
        losses=losses,
    )


# ---------------------------------------------------------------------------
# Saving and reloading
# ---------------------------------------------------------------------------


def save_head(module: Any, path: Path, *, meta: dict[str, object] | None = None) -> Path:
    """Write the trained neck + decoder to ``path``.

    Only the trained submodules are stored. The frozen encoder is already on
    disk in the model registry, addressed by digest, and writing a second
    525 MB copy per adapter would be the single largest thing this app ever
    wrote for no information gained.
    """
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"format": HEAD_FORMAT, "meta": dict(meta or {})}
    for name in HEAD_MODULES:
        payload[name] = {
            key: value.detach().cpu()
            for key, value in getattr(module, name).state_dict().items()
        }
    torch.save(payload, str(path))
    return path


def load_head(module: Any, path: Path) -> dict[str, object]:
    """Load a saved head onto a freshly built module.

    Used for the reference's verification step: an adapter that does not
    reproduce its own reported Dice when reloaded is not a deliverable.
    """
    torch = _torch()
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    fmt = payload.get("format")
    if fmt != HEAD_FORMAT:
        raise HeadAdaptationUnavailable(
            f"{Path(path).name} is not a QuantEM adapted head (format={fmt!r})."
        )
    for name in HEAD_MODULES:
        getattr(module, name).load_state_dict(payload[name])
    module.eval()
    return dict(payload.get("meta") or {})


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaledCrop:
    """A crop moved onto the grid the model works on.

    Training windows must be cut at *model* scale. The reference's crops were
    extracted already resampled to the model's canonical 8 nm; QuantEM extracts
    at the asset's native scale, so the resampling that inference does per run
    has to happen here too — otherwise the decoder is calibrated on organelles
    of the wrong apparent size.
    """

    name: str
    em: np.ndarray
    gt: np.ndarray
    valid: np.ndarray
    context: resample.ResampleContext


def to_model_scale_crop(crop: CropArrays, *, canonical_nm: float | None) -> ScaledCrop:
    """Resample one crop and its labels to the model's canonical pixel size."""
    if crop.em is None:
        raise ValueError(f"crop {crop.name!r} has no EM pixels loaded")
    context = resample.plan_resample(
        crop.em.shape[:2], getattr(crop, "pixel_size_nm", None), canonical_nm
    )
    gt, valid = masks_to_model_scale(crop.gt, crop.valid, context)
    return ScaledCrop(
        name=crop.name,
        em=resample.to_model_scale(crop.em, context),
        gt=gt,
        valid=valid,
        context=context,
    )


def masks_to_model_scale(
    gt: np.ndarray, valid: np.ndarray, context: resample.ResampleContext
) -> tuple[np.ndarray, np.ndarray]:
    """Move the labels to the grid the model predicted on, NEAREST both ways.

    The alternative — upsampling the probability map to native and thresholding
    it there — re-decides the boundary on interpolated values the model never
    produced (see :mod:`quantem.inference.resample`). Scoring happens where the
    prediction was made.
    """
    if context.is_identity:
        return gt, valid
    nearest = {
        "downscale_interpolation": cv2.INTER_NEAREST,
        "upscale_interpolation": cv2.INTER_NEAREST,
    }
    return (
        resample.to_model_scale(gt, context, **nearest),
        resample.to_model_scale(valid, context, **nearest),
    )

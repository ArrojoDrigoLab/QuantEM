"""Masked head adaptation with complete annotation coverage and validation telemetry."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
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

#: Tile-scaled training schedule. Fifteen or fewer tiles use the published
#: 300-step recipe; each tile after that adds 20 steps until 30 tiles reaches
#: the 600-step ceiling. These live beside :class:`AdaptConfig` because the API
#: planner and the worker must use one policy.
MIN_ADAPT_STEPS = 300
STEPS_PER_TRAINING_TILE = 20
MAX_ADAPT_STEPS = 20000
QUICK_ADAPT_STEPS = 600


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
    schedule: str = "quick"
    validation_fraction: float = 0.0
    eval_interval: int = 100
    patience: int = 0
    min_delta: float = 1e-4
    ignore_index: int = IGNORE

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "schedule": self.schedule,
            "validation_fraction": self.validation_fraction,
            "eval_interval": self.eval_interval,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }


def steps_for_training_tiles(tile_count: int, *, schedule: str = "quick") -> int:
    """Optimizer steps for one round containing ``tile_count`` training tiles."""
    scaled = max(0, int(tile_count)) * STEPS_PER_TRAINING_TILE
    ceiling = MAX_ADAPT_STEPS if schedule == "extended" else QUICK_ADAPT_STEPS
    return max(int(tile_count), min(ceiling, max(MIN_ADAPT_STEPS, scaled)))


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


def window_starts(length: int, tile: int, stride: int) -> list[int]:
    """Include the final edge even when it does not land on the stride."""
    end = max(0, length - tile)
    return sorted(set(range(0, end + 1, stride)) | {end})


def iter_windows(
    valid: np.ndarray, tile: int, *, config: AdaptConfig = AdaptConfig()
) -> Iterator[tuple[int, int]]:
    """``(y, x)`` of every training window a ``valid`` mask yields.

    The one definition of "how many tiles is this region worth". Both the
    trainer (:func:`build_patches`) and the pre-run count the dialog shows
    (:func:`quantem.finetune.scope.count_tiles`) walk it, so the number the user
    was promised is the number the loop produces — a second implementation would
    have drifted the first time either rule changed.

    ``valid`` must already be padded to at least ``tile`` on both edges; see
    :func:`_pad_to_tile`.
    """
    stride = max(1, tile // 2)
    height, width = valid.shape[:2]
    rows, cols = np.nonzero(valid)
    if not len(rows):
        return

    def starts(low: int, high: int, length: int) -> list[int]:
        span = high - low
        if span <= tile:
            return [max(0, min(length - tile, low - (tile - span) // 2))]
        return [low + offset for offset in window_starts(span, tile, stride)]

    for y in starts(int(rows.min()), int(rows.max()) + 1, height):
        for x in starts(int(cols.min()), int(cols.max()) + 1, width):
            window = valid[y : y + tile, x : x + tile]
            if window.shape != (tile, tile):
                continue
            if not window.any():
                continue
            yield y, x


def build_patches(
    crops: Sequence[CropArrays],
    tile: int,
    *,
    image_mean: float,
    image_std: float,
    config: AdaptConfig = AdaptConfig(),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut normalised ``(image, target)`` training windows out of the crops.

    Every window containing an annotated pixel contributes, including empty
    background regions. Unannotated image context and padding are ignored.
    """
    patches: list[tuple[np.ndarray, np.ndarray]] = []

    for crop in crops:
        if crop.em is None:
            raise ValueError(f"crop {crop.name!r} has no EM pixels loaded")
        em, gt, valid = _pad_to_tile(crop.em, crop.gt, crop.valid, tile)
        normalised = normalize_tile(em, image_mean, image_std)
        for y, x in iter_windows(valid, tile, config=config):
            window_valid = valid[y : y + tile, x : x + tile]
            target = gt[y : y + tile, x : x + tile].astype(np.int64)
            target[window_valid == 0] = config.ignore_index
            patches.append((normalised[y : y + tile, x : x + tile].copy(), target))
    return patches


def pad_masks_to_tile(
    gt: np.ndarray, valid: np.ndarray, tile: int
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_pad_to_tile` for a caller that has labels but no image pixels.

    Counting tiles must not need the image off disk: the dialog answers before
    anything has been decoded.
    """
    height, width = valid.shape[:2]
    pad_y = max(0, tile - height)
    pad_x = max(0, tile - width)
    if not pad_y and not pad_x:
        return gt, valid
    pads = ((0, pad_y), (0, pad_x))
    return (
        np.pad(gt, pads, mode="constant", constant_values=0),
        np.pad(valid, pads, mode="constant", constant_values=0),
    )


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

    if not torch.any(target != ignore_index):
        return logits.sum() * 0
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
    history: list[dict[str, float | int | None]] = field(default_factory=list)
    selected_step: int = 0
    stopped_early: bool = False
    patches_seen: int = 0

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
            "history": self.history,
            "selected_step": self.selected_step,
            "stopped_early": self.stopped_early,
            "patches_seen": self.patches_seen,
            "coverage_fraction": self.patches_seen / max(1, self.n_patches),
        }


def train_head(
    module: Any,
    patches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    device: str = "cpu",
    config: AdaptConfig = AdaptConfig(),
    validation_patches: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> HeadTrainingResult:
    """Run the head-only loop over ``patches``.

    ``should_cancel`` is polled every step so a user can stop a run that is
    taking longer than they expected; the partially trained head is returned
    rather than discarded, and the caller decides what to do with it.
    """
    if not patches:
        raise HeadAdaptationUnavailable("No training window contains an annotated pixel.")

    torch = _torch()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    trainable, n_trainable = freeze_to_head(module)
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    tile = int(patches[0][0].shape[0])
    logger.info(
        "Head-only adaptation: %d patches, %.3f M trainable params, %d steps on %s",
        len(patches),
        n_trainable / 1e6,
        config.steps,
        device,
    )

    # Frozen encoder buffers and stochastic layers must stay fixed, too.
    module.eval()
    for name in HEAD_MODULES:
        getattr(module, name).train()
    history = []
    best_loss, best_state, selected_step, stale = float("inf"), None, 0, 0
    order = rng.permutation(len(patches))
    seen = set()
    stopped_early = False
    losses: list[float] = []
    started = time.time()
    for step in range(config.steps):
        if should_cancel is not None and should_cancel():
            logger.info("Head adaptation cancelled at step %d", step)
            break
        if step and step % len(patches) == 0:
            order = rng.permutation(len(patches))
        patch_index = int(order[step % len(patches)])
        seen.add(patch_index)
        image, target = augment(*patches[patch_index], rng)
        x = torch.from_numpy(image)[None, None].float().to(device)
        y = torch.from_numpy(target).to(device)
        loss = head_loss(module(x), y, ignore_index=config.ignore_index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if (step + 1) % config.eval_interval == 0 or step + 1 == config.steps:
            val_loss = None
            val_dice = None
            if validation_patches:
                module.eval()
                values, dice = [], []
                with torch.no_grad():
                    for vi, vt in validation_patches:
                        vx = torch.from_numpy(vi)[None, None].float().to(device)
                        vy = torch.from_numpy(vt).to(device)
                        logits = module(vx)
                        values.append(float(head_loss(logits, vy).item()))
                        valid = vy != config.ignore_index
                        pred = (logits.argmax(1)[0] == 1) & valid
                        truth = (vy == 1) & valid
                        denom = int(pred.sum() + truth.sum())
                        if denom:
                            dice.append(float(2 * (pred & truth).sum() / denom))
                val_loss = float(np.mean(values))
                val_dice = float(np.mean(dice)) if dice else None
                if val_loss < best_loss - config.min_delta:
                    best_loss, selected_step, stale = val_loss, step + 1, 0
                    best_state = {
                        name: {
                            k: v.detach().cpu().clone()
                            for k, v in getattr(module, name).state_dict().items()
                        }
                        for name in HEAD_MODULES
                    }
                else:
                    stale += 1
                for name in HEAD_MODULES:
                    getattr(module, name).train()
            history.append(
                {
                    "step": step + 1,
                    "train_loss": float(np.mean(losses[-config.eval_interval :])),
                    "validation_loss": val_loss,
                    "validation_dice": val_dice,
                }
            )
        if on_progress is not None:
            on_progress(
                AdaptProgress(
                    step=step,
                    total_steps=config.steps,
                    loss=losses[-1],
                    elapsed_s=time.time() - started,
                )
            )

        if (
            validation_patches
            and config.patience
            and stale >= config.patience
            and len(seen) == len(patches)
        ):
            stopped_early = True
            break

    if best_state is not None:
        for name in HEAD_MODULES:
            getattr(module, name).load_state_dict(best_state[name])

    module.eval()
    return HeadTrainingResult(
        steps=len(losses),
        trainable_params=n_trainable,
        n_patches=len(patches),
        tile=tile,
        seconds=time.time() - started,
        losses=losses,
        history=history,
        selected_step=selected_step or len(losses),
        stopped_early=stopped_early,
        patches_seen=len(seen),
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
            key: value.detach().cpu() for key, value in getattr(module, name).state_dict().items()
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
    """Resample supervision without treating unknown source pixels as background.

    At reduced resolution a cell is valid if any annotated source area covers
    it; its label is the majority of that annotated area only. This preserves
    even subpixel completed areas. Upsampling uses nearest-neighbour labels.
    """
    if context.is_identity:
        return gt, valid
    interpolation = cv2.INTER_AREA if context.factor < 1 else cv2.INTER_NEAREST
    opts = {"downscale_interpolation": interpolation, "upscale_interpolation": interpolation}
    coverage = resample.to_model_scale(valid.astype(np.float32), context, **opts)
    foreground = resample.to_model_scale((gt * valid).astype(np.float32), context, **opts)
    present = coverage > 0
    return ((foreground >= coverage * 0.5) & present).astype(np.uint8), present.astype(np.uint8)

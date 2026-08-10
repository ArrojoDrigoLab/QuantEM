"""Load a model once, run it over a region.

The model is loaded once into this process and kept in :data:`_MODEL_CACHE`,
rather than being re-read from disk on every call. A 4-crop ROI therefore pays
one ViT-L load, not four.

Structure of one call::

    handle = load_model("quantem:mito")          # cached per (pack, device)
    pred   = predict_region(handle, image_uint8, pixel_size_nm=4.2)
    # pred.prob is at MODEL scale; threshold it, then map the mask back with
    # pred.context (see quantem.inference.resample for why that order).

Everything except the model *forward* is independently testable: resampling,
padding, the tile plan, Hann blending, normalisation, the softmax/foreground
reduction, the cache and its eviction. Pass ``forward=`` to
:func:`predict_region` to drive the whole path with a stand-in.

The forward itself is real. The released checkpoints are bare ``state_dict``s,
so :func:`build_module` rebuilds the architecture each head was trained as --
neck, decoder and adapter wiring from :mod:`quantem.inference._fig3`, encoder
from :mod:`quantem.inference.encoders` -- driven by the pack's own
``resolved_config.yaml``, and loads the head into it strictly. A pack that has
been through :mod:`quantem.inference.export` skips the encoder rebuild entirely
and loads a TorchScript artifact instead.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from . import resample, tiling
from .device import autocast_dtype, empty_cache, select_device
from .specs import MODEL_SPECS, ModelSpec

if TYPE_CHECKING:  # torch must not be imported at module scope; see _torch()
    from .encoders import EncoderContract

logger = logging.getLogger(__name__)

#: How many loaded models to keep resident. A ViT-L pack is >1 GB of weights;
#: two is enough to flip between families without thrashing, and bounded.
MAX_CACHED_MODELS = 2

TileForward = Callable[[np.ndarray], np.ndarray]


class ModelUnavailableError(RuntimeError):
    """A model cannot be run right now. Carries a user-facing explanation."""


class ModelWeightsNotInstalled(ModelUnavailableError):
    """The weights for this pack have not been downloaded yet."""


class ModelArchitectureUnavailable(ModelUnavailableError):
    """Weights are present but nothing here can turn them into a module."""


def _torch() -> ModuleType:
    import torch  # noqa: PLC0415 -- deliberately lazy; keeps Django startup fast

    return torch


# --- Weight resolution ------------------------------------------------------


@dataclass(frozen=True)
class ModelFiles:
    """On-disk artifacts for one model pack."""

    pack_id: str
    head_path: Path
    encoder_path: Path | None = None
    #: The pack's released ``resolved_config.yaml`` -- the neck/decoder/adapt
    #: choice the head was trained with, read as shipped.
    config_path: Path | None = None
    #: The encoder family's ``checkpoint_index.json``.
    index_path: Path | None = None
    #: A TorchScript encoder built by :mod:`quantem.inference.export`, if any.
    #: When present the pack runs without timm or dinov3.
    export_path: Path | None = None


def resolve_model_files(pack_id: str) -> ModelFiles:
    """Locate the installed, checksum-recorded artifacts for a pack.

    Paths belong to the model registry, not to inference: the cache is
    content-addressed by SHA-256 and is the only thing that knows where a
    verified blob landed.

    Raises:
        ModelWeightsNotInstalled: if the registry cache cannot produce the files.
    """
    from quantem.registry import cache as registry_cache

    try:
        resolved = registry_cache.resolve_pack(pack_id)
    except registry_cache.PackNotInstalled as exc:
        raise ModelWeightsNotInstalled(str(exc)) from exc

    return ModelFiles(
        pack_id=pack_id,
        head_path=Path(resolved.head_path),
        encoder_path=Path(resolved.encoder_path) if resolved.encoder_path else None,
        config_path=Path(resolved.config_path) if resolved.config_path else None,
        index_path=Path(resolved.index_path) if resolved.index_path else None,
        export_path=Path(resolved.export_path) if resolved.export_path else None,
    )


# --- Loaded model -----------------------------------------------------------


def normalize_tile(tile: np.ndarray, mean: float, std: float) -> np.ndarray:
    """uint8 [0,255] EM -> float32 ``(x/255 - mean) / std``.

    No per-tile percentile normalisation: ``mean``/``std`` are fixed constants
    from the model spec, never derived from the tile itself, or adjacent windows
    would be normalised differently and the blend would show seams.

    Callers pass :attr:`~quantem.inference.specs.ModelSpec.input_mean` /
    ``input_std`` -- the scaling of the *tensor handed to the model*, which for
    the OmniEM family is not the encoder's corpus statistics. See
    :class:`~quantem.inference.specs.FamilySpec`.
    """
    x = np.asarray(tile, dtype=np.float32) / 255.0
    return (x - float(mean)) / float(std)


@dataclass
class LoadedModel:
    """A model resident in this process, ready to run tiles."""

    spec: ModelSpec
    device: str
    #: The assembled ``SegModel`` (or a stand-in when ``forward`` is supplied).
    #: Untyped because torch is not imported at module scope.
    module: Any
    files: ModelFiles | None = None
    #: Optional replacement for the torch forward. Lets the whole pipeline run
    #: without torch (tests, and a finetune harness that owns its own loop).
    forward: TileForward | None = None
    #: How the encoder was built: ``"exported"`` | ``"timm"`` | ``"dinov3"``.
    #: Recorded because it is provenance -- an exported artifact is a different
    #: (frozen, digest-covered) object from a graph rebuilt at run time.
    encoder_tier: str = "unknown"
    #: Cache identity, when this is not simply the released pack. An adapted
    #: model has the same ``pack_id`` and *different weights*, so it must not
    #: share a cache slot with the pack it was fitted from -- see
    #: :func:`load_adapted_model`.
    cache_key: str | None = None
    #: The adapter whose head was loaded over the released one, if any. Carried
    #: because it is provenance: two runs of "quantem:mito" are not the same run
    #: when one of them is wearing a user's trained head.
    adapter_id: str | None = None

    @property
    def pack_id(self) -> str:
        return self.spec.pack_id

    @property
    def key(self) -> tuple[str, str]:
        """This model's slot in :data:`_MODEL_CACHE`."""
        return (self.cache_key or self.spec.pack_id, self.device)

    def forward_tile(self, tile: np.ndarray) -> np.ndarray:
        """Foreground probability for one square window.

        Args:
            tile: uint8 grayscale ``[t, t]`` at model scale.

        Returns:
            float32 ``[t, t]`` foreground probability in ``[0, 1]``.
        """
        if self.forward is not None:
            return self.forward(tile)

        torch = _torch()
        x = normalize_tile(tile, self.spec.input_mean, self.spec.input_std)
        xt = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(self.device)

        dtype = autocast_dtype(self.device)
        with torch.no_grad():
            if dtype is not None:
                with torch.autocast(device_type=self.device, dtype=dtype):
                    logits = self.module(xt)
            else:
                logits = self.module(xt)
            probs = torch.softmax(logits[0].float(), dim=0)
            fg = probs[1] if probs.shape[0] == 2 else probs[1:].amax(dim=0)
            return fg.cpu().numpy().astype(np.float32)


def build_module(
    files: ModelFiles,
    spec: ModelSpec,
    device: str,
    *,
    allow_eager_encoder: bool = True,
) -> tuple[object, str]:
    """Assemble the segmentation model for a pack on ``device``.

    The released checkpoints are bare ``state_dict``s, so this rebuilds the
    architecture they were trained as -- neck, decoder and adapter wiring from
    :mod:`quantem.inference._fig3`, encoder from
    :mod:`quantem.inference.encoders` -- and loads the head into it. The head's
    own ``resolved_config.yaml`` decides the graph; nothing about the shape of
    the model is guessed here.

    Returns ``(module, encoder_tier)``.

    Raises:
        ModelArchitectureUnavailable: when the encoder cannot be built (no
            exported artifact and the eager tier's package is missing) or the
            head does not load cleanly into the rebuilt graph.
    """
    from ._fig3.load_head import HeadLoadError, build_and_load_head
    from ._fig3.schema import load_head_config
    from .encoders import EncoderManifest, EncoderUnavailable, build_encoder

    if files.config_path is None or not files.config_path.exists():
        raise ModelArchitectureUnavailable(
            f"{spec.pack_id}: no resolved_config.yaml beside the weights. The head is a "
            "bare state_dict; without its config there is nothing that says which neck "
            "and decoder to rebuild. Reinstall the pack."
        )
    cfg = load_head_config(files.config_path)

    manifest = None
    if files.index_path is not None and files.index_path.exists():
        manifest = EncoderManifest.from_index(files.index_path)

    # A pack adapted with `full` embeds its whole fine-tuned encoder in the head,
    # so it can build the module shape from those tensors if the shared blob is
    # not installed. The head overwrites the weights either way.
    skeleton_state = None
    if spec.embeds_encoder and files.encoder_path is None:
        torch = _torch()
        skeleton_state = torch.load(
            str(files.head_path), map_location="cpu", weights_only=False
        ).get("encoder_trainable")

    try:
        bundle = build_encoder(
            manifest=manifest,
            encoder_path=files.encoder_path,
            export_path=files.export_path,
            apply_encoder_norm=cfg.encoder.apply_encoder_norm,
            device=device,
            skeleton_state=skeleton_state,
            allow_eager=allow_eager_encoder,
        )
    except EncoderUnavailable as exc:
        raise ModelArchitectureUnavailable(f"{spec.pack_id}: {exc}") from exc

    contract = bundle.contract
    _check_contract(spec, contract)

    if contract.tier != "exported":
        # The slow path, and it must never be silent: without encoder_ts.pt the
        # encoder is rebuilt from the raw weights on every cold start -- ~4.5
        # minutes instead of ~30 seconds for a whole-image run on CPU -- and
        # nothing on screen said why. See _repair_export below for the rewrite
        # that stops the next start paying it again.
        expected = files.head_path.parent / _exported_encoder_name()
        logger.warning(
            "%s: no exported TorchScript encoder at %s; falling back to the "
            "slow eager path (rebuilding the encoder from raw weights, tier "
            "'%s'). Model load takes minutes instead of seconds this way.",
            spec.pack_id,
            expected,
            contract.tier,
        )

    try:
        model, info = build_and_load_head(cfg, bundle.module, files.head_path, device=device)
    except HeadLoadError as exc:
        raise ModelArchitectureUnavailable(f"{spec.pack_id}: {exc}") from exc

    logger.info(
        "Built %s: %s neck + %s decoder, adapt=%s, taps=%s, encoder tier=%s",
        spec.pack_id, info["neck"], info["decoder"], info["adapt"],
        info["layers"], contract.tier,
    )

    if contract.tier != "exported":
        # Repair the export so only THIS start pays the eager build. The
        # encoder in hand already carries the pack's encoder-side tensors
        # (build_and_load_head just applied them), which is exactly the state
        # the build-time export traces. Best-effort by contract: a failed
        # rewrite is logged and the run continues on the eager module.
        _repair_export(files, spec, cfg, bundle, model, device)

    return model, contract.tier


def _exported_encoder_name() -> str:
    from .encoders import EXPORTED_ENCODER_NAME

    return EXPORTED_ENCODER_NAME


def _repair_export(
    files: ModelFiles,
    spec: ModelSpec,
    cfg: Any,
    bundle: Any,
    model: Any,
    device: str,
) -> None:
    """Best-effort rewrite of a missing ``encoder_ts.pt`` beside the pack.

    Called only when the eager fallback just ran. Reuses the encoder that was
    just built (head tensors applied), so the added cost is the trace and its
    on-disk verification, not a second multi-minute build. The write is atomic
    (tmp then rename, inside :func:`quantem.inference.export.export_built_encoder`)
    and the registry resolves the export by existence, so the next
    ``load_model`` takes the fast tier with no further bookkeeping.

    **Never raises.** A pack directory that cannot be written (read-only
    install, disk full) or a trace that fails verification leaves the run on
    the eager module it already has; the failure is logged with the reason.
    """
    target = files.head_path.parent / _exported_encoder_name()
    if target.exists():
        return
    try:
        from .export import export_built_encoder

        result = export_built_encoder(
            spec.pack_id,
            bundle.module,
            list(model.layers),
            adapt=cfg.encoder.adapt,
            apply_encoder_norm=bool(cfg.encoder.apply_encoder_norm),
            source_tier=bundle.contract.tier,
            depth=int(bundle.contract.depth),
            embedding_dim=int(bundle.contract.embedding_dim),
            patch_size=int(bundle.contract.patch_size),
            input_mean=float(bundle.contract.input_mean),
            input_std=float(bundle.contract.input_std),
            output=target,
            device=device,
        )
        logger.warning(
            "%s: rewrote the missing TorchScript encoder to %s "
            "(max|diff| vs eager: %.3e). The next model load takes the fast "
            "exported path again.",
            spec.pack_id,
            result.path,
            result.max_abs_diff,
        )
    except Exception:
        logger.warning(
            "%s: could not rewrite the missing TorchScript encoder beside the "
            "pack; every cold start keeps paying the slow eager build until "
            "the pack is reinstalled or exported (python -m "
            "quantem.inference.export %s).",
            spec.pack_id,
            spec.pack_id,
            exc_info=True,
        )


def _check_contract(spec: ModelSpec, contract: EncoderContract) -> None:
    """Fail loudly when the built encoder disagrees with the pack's spec.

    Every field here is one that produces a *plausible but wrong* segmentation
    when it drifts rather than an exception: the wrong input scaling shifts the
    encoder off its training distribution (and mis-feeds the ER neck's raw-image
    branch), and the wrong patch size silently changes the tap grid resolution.
    """
    problems = []
    exported_for = getattr(contract, "pack_id", "") or ""
    if exported_for and exported_for != spec.pack_id:
        problems.append(
            f"the exported encoder was built for {exported_for!r}, not {spec.pack_id!r}"
        )
    if abs(contract.input_mean - spec.input_mean) > 1e-9 or abs(
        contract.input_std - spec.input_std
    ) > 1e-9:
        problems.append(
            f"input normalisation {contract.input_mean}/{contract.input_std} != "
            f"spec {spec.input_mean}/{spec.input_std}"
        )
    if contract.patch_size != spec.patch_size:
        problems.append(f"patch size {contract.patch_size} != spec {spec.patch_size}")
    if spec.tile_size % contract.patch_size != 0:
        problems.append(
            f"tile {spec.tile_size} is not a multiple of patch {contract.patch_size}"
        )
    if problems:
        raise ModelArchitectureUnavailable(
            f"{spec.pack_id}: the built encoder does not match the pack spec "
            f"({'; '.join(problems)}). Refusing to run a model whose output would look "
            "correct and not be."
        )


# --- Module-level cache -----------------------------------------------------

_MODEL_CACHE: OrderedDict[tuple[str, str], LoadedModel] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def load_model(pack_id: str, device: str | None = None) -> LoadedModel:
    """Load (or reuse) a model pack on the selected device.

    Repeated calls with the same ``(pack_id, device)`` return the same object;
    the least recently used model is evicted past :data:`MAX_CACHED_MODELS`.

    Raises:
        ValueError: unknown pack id.
        ModelWeightsNotInstalled / ModelArchitectureUnavailable: see above.
    """
    spec = MODEL_SPECS.get(pack_id)
    if spec is None:
        raise ValueError(f"Unknown model pack: {pack_id!r}")
    resolved_device = select_device(device)
    key = (pack_id, resolved_device)

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            _MODEL_CACHE.move_to_end(key)
            return cached

    # Loading is slow and must not hold the lock; a concurrent duplicate load is
    # wasteful but harmless, and the second one is discarded below.
    files = resolve_model_files(pack_id)
    module, tier = build_module(files, spec, resolved_device)
    loaded = LoadedModel(
        spec=spec,
        device=resolved_device,
        module=module,
        files=files,
        encoder_tier=tier,
    )
    logger.info("Loaded model pack %s on %s (encoder tier: %s)", pack_id, resolved_device, tier)

    with _CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            return existing
        _MODEL_CACHE[key] = loaded
        _MODEL_CACHE.move_to_end(key)
        while len(_MODEL_CACHE) > MAX_CACHED_MODELS:
            evicted_key, evicted = _MODEL_CACHE.popitem(last=False)
            logger.info("Evicting cached model %s (%s)", *evicted_key)
            empty_cache(evicted.device)
    return loaded


def adapted_cache_key(pack_id: str, head_path: str | Path) -> str:
    """Cache identity for ``pack_id`` wearing the head at ``head_path``.

    The file's modification time is part of the key: re-running an adaptation
    writes a new head to the *same* path, and a stale module in the cache would
    then serve the previous fit under the new adapter's name.
    """
    path = Path(head_path)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return f"{pack_id}+head:{path}@{stamp}"


def load_adapted_model(
    pack_id: str,
    head_path: str | Path,
    device: str | None = None,
    *,
    adapter_id: str | None = None,
) -> LoadedModel:
    """Load a pack with a user-trained neck + decoder loaded over its own.

    Guided fine-tuning saves only the trained submodules (see
    :func:`quantem.finetune.adapt.save_head`); the encoder is the released one,
    still frozen, still in the registry cache addressed by digest. So this
    builds the pack exactly as :func:`load_model` would and then overwrites the
    head.

    The module is built **fresh** rather than taken from the shared cache, and
    is stored under its own key. Mutating the cached released model in place
    would silently give an adapted head to every later run of that pack,
    including runs on segmentations the user never applied the adapter to.

    ``quantem.finetune`` is imported lazily so that :mod:`quantem.inference`
    keeps working on an install without it.

    Raises:
        ValueError: unknown pack id.
        ModelWeightsNotInstalled / ModelArchitectureUnavailable: as
            :func:`load_model`.
        FileNotFoundError: the adapted head is missing.
        quantem.finetune.adapt.HeadAdaptationUnavailable: the file is not a
            QuantEM adapted head.
    """
    spec = MODEL_SPECS.get(pack_id)
    if spec is None:
        raise ValueError(f"Unknown model pack: {pack_id!r}")
    head_path = Path(head_path)
    if not head_path.is_file():
        raise FileNotFoundError(
            f"Adapted head for {pack_id} is missing at {head_path}; the adapter "
            "cannot be applied."
        )

    resolved_device = select_device(device)
    key = (adapted_cache_key(pack_id, head_path), resolved_device)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            _MODEL_CACHE.move_to_end(key)
            return cached

    from quantem.finetune.adapt import load_head  # noqa: PLC0415 -- optional dependency

    files = resolve_model_files(pack_id)
    module, tier = build_module(files, spec, resolved_device)
    load_head(module, head_path)
    loaded = LoadedModel(
        spec=spec,
        device=resolved_device,
        module=module,
        files=files,
        encoder_tier=tier,
        cache_key=key[0],
        adapter_id=adapter_id,
    )
    logger.info(
        "Loaded model pack %s on %s with adapted head %s (adapter=%s, encoder tier: %s)",
        pack_id, resolved_device, head_path, adapter_id or "?", tier,
    )

    with _CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            return existing
        _MODEL_CACHE[key] = loaded
        _MODEL_CACHE.move_to_end(key)
        while len(_MODEL_CACHE) > MAX_CACHED_MODELS:
            evicted_key, evicted = _MODEL_CACHE.popitem(last=False)
            logger.info("Evicting cached model %s (%s)", *evicted_key)
            empty_cache(evicted.device)
    return loaded


def cache_model(model: LoadedModel) -> None:
    """Insert an already-built model into the cache (used by tests and finetune)."""
    with _CACHE_LOCK:
        _MODEL_CACHE[model.key] = model
        _MODEL_CACHE.move_to_end(model.key)


def cached_model_keys() -> list[tuple[str, str]]:
    with _CACHE_LOCK:
        return list(_MODEL_CACHE)


def clear_model_cache() -> None:
    """Drop every resident model and release allocator caches."""
    with _CACHE_LOCK:
        devices = {model.device for model in _MODEL_CACHE.values()}
        _MODEL_CACHE.clear()
    for device in devices:
        empty_cache(device)


# --- Region prediction ------------------------------------------------------


@dataclass(frozen=True)
class RegionPrediction:
    """Result of running a model over one region.

    ``prob`` is at **model scale** -- the grid the model actually predicted on.
    Threshold it there, then use ``context`` to map the binary mask back to
    native pixels. ``context.is_identity`` is True when no resampling happened.
    """

    prob: np.ndarray
    context: resample.ResampleContext
    plan: tiling.TilePlan

    @property
    def native_shape(self) -> tuple[int, int]:
        return self.context.native_shape


def estimate_tiles(
    spec: ModelSpec,
    native_shape: tuple[int, int],
    pixel_size_nm: float | None = None,
    overlap: float = tiling.DEFAULT_OVERLAP,
) -> int:
    """Window count a region will need, for progress reporting."""
    context = resample.plan_resample(native_shape, pixel_size_nm, spec.canonical_nm)
    return tiling.estimate_tile_count(context.model_shape, spec.tile_size, overlap)


def predict_region(
    model: LoadedModel,
    image: np.ndarray,
    *,
    pixel_size_nm: float | None = None,
    overlap: float = tiling.DEFAULT_OVERLAP,
    on_progress: Callable[[float], None] | None = None,
    forward: TileForward | None = None,
    out: np.ndarray | None = None,
) -> RegionPrediction:
    """Sliding-window inference over a whole region.

    Args:
        model: a loaded pack.
        image: uint8 grayscale ``[H, W]`` at the asset's native resolution.
        pixel_size_nm: the asset's true pixel size. Required whenever the model
            declares a ``canonical_nm``; without it the region is fed at native
            scale and the result is only as good as that coincidence.
        overlap: window overlap fraction (0.25 as published).
        on_progress: called with a fraction in ``[0, 1]``.
        forward: override the tile forward (tests, or a finetune harness).
        out: optional destination for the model-scale map (e.g. a memmap).

    Returns:
        A :class:`RegionPrediction` at model scale.
    """
    spec = model.spec
    if spec.canonical_nm is not None and not pixel_size_nm:
        logger.warning(
            "%s expects %.1f nm/px but the asset has no pixel_size_nm; "
            "running at native scale",
            spec.pack_id,
            spec.canonical_nm,
        )

    context = resample.plan_resample(
        image.shape[:2], pixel_size_nm, spec.canonical_nm
    )
    scaled = resample.to_model_scale(image, context)
    padded, _pads = tiling.pad_for_tiling(scaled, spec.tile_size, spec.patch_size)
    plan = tiling.plan_tiles(padded.shape[:2], spec.tile_size, overlap)

    predict_tile = forward or model.forward_tile
    padded_prob = tiling.blend_region(
        plan,
        lambda tile: predict_tile(padded[tile.slices]),
        on_progress=on_progress,
        out=out,
    )
    prob = padded_prob[: scaled.shape[0], : scaled.shape[1]]
    return RegionPrediction(prob=np.asarray(prob, dtype=np.float32), context=context, plan=plan)


def predict_region_streaming(
    model: LoadedModel,
    plan: tiling.TilePlan,
    read_tile: Callable[[tiling.Tile], np.ndarray],
    on_band: tiling.BandSink,
    *,
    on_progress: Callable[[float], None] | None = None,
    forward: TileForward | None = None,
) -> None:
    """Bounded-memory variant: the caller supplies tiles and consumes bands.

    Nothing region-sized is allocated, so this is the path for a gigapixel
    asset. ``read_tile`` must return a uint8 ``[t, t]`` window at model scale
    (the caller owns reading and resampling from the source image);
    ``on_band(y0, band)`` receives normalised float32 row-bands in order.
    """
    predict_tile = forward or model.forward_tile
    tiling.blend_region_streaming(
        plan,
        lambda tile: predict_tile(read_tile(tile)),
        on_band,
        on_progress=on_progress,
    )

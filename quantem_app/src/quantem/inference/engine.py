"""Load a model once, run it over a region.

The model is loaded once into this process and kept in :data:`_MODEL_CACHE`,
rather than being re-read from disk on every call. A 4-crop ROI therefore pays
one ViT-L load, not four.

Structure of one call::

    handle = load_model("quantem:mito")          # cached per (pack, device)
    pred   = predict_region(handle, image_uint8, pixel_size_nm=4.2)
    # pred.prob is at MODEL scale and nothing is decided there: bring the field
    # back to native pixels with pred.context, quantise, then threshold that
    # stored map (see quantem.inference.resample for why that order).

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
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from . import resample, tiling
from .device import (
    autocast_dtype,
    device_type,
    empty_cache,
    is_out_of_memory,
    select_device,
    tile_batch_for,
    total_memory_bytes,
)
from .specs import MODEL_SPECS, ModelSpec

if TYPE_CHECKING:  # torch must not be imported at module scope; see _torch()
    from .encoders import EncoderContract

logger = logging.getLogger(__name__)

#: Fewest models kept resident, whatever the machine. Two is enough to flip
#: between two families without thrashing, and it is what a small machine gets:
#: a ViT-L pack is ~1 540 MB of VRAM at batch 1 and ViT-B 588-768 MB, so four
#: resident ViT-L packs would not fit the 4 GB laptop floor of ruling R3.
#:
#: KNOWN TENSION, measured, deliberately left in place. One run per image walks
#: all four organelle packs in a single job, and at two slots that run evicts
#: the first pack before it reaches the fourth: **45.6 s of repeated cold model
#: loads for 4.8 s of warm work**, and a re-run of the same image reloads every
#: one of them. On a machine with the RAM for it :func:`model_cache_slots`
#: returns :data:`MAX_MODEL_CACHE_SLOTS` and the problem does not arise; on a
#: laptop the thrash is the price of fitting in memory at all. Whether a
#: four-organelle run on a small machine should instead order its packs to
#: evict least, or accept the reload, is an open question for measurement -- not
#: something to fix by raising this number, which would trade a slow run for an
#: out-of-memory one.
MAX_CACHED_MODELS = 2

#: Most models kept resident. Four is the whole organelle set, which is the
#: rotation that matters: a library segmented for all four organelles walks the
#: same four packs once per image, and at two slots every one of those runs
#: reloads the weights *and* pays the accelerator's ~30 s graph warm-up again.
MAX_MODEL_CACHE_SLOTS = 4

#: Support override for the resident-model count.
MODEL_CACHE_SLOTS_ENV_VAR = "QUANTEM_MODEL_CACHE_SLOTS"

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
    #: Where that adapted head came from, so a device fallback can rebuild the
    #: same model somewhere else rather than silently reverting to the released
    #: one -- a user's fitted head is not an optional detail.
    adapter_head_path: Path | None = None
    #: Encoder width, from the built encoder's own contract. Only used to size
    #: the tile batch against the measured VRAM table.
    embedding_dim: int = 0
    #: Windows per forward pass. Decided once, at load, by
    #: :func:`quantem.inference.device.tile_batch_for` and then **proved by
    #: running one** -- see :func:`prepare_for_device`.
    tile_batch: int = 1
    #: Plain-language sentences about how this model ended up where it is, set
    #: once at load. Sticky, because they are true of every run this model
    #: serves ("this model cannot use the graphics card").
    load_notices: tuple[str, ...] = ()
    #: Plain-language sentences about *this* run. Cleared by
    #: :func:`predict_region` on entry and collected by it on the way out, so a
    #: cached model does not report last week's fallback on today's run.
    run_notices: list[str] = field(default_factory=list)

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
        return self.forward_tiles([tile])[0]

    def forward_tiles(self, tiles: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Foreground probability for several windows in one forward pass.

        MEASURED: eight windows per pass runs a ViT-B at 57.3 tiles/s against
        26.8 one at a time -- 2.1x on top of the device win. On the CPU the same
        change measured 1.32 -> 1.35 tiles/s, i.e. nothing, because one window
        already saturates every core; :func:`~quantem.inference.device.
        tile_batch_for` therefore returns 1 there and this loops.

        Running out of accelerator memory is handled here rather than left to
        surface as a traceback, because it is recoverable and because the user
        is owed the sentence rather than the exception: the batch is halved and
        retried, and if a single window will not fit the model moves to the
        processor and the run finishes there. Both outcomes are recorded in
        :attr:`run_notices`.
        """
        if self.forward is not None:
            return [self.forward(tile) for tile in tiles]
        if not tiles:
            return []
        limit = max(1, self.tile_batch)
        if len(tiles) > limit:
            # Only reachable after a fallback lowered the ceiling mid-run.
            out: list[np.ndarray] = []
            for start in range(0, len(tiles), limit):
                out.extend(self.forward_tiles(tiles[start:start + limit]))
            return out
        try:
            return self._forward_batch(tiles)
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            return self._recover_from_out_of_memory(list(tiles), exc)

    # --- internals ---

    def _forward_batch(self, tiles: Sequence[np.ndarray]) -> list[np.ndarray]:
        """One forward pass over ``tiles``. No recovery: the caller owns that.

        On CUDA a short batch is **padded up to the full one** and the extra
        outputs discarded. That looks wasteful and is the opposite: MEASURED,
        the first pass through a TorchScript graph at a given input shape costs
        ~20-30 s of cuDNN autotune and profiling-executor specialisation on this
        card, and the specialisation is per shape. Without padding, a run whose
        last batch is short pays that a second time -- 20 s to save the 7 tile
        forwards (~0.2 s) that padding spends. The duplicated windows are
        dropped before anything sees them, so the result is untouched.
        """
        torch = _torch()
        want = len(tiles)
        if (
            device_type(self.device) == "cuda"
            and self.tile_batch > 1
            and 0 < want < self.tile_batch
        ):
            tiles = list(tiles) + [tiles[-1]] * (self.tile_batch - want)
        stacked = np.stack(
            [
                normalize_tile(tile, self.spec.input_mean, self.spec.input_std)
                for tile in tiles
            ]
        )
        xt = torch.from_numpy(np.ascontiguousarray(stacked))[:, None].to(self.device)

        dtype = autocast_dtype(self.device)
        with torch.no_grad():
            if dtype is not None:
                with torch.autocast(device_type=device_type(self.device), dtype=dtype):
                    logits = self.module(xt)
            else:
                logits = self.module(xt)
            probs = torch.softmax(logits.float(), dim=1)
            fg = probs[:, 1] if probs.shape[1] == 2 else probs[:, 1:].amax(dim=1)
            out = fg.cpu().numpy().astype(np.float32)
        return [np.ascontiguousarray(out[i]) for i in range(want)]

    def _recover_from_out_of_memory(
        self, tiles: list[np.ndarray], exc: BaseException
    ) -> list[np.ndarray]:
        """Halve the batch, then move to the processor. Never re-raise an OOM."""
        empty_cache(self.device)
        if len(tiles) > 1:
            half = max(1, len(tiles) // 2)
            self._shrink_batch(half)
            first = self.forward_tiles(tiles[:half])
            return first + self.forward_tiles(tiles[half:])

        self._fall_back_to_processor(cause=exc)
        return self.forward_tiles(tiles)

    def _shrink_batch(self, size: int) -> None:
        if size >= self.tile_batch:
            return
        self.tile_batch = size
        self._note_run(_SMALLER_BATCHES)

    def _fall_back_to_processor(self, *, cause: BaseException | None = None) -> None:
        """Rebuild this model on the CPU and keep going.

        Not ``module.to("cpu")``: a graph traced on the accelerator carries
        device-locked constants and moving it does not move them, which is the
        same defect that makes the artifact names device-tagged. The honest move
        is to build the model the CPU way -- which also picks up the CPU
        artifact rather than the accelerator one.
        """
        if device_type(self.device) == "cpu":
            # Out of *host* memory. There is nowhere further down to go, and
            # pretending otherwise would loop.
            raise MemoryError(
                "There was not enough memory to run the model over this image. "
                "Try a smaller region."
            ) from cause
        was = self.device
        logger.warning(
            "%s ran out of memory on %s; moving this run to the CPU.",
            self.spec.pack_id, was, exc_info=cause is not None,
        )
        if self.files is None:
            raise RuntimeError(
                f"{self.spec.pack_id}: cannot move to the processor without the "
                "pack's files."
            ) from cause
        module, tier = build_module(self.files, self.spec, "cpu")
        if self.adapter_head_path is not None:
            from quantem.finetune.adapt import load_head  # noqa: PLC0415

            load_head(module, self.adapter_head_path)
        old, self.module = self.module, module
        self.device = "cpu"
        self.encoder_tier = tier
        self.tile_batch = 1
        del old
        empty_cache(was)
        self._note_run(
            "This run moved to the processor part-way through: the graphics card "
            "ran out of memory. The result is complete; it took longer than it "
            "would have on the graphics card."
        )

    def _note_run(self, sentence: str) -> None:
        if sentence not in self.run_notices:
            self.run_notices.append(sentence)


def build_module(
    files: ModelFiles,
    spec: ModelSpec,
    device: str,
    *,
    allow_eager_encoder: bool = True,
    allow_exported_encoder: bool = True,
) -> tuple[object, str]:
    """Assemble the segmentation model for a pack on ``device``.

    The released checkpoints are bare ``state_dict``s, so this rebuilds the
    architecture they were trained as -- neck, decoder and adapter wiring from
    :mod:`quantem.inference._fig3`, encoder from
    :mod:`quantem.inference.encoders` -- and loads the head into it. The head's
    own ``resolved_config.yaml`` decides the graph; nothing about the shape of
    the model is guessed here.

    The exported artifact it looks for is the one tagged with ``device``
    (:func:`quantem.inference.encoders.exported_encoder_name`), because a traced
    graph is not portable between devices. When there is no artifact for this
    device the encoder is rebuilt eagerly and :func:`_repair_export` writes the
    tagged one, so only the first run on a given device pays that.

    Args:
        allow_eager_encoder: False to require an exported artifact.
        allow_exported_encoder: False to ignore any artifact and rebuild. Set by
            :func:`load_model` when an artifact loaded but would not run here.

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

    export_path = _export_path_for(files, device) if allow_exported_encoder else None
    try:
        bundle = build_encoder(
            manifest=manifest,
            encoder_path=files.encoder_path,
            export_path=export_path,
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
        # The slow path, and it must never be silent: without the exported
        # encoder the encoder is rebuilt from the raw weights on every cold
        # start -- ~4.5 minutes instead of ~30 seconds for a whole-image run on
        # CPU -- and nothing on screen said why. See _repair_export below for
        # the rewrite that stops the next start paying it again.
        expected = files.head_path.parent / _exported_encoder_name(device)
        logger.warning(
            "%s: no exported TorchScript encoder for %s at %s; falling back to "
            "the slow eager path (rebuilding the encoder from raw weights, tier "
            "'%s'). Model load takes minutes instead of seconds this way.",
            spec.pack_id,
            device,
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


def _exported_encoder_name(device: str = "cpu") -> str:
    from .encoders import exported_encoder_name

    return exported_encoder_name(device)


def _export_path_for(files: ModelFiles, device: str) -> Path | None:
    """The exported artifact to try on ``device``, best first.

    On the CPU that is the artifact the registry already resolved -- the name
    every bundle ships. On an accelerator, a device-tagged artifact if one has
    been written, **otherwise the shipped one anyway**.

    Trying the shipped one is the point. Portability is a property of the
    encoder, not a rule: MEASURED on a Quadro RTX 8000, the OmniEM ViT-L's
    CPU-traced artifact runs on CUDA unchanged, and refusing it on principle
    would have cost every OmniEM pack a ~60 s eager rebuild and 1.2 GB of extra
    disk to reproduce a file that already worked. The QuantEM ViT-B's does not
    run there, and that is discovered by
    :func:`prepare_for_device` running it -- one forward pass, at load, before
    any of the user's tiles.
    """
    if device_type(device) != "cpu":
        tagged = files.head_path.parent / _exported_encoder_name(device)
        if tagged.exists():
            return tagged
    return files.export_path


#: ``(pack_id, device)`` pairs whose export this process has already tried and
#: failed. Not persisted, and deliberately not: an install, a torch upgrade or a
#: different card can all change the answer, and a stale file on disk claiming
#: otherwise would be worse than repeating the work once per process.
#:
#: It exists because a refusal is not free. A QuantEM ViT-B traced on CUDA does
#: not survive its own verification (see export._VERIFY_PASSES), and paying the
#: trace to be told so again on every cold start MEASURED 51 s against 25 s for
#: the eager build alone -- for a file that will never be written.
_EXPORT_REFUSED: set[tuple[str, str]] = set()


def _repair_export(
    files: ModelFiles,
    spec: ModelSpec,
    cfg: Any,
    bundle: Any,
    model: Any,
    device: str,
) -> None:
    """Best-effort rewrite of the missing exported encoder beside the pack.

    Called only when the eager fallback just ran. Reuses the encoder that was
    just built (head tensors applied), so the added cost is the trace and its
    on-disk verification, not a second multi-minute build. The write is atomic
    (tmp then rename, inside :func:`quantem.inference.export.export_built_encoder`)
    and the registry resolves the export by existence, so the next
    ``load_model`` takes the fast tier with no further bookkeeping.

    **It writes the name that belongs to ``device``, and only that name.** This
    function used to write the plain CPU name whatever device the run was using,
    which was observed putting a 341 MB CUDA-traced graph into a shared pack
    directory: the next CPU run of that pack then failed with the mirror
    device error, permanently, silently, with nothing on screen to explain it.
    A device-tagged target makes that unreachable rather than unlikely -- there
    is no longer a filename a CUDA trace and a CPU trace can both claim.

    **Never raises**, and never replaces an existing artifact (the exporter
    refuses to as well, so a race between two workers loses the write rather
    than the file). A pack directory that cannot be written (read-only install,
    disk full) or a trace that fails verification leaves the run on the eager
    module it already has; the failure is logged with the reason -- once per
    process per device, see :data:`_EXPORT_REFUSED`.
    """
    target = files.head_path.parent / _exported_encoder_name(device)
    if target.exists():
        return
    memo = (spec.pack_id, device_type(device))
    if memo in _EXPORT_REFUSED:
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
            "%s: wrote the missing TorchScript encoder for %s to %s "
            "(max|diff| vs eager: %.3e). The next model load on this device "
            "takes the fast exported path again.",
            spec.pack_id,
            device,
            result.path,
            result.max_abs_diff,
        )
    except Exception:
        _EXPORT_REFUSED.add(memo)
        logger.warning(
            "%s: could not write the TorchScript encoder for %s beside the "
            "pack; every cold start on this device keeps paying the eager "
            "build. This process will not try again (python -m "
            "quantem.inference.export %s to see the reason on its own).",
            spec.pack_id,
            device,
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


# --- Getting a built model onto a device it can actually run on -------------


class AcceleratorUnusable(RuntimeError):
    """A model built for an accelerator will not execute on it.

    Internal: :func:`load_model` catches this and rebuilds on the CPU. It never
    reaches a caller, and its text is a log line, not user copy.

    ``out_of_memory`` separates the two reasons a user is owed different words
    for: a card that cannot run this model at all, and a card that could but has
    no room right now.
    """

    def __init__(self, message: str, *, out_of_memory: bool = False) -> None:
        super().__init__(message)
        self.out_of_memory = out_of_memory


#: Forward passes a warm-up runs. **Two, and the second is the expensive one.**
#: TorchScript's profiling executor records tensor profiles on the first
#: execution of a graph and only compiles the specialised, fused version on the
#: second -- so a single warm pass proves the model runs and leaves the ~20-30 s
#: cuDNN autotune to land on the user's first real tile. MEASURED on a Quadro
#: RTX 8000 with the OmniEM ViT-L: one warm pass, then a real window, cost
#: 14 s + 23 s; see the load/predict split in the package report.
_WARMUP_PASSES = 2


def _warm_up(module: Any, spec: ModelSpec, device: str, batch: int) -> None:
    """Run real forward passes of ``batch`` windows. Raises what it hits.

    Two jobs, which is why this exists at load rather than being left to the
    first tile.

    **It proves the model runs here.** A TorchScript encoder traced on another
    device loads without complaint and then dies inside the graph on the first
    forward -- that is how four of the eight shipped packs behaved on CUDA. A
    failure that arrives at load can be recovered from; the same failure 400
    tiles into a whole-image run cannot.

    **And it pays the warm-up here rather than there.** The cost is per graph
    *shape*, which is why the batch used here is the batch the run will use and
    why :meth:`LoadedModel._forward_batch` pads a short final batch back up to
    it.
    """
    torch = _torch()
    tile = int(spec.tile_size)
    probe = np.zeros((max(1, batch), tile, tile), dtype=np.float32)
    xt = torch.from_numpy(probe)[:, None].to(device)
    with torch.no_grad():
        for _ in range(_WARMUP_PASSES):
            module(xt)
    if device_type(device) == "cuda":
        torch.cuda.synchronize()


def prepare_for_device(model: LoadedModel) -> None:
    """Choose the tile batch, prove it runs, and fall back until it does.

    The ladder, in order, each rung measured or observed rather than guessed:

    1. Ask for the batch the free VRAM supports (a lookup over measured working
       sets), and run it. Almost always the end of the story.
    2. Out of memory: halve and retry, down to one window. A card that is busy
       with something else is the common cause and it is transient.
    3. Anything else at batch > 1: the graph refuses batches. Drop to one
       window rather than losing the accelerator entirely -- TorchScript
       specialises per input shape and a graph traced at one window is not
       obliged to accept eight.
    4. Still failing at one window: the accelerator cannot run this model, and
       :func:`load_model` rebuilds it on the CPU with a sentence saying so.

    A no-op off CUDA, where the batch is 1 and there is no warm-up worth paying
    at load time.
    """
    if device_type(model.device) != "cuda" or model.forward is not None:
        model.tile_batch = 1
        return

    wanted = tile_batch_for(model.device, embedding_dim=model.embedding_dim)
    batch = wanted
    while True:
        try:
            _warm_up(model.module, model.spec, model.device, batch)
        except Exception as exc:
            empty_cache(model.device)
            if is_out_of_memory(exc) and batch > 1:
                batch = max(1, batch // 2)
                logger.warning(
                    "%s: not enough graphics memory for %d windows per pass; "
                    "trying %d.", model.pack_id, batch * 2, batch,
                )
                continue
            if batch > 1:
                logger.warning(
                    "%s: this model's graph did not accept %d windows in one "
                    "pass; running one window at a time.",
                    model.pack_id, batch, exc_info=True,
                )
                batch = 1
                continue
            unusable = AcceleratorUnusable(
                f"{model.pack_id} could not run on {model.device}: {exc}",
                out_of_memory=is_out_of_memory(exc),
            )
            raise unusable from exc
        model.tile_batch = batch
        if batch < wanted:
            model.load_notices = (*model.load_notices, _SMALLER_BATCHES)
        logger.info(
            "%s warm on %s at %d window(s) per pass (encoder tier: %s)",
            model.pack_id, model.device, batch, model.encoder_tier,
        )
        return


def _organelle_label(spec: ModelSpec) -> str:
    """The organelle's own name, as the rest of the app writes it."""
    from quantem.segmentation.type_definitions import (  # noqa: PLC0415
        BUILTIN_SEGMENTATION_TYPES_BY_INTERNAL_NAME,
    )

    from .specs import ORGANELLES  # noqa: PLC0415

    organelle = ORGANELLES.get(spec.organelle)
    if organelle is None:
        return "this"
    definition = BUILTIN_SEGMENTATION_TYPES_BY_INTERNAL_NAME.get(
        organelle.internal_name
    )
    return definition.long_name.lower() if definition else spec.organelle


#: Said when the graphics card had room for the model but not for the batch.
_SMALLER_BATCHES = (
    "This run used smaller batches than usual because the graphics card was "
    "short of memory. The result is the same; it took a little longer."
)


def _cannot_use_accelerator_notice(
    spec: ModelSpec, device: str, reason: BaseException | None = None
) -> str:
    """One sentence for a model that had to leave the accelerator at load.

    Names the organelle rather than the pack id: a pack id is an internal model
    name and the user never chose one. Says the consequence and stops -- there
    is nothing the reader can do about a model their card cannot execute, so
    asking them to do something would be noise. Running out of memory is a
    different situation and gets different words, because it *is* actionable
    (close the other program using the card) and because "cannot run" would be
    untrue of a card that ran the same model yesterday.
    """
    where = "graphics card" if device_type(device) == "cuda" else "graphics chip"
    if isinstance(reason, AcceleratorUnusable) and reason.out_of_memory:
        # No figure. The card's free-memory reading is not the limit that was
        # actually hit whenever anything caps the process below the card --
        # a container, MIG, another allocator setting -- and invariant I-13
        # says a number we show is the number that happened. Nothing here is
        # sure enough of one to print it.
        return (
            f"This run used the processor: there was not enough memory on the "
            f"{where} for the {_organelle_label(spec)} model. The result is "
            f"complete; it took longer than it would have on the {where}."
        )
    return (
        f"The {_organelle_label(spec)} model cannot run on this {where}, so this "
        "run used the processor instead. The result is complete."
    )


def _encoder_embedding_dim(module: Any) -> int:
    return int(getattr(getattr(module, "encoder", None), "embedding_dim", 0) or 0)


def model_cache_slots(device: str = "cpu") -> int:
    """How many loaded models to keep resident in this process.

    A whole-library run walks the same packs once per image, and every eviction
    costs not only the weights but the accelerator's graph warm-up -- MEASURED
    ~30 s for a ViT-L on CUDA, against ~37 ms for a warm tile. Two slots cannot
    hold a four-organelle rotation, so at two every run in that rotation paid
    both again, on every image. Four holds it.

    Sized from what the machine has, per owner ruling R2: one place decides, and
    it forks only where the lever is large. **Total** VRAM decides on CUDA, not
    free -- free shrinks as the cache fills, which would make the cache evict
    itself. A ViT-L is ~1.2 GB of resident weights, so four of them plus the
    measured working set wants around 7 GB and a 4 GB laptop card must not try.
    Host RAM decides everywhere else, read from the machine profile rather than
    probed again here.
    """
    override = os.environ.get(MODEL_CACHE_SLOTS_ENV_VAR, "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning(
                "Ignoring %s=%r: it is not a whole number of models",
                MODEL_CACHE_SLOTS_ENV_VAR, override,
            )

    if device_type(device) == "cuda":
        total = total_memory_bytes(device)
        if total is None:
            return MAX_CACHED_MODELS
        gib = total / (1024 ** 3)
        if gib >= 12.0:
            return MAX_MODEL_CACHE_SLOTS
        return MAX_CACHED_MODELS if gib >= 6.0 else 1

    try:
        from quantem.core.machine import get_machine_profile  # noqa: PLC0415

        ram = get_machine_profile().total_ram_bytes
    except Exception:
        logger.debug("no machine profile; keeping the default cache size", exc_info=True)
        return MAX_CACHED_MODELS
    if ram is None:
        return MAX_CACHED_MODELS
    return MAX_MODEL_CACHE_SLOTS if ram >= 32 * (1024 ** 3) else MAX_CACHED_MODELS


# --- Module-level cache -----------------------------------------------------

_MODEL_CACHE: OrderedDict[tuple[str, str], LoadedModel] = OrderedDict()
_CACHE_LOCK = threading.Lock()

#: Whether this process has already told torch how wide to run. One decision per
#: process, taken the first time a model is loaded.
_THREADS_CONFIGURED = False


def configure_torch_threads() -> int | None:
    """Give torch the intra-op width the machine profile chose. Once.

    The profile is detected in one place (:mod:`quantem.core.machine`, owner
    ruling **R2**: detect capability once, express it as one profile, do not
    scatter ``os.cpu_count()`` calls). It already pins BLAS/OpenMP through the
    environment before numpy is imported; torch's own intra-op pool is *not*
    covered by that pin and defaults to every core it can see, which on a
    workstation means the sequential organelles of one run fight each other for
    cores and, on the 8 GB laptop of **R3**, means one thread arena per core.

    Never raises and never overrules a user: an explicit ``TORCH_NUM_THREADS``
    or a value the caller already set is left alone. Returns the width applied,
    or None when nothing was changed.
    """
    global _THREADS_CONFIGURED
    if _THREADS_CONFIGURED:
        return None
    _THREADS_CONFIGURED = True
    try:
        from quantem.core.machine import get_machine_profile  # noqa: PLC0415

        wanted = int(get_machine_profile().torch_threads)
        if wanted <= 0:
            return None
        torch = _torch()
        if torch.get_num_threads() == wanted:
            return None
        torch.set_num_threads(wanted)
    except Exception:
        logger.debug("could not set the torch thread count", exc_info=True)
        return None
    logger.info("torch intra-op threads set to %d", wanted)
    return wanted


def _evict_to_fit(device: str) -> None:
    """Trim the cache to :func:`model_cache_slots`. Caller holds the lock."""
    slots = max(1, model_cache_slots(device))
    while len(_MODEL_CACHE) > slots:
        evicted_key, evicted = _MODEL_CACHE.popitem(last=False)
        logger.info("Evicting cached model %s (%s)", *evicted_key)
        empty_cache(evicted.device)


def load_model(pack_id: str, device: str | None = None) -> LoadedModel:
    """Load (or reuse) a model pack on the selected device.

    Repeated calls with the same ``(pack_id, device)`` return the same object;
    the least recently used model is evicted past :func:`model_cache_slots`.

    On an accelerator the model is **run once before this returns**, which both
    proves it executes there and pays the graph warm-up at load rather than
    inside the first tile. A model that will not run on the accelerator is
    rebuilt on the CPU and says so in
    :attr:`LoadedModel.load_notices`; it is not an error, because the answer to
    "this model cannot use your graphics card" is a slower run, not a failed one.

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

    configure_torch_threads()
    # Loading is slow and must not hold the lock; a concurrent duplicate load is
    # wasteful but harmless, and the second one is discarded below.
    files = resolve_model_files(pack_id)
    loaded = _build_and_prepare(spec, files, resolved_device)
    logger.info(
        "Loaded model pack %s on %s (encoder tier: %s)",
        pack_id, loaded.device, loaded.encoder_tier,
    )

    key = (pack_id, loaded.device)
    with _CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            return existing
        _MODEL_CACHE[key] = loaded
        _MODEL_CACHE.move_to_end(key)
        _evict_to_fit(loaded.device)
    return loaded


def _build_and_prepare(
    spec: ModelSpec,
    files: ModelFiles,
    device: str,
    *,
    adapter_head: Path | None = None,
    adapter_id: str | None = None,
    cache_key: str | None = None,
) -> LoadedModel:
    """Build a pack on ``device``, then get it into a state that runs.

    Three attempts, in the order that costs least. Each is a build followed by
    one real forward pass, because loading a traced graph proves nothing about
    whether it will execute.

    1. **The shipped artifact on the asked-for device.** Almost always the end
       of it -- and on CUDA it is also where the OmniEM family stays, because
       its CPU-traced artifact runs there unchanged.
    2. **Rebuild without the artifact, same device.** For an encoder whose trace
       is device-locked this is what works: an eager encoder is built *on* the
       device rather than replayed onto it, and :func:`_repair_export` then
       writes the device-tagged artifact so the next run is fast again.
    3. **The CPU.** Reached when the accelerator can run neither, which is the
       real situation for a release-bundle QuantEM pack on a machine without
       Meta's DINOv3 package: the artifact will not execute on the card and
       there is nothing installed to rebuild it from. The run is not failed
       over it -- it moves to the processor and the model carries the sentence
       that says so, once, in the app's own words. A slower run beats no run.
    """
    on_accelerator = device_type(device) != "cpu"
    attempts: list[tuple[str, dict]] = [(device, {})]
    if on_accelerator:
        attempts += [(device, {"allow_exported_encoder": False}), ("cpu", {})]

    last: Exception | None = None
    tried_eager_here = False
    for where, options in attempts:
        without_artifact = options.get("allow_exported_encoder") is False
        if without_artifact and tried_eager_here:
            continue  # the failing attempt was already the eager one
        try:
            model = _build_on(
                spec, files, where, adapter_head, adapter_id, cache_key, **options
            )
        except Exception as exc:
            # An out-of-memory here is the weights not fitting on the card at
            # all, before a single window has run. It belongs on this ladder for
            # the same reason the warm-up's does -- there is a device below that
            # will work -- and it must not reach the user as a CUDA traceback.
            if not isinstance(exc, ModelUnavailableError) and not is_out_of_memory(exc):
                raise
            last = _as_unusable(exc, spec, where)
            tried_eager_here = tried_eager_here or without_artifact
            logger.warning("%s: could not build on %s: %s", spec.pack_id, where, exc)
            empty_cache(where)
            continue
        tier = model.encoder_tier
        try:
            prepare_for_device(model)
        except AcceleratorUnusable as exc:
            last = exc
            tried_eager_here = tried_eager_here or tier != "exported"
            logger.warning("%s", exc)
            del model
            empty_cache(where)
            continue
        if on_accelerator and device_type(where) == "cpu":
            model.load_notices = (
                _cannot_use_accelerator_notice(spec, device, last),
            )
        return model

    # Every route failed, including the CPU one. Nothing here can invent a
    # model; the last reason is the most specific one there is.
    if getattr(last, "out_of_memory", False):
        raise ModelUnavailableError(
            f"There was not enough memory to load the {_organelle_label(spec)} "
            "model on this computer. Close other programs and try again, or run "
            "this on a smaller region of the image."
        ) from last
    if last is not None:
        raise last
    raise ModelArchitectureUnavailable(
        f"{spec.pack_id}: no way to build this model on this machine."
    )


def _as_unusable(
    exc: BaseException, spec: ModelSpec, device: str
) -> Exception:
    """Normalise a build failure into something the ladder can reason about."""
    if isinstance(exc, ModelUnavailableError):
        return exc
    unusable = AcceleratorUnusable(
        f"{spec.pack_id} could not be built on {device}: {exc}", out_of_memory=True
    )
    return unusable


def _build_on(
    spec: ModelSpec,
    files: ModelFiles,
    device: str,
    adapter_head: Path | None,
    adapter_id: str | None,
    cache_key: str | None,
    **options,
) -> LoadedModel:
    module, tier = build_module(files, spec, device, **options)
    if adapter_head is not None:
        from quantem.finetune.adapt import load_head  # noqa: PLC0415

        load_head(module, adapter_head)
    return LoadedModel(
        spec=spec,
        device=device,
        module=module,
        files=files,
        encoder_tier=tier,
        cache_key=cache_key,
        adapter_id=adapter_id,
        adapter_head_path=adapter_head,
        embedding_dim=_encoder_embedding_dim(module),
    )


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

    files = resolve_model_files(pack_id)
    loaded = _build_and_prepare(
        spec,
        files,
        resolved_device,
        adapter_head=head_path,
        adapter_id=adapter_id,
        cache_key=key[0],
    )
    logger.info(
        "Loaded model pack %s on %s with adapted head %s (adapter=%s, encoder tier: %s)",
        pack_id, loaded.device, head_path, adapter_id or "?", loaded.encoder_tier,
    )

    key = (key[0], loaded.device)
    with _CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            return existing
        _MODEL_CACHE[key] = loaded
        _MODEL_CACHE.move_to_end(key)
        _evict_to_fit(loaded.device)
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
    Nothing is decided there: pass it with ``context`` to
    :func:`quantem.inference.resample.probability_to_native_uint8`, which brings
    the field to the image's own pixels and quantises it, and threshold that.
    ``context.is_identity`` is True when no resampling happened.
    """

    prob: np.ndarray
    context: resample.ResampleContext
    plan: tiling.TilePlan
    #: Plain-language sentences about anything that changed how this run was
    #: computed -- so far, running short of graphics memory. Empty on the
    #: ordinary path. The device the run *finished* on is
    #: ``model.device``, which a fallback will have changed.
    notices: tuple[str, ...] = ()

    @property
    def native_shape(self) -> tuple[int, int]:
        return self.context.native_shape


def estimate_tiles(
    spec: ModelSpec,
    native_shape: tuple[int, int],
    pixel_size_nm: float | None = None,
    overlap: float = tiling.DEFAULT_OVERLAP,
) -> int:
    """Window count a region will need, for progress reporting.

    Exact, not an estimate despite the name: it runs the same two steps
    :func:`predict_region` does before the plan exists -- resample to model
    scale, pad to a whole number of patches -- so the number quoted to a user
    before the run is the number the loop counts to. It used to skip the
    padding step, which is a whole extra row or column of windows whenever the
    model shape happens to land just past a stride boundary.
    """
    context = resample.plan_resample(native_shape, pixel_size_nm, spec.canonical_nm)
    return tiling.count_tiles_for_region(
        context.model_shape, spec.tile_size, spec.patch_size, overlap
    )


def predict_region(
    model: LoadedModel,
    image: np.ndarray,
    *,
    pixel_size_nm: float | None = None,
    overlap: float = tiling.DEFAULT_OVERLAP,
    on_progress: Callable[[float], None] | None = None,
    on_tile: tiling.TileCounter | None = None,
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
        on_tile: called ``(done, total)`` after each window, in whole tiles.
            ``total`` is the plan's, so it is authoritative even if the caller
            pre-announced a different number.
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

    model.run_notices.clear()
    padded_prob = tiling.blend_region_batched(
        plan,
        _batch_predictor(model, padded, forward),
        # The ceiling for the run. An out-of-memory fallback lowers
        # ``model.tile_batch`` mid-run and ``forward_tiles`` then splits each
        # batch down to it, so the loop does not have to know.
        batch=1 if forward is not None else max(1, model.tile_batch),
        on_progress=on_progress,
        on_tile=on_tile,
        out=out,
    )
    prob = padded_prob[: scaled.shape[0], : scaled.shape[1]]
    return RegionPrediction(
        prob=np.asarray(prob, dtype=np.float32),
        context=context,
        plan=plan,
        notices=tuple(model.run_notices),
    )


def _batch_predictor(
    model: LoadedModel,
    padded: np.ndarray,
    forward: TileForward | None,
) -> Callable[[list[tiling.Tile]], list[np.ndarray]]:
    """Windows out of ``padded``, through the model, in one call each batch."""
    if forward is not None:
        return lambda tiles: [forward(padded[tile.slices]) for tile in tiles]
    return lambda tiles: model.forward_tiles([padded[tile.slices] for tile in tiles])


def predict_region_streaming(
    model: LoadedModel,
    plan: tiling.TilePlan,
    read_tile: Callable[[tiling.Tile], np.ndarray],
    on_band: tiling.BandSink,
    *,
    on_progress: Callable[[float], None] | None = None,
    on_tile: tiling.TileCounter | None = None,
    forward: TileForward | None = None,
) -> None:
    """Bounded-memory variant: the caller supplies tiles and consumes bands.

    Nothing region-sized is allocated, so this is the path for a gigapixel
    asset. ``read_tile`` must return a uint8 ``[t, t]`` window at model scale
    (the caller owns reading and resampling from the source image);
    ``on_band(y0, band)`` receives normalised float32 row-bands in order.

    Batching applies here too, and costs ``batch x tile x tile`` bytes of
    windows read ahead -- 2 MB at eight 512-px windows, against the hundreds of
    megabytes this function exists to avoid.
    """
    def predict_tiles(tiles: list[tiling.Tile]) -> list[np.ndarray]:
        windows = [read_tile(tile) for tile in tiles]
        if forward is not None:
            return [forward(window) for window in windows]
        return model.forward_tiles(windows)

    batch = 1 if forward is not None else max(1, model.tile_batch)
    model.run_notices.clear()
    tiling.blend_region_streaming_batched(
        plan,
        predict_tiles,
        on_band,
        batch=batch,
        on_progress=on_progress,
        on_tile=on_tile,
    )

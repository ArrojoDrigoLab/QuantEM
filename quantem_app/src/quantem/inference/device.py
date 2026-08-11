"""Compute-device selection: ``cuda | mps | cpu``.

Offering only ``auto | cuda | cpu`` would leave every Apple Silicon machine
silently running a ViT-L on the CPU -- minutes per tile instead of sub-second.
QuantEM is a desktop app and Macs are a first-class target, so MPS is selected
here whenever torch reports it is built and available.

Torch is imported lazily inside each function. Importing it at module scope
would add seconds to Django startup on a machine that may never run inference.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Preference order when the caller asks for "auto".
#: What ``auto`` picks, in order. **MPS is deliberately absent.**
#:
#: Apple's Metal backend has no float64 at all, and a real segmentation run on
#: an Apple Silicon machine died with "Cannot convert a MPS Tensor to float64
#: dtype" -- no objects written. Somewhere in the forward path a tensor is
#: float64, which CPU and CUDA accept silently and MPS cannot. Beyond that
#: crash, no run on this backend has ever been checked for numerical agreement
#: with CPU, and this application's output is measurements people publish.
#:
#: So macOS runs on the processor by default. A user who wants to try Metal can
#: ask for it explicitly with ``QUANTEM_DEVICE=mps`` or ``--device mps``, which
#: still resolves through :func:`select_device` below; it is opt-in, not a
#: default anybody gets by owning a Mac. Re-enabling it here needs two things:
#: the float64 found and fixed, and a CPU-versus-MPS parity measurement of the
#: kind the CUDA work produced (probability map, mask IoU, per-object geometry).
AUTO_ORDER: tuple[str, ...] = ("cuda", "cpu")

VALID_DEVICES = frozenset({"auto", "cuda", "mps", "cpu"})

#: Env override, useful for support ("does it work on CPU?") without a UI knob.
DEVICE_ENV_VAR = "QUANTEM_DEVICE"


def _torch():
    import torch  # noqa: PLC0415 -- deliberately lazy

    return torch


def cuda_available() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:  # torch missing or a broken driver
        logger.debug("CUDA availability check failed", exc_info=True)
        return False


def mps_available() -> bool:
    """True on Apple Silicon with a torch built against the MPS backend."""
    try:
        backend = getattr(_torch().backends, "mps", None)
        if backend is None:
            return False
        return bool(backend.is_available() and backend.is_built())
    except Exception:
        logger.debug("MPS availability check failed", exc_info=True)
        return False


def available_devices() -> list[str]:
    """Devices usable right now, best first. Always contains ``"cpu"``."""
    devices = []
    if cuda_available():
        devices.append("cuda")
    if mps_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def select_device(preference: str | None = None) -> str:
    """Resolve a device preference to a concrete ``cuda``/``mps``/``cpu``.

    Args:
        preference: ``"auto"``/None, or an explicit device. An explicit device
            that is not available falls back to the next best one with a
            warning rather than failing the run -- a user who picked CUDA on a
            machine whose driver disappeared still wants their segmentation.

    Returns:
        One of ``"cuda"``, ``"mps"``, ``"cpu"``.
    """
    requested = (preference or os.environ.get(DEVICE_ENV_VAR) or "auto").strip().lower()
    if requested not in VALID_DEVICES:
        logger.warning("Unknown device %r; falling back to auto", requested)
        requested = "auto"

    available = available_devices()
    if requested == "auto":
        for candidate in AUTO_ORDER:
            if candidate in available:
                return candidate
        return "cpu"

    if requested in available:
        return requested

    fallback = available[0]
    logger.warning(
        "Requested device %r is not available; using %r instead", requested, fallback
    )
    return fallback


def torch_device(preference: str | None = None):
    """Return a ``torch.device`` for the resolved preference."""
    return _torch().device(select_device(preference))


def describe_device(device: str | None = None) -> str:
    """Human-readable device label for logs and the UI."""
    name = select_device(device) if device in (None, "auto") else device
    if name == "cuda":
        try:
            return f"CUDA ({_torch().cuda.get_device_name(0)})"
        except Exception:
            return "CUDA"
    if name == "mps":
        return "Apple GPU (MPS)"
    return "CPU"


#: The arithmetic every device runs in. One value, recorded in provenance, so
#: that the day a family gets a different default every object written before
#: it is distinguishable from every object written after.
PRECISION = "fp32"


def autocast_dtype(device: str):
    """Autocast dtype for a device. **Always None: every device runs fp32.**

    This used to return bf16 on CUDA whenever ``torch.cuda.is_bf16_supported()``
    said yes. That call counts *emulation*, so on Turing -- which has no bf16
    tensor cores -- it returns True and the app picked the one dtype the
    hardware cannot do natively. Measured on a Quadro RTX 8000 at 4.56 MP, that
    made the ViT-B **1.78x slower than fp32** and the ViT-L **5.5x slower than
    fp16**, while using more VRAM than either (autocast keeps fp32 masters
    alongside the cast copies). ``is_bf16_supported(including_emulation=False)``
    returns False on that card and is the guard anyone re-enabling autocast must
    use -- but the reason autocast is off is not speed, it is the numbers:

    ===========================  =========  =========  =========
    60.73 MP, 464 objects        GPU fp32   GPU fp16   GPU bf16
    ===========================  =========  =========  =========
    mask IoU vs CPU               0.99913    0.99370    0.98333
    objects matched (CPU 464)     462        459        451
    total segmented area          +0.002 %   0.083 %    0.104 %
    worst single object, area     5.1 %      23.3 %     58.1 %
    ===========================  =========  =========  =========

    Population numbers survive all three. Per-object area, perimeter and
    circularity -- what this app exists to produce -- do not. Owner ruling R5
    says a GPU run and a CPU run may be compared; fp32 is what makes that true,
    and 1.14x against fp16 is not a trade worth a 23 % worst-object error.

    Kept as a function rather than deleted because the call sites read better
    with it and because the day a measured family-specific exception arrives
    (GPU_DESIGN P8: ViT-L fp16 parity at scale) it lands here, once, and is
    recorded in provenance rather than assumed.
    """
    del device  # every device, deliberately
    return None


def device_type(device: str | None) -> str:
    """``"cuda:1"`` -> ``"cuda"``. The kind, not the index."""
    name = str(device or "cpu").strip().lower()
    return name.split(":", 1)[0] or "cpu"


def free_memory_bytes(device: str) -> int | None:
    """Free accelerator memory right now, or None when there is no figure.

    CUDA only. MPS is unified memory -- the free figure there is the machine's
    RAM, which the machine profile already owns, and asking torch for it would
    invite a second, disagreeing answer.
    """
    if device_type(device) != "cuda":
        return None
    try:
        free, _total = _torch().cuda.mem_get_info()
        return int(free)
    except Exception:
        logger.debug("could not read free VRAM for %s", device, exc_info=True)
        return None


def total_memory_bytes(device: str) -> int | None:
    """Total accelerator memory, or None when there is no figure."""
    if device_type(device) != "cuda":
        return None
    try:
        _free, total = _torch().cuda.mem_get_info()
        return int(total)
    except Exception:
        logger.debug("could not read total VRAM for %s", device, exc_info=True)
        return None


def is_out_of_memory(exc: BaseException) -> bool:
    """True for an accelerator out-of-memory failure, on any torch version.

    ``torch.OutOfMemoryError`` is a subclass of ``RuntimeError`` and only
    exists from 2.5; older torch raises a bare ``RuntimeError`` whose message
    starts with "CUDA out of memory". MPS raises its own wording. Checked in
    that order so the typed exception wins where it exists.
    """
    try:
        oom = getattr(_torch(), "OutOfMemoryError", None)
    except Exception:
        oom = None
    if oom is not None and isinstance(exc, oom):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    return "out of memory" in text or "mps backend out of memory" in text


# --- Tiles per forward pass -------------------------------------------------

#: Windows per forward pass the batching fork will consider. Powers of two
#: because that is what was measured; anything between them is interpolation.
TILE_BATCH_CHOICES: tuple[int, ...] = (1, 2, 4, 8)

#: The largest batch chosen **automatically**, as opposed to asked for.
#:
#: MEASURED at fp32, which is what ships. On a real 52.9 MP EM image with the
#: OmniEM ViT-L: 7.5 tiles/s at one window, 7.8 at two, **8.1 at four, 8.1 at
#: eight** -- so four takes the whole gain and eight buys nothing for another
#: 1.1 GB of VRAM (3 504 -> 4 627 MiB). The ViT-B's fp32 curve from
#: ``gpu_measure`` agrees on the shape: 18.1 / 24.5 / 26.0 / 27.0 tiles/s, i.e.
#: four is 96 % of eight's throughput for 59 % of its memory (1 247 vs
#: 2 120 MiB). Memory is the scarce thing on the 4-8 GB laptop cards owner
#: ruling R3 targets, and this is where the curve flattens.
#:
#: ``QUANTEM_TILE_BATCH`` still reaches eight for anyone measuring.
MAX_AUTOMATIC_TILE_BATCH = 4

#: Env override for support ("run it one tile at a time and see").
TILE_BATCH_ENV_VAR = "QUANTEM_TILE_BATCH"

#: Free VRAM left unclaimed. The desktop compositor was holding 900 MB on the
#: measurement machine while nothing was running, and the allocator's own
#: fragmentation is not in the table below.
TILE_BATCH_HEADROOM_MIB = 1536

#: MEASURED peak CUDA working set, MiB, for a whole forward pass at N windows
#: per batch (gpu_measure section 3, Quadro RTX 8000, 512/518 windows).
#:
#: ``base``  is the QuantEM ViT-B, fp32, which is what ships.
#: ``large`` is the OmniEM ViT-L. Its 2/4/8 rows were measured under fp16
#: autocast rather than fp32, and are used here **because they are the larger
#: figure**: at batch 1 the same model measured 2 242 MB under autocast against
#: 1 540 MB in fp32, so the autocast curve is an upper bound on the fp32 one.
#: An over-estimate costs a smaller batch; an under-estimate costs an OOM.
_TILE_BATCH_VRAM_MIB: dict[str, dict[int, int]] = {
    "base": {1: 603, 2: 810, 4: 1247, 8: 2120},
    "large": {1: 2259, 2: 2639, 4: 3318, 8: 4753},
}

#: Embedding width at or above which a model is costed as ``large``.
_LARGE_EMBEDDING_DIM = 1024


def _size_class(embedding_dim: int | None) -> str:
    return "large" if int(embedding_dim or 0) >= _LARGE_EMBEDDING_DIM else "base"


def tile_batch_for(
    device: str,
    *,
    embedding_dim: int | None = None,
    free_bytes: int | None = None,
) -> int:
    """How many windows to push through one forward pass.

    **CUDA only.** On the CPU batching buys nothing at all -- MEASURED
    1.32 -> 1.35 tiles/s -- because a single window already occupies every core,
    so it would spend memory for no time. MPS is unified memory and its budget
    belongs to the machine profile, not to a VRAM query. Both stay at one
    window.

    On CUDA the size is a **lookup, not a formula**: the largest batch, up to
    :data:`MAX_AUTOMATIC_TILE_BATCH`, whose *measured* working set fits in the
    free VRAM with :data:`TILE_BATCH_HEADROOM_MIB` held back. A machine with
    less free memory than even one window needs still gets 1 -- the answer to
    "there is not enough memory" is the out-of-memory ladder in
    :meth:`quantem.inference.engine.LoadedModel.forward_tiles`, not a batch of
    zero.

    What it is worth is modest and is stated where the ceiling is set: at fp32
    it measured 1.08x on the ViT-L and (from ``gpu_measure``'s table) 1.44x on
    the ViT-B. The 2.1x figure that motivated batching was fp16, which owner
    decision D1 does not ship.
    """
    override = os.environ.get(TILE_BATCH_ENV_VAR, "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning(
                "Ignoring %s=%r: it is not a whole number of tiles",
                TILE_BATCH_ENV_VAR,
                override,
            )
    if device_type(device) != "cuda":
        return 1

    budget_mib = free_bytes if free_bytes is not None else free_memory_bytes(device)
    if budget_mib is None:
        return 1
    budget_mib = budget_mib / (1024 * 1024) - TILE_BATCH_HEADROOM_MIB
    table = _TILE_BATCH_VRAM_MIB[_size_class(embedding_dim)]
    chosen = 1
    for batch in TILE_BATCH_CHOICES:
        if batch > MAX_AUTOMATIC_TILE_BATCH:
            break
        if table.get(batch) is not None and table[batch] <= budget_mib:
            chosen = batch
    return chosen


def empty_cache(device: str) -> None:
    """Release cached allocator blocks after a large run. Best-effort."""
    try:
        torch = _torch()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        logger.debug("empty_cache failed for %s", device, exc_info=True)

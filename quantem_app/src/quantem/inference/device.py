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
AUTO_ORDER: tuple[str, ...] = ("cuda", "mps", "cpu")

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


def autocast_dtype(device: str):
    """Preferred autocast dtype for a device, or None when autocast is off.

    CUDA gets bf16/fp16. MPS autocast support is uneven across torch releases
    and a wrong dtype there produces NaNs rather than a speedup, so MPS and CPU
    run in fp32.
    """
    if device != "cuda":
        return None
    torch = _torch()
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        logger.debug("bf16 support check failed", exc_info=True)
    return torch.float16


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

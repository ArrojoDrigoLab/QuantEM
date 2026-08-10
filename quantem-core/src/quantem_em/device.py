"""Device selection, capability reporting, and a calibrated speed estimate.

The fine-tuning and inference widgets both need to tell a user how long something will take
*before* they commit to it. Rather than ship a hardware table, we measure: run a few warm-up steps
and extrapolate. The reference numbers below are only a sanity check.
"""

from __future__ import annotations

import os

#: Measured on the campaign GPU, head-only fine-tuning, 300 steps (gk_gold_seg/results/finetune_cv).
#: Used to sanity-check a live measurement, never as the estimate itself.
REFERENCE_FINETUNE_SEC_PER_STEP = {"quantem": 0.059, "omniem": 0.167}


def resolve(device: str = "auto"):
    """``"auto"`` -> cuda, else mps, else cpu. Anything else is passed through."""
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def describe(device=None) -> dict:
    """Human-facing capability summary for the About panel and the run estimate."""
    import torch

    dev = resolve("auto") if device is None else torch.device(device)
    out = {
        "device": str(dev),
        "torch": torch.__version__,
        "accelerated": dev.type != "cpu",
    }
    if dev.type == "cuda":
        i = dev.index or 0
        p = torch.cuda.get_device_properties(i)
        out.update(
            name=p.name,
            total_memory_gb=round(p.total_memory / 1e9, 1),
            capability="sm_{}{}".format(*torch.cuda.get_device_capability(i)),
            cuda=torch.version.cuda,
        )
        out["summary"] = f"CUDA · {p.name} · {out['total_memory_gb']} GB"
    elif dev.type == "mps":
        out["name"] = "Apple GPU (MPS)"
        out["summary"] = "Apple GPU (MPS)"
    else:
        threads = torch.get_num_threads()
        out.update(name="CPU", threads=threads)
        out["summary"] = f"CPU only · {threads} threads"
    return out


def free_memory_bytes(device=None) -> int | None:
    """Free memory on the device, or system RAM if on CPU. ``None`` when unknown."""
    import torch

    dev = resolve("auto") if device is None else torch.device(device)
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev.index or 0)
        return int(free)
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """'about 40 seconds' / 'about 3 minutes' / 'about 1 hour 20 minutes'."""
    s = max(0.0, float(seconds))
    if s < 90:
        return f"about {int(round(s))} seconds"
    m = s / 60.0
    if m < 90:
        return f"about {int(round(m))} minutes"
    h, rem = divmod(int(round(m)), 60)
    return f"about {h} hour{'s' if h != 1 else ''}" + (f" {rem} minutes" if rem else "")

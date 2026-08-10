"""Machine probing.

`probe_system()` collects everything needed to size a run on an unknown server: GPUs
(via nvidia-smi), CUDA/torch versions, CPU cores, RAM, disks + local NVMe candidates,
SLURM availability, open-file ulimit, and distributed-relevant env vars. The training runner
records its output as `system_info.json` in the run directory.
Everything degrades gracefully when GPUs / psutil are absent (e.g. a CPU-only host).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from typing import Any

# --------------------------------------------------------------------------- #
# Static probe
# --------------------------------------------------------------------------- #
def _run(args: list[str], timeout: int = 20) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True, timeout=timeout).strip()
    except Exception:
        return None

def _nvidia_smi() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False, "gpus": []}
    raw = _run(["nvidia-smi"])
    if raw is None:
        return out
    out["available"] = True
    out["raw"] = raw
    q = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if q:
        for line in q.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                out["gpus"].append(
                    {
                        "index": _int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": _int(parts[2]),
                        "memory_used_mb": _int(parts[3]),
                        "driver_version": parts[4],
                        "compute_cap": parts[5],
                    }
                )
    return out

def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"logical": os.cpu_count()}
    try:
        import psutil

        info["physical"] = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
        info["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception:
        pass
    return info

def _disk_info() -> list[dict[str, Any]]:
    disks: list[dict[str, Any]] = []
    candidates = ["/", "/scratch", "/local", "/mnt", "/tmp", os.getcwd()]
    seen: set[str] = set()
    for c in candidates:
        if not os.path.isdir(c):
            continue
        try:
            total, used, free = shutil.disk_usage(c)
        except Exception:
            continue
        key = f"{total}:{free}"
        if key in seen:
            continue
        seen.add(key)
        disks.append(
            {
                "path": c,
                "total_gb": round(total / 1e9, 1),
                "free_gb": round(free / 1e9, 1),
            }
        )
    return disks

def _nvme_candidates() -> list[str]:
    """Best-effort local-NVMe scratch candidates to stage shards onto."""
    cands = []
    for p in ("/scratch", "/local", "/local_scratch", "/raid", "/nvme", "/mnt/nvme", "/mnt/scratch"):
        if os.path.isdir(p) and os.access(p, os.W_OK):
            cands.append(p)
    return cands

def _slurm_info() -> dict[str, Any]:
    return {
        "sbatch_available": shutil.which("sbatch") is not None,
        "srun_available": shutil.which("srun") is not None,
        "in_slurm_alloc": any(k.startswith("SLURM_") for k in os.environ),
        "slurm_env": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
    }

def _ulimit_nofile() -> int | None:
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return soft
    except Exception:
        return None

def _dist_env() -> dict[str, str]:
    keys = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "OMP_NUM_THREADS",
        "TORCH_DISTRIBUTED_DEBUG",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}

def probe_system() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["torch_device_count"] = torch.cuda.device_count()
            info["torch_bf16_supported"] = getattr(torch.cuda, "is_bf16_supported", lambda: None)()
    except Exception:
        info["torch_version"] = None
    info["nvidia_smi"] = _nvidia_smi()
    info["cpu"] = _cpu_info()
    info["disks"] = _disk_info()
    info["nvme_candidates"] = _nvme_candidates()
    info["slurm"] = _slurm_info()
    info["ulimit_nofile"] = _ulimit_nofile()
    info["distributed_env"] = _dist_env()
    return info

def _int(s: str) -> int | None:
    try:
        return int(float(s))
    except Exception:
        return None

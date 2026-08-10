"""Reproducibility helpers: seeding, git commit capture, environment dumps."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python / numpy / torch RNGs. Returns the seed for logging."""
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed

def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker distinctly but reproducibly."""
    import random

    try:
        import torch

        base = torch.initial_seed() % (2**31)
    except Exception:
        base = 0
    seed = (base + worker_id) % (2**31)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

def get_git_commit(repo_dir: str | os.PathLike | None = None) -> dict[str, Any]:
    """Capture git commit/branch/dirty state for the repo containing this package, plus the DINOv3
    commit pinned in ``third_party/dinov3.pin``."""
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parents[2]
    repo_dir = str(repo_dir)

    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=repo_dir, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    dinov3_pin = None
    pin = Path(repo_dir) / "third_party" / "dinov3.pin"
    if pin.exists():
        for line in pin.read_text(encoding="utf-8").splitlines():
            if line.startswith("DINOV3_COMMIT="):
                dinov3_pin = line.split("=", 1)[1].strip()
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_files": (status.splitlines() if status else []),
        "dinov3_pin": dinov3_pin,
    }

def collect_environment() -> dict[str, Any]:
    """Collect interpreter + key package versions (no heavy imports forced)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }
    pkgs: dict[str, str] = {}
    try:
        from importlib.metadata import version

        for name in (
            "torch",
            "torchvision",
            "torchmetrics",
            "timm",
            "numpy",
            "pillow",
            "webdataset",
            "omegaconf",
            "pyarrow",
            "pandas",
            "einops",
            "tensorboard",
            "nvidia-ml-py",
        ):
            try:
                pkgs[name] = version(name)
            except Exception:
                pass
    except Exception:
        pass
    info["packages"] = pkgs
    # CUDA / device info (best effort, no failure if torch missing).
    try:
        import torch

        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return info

def dump_environment(path: str | os.PathLike) -> dict[str, Any]:
    """Write a human-readable environment.txt and return the collected dict."""
    env = collect_environment()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"python: {env.get('python')}",
        f"platform: {env.get('platform')}",
        f"machine: {env.get('machine')}",
        f"executable: {env.get('executable')}",
        f"torch_cuda_available: {env.get('torch_cuda_available')}",
        f"torch_cuda_version: {env.get('torch_cuda_version')}",
        f"cuda_devices: {env.get('cuda_devices')}",
        "packages:",
    ]
    for k, v in sorted(env.get("packages", {}).items()):
        lines.append(f"  {k}=={v}")
    # Also append a best-effort pip freeze for full reproducibility.
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, text=True
        )
        lines.append("\n# pip freeze\n" + freeze)
    except Exception:
        pass
    p.write_text("\n".join(lines), encoding="utf-8")
    return env

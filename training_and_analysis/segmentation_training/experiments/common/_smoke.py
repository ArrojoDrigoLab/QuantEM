"""Shared CPU-smoke scaffolding for the experiments tests — synthetic derived dataset + mock encoder.

No real weights, no CUDA, no numpy-BLAS/sklearn (runs without a GPU). Mirrors segmentation_training/tests: build a tiny
segmentations-shaped corpus, run the real ``segmentation_training.dataprep.build_dataset`` to a derived root, and write a
randomly-initialised mock checkpoint (a ``dinov3`` ViT-S at patch 16, single channel). Every experiment
smoke test drives its arm through these so the train/eval plumbing is exercised end-to-end on CPU.
"""

from __future__ import annotations

import types
from pathlib import Path


def build_derived(corpus_root, out, organelle: str = "er", target_nm: float = 0.0, scale_mode: str = "canonical"):
    """Run the real dataprep on a synthetic corpus → a derived segmentation_training dataset at ``out``."""
    from segmentation_training.dataprep.build_dataset import run
    run(types.SimpleNamespace(corpus_root=str(corpus_root), out=str(out), organelles=[organelle],
                              splits=None, context_frac=0.5, limit=0, null_scale_policy="drop",
                              target_nm=float(target_nm), scale_mode=scale_mode))
    return Path(out)


def mock_run_dir(path, framework: str = "dinov3", arch: str = "vit_small"):
    from segmentation_training._synthetic import write_mock_checkpoint
    return write_mock_checkpoint(Path(path), framework, arch=arch)


def synthetic_setup(tmp_path, organelle: str = "er", *, target_nm: float = 0.0, tile_size: int = 64):
    """Full CPU-smoke setup. Returns a small namespace carrying the synthetic corpus, the derived
    ``data_root``, the mock-encoder ``run_dir``, the loaded ``encoder``, and the organelle and tile size.

    ``target_nm=0.0`` = per-organelle canonical; pass a value to smoke a scale variant.
    """
    from em_ssl.utils.checkpoint_index import CheckpointIndex
    from segmentation_training._synthetic import build_synthetic_corpus
    from segmentation_training.harness.encoders import FrozenEncoder, select_checkpoints

    corpus = build_synthetic_corpus(Path(tmp_path) / "segmentations")
    data_root = build_derived(corpus["root"], Path(tmp_path) / "derived", organelle=organelle,
                              target_nm=target_nm)
    run_dir = mock_run_dir(Path(tmp_path) / "mock_encoder")
    idx = CheckpointIndex.load(run_dir)
    rec = select_checkpoints(idx, n=1)[-1]
    enc = FrozenEncoder.from_manifest(rec.path, idx.manifest, tile_size=tile_size)
    return types.SimpleNamespace(corpus=corpus, data_root=str(data_root), run_dir=str(run_dir),
                                 encoder=enc, organelle=organelle, tile_size=tile_size)


def smoke_cfg(organelle: str = "er", *, decoder: str = "dpt", neck: str = "resnet34_detail",
              adapt: str = "lora", tile_size: int = 64, task: str | None = None, max_steps: int = 2):
    """A tiny encoder adaptation-shaped SegConfig for CPU smoke (mock-encoder tile sizes; 2 steps)."""
    from segmentation_training.config.schema import SegConfig
    return SegConfig.from_dict({
        "name": f"smoke_{organelle}_{decoder}", "device": "cpu", "amp": False, "num_workers": 0,  # 0 = no worker spawn (Windows-safe)
        "encoder": {"run_dir": ".", "tile_size": tile_size, "feature_layers": "last4",
                    "adapt": adapt, "adapt_params": {"rank": 8, "conv": True}},
        "neck": {"type": neck}, "decoder": {"type": decoder},
        "loss": {"terms": [{"type": "dice_bce"}]},
        "data": {"organelle": organelle, "task": (task or ("instance" if organelle == "mito" else "semantic"))},
        "optim": {"max_steps": max_steps, "warmup_steps": 1, "batch_size": 2, "seed": 0},
        "eval": {"overlap": 0.0, "bootstrap_n": 0},
    })

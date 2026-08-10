"""CPU end-to-end test: encoder -> neck -> decoder -> loss train + sliding-window eval.

Exercises multiple neck/decoder/organelle/loss combinations through the full stack in one process, so
that the modules are verified in composition rather than only in isolation. Mock DINOv3 ``vit_small``
encoder, tile 64, device cpu, max_steps 2 — a smoke test, not a convergence test.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from em_ssl.utils.checkpoint_index import CheckpointIndex
from segmentation_training.config.schema import SegConfig
from segmentation_training.harness.encoders import FrozenEncoder, select_checkpoints


def _derived(root, out, organelles):
    from segmentation_training.dataprep.build_dataset import run
    args = types.SimpleNamespace(
        corpus_root=str(root), out=str(out), organelles=list(organelles), splits=None,
        context_frac=0.5, limit=0, null_scale_policy="drop", target_nm=0.0)
    run(args)
    return out


def _encoder(tmp_path):
    from segmentation_training._synthetic import write_mock_checkpoint
    run_dir = write_mock_checkpoint(tmp_path / "mock_encoder", "dinov3", arch="vit_small")
    idx = CheckpointIndex.load(run_dir)
    rec = select_checkpoints(idx, n=1)[-1]
    return FrozenEncoder.from_manifest(rec.path, idx.manifest, tile_size=64)


def _cfg(organelle, neck, decoder, terms, task):
    return SegConfig.from_dict({
        "name": f"{organelle}_{neck}_{decoder}",
        "device": "cpu", "amp": False, "num_workers": 0,
        "encoder": {"tile_size": 64, "feature_layers": "last4"},
        "neck": {"type": neck},
        "decoder": {"type": decoder},
        "loss": {"terms": [{"type": t, "weight": 1.0} for t in terms]},
        "data": {"organelle": organelle, "num_classes": 2, "task": task, "min_fg_frac_keep": 0.0},
        "optim": {"max_steps": 2, "warmup_steps": 1, "batch_size": 2, "lr": 1e-3, "seed": 0},
        "eval": {"overlap": 0.0, "bootstrap_n": 0},
    })


@pytest.fixture(scope="module")
def derived(tmp_path_factory):
    from segmentation_training._synthetic import build_synthetic_corpus
    root = tmp_path_factory.mktemp("seg")
    build_synthetic_corpus(root)
    out = tmp_path_factory.mktemp("seg_data")
    _derived(root, out, ["er", "mito"])
    return out


@pytest.mark.parametrize("organelle,neck,decoder,terms,task", [
    ("er", "naive_1x1", "upernet", ["dice_bce"], "semantic"),
    ("er", "resnet34_detail", "dpt", ["dice_bce", "cldice"], "semantic"),
    ("mito", "naive_1x1", "nnunet_convnext_unet", ["dice_bce"], "instance"),
])
def test_e2e_train_eval(derived, tmp_path, organelle, neck, decoder, terms, task):
    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.evaluate import evaluate_head
    from segmentation_training.harness.train import train_segmodel

    enc = _encoder(tmp_path)
    cfg = _cfg(organelle, neck, decoder, terms, task)
    group = cfg.data.resolved_group()
    train_recs = load_manifest(derived, group, "train")
    test_recs = load_manifest(derived, group, "test")
    assert train_recs and test_recs

    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")
    out = evaluate_head(model, test_recs, cfg, str(derived), device="cpu",
                        mean=enc.image_mean, std=enc.image_std)
    assert "macro" in out["summary"]
    assert out["summary"]["macro"].get("dice") is not None
    if task == "instance":
        assert out["summary"]["macro"].get("pq") is not None



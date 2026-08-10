"""CPU smoke: imports, metrics (ignore/empty/instance), config, mock-encoder taps, corpus.

All runs without a GPU (numpy + scipy.ndimage + torch CPU; no sklearn/skimage). Encoder taps are
exercised against the mock checkpoint written by ``_synthetic.write_mock_checkpoint``, a randomly
initialised 1-channel DINOv3 ViT-S, so those tests need DINOv3 installed.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_imports():
    import segmentation_training  # noqa: F401
    from segmentation_training import BACKGROUND, FOREGROUND, IGNORE_INDEX
    from segmentation_training.config.schema import SegConfig  # noqa: F401
    from segmentation_training.harness import instance_metrics, metrics  # noqa: F401

    assert (BACKGROUND, FOREGROUND, IGNORE_INDEX) == (0, 1, 255)


def test_per_crop_metrics_ignore_and_perfect():
    from segmentation_training.constants import FOREGROUND, IGNORE_INDEX
    from segmentation_training.harness.metrics import per_crop_metrics

    gt = np.zeros((40, 40), np.uint8)
    gt[10:30, 10:30] = FOREGROUND
    gt[:3, :] = IGNORE_INDEX  # ignore band
    pred = gt == FOREGROUND
    m = per_crop_metrics(pred, gt, organelle="er")
    assert m["dice"] > 0.99 and m["iou"] > 0.99
    assert m["cldice"] is not None  # clDice fires for organelle == 'er'
    assert not m["excluded"]
    # a false positive inside the ignore band is not counted
    pred2 = pred.copy()
    pred2[:3, :] = True
    m2 = per_crop_metrics(pred2, gt, organelle="er")
    assert abs(m2["dice"] - m["dice"]) < 1e-6


def test_both_empty_excluded():
    from segmentation_training.harness.metrics import aggregate, per_crop_metrics

    gt = np.zeros((16, 16), np.uint8)
    m = per_crop_metrics(np.zeros((16, 16), bool), gt)
    assert m["excluded"] is True
    agg = aggregate([m])
    assert agg["n_crops"] == 1 and agg["n_evaluated"] == 0 and agg["n_excluded_both_empty"] == 1


def test_instance_metrics_perfect_and_split():
    from segmentation_training.harness.instance_metrics import instance_metrics, postproc_instances

    gt_inst = np.zeros((48, 48), np.int32)
    gt_inst[8:24, 8:24] = 5  # single instance id 5
    valid = np.ones((48, 48), bool)
    prob = (gt_inst > 0).astype(np.float32)
    m = instance_metrics(prob, gt_inst, valid, threshold=0.5, min_size=4)
    assert m["pq"] > 0.99 and m["n_gt_inst"] == 1 and m["n_pred_inst"] == 1
    # deterministic post-proc splits two separated blobs into two instances
    prob2 = np.zeros((48, 48), np.float32)
    prob2[4:12, 4:12] = 1.0
    prob2[30:40, 30:40] = 1.0
    lab = postproc_instances(prob2, valid, 0.5, 0)
    assert lab.max() == 2


def test_config_roundtrip_and_unknown_keys(tmp_path):
    import yaml

    from segmentation_training.config.schema import SegConfig, load_seg_config

    raw = {
        "name": "E6_mito_naive",
        "data": {"organelle": "mito", "task": "instance", "bogus": 1},
        "neck": {"type": "resnet34_detail", "params": {"norm": "instancenorm"}},
        "decoder": {"type": "panoptic_deeplab"},
        "loss": {"terms": [{"type": "dice_bce", "weight": 1.0}, {"type": "cldice", "weight": 0.5}]},
        "unknown_top": 7,
    }
    p = tmp_path / "arm.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_seg_config(p)
    assert cfg.name == "E6_mito_naive"
    assert cfg.data.organelle == "mito" and cfg.data.resolved_canonical_nm() == 8.0
    assert cfg.data.resolved_group() == "group2_mito"
    assert cfg.neck.type == "resnet34_detail" and cfg.neck.params["norm"] == "instancenorm"
    assert len(cfg.loss.terms) == 2 and cfg.loss.terms[1].type == "cldice"
    assert cfg.eval.instance_min_size == 16  # default preserved
    # unknown keys are dropped (round-trips through to_dict without them)
    d = cfg.to_dict()
    assert "unknown_top" not in d and "bogus" not in d["data"]


def test_mock_mae_encoder_taps(mock_checkpoint):
    import torch

    from em_ssl.utils.checkpoint_index import CheckpointIndex
    from segmentation_training.config.schema import EncoderSpec
    from segmentation_training.harness.encoders import FrozenEncoder, select_checkpoints

    run_dir = mock_checkpoint("dinov3", arch="vit_small")
    idx = CheckpointIndex.load(run_dir)
    rec = select_checkpoints(idx, n=1)[-1]
    enc = FrozenEncoder.from_manifest(rec.path, idx.manifest, tile_size=64)
    layers = EncoderSpec(feature_layers="last4").resolved_layers(idx.manifest.depth)
    assert layers == [8, 9, 10, 11]
    feats = enc.extract(torch.zeros(1, 1, 64, 64), layers)
    assert len(feats) == 4
    for f in feats:
        assert tuple(f.shape) == (1, 384, 4, 4)
    # encoder is frozen
    assert all(not p.requires_grad for p in enc.backbone.parameters())


def test_synthetic_corpus_layout(synthetic_corpus):
    root = synthetic_corpus["root"]
    assert (root / "splits" / "group2_mito.csv").exists()
    assert (root / "splits" / "group2_er.csv").exists()
    assert (root / "crops_metadata.csv").exists()
    # the null-scale crop exists and is flagged unknown in crops_metadata
    meta = (root / "crops_metadata.csv").read_text(encoding="utf-8")
    assert "ds_null_scale" in meta and "unknown" in meta
    # OpenOrganelle dual-orientation raw tiles present
    oo = root / "openOrganelle" / "jrc_synthetic" / "crop0"
    assert (oo / "raw_xy.tif").exists() and (oo / "raw_xz.tif").exists()
    assert (oo / "seg_er.tif").exists() and (oo / "seg_mito.tif").exists()


@pytest.mark.parametrize("fw", ["dinov3"])
def test_mock_dinov3_encoder_optional(mock_checkpoint, fw):
    pytest.importorskip("dinov3")
    import torch

    from em_ssl.utils.checkpoint_index import CheckpointIndex
    from segmentation_training.harness.encoders import FrozenEncoder, select_checkpoints

    run_dir = mock_checkpoint(fw, arch="vit_small")
    idx = CheckpointIndex.load(run_dir)
    rec = select_checkpoints(idx, n=1)[-1]
    enc = FrozenEncoder.from_manifest(rec.path, idx.manifest, tile_size=64)
    feats = enc.extract(torch.zeros(1, 1, 64, 64), [8, 9, 10, 11])
    assert len(feats) == 4 and tuple(feats[0].shape) == (1, 384, 4, 4)

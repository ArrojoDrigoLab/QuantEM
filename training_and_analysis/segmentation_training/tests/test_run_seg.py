"""CPU test: the single-arm run spine (run_seg.run_arm) and aggregation, on a mock encoder."""

from __future__ import annotations

import json
import types
from pathlib import Path


def _build_derived(root, out):
    from segmentation_training.dataprep.build_dataset import run
    run(types.SimpleNamespace(corpus_root=str(root), out=str(out), organelles=["er"], splits=None,
                              context_frac=0.5, limit=0, null_scale_policy="drop", target_nm=0.0))


def test_run_arm_and_aggregate(synthetic_corpus, tmp_path):
    from segmentation_training._synthetic import write_mock_checkpoint
    from segmentation_training.config.schema import SegConfig
    from segmentation_training.harness import aggregate
    from segmentation_training.harness.run_seg import run_arm

    data_root = tmp_path / "seg_data"
    _build_derived(synthetic_corpus["root"], data_root)
    run_dir = write_mock_checkpoint(tmp_path / "mock_encoder", "dinov3", arch="vit_small")

    cfg = SegConfig.from_dict({
        "name": "E6_er_naive_smoke", "device": "cpu", "amp": False, "num_workers": 0,
        "encoder": {"run_dir": str(run_dir), "tile_size": 64},
        "neck": {"type": "naive_1x1"}, "decoder": {"type": "upernet"},
        "loss": {"terms": [{"type": "dice_bce"}]},
        "data": {"organelle": "er", "task": "semantic"},
        "optim": {"max_steps": 2, "warmup_steps": 1, "batch_size": 2},
        "eval": {"overlap": 0.0, "bootstrap_n": 0},
    })
    out = tmp_path / "runs" / "segmentation_training" / cfg.name
    run_arm(cfg, str(data_root), str(out), device="cpu", max_steps=2)

    assert (out / "results.json").exists() and (out / "results.csv").exists()
    assert (out / "run.json").exists() and (out / "resolved_config.yaml").exists()
    assert (out / "head.pt").exists()
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert "test" in res["splits"] and "dice" in res["splits"]["test"]["macro"]

    # Aggregation discovers the arm by its run.json marker
    rows = aggregate.collect(tmp_path / "runs" / "segmentation_training")
    assert rows and any(r["arm"] == cfg.name for r in rows)
    agg_out = tmp_path / "agg"
    aggregate.write_csv(rows, _mk(agg_out) / "all_results.csv")
    aggregate.write_markdown(rows, agg_out / "summary.md")
    assert (agg_out / "all_results.csv").read_text(encoding="utf-8").strip()
    assert "## er" in (agg_out / "summary.md").read_text(encoding="utf-8")


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _loader_cfg(tile):
    from segmentation_training.config.schema import SegConfig
    return SegConfig.from_dict({
        "name": "loader_test", "device": "cpu", "amp": False, "num_workers": 0,
        "encoder": {"run_dir": ".", "tile_size": tile},
        "neck": {"type": "naive_1x1"}, "decoder": {"type": "upernet"},
        "loss": {"terms": [{"type": "dice_bce"}]},
        "data": {"organelle": "er", "task": "semantic", "aug_flip": False, "aug_rot90": False,
                 "aug_elastic": False, "aug_brightness": 0.0, "aug_contrast": 0.0, "aug_gamma": 0.0,
                 "aug_noise_std": 0.0, "min_fg_frac_keep": 0.0},
        "optim": {"seed": 0},
    })


def test_v2_loader_crop_contains_edge_annotation():
    """The segmentation_training loader crops to contain the annotation (records carrying
    ``annotation_bbox_in_tile_xyxy`` take this path; the rest take a random crop), which covers
    annotations at the edge of the region. PairedAug and inst gating are unaffected."""
    import numpy as np

    from segmentation_training.constants import IGNORE_INDEX
    from segmentation_training.harness.dataset import SegTrainDataset

    ds = SegTrainDataset([], data_root=".", cfg=_loader_cfg(64), mean=0.5, std=0.25)
    em = np.zeros((128, 128), np.uint8)
    mask = np.full((128, 128), IGNORE_INDEX, np.uint8)
    mask[112:124, 112:124] = 1  # annotation near the far corner
    for _ in range(20):
        ec, mc, ic = ds._crop_containing(em.copy(), mask.copy(), None, [112, 112, 124, 124])
        assert ec.shape == (64, 64) and mc.shape == (64, 64) and ic is None
        assert int((mc == 1).sum()) == 144  # the whole annotation is inside every crop

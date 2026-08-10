"""Input-scale CPU smoke: config gen · multi-scale fusion · two-scale train+eval · native-composition
report. Fast (light neck, 2 recs)."""

from __future__ import annotations


def _cfg(organelle="er", neck="naive_1x1", decoder="upernet", tile=32, steps=2):
    from segmentation_training.experiments.common._smoke import smoke_cfg
    c = smoke_cfg(organelle, decoder=decoder, neck=neck, adapt="lora", tile_size=tile, max_steps=steps)
    return c


def test_gen_scale_configs(tmp_path):
    from segmentation_training.experiments.scale.run_scale import gen_scale_configs
    droots = gen_scale_configs(tmp_path / "cfgs", scale_root="<path>")
    assert (tmp_path / "cfgs" / "data_roots.json").exists()
    assert "scale_er_4nm" in droots and "scale_mito_16nm" in droots
    assert "scale_er_1nm" not in droots               # ER 1nm dropped (test set 100% upsampled below 4.68nm)
    assert (tmp_path / "cfgs" / "scale_er_native.yaml").exists()
    # every generated config loads back as a valid SegConfig with the encoder adaptation recipe
    from segmentation_training.config.schema import load_seg_config
    cfg = load_seg_config(str(tmp_path / "cfgs" / "scale_er_4nm.yaml"))
    assert cfg.encoder.adapt == "lora" and cfg.neck.type == "resnet34_detail" and cfg.decoder.type == "dpt"
    assert cfg.data.canonical_nm == 4.0


def test_multiscale_eval(er_setup):
    from segmentation_training.experiments.common.base_model import build_mock_base
    from segmentation_training.experiments.scale.multiscale import evaluate_multiscale
    from segmentation_training.harness.dataset import load_manifest

    base = build_mock_base(er_setup.run_dir + "_mb", organelle="er", decoder="upernet", neck="naive_1x1",
                           adapt="lora", tile_size=32)
    recs = load_manifest(er_setup.data_root, "group2_er", "test")[:2]
    out = evaluate_multiscale(base.model, recs, base.cfg, er_setup.data_root, "cpu", base.mean, base.std,
                              scales=(1.0, 1.5), fuse="mean")
    assert out["summary"]["macro"].get("dice") is not None
    assert out["summary"]["multiscale"]["scales"] == [1.0, 1.5]


def test_two_scale_train_eval(er_setup):
    from segmentation_training.experiments.scale import two_scale as ts
    from segmentation_training.harness.dataset import load_manifest

    cfg = _cfg("er", neck="naive_1x1", decoder="upernet", tile=32, steps=2)
    train_recs = load_manifest(er_setup.data_root, "group2_er", "train")
    test_recs = load_manifest(er_setup.data_root, "group2_er", "test")[:2]
    model = ts.train_two_scale(cfg, er_setup.encoder, train_recs, er_setup.data_root, "cpu",
                               fuse="xattn", coarse_factor=2)
    model._fuse = "xattn"
    assert sum(p.numel() for p in model.trainable_parameters()) > 0
    ev = ts.evaluate_two_scale(model, test_recs, cfg, er_setup.data_root, "cpu",
                               er_setup.encoder.image_mean, er_setup.encoder.image_std, coarse_factor=2)
    assert ev["summary"]["macro"].get("dice") is not None
    assert ev["summary"]["two_scale"]["coarse_factor"] == 2


def test_two_scale_concat_fusion(er_setup):
    from segmentation_training.experiments.scale import two_scale as ts
    from segmentation_training.harness.dataset import load_manifest

    cfg = _cfg("er", neck="naive_1x1", decoder="upernet", tile=32, steps=1)
    train_recs = load_manifest(er_setup.data_root, "group2_er", "train")
    model = ts.train_two_scale(cfg, er_setup.encoder, train_recs, er_setup.data_root, "cpu",
                               fuse="concat", coarse_factor=2)
    import torch
    y = model(torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 32, 32))
    assert y.shape[1] == 2  # num_classes


def test_scale_composition():
    from segmentation_training.experiments.scale.scale_report import native_composition
    recs = [{"src_nm_col": 8.0, "resample_factor": [4.0, 4.0]},
            {"src_nm_col": 1.0, "resample_factor": [0.5, 0.5]},
            {"src_nm_col": 2.0, "resample_factor": [1.0, 1.0]}]
    c = native_composition(recs)
    assert c["n"] == 3 and abs(c["frac_upsampled"] - 1 / 3) < 1e-6
    assert abs(c["frac_downsampled"] - 1 / 3) < 1e-6 and abs(c["frac_near_native"] - 1 / 3) < 1e-6



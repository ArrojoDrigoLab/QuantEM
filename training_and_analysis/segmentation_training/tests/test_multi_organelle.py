"""Multi-organelle CPU smoke: config gen · DoDNet head build/forward (both codes) · mixed dataset ·
shared-DoDNet train+eval per organelle · per-organelle-LoRA train. Fast (light neck, tile 32, 2 steps, 2 recs).
"""

from __future__ import annotations


def _cfg(organelle="er", neck="naive_1x1", decoder="dodnet", tile=32, steps=2):
    from segmentation_training.experiments.common._smoke import smoke_cfg
    return smoke_cfg(organelle, decoder=decoder, neck=neck, adapt="lora", tile_size=tile, max_steps=steps)


def test_gen_multi_configs(tmp_path):
    from segmentation_training.experiments.multi_organelle.run_multi_organelle import gen_configs
    roots = gen_configs(tmp_path / "cfgs", organelles=("mito", "er"))
    assert (tmp_path / "cfgs" / "multi_dodnet.yaml").exists()
    assert (tmp_path / "cfgs" / "multi_perorg_mito.yaml").exists()
    assert (tmp_path / "cfgs" / "multi_perorg_er.yaml").exists()
    assert (tmp_path / "cfgs" / "data_roots.json").exists()
    assert "mito" in roots and "er" in roots
    # the dodnet config loads back as a valid SegConfig with the dodnet decoder + K=2 controller.
    from segmentation_training.config.schema import load_seg_config
    cfg = load_seg_config(str(tmp_path / "cfgs" / "multi_dodnet.yaml"))
    assert cfg.decoder.type == "dodnet"
    assert int(cfg.decoder.params.get("n_organelles")) == 2
    assert cfg.encoder.adapt == "lora"


def test_dodnet_head_builds_and_forwards_both_codes():
    """DoDNetHead builds via the registry and forwards for both organelle codes with distinct outputs."""
    import torch

    # dodnet is registered into DECODERS at import of dodnet_head.
    from segmentation_training.experiments.multi_organelle import dodnet_head  # noqa: F401
    from segmentation_training.config.schema import DecoderSpec
    from segmentation_training.models.decoders import DECODERS, build_decoder

    assert "dodnet" in DECODERS
    strides = (4, 8, 16, 32)
    in_ch = 32
    spec = DecoderSpec.from_dict({"type": "dodnet",
                                  "params": {"channels": 32, "n_organelles": 2, "mid_channels": 6,
                                             "n_dynamic": 3, "mechanism": "dynamic"}})
    head = build_decoder(spec, in_ch, strides, 2)
    # pyramid: fine..coarse maps at the four strides, base tile = 32 -> H/4=8 finest.
    pyr = [torch.randn(2, in_ch, 32 // s, 32 // s) for s in strides]
    head.set_organelle_code(0)
    y0 = head(pyr, out_hw=(32, 32))
    head.set_organelle_code(1)
    y1 = head(pyr, out_hw=(32, 32))
    assert y0.shape == (2, 2, 32, 32) and y1.shape == (2, 2, 32, 32)
    assert head.aux_logits == []
    # different organelle codes -> different dynamic filters -> different logits (controller is trained-from
    # random init, so at least not byte-identical).
    assert not torch.allclose(y0, y1)
    # batched per-sample codes also work (mito for sample 0, ER for sample 1).
    import torch as _t
    head.set_organelle_code(_t.tensor([[1.0, 0.0], [0.0, 1.0]]))
    yb = head(pyr, out_hw=(32, 32))
    assert yb.shape == (2, 2, 32, 32)


def test_dodnet_head_film_moe_variant():
    """The FiLM-MoE fallback mechanism also builds + forwards."""
    import torch

    from segmentation_training.experiments.multi_organelle.dodnet_head import DoDNetHead
    strides = (4, 8, 16, 32)
    head = DoDNetHead(32, strides, 2, channels=32, n_organelles=2, mid_channels=6, mechanism="film_moe")
    # batch 2: GroupNorm on the 1x1 coarsest pyramid level needs >1 value per channel in train mode (the
    # same constraint the native _SharedDecoderTrunk has; reported runs use batch>=8).
    pyr = [torch.randn(2, 32, 32 // s, 32 // s) for s in strides]
    head.set_organelle_code(1)
    y = head(pyr, out_hw=(32, 32))
    assert y.shape == (2, 2, 32, 32)


def test_mixed_dataset_yields_both_organelles_with_codes(er_setup, mito_setup):
    """MixedOrganelleDataset yields crops from both organelles, each with the correct one-hot code."""
    from segmentation_training.experiments.multi_organelle.mixed_dataset import (
        build_mixed_dataset, organelle_index)
    from segmentation_training.harness.dataset import load_manifest

    cfg = _cfg("er", tile=32, steps=2)
    er_recs = load_manifest(er_setup.data_root, "group2_er", "train")
    mito_recs = load_manifest(mito_setup.data_root, "group2_mito", "train")
    per_org = {"mito": (mito_recs, mito_setup.data_root), "er": (er_recs, er_setup.data_root)}
    ds = build_mixed_dataset(per_org, cfg, er_setup.encoder.image_mean, er_setup.encoder.image_std,
                             patch_size=er_setup.encoder.patch_size, n_organelles=2)
    assert len(ds) == len(er_recs) + len(mito_recs) > 0
    seen = set()
    for i in range(len(ds)):
        item = ds[i]
        oidx = int(item["org_idx"])
        seen.add(oidx)
        assert item["org_code"].shape == (2,)
        assert int(item["org_code"].argmax()) == oidx
    assert organelle_index("mito") in seen and organelle_index("er") in seen
    # collate produces a uniform-key batch with an inst map (padded for semantic ER crops).
    batch = ds.collate([ds[0], ds[len(er_recs)]])  # one from each subset boundary
    assert batch["image"].shape[0] == 2 and batch["org_code"].shape == (2, 2)


def test_train_multi_and_evaluate_multi(er_setup, mito_setup):
    """train_multi runs 1-2 steps on the mixed dataset; evaluate_multi produces dice for each organelle."""
    from segmentation_training.experiments.multi_organelle.train_multi import evaluate_multi, train_multi
    from segmentation_training.harness.dataset import load_manifest

    cfg = _cfg("er", neck="naive_1x1", decoder="dodnet", tile=32, steps=2)
    er_train = load_manifest(er_setup.data_root, "group2_er", "train")
    mito_train = load_manifest(mito_setup.data_root, "group2_mito", "train")
    train_per_org = {"mito": (mito_train, mito_setup.data_root), "er": (er_train, er_setup.data_root)}
    # both mock encoders share the same mock manifest stats; use the er encoder for the shared base.
    model = train_multi(cfg, er_setup.encoder, train_per_org, device="cpu", n_organelles=2)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0

    er_test = load_manifest(er_setup.data_root, "group2_er", "test")[:2]
    mito_test = load_manifest(mito_setup.data_root, "group2_mito", "test")[:2]
    eval_per_org = {"mito": (mito_test, mito_setup.data_root), "er": (er_test, er_setup.data_root)}
    out = evaluate_multi(model, eval_per_org, cfg, "cpu", er_setup.encoder.image_mean,
                         er_setup.encoder.image_std, n_organelles=2)
    assert set(out.keys()) == {"mito", "er"}
    assert out["er"]["summary"]["macro"].get("dice") is not None
    assert out["mito"]["summary"]["macro"].get("dice") is not None


def test_per_organelle_lora_trains(er_setup):
    """per_organelle_lora trains one organelle's own-LoRA head for 1-2 steps + evals to a dice."""
    from segmentation_training.experiments.multi_organelle.per_organelle_lora import (
        evaluate_per_organelle, train_per_organelle_lora)
    from segmentation_training.harness.dataset import load_manifest

    # ER specialist: build_per_organelle_config forces neck=resnet34_detail + decoder=dpt from
    # config_templates.ORG_RECIPE; use a light tile + 2 steps.
    cfg = _cfg("er", neck="resnet34_detail", decoder="dpt", tile=32, steps=2)
    train_recs = load_manifest(er_setup.data_root, "group2_er", "train")
    model = train_per_organelle_lora(cfg, er_setup.encoder, train_recs, er_setup.data_root, "cpu",
                                     organelle="er")
    assert sum(p.numel() for p in model.trainable_parameters()) > 0
    test_recs = load_manifest(er_setup.data_root, "group2_er", "test")[:2]
    out = evaluate_per_organelle(model, test_recs, cfg, er_setup.data_root, "cpu",
                                 er_setup.encoder.image_mean, er_setup.encoder.image_std, organelle="er")
    assert out["summary"]["macro"].get("dice") is not None


def test_subset_code_map_non_prefix_no_crash():
    """A non-prefix organelle subset (e.g. mito+ld) maps to subset-local slots 0..K-1 rather than the
    fixed ORGANELLE_ORDER index, keeping every one_hot index inside K."""
    from segmentation_training.experiments.multi_organelle.mixed_dataset import subset_code_map, one_hot
    for sel in (["mito", "ld"], ["er", "nucleus"], ["ld", "er"], ["mito", "er"]):
        cm = subset_code_map(sel)
        k = len(cm)
        assert sorted(cm.values()) == list(range(k))          # dense 0..K-1
        for o in sel:
            assert one_hot(cm[o], k).shape[0] == k              # in-bounds (no IndexError)


def _per_org(er_setup, mito_setup):
    from segmentation_training.harness.dataset import load_manifest
    return {"er": (load_manifest(er_setup.data_root, "group2_er", "train"), er_setup.data_root),
            "mito": (load_manifest(mito_setup.data_root, "group2_mito", "train"), mito_setup.data_root)}


def test_seed_stats_wash_and_help():
    from segmentation_training.experiments.common.seed_stats import compare_arms
    assert compare_arms([0.50, 0.51, 0.49], [0.50, 0.49, 0.51], tie_k=1.0)["verdict"] == "wash"
    assert compare_arms([0.60, 0.61, 0.59], [0.50, 0.49, 0.51], tie_k=1.0)["verdict"] == "help"
    assert compare_arms([0.40, 0.41, 0.39], [0.50, 0.49, 0.51], tie_k=1.0)["verdict"] == "hurt"


def test_mixed_dataset_balance_and_ratio(er_setup, mito_setup):
    import collections
    from segmentation_training.experiments.common._smoke import smoke_cfg
    from segmentation_training.experiments.multi_organelle.mixed_dataset import build_mixed_dataset
    cfg = smoke_cfg("er", tile_size=32)
    per = _per_org(er_setup, mito_setup)
    raw = build_mixed_dataset(per, cfg, er_setup.encoder.image_mean, er_setup.encoder.image_std,
                              patch_size=16, balance="raw")
    bal = build_mixed_dataset(per, cfg, er_setup.encoder.image_mean, er_setup.encoder.image_std,
                              patch_size=16, balance="balanced")
    r = raw.ratio_report()
    assert set(r["raw_counts"]) == {"er", "mito"} and r["balance"] == "raw"
    # balanced index has equal representation per organelle-subset
    cnt = collections.Counter(si for si, _ in bal.index)
    assert len(set(cnt.values())) == 1


def test_mixed_collate_instance_ignore():
    import torch
    from segmentation_training.experiments.multi_organelle.mixed_dataset import MixedOrganelleDataset, INSTANCE_IGNORE
    er = {"image": torch.zeros(1, 32, 32), "target": torch.zeros(32, 32, dtype=torch.long),
          "org_idx": torch.tensor(0), "org_code": torch.tensor([1., 0.]), "_task": "semantic"}
    mito = {"image": torch.zeros(1, 32, 32), "target": torch.zeros(32, 32, dtype=torch.long),
            "org_idx": torch.tensor(1), "org_code": torch.tensor([0., 1.]),
            "inst": torch.zeros(32, 32, dtype=torch.long), "_task": "instance"}
    b = MixedOrganelleDataset.collate(None, [er, mito])              # collate uses only `batch`
    assert b["inst_task"].tolist() == [False, True]
    assert (b["inst"][0] == INSTANCE_IGNORE).all()                  # ER = IGNORE, not zero-supervised
    assert (b["inst"][1] == 0).all()                               # mito real (all-bg here)


def test_train_multi_reports_step_allocation(er_setup, mito_setup):
    from segmentation_training.experiments.common._smoke import smoke_cfg
    from segmentation_training.experiments.multi_organelle.train_multi import train_multi
    cfg = smoke_cfg("er", tile_size=32, max_steps=2)
    per = _per_org(er_setup, mito_setup)
    model = train_multi(cfg, er_setup.encoder, per, device="cpu", n_organelles=2, mid_channels=8, balance="balanced")
    ts = model._train_stats
    assert set(ts["per_organelle_step_fraction"]) == {"er", "mito"}
    assert ts["balance"] == "balanced" and "dataset_ratio" in ts


def _fake_report(d, arm, organelle, extra, dice=None, inst_pq=None):
    import json
    d.mkdir(parents=True, exist_ok=True)
    macro = {}
    if dice is not None:
        macro["dice"] = dice
    if inst_pq is not None:
        macro["inst_pq"] = inst_pq
    rep = {"arm": arm, "organelle": organelle, "splits": {"test": {"macro": macro}}, "extra": extra}
    (d / f"report_{arm}.json").write_text(json.dumps(rep), encoding="utf-8")


def test_multi_compare_wash_default(tmp_path):
    from segmentation_training.experiments.multi_organelle.run_multi_organelle import compare
    runs = tmp_path / "runs"
    for s, pq in enumerate([0.48, 0.50, 0.46]):    # specialist mito ~0.48 (sd~0.016)
        _fake_report(runs / f"spec_s{s}", f"specialist_mito_s{s}", "mito", {"arm": "specialist", "seed": s},
                     inst_pq=pq, dice=0.70)
    for s, pq in enumerate([0.485, 0.505, 0.465]):  # shared-dodnet Δ~0.005 << seed band ~0.016 -> wash
        _fake_report(runs / f"dod_s{s}", f"dodnet_mito_mid8_raw_s{s}", "mito",
                     {"arm": "shared-dodnet", "seed": s, "mid_channels": 8, "balance": "raw",
                      "train_stats": {"per_organelle_step_fraction": {"mito": 0.13, "er": 0.87},
                                      "dataset_ratio": {"max_over_min_ratio": 7.0}}},
                     inst_pq=pq, dice=0.71)
    rep = compare(runs, out_dir=tmp_path / "out", tie_k=1.0)
    mito = rep["mito"]
    assert mito["primary_metric"] == "instance"       # mito read off instance
    assert mito["variants"]["mid8_raw"]["metrics"]["instance"]["verdict"] == "wash"
    assert (tmp_path / "out" / "multi_organelle_verdict.json").exists()

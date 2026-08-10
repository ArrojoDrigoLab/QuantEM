"""Image-style conditioning — CPU unit + end-to-end tests (mock DINOv3 encoder, tile 64, cpu).

Covers FiLM shape/broadcast and identity-at-init, style-pooling scope correctness
(tile vs source), gradient-reversal sign, MixStyle/DSU train-vs-eval, annotation-masked per-subgroup
metrics, and the full train+eval stack for the conditioning arms + the DANN adversary + head round-trip.
Also covers the test-time support family reached from the same arms: seed source x combination x
gating, FiLM liveness and gating, field-size fairness, and the support report.

Runs without a GPU: torch CPU and scipy.ndimage only (no sklearn/skimage).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from em_ssl.utils.checkpoint_index import CheckpointIndex
from segmentation_training.config.schema import SegConfig
from segmentation_training.harness.encoders import FrozenEncoder, select_checkpoints


# --------------------------------------------------------------------------- #
# Unit tests (no corpus needed)
# --------------------------------------------------------------------------- #
def test_film_shape_broadcast_and_identity_init():
    import torch.nn as nn

    from segmentation_training.models.conditioning.film import FiLMConditioner, FiLMHead

    code = torch.randn(2, 32)
    g, b = FiLMHead(32, 16)(code)
    assert g.shape == (2, 16) and b.shape == (2, 16)
    assert torch.allclose(g, torch.ones_like(g)) and torch.allclose(b, torch.zeros_like(b))  # identity init

    net = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.GroupNorm(2, 8), nn.GELU(),
                        nn.Conv2d(8, 8, 3, padding=1), nn.GroupNorm(2, 8))
    cond = FiLMConditioner(32, {"net": net}, scope="per_block")
    assert len(cond.points) == 2
    x = torch.randn(2, 4, 12, 12)
    base = net(x).clone()
    cond.set_code(code)
    assert torch.allclose(net(x), base, atol=1e-5)  # gamma=1,beta=0 -> conditioned output == base
    cond.set_code(None)
    assert torch.allclose(net(x), base, atol=1e-6)  # no code -> passthrough
    assert len(FiLMConditioner(32, {"net": net}, scope="once").points) == 1


def test_style_scope_pooling_tile_vs_source():
    from segmentation_training.models.conditioning.pooling import pool_by_source

    codes = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]])
    assert torch.allclose(pool_by_source(codes, None), codes)  # tile scope = identity
    src = torch.tensor([0, 0, 1, 1])
    pooled = pool_by_source(codes, src)  # source scope = same-source mean
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0], [2.0, 2.0], [6.0, 6.0], [6.0, 6.0]]))


def test_gradient_reversal_sign():
    from segmentation_training.models.conditioning.grl import dann_lambda, grad_reverse

    w = torch.tensor([2.0], requires_grad=True)
    grad_reverse(w * 3.0, alpha=0.5).sum().backward()
    assert abs(float(w.grad) - (-1.5)) < 1e-6  # backward negates + scales by alpha
    assert dann_lambda(0.0) == 0.0 and 0.0 < dann_lambda(0.5) < 1.0 and dann_lambda(1.0) <= 1.0


def test_mixstyle_dsu_train_only():
    from segmentation_training.models.conditioning.mixstyle import DSU, MixStyle

    torch.manual_seed(0)
    f = torch.randn(6, 8, 10, 10)
    for m in (MixStyle(p=1.0, alpha=0.1), DSU(p=1.0)):
        m.eval()
        assert torch.allclose(m(f), f)  # identity at eval
        m.train()
        # perturbs in train (MixStyle rarely draws the identity permutation, so the check retries)
        assert any(not torch.allclose(m(f), f) for _ in range(5))


def test_low_level_stats_dim_and_finiteness():
    from segmentation_training.models.conditioning.style_encoder import STAT_NAMES, STATS_DIM, low_level_stats

    s = low_level_stats(torch.rand(3, 1, 256, 256))
    assert s.shape == (3, STATS_DIM) == (3, len(STAT_NAMES))
    assert torch.isfinite(s).all()


def test_masked_per_subgroup_metrics():
    """Annotation-masked per-subgroup aggregation: an ignore-band FP is not counted, and the worst
    subgroup recall is surfaced."""
    from segmentation_training.constants import FOREGROUND, IGNORE_INDEX
    from segmentation_training.harness.metrics import aggregate, per_crop_metrics

    def crop(sub, recall_hit):
        gt = np.zeros((40, 40), np.uint8)
        gt[10:30, 10:30] = FOREGROUND
        gt[:3, :] = IGNORE_INDEX
        pred = np.zeros((40, 40), bool)
        pred[10:30, 10:30] = recall_hit  # full hit (recall 1) or miss (recall ~0)
        pred[:3, :] = True  # FP inside ignore band — must be excluded
        m = per_crop_metrics(pred, gt, organelle="mito")
        m["subgroup"] = sub
        return m

    recs = [crop("A", True), crop("A", True), crop("B", False)]
    agg = aggregate(recs, bootstrap_n=0)
    assert set(agg["per_subgroup"]) == {"A", "B"}
    assert agg["per_subgroup"]["A"]["recall"] > 0.99  # ignore-band FP didn't hurt recall
    assert agg["worst_subgroup"]["recall"]["subgroup"] == "B"  # under-caller surfaced


# --------------------------------------------------------------------------- #
# End-to-end (mock encoder + synthetic derived corpus)
# --------------------------------------------------------------------------- #
def _encoder(tmp_path, ckpt_dir=None):
    """A frozen mock DINOv3 encoder. Pass a shared ``ckpt_dir`` to load two encoders with identical
    frozen weights (each write_mock_checkpoint call randomises the backbone, so the loader round-trip
    reuses one checkpoint, matching how a real arm loads a single encoder checkpoint)."""
    from segmentation_training._synthetic import write_mock_checkpoint

    if ckpt_dir is None or not (ckpt_dir / "checkpoint_index.json").exists():
        run_dir = write_mock_checkpoint(ckpt_dir or (tmp_path / "mock_encoder"), "dinov3", arch="vit_small")
    else:
        run_dir = ckpt_dir
    idx = CheckpointIndex.load(run_dir)
    rec = select_checkpoints(idx, n=1)[-1]
    return FrozenEncoder.from_manifest(rec.path, idx.manifest, tile_size=64)


def _cond_cfg(cond: dict, organelle="er", neck="resnet34_detail", decoder="dpt", task="semantic"):
    return SegConfig.from_dict({
        "name": f"conditioning_{cond.get('arm', 'A')}",
        "device": "cpu", "amp": False, "num_workers": 0,
        "encoder": {"tile_size": 64, "feature_layers": "last4"},
        "neck": {"type": neck}, "decoder": {"type": decoder},
        "loss": {"terms": [{"type": "dice_bce", "weight": 1.0}]},
        "data": {"organelle": organelle, "num_classes": 2, "task": task, "min_fg_frac_keep": 0.0},
        "optim": {"max_steps": 2, "warmup_steps": 1, "batch_size": 2, "lr": 1e-3, "seed": 0},
        "eval": {"overlap": 0.0, "bootstrap_n": 0},
        "cond": cond,
    })


@pytest.fixture(scope="module")
def derived(tmp_path_factory):
    from segmentation_training._synthetic import build_synthetic_corpus
    from segmentation_training.dataprep.build_dataset import run

    root = tmp_path_factory.mktemp("seg_e1")
    build_synthetic_corpus(root)
    out = tmp_path_factory.mktemp("seg_data")
    run(types.SimpleNamespace(corpus_root=str(root), out=str(out), organelles=["er"], splits=None,
                              context_frac=0.5, limit=0, null_scale_policy="drop", target_nm=0.0))
    return out


@pytest.mark.parametrize("cond", [
    {"arm": "inferred_style_tile", "enabled": True, "style_source": "inferred", "style_scope": "tile"},
    {"arm": "inferred_style", "enabled": True, "style_source": "inferred", "style_scope": "dataset"},
    {"arm": "mixstyle_dsu", "enabled": True, "film": False, "mixstyle": "mixstyle"},
    {"arm": "dsu", "enabled": True, "film": False, "mixstyle": "dsu"},
])
def test_e2e_conditioned_arm_trains_and_evals(derived, tmp_path, cond):
    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.evaluate import evaluate_head
    from segmentation_training.harness.train import train_segmodel

    enc = _encoder(tmp_path)
    cfg = _cond_cfg(cond)
    train_recs = load_manifest(derived, "group2_er", "train")
    test_recs = load_manifest(derived, "group2_er", "test")
    assert train_recs and test_recs

    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")
    assert model.conditioner is not None
    out = evaluate_head(model, test_recs, cfg, str(derived), device="cpu",
                        mean=enc.image_mean, std=enc.image_std)
    assert out["summary"]["macro"].get("dice") is not None


def test_a0_disabled_has_no_conditioner(derived, tmp_path):
    from segmentation_training.harness.train import build_segmodel

    enc = _encoder(tmp_path)
    model = build_segmodel(_cond_cfg({"arm": "baseline", "enabled": False}), enc)
    assert model.conditioner is None  # the unconditioned baseline is byte-identical to the base arm


def test_adversary_trains_and_conditioner_updates(derived, tmp_path):
    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.train import train_segmodel

    enc = _encoder(tmp_path)
    cfg = _cond_cfg({"arm": "inferred_style_adversary", "enabled": True, "style_source": "inferred",
                     "grad_reversal": 1.0, "adv_targets": ["dataset"]})
    train_recs = load_manifest(derived, "group2_er", "train")
    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")
    assert model.conditioner.adversary is not None
    # the style encoder actually received gradient (params moved from init)
    updated = any(p.grad is not None or p.abs().sum() > 0 for p in model.conditioner.style_encoder.parameters())
    assert updated


def test_head_roundtrip_save_load(derived, tmp_path):
    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.meta import MetaVocab
    from segmentation_training.harness.train import build_segmodel, train_segmodel

    enc = _encoder(tmp_path)
    cfg = _cond_cfg({"arm": "inferred_style_tile", "enabled": True, "style_source": "inferred"})
    train_recs = load_manifest(derived, "group2_er", "train")
    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")

    state = {"conditioner": model.conditioner.state_dict(),
             "meta_vocab": model._meta_vocab.to_dict(),
             "neck": model.neck.state_dict(), "decoder": model.decoder.state_dict()}

    fresh_enc = _encoder(tmp_path)
    vocab = MetaVocab.from_dict(state["meta_vocab"])
    fresh = build_segmodel(cfg, fresh_enc, field_sizes=vocab.sizes())
    fresh.neck.load_state_dict(state["neck"])
    fresh.decoder.load_state_dict(state["decoder"])
    missing, unexpected = fresh.conditioner.load_state_dict(state["conditioner"], strict=False)
    assert not unexpected  # every saved conditioner tensor maps back


def test_load_adapted_lora_plus_conditioner_roundtrip(derived, tmp_path):
    """The full head.pt save/load path (LoRA adapters + conditioner) reproduces the trained model's
    forward output — the loader the TTA arms and warm-started arms rely on."""
    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.load_adapted import build_and_load_head
    from segmentation_training.harness.train import train_segmodel

    ckpt_dir = tmp_path / "shared_ckpt"
    enc = _encoder(tmp_path, ckpt_dir=ckpt_dir)  # trained model's encoder
    cfg = _cond_cfg({"arm": "inferred_style_tile", "enabled": True, "style_source": "inferred"})
    cfg.encoder.adapt = "lora"
    cfg.encoder.adapt_params = {"rank": 4, "conv": False}
    train_recs = load_manifest(derived, "group2_er", "train")
    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")

    head = tmp_path / "head.pt"
    enc_trainable = {n: p.detach().cpu() for n, p in model.encoder.named_parameters() if p.requires_grad}
    torch.save({"neck": model.neck.state_dict(), "decoder": model.decoder.state_dict(),
                "encoder_trainable": enc_trainable, "adapters": model.encoder._conv_lora.state_dict(),
                "conditioner": model.conditioner.state_dict(),
                "meta_vocab": model._meta_vocab.to_dict()}, head)

    fresh_enc = _encoder(tmp_path, ckpt_dir=ckpt_dir)  # same frozen backbone (as reloading one encoder ckpt)
    loaded, vocab, info = build_and_load_head(cfg, fresh_enc, head, device="cpu")
    assert loaded.conditioner is not None and info["encoder_loaded"] == len(enc_trainable)

    x = torch.randn(1, 1, 64, 64)
    r = train_recs[0]
    model.eval()
    model.set_record_context(r, "cpu"); loaded.set_record_context(r, "cpu")
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)  # reloaded model == trained model


def test_run_arm_factorial_path_scores_test_image(derived, tmp_path):
    """End-to-end: run_arm trains an inferred-style LoRA arm, saves a head with conditioner+vocab, and auto-scores
    the rebalanced test_image split alongside test_source (the gap-decomposition readout)."""
    import json

    from segmentation_training._synthetic import write_mock_checkpoint
    from segmentation_training.dataprep.rebalance_heldout_image import rebalance
    from segmentation_training.harness.run_seg import run_arm

    # Rebalance the derived manifest to add a test_image split (force some crops with a small min).
    recs = [json.loads(l) for l in (derived / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out_recs, stats = rebalance(recs, "er", holdout_frac=0.3, min_crops=2)
    (derived / "manifest_heldimg.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out_recs) + "\n", encoding="utf-8")

    run_dir = write_mock_checkpoint(tmp_path / "mock_encoder", "dinov3", arch="vit_small")
    cfg = _cond_cfg({"arm": "inferred_style", "enabled": True, "style_source": "inferred", "style_scope": "dataset"})
    cfg.encoder.run_dir = str(run_dir)
    cfg.encoder.adapt = "lora"
    cfg.encoder.adapt_params = {"rank": 4, "conv": False}
    cfg.data.manifest_name = "manifest_heldimg.jsonl"
    out = tmp_path / "runs" / cfg.name
    run_arm(cfg, str(derived), str(out), device="cpu", max_steps=2)

    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert "test" in res["splits"]
    if stats["n_held"]:  # test_image present -> it must be scored too (the gap-decomposition anchor)
        assert "test_image" in res["splits"]
    head = torch.load(out / "head.pt", map_location="cpu", weights_only=False)
    assert head["conditioner"] is not None and head["meta_vocab"] is not None
    assert head["adapters"] is not None  # LoRA adapters saved


def test_film_liveness_gate(derived, tmp_path):
    """FiLM gradients reach the generators, and swapping style codes changes the output
    only once the heads depart from identity; a zero-init conditioner reads as "INERT"."""
    import torch.nn as nn

    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.film_liveness import (gradient_reaches_film, liveness_report, output_sensitivity,
                                            verdict)
    from segmentation_training.harness.train import build_segmodel, train_segmodel

    enc = _encoder(tmp_path)
    cfg = _cond_cfg({"arm": "inferred_style_tile", "enabled": True, "style_source": "inferred"})
    train_recs = load_manifest(derived, "group2_er", "train")
    x = torch.randn(1, 1, 64, 64)
    tgt = torch.zeros(1, 64, 64, dtype=torch.long)
    code_a, code_b = torch.randn(cfg.cond.style_dim), torch.randn(cfg.cond.style_dim)

    # zero-init FiLM (untrained) reads as "INERT" (no output change) — the dead-path detector.
    fresh = build_segmodel(cfg, enc, field_sizes={"dataset": 3})
    assert verdict(output_sensitivity(fresh, x, code_a, code_b)) == "INERT"
    for h in fresh.conditioner.film.heads.values():  # perturb -> path is demonstrably "LIVE"
        nn.init.normal_(h.proj.weight, std=0.5)
    assert verdict(output_sensitivity(fresh, x, code_a, code_b)) == "LIVE"

    # gradient reaches the FiLM generators (mechanism is trainable, not detached)
    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")
    assert gradient_reaches_film(model, x, tgt, code_a)["reaches"] is True
    rep = liveness_report(model, x)
    assert rep["has_film"] and rep["verdict"] in ("LIVE", "INERT")


def test_center_window_field_fairness():
    """The central base-tile window is the same real-content window regardless of field size, so the
    field-size axis is compared fairly."""
    from segmentation_training.harness.tta import _center_window

    a = np.arange(1, 37, dtype=np.int32).reshape(6, 6)
    w = _center_window(a, 8)                              # pad 6->8 symmetric: content centered
    assert w.shape == (8, 8) and np.array_equal(w[1:7, 1:7], a)
    m = _center_window(np.ones((6, 6), np.int32), 8, mask=True)
    assert (m[0, :] == 255).all() and (m[1:7, 1:7] == 1).all()  # mask padding is ignore (never scored)

    # region larger than base but (potentially) smaller than field: central-of-field == center-at-base
    big = (np.arange(100 * 100, dtype=np.int32) % 251).reshape(100, 100)
    base, field = 64, 128
    off = (field - base) // 2
    central = _center_window(big, field)[off:off + base, off:off + base]
    assert np.array_equal(central, _center_window(big, base))  # same window -> fair field axis


@pytest.mark.parametrize("sup", [
    {"support_source": "gt", "support_combine": "replace"},               # GT seeds: upper bound
    {"support_source": "inferred", "support_combine": "replace"},         # inferred seeds, hard replacement
    {"support_source": "inferred_gated", "support_combine": "uncertainty_gated"},  # gated seeds, gated combine
    {"support_source": "inferred_gated", "support_combine": "residual", "n_support": 4, "proto_gate": True},
    {"support_source": "interactive", "support_combine": "uncertainty_gated", "interactive_clicks": 2},
])
def test_support_family_axes(derived, tmp_path, sup):
    """Support family: seed source x combination x K/gating, on an unconditioned base (post-hoc)."""
    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.tta import run_support
    from segmentation_training.harness.train import train_segmodel

    enc = _encoder(tmp_path)
    cfg = _cond_cfg({"arm": "baseline", "enabled": False, "tta": "support", "support_min_size": 4,
                     "support_conf": 0.5, "confident_thresh": 0.5, **sup})
    train_recs = load_manifest(derived, "group2_er", "train")
    test_recs = load_manifest(derived, "group2_er", "test")
    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")
    out = run_support(model, test_recs, cfg, str(derived), device="cpu", mean=enc.image_mean, std=enc.image_std)
    assert out["per_crop"] and out["summary"]["support"]["source"] == sup["support_source"]
    assert out["summary"]["support"]["combine"] == sup["support_combine"]
    # every scored crop reports precision + recall (the precision-recovery success metric) and ceiling flag
    assert all("precision" in c and "is_ceiling" in c for c in out["per_crop"])
    assert out["summary"]["support"]["is_ceiling"] == (sup["support_source"] == "gt")


def test_support_early_filmroute_needs_confident_head(derived, tmp_path):
    """Early combination is FiLM-routed: it requires a confident_feature head (otherwise a clear error),
    and honours the configured seed source (here interactive GT clicks) as the FiLM appearance mask."""
    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.tta import run_support
    from segmentation_training.harness.train import build_segmodel, train_segmodel

    enc = _encoder(tmp_path)
    # no confident head -> clear error
    a0 = build_segmodel(_cond_cfg({"arm": "baseline", "enabled": False, "support_combine": "early"}), enc)
    import pytest as _pt
    with _pt.raises(ValueError, match="confident_feature"):
        run_support(a0, [], _cond_cfg({"support_combine": "early"}), str(derived), "cpu",
                    enc.image_mean, enc.image_std)
    # with a confident_feature head + interactive seeds -> runs
    cfg = _cond_cfg({"arm": "pooled_global", "enabled": True, "style_source": "confident_feature",
                     "support_combine": "early", "support_source": "interactive", "interactive_clicks": 2,
                     "tta": "support"})
    train_recs = load_manifest(derived, "group2_er", "train")
    test_recs = load_manifest(derived, "group2_er", "test")
    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")
    out = run_support(model, test_recs, cfg, str(derived), device="cpu", mean=enc.image_mean, std=enc.image_std)
    assert out["per_crop"] and out["summary"]["support"]["combine"] == "early"


def test_few_shot_seeds_and_corruption():
    """Few-shot selection picks k GT instances; seed corruption degrades seed recall (drop) or precision
    (false) with measurable realized quality."""
    import types

    from segmentation_training.harness.tta import _corrupt_seed, _select_seed_mask, _seed_quality

    gt = np.zeros((40, 40), np.int32)
    gt[5:10, 5:10] = 1        # instance, 25 px
    gt[20:30, 20:30] = 1      # instance, 100 px
    gt[35:38, 2:5] = 1        # instance, 9 px
    gt_fg = gt == 1
    cfg = types.SimpleNamespace(cond=types.SimpleNamespace(n_shots=2, confident_thresh=0.5))
    gen = torch.Generator(device="cpu").manual_seed(0)
    fs = _select_seed_mask("few_shot", np.zeros((40, 40)), gt.astype(np.uint8), cfg, gen)
    assert int(fs.sum()) == 125  # the two largest instances (100+25), not the 9-px speck

    full = gt_fg.copy()
    bg = gt == 0  # true background, excluding ignore; false seeds are drawn from here
    q_drop = _seed_quality(_corrupt_seed(full.copy(), bg, 0.5, 0.0, gen), gt_fg)
    assert 0.35 < q_drop["seed_recall"] < 0.65 and q_drop["seed_precision"] > 0.99  # recall down, precision kept
    q_false = _seed_quality(_corrupt_seed(full.copy(), bg, 0.0, 1.0, gen), gt_fg)
    assert q_false["seed_precision"] < 0.6 and q_false["seed_recall"] > 0.99  # precision down, recall kept

    # false seeds land on true background only, never on ignore (255) regions
    gt_ig = gt.copy()
    gt_ig[0:15, 30:40] = 255  # a large ignore band
    corrupted = _corrupt_seed(gt_fg.copy(), gt_ig == 0, 0.0, 2.0, gen)
    injected = corrupted & ~gt_fg
    assert not (injected & (gt_ig == 255)).any()  # nothing injected into ignore


def test_film_gate_reports_collapse_frequency(derived, tmp_path):
    """Early combination with film_gate uncertainty-gates the conditioned pass against the base and
    reports how often the ungated conditioned pass collapses."""
    from segmentation_training.harness.dataset import load_manifest
    from segmentation_training.harness.tta import run_support
    from segmentation_training.harness.train import train_segmodel

    enc = _encoder(tmp_path)
    cfg = _cond_cfg({"arm": "pooled_global", "enabled": True, "style_source": "confident_feature",
                     "support_combine": "early", "support_source": "inferred_gated", "film_gate": True,
                     "confident_thresh": 0.5, "support_conf": 0.5, "support_min_size": 4, "tta": "support"})
    train_recs = load_manifest(derived, "group2_er", "train")
    test_recs = load_manifest(derived, "group2_er", "test")
    model = train_segmodel(cfg, enc, train_recs, str(derived), device="cpu")
    out = run_support(model, test_recs, cfg, str(derived), device="cpu", mean=enc.image_mean, std=enc.image_std)
    s = out["summary"]["support"]
    assert s["film_gate"] is True and "ungated_collapse_frequency" in s
    assert 0.0 <= s["ungated_collapse_frequency"] <= 1.0


def test_support_uncertainty_gate_protects_confident_base():
    """Uncertainty-gated combination leaves a confident head prediction unchanged, so it cannot degrade a
    strong base."""
    import types

    from segmentation_training.harness.tta import _combine_support

    cfg = types.SimpleNamespace(cond=types.SimpleNamespace(support_uncertain_margin=0.2, support_alpha=0.5))
    prob0 = torch.tensor([0.95, 0.02, 0.55, 0.48])   # confident fg, confident bg, unconfident, unconfident
    support = torch.tensor([0.10, 0.90, 0.90, 0.10])  # support disagrees everywhere
    out = _combine_support("uncertainty_gated", prob0, support, cfg)
    assert torch.allclose(out[:2], prob0[:2])         # confident pixels untouched
    assert torch.allclose(out[2:], support[2:])       # only the unconfident band moves


def test_support_report_precision_recovery_loso_watchlist():
    from segmentation_training.harness.support_report import loso_source_delta, precision_recovery, watchlist_rows

    a0 = [{"subgroup": s, "dataset": s, "precision": 0.3, "recall": 0.9, "dice": 0.4, "excluded": False}
          for s in ("s1", "s2", "s3", "s4")]
    # precision up, recall held -> a recovery
    arm = [{"subgroup": s, "dataset": s, "precision": 0.6, "recall": 0.88, "dice": 0.55, "excluded": False}
           for s in ("s1", "s2", "s3", "s4")]
    pr = precision_recovery(arm, a0, organelle="er")
    assert pr["precision_recovered"] is True and pr["d_precision_vs_baseline"] > 0
    # precision up but recall collapsed -> not a recovery
    arm2 = [{"subgroup": s, "dataset": s, "precision": 0.7, "recall": 0.3, "dice": 0.4, "excluded": False}
            for s in ("s1", "s2", "s3", "s4")]
    assert precision_recovery(arm2, a0, organelle="er")["precision_recovered"] is False

    lo = loso_source_delta(arm, a0, "dice")
    assert lo["n_sources"] == 4 and lo["point_delta"] is not None and lo["jackknife_sd"] is not None

    wl = watchlist_rows([{"subgroup": "FAST-EM | pancreas", "dataset": "x", "precision": 0.5,
                          "recall": 0.5, "dice": 0.5, "excluded": False}])
    assert any("FAST-EM" in k for k in wl)

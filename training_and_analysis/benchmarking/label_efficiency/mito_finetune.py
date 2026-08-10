"""Per-dataset fine-tune of the mito ViT models on the GK train confirmed-area crops, eval held-out
test. Modes on the light->heavy ladder: head_only (neck+decoder), lora (omni_cem's native adapters),
lastN (unfreeze last-N ViT blocks). Valid-mask-aware loss (ignore outside the labeled ROI). Reports
base vs adapted, each threshold-calibrated on train for a fair 'best achievable' number + wall-clock.
Writes ft_<model>_<mode>.json into --gt-root.

Ground-truth layout: see mito_vit.py; train/test come from split.json unless --train-crops and
--test-crops override them (e.g. for leave-one-image-out). Run from training_and_analysis/ with
PYTHONPATH=. in the quantem-segmentation-training environment
(../../segmentation_training/environment.yml).

Usage:
    python benchmarking/label_efficiency/mito_finetune.py <qem_cem|omni_cem> <head_only|lora|lastN>
        --gt-root DIR --head-dir DIR --backbone-dir DIR [--steps 300] [--last-n 4]
        [--train-crops a,b --test-crops c,d]
"""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path

from mito_vit import load_vit, load_crops, split, masked_dice
from segmentation_training.harness.evaluate import predict_region, _round_up
from segmentation_training.harness.dataset import normalize_em
from segmentation_training.hooks.encoder_adaptation import _unfreeze_last_n_blocks

ap = argparse.ArgumentParser(description="Per-dataset fine-tune of a released mito ViT model.")
ap.add_argument("model", choices=("qem_cem", "omni_cem"))
ap.add_argument("mode", choices=("head_only", "lora", "lastN"))
ap.add_argument("--steps", type=int, default=300, help="adaptation steps")
ap.add_argument("--last-n", type=int, default=4, help="ViT blocks to unfreeze in lastN mode")
ap.add_argument("--gt-root", required=True, help="ground-truth crop directory (layout: see mito_vit.py)")
ap.add_argument("--head-dir", required=True, help="directory containing the trained head.pt")
ap.add_argument("--backbone-dir", required=True,
                help="encoder pretraining run directory (holds checkpoint_index.json)")
ap.add_argument("--train-crops", default=None,
                help="comma-separated crop names; with --test-crops, overrides split.json")
ap.add_argument("--test-crops", default=None,
                help="comma-separated crop names; with --train-crops, overrides split.json")
args = ap.parse_args()

MODEL = args.model; MODE = args.mode
STEPS = args.steps
LASTN = args.last_n
GT = Path(args.gt_root)
SEED = 0; torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
IGN = 255

model, cfg, enc, dev = load_vit(MODEL, args.head_dir, args.backbone_dir)
mean, std = enc.image_mean, enc.image_std
if args.train_crops and args.test_crops:
    train, test = args.train_crops.split(","), args.test_crops.split(",")   # leave-one-image-out override
else:
    train, test = split(GT)
crops = load_crops(GT)
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def eval_calibrated(tag):
    model.eval()
    preds = {}
    for n in (train + test):
        preds[n] = predict_region(model, crops[n][0], cfg, mean, std, dev)
    def md(names, thr):
        return float(np.mean([masked_dice(preds[n], crops[n][1], crops[n][2], thr) for n in names]))
    cthr = max(THRS, key=lambda t: md(train, t))
    return {"test_thr0.5": md(test, 0.5), "test_cal": md(test, cthr), "cal_thr": float(cthr),
            "train_cal": md(train, cthr), "per_test": {n: masked_dice(preds[n], crops[n][1], crops[n][2], cthr) for n in test}}


base = eval_calibrated("base")
print(f"[{MODEL}/{MODE}] BASE test cal={base['test_cal']:.4f} (thr {base['cal_thr']}) per-test={ {k:round(v,3) for k,v in base['per_test'].items()} }", flush=True)

# ---- build tile patches from TRAIN crops (valid-aware target; ignore outside ROI) ----
patch = int(getattr(model.encoder, "patch_size", 16))
t = _round_up(int(cfg.encoder.tile_size), patch)  # patch-multiple tile (518 for patch14, 512 for patch16)
pats = []
for n in train:
    em, gt, valid = crops[n]
    xn = normalize_em(em, mean, std)
    for y in range(0, max(1, em.shape[0]-t+1), t//2):
        for x in range(0, max(1, em.shape[1]-t+1), t//2):
            v = valid[y:y+t, x:x+t]
            if v.shape != (t, t) or v.sum() < 0.2*t*t:  # need enough labeled area
                continue
            g = gt[y:y+t, x:x+t].astype(np.int64)
            g[v == 0] = IGN
            pats.append((xn[y:y+t, x:x+t].copy(), g))
print(f"  train patches: {len(pats)}", flush=True)

# ---- set trainable params per mode ----
for p in model.parameters(): p.requires_grad_(False)
for m in (model.neck, model.decoder):
    for p in m.parameters(): p.requires_grad_(True)
if MODE == "lora":
    lora = getattr(model.encoder, "_conv_lora", None)
    assert lora is not None, f"{MODEL} has no _conv_lora (not a lora-adapt model)"
    for p in lora.parameters(): p.requires_grad_(True)
elif MODE == "lastN":
    _unfreeze_last_n_blocks(model.encoder, LASTN)
elif MODE != "head_only":
    raise SystemExit(f"unknown mode {MODE}")
tp = [p for p in model.parameters() if p.requires_grad]
n_tp = sum(p.numel() for p in tp)
opt = torch.optim.AdamW(tp, lr=1e-4, weight_decay=1e-4)
print(f"  mode={MODE} trainable_params={n_tp/1e6:.2f}M", flush=True)


def aug(im, g):
    if rng.random() < 0.5: im, g = im[:, ::-1].copy(), g[:, ::-1].copy()
    if rng.random() < 0.5: im, g = im[::-1].copy(), g[::-1].copy()
    k = int(rng.integers(4)); im, g = np.rot90(im, k).copy(), np.rot90(g, k).copy()
    return im, g


def loss_fn(logits, tgt):
    ce = F.cross_entropy(logits, tgt[None], ignore_index=IGN)
    prob = torch.softmax(logits, 1)[:, 1]
    valid = (tgt != IGN).float()[None]; g = (tgt == 1).float()[None]
    inter = (prob*g*valid).sum(); denom = ((prob+g)*valid).sum()
    return ce + (1 - (2*inter+1)/(denom+1))


model.train()
t0 = time.time()
for step in range(STEPS):
    im, g = aug(*pats[int(rng.integers(len(pats)))])
    xt = torch.from_numpy(im)[None, None].float().to(dev)
    tt = torch.from_numpy(g).to(dev)
    logits = model(xt)
    loss = loss_fn(logits, tt)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 75 == 0 or step == STEPS-1:
        print(f"    step {step:3d} loss {loss.item():.3f}", flush=True)
train_s = time.time() - t0

ad = eval_calibrated("adapted")
out = {"model": MODEL, "mode": MODE, "steps": STEPS, "trainable_M": round(n_tp/1e6, 3),
       "train_sec": round(train_s, 1), "base": base, "adapted": ad,
       "delta_cal": round(ad["test_cal"] - base["test_cal"], 4)}
(GT / f"ft_{MODEL}_{MODE}{'' if MODE!='lastN' else LASTN}.json").write_text(json.dumps(out, indent=2))
print(f"\n=== {MODEL} / {MODE} ({n_tp/1e6:.2f}M params, {train_s:.0f}s, {STEPS} steps) ===")
print(f"  BASE    test cal = {base['test_cal']:.4f}")
print(f"  ADAPTED test cal = {ad['test_cal']:.4f}   (thr {ad['cal_thr']})  Δ={ad['test_cal']-base['test_cal']:+.4f}")
print(f"  per-test adapted: { {k:round(v,3) for k,v in ad['per_test'].items()} }")

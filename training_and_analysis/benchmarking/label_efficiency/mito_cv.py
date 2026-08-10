"""Cross-validated labeling-efficiency curve for per-dataset mito adaptation.
Image-disjoint CV (hold out M whole images as TEST -> no within-image leakage), vary #training
regions k (sampled from the other images), repeat R times with random train/test image swaps.
Loads the model ONCE and resets the trainable state per trial. Reports mean±std test Dice vs k
and the minimum k to exceed 0.90. Writes cv_<model>_<mode>.json into --gt-root.

Ground-truth layout: see mito_vit.py. Run from training_and_analysis/ with PYTHONPATH=. in the
quantem-segmentation-training environment (../../segmentation_training/environment.yml).

Usage:
    python benchmarking/label_efficiency/mito_cv.py <qem_cem|omni_cem> <head_only|lora|lastN>
        --gt-root DIR --head-dir DIR --backbone-dir DIR [--trials 8] [--steps 250]
        [--m-test 2] [--k-list 1,2,3,4,6,8,10]
"""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

from mito_vit import load_vit, load_crops, masked_dice
from segmentation_training.harness.evaluate import predict_region, _round_up
from segmentation_training.harness.dataset import normalize_em
from segmentation_training.hooks.encoder_adaptation import _unfreeze_last_n_blocks

ap = argparse.ArgumentParser(description="Image-disjoint CV labeling-efficiency curve (ViT models).")
ap.add_argument("model", choices=("qem_cem", "omni_cem"))
ap.add_argument("mode", choices=("head_only", "lora", "lastN"))
ap.add_argument("--trials", type=int, default=8, help="repeats R per k (random train/test image swaps)")
ap.add_argument("--steps", type=int, default=250, help="adaptation steps per trial")
ap.add_argument("--gt-root", required=True, help="ground-truth crop directory (layout: see mito_vit.py)")
ap.add_argument("--head-dir", required=True, help="directory containing the trained head.pt")
ap.add_argument("--backbone-dir", required=True,
                help="encoder pretraining run directory (holds checkpoint_index.json)")
ap.add_argument("--m-test", type=int, default=2, help="hold out this many whole images as test")
ap.add_argument("--k-list", default="1,2,3,4,6,8,10", help="comma-separated #training-region counts")
args = ap.parse_args()

MODEL = args.model; MODE = args.mode
R = args.trials
STEPS = args.steps
GT = Path(args.gt_root)
M_TEST = args.m_test
K_LIST = [int(x) for x in args.k_list.split(",")]
IGN = 255
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)

model, cfg, enc, dev = load_vit(MODEL, args.head_dir, args.backbone_dir)
mean, std = enc.image_mean, enc.image_std
patch = int(getattr(model.encoder, "patch_size", 16))
tile = _round_up(int(cfg.encoder.tile_size), patch)
crops = load_crops(GT)
images = defaultdict(list)
for n in crops: images[n.split("_")[0]].append(n)
img_ids = sorted(images)
print(f"[{MODEL}/{MODE}] {len(crops)} regions across {len(img_ids)} images; M_TEST={M_TEST} R={R} STEPS={STEPS}", flush=True)

init_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_trainable():
    for p in model.parameters(): p.requires_grad_(False)
    for m in (model.neck, model.decoder):
        for p in m.parameters(): p.requires_grad_(True)
    if MODE == "lora":
        for p in model.encoder._conv_lora.parameters(): p.requires_grad_(True)
    elif MODE == "lastN":
        _unfreeze_last_n_blocks(model.encoder, 4)


def make_patches(regions):
    pats = []
    for n in regions:
        em, gt, valid = crops[n]; xn = normalize_em(em, mean, std)
        for y in range(0, max(1, em.shape[0]-tile+1), tile//2):
            for x in range(0, max(1, em.shape[1]-tile+1), tile//2):
                v = valid[y:y+tile, x:x+tile]
                if v.shape != (tile, tile) or v.sum() < 0.2*tile*tile:
                    continue
                g = gt[y:y+tile, x:x+tile].astype(np.int64); g[v == 0] = IGN
                pats.append((xn[y:y+tile, x:x+tile].copy(), g))
    return pats


def loss_fn(logits, tgt):
    ce = F.cross_entropy(logits, tgt[None], ignore_index=IGN)
    prob = torch.softmax(logits, 1)[:, 1]
    valid = (tgt != IGN).float()[None]; g = (tgt == 1).float()[None]
    return ce + (1 - (2*(prob*g*valid).sum()+1)/(((prob+g)*valid).sum()+1))


def train_eval(train_regions, test_regions, seed):
    model.load_state_dict(init_sd)                 # reset to base
    set_trainable()
    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=1e-4, weight_decay=1e-4)
    pats = make_patches(train_regions)
    if not pats:
        return None
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(STEPS):
        im, g = pats[int(rng.integers(len(pats)))]
        if rng.random() < 0.5: im, g = im[:, ::-1].copy(), g[:, ::-1].copy()
        if rng.random() < 0.5: im, g = im[::-1].copy(), g[::-1].copy()
        k = int(rng.integers(4)); im, g = np.rot90(im, k).copy(), np.rot90(g, k).copy()
        xt = torch.from_numpy(im)[None, None].float().to(dev); tt = torch.from_numpy(g).to(dev)
        loss = loss_fn(model(xt), tt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    preds = {n: predict_region(model, crops[n][0], cfg, mean, std, dev) for n in (train_regions + test_regions)}
    def md(names, thr): return float(np.mean([masked_dice(preds[n], crops[n][1], crops[n][2], thr) for n in names]))
    cthr = max(THRS, key=lambda t: md(train_regions, t))
    return md(test_regions, cthr)


t0 = time.time()
curve = {}
for k in K_LIST:
    ds = []
    for trial in range(R):
        rng = np.random.default_rng(1000*k + trial)
        order = list(img_ids); rng.shuffle(order)
        test_imgs, train_imgs = order[:M_TEST], order[M_TEST:]
        pool = [r for im in train_imgs for r in images[im]]
        if len(pool) < k:
            continue
        train_regions = list(rng.choice(pool, size=k, replace=False))
        test_regions = [r for im in test_imgs for r in images[im]]
        d = train_eval(train_regions, test_regions, seed=1000*k+trial)
        if d is not None:
            ds.append(d)
    curve[k] = (float(np.mean(ds)), float(np.std(ds)), len(ds))
    print(f"  k={k:2}  test Dice = {np.mean(ds):.4f} ± {np.std(ds):.4f}  (n={len(ds)})  [{time.time()-t0:.0f}s]", flush=True)

# base (no training) cross-image reference: reset + eval on random 2-image holdouts
model.load_state_dict(init_sd); model.eval()
base_ds = []
for trial in range(R):
    rng = np.random.default_rng(9000+trial); order = list(img_ids); rng.shuffle(order)
    test_regions = [r for im in order[:M_TEST] for r in images[im]]
    tr = [r for im in order[M_TEST:] for r in images[im]]
    preds = {n: predict_region(model, crops[n][0], cfg, mean, std, dev) for n in (tr+test_regions)}
    cthr = max(THRS, key=lambda t: np.mean([masked_dice(preds[n], crops[n][1], crops[n][2], t) for n in tr]))
    base_ds.append(float(np.mean([masked_dice(preds[n], crops[n][1], crops[n][2], cthr) for n in test_regions])))
base_mean = float(np.mean(base_ds))

min_k = next((k for k in K_LIST if curve[k][0] > 0.90), None)
out = {"model": MODEL, "mode": MODE, "R": R, "steps": STEPS, "m_test": M_TEST,
       "base_crossimg": base_mean, "curve": {str(k): curve[k] for k in K_LIST}, "min_k_over_0.90": min_k}
(GT / f"cv_{MODEL}_{MODE}.json").write_text(json.dumps(out, indent=2))
print(f"\n=== {MODEL}/{MODE} labeling-efficiency curve (image-disjoint CV) ===")
print(f"  base (no training, cross-image) = {base_mean:.4f}")
for k in K_LIST:
    m, s, n = curve[k]; print(f"  {k:2} train regions -> {m:.4f} ± {s:.4f}")
print(f"  min #train regions for >0.90: {min_k}")

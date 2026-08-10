"""Deployable GK QuantEM ViT-B checkpoint, head-only adapted on k=2 training crops.

Direct counterpart to train_mitonet_k2_deploy.py for an apples-to-apples baseline comparison: SAME 2
training crops from 2 separate images, SAME image-disjoint held-out set, SAME calibrated-threshold
protocol. Freezes the backbone, trains only neck+decoder (head-only) with the ViT recipe (300 steps,
valid-mask-aware CE+soft-Dice, flips+rot90 aug). Saves an adapted head.pt in the native format
{neck, decoder, encoder_trainable, adapters, conditioner, meta_vocab} — reloadable by
segmentation_training.harness.load_adapted.build_and_load_head — and VERIFIES the saved head
reloads to the same Dice.

Ground-truth layout: see mito_vit.py. Run from training_and_analysis/ with PYTHONPATH=. in the
quantem-segmentation-training environment (../../segmentation_training/environment.yml).

Usage:
    python benchmarking/label_efficiency/train_qem_k2_deploy.py --gt-root DIR --head-dir DIR
        --backbone-dir DIR --out-dir DIR [--model qem_cem] [--train-crops a_0,b_0] [--steps 300]
"""
import argparse, glob, json, os, time
import numpy as np, torch, torch.nn.functional as F

from mito_vit import load_vit, load_crops, masked_dice, SPECS
from segmentation_training.harness.evaluate import predict_region, _round_up
from segmentation_training.harness.dataset import normalize_em
from segmentation_training.harness.load_adapted import build_and_load_head
from segmentation_training.config.schema import load_seg_config
from segmentation_training.harness.run_seg import resolve_encoder, resolve_device

ap = argparse.ArgumentParser(description="Train + save the deployable head-only k=2 ViT checkpoint.")
ap.add_argument("--gt-root", required=True, help="ground-truth crop directory (layout: see mito_vit.py)")
ap.add_argument("--head-dir", required=True, help="directory containing the BASE head.pt for --model")
ap.add_argument("--backbone-dir", required=True,
                help="encoder pretraining run directory (holds checkpoint_index.json)")
ap.add_argument("--out-dir", required=True, help="output directory for the adapted checkpoint + JSON sidecar")
ap.add_argument("--model", default="qem_cem", choices=sorted(SPECS))
ap.add_argument("--train-crops", default="5efb1b60_0,d0ccc5eb_0",
                help="comma-separated crop names drawn from 2 separate images (default: the released checkpoint's crops)")
ap.add_argument("--steps", type=int, default=300)
args = ap.parse_args()

GT = args.gt_root
OUT_DIR = args.out_dir
MODEL = args.model
TRAIN_CROPS = args.train_crops.split(",")
STEPS = args.steps
IGN, SEED = 255, 0
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)

allids = [os.path.basename(f)[:-7] for f in sorted(glob.glob(f"{GT}/*_em.npy"))
          if os.path.exists(f.replace("_em", "_gt")) and os.path.exists(f.replace("_em", "_valid"))]
train_imgs = {c.split("_")[0] for c in TRAIN_CROPS}
eval_ids = [n for n in allids if n.split("_")[0] not in train_imgs]     # image-disjoint held-out
crops = load_crops(GT, set(TRAIN_CROPS) | set(eval_ids))
print(f"[{MODEL}] train {TRAIN_CROPS} (images {sorted(train_imgs)}) | image-disjoint held-out: {eval_ids}", flush=True)

model, cfg, enc, dev = load_vit(MODEL, args.head_dir, args.backbone_dir)
mean, std = enc.image_mean, enc.image_std


def eval_all():
    model.eval()
    preds = {n: predict_region(model, crops[n][0], cfg, mean, std, dev) for n in (TRAIN_CROPS + eval_ids)}
    def md(names, thr):
        vals = [masked_dice(preds[n], crops[n][1], crops[n][2], thr) for n in names]
        vals = [v for v in vals if v is not None]; return float(np.mean(vals)) if vals else None
    cthr = max(THRS, key=lambda t: (md(TRAIN_CROPS, t) or 0))
    per = {n: (None if md([n], cthr) is None else round(md([n], cthr), 4)) for n in eval_ids}
    return {"thr": float(cthr), "train": md(TRAIN_CROPS, cthr), "heldout": md(eval_ids, cthr), "per_heldout": per}


base = eval_all()
print(f"BASE  thr={base['thr']}  train={base['train']:.3f}  held-out={base['heldout']:.3f}", flush=True)

# ---- build training patches from the 2 train crops (valid-aware; ignore outside ROI) ----
patch = int(getattr(model.encoder, "patch_size", 16))
t = _round_up(int(cfg.encoder.tile_size), patch)
pats = []
for n in TRAIN_CROPS:
    em, gt, valid = crops[n]; xn = normalize_em(em, mean, std)
    for y in range(0, max(1, em.shape[0] - t + 1), t // 2):
        for x in range(0, max(1, em.shape[1] - t + 1), t // 2):
            v = valid[y:y + t, x:x + t]
            if v.shape != (t, t) or v.sum() < 0.2 * t * t:
                continue
            g = gt[y:y + t, x:x + t].astype(np.int64); g[v == 0] = IGN
            pats.append((xn[y:y + t, x:x + t].copy(), g))
print(f"  train patches: {len(pats)}", flush=True)

# ---- head-only: freeze everything, train neck+decoder ----
for p in model.parameters(): p.requires_grad_(False)
for m in (model.neck, model.decoder):
    for p in m.parameters(): p.requires_grad_(True)
tp = [p for p in model.parameters() if p.requires_grad]
n_tp = sum(p.numel() for p in tp)
opt = torch.optim.AdamW(tp, lr=1e-4, weight_decay=1e-4)
print(f"  head-only trainable_params={n_tp/1e6:.2f}M", flush=True)


def aug(im, g):
    if rng.random() < 0.5: im, g = im[:, ::-1].copy(), g[:, ::-1].copy()
    if rng.random() < 0.5: im, g = im[::-1].copy(), g[::-1].copy()
    k = int(rng.integers(4)); return np.rot90(im, k).copy(), np.rot90(g, k).copy()


def loss_fn(logits, tgt):
    ce = F.cross_entropy(logits, tgt[None], ignore_index=IGN)
    prob = torch.softmax(logits, 1)[:, 1]
    valid = (tgt != IGN).float()[None]; g = (tgt == 1).float()[None]
    inter = (prob * g * valid).sum(); denom = ((prob + g) * valid).sum()
    return ce + (1 - (2 * inter + 1) / (denom + 1))


model.train(); t0 = time.time()
for step in range(STEPS):
    im, g = aug(*pats[int(rng.integers(len(pats)))])
    logits = model(torch.from_numpy(im)[None, None].float().to(dev))
    loss = loss_fn(logits, torch.from_numpy(g).to(dev))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 75 == 0 or step == STEPS - 1:
        print(f"    step {step:3d} loss {loss.item():.3f}", flush=True)
train_sec = time.time() - t0

ad = eval_all()
print(f"ADAPT thr={ad['thr']}  train={ad['train']:.3f}  held-out={ad['heldout']:.3f}  ({train_sec:.1f}s train)", flush=True)

# ---- save adapted head.pt in native format (carry base encoder-side state; swap neck+decoder) ----
cfg_path, _step = SPECS[MODEL]
base_ckpt = torch.load(f"{args.head_dir}/head.pt", map_location="cpu", weights_only=False)
base_ckpt["neck"] = {k: v.detach().cpu() for k, v in model.neck.state_dict().items()}
base_ckpt["decoder"] = {k: v.detach().cpu() for k, v in model.decoder.state_dict().items()}
ckpt_path = os.path.join(OUT_DIR, f"{MODEL}_GK_headonly_k2.pt")
torch.save(base_ckpt, ckpt_path)

# ---- VERIFY: reload the saved head onto a fresh encoder and re-eval held-out ----
cfg2 = load_seg_config(cfg_path); cfg2.encoder.run_dir = str(args.backbone_dir); cfg2.encoder.checkpoint_step = _step
dev2 = resolve_device("cuda"); enc2, _ = resolve_encoder(cfg2, dev2); enc2.to(dev2)
model2, _v, info = build_and_load_head(cfg2, enc2, ckpt_path, device=dev2)
model2.eval()
preds2 = {n: predict_region(model2, crops[n][0], cfg2, enc2.image_mean, enc2.image_std, dev2) for n in eval_ids}
reload_held = float(np.mean([masked_dice(preds2[n], crops[n][1], crops[n][2], ad["thr"])
                             for n in eval_ids if masked_dice(preds2[n], crops[n][1], crops[n][2], ad["thr"]) is not None]))
print(f"VERIFY reloaded head held-out={reload_held:.3f}  (train-time {ad['heldout']:.3f}; load info {info})", flush=True)

meta = {
    "checkpoint": os.path.basename(ckpt_path),
    "format": "head.pt {neck,decoder,encoder_trainable,adapters,conditioner,meta_vocab}; "
              "load via segmentation_training.harness.load_adapted.build_and_load_head onto the frozen backbone below",
    "model": MODEL, "adaptation": "head_only (backbone frozen; neck+decoder trained)",
    "backbone_run_dir": str(args.backbone_dir), "backbone_checkpoint_step": _step,
    "base_head": f"{args.head_dir}/head.pt", "config": cfg_path,
    "steps": STEPS, "lr": 1e-4, "trainable_M": round(n_tp / 1e6, 3), "tile": t, "patch": patch,
    "train_crops": TRAIN_CROPS, "train_images": sorted(train_imgs),
    "held_out_crops_image_disjoint": eval_ids,
    "calibrated_threshold": ad["thr"],
    "dice": {"base_train": round(base["train"], 4), "base_heldout": round(base["heldout"], 4),
             "adapted_train": round(ad["train"], 4), "adapted_heldout": round(ad["heldout"], 4),
             "reloaded_heldout": round(reload_held, 4)},
    "per_heldout_crop_dice": ad["per_heldout"],
    "train_seconds_on_this_gpu": round(train_sec, 1),
    "note": ("Same 2 crops / same image-disjoint held-out as MitoNet_GK_FTnone_k2 for a matched baseline. "
             "k=2 is one draw; ViT-B head-only k=2 CV = 0.868 +/- 0.049."),
}
json.dump(meta, open(ckpt_path.replace(".pt", ".json"), "w"), indent=2)
print("SAVED", ckpt_path, flush=True)
print(json.dumps(meta["dice"], indent=2))

"""Deployable GK MitoNet checkpoint, fine-tuned on k=2 training crops.

empanada's recommended fine-tune recipe (finetune_layer=none -> encoder frozen, decoders+heads
trained), fp32 (empanada's PointRend scatter_ corrupts under fp16; train fp32). Trains on 2
confirmed-area crops drawn from 2 SEPARATE GK images, then saves the updated model as a drop-in
TorchScript .pth (same format as MitoNet_v1_mini.pth, loadable by any empanada-based pipeline)
plus a JSON sidecar with the calibrated semantic threshold, provenance, and base-vs-adapted Dice
measured image-disjoint on all crops from the OTHER images.

Environment: ../comparators/envs/empanada.yml. Ground truth (in-house, not distributed): flat
directory of <name>_em.npy (uint8 2-D EM crop, 8 nm/px), <name>_inst.npy (instance-labeled mask),
<name>_gt.npy (binary mask), <name>_valid.npy (labeled-ROI mask); crop names are <image>_<index>
with the prefix before the first underscore identifying the source image.

Usage:
    python train_mitonet_k2_deploy.py --gt-root DIR --weights MitoNet_v1_mini.pth --out-dir DIR
        [--ft none|all] [--train-crops a_0,b_0]
"""
import argparse, glob, json, os, time
import numpy as np, torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from empanada.losses import PanopticLoss
from empanada.data.utils import heatmap_and_offsets

ap = argparse.ArgumentParser(description="Train + save the deployable fine-tuned k=2 MitoNet checkpoint.")
ap.add_argument("--gt-root", required=True, help="ground-truth crop directory (layout in module docstring)")
ap.add_argument("--weights", required=True,
                help="base MitoNet_v1_mini.pth TorchScript weights, downloaded from its original source (see ../comparators/WEIGHTS.md)")
ap.add_argument("--out-dir", required=True, help="output directory for the adapted checkpoint + JSON sidecar")
ap.add_argument("--train-crops", default="5efb1b60_0,d0ccc5eb_0",
                help="comma-separated crop names drawn from 2 separate images (default: the released checkpoint's crops)")
ap.add_argument("--ft", default="none", choices=("none", "all"),
                help="finetune_layer: none = encoder frozen (decoders+heads), all = everything")
args = ap.parse_args()

GT = args.gt_root
BASE = args.weights
OUT_DIR = args.out_dir
TRAIN_CROPS = args.train_crops.split(",")
FT = args.ft
MEAN, STD, TILE, BSZ, ITERS, MAXLR = 0.57571, 0.12765, 256, 16, 100, 1e-3
dev = "cuda"
os.makedirs(OUT_DIR, exist_ok=True)
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load(n):
    return (np.load(f"{GT}/{n}_em.npy"), np.load(f"{GT}/{n}_inst.npy"),
            np.load(f"{GT}/{n}_gt.npy"), np.load(f"{GT}/{n}_valid.npy"))


allids = [os.path.basename(f)[:-7] for f in sorted(glob.glob(f"{GT}/*_em.npy"))
          if os.path.exists(f.replace("_em", "_inst"))]
train_imgs = {c.split("_")[0] for c in TRAIN_CROPS}
eval_ids = [n for n in allids if n.split("_")[0] not in train_imgs]   # image-disjoint held-out
crops = {n: load(n) for n in sorted(set(TRAIN_CROPS) | set(eval_ids))}
print(f"train crops {TRAIN_CROPS} (images {sorted(train_imgs)}) | image-disjoint held-out: {eval_ids}", flush=True)

model = torch.jit.load(BASE, map_location=dev).to(dev)
loss_fn = PanopticLoss(ce_weight=1, mse_weight=200, l1_weight=0.01, top_k_percent=0.2).to(dev)
TF = A.Compose([
    A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0), A.PadIfNeeded(TILE, TILE, border_mode=0),
    A.RandomCrop(TILE, TILE), A.Rotate(limit=180, border_mode=0), A.RandomBrightnessContrast(0.3, 0.3),
    A.HorizontalFlip(), A.VerticalFlip(), A.Normalize(mean=(MEAN,), std=(STD,), max_pixel_value=255.0), ToTensorV2()])


def sample_batch(regions, rng):
    ims, sems, hms, offs = [], [], [], []
    for _ in range(BSZ):
        n = regions[int(rng.integers(len(regions)))]
        em, inst, _g, _v = crops[n]
        o = TF(image=em[..., None], mask=inst.astype(np.int32)); m = np.asarray(o["mask"])
        hm, off = heatmap_and_offsets(m.astype(np.int32))
        ims.append(o["image"]); sems.append((m > 0).astype(np.float32)); hms.append(hm); offs.append(off)
    x = torch.stack(ims).float().to(dev)
    return x, {"sem": torch.from_numpy(np.stack(sems)).float().to(dev),
               "ctr_hmp": torch.from_numpy(np.stack(hms)).float().to(dev),
               "offsets": torch.from_numpy(np.stack(offs)).float().to(dev)}


def predict_sem(em):
    H, W = em.shape
    Hp, Wp = max(TILE, ((H + 127) // 128) * 128), max(TILE, ((W + 127) // 128) * 128)
    emp = np.zeros((Hp, Wp), np.uint8); emp[:H, :W] = em
    acc = np.zeros((Hp, Wp), np.float32); cnt = np.zeros((Hp, Wp), np.float32)
    ys = list(range(0, Hp - TILE + 1, 192)) or [0]; xs = list(range(0, Wp - TILE + 1, 192)) or [0]
    if ys[-1] != Hp - TILE: ys.append(Hp - TILE)
    if xs[-1] != Wp - TILE: xs.append(Wp - TILE)
    model.eval()
    for y in ys:
        for x0 in xs:
            t = (emp[y:y + TILE, x0:x0 + TILE].astype(np.float32) / 255.0 - MEAN) / STD
            with torch.no_grad():
                out = model(torch.from_numpy(t)[None, None].float().to(dev))
            p = torch.sigmoid(out["sem_logits"])[0, 0].cpu().numpy()
            acc[y:y + TILE, x0:x0 + TILE] += p; cnt[y:y + TILE, x0:x0 + TILE] += 1
    return (acc / np.maximum(cnt, 1))[:H, :W]


def mdice(pred, gt, valid, thr):
    p = ((pred >= thr) & (valid > 0)); g = ((gt > 0) & (valid > 0))
    d = p.sum() + g.sum(); return None if d == 0 else 2.0 * (p & g).sum() / d


def scored(names, thr):
    vals = [mdice(preds[n], crops[n][2], crops[n][3], thr) for n in names]
    vals = [v for v in vals if v is not None]; return float(np.mean(vals)) if vals else None


# --- base (pre-adaptation) reference, image-disjoint ---
preds = {n: predict_sem(crops[n][0]) for n in crops}
base_thr = max(THRS, key=lambda t: (scored(TRAIN_CROPS, t) or 0))
base_train, base_held = scored(TRAIN_CROPS, base_thr), scored(eval_ids, base_thr)
print(f"BASE  thr={base_thr}  train={base_train:.3f}  held-out={base_held:.3f}", flush=True)

# --- fine-tune (empanada recipe) ---
for nm, p in model.named_parameters():
    p.requires_grad_(FT == "all" or not nm.startswith("encoder."))
tp = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(tp, lr=MAXLR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAXLR, total_steps=ITERS, pct_start=0.3)
rng = np.random.default_rng(0); model.train()
t0 = time.time()
for it in range(ITERS):
    x, tgt = sample_batch(TRAIN_CROPS, rng)
    out = model(x); loss, _ = loss_fn(out, tgt)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
train_sec = time.time() - t0

# --- adapted eval + threshold calibration (on the 2 train crops) ---
preds = {n: predict_sem(crops[n][0]) for n in crops}
adapt_thr = max(THRS, key=lambda t: (scored(TRAIN_CROPS, t) or 0))
adapt_train, adapt_held = scored(TRAIN_CROPS, adapt_thr), scored(eval_ids, adapt_thr)
per_held = {n: round(mdice(preds[n], crops[n][2], crops[n][3], adapt_thr), 4)
            for n in eval_ids if mdice(preds[n], crops[n][2], crops[n][3], adapt_thr) is not None}
print(f"ADAPT thr={adapt_thr}  train={adapt_train:.3f}  held-out={adapt_held:.3f}  ({train_sec:.1f}s train)", flush=True)

model.eval()
ckpt = os.path.join(OUT_DIR, f"MitoNet_GK_FT{FT}_k2.pth")
torch.jit.save(model, ckpt)
meta = {
    "checkpoint": os.path.basename(ckpt),
    "format": "TorchScript (torch.jit.save) — drop-in replacement for MitoNet_v1_mini.pth in empanada-based pipelines",
    "base_model": "MitoNet_v1_mini.pth",
    "recipe": "empanada finetune", "finetune_layer": FT, "iters": ITERS, "bsz": BSZ, "max_lr": MAXLR,
    "precision": "fp32 (no autocast; empanada PointRend requires it)",
    "norm_mean": MEAN, "norm_std": STD, "tile": TILE,
    "train_crops": TRAIN_CROPS, "train_images": sorted(train_imgs),
    "held_out_crops_image_disjoint": eval_ids,
    "calibrated_semantic_threshold": float(adapt_thr),
    "dice": {"base_train": round(base_train, 4), "base_heldout": round(base_held, 4),
             "adapted_train": round(adapt_train, 4), "adapted_heldout": round(adapt_held, 4)},
    "per_heldout_crop_dice": per_held,
    "train_seconds_on_this_gpu": round(train_sec, 1),
    "note": ("k=2 is a single draw from a high-variance regime (MitoNet none CV at k=2 = 0.697 +/- 0.183); "
             "this checkpoint's held-out number is one sample, not the CV mean. Semantic threshold is for "
             "reproducing the masked-Dice eval; empanada inference applies its own postprocessing."),
}
json.dump(meta, open(ckpt.replace(".pth", ".json"), "w"), indent=2)
print("SAVED", ckpt, flush=True)
print(json.dumps(meta["dice"], indent=2))

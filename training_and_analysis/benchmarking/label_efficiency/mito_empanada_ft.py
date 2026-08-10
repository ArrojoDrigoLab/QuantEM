"""Fine-tune MitoNet with empanada's OWN recommended config (environment:
../comparators/envs/empanada.yml). Uses empanada.losses.PanopticLoss (mse 200 / l1 .01 /
bootstrap-BCE top20%) + empanada's heatmap_and_offsets panoptic targets on the trainable
MitoNet_v1_mini ScriptModule, with the shipped finetune recipe (finetune_layer=none head-only
analogue / all; bsz16; RandomScale+RandomCrop256; Rotate180; RBC .3; flips; OneCycleLR AdamW wd.1
max_lr 1e-3; iters). Trains fp32 throughout: empanada's PointRend scatter_ corrupts under fp16.
Image-disjoint CV over #train regions, mirroring the ViT harness. Eval = masked semantic Dice
(calibrated on train), same metric as the ViT runs. Writes cv_<model-name>_<ft>.json into
--gt-root.

Ground truth (in-house, not distributed): flat directory of <name>_em.npy (uint8 2-D EM crop,
8 nm/px), <name>_inst.npy (instance-labeled mask), <name>_gt.npy (binary mask), <name>_valid.npy
(labeled-ROI mask); crop names are <image>_<index> with the prefix before the first underscore
identifying the source image.

Usage:
    python mito_empanada_ft.py <none|stage4|all> --gt-root DIR --weights MitoNet_v1_mini.pth
        [--iters 100] [--trials 8] [--max-lr 1e-3] [--m-test 2] [--k-list 1,2,3,4,6,8,10]
"""
import argparse, glob, json, os, time
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from empanada.losses import PanopticLoss
from empanada.data.utils import heatmap_and_offsets

ap = argparse.ArgumentParser(description="Image-disjoint CV labeling-efficiency curve (MitoNet, empanada recipe).")
ap.add_argument("ft", choices=("none", "stage4", "all"),
                help="finetune_layer: none = encoder frozen (decoders+heads), stage4 = also encoder layer4, all = everything")
ap.add_argument("--iters", type=int, default=100, help="fine-tune iterations per trial")
ap.add_argument("--trials", type=int, default=8, help="repeats R per k (random train/test image swaps)")
ap.add_argument("--max-lr", type=float, default=1e-3, help="OneCycleLR max learning rate")
ap.add_argument("--gt-root", required=True, help="ground-truth crop directory (layout in module docstring)")
ap.add_argument("--weights", required=True,
                help="MitoNet_v1_mini.pth TorchScript weights, downloaded from its original source (see ../comparators/WEIGHTS.md)")
ap.add_argument("--model-name", default="mitonet", help="label used in the output filename and JSON")
ap.add_argument("--mean", type=float, default=0.57571, help="normalization mean (MitoNet_v1_mini stats)")
ap.add_argument("--std", type=float, default=0.12765, help="normalization std (MitoNet_v1_mini stats)")
ap.add_argument("--m-test", type=int, default=2, help="hold out this many whole images as test")
ap.add_argument("--k-list", default="1,2,3,4,6,8,10", help="comma-separated #training-region counts")
args = ap.parse_args()

GT = args.gt_root
MODEL_PATH = args.weights
MODEL_NAME = args.model_name
MEAN = args.mean
STD = args.std
FT = args.ft
ITERS = args.iters
R = args.trials
MAXLR = args.max_lr
BSZ, TILE = 16, 256
M_TEST = args.m_test
K_LIST = [int(x) for x in args.k_list.split(",")]
dev = "cuda"

crops = {}
for emf in sorted(glob.glob(f"{GT}/*_em.npy")):
    n = os.path.basename(emf)[:-7]
    if os.path.exists(f"{GT}/{n}_inst.npy"):
        crops[n] = (np.load(emf), np.load(f"{GT}/{n}_inst.npy"), np.load(f"{GT}/{n}_gt.npy"), np.load(f"{GT}/{n}_valid.npy"))
images = defaultdict(list)
for n in crops: images[n.split("_")[0]].append(n)
img_ids = sorted(images)
print(f"[MitoNet/{FT}] {len(crops)} regions / {len(img_ids)} images; ITERS={ITERS} R={R} bsz={BSZ} max_lr={MAXLR}", flush=True)

model = torch.jit.load(MODEL_PATH, map_location=dev).to(dev)
init_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
loss_fn = PanopticLoss(ce_weight=1, mse_weight=200, l1_weight=0.01, top_k_percent=0.2).to(dev)

TF = A.Compose([
    A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
    A.PadIfNeeded(TILE, TILE, border_mode=0),
    A.RandomCrop(TILE, TILE),
    A.Rotate(limit=180, border_mode=0),
    A.RandomBrightnessContrast(0.3, 0.3),
    A.HorizontalFlip(), A.VerticalFlip(),
    A.Normalize(mean=(MEAN,), std=(STD,), max_pixel_value=255.0),
    ToTensorV2(),
])


def set_trainable():
    for n, p in model.named_parameters():
        if FT == "all":
            p.requires_grad_(True)
        elif FT == "stage4":
            p.requires_grad_(not n.startswith("encoder.") or n.startswith("encoder.layer4"))
        else:  # none: decoders + heads only, encoder frozen
            p.requires_grad_(not n.startswith("encoder."))


def sample_batch(regions, rng):
    ims, sems, hms, offs = [], [], [], []
    for _ in range(BSZ):
        n = regions[int(rng.integers(len(regions)))]
        em, inst, _gt, valid = crops[n]
        # instances outside the labeled ROI are already zeroed in the input; pass instance map as albumentations mask
        o = TF(image=em[..., None], mask=inst.astype(np.int32))
        img = o["image"]; m = o["mask"].numpy() if hasattr(o["mask"], "numpy") else np.asarray(o["mask"])
        sem = (m > 0).astype(np.float32)
        hm, off = heatmap_and_offsets(m.astype(np.int32))
        ims.append(img); sems.append(sem); hms.append(hm); offs.append(off)
    x = torch.stack(ims).float().to(dev)
    tgt = {"sem": torch.from_numpy(np.stack(sems)).float().to(dev),
           "ctr_hmp": torch.from_numpy(np.stack(hms)).float().to(dev),
           "offsets": torch.from_numpy(np.stack(offs)).float().to(dev)}
    return x, tgt


def predict_sem(em):
    """Sliding-window 256 semantic prob (sigmoid of sem_logits), Hann-free avg blend."""
    H, W = em.shape
    Hp, Wp = max(TILE, ((H+127)//128)*128), max(TILE, ((W+127)//128)*128)
    emp = np.zeros((Hp, Wp), np.uint8); emp[:H, :W] = em
    acc = np.zeros((Hp, Wp), np.float32); cnt = np.zeros((Hp, Wp), np.float32)
    st = 192
    ys = list(range(0, Hp-TILE+1, st)) or [0]; xs = list(range(0, Wp-TILE+1, st)) or [0]
    if ys[-1] != Hp-TILE: ys.append(Hp-TILE)
    if xs[-1] != Wp-TILE: xs.append(Wp-TILE)
    model.eval()
    for y in ys:
        for x0 in xs:
            t = (emp[y:y+TILE, x0:x0+TILE].astype(np.float32)/255.0 - MEAN)/STD
            xt = torch.from_numpy(t)[None, None].float().to(dev)
            with torch.no_grad():
                out = model(xt)
            p = torch.sigmoid(out["sem_logits"])[0, 0].cpu().numpy()
            acc[y:y+TILE, x0:x0+TILE] += p; cnt[y:y+TILE, x0:x0+TILE] += 1
    return (acc/np.maximum(cnt, 1))[:H, :W]


def mdice(pred, gt, valid, thr):
    p = ((pred >= thr) & (valid > 0)); g = ((gt > 0) & (valid > 0))
    d = p.sum()+g.sum(); return None if d == 0 else 2.0*(p & g).sum()/d


THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def train_eval(train_regions, test_regions, seed):
    model.load_state_dict(init_sd); set_trainable()
    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=MAXLR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAXLR, total_steps=ITERS, pct_start=0.3)
    rng = np.random.default_rng(seed)
    model.train()
    for it in range(ITERS):
        x, tgt = sample_batch(train_regions, rng)
        # empanada's PointRend scatter_ corrupts under fp16; train fp32
        out = model(x)
        loss, _ = loss_fn(out, tgt)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    preds = {n: predict_sem(crops[n][0]) for n in (train_regions+test_regions)}
    def md(names, thr): return float(np.mean([mdice(preds[n], crops[n][2], crops[n][3], thr) for n in names]))
    cthr = max(THRS, key=lambda t: md(train_regions, t))
    return md(test_regions, cthr)


# base (no training) cross-image reference
model.load_state_dict(init_sd)
base_ds = []
for trial in range(R):
    rng = np.random.default_rng(9000+trial); order = list(img_ids); rng.shuffle(order)
    te = [r for im in order[:M_TEST] for r in images[im]]; tr = [r for im in order[M_TEST:] for r in images[im]]
    preds = {n: predict_sem(crops[n][0]) for n in (tr+te)}
    cthr = max(THRS, key=lambda t: np.mean([mdice(preds[n], crops[n][2], crops[n][3], t) for n in tr]))
    base_ds.append(float(np.mean([mdice(preds[n], crops[n][2], crops[n][3], cthr) for n in te])))
base_mean = float(np.mean(base_ds))
print(f"  base (no-train, cross-image) = {base_mean:.4f}", flush=True)

t0 = time.time(); curve = {}
for k in K_LIST:
    ds, times = [], []
    for trial in range(R):
        rng = np.random.default_rng(1000*k+trial); order = list(img_ids); rng.shuffle(order)
        test_imgs, train_imgs = order[:M_TEST], order[M_TEST:]
        pool = [r for im in train_imgs for r in images[im]]
        if len(pool) < k: continue
        tr = list(rng.choice(pool, size=k, replace=False)); te = [r for im in test_imgs for r in images[im]]
        tt = time.time(); d = train_eval(tr, te, seed=1000*k+trial); times.append(time.time()-tt)
        ds.append(d)
    curve[k] = (float(np.mean(ds)), float(np.std(ds)), len(ds), float(np.mean(times)))
    print(f"  k={k:2} Dice={np.mean(ds):.4f} ± {np.std(ds):.4f} (n={len(ds)}, {np.mean(times):.0f}s/run) [{time.time()-t0:.0f}s]", flush=True)

min_k = next((k for k in K_LIST if curve[k][0] > 0.90), None)
out = {"model": MODEL_NAME, "recipe": "empanada", "finetune_layer": FT, "iters": ITERS, "max_lr": MAXLR,
       "R": R, "base_crossimg": base_mean, "curve": {str(k): curve[k] for k in K_LIST}, "min_k_over_0.90": min_k,
       "sec_per_run": float(np.mean([curve[k][3] for k in K_LIST]))}
open(f"{GT}/cv_{MODEL_NAME}_{FT}.json", "w").write(json.dumps(out, indent=2))
print(f"\n=== MitoNet empanada fine-tune (finetune_layer={FT}) labeling-efficiency ===")
print(f"  base cross-image = {base_mean:.4f}")
for k in K_LIST:
    m, s, n, tm = curve[k]; print(f"  {k:2} regions -> {m:.4f} ± {s:.4f}   ({tm:.0f}s/run)")
print(f"  min #regions for >0.90: {min_k}   | mean {out['sec_per_run']:.0f}s per fine-tune")

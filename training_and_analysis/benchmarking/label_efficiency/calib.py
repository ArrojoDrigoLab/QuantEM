"""Cheapest adaptation rung: per-dataset threshold calibration.
Fit ONE scalar (fg threshold) on the train crops' saved base predictions, apply to the test crops.
No GPU, no training. Also reports the per-crop oracle threshold (ceiling). Reads split.json,
pred_<model>_<name>.npy (written by mito_vit.py), <name>_gt.npy and <name>_valid.npy from
--gt-root.

Usage: python calib.py --gt-root DIR"""
import argparse, json
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser(description="Per-dataset threshold calibration from saved base predictions.")
ap.add_argument("--gt-root", required=True, help="ground-truth crop directory holding the saved pred_*.npy")
GT = Path(ap.parse_args().gt_root)
split = json.loads((GT / "split.json").read_text())
train, test = split["train"], split["test"]


def dice(pred_bin, gt, valid):
    p = (pred_bin & (valid > 0)).astype(np.uint8)
    g = ((gt > 0) & (valid > 0)).astype(np.uint8)
    d = int(p.sum() + g.sum())
    return None if d == 0 else 2.0 * int((p & g).sum()) / d


def load(model, name):
    pr = GT / f"pred_{model}_{name}.npy"
    if not pr.exists():
        return None
    return np.load(pr), np.load(GT / f"{name}_gt.npy"), np.load(GT / f"{name}_valid.npy")


THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)
for model in ("qem_cem", "omni_cem"):
    data = {n: load(model, n) for n in (train + test)}
    if any(v is None for v in data.values()):
        print(f"[{model}] missing preds; skip"); continue

    def mean_dice(names, thr):
        ds = [dice(p >= thr, g, v) for (p, g, v) in (data[n] for n in names)]
        ds = [d for d in ds if d is not None]
        return float(np.mean(ds)) if ds else None

    base_test = mean_dice(test, 0.5)
    # calibrate threshold on TRAIN
    best_thr, best_train = max(((t, mean_dice(train, t)) for t in THRS), key=lambda kv: kv[1])
    cal_test = mean_dice(test, best_thr)
    # oracle per-crop test threshold (ceiling, not achievable without test labels)
    oracle = np.mean([max(dice(p >= t, g, v) or 0 for t in THRS) for (p, g, v) in (data[n] for n in test)])
    print(f"[{model}] base(thr0.5) test={base_test:.4f} | calibrated thr={best_thr} "
          f"(train {best_train:.4f}) -> test={cal_test:.4f}  (Δ{cal_test-base_test:+.4f}) | per-crop-oracle test={oracle:.4f}")
    for n in test:
        p, g, v = data[n]
        print(f"      {n}: base={dice(p>=0.5,g,v):.4f} calibrated={dice(p>=best_thr,g,v):.4f}")

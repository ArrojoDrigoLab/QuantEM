"""Shared loader + eval utilities for the released mitochondria ViT models.
SPECS maps each model name to its released configuration and encoder checkpoint step; the encoder
run directory and the head directory are always supplied by the caller. Exposes load_vit,
predict_fg, masked_dice, and load_crops/split so every experiment in this folder (base eval,
cross-validation, LoRA/head fine-tuning) shares one code path.

Ground-truth crops (in-house immuno-EM annotations, not distributed) are read from a flat
directory of numpy arrays:
    <name>_em.npy     uint8 2-D EM crop (8 nm/px)
    <name>_gt.npy     binary organelle mask
    <name>_valid.npy  labeled-ROI mask (1 = annotated pixel)
    split.json        {"train": [names], "test": [names]}
Crop names are <image>_<index>; the text before the first underscore identifies the source image
(image-disjoint splits group crops by this prefix).

Requires the segmentation_training package importable: run from training_and_analysis/ with
PYTHONPATH=. in the quantem-segmentation-training environment
(../../segmentation_training/environment.yml).

Usage (writes pred_<model>_<name>.npy + eval_<model>.json into --gt-root):
    python benchmarking/label_efficiency/mito_vit.py <qem_cem|omni_cem>
        --gt-root DIR --head-dir DIR --backbone-dir DIR
"""
import glob, json
import numpy as np
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "segmentation_training" / "configs" / "released_models"

# name -> (released config yaml, encoder checkpoint_step)
SPECS = {
    "omni_cem": (str(_CONFIG_DIR / "mitochondria_omniem.yaml"), 0),
    "qem_cem":  (str(_CONFIG_DIR / "mitochondria_quantem.yaml"), 674999),
}


def load_vit(name, head_dir, backbone_dir, device="cuda"):
    from segmentation_training.config.schema import load_seg_config
    from segmentation_training.harness.run_seg import resolve_encoder, resolve_device
    from segmentation_training.harness.load_adapted import build_and_load_head
    cfg_path, step = SPECS[name]
    cfg = load_seg_config(cfg_path)
    cfg.encoder.run_dir = str(backbone_dir)
    cfg.encoder.checkpoint_step = step
    dev = resolve_device(device)
    enc, _ = resolve_encoder(cfg, dev); enc.to(dev)
    model, *_ = build_and_load_head(cfg, enc, f"{head_dir}/head.pt", device=dev)
    model.eval()
    return model, cfg, enc, dev


def predict_fg(model, cfg, enc, dev, em):
    from segmentation_training.harness.evaluate import predict_region
    return predict_region(model, em, cfg, enc.image_mean, enc.image_std, dev)


def masked_dice(pred, gt, valid, thr=0.5):
    p = ((np.asarray(pred) >= thr) & (valid > 0)).astype(np.uint8)
    g = ((np.asarray(gt) > 0) & (valid > 0)).astype(np.uint8)
    inter = int((p & g).sum()); denom = int(p.sum() + g.sum())
    return None if denom == 0 else 2.0 * inter / denom


def load_crops(gt_root, names=None):
    gt_root = Path(gt_root)
    crops = {}
    for emf in sorted(glob.glob(str(gt_root / "*_em.npy"))):
        name = Path(emf).name[:-7]
        if names is not None and name not in names:
            continue
        gtf, vf = gt_root / f"{name}_gt.npy", gt_root / f"{name}_valid.npy"
        if gtf.exists() and vf.exists():
            crops[name] = (np.load(emf), np.load(gtf), np.load(vf))
    return crops


def split(gt_root):
    s = json.loads((Path(gt_root) / "split.json").read_text())
    return s["train"], s["test"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate a released mitochondria ViT model on the GT crops.")
    ap.add_argument("model", choices=sorted(SPECS), help="which released model to load")
    ap.add_argument("--gt-root", required=True, help="ground-truth crop directory (layout in module docstring)")
    ap.add_argument("--head-dir", required=True, help="directory containing the trained head.pt for this model")
    ap.add_argument("--backbone-dir", required=True,
                    help="encoder pretraining run directory (holds checkpoint_index.json)")
    args = ap.parse_args()
    name = args.model
    GT = Path(args.gt_root)
    train, test = split(GT)
    model, cfg, enc, dev = load_vit(name, args.head_dir, args.backbone_dir)
    print(f"[{name}] loaded  arch={getattr(enc,'arch','?')} fg_thr={float(getattr(cfg.eval,'fg_threshold',0.5))}", flush=True)
    crops = load_crops(GT)
    rows = {}
    for cname, (em, gt, valid) in crops.items():
        fg = predict_fg(model, cfg, enc, dev, em)
        d = masked_dice(fg, gt, valid)
        rows[cname] = d
        tag = "TEST " if cname in test else "train"
        print(f"  {tag} {cname:14} dice={d if d is None else round(d,4)}", flush=True)
        np.save(GT / f"pred_{name}_{cname}.npy", fg.astype(np.float32))
    td = [rows[c] for c in test if rows.get(c) is not None]
    trd = [rows[c] for c in train if rows.get(c) is not None]
    print(f"\n[{name}] TEST mean dice = {np.mean(td):.4f}   (train mean = {np.mean(trd):.4f})")
    out = GT / f"eval_{name}.json"
    out.write_text(json.dumps({"model": name, "rows": rows,
                               "test_mean": float(np.mean(td)), "train_mean": float(np.mean(trd))}, indent=2))
    print(f"  wrote {out}")

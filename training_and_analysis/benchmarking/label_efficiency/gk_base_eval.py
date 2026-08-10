"""Zero-label base mito Dice on the GK GT crops, inside the labeled ROI (valid mask).
Bases:
  qem | omni : released mitochondria ViT models loaded through segmentation_training; requires
               --head-dir and --backbone-dir. Run from training_and_analysis/ with PYTHONPATH=.
               in the quantem-segmentation-training environment
               (../../segmentation_training/environment.yml).
  mitonet    : MitoNet_v1_mini.pth TorchScript weights (--weights; see ../comparators/WEIGHTS.md),
               sliding-window semantic probability at native resolution
               (environment: ../comparators/envs/empanada.yml).
Writes eval_<base>.json into --gt-root. `gk_base_eval.py selftest` checks the Dice math with no
model. Aggregate across the JSONs to compare bases.

Ground truth (in-house, not distributed): flat directory of <name>_em.npy (uint8 2-D EM crop,
8 nm/px), <name>_gt.npy (binary mask), <name>_valid.npy (labeled-ROI mask); crop names are
<image>_<index> with the prefix before the first underscore identifying the source image."""
import glob, json
import numpy as np
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "segmentation_training" / "configs" / "released_models"


def masked_dice(pred, gt, valid, thr=0.5):
    p = ((np.asarray(pred) >= thr) & (valid > 0)).astype(np.uint8)
    g = ((np.asarray(gt) > 0) & (valid > 0)).astype(np.uint8)
    inter = int((p & g).sum()); denom = int(p.sum() + g.sum())
    if denom == 0:
        return None                      # no fg either way inside ROI -> undefined
    return 2.0 * inter / denom


def load_crops(gt_root):
    gt_root = Path(gt_root)
    crops = []
    for emf in sorted(glob.glob(str(gt_root / "*_em.npy"))):
        name = Path(emf).name[:-7]
        gtf, vf = gt_root / f"{name}_gt.npy", gt_root / f"{name}_valid.npy"
        if gtf.exists() and vf.exists():
            crops.append((name, np.load(emf), np.load(gtf), np.load(vf)))
    return crops


def selftest():
    # fabricate: valid=full, gt=disk; perfect pred -> 1.0, empty pred -> 0.0/None, half -> known
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    gt = (((yy - 32) ** 2 + (xx - 32) ** 2) < 12 ** 2).astype(np.uint8)
    valid = np.ones((H, W), np.uint8)
    assert abs(masked_dice(gt.astype(float), gt, valid) - 1.0) < 1e-9, "perfect should be 1.0"
    assert masked_dice(np.zeros((H, W)), gt, valid) == 0.0, "empty pred vs fg gt -> 0"
    assert masked_dice(np.zeros((H, W)), np.zeros((H, W)), valid) is None, "no fg both -> None"
    # valid-mask must exclude outside-ROI: pred fires outside ROI, should not count
    pred = np.zeros((H, W)); pred[:, :] = 1.0
    v2 = np.zeros((H, W), np.uint8); v2[gt > 0] = 1        # ROI == gt region
    assert abs(masked_dice(pred, gt, v2) - 1.0) < 1e-9, "outside-ROI pred ignored"
    print("selftest OK: masked_dice perfect=1.0 empty=0.0 nofg=None roi-masking=OK")


def run(base, args):
    GT = Path(args.gt_root)
    crops = load_crops(GT)
    if not crops:
        print(f"[{base}] no GT crops in {GT}."); return
    preds_fn = None
    if base in ("qem", "omni"):
        import torch  # noqa
        from segmentation_training.config.schema import load_seg_config
        from segmentation_training.harness.run_seg import resolve_encoder, resolve_device
        from segmentation_training.harness.load_adapted import build_and_load_head
        from segmentation_training.harness.evaluate import predict_region
        spec = {"qem": ("mitochondria_quantem.yaml", 674999),
                "omni": ("mitochondria_omniem.yaml", 0)}[base]
        cfg = load_seg_config(str(_CONFIG_DIR / spec[0]))
        cfg.encoder.run_dir = str(args.backbone_dir); cfg.encoder.checkpoint_step = spec[1]
        dev = resolve_device("cuda"); enc, _ = resolve_encoder(cfg, dev); enc.to(dev)
        model, *_ = build_and_load_head(cfg, enc, f"{args.head_dir}/head.pt", device=dev)
        model.eval()
        preds_fn = lambda em: predict_region(model, em, cfg, enc.image_mean, enc.image_std, dev)
    elif base == "mitonet":
        import torch
        TILE, MEAN, STD = 256, 0.57571, 0.12765     # MitoNet_v1_mini normalization stats
        model = torch.jit.load(args.weights, map_location="cuda").to("cuda")
        model.eval()

        def _mn(em):
            H, W = em.shape
            Hp, Wp = max(TILE, ((H + 127) // 128) * 128), max(TILE, ((W + 127) // 128) * 128)
            emp = np.zeros((Hp, Wp), np.uint8); emp[:H, :W] = em
            acc = np.zeros((Hp, Wp), np.float32); cnt = np.zeros((Hp, Wp), np.float32)
            ys = list(range(0, Hp - TILE + 1, 192)) or [0]; xs = list(range(0, Wp - TILE + 1, 192)) or [0]
            if ys[-1] != Hp - TILE: ys.append(Hp - TILE)
            if xs[-1] != Wp - TILE: xs.append(Wp - TILE)
            for y in ys:
                for x0 in xs:
                    t = (emp[y:y + TILE, x0:x0 + TILE].astype(np.float32) / 255.0 - MEAN) / STD
                    with torch.no_grad():
                        out = model(torch.from_numpy(t)[None, None].float().to("cuda"))
                    p = torch.sigmoid(out["sem_logits"])[0, 0].cpu().numpy()
                    acc[y:y + TILE, x0:x0 + TILE] += p; cnt[y:y + TILE, x0:x0 + TILE] += 1
            return (acc / np.maximum(cnt, 1))[:H, :W]
        preds_fn = _mn
    else:
        raise SystemExit(f"unknown base {base}")

    rows = []
    for name, em, gt, valid in crops:
        pred = preds_fn(em)
        d = masked_dice(pred, gt, valid)
        rows.append({"name": name, "dice": d, "gt_px": int((gt > 0).sum()), "valid_px": int((valid > 0).sum())})
        print(f"  {name:26} dice={('%.3f' % d) if d is not None else 'None':>6} "
              f"gt_px={int((gt>0).sum())}", flush=True)
    valid_d = [r["dice"] for r in rows if r["dice"] is not None]
    mean = float(np.mean(valid_d)) if valid_d else None
    print(f"\n[{base}] crops={len(rows)} mean_dice={mean}")
    (GT / f"eval_{base}.json").write_text(json.dumps({"base": base, "mean_dice": mean, "rows": rows}, indent=2))
    print(f"  wrote {GT / f'eval_{base}.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Zero-label base Dice on the GK GT crops.")
    ap.add_argument("base", nargs="?", default="selftest", choices=("qem", "omni", "mitonet", "selftest"))
    ap.add_argument("--gt-root", help="ground-truth crop directory (required except for selftest)")
    ap.add_argument("--head-dir", help="qem|omni: directory containing the trained head.pt")
    ap.add_argument("--backbone-dir",
                    help="qem|omni: encoder pretraining run directory (holds checkpoint_index.json)")
    ap.add_argument("--weights", help="mitonet: MitoNet_v1_mini.pth TorchScript weights (see ../comparators/WEIGHTS.md)")
    args = ap.parse_args()
    if args.base == "selftest":
        selftest()
    else:
        if not args.gt_root:
            raise SystemExit("--gt-root is required")
        if args.base in ("qem", "omni") and not (args.head_dir and args.backbone_dir):
            raise SystemExit("--head-dir and --backbone-dir are required for qem|omni")
        if args.base == "mitonet" and not args.weights:
            raise SystemExit("--weights is required for mitonet")
        run(args.base, args)

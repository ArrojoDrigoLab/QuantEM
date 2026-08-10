"""DeepContact ER segmentation (Liu et al. 2022) on the benchmark ER test crops.
Faithful re-implementation of DeepContact's ER inference (add_er / main_predict):
  * grayscale->RGB, resolution-normalize to ~10 nm/px (crop_size = 1024*10/res),
  * 1024x1024 sliding windows, smp encoder=resnext101_32x8d, sigmoid > 0.3,
  * model variant matched to modality (TEM->tem, FIB/volume->cell, SEM/SBF->sem).
Runs in the `deepcontact_er` env (torch + smp 0.5.0). PM-cropping (needs manual PM masks)
is skipped -> honest out-of-the-box application.

Usage: python run_deepcontact_er.py er --weights W --seg-root S --results-root R
"""
import os, sys, argparse
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from run_utils import run_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
THRESH = 0.3
TILE = 1024
OVERLAP = 0.2
MAXSIDE = 2048          # cap resolution-normalized size (avoid runaway upsampling)
DEV = "cuda"


def build_er_model(path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    is_unet = any("decoder.blocks" in k for k in sd)
    Arch = smp.Unet if is_unet else smp.FPN
    model = Arch(encoder_name="resnext101_32x8d", encoder_weights=None, classes=1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    arch = "Unet" if is_unet else "FPN"
    print(f"  loaded {os.path.basename(path)} as {arch} "
          f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    return model.eval().to(DEV)


# Fair variant per dataset: TEM->tem; SEM-family cultured cells->cell (U-2 OS model);
# SEM-family tissue->sem (Sertoli tissue model).
ER_VARIANT = {
    "empiar_12885_aive": "cell",           # FIB-SEM, cultured + muscle cells
    "empiar_10994_hela_sbfsem": "cell",    # SBF-SEM, HeLa (cultured)
    "deepcontact_tem": "tem",              # TEM, COS-7
    "empiar_13156_hela_stard3_er": "cell", # FIB-SEM, HeLa (cultured)
    "lab_islet_liver_er": "sem",           # SEM, islet + liver tissue
}
def variant_for(spec):
    return ER_VARIANT.get(spec.dataset, "cell")


@torch.no_grad()
def infer_tile(model, tile_rgb):
    x = tile_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEV)
    logit = model(x)
    return (torch.sigmoid(logit)[0, 0] > THRESH).cpu().numpy()


def predict_full(model, img_gray, voxel_nm):
    rgb = np.repeat(img_gray[..., None], 3, axis=2)
    H0, W0 = img_gray.shape
    res = voxel_nm if (voxel_nm and voxel_nm > 0) else 10.0
    scale = res / 10.0                       # native -> 10 nm/px
    Hs = max(1, int(round(H0 * scale)))
    Ws = max(1, int(round(W0 * scale)))
    if max(Hs, Ws) > MAXSIDE:
        c = MAXSIDE / max(Hs, Ws); Hs, Ws = max(1, int(Hs * c)), max(1, int(Ws * c))
    rs = cv2.resize(rgb, (Ws, Hs), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.zeros((Hs, Ws), np.float32)
    step = max(1, int(TILE * (1 - OVERLAP)))
    ys = list(range(0, max(1, Hs - TILE + 1), step)) or [0]
    xs = list(range(0, max(1, Ws - TILE + 1), step)) or [0]
    if ys[-1] != max(0, Hs - TILE):
        ys.append(max(0, Hs - TILE))
    if xs[-1] != max(0, Ws - TILE):
        xs.append(max(0, Ws - TILE))
    for y in ys:
        for x in xs:
            tile = rs[y:y + TILE, x:x + TILE]
            th, tw = tile.shape[:2]
            pad = np.zeros((TILE, TILE, 3), np.uint8)
            pad[:th, :tw] = tile
            m = infer_tile(model, pad)[:th, :tw]
            out[y:y + th, x:x + tw] = np.maximum(out[y:y + th, x:x + tw], m.astype(np.float32))
    mask = cv2.resize(out, (W0, H0), interpolation=cv2.INTER_NEAREST) > 0.5
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("organelle", choices=["er"])
    ap.add_argument("--weights", required=True,
                    help="directory holding {tem,sem,cell}_er.pth (see WEIGHTS.md)")
    ap.add_argument("--seg-root", required=True, help="root of the benchmark crop corpus")
    ap.add_argument("--results-root", required=True,
                    help="output root for per-crop CSVs + prediction PNGs")
    ap.add_argument("--no-preds", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    models = {v: build_er_model(os.path.join(args.weights, f"{v}_er.pth"))
              for v in ("cell", "sem", "tem")}

    def predict_fn(em, spec):
        v = variant_for(spec)
        mask = predict_full(models[v], np.asarray(em), spec.voxel_nm)
        return {"binary": mask, "instance": None, "_variant": v}

    run_model("deepcontact_er", args.organelle, predict_fn,
              seg_root=args.seg_root, results_root=args.results_root,
              save_preds=not args.no_preds, want_instance=False,
              only_datasets=args.datasets, merge=args.merge)


if __name__ == "__main__":
    main()

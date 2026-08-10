"""ALTERNATE empanada run using the `inference_scale` knob (does NOT overwrite the
native results — writes model names '<model>_scaled').

empanada's inference_scale is an IMAGE RESIZE, not a model parameter: the input is
downsampled by `scale`, the panoptic engine runs at that lower resolution, and the
output labels are upsampled back by `scale` (the engine's `upsampling` arg, which must
be a power of 2). `scale` is chosen per-crop to bring each crop's pixel size toward the
model's target nm/px (only downsampling high-res crops; coarse crops stay native).

Usage: python run_empanada_scaled.py <mitonet|nucleonet|lipidnet> <organelle> --target-nm T
           --weights W --seg-root S --results-root R
"""
import os, sys, math, argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from run_utils import run_model
from empanada.inference.engines import PanopticDeepLabRenderEngine

MEAN, STD = 0.57571, 0.12765
# target nm/px per model (documented): MitoNet trains 6-24 nm/px -> ~12;
# NucleoNet/DropNet preprint: optimal 15-40 nm/px (best ~15-30) -> ~20.
MODELS = {
    "mitonet":   dict(file="MitoNet_v1.pth",  padding=16,  organelle="mito",    target_nm=12.0),
    "nucleonet": dict(file="NucleoNet_v2.pth", padding=512, organelle="nucleus", target_nm=20.0),
    "lipidnet":  dict(file="DropNet_v1.pth",   padding=512, organelle="ld",      target_nm=20.0),
}
POW2 = [1, 2, 4, 8]


def make_engine(cfg, weights):
    model = torch.jit.load(os.path.join(weights, cfg["file"]), map_location="cuda").eval()
    return PanopticDeepLabRenderEngine(
        model, thing_list=[1], label_divisor=1000, stuff_area=64, void_label=0,
        nms_threshold=0.1, nms_kernel=7, confidence_thr=0.5,
        padding_factor=cfg["padding"], coarse_boundaries=True)


def preprocess(img):
    mx = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 255.0
    x = ((img.astype(np.float32)) - MEAN * mx) / (STD * mx)
    return torch.from_numpy(x)[None, None]


def choose_scale(voxel_nm, target_nm):
    if not voxel_nm or voxel_nm <= 0:
        return 1
    ideal = target_nm / voxel_nm            # >1 => data finer than target => downsample
    if ideal <= 1.25:
        return 1
    return min(POW2, key=lambda p: abs(p - ideal)) if ideal < 8 else 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("organelle", choices=["mito", "nucleus", "ld"])
    ap.add_argument("--weights", required=True,
                    help="directory holding MitoNet_v1.pth / NucleoNet_v2.pth / DropNet_v1.pth (see WEIGHTS.md)")
    ap.add_argument("--seg-root", required=True, help="root of the benchmark crop corpus")
    ap.add_argument("--results-root", required=True,
                    help="output root for per-crop CSVs + prediction PNGs")
    ap.add_argument("--target-nm", type=float, default=None)
    ap.add_argument("--force-scale", type=int, default=None,
                    help="override choose_scale with a constant downsample factor (experiment)")
    ap.add_argument("--tag", default="scaled", help="output model name suffix (<model>_<tag>)")
    ap.add_argument("--no-preds", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = MODELS[args.model]
    assert args.organelle == cfg["organelle"]
    target_nm = args.target_nm or cfg["target_nm"]
    engine = make_engine(cfg, args.weights)

    def predict_fn(em, spec):
        img = np.asarray(em)
        if img.ndim == 3:
            img = img[..., 0]
        H, W = img.shape
        scale = args.force_scale if args.force_scale else choose_scale(spec.voxel_nm, target_nm)
        with torch.no_grad():
            if scale == 1:
                pan = engine(preprocess(img).cuda(), size=(H, W), upsampling=1)
            else:
                ds = cv2.resize(img, (max(1, W // scale), max(1, H // scale)), interpolation=cv2.INTER_AREA)
                pan = engine(preprocess(ds).cuda(), size=(H, W), upsampling=scale)
        pan = np.asarray(pan.squeeze().cpu() if torch.is_tensor(pan) else pan).astype(np.int32)
        if pan.shape != (H, W):
            pan = cv2.resize(pan, (W, H), interpolation=cv2.INTER_NEAREST)
        uids = np.unique(pan)
        inst = np.zeros_like(pan)
        nid = 0
        for u in uids:
            if u != 0:
                nid += 1; inst[pan == u] = nid
        return {"binary": inst > 0, "instance": inst, "_scale": scale}

    run_model(f"{args.model}_{args.tag}", args.organelle, predict_fn,
              seg_root=args.seg_root, results_root=args.results_root,
              save_preds=not args.no_preds, want_instance=True,
              extra_meta={"target_nm": target_nm, "force_scale": args.force_scale},
              only_datasets=args.datasets, merge=args.merge)


if __name__ == "__main__":
    main()

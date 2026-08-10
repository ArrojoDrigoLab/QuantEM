"""Empanada panoptic models on the benchmark test crops (runs in the `empanada` env):
  mitonet   -> MitoNet_v1   -> mitochondria
  nucleonet -> NucleoNet_v2 -> nuclei
  lipidnet  -> DropNet_v1   -> lipid droplets
Uses empanada.inference.engines.PanopticDeepLabRenderEngine directly (no napari).
Norm: mean=0.57571 std=0.12765 (single-channel EM). Returns instance labels.

Usage: python run_empanada.py <mitonet|nucleonet|lipidnet> <organelle>
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

# NATIVE RESOLUTION: empanada runs at the image's native pixel resolution (its only
# optional knob is inference_scale, default 1 = native). The crops are NOT resized:
# they are capped at the 4096 canvas which fits in ~6 GB, so no downscaling/tiling
# is needed.
MODELS = {
    "mitonet":   dict(file="MitoNet_v1.pth",  padding=16,  organelle="mito"),
    "nucleonet": dict(file="NucleoNet_v2.pth", padding=512, organelle="nucleus"),
    "lipidnet":  dict(file="DropNet_v1.pth",   padding=512, organelle="ld"),
}


def make_engine(cfg, weights):
    model = torch.jit.load(os.path.join(weights, cfg["file"]), map_location="cuda").eval()
    return PanopticDeepLabRenderEngine(
        model, thing_list=[1], label_divisor=1000, stuff_area=64, void_label=0,
        nms_threshold=0.1, nms_kernel=7, confidence_thr=0.5,
        padding_factor=cfg["padding"], coarse_boundaries=True)


def preprocess(img):
    mx = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 255.0
    x = img.astype(np.float32)
    x = (x - MEAN * mx) / (STD * mx)
    return torch.from_numpy(x)[None, None]      # (1,1,H,W)


def run_engine(engine, img):
    t = preprocess(img).cuda()
    with torch.no_grad():
        pan = engine(t, size=img.shape, upsampling=1)
    pan = np.asarray(pan.squeeze().cpu() if torch.is_tensor(pan) else pan).astype(np.int32)
    return pan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("organelle", choices=["mito", "nucleus", "ld"])
    ap.add_argument("--weights", required=True,
                    help="directory holding MitoNet_v1.pth / NucleoNet_v2.pth / DropNet_v1.pth (see WEIGHTS.md)")
    ap.add_argument("--seg-root", required=True, help="root of the benchmark crop corpus")
    ap.add_argument("--results-root", required=True,
                    help="output root for per-crop CSVs + prediction PNGs")
    ap.add_argument("--no-preds", action="store_true")
    args = ap.parse_args()
    cfg = MODELS[args.model]
    assert args.organelle == cfg["organelle"], f"{args.model} is for {cfg['organelle']}"
    engine = make_engine(cfg, args.weights)

    def predict_fn(em, spec):
        img = np.asarray(em)
        if img.ndim == 3:
            img = img[..., 0]
        # NATIVE resolution: run the engine on the image as-is (no resize).
        pan = run_engine(engine, img)
        # relabel packed panoptic ids -> contiguous instance labels
        uids = np.unique(pan)
        remap = {u: i for i, u in enumerate(uids)}   # 0 stays 0 (first if present)
        if 0 not in remap:
            remap = {u: i + 1 for i, u in enumerate(uids)}
        inst = np.zeros_like(pan)
        for u, i in remap.items():
            if u != 0:
                inst[pan == u] = i
        return {"binary": inst > 0, "instance": inst}

    run_model(args.model, args.organelle, predict_fn,
              seg_root=args.seg_root, results_root=args.results_root,
              save_preds=not args.no_preds, want_instance=True)


if __name__ == "__main__":
    main()

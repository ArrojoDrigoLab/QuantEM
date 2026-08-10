"""micro-sam (vit_l_em_organelles) Automatic Instance Segmentation on the benchmark test
crops. Runs in the `microsam` env. Model covers all 4 organelles (strong for mito;
best-effort for nucleus/ER/LD -- benchmarked honestly).

Usage: python run_microsam.py <organelle> [--model vit_l_em_organelles]
           --seg-root S --results-root R [--no-preds]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from run_utils import run_model

import torch
from skimage.transform import resize as sk_resize
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

TILE = 1536          # run tiled AIS above this size
TILE_SHAPE = (1024, 1024)
HALO = (128, 128)
MAXSIDE = 2048       # downscale above this (avoid 16-tile 4096 runs); labels upscaled back


def build(model_type, tiled):
    return get_predictor_and_segmenter(model_type=model_type, device="cuda",
                                       segmentation_mode="ais", is_tiled=tiled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("organelle", choices=["mito", "er", "nucleus", "ld"])
    ap.add_argument("--model", default="vit_l_em_organelles")
    ap.add_argument("--seg-root", required=True, help="root of the benchmark crop corpus")
    ap.add_argument("--results-root", required=True,
                    help="output root for per-crop CSVs + prediction PNGs")
    ap.add_argument("--no-preds", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    pred_s, seg_s = build(args.model, tiled=False)
    pred_t, seg_t = build(args.model, tiled=True)

    def predict_fn(em, spec):
        img = np.asarray(em)
        if img.ndim == 3:
            img = img[..., 0]
        # light percentile contrast stretch to 0..255 (helps low-contrast EM)
        lo, hi = np.percentile(img, (1, 99))
        if hi > lo:
            img = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        # NATIVE resolution: run micro-sam at the image's native resolution. Large images
        # are handled by micro-sam's own tiling (below), NOT by downscaling.
        big = max(img.shape) > TILE
        try:
            if big:
                inst = automatic_instance_segmentation(
                    predictor=pred_t, segmenter=seg_t, input_path=img, ndim=2,
                    tile_shape=TILE_SHAPE, halo=HALO, verbose=False)
            else:
                inst = automatic_instance_segmentation(
                    predictor=pred_s, segmenter=seg_s, input_path=img, ndim=2, verbose=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            inst = automatic_instance_segmentation(
                predictor=pred_t, segmenter=seg_t, input_path=img, ndim=2,
                tile_shape=TILE_SHAPE, halo=HALO, verbose=False)
        inst = np.asarray(inst).astype(np.int32)
        return {"binary": inst > 0, "instance": inst}

    run_model(f"microsam_{args.model}", args.organelle, predict_fn,
              seg_root=args.seg_root, results_root=args.results_root,
              save_preds=not args.no_preds, want_instance=True,
              extra_meta={"model_type": args.model},
              only_datasets=args.datasets, merge=args.merge)


if __name__ == "__main__":
    main()

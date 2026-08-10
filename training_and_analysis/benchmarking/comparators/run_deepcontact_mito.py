"""DeepContact mitochondria (Mask R-CNN, Matterport) on the benchmark mito test crops.
Runs in the `deepcontact_mito` env (py3.7 / tf1.15 / keras2.1.6, CPU). Faithful to
DeepContact: grayscale->RGB, resolution-normalize to ~10 nm/px, 1024 sliding windows,
resnet101 Mask R-CNN, union instance masks. Weight variant matched per dataset.

Usage: python run_deepcontact_mito.py mito --deepcontact-repo REPO --weights W
           --seg-root S --results-root R
"""
import os, sys, argparse, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU-only (tf1.15)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from run_utils import run_model

TILE, OVERLAP = 1024, 0.2
MAXSIDE = 2048          # cap resolution-normalized size (avoid runaway upsampling)

# Fair variant per dataset: TEM->tem; SEM-family tissue->sem (Sertoli model);
# SEM-family cultured->cell (U-2 OS model). empiar_10982 is multimodal -> per-crop.
DATASET_VARIANT = {
    "zenodo_mitoem2": "sem",               # FIB-SEM, cerebellum tissue
    "empiar_10982_mitonet_benchmark": None, # mixed TEM/FIB/SBF -> per-crop modality
    "orgsegnet_plant": "tem",              # TEM, plant
    "deeppi_em_skeletal_muscle": "tem",    # TEM, mouse muscle
    "deepcontact_tem": "tem",              # TEM, COS-7
}


def variant_for(spec):
    v = DATASET_VARIANT.get(spec.dataset, "cell")
    if v is None:                          # multimodal dataset: match each crop's modality
        return "tem" if "tem" in (spec.modality or "").lower() else "sem"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("organelle", choices=["mito"])
    ap.add_argument("--deepcontact-repo", required=True,
                    help="path to a DeepContact clone (provides mrcnn + config; inserted on sys.path; see WEIGHTS.md)")
    ap.add_argument("--weights", required=True,
                    help="directory holding {tem,sem,cell}_mito.h5 (see WEIGHTS.md)")
    ap.add_argument("--seg-root", required=True, help="root of the benchmark crop corpus")
    ap.add_argument("--results-root", required=True,
                    help="output root for per-crop CSVs + prediction PNGs")
    ap.add_argument("--no-preds", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.deepcontact_repo)
    from mrcnn import model as modellib
    from config.mrcnn_config import MitochondrionInferenceConfig

    cfg = MitochondrionInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=cfg, model_dir=tempfile.mkdtemp())
    state = {"variant": None}

    def ensure(variant):
        if state["variant"] != variant:
            model.load_weights(os.path.join(args.weights, f"{variant}_mito.h5"), by_name=True)
            state["variant"] = variant
            print(f"  loaded {variant}_mito.h5", flush=True)

    def predict_full(img_gray, voxel_nm):
        rgb = np.repeat(img_gray[..., None], 3, axis=2)
        H0, W0 = img_gray.shape
        res = voxel_nm if (voxel_nm and voxel_nm > 0) else 10.0
        scale = min(5.0, max(0.2, res / 10.0))
        Hs, Ws = max(1, int(round(H0 * scale))), max(1, int(round(W0 * scale)))
        if max(Hs, Ws) > MAXSIDE:
            c = MAXSIDE / max(Hs, Ws); Hs, Ws = max(1, int(Hs * c)), max(1, int(Ws * c))
        rs = cv2.resize(rgb, (Ws, Hs), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        binout = np.zeros((Hs, Ws), np.uint8)
        instout = np.zeros((Hs, Ws), np.int32)
        nid = 0
        step = max(1, int(TILE * (1 - OVERLAP)))
        ys = sorted(set(list(range(0, max(1, Hs - TILE + 1), step)) + [max(0, Hs - TILE)]))
        xs = sorted(set(list(range(0, max(1, Ws - TILE + 1), step)) + [max(0, Ws - TILE)]))
        for y in ys:
            for x in xs:
                tile = rs[y:y + TILE, x:x + TILE]
                th, tw = tile.shape[:2]
                pad = np.zeros((TILE, TILE, 3), np.uint8); pad[:th, :tw] = tile
                r = model.detect([pad], verbose=0)[0]
                masks = r["masks"]
                if masks.size == 0:
                    continue
                for k in range(masks.shape[2]):
                    mk = masks[:th, :tw, k]
                    if mk.sum() == 0:
                        continue
                    ys_, xs_ = np.where(mk)
                    gy, gx = ys_ + y, xs_ + x
                    binout[gy, gx] = 1
                    free = instout[gy, gx] == 0
                    if free.any():
                        nid += 1
                        instout[gy[free], gx[free]] = nid
        binm = cv2.resize(binout, (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
        instm = cv2.resize(instout, (W0, H0), interpolation=cv2.INTER_NEAREST)
        return binm, instm

    def predict_fn(em, spec):
        v = variant_for(spec)
        ensure(v)
        b, inst = predict_full(np.asarray(em), spec.voxel_nm)
        return {"binary": b, "instance": inst, "_variant": v}

    run_model("deepcontact_mito", args.organelle, predict_fn,
              seg_root=args.seg_root, results_root=args.results_root,
              save_preds=not args.no_preds, want_instance=True,
              only_datasets=args.datasets, merge=args.merge)


if __name__ == "__main__":
    main()

"""OrgSegNet (Feng et al. 2023, PSPNet/MMSegmentation) on the benchmark mito + nucleus crops.
OrgSegNet segments plant-cell organelles: 0=bg 1=chloroplast 2=mito 3=vacuole 4=nucleus
(single multiclass model); mito (==2) and nucleus (==4) are extracted. Runs in the
`orgsegnet` env (torch1.13/cu116, mmcv 2.0.0rc4). orgsegnet_plant is OrgSegNet's
own official test split -> in-training-domain (flagged in figures).

Usage: python run_orgsegnet.py <mito|nucleus> --orgsegnet-repo REPO --weights W
           --seg-root S --results-root R
"""
import os, sys, argparse, tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from run_utils import run_model

VAL = {"mito": 2, "nucleus": 4}


def build_model(config, ckpt):
    from mmseg.apis import init_model
    try:
        from mmseg.utils import register_all_modules
        register_all_modules(init_default_scope=True)
    except Exception:
        pass
    return init_model(config, ckpt, device="cuda:0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("organelle", choices=["mito", "nucleus"])
    ap.add_argument("--orgsegnet-repo", required=True,
                    help="path to an OrgSegNet clone (provides mmseg + configs; inserted on sys.path; see WEIGHTS.md)")
    ap.add_argument("--weights", required=True,
                    help="path to OrgSegNet_iter_Version1.pth (see WEIGHTS.md)")
    ap.add_argument("--seg-root", required=True, help="root of the benchmark crop corpus")
    ap.add_argument("--results-root", required=True,
                    help="output root for per-crop CSVs + prediction PNGs")
    ap.add_argument("--no-preds", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.orgsegnet_repo)
    config = os.path.join(args.orgsegnet_repo, "configs", "OrgSegNet", "OrgSeg_PlantCell_768x512.py")
    tmp = tempfile.mkdtemp()
    from mmseg.apis import inference_model
    model = build_model(config, args.weights)
    val = VAL[args.organelle]
    tmpf = os.path.join(tmp, f"cur_{args.organelle}.png")

    def predict_fn(em, spec):
        em = np.asarray(em)
        if em.ndim == 3:
            em = em[..., 0]
        # STANDARD OrgSegNet inference: feed the native-resolution crop and let the model's
        # own test pipeline (Resize(768,512, keep_ratio) + slide-window 512/256) run. This is
        # the published pipeline.
        cv2.imwrite(tmpf, em)
        result = inference_model(model, tmpf)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        if pred.shape != em.shape:
            pred = cv2.resize(pred, (em.shape[1], em.shape[0]), interpolation=cv2.INTER_NEAREST)
        return {"binary": pred == val, "instance": None}

    run_model("orgsegnet", args.organelle, predict_fn,
              seg_root=args.seg_root, results_root=args.results_root,
              save_preds=not args.no_preds, want_instance=False)


if __name__ == "__main__":
    main()

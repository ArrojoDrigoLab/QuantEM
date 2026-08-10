"""incasem (Kirchhausen lab, 3D FIB-SEM organelle U-Nets) on the benchmark mito/ER crops.
incasem models are inherently 3D (~100+ z-slices of context); the benchmark test crops
are single 2D slices, so each slice is z-replicated into a thin slab, run through the
valid-padded 3D U-Net, and the central output slice is taken. Data is resampled to the
model's native 5 nm/px (+ CLAHE), per incasem's requirements. Runs in the `incasem` env.

Usage: python run_incasem.py <mito|er> --ckpt-mito P|--ckpt-er P --seg-root S --results-root R
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from run_utils import run_model
import torch
import incasem as fos
from skimage.exposure import equalize_adapthist
from skimage.transform import resize as sk_resize


def _resize(img, out_hw, order=1):
    return sk_resize(img, out_hw, order=order, preserve_range=True,
                     anti_aliasing=(order > 0 and out_hw[0] < img.shape[0])).astype(np.float32)

DEV = "cuda"
CTX, OUT = 47, 110                    # valid 3D U-Net: input 204 -> output 110 (per axis)
IN = OUT + 2 * CTX                    # 204
ZIN = IN                             # cubic 204^3 block (only valid size); take middle z-slice
MAXSIDE = 1400                        # cap resampled size (incasem is 5nm; far-off data capped)


def build_model(ckpt):
    m = fos.torch.models.Unet(
        in_channels=1, num_fmaps=32, fmap_inc_factor=2,
        downsample_factors=((2, 2, 2), (2, 2, 2), (2, 2, 2)),
        num_fmaps_out=2, constant_upsample=True, padding="valid",
    ).to(DEV).eval()
    sd = torch.load(ckpt, map_location=DEV)
    sd = sd.get("model_state_dict", sd)
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def forward_block(model, block2d):
    # block2d: (IN,IN) float32 in [-1,1]; replicate to z-slab
    vol = np.repeat(block2d[None], ZIN, axis=0)          # (ZIN,IN,IN)
    x = torch.from_numpy(vol)[None, None].to(DEV)        # (1,1,Z,Y,X)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0, 1]           # (Zo,Yo,Xo)
    zmid = probs.shape[0] // 2
    return probs[zmid].cpu().numpy()                     # (OUT,OUT)


def predict_full(model, img_gray, voxel_nm):
    H0, W0 = img_gray.shape
    res = voxel_nm if (voxel_nm and voxel_nm > 0) else 5.0
    scale = res / 5.0
    Hs, Ws = max(1, int(round(H0 * scale))), max(1, int(round(W0 * scale)))
    if max(Hs, Ws) > MAXSIDE:                            # cap runaway upsampling
        c = MAXSIDE / max(Hs, Ws); Hs, Ws = max(1, int(Hs * c)), max(1, int(Ws * c))
    eq = equalize_adapthist(img_gray, clip_limit=0.02).astype(np.float32)   # [0,1]
    rs = _resize(eq, (Hs, Ws), order=1)
    x = rs * 2.0 - 1.0
    pad = np.pad(x, CTX, mode="reflect")
    fg = np.zeros((Hs, Ws), np.float32)
    oys = list(range(0, Hs, OUT))
    oxs = list(range(0, Ws, OUT))
    for oy in oys:
        for ox in oxs:
            block = pad[oy:oy + IN, ox:ox + IN]
            bh, bw = block.shape
            if bh < IN or bw < IN:
                block = np.pad(block, ((0, IN - bh), (0, IN - bw)), mode="reflect")
            out = forward_block(model, block)
            h = min(OUT, Hs - oy); w = min(OUT, Ws - ox)
            fg[oy:oy + h, ox:ox + w] = out[:h, :w]
    mask = _resize(fg, (H0, W0), order=1) > 0.5
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("organelle", choices=["mito", "er"])
    ap.add_argument("--ckpt-mito", default=None,
                    help="path to model_checkpoint_1847_mito_CF.pt (required for mito; see WEIGHTS.md)")
    ap.add_argument("--ckpt-er", default=None,
                    help="path to model_checkpoint_1841_er_CF.pt (required for er; see WEIGHTS.md)")
    ap.add_argument("--seg-root", required=True, help="root of the benchmark crop corpus")
    ap.add_argument("--results-root", required=True,
                    help="output root for per-crop CSVs + prediction PNGs")
    ap.add_argument("--no-preds", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    ckpt = {"mito": args.ckpt_mito, "er": args.ckpt_er}[args.organelle]
    if not ckpt:
        ap.error(f"--ckpt-{args.organelle} is required for organelle '{args.organelle}'")
    model = build_model(ckpt)

    MARGIN = 64

    def predict_fn(em, spec):
        em = np.asarray(em)
        H, W = em.shape
        # only run inside the annotation window (+margin); scoring is restricted there anyway
        if spec.annotation_bbox and spec.valid_region:
            vx0, vy0 = int(spec.valid_region[0]), int(spec.valid_region[1])
            ax0, ay0, ax1, ay1 = [int(v) for v in spec.annotation_bbox]
            lx0 = max(0, ax0 - vx0 - MARGIN); ly0 = max(0, ay0 - vy0 - MARGIN)
            lx1 = min(W, ax1 - vx0 + MARGIN); ly1 = min(H, ay1 - vy0 + MARGIN)
        else:
            lx0, ly0, lx1, ly1 = 0, 0, W, H
        if lx1 <= lx0 or ly1 <= ly0:
            return {"binary": np.zeros((H, W), bool), "instance": None}
        sub = em[ly0:ly1, lx0:lx1]
        m = predict_full(model, sub, spec.voxel_nm)
        full = np.zeros((H, W), bool)
        full[ly0:ly1, lx0:lx1] = m
        return {"binary": full, "instance": None}

    run_model("incasem", args.organelle, predict_fn,
              seg_root=args.seg_root, results_root=args.results_root,
              save_preds=not args.no_preds, want_instance=False,
              only_datasets=args.datasets, merge=args.merge)


if __name__ == "__main__":
    main()

"""Shared runner: apply a model's predict_fn to every benchmark test crop, score it
region-masked, and write per-crop metrics (+ optional prediction PNGs)."""
import os, csv, time, traceback
import numpy as np

from benchmark_common import iter_test_crops, load_crop, BENCHMARK_DATASETS, DATASET_LABEL
import metrics as M


def _save_pred(preds_root, model, spec, pred_bin, valid_slice):
    try:
        import tifffile
        y0, y1, x0, x1 = valid_slice
        sub = (pred_bin[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        d = os.path.join(preds_root, model, spec.organelle, spec.dataset)
        os.makedirs(d, exist_ok=True)
        tifffile.imwrite(os.path.join(d, f"{spec.crop_id}.png"), sub)
    except Exception:
        pass


def run_model(model_name, organelle, predict_fn, seg_root, results_root,
              save_preds=True, want_instance=True, extra_meta=None,
              compute_topology=True, progress_every=10, only_datasets=None, merge=False):
    """
    predict_fn(em_crop_2d_uint8, spec) -> dict with keys:
        'binary'   : (h,w) bool/uint8 foreground over the em_crop (required)
        'instance' : (h,w) int labels over the em_crop (optional)
    where (h,w) == em_crop.shape (the real-EM valid_region crop).

    seg_root      : root directory of the segmentation crop datasets (one
                    sub-directory per dataset, plus splits/).
    results_root  : output root; per-crop CSVs are written to
                    <results_root>/per_crop/ and prediction PNGs to
                    <results_root>/preds/.
    only_datasets : optional collection of dataset ids; restricts the run to
                    those datasets (re-run filter, typically paired with
                    merge=True to keep the other datasets' existing rows).
    """
    per_crop_dir = os.path.join(results_root, "per_crop")
    os.makedirs(per_crop_dir, exist_ok=True)
    preds_root = os.path.join(results_root, "preds")
    out_csv = os.path.join(per_crop_dir, f"{model_name}__{organelle}.csv")
    rows = []
    specs = list(iter_test_crops(organelle, seg_root=seg_root))
    if only_datasets:
        specs = [s for s in specs if s.dataset in only_datasets]
    t0 = time.time()
    for i, spec in enumerate(specs):
        rec = dict(model=model_name, organelle=organelle, dataset=spec.dataset,
                   dataset_label=DATASET_LABEL.get(spec.dataset, spec.dataset),
                   crop_id=spec.crop_id, coverage=spec.coverage_tier,
                   voxel_nm=spec.voxel_nm, source_image=spec.source_image)
        if extra_meta:
            rec.update(extra_meta)
        try:
            data = load_crop(spec, want_instance=want_instance)
            em, emask = data["em"], data["eval_mask"]
            H, W = data["shape"]
            y0, y1, x0, x1 = data["valid_slice"]
            em_crop = em[y0:y1, x0:x1]
            if em_crop.size == 0:
                em_crop = em
                y0, y1, x0, x1 = 0, H, 0, W
            t = time.time()
            pred = predict_fn(em_crop, spec)
            rec["infer_s"] = round(time.time() - t, 3)

            pb = np.zeros((H, W), bool)
            sub = np.asarray(pred["binary"]) > 0
            pb[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub[:y1 - y0, :x1 - x0]

            sm = M.semantic_metrics(pb, data["gt_binary"], emask)
            if compute_topology:
                sm["cldice"] = M.cldice(pb, data["gt_binary"], emask)
                sm["boundary_f1"] = M.boundary_f1(pb, data["gt_binary"], emask)
            rec.update(sm)

            pi = pred.get("instance")
            if pi is not None and data["gt_instance"] is not None:
                pinst = np.zeros((H, W), np.int32)
                pia = np.asarray(pi).astype(np.int32)
                pinst[y0:y0 + pia.shape[0], x0:x0 + pia.shape[1]] = pia[:y1 - y0, :x1 - x0]
                im = M.instance_metrics(pinst, data["gt_instance"], emask)
                rec.update({("inst_" + k): v for k, v in im.items()})

            if save_preds:
                _save_pred(preds_root, model_name, spec, pb, (y0, y1, x0, x1))
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        rows.append(rec)
        if progress_every and (i % progress_every == 0 or i == len(specs) - 1):
            done = i + 1
            el = time.time() - t0
            print(f"[{model_name}/{organelle}] {done}/{len(specs)} "
                  f"({spec.dataset}) {el:.0f}s", flush=True)

    if merge and os.path.exists(out_csv):
        rerun = {(r["dataset"], r["crop_id"]) for r in rows}
        old = [r for r in csv.DictReader(open(out_csv)) if (r["dataset"], r["crop_id"]) not in rerun]
        rows = old + rows
        print(f"[merge] kept {len(old)} existing rows + {len(rerun)} re-run", flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"WROTE {out_csv}  ({len(rows)} rows)", flush=True)
    return out_csv, rows

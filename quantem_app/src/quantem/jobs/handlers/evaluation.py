"""Evaluate without training, changing predictions, or changing annotations."""

import numpy as np

from quantem.finetune.calibrate import masked_dice, masked_iou
from quantem.finetune.evaluation import crop_fingerprint, evaluation_inputs
from quantem.inference import engine, resample
from quantem.jobs.constants import JOB_TYPE_EVALUATE_MODEL
from quantem.jobs.registry import job_handler


@job_handler(JOB_TYPE_EVALUATE_MODEL)
def handle_evaluate_model(payload, reporter, cancel):
    cancel.check_cancelled()
    scope, base, adapter, crops, exposure = evaluation_inputs(payload, load_em=True)
    selected = {row["id"]: row for row in payload["rois"]}
    crops = [crop for crop in crops if crop.id in selected]
    if len(crops) != len(selected) or any(
        crop_fingerprint(crop) != selected[crop.id]["input_fingerprint"] for crop in crops
    ):
        raise ValueError("The selected annotations changed. Preview and run the evaluation again.")
    reporter.update(progress=5, message="Loading the model for evaluation")
    model = (
        engine.load_adapted_model(base, adapter.head_file, adapter_id=str(adapter.id))
        if adapter and adapter.head_file
        else engine.load_model(base)
    )
    threshold = payload["threshold"]
    rows = []
    for index, crop in enumerate(crops):
        cancel.check_cancelled()
        predicted = engine.predict_region(model, crop.em, pixel_size_nm=crop.pixel_size_nm)
        stored = resample.probability_to_native_uint8(predicted.prob, predicted.context)
        foreground = resample.binarize_quantized(stored, threshold)
        rows.append(
            {
                "roi_id": crop.id,
                "name": crop.name,
                "image_id": crop.image_key,
                "dice": masked_dice(foreground, crop.gt, crop.valid, 0.5),
                "iou": masked_iou(foreground, crop.gt, crop.valid, 0.5),
                "annotated_pixels": crop.annotated_px,
                "training_overlap": crop.image_key in exposure,
                "input_fingerprint": crop_fingerprint(crop),
            }
        )
        reporter.update(
            progress=5 + 95 * (index + 1) / len(crops),
            message=f"Evaluated {index + 1} of {len(crops)} ROIs",
        )
    cancel.check_cancelled()
    return {
        "base_model": base,
        "adapter_id": str(adapter.id) if adapter else None,
        "threshold": threshold,
        "per_roi": rows,
        "mean_dice": float(np.mean([row["dice"] for row in rows if row["dice"] is not None]))
        if any(row["dice"] is not None for row in rows)
        else None,
        "training_overlap": any(row["training_overlap"] for row in rows),
        "notes": [
            "Scores compare thresholded native-grid predictions with completed annotations. Empty prediction and ground truth produce an undefined score."
        ]
        + (
            [
                "Some ROIs are on images used to train this model or an earlier version. These scores are not independent held-out evaluation."
            ]
            if any(row["training_overlap"] for row in rows)
            else []
        ),
    }

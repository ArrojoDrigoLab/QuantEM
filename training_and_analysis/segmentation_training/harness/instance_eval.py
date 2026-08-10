"""Shared instance evaluation — the single boundary the harnesses import.

This module is the interface for mitochondria and instance metrics. The default `instance_metrics`
PQ is a semantic-foreground score computed by connected components, not an instance-segmentation
score.

Metric definitions: the decoder-agnostic "instance" PQ/AP is computed by `instance_metrics` =
threshold the semantic softmax + connected-components (`postproc_instances`). That is a
*semantic-foreground-quality-via-CC* score — it never runs any decoder's instance mechanism, so it
does not test instance segmentation. This module exposes the true-instance path (each decoder's own
post-proc) and reports both numbers side by side (dual metric).

Contract — for a decoder to be scored on true-instance metrics:
  1. In `forward()`, populate `self.aux_logits` with the instance-head outputs
     (affinity head: `[affinities]`; panoptic: `[center_logit, offset]`; query: `[per_query_masks]`).
  2. Implement `native_instance_labels(self, aux, fg) -> np.ndarray[H,W] int32` that turns the (full-region,
     sliding-window-accumulated) `aux` into an instance-id map via the decoder's own post-proc
     (mutex-watershed on affinities / center-offset grouping / per-query-mask argmax).
`segmentation_training.harness.evaluate.evaluate_head` (the shared eval used by `run_seg`) then reports
`inst_pq / inst_sq / inst_rq / inst_ap / inst_vi` alongside the semantic-CC `pq/ap/...`. No further wiring
is required: runs through `run_seg` / `evaluate_head` expose the `inst_*` keys. `AffinityMWS`,
`PanopticDeepLab` and the Mask2Former query decoder (`Mask2FormerHFBridge`, registered as
`mask2former_query_hf`) implement the contract.

For a standalone dual-metric score outside `evaluate_head` (e.g. a custom eval loop), call
`dual_instance_metrics`. Reusable building blocks (targets / post-procs / from-labels scorer) are
re-exported below.
"""
from __future__ import annotations

import numpy as np

# Re-exports: the reusable pieces (import from here so there is a single boundary to depend on).
from segmentation_training.models.instance_targets import seg_to_affinities, seg_to_center_offset          # noqa: F401
from segmentation_training.models.decoders import mutex_watershed_postproc, panoptic_instance_postproc     # noqa: F401
from segmentation_training.harness.instance_metrics import (                                               # noqa: F401
    instance_metrics,                 # semantic-CC (decoder-agnostic)
    instance_metrics_from_labels,     # score a decoder's own native label map (true-instance)
    postproc_instances,
)


def has_native_instance(model_or_decoder) -> bool:
    """True if the arm exposes a native instance post-proc (so true-instance metrics are computable)."""
    dec = getattr(model_or_decoder, "decoder", model_or_decoder)
    return hasattr(dec, "native_instance_labels")


def dual_instance_metrics(decoder, aux, prob: np.ndarray, gt_inst: np.ndarray, valid: np.ndarray, *,
                          fg_threshold: float = 0.5, min_size: int = 16) -> dict:
    """Both metrics for one region, side by side.

    (1) semantic-CC: `instance_metrics(prob, ...)` — pq/sq/rq/ap/vi from threshold+CC on the semantic softmax.
    (2) true-instance: `decoder.native_instance_labels(aux, fg)` -> label map -> `instance_metrics_from_labels`,
        stored under `inst_*` keys.

    `aux` = the decoder's sliding-window-accumulated `aux_logits` (as `predict_region(..., collect_aux=True)`
    returns). Returns a dict with the semantic-CC keys and (if the decoder has a native head) the `inst_*` keys.
    A native-postproc failure never breaks the semantic-CC number (recorded under `inst_error`).
    """
    m = instance_metrics(prob, gt_inst, valid, threshold=fg_threshold, min_size=min_size)
    if aux and hasattr(decoder, "native_instance_labels"):
        try:
            pred_bin = np.asarray(prob) >= fg_threshold
            lab = decoder.native_instance_labels(aux, pred_bin & np.asarray(valid).astype(bool))
            im = instance_metrics_from_labels(lab, gt_inst, valid, prob)
            m.update({f"inst_{k}": v for k, v in im.items()
                      if k in ("pq", "sq", "rq", "ap", "vi", "n_pred_inst")})
        except Exception as exc:  # noqa: BLE001 — true-instance must never break the semantic-CC eval
            m["inst_error"] = str(exc)[:120]
    return m

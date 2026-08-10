"""Dual-metric: the connected-component instance metric agrees with instance_metrics_from_labels,
and instance_metrics_from_labels scores a given label map correctly."""
import numpy as np

from segmentation_training.harness.instance_metrics import (
    instance_metrics,
    instance_metrics_from_labels,
    postproc_instances,
)


def _two_blobs():
    gt = np.zeros((16, 16), np.int64)
    gt[2:6, 2:6] = 1
    gt[10:14, 10:14] = 2
    valid = np.ones((16, 16), bool)
    return gt, valid


def test_refactor_behaviour_preserving():
    # the CC metric equals from_labels applied to the same CC labelling
    gt, valid = _two_blobs()
    prob = (gt > 0).astype(np.float32) * 0.9
    m_cc = instance_metrics(prob, gt, valid, threshold=0.5, min_size=0)
    cc_lab = postproc_instances(prob, valid, 0.5, 0)
    m_fl = instance_metrics_from_labels(cc_lab, gt, valid, prob)
    for k in ("pq", "sq", "rq", "ap"):
        assert abs(m_cc[k] - m_fl[k]) < 1e-6, f"{k}: {m_cc[k]} vs {m_fl[k]}"
    assert m_cc["pq"] > 0.9  # near-perfect prediction


def test_from_labels_perfect_and_wrong():
    gt, valid = _two_blobs()
    prob = (gt > 0).astype(np.float32)
    # pred == gt -> PQ 1
    assert instance_metrics_from_labels(gt.copy(), gt, valid, prob)["pq"] == 1.0
    # one merged instance (both blobs same id) -> RQ drops (1 pred can't match 2 gt)
    merged = (gt > 0).astype(np.int64)  # single instance
    m = instance_metrics_from_labels(merged, gt, valid, prob)
    assert m["n_pred_inst"] == 1 and m["rq"] < 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL PASS")

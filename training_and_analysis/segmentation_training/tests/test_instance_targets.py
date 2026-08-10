"""Correctness tests for the instance-target derivation (segmentation_training/models/instance_targets.py)
and the panoptic grouping that consumes it (segmentation_training/models/decoders.py).

Verifies the affinity, centre and offset targets so that the instance head is supervised by the
targets its metric names, and that ``panoptic_instance_postproc`` recovers instances from them."""
import torch

from segmentation_training.models.instance_targets import (
    seg_to_affinities,
    seg_to_center_offset,
    DEFAULT_MWS_OFFSETS,
)


def test_affinity_within_across_and_border():
    # cols: [inst1, inst1, background, inst2]
    inst = torch.tensor([[1, 1, 0, 2]] * 4, dtype=torch.long)  # [4,4]
    aff, valid = seg_to_affinities(inst, [(0, 1)])  # horizontal neighbour
    # within instance 1 (x=0 -> x=1): attractive = 1
    assert aff[0, 0, 0, 0].item() == 1.0
    # instance-1 -> background (x=1 -> x=2): 0
    assert aff[0, 0, 0, 1].item() == 0.0
    # background source (x=2): not foreground -> 0
    assert aff[0, 0, 0, 2].item() == 0.0
    # last column (x=3): shifted x=4 out of bounds -> invalid, aff 0
    assert valid[0, 0, 0, 3].item() is False or valid[0, 0, 0, 3].item() == False  # noqa: E712
    assert aff[0, 0, 0, 3].item() == 0.0
    # in-bounds pixel is valid
    assert valid[0, 0, 0, 0].item() == True  # noqa: E712


def test_affinity_no_wraparound():
    # a vertical offset does not wrap the bottom row to the top
    inst = torch.tensor([[1], [1], [2], [2]], dtype=torch.long)  # [4,1]
    aff, valid = seg_to_affinities(inst, [(1, 0)])  # look one row down
    # row 2 (inst 1) -> row 3 (inst 2): different -> 0
    assert aff[0, 0, 1, 0].item() == 0.0
    # bottom row (y=3): target y=4 out of bounds -> invalid
    assert valid[0, 0, 3, 0].item() == False  # noqa: E712


def test_center_offset_single_instance():
    inst = torch.zeros((5, 5), dtype=torch.long)
    inst[1:4, 1:4] = 1  # 3x3 block, centroid (2,2)
    center, offset, fg = seg_to_center_offset(inst, sigma=1.0)
    assert center.shape == (1, 1, 5, 5) and offset.shape == (1, 2, 5, 5)
    # center peak at the centroid, value ~1
    assert abs(center[0, 0, 2, 2].item() - 1.0) < 1e-4
    assert int(center[0, 0].argmax().item()) == 2 * 5 + 2
    # offset at (1,1) points to centroid (2,2): (+1,+1)
    assert abs(offset[0, 0, 1, 1].item() - 1.0) < 1e-5
    assert abs(offset[0, 1, 1, 1].item() - 1.0) < 1e-5
    # offset at the centroid is zero; background offset is zero; fg mask correct
    assert abs(offset[0, 0, 2, 2].item()) < 1e-5
    assert abs(offset[0, 0, 0, 0].item()) < 1e-5
    assert fg[0, 0, 2, 2].item() == 1.0 and fg[0, 0, 0, 0].item() == 0.0


def test_shapes_and_default_offsets():
    inst = torch.randint(0, 4, (2, 8, 8))
    aff, valid = seg_to_affinities(inst, DEFAULT_MWS_OFFSETS)
    assert aff.shape == (2, len(DEFAULT_MWS_OFFSETS), 8, 8)
    assert valid.shape == aff.shape
    assert aff.dtype == torch.float32 and valid.dtype == torch.bool


def test_panoptic_postproc_two_instances():
    import numpy as np
    from segmentation_training.models.decoders import panoptic_instance_postproc
    inst = torch.zeros((16, 16), dtype=torch.long)
    inst[2:5, 2:5] = 1      # centroid (3,3)
    inst[11:14, 11:14] = 2  # centroid (12,12)
    center, offset, fg = seg_to_center_offset(inst, sigma=1.0)
    lab = panoptic_instance_postproc(center[0], offset[0], fg[0, 0], center_thresh=0.5, nms_kernel=5)
    ids = set(int(i) for i in np.unique(lab)) - {0}
    assert len(ids) == 2, f"expected 2 instances, got {ids}"
    assert lab[3, 3] > 0 and lab[12, 12] > 0 and lab[3, 3] != lab[12, 12]
    assert lab[0, 0] == 0  # background stays 0


def test_panoptic_postproc_no_centers_fallback():
    import numpy as np
    from segmentation_training.models.decoders import panoptic_instance_postproc
    # flat (no peaks above threshold) -> CC fallback on fg still yields the instance
    center = torch.zeros((1, 8, 8))
    offset = torch.zeros((2, 8, 8))
    fg = torch.zeros((8, 8), dtype=torch.bool)
    fg[1:4, 1:4] = True
    lab = panoptic_instance_postproc(center, offset, fg, center_thresh=0.5)
    assert (np.asarray(lab) > 0).sum() == 9  # the 3x3 fg block labelled


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL PASS")

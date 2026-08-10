"""instance-head losses: gradient flow, safe no-op when unwired, and inst plumbing through CombinedLoss."""
import torch

from segmentation_training.models.losses import AffinityLoss, PanopticInstanceLoss, CombinedLoss, DiceBCELoss
from segmentation_training.models.decoders import AffinityMWS


def _inst():
    inst = torch.zeros((1, 16, 16), dtype=torch.long)
    inst[0, 2:7, 2:7] = 1
    inst[0, 9:14, 9:14] = 2
    return inst


def _target(inst):
    return (inst > 0).long()  # [B,H,W] semantic fg (for the dense terms)


def test_affinity_loss_grad_and_noop():
    inst = _inst()
    n = len(AffinityMWS.DEFAULT_OFFSETS)
    aff = torch.rand((1, n, 16, 16), requires_grad=True)
    logits = torch.zeros((1, 2, 16, 16))
    L = AffinityLoss()
    v = L(logits, _target(inst), aux_logits=[aff], inst=inst)
    assert v.item() > 0
    v.backward()
    assert aff.grad is not None and aff.grad.abs().sum() > 0
    # no-op when not an affinity head / no inst
    assert L(logits, _target(inst), aux_logits=None, inst=None).item() == 0.0
    wrong = torch.rand((1, 3, 16, 16))  # channel count != n_offsets
    assert L(logits, _target(inst), aux_logits=[wrong], inst=inst).item() == 0.0


def test_panoptic_loss_grad_and_noop():
    inst = _inst()
    center = torch.zeros((1, 1, 16, 16), requires_grad=True)
    offset = torch.zeros((1, 2, 16, 16), requires_grad=True)
    logits = torch.zeros((1, 2, 16, 16))
    L = PanopticInstanceLoss()
    v = L(logits, _target(inst), aux_logits=[center, offset], inst=inst)
    assert v.item() > 0
    v.backward()
    assert center.grad is not None and offset.grad is not None
    assert L(logits, _target(inst), aux_logits=None, inst=None).item() == 0.0


def test_combined_loss_inst_plumbing():
    inst = _inst()
    n = len(AffinityMWS.DEFAULT_OFFSETS)
    aff = torch.rand((1, n, 16, 16))
    logits = torch.randn((1, 2, 16, 16))
    L = CombinedLoss([DiceBCELoss(), AffinityLoss()], [1.0, 1.0], ["dice_bce", "affinity"], add_auto_ce=False)
    total, report = L(logits, _target(inst), aux_logits=[aff], inst=inst)
    assert report["affinity"] > 0, "inst must reach the affinity term through CombinedLoss"
    assert "dice_bce" in report
    # inst=None -> affinity term no-ops, dice_bce unaffected
    _, report2 = L(logits, _target(inst), aux_logits=[aff], inst=None)
    assert report2["affinity"] == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL PASS")

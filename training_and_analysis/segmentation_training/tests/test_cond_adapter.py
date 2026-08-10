"""Conditional-adapter unit tests (torch-only, CPU safe). Validates the core mechanism without an
encoder: identity-at-init, per-code output dependence, gradient flow, prefix preservation, controller wiring."""
import torch

from segmentation_training.harness.adapters import CondLoRAAdapter, CondLoRAController


def _mk(mode, D=32, r=8, cd=64):
    return CondLoRAAdapter(D, n_prefix=1, rank=r, conv=True, cond_dim=cd, mode=mode, n_experts=4)


def test_identity_at_init():
    B, npfx, P, D, cd = 2, 1, 16, 32, 64  # P=16 -> 4x4 grid (conv path active)
    for mode in ("hyper", "moe"):
        ad = _mk(mode, D=D, cd=cd)
        tokens = torch.randn(B, npfx + P, D)
        ad.set_code(torch.randn(B, cd))
        out = ad(tokens)
        assert out.shape == tokens.shape
        assert torch.allclose(out, tokens, atol=1e-6), f"{mode}: not identity at init (up/experts zero)"
        assert torch.allclose(out[:, :npfx], tokens[:, :npfx]), f"{mode}: prefix tokens altered"


def test_code_dependence_after_departure():
    B, npfx, P, D, cd = 2, 1, 16, 32, 64
    for mode in ("hyper", "moe"):
        ad = _mk(mode, D=D, cd=cd)
        tokens = torch.randn(B, npfx + P, D)
        if mode == "hyper":
            torch.nn.init.normal_(ad.up.weight, std=0.1)
            torch.nn.init.normal_(ad.mod.weight, std=0.3)
        else:
            for u in ad.up:
                torch.nn.init.normal_(u.weight, std=0.1)
            torch.nn.init.normal_(ad.gate.weight, std=1.0)
        ad.set_code(torch.randn(B, cd)); o1 = ad(tokens)
        ad.set_code(torch.randn(B, cd)); o2 = ad(tokens)
        assert not torch.allclose(o1, o2, atol=1e-5), f"{mode}: output not code-dependent after departure"


def test_grad_flow():
    B, npfx, P, D, cd = 2, 1, 16, 32, 64
    for mode in ("hyper", "moe"):
        ad = _mk(mode, D=D, cd=cd)
        tokens = torch.randn(B, npfx + P, D)
        ad.set_code(torch.randn(B, cd, requires_grad=False))
        ad(tokens).pow(2).sum().backward()
        assert ad.down.weight.grad is not None
        cond_param = ad.mod if mode == "hyper" else ad.gate
        # The up-projections are zero at init, so in both modes the conditioning head's gradient is
        # zero-valued on the first step; only its presence on the graph is asserted.
        assert cond_param.weight.grad is not None, f"{mode}: no grad on conditioning head"


def test_controller_sets_codes():
    B, cd = 2, 64
    adapters = torch.nn.ModuleList([_mk("hyper", cd=cd), _mk("hyper", cd=cd)])
    ctrl = CondLoRAController(adapters, cond_source="image", cond_dim=cd)
    img = torch.randn(B, 1, 64, 64)
    ctrl.before_forward(img)
    for ad in adapters:
        assert ad._code is not None and ad._code.shape == (B, cd)


def test_source_controller():
    B, cd = 3, 64
    adapters = torch.nn.ModuleList([_mk("hyper", cd=cd)])
    ctrl = CondLoRAController(adapters, cond_source="source", cond_dim=cd, source_vocab=10)
    ctrl.before_forward(torch.randn(B, 1, 32, 32), source_ids=torch.tensor([0, 3, 7]))
    assert adapters[0]._code.shape == (B, cd)


if __name__ == "__main__":
    test_identity_at_init()
    test_code_dependence_after_departure()
    test_grad_flow()
    test_controller_sets_codes()
    test_source_controller()
    print("All conditional-adapter tests passed.")

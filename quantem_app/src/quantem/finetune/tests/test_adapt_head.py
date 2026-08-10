"""Head-only training itself, on a toy module.

Marked ``slow`` because it needs torch, which the default lane does not have.
It runs on the CPU in a couple of seconds — no GPU and no downloaded weights —
because what is being checked is the *recipe*, not the released model: that only
the head moves, that ignored pixels do not contribute, and that a saved head
reloads to the same predictions. The last one is the reference implementation's
own final assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="head adaptation needs torch")

from quantem.finetune.adapt import (  # noqa: E402 -- after the torch skip
    IGNORE,
    AdaptConfig,
    HeadAdaptationUnavailable,
    freeze_to_head,
    head_loss,
    load_head,
    save_head,
    train_head,
)

pytestmark = pytest.mark.slow

TILE = 64


class TinyModel(torch.nn.Module):
    """An encoder + neck + decoder with the shape the trainer expects."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.neck = torch.nn.Conv2d(4, 4, 1)
        self.decoder = torch.nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.decoder(self.neck(self.encoder(x)))


class EncoderOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(1, 2, 1)

    def forward(self, x):
        return self.encoder(x)


def _patches(n: int = 4, *, ignore_rows: int = 0):
    """Bright squares to find, optionally with an unannotated band."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        image = rng.normal(0.0, 0.2, size=(TILE, TILE)).astype(np.float32)
        target = np.zeros((TILE, TILE), dtype=np.int64)
        y, x = rng.integers(4, TILE - 20, size=2)
        image[y : y + 16, x : x + 16] += 2.0
        target[y : y + 16, x : x + 16] = 1
        if ignore_rows:
            target[:ignore_rows] = IGNORE
        out.append((image, target))
    return out


class TestFreezing:
    def test_only_the_neck_and_decoder_are_trainable(self):
        model = TinyModel()
        params, count = freeze_to_head(model)
        assert count == sum(
            p.numel() for m in (model.neck, model.decoder) for p in m.parameters()
        )
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert len(params) == 4  # weight + bias, neck and decoder

    def test_a_model_with_no_separable_head_is_refused_clearly(self):
        with pytest.raises(HeadAdaptationUnavailable, match="does not expose"):
            freeze_to_head(EncoderOnly())


class TestLoss:
    def test_ignored_pixels_do_not_contribute(self):
        """The completed-ROI contract, expressed as a gradient: changing the
        prediction where the user never annotated must not change the loss."""
        torch.manual_seed(0)
        logits = torch.randn(1, 2, 8, 8)
        target = torch.zeros(8, 8, dtype=torch.long)
        target[:4] = IGNORE
        target[6:, 6:] = 1

        baseline = head_loss(logits, target).item()
        shifted = logits.clone()
        shifted[:, :, :4] += 5.0  # only inside the ignored band
        assert head_loss(shifted, target).item() == pytest.approx(baseline, abs=1e-6)

    def test_a_perfect_prediction_scores_better_than_an_inverted_one(self):
        target = torch.zeros(8, 8, dtype=torch.long)
        target[2:6, 2:6] = 1
        good = torch.stack([(target == 0).float(), (target == 1).float()])[None] * 8
        bad = good.flip(1)
        assert head_loss(good, target).item() < head_loss(bad, target).item()


class TestTrainHead:
    def test_the_encoder_does_not_move_and_the_head_does(self):
        model = TinyModel()
        before_encoder = model.encoder.weight.detach().clone()
        before_decoder = model.decoder.weight.detach().clone()

        result = train_head(
            model, _patches(), config=AdaptConfig(steps=25, lr=1e-2, seed=0)
        )

        assert result.steps == 25
        assert torch.equal(model.encoder.weight, before_encoder)
        assert not torch.equal(model.decoder.weight, before_decoder)
        assert result.trainable_params == sum(
            p.numel() for m in (model.neck, model.decoder) for p in m.parameters()
        )

    def test_the_loss_goes_down(self):
        model = TinyModel()
        result = train_head(
            model, _patches(8), config=AdaptConfig(steps=60, lr=1e-2, seed=0)
        )
        assert np.mean(result.losses[-10:]) < np.mean(result.losses[:10])

    def test_progress_is_reported_with_an_eta(self):
        seen = []
        train_head(
            TinyModel(),
            _patches(2),
            config=AdaptConfig(steps=5, lr=1e-2, seed=0),
            on_progress=seen.append,
        )
        assert [p.step for p in seen] == [0, 1, 2, 3, 4]
        assert seen[-1].fraction == 1.0
        assert seen[0].eta_s >= 0.0

    def test_cancellation_stops_the_run_and_keeps_what_was_trained(self):
        model = TinyModel()
        calls = {"n": 0}

        def cancelled() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        result = train_head(
            model,
            _patches(2),
            config=AdaptConfig(steps=100, lr=1e-2, seed=0),
            should_cancel=cancelled,
        )
        assert result.steps == 3

    def test_no_usable_window_is_a_clear_refusal(self):
        with pytest.raises(HeadAdaptationUnavailable, match="No training window"):
            train_head(TinyModel(), [], config=AdaptConfig(steps=5))


class TestSaveAndReload:
    def test_a_saved_head_reloads_to_the_same_predictions(self, tmp_path):
        """The reference's last act, and the reason it exists: an adapter that
        does not reproduce its own number when reloaded is not a deliverable."""
        torch.manual_seed(0)
        model = TinyModel()
        train_head(model, _patches(), config=AdaptConfig(steps=20, lr=1e-2, seed=0))

        probe = torch.randn(1, 1, TILE, TILE)
        with torch.no_grad():
            expected = model(probe)

        path = save_head(model, tmp_path / "head.pt", meta={"base_model": "toy"})
        assert path.exists()

        # A fresh build of the same pack: the encoder is the same frozen blob
        # from the registry, the head is whatever the base checkpoint had.
        fresh = TinyModel()
        fresh.encoder.load_state_dict(model.encoder.state_dict())
        # Its head must start somewhere else, or the check below proves nothing.
        with torch.no_grad():
            assert not torch.allclose(fresh(probe), expected)

        meta = load_head(fresh, path)
        assert meta["base_model"] == "toy"
        with torch.no_grad():
            assert torch.allclose(fresh(probe), expected, atol=1e-6)

    def test_the_encoder_is_not_copied_into_the_head_file(self):
        """A 525 MB encoder per adapter would be the largest thing this app ever
        wrote, for nothing: the registry already has it, addressed by digest."""
        import tempfile
        from pathlib import Path

        model = TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_head(model, Path(tmp) / "head.pt")
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        assert set(payload) == {"format", "meta", "neck", "decoder"}

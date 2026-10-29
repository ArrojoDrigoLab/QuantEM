from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from quantem.sam.geometry import Box, Crop
from quantem.sam.prompt import PromptResult, run_prompt


def test_direct_backend_load_obeys_the_same_priority_lock_order():
    from quantem.core.compute_admission import interactive_pending
    from quantem.sam.backends import get_backend, reset_backend

    def build():
        assert interactive_pending()
        return SimpleNamespace(device="cpu")

    reset_backend()
    try:
        with (
            patch("quantem.sam.backends.stub_mode", return_value=True),
            patch("quantem.sam.backends.stub.StubBackend", side_effect=build),
        ):
            assert get_backend().device == "cpu"
    finally:
        reset_backend()


def test_cached_cpu_backend_uses_cpu_lane_when_cuda_is_preferred():
    lanes = []
    backend = SimpleNamespace(device="cpu")
    result = PromptResult([], Crop(0, 0, 100, 100), True, "cpu", "test", 0, 0)

    @contextmanager
    def slot(device, **kwargs):
        lanes.append(device)
        yield 0.025

    with (
        patch("quantem.inference.device.select_device", return_value="cuda"),
        patch("quantem.inference.device.total_memory_bytes", return_value=None),
        patch("quantem.sam.prompt.get_backend", return_value=backend),
        patch("quantem.sam.prompt._predict_crop", return_value=result),
        patch("quantem.core.compute_admission.compute_slot", slot),
    ):
        answer = run_prompt(
            segmentation_id="test",
            openable=None,
            box=Box(10, 10, 30, 30),
            image_width=100,
            image_height=100,
        )
    assert lanes == ["cpu"]
    assert answer.queue_ms >= 25

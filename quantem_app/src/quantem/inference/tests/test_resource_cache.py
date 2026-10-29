import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from quantem.inference import engine
from quantem.inference.specs import MODEL_SPECS


def test_vram_budget_queries_the_assigned_device():
    from quantem.inference import device

    with patch.object(device, "_torch") as torch:
        cuda = torch.return_value.cuda
        cuda.mem_get_info.return_value = (2 * 1024**3, 6 * 1024**3)
        assert device.free_memory_bytes("cuda:1") == 2 * 1024**3
        assert device.total_memory_bytes("cuda:1") == 6 * 1024**3
        assert all(call.args == ("cuda:1",) for call in cuda.mem_get_info.call_args_list)


def test_allocator_release_uses_the_assigned_device_context():
    from quantem.inference import device

    with patch.object(device, "_torch") as torch:
        cuda = torch.return_value.cuda
        device.empty_cache("cuda:1")
        cuda.device.assert_called_once_with("cuda:1")
        cuda.device.return_value.__enter__.assert_called_once()
        cuda.empty_cache.assert_called_once()
        cuda.device.return_value.__exit__.assert_called_once()


def test_concurrent_cold_loads_share_one_allocation():
    engine.clear_model_cache()
    model = engine.LoadedModel(spec=MODEL_SPECS["quantem:mito"], device="cpu", module=None)

    def build(*args):
        time.sleep(0.05)
        return model

    with (
        patch.object(engine, "select_device", return_value="cpu"),
        patch.object(engine, "configure_torch_threads"),
        patch.object(engine, "resolve_model_files"),
        patch.object(engine, "_build_and_prepare", side_effect=build) as loader,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        models = list(pool.map(lambda _: engine.load_model("quantem:mito"), range(4)))
    assert loader.call_count == 1
    assert all(item is model for item in models)
    engine.clear_model_cache()


def test_byte_budget_evicts_before_building_next_model():
    engine.clear_model_cache()
    first = engine.LoadedModel(spec=MODEL_SPECS["omniem:mito"], device="cpu", module=None)
    engine.cache_model(first)
    with (
        patch.object(engine, "model_cache_slots", return_value=4),
        patch.object(engine, "model_cache_budget_bytes", return_value=1600 * 1024**2),
        patch.object(engine, "empty_cache"),
    ):
        engine._make_load_room("quantem:mito", "cpu")
        assert engine.cached_model_keys() == []


def test_cpu_budget_reserves_room_on_small_machines():
    with patch(
        "quantem.core.machine.get_machine_profile",
        return_value=SimpleNamespace(total_ram_bytes=8 * 1024**3),
    ):
        assert engine.model_cache_budget_bytes("cpu") < 1024**3


def test_loading_cpu_model_does_not_evict_an_independent_gpu_cache():
    engine.clear_model_cache()
    gpu = engine.LoadedModel(spec=MODEL_SPECS["quantem:mito"], device="cuda", module=None)
    engine.cache_model(gpu)
    with engine._model_lease(gpu), patch.object(engine, "model_cache_slots", return_value=1):
        engine._make_load_room("omniem:mito", "cpu")
    assert engine.cached_model_keys() == [("quantem:mito", "cuda")]
    engine.clear_model_cache()


def test_cached_lookup_does_not_wait_behind_a_load_waiting_for_its_prediction():
    import threading

    engine.clear_model_cache()
    first = engine.LoadedModel(spec=MODEL_SPECS["quantem:mito"], device="cpu", module=None)
    second = engine.LoadedModel(spec=MODEL_SPECS["omniem:mito"], device="cpu", module=None)
    engine.cache_model(first)
    waiting = threading.Event()
    make_room = engine._make_load_room

    def admit(*args):
        waiting.set()
        make_room(*args)

    with (
        patch.object(engine, "model_cache_slots", return_value=1),
        patch.object(engine, "select_device", return_value="cpu"),
        patch.object(engine, "configure_torch_threads"),
        patch.object(engine, "resolve_model_files"),
        patch.object(engine, "_build_and_prepare", return_value=second),
        patch.object(engine, "_make_load_room", side_effect=admit),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        with engine._model_lease(first):
            cold = pool.submit(engine.load_model, "omniem:mito")
            assert waiting.wait(5)
            warm = pool.submit(engine.load_model, "quantem:mito")
            assert warm.result(timeout=2) is first
        assert cold.result(timeout=5) is second
    engine.clear_model_cache()


def test_active_prediction_is_not_evicted_to_admit_a_second_model():
    import threading

    engine.clear_model_cache()
    first = engine.LoadedModel(spec=MODEL_SPECS["quantem:mito"], device="cpu", module=None)
    engine.cache_model(first)
    entered = threading.Event()

    def make_room():
        entered.set()
        engine._make_load_room("omniem:mito", "cpu")

    with (
        patch.object(engine, "model_cache_slots", return_value=1),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        with engine._model_lease(first):
            task = pool.submit(make_room)
            assert entered.wait(5)
            assert not task.done()
            assert engine.cached_model_keys() == [("quantem:mito", "cpu")]
        task.result(timeout=5)
    assert engine.cached_model_keys() == []


def test_device_parking_waits_for_another_callers_forward_pass():
    import threading
    from contextlib import contextmanager

    import numpy as np

    entered = threading.Event()
    release = threading.Event()
    parked = threading.Event()

    def move(device):
        if device == "cpu":
            parked.set()

    model = engine.LoadedModel(
        spec=MODEL_SPECS["quantem:mito"], device="cuda", module=SimpleNamespace(to=move)
    )

    @contextmanager
    def admission(device, *, on_yield, on_resume):
        if entered.is_set():
            on_yield()
            on_resume()
        yield 0

    def forward(tiles):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
            assert not parked.is_set()
        return tiles

    tile = np.zeros((2, 2), dtype=np.uint8)
    with (
        patch("quantem.core.compute_admission.compute_slot", admission),
        patch.object(engine, "total_memory_bytes", return_value=4 * 1024**3),
        patch.object(engine, "empty_cache"),
        patch.object(model, "_forward_batch", side_effect=forward),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(model.forward_tiles, [tile])
        assert entered.wait(5)
        second = pool.submit(model.forward_tiles, [tile])
        try:
            assert not parked.wait(0.1)
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert parked.is_set()

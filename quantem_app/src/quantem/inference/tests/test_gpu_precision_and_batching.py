"""fp32 everywhere, and a batch that cannot change the answer.

Two owner decisions are pinned here.

**D1 -- GPU runs use fp32.** ``autocast_dtype`` used to return bf16 whenever
``torch.cuda.is_bf16_supported()`` said yes; that call counts *emulation*, so on
a Turing card -- which has no bf16 tensor cores -- it returns True and the app
selected the one dtype the hardware cannot do natively. Verified on the
measurement machine's Quadro RTX 8000: ``is_bf16_supported()`` True,
``is_bf16_supported(including_emulation=False)`` False. MEASURED cost of that
choice at 60.73 MP: mask IoU 0.983 against fp32's 0.999, 13 of 464 objects
changing identity, and one object's area moving 58 %.

**F2 -- batching is a speed fork, not a numerical one.** The windows, their
order and their Hann weights are identical whatever the batch, so the blended
map must be identical too. This asserts that on the blending side exactly; the
device side (where a different cuDNN kernel *can* move the last bit) was
measured separately on a real 52.9 MP EM image: max 1 uint8 level on 10 pixels
of 52 920 320, and identical object counts at three thresholds.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.inference import device as device_mod
from quantem.inference import tiling

SRC = Path(__file__).resolve().parents[3]


# --- D1 ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cuda", "cuda:1", "mps", "cpu"])
def test_no_device_gets_autocast(name):
    assert device_mod.autocast_dtype(name) is None


def test_nothing_in_the_tree_asks_torch_whether_bf16_is_supported():
    """The regression guard for D1.

    ``is_bf16_supported()`` is not a wrong function to call, it is a wrong
    function to *believe*: its zero-argument form counts emulated bf16. The
    honest question is ``including_emulation=False``, and since fp32 is now the
    answer for every device there is no reason for either call to exist. If one
    comes back, it comes back with an argument for the numbers above.
    """
    offenders = []
    for path in SRC.rglob("quantem/**/*.py"):
        if "tests" in path.parts or "_fig3" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "is_bf16_supported"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], (
        "bf16 selection is back in " + ", ".join(offenders) + ". See D1: on "
        "pre-Ampere hardware is_bf16_supported() counts emulation and the app "
        "picks the slowest, hungriest and least faithful of the three dtypes."
    )


def test_precision_is_recorded_as_one_value():
    assert device_mod.PRECISION == "fp32"


# --- F2: how many windows per pass -------------------------------------------


def test_the_processor_never_batches():
    """MEASURED 1.32 -> 1.35 tiles/s: one window already uses every core."""
    assert device_mod.tile_batch_for("cpu") == 1
    assert device_mod.tile_batch_for("mps") == 1


def test_a_big_card_gets_the_ceiling_and_a_small_one_gets_one_window():
    big = device_mod.tile_batch_for("cuda", embedding_dim=768, free_bytes=48 * 1024**3)
    assert big == device_mod.MAX_AUTOMATIC_TILE_BATCH

    # A ViT-L with 3 GB free: 1 536 MiB of headroom leaves ~1 536 MiB, under the
    # 2 259 MiB the measured table wants even for one window.
    small = device_mod.tile_batch_for("cuda", embedding_dim=1024, free_bytes=3 * 1024**3)
    assert small == 1


def test_the_ceiling_is_four_because_that_is_where_the_curve_flattens():
    assert device_mod.MAX_AUTOMATIC_TILE_BATCH == 4
    assert device_mod.tile_batch_for("cuda", embedding_dim=768, free_bytes=64 * 1024**3) == 4


def test_support_can_force_a_batch(monkeypatch):
    monkeypatch.setenv(device_mod.TILE_BATCH_ENV_VAR, "8")
    assert device_mod.tile_batch_for("cpu") == 8
    monkeypatch.setenv(device_mod.TILE_BATCH_ENV_VAR, "nonsense")
    assert device_mod.tile_batch_for("cpu") == 1
    os.environ.pop(device_mod.TILE_BATCH_ENV_VAR, None)


# --- F2: batching must not change the blend ----------------------------------


def _blend(shape, tile, batch, seed=3):
    plan = tiling.plan_tiles(shape, tile, overlap=0.25)
    rng = np.random.default_rng(seed)
    fields = {(t.y, t.x): rng.random((tile, tile)).astype(np.float32) for t in plan.tiles()}
    seen: list[tuple[int, int]] = []

    def predict(tiles):
        seen.extend((t.y, t.x) for t in tiles)
        return [fields[(t.y, t.x)] for t in tiles]

    out = tiling.blend_region_batched(plan, predict, batch=batch)
    return out, seen, plan.n_tiles


def test_a_batched_blend_is_the_same_array_as_a_one_at_a_time_blend():
    base, order1, total = _blend((900, 1300), 256, batch=1)
    for batch in (2, 3, 4, 8, 64):
        got, order, _ = _blend((900, 1300), 256, batch=batch)
        np.testing.assert_array_equal(got, base, err_msg=f"batch {batch} changed the blended map")
        # Same windows, same order: batching slices the sequence, it does not
        # reorder it, which is what BandBlender's row-major contract needs.
        assert order == order1
        assert len(order) == total


def test_every_window_is_reported_once_however_they_are_grouped():
    plan = tiling.plan_tiles((900, 1300), 256, overlap=0.25)

    def blank(tiles):
        return [np.zeros((256, 256), np.float32) for _ in tiles]

    for batch in (1, 3, 8):
        counts: list[tuple[int, int]] = []
        tiling.blend_region_batched(
            plan,
            blank,
            batch=batch,
            on_tile=lambda done, total, sink=counts: sink.append((done, total)),
        )
        assert counts[0] == (1, plan.n_tiles)
        assert counts[-1] == (plan.n_tiles, plan.n_tiles)
        assert [d for d, _ in counts] == list(range(1, plan.n_tiles + 1))


def test_a_predictor_that_loses_a_window_is_caught():
    plan = tiling.plan_tiles((600, 600), 256, overlap=0.25)
    with pytest.raises(ValueError, match="returned 1 maps for"):
        tiling.blend_region_batched(
            plan,
            lambda tiles: [np.zeros((256, 256), np.float32)],
            batch=4,
        )

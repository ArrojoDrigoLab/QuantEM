"""A traced encoder belongs to a device, and running out of memory is not a crash.

**D2 -- the exported encoder is device-tagged.** A TorchScript artifact is not
portable, which the exporter believed for a year and four of the eight shipped
packs disproved. Reproduced in isolation on the real card: a DINOv3-shaped ViT-B
traced on the CPU dies on CUDA inside ``rope``'s
``_get_pos_embed_from_coords`` -- the graph divides coordinates it created on
the CPU by a ``periods`` buffer that ``map_location`` moved -- and a trace made
on CUDA dies the mirror death on the CPU. Freezing is not the cause: the
unfrozen trace fails identically both ways.

Portability is a property of the encoder, not a rule: the OmniEM ViT-L's
CPU trace runs on CUDA unchanged, and MEASURED, refusing it on principle cost
~60 s of eager rebuild and 1.2 GB of extra disk per pack to reproduce a file
that already worked. So the shipped artifact is *tried* and one real forward
pass at load decides.

**D3 -- nothing caught ``torch.OutOfMemoryError``.** Now the batch halves, and
if a single window will not fit the model moves to the processor and finishes
there. Verified against the real allocator by capping it below the model's
measured floor; these pin the logic without a card.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from quantem.inference import encoders, engine
from quantem.inference.specs import MODEL_SPECS

# --- The filename ------------------------------------------------------------


def test_the_cpu_keeps_the_name_every_bundle_ships():
    assert encoders.exported_encoder_name("cpu") == "encoder_ts.pt"
    assert encoders.exported_encoder_name() == encoders.EXPORTED_ENCODER_NAME


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda", "encoder_ts.cuda.pt"),
        ("cuda:3", "encoder_ts.cuda.pt"),
        ("mps", "encoder_ts.mps.pt"),
    ],
)
def test_an_accelerator_trace_has_nowhere_to_go_but_its_own_name(device, expected):
    assert encoders.exported_encoder_name(device) == expected
    assert encoders.exported_encoder_name(device) != encoders.EXPORTED_ENCODER_NAME


def test_the_index_is_not_part_of_the_name():
    """Two cards in one machine share an artifact; they are the same device
    kind and a per-index file would be two copies of one 1.2 GB trace."""
    assert encoders.exported_encoder_name("cuda:0") == encoders.exported_encoder_name("cuda:7")


# --- The stamp inside the archive --------------------------------------------


class _FakeScript:
    def eval(self):
        return self


def _load_with_meta(tmp_path, meta, device):
    path = tmp_path / "encoder_ts.pt"
    path.write_bytes(b"not really torchscript")

    def fake_load(_path, map_location=None, _extra_files=None):
        _extra_files[encoders.EXPORT_META_FILE] = json.dumps(meta).encode()
        return _FakeScript()

    with patch("quantem.inference.encoders.torch.jit.load", fake_load):
        return encoders.load_exported_encoder(path, device=device)


_META = {
    "pack_id": "omniem:mito",
    "depth": 24,
    "embedding_dim": 1024,
    "patch_size": 14,
    "input_mean": 0.0,
    "input_std": 1.0,
    "layers": [5, 11, 17, 23],
    "traced_tile": 518,
}


def test_an_unstamped_artifact_is_tried_on_any_device(tmp_path):
    """Every bundle shipped so far is unstamped, and the OmniEM one runs on
    CUDA. Refusing it would break a path that works."""
    for device in ("cpu", "cuda", "mps"):
        loaded = _load_with_meta(tmp_path, dict(_META), device)
        assert loaded.traced_device == "cpu"


def test_a_stamped_artifact_is_refused_on_the_wrong_device(tmp_path):
    meta = dict(_META, traced_device="cuda")
    with pytest.raises(encoders.EncoderUnavailable, match="traced on 'cuda'"):
        _load_with_meta(tmp_path, meta, "cpu")
    # ...and accepted on the right one.
    assert _load_with_meta(tmp_path, meta, "cuda:2").traced_device == "cuda"


# --- Which artifact a device reaches for --------------------------------------


def _files(tmp_path):
    head = tmp_path / "head.pt"
    head.write_bytes(b"h")
    return engine.ModelFiles(
        pack_id="omniem:mito",
        head_path=head,
        export_path=tmp_path / "encoder_ts.pt",
    )


def test_the_cpu_reaches_for_the_registry_resolved_artifact(tmp_path):
    files = _files(tmp_path)
    assert engine._export_path_for(files, "cpu") == files.export_path


def test_an_accelerator_prefers_its_own_artifact_when_one_exists(tmp_path):
    files = _files(tmp_path)
    tagged = tmp_path / "encoder_ts.cuda.pt"
    tagged.write_bytes(b"cuda trace")
    assert engine._export_path_for(files, "cuda") == tagged


def test_an_accelerator_falls_back_to_the_shipped_artifact(tmp_path):
    """Not blind faith: prepare_for_device runs it before the user's tiles."""
    files = _files(tmp_path)
    assert engine._export_path_for(files, "cuda") == files.export_path


# --- Running out of memory ----------------------------------------------------


def _oom():
    import torch

    return torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB.")


def _model(tile_batch: int, tmp_path) -> engine.LoadedModel:
    return engine.LoadedModel(
        spec=MODEL_SPECS["omniem:mito"],
        device="cuda",
        module=object(),
        files=_files(tmp_path),
        tile_batch=tile_batch,
        embedding_dim=1024,
    )


def test_a_batch_that_does_not_fit_is_halved_and_said_so(tmp_path):
    model = _model(4, tmp_path)
    calls: list[int] = []

    def forward(tiles):
        calls.append(len(tiles))
        if len(tiles) > 1:
            raise _oom()
        return [np.zeros((4, 4), np.float32) for _ in tiles]

    with patch.object(engine.LoadedModel, "_forward_batch", staticmethod(forward)):
        out = model.forward_tiles([np.zeros((4, 4), np.uint8)] * 4)

    assert len(out) == 4
    assert calls[0] == 4  # asked for four
    assert model.tile_batch == 1  # ...and settled on one
    assert model.run_notices == [
        "This run used smaller batches than usual because the graphics card was "
        "short of memory. The result is the same; it took a little longer."
    ]


def test_one_window_that_does_not_fit_moves_the_run_to_the_processor(tmp_path):
    model = _model(1, tmp_path)
    rebuilt = SimpleNamespace(built=0)

    def forward(self, tiles):
        if self.device == "cuda":
            raise _oom()
        return [np.full((4, 4), 0.25, np.float32) for _ in tiles]

    def fake_build(files, spec, device, **kwargs):
        rebuilt.built += 1
        assert device == "cpu"
        return object(), "exported"

    with (
        patch.object(engine.LoadedModel, "_forward_batch", forward),
        patch.object(engine, "build_module", fake_build),
    ):
        out = model.forward_tiles([np.zeros((4, 4), np.uint8)])

    assert rebuilt.built == 1
    assert model.device == "cpu"
    assert len(out) == 1 and float(out[0][0, 0]) == 0.25  # the run finished
    assert model.run_notices == [
        "This run moved to the processor part-way through: the graphics card ran "
        "out of memory. The result is complete; it took longer than it would "
        "have on the graphics card."
    ]


def test_the_processor_has_nowhere_further_to_fall(tmp_path):
    """A host out-of-memory is a real failure, and looping on it would hang."""
    model = _model(1, tmp_path)
    model.device = "cpu"

    def forward(self, tiles):
        raise _oom()

    with patch.object(engine.LoadedModel, "_forward_batch", forward):
        with pytest.raises(MemoryError, match="not enough memory"):
            model.forward_tiles([np.zeros((4, 4), np.uint8)])


def test_a_fallback_keeps_the_user_s_own_head(tmp_path):
    """A user's fitted head is not an optional detail: rebuilding on the CPU
    without it would silently serve the released model under the adapter's
    name."""
    model = _model(1, tmp_path)
    head = tmp_path / "adapted_head.pt"
    head.write_bytes(b"head")
    model.adapter_head_path = head
    model.adapter_id = "adapter-1"
    loaded: list[Path] = []

    def forward(self, tiles):
        if self.device == "cuda":
            raise _oom()
        return [np.zeros((4, 4), np.float32) for _ in tiles]

    fake_adapt = SimpleNamespace(load_head=lambda module, path: loaded.append(path))
    with (
        patch.object(engine.LoadedModel, "_forward_batch", forward),
        patch.object(engine, "build_module", lambda *a, **k: (object(), "exported")),
        patch.dict("sys.modules", {"quantem.finetune.adapt": fake_adapt}),
    ):
        model.forward_tiles([np.zeros((4, 4), np.uint8)])

    assert loaded == [head]


def test_an_error_that_is_not_out_of_memory_is_not_swallowed(tmp_path):
    model = _model(4, tmp_path)

    def forward(self, tiles):
        raise RuntimeError("the graph is wrong")

    with patch.object(engine.LoadedModel, "_forward_batch", forward):
        with pytest.raises(RuntimeError, match="the graph is wrong"):
            model.forward_tiles([np.zeros((4, 4), np.uint8)])


def test_the_out_of_memory_recogniser_knows_both_shapes():
    from quantem.inference.device import is_out_of_memory

    assert is_out_of_memory(_oom())
    assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert is_out_of_memory(RuntimeError("MPS backend out of memory"))
    assert not is_out_of_memory(RuntimeError("Expected all tensors on same device"))
    assert not is_out_of_memory(ValueError("nope"))


# --- What the user is told -----------------------------------------------------


def test_the_two_reasons_a_run_leaves_the_card_read_differently():
    spec = MODEL_SPECS["quantem:mito"]
    cannot = engine._cannot_use_accelerator_notice(spec, "cuda", None)
    no_room = engine._cannot_use_accelerator_notice(
        spec, "cuda", engine.AcceleratorUnusable("x", out_of_memory=True)
    )
    assert cannot != no_room
    assert "cannot run" in cannot
    assert "not enough memory" in no_room
    for sentence in (cannot, no_room):
        assert "mitochondria" in sentence  # the organelle, not the pack id
        assert "quantem:mito" not in sentence
        assert "cuda" not in sentence
        assert "result is complete" in sentence


def test_no_device_sentence_carries_an_internal_name():
    """Invariant I-12: no user-facing string names a module, a command, an
    endpoint, an exception class or a pack id."""
    spec = MODEL_SPECS["omniem:ld"]
    sentences = [
        engine._SMALLER_BATCHES,
        engine._cannot_use_accelerator_notice(spec, "cuda", None),
        engine._cannot_use_accelerator_notice(
            spec, "cuda", engine.AcceleratorUnusable("x", out_of_memory=True)
        ),
    ]
    forbidden = (
        "quantem.",
        "torch",
        "cuda",
        "mps",
        "encoder_ts",
        "OutOfMemoryError",
        "pip install",
        "GET ",
        "POST ",
        "/api/",
        "omniem:",
        "fp32",
        "bf16",
    )
    for sentence in sentences:
        for token in forbidden:
            assert token not in sentence, f"{token!r} leaked into {sentence!r}"
